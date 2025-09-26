#!/usr/bin/env python3
"""
软体臂训练 - 基于官方OpenPi train_pytorch.py的最小修改版本
直接复用官方逻辑，只替换数据加载和模型创建部分
"""

import dataclasses
import gc
import logging
import os
import platform
import shutil
import sys
import time

import jax
import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
import torch.nn.parallel
import tqdm
import wandb

# 添加OpenPi路径
sys.path.append('/home/cx/AET_FOR_RL/vla/参考模型/openpi/src')

# 导入官方OpenPi组件
import openpi.models.pi0_config
import openpi.models_pytorch.pi0_pytorch
import openpi.shared.normalize as _normalize
import openpi.training.config as _config

# 导入我们的扩展
sys.path.append('/home/cx/AET_FOR_RL/vla/openpi_soft_arm_training')
from models.pi0_graph_extension import PI0PytorchWithGraph
from data.soft_arm_data_adapter import create_soft_arm_openpi_dataloader

# 复用官方的日志和DDP函数
def init_logging():
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    else:
        logger.handlers[0].setFormatter(formatter)

def setup_ddp():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1
    if use_ddp and not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend, init_method="env://")

        if os.environ.get("TORCH_DISTRIBUTED_DEBUG") is None:
            os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"

    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    return use_ddp, local_rank, device

def cleanup_ddp():
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()

def set_seed(seed: int, local_rank: int):
    torch.manual_seed(seed + local_rank)
    np.random.seed(seed + local_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + local_rank)

# 我们的数据加载函数
def build_datasets(processed_data_dir: str, batch_size: int):
    train_loader = create_soft_arm_openpi_dataloader(
        processed_data_dir, 'train', batch_size
    )

    # 创建dummy data_config
    class DataConfig:
        def __init__(self):
            self.norm_stats = None
            self.asset_id = None

    return train_loader, DataConfig()

def create_debug_config():
    """创建调试用的配置，模仿官方TrainConfig"""

    @dataclasses.dataclass
    class DebugConfig:
        # 基础配置
        seed: int = 42
        batch_size: int = 1
        num_train_steps: int = 100
        save_interval: int = 50
        log_interval: int = 10

        # 模型配置
        pytorch_training_precision: str = "bfloat16"

        # 学习率配置
        lr_schedule: object = None

        # 优化器配置
        optimizer: object = None

        # 路径配置
        checkpoint_dir: str = "/home/cx/AET_FOR_RL/vla/openpi_soft_arm_training/debug_checkpoints"

        # wandb配置
        wandb_enabled: bool = False
        project_name: str = "soft-arm-debug"
        exp_name: str = "debug"

        # 预训练权重
        pytorch_weight_path: str = None

        # 数据配置
        processed_data_dir: str = "/home/cx/AET_FOR_RL/vla/openpi_soft_arm_training/data/processed"

        # 恢复配置
        resume: bool = False
        overwrite: bool = False

    # 创建学习率配置
    @dataclasses.dataclass
    class LRSchedule:
        warmup_steps: int = 100
        peak_lr: float = 5e-5
        decay_steps: int = 100
        decay_lr: float = 1e-6

    # 创建优化器配置
    @dataclasses.dataclass
    class OptimizerConfig:
        b1: float = 0.9
        b2: float = 0.95
        eps: float = 1e-8
        weight_decay: float = 1e-5
        clip_gradient_norm: float = 1.0

    config = DebugConfig()
    config.lr_schedule = LRSchedule()
    config.optimizer = OptimizerConfig()

    return config

def train_loop():
    """主训练循环 - 基于官方train_pytorch.py"""

    init_logging()

    use_ddp, local_rank, device = setup_ddp()
    is_main = (not use_ddp) or (dist.get_rank() == 0)

    # 创建配置
    config = create_debug_config()
    set_seed(config.seed, local_rank)

    # 创建checkpoint目录
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # 构建数据集
    loader, data_config = build_datasets(config.processed_data_dir, config.batch_size)

    # 构建模型 - 使用300m变体，维度匹配
    model_cfg = openpi.models.pi0_config.Pi0Config(
        dtype=config.pytorch_training_precision,
        action_dim=10,  # 软体臂10DoF
        action_horizon=8,  # 减半节省内存
        max_token_len=1024,
        paligemma_variant="gemma_2b_lora",  # 使用LoRA微调
        action_expert_variant="gemma_2b_lora",  # 两边都用LoRA
        pi05=True,
    )

    # 使用我们的扩展模型，先禁用graph验证基础训练
    model = PI0PytorchWithGraph(model_cfg, enable_graph=False).to(device)

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        logging.info("Enabled gradient checkpointing for PI0Pytorch model")

    # DDP包装
    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
        )

    # 优化器 - 使用官方方式
    warmup_steps = config.lr_schedule.warmup_steps
    peak_lr = config.lr_schedule.peak_lr
    decay_steps = config.lr_schedule.decay_steps
    end_lr = config.lr_schedule.decay_lr

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )

    def lr_schedule(step: int):
        if step < warmup_steps:
            init_lr = peak_lr / (warmup_steps + 1)
            return init_lr + (peak_lr - init_lr) * step / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
        cos = 0.5 * (1 + np.cos(np.pi * progress))
        return end_lr + (peak_lr - end_lr) * cos

    # 训练循环 - 复用官方逻辑
    model.train()
    start_time = time.time()
    infos = []
    global_step = 0

    if is_main:
        logging.info(f"Running on: {platform.node()}")
        logging.info(f"Training config: batch_size={config.batch_size}, num_train_steps={config.num_train_steps}")

    pbar = tqdm.tqdm(total=config.num_train_steps, initial=global_step, desc="Training", disable=not is_main) if is_main else None

    while global_step < config.num_train_steps:
        for observation, actions in loader:
            if global_step >= config.num_train_steps:
                break

            # 数据处理 - 正确处理我们的ObservationWrapper
            # 直接转移observation的属性到设备
            observation.images = jax.tree.map(
                lambda x: x.to(device) if hasattr(x, 'to') else x,
                observation.images
            )
            observation.image_masks = jax.tree.map(
                lambda x: x.to(device) if hasattr(x, 'to') else x,
                observation.image_masks
            )
            observation.state = observation.state.to(device)

            # 转移token相关属性到设备
            if observation.tokenized_prompt is not None:
                observation.tokenized_prompt = observation.tokenized_prompt.to(device)
            if observation.tokenized_prompt_mask is not None:
                observation.tokenized_prompt_mask = observation.tokenized_prompt_mask.to(device)
            if observation.token_ar_mask is not None:
                observation.token_ar_mask = observation.token_ar_mask.to(device)
            if observation.token_loss_mask is not None:
                observation.token_loss_mask = observation.token_loss_mask.to(device)

            # 提取图数据
            obs_dict = observation.to_dict()
            graph_data = obs_dict.get('graph_data', None)
            if graph_data and isinstance(graph_data, dict):
                graph_data = {k: v.to(device) if hasattr(v, 'to') else v for k, v in graph_data.items()}

            actions = actions.to(torch.float32).to(device)

            # 更新学习率
            for pg in optim.param_groups:
                pg["lr"] = lr_schedule(global_step)

            # 前向传播 - 传递图数据给扩展模型
            losses = model(observation, actions, graph_data=graph_data)
            if isinstance(losses, (list, tuple)):
                losses = torch.stack(losses)
            elif not isinstance(losses, torch.Tensor):
                losses = torch.tensor(losses, device=device, dtype=torch.float32)

            loss = losses.mean()

            # 反向传播
            loss.backward()

            # 梯度裁剪
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.optimizer.clip_gradient_norm)

            # 优化器步骤
            optim.step()
            optim.zero_grad(set_to_none=True)

            # 收集统计信息
            if is_main:
                infos.append({
                    "loss": loss.item(),
                    "learning_rate": optim.param_groups[0]["lr"],
                    "grad_norm": float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm,
                })

            if is_main and (global_step % config.log_interval == 0):
                elapsed = time.time() - start_time
                avg_loss = sum(info["loss"] for info in infos) / len(infos)
                avg_lr = sum(info["learning_rate"] for info in infos) / len(infos)

                logging.info(f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} time={elapsed:.1f}s")
                start_time = time.time()
                infos = []

            global_step += 1

            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{optim.param_groups[0]['lr']:.2e}",
                })

    if pbar is not None:
        pbar.close()

    cleanup_ddp()

if __name__ == "__main__":
    train_loop()