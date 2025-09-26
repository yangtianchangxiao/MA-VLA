#!/usr/bin/env python3
"""
软体臂Graph VLA训练脚本 - 基于官方OpenPi训练脚本的最小化修改
符合Linus原则: 复用所有官方基础设施，只替换必要组件

使用方法:
单GPU: python scripts/train_soft_arm.py
多GPU: torchrun --standalone --nnodes=1 --nproc_per_node=8 scripts/train_soft_arm.py
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
import yaml

# 添加OpenPi路径
sys.path.append('/home/cx/AET_FOR_RL/vla/参考模型/openpi/src')

# 导入官方OpenPi组件
import openpi.models.pi0_config
import openpi.shared.normalize as _normalize
import openpi.training.config as _config

# 导入我们的扩展组件
sys.path.append('/home/cx/AET_FOR_RL/vla/openpi_soft_arm_training')
from models.pi0_graph_extension import PI0PytorchWithGraph, create_soft_arm_pi0_config
from data.soft_arm_data_adapter import create_soft_arm_openpi_dataloader


class SoftArmTrainConfig:
    """软体臂训练配置 - 扩展官方配置"""

    def __init__(self, config_path: str = None):
        # 加载YAML配置
        if config_path is None:
            config_path = "/home/cx/AET_FOR_RL/vla/openpi_soft_arm_training/configs/soft_arm_config.yaml"

        with open(config_path, 'r') as f:
            self.config_dict = yaml.safe_load(f)

        # 基础训练配置
        self.exp_name = self.config_dict['experiment']['name']
        self.project_name = self.config_dict.get('logging', {}).get('wandb_project', 'soft-arm-graph-vla')

        # 数据配置
        data_config = self.config_dict['data']
        self.processed_data_dir = data_config['processed_dir']
        self.batch_size = data_config['batch_size']
        self.num_workers = data_config.get('num_workers', 4)

        # 训练配置
        train_config = self.config_dict['training']
        self.num_train_steps = train_config['max_steps']
        self.save_interval = train_config['save_interval']
        self.log_interval = train_config['log_interval']

        # 模型配置
        model_config = self.config_dict['model']
        self.action_dim = model_config['max_dof']  # 10DoF
        self.action_horizon = model_config['action_chunk_size']  # 16
        self.max_token_len = model_config.get('max_token_len', 1024)
        self.enable_graph = model_config.get('graph', {}).get('enabled', True)

        # 优化器配置
        optimizer_config = train_config['optimizer']
        self.peak_lr = float(optimizer_config['lr'])
        self.weight_decay = float(optimizer_config['weight_decay'])
        self.b1, self.b2 = optimizer_config['betas']
        self.clip_gradient_norm = float(train_config['grad_clip_norm'])

        # 学习率调度
        self.warmup_steps = 1000  # 固定warmup
        self.decay_steps = self.num_train_steps
        self.decay_lr = float(train_config['scheduler']['eta_min'])

        # 系统配置
        output_config = self.config_dict['output']
        self.checkpoint_dir_base = output_config['checkpoint_dir']
        self.wandb_enabled = self.config_dict.get('logging', {}).get('use_wandb', False)  # 默认关闭wandb

        # DDP配置
        self.pytorch_training_precision = torch.bfloat16  # 混合精度
        self.seed = 42

        # 构建完整的checkpoint路径
        self.checkpoint_dir = os.path.join(self.checkpoint_dir_base, self.exp_name)

        # 其他选项
        self.resume = False
        self.overwrite = False
        self.pytorch_weight_path = model_config.get('pretrained_checkpoint')

        print(f"✅ 软体臂训练配置已加载:")
        print(f"   实验名称: {self.exp_name}")
        print(f"   数据目录: {self.processed_data_dir}")
        print(f"   批量大小: {self.batch_size}")
        print(f"   训练步数: {self.num_train_steps}")
        print(f"   动作维度: {self.action_dim}DoF × {self.action_horizon}步")
        print(f"   图支持: {'✅' if self.enable_graph else '❌'}")


def init_logging():
    """初始化日志系统 - 复用官方实现"""
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
    """设置分布式训练 - 复用官方实现"""
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


def build_soft_arm_datasets(config: SoftArmTrainConfig):
    """构建软体臂数据集 - 我们的自定义实现"""

    # 训练数据加载器
    train_loader = create_soft_arm_openpi_dataloader(
        processed_data_dir=config.processed_data_dir,
        split='train',
        batch_size=config.batch_size
    )

    # 验证数据加载器
    val_loader = create_soft_arm_openpi_dataloader(
        processed_data_dir=config.processed_data_dir,
        split='val',
        batch_size=config.batch_size
    )

    # 获取数据配置
    data_config = train_loader.data_config()

    print(f"✅ 软体臂数据集已构建")
    print(f"   训练加载器: {len(train_loader.dataset)} 个样本")
    print(f"   验证加载器: {len(val_loader.dataset)} 个样本")

    return train_loader, data_config


def create_soft_arm_model(config: SoftArmTrainConfig, device):
    """创建软体臂模型 - 我们的核心修改"""

    # 创建Pi0配置
    pi0_config = create_soft_arm_pi0_config(
        action_dim=config.action_dim,
        action_horizon=config.action_horizon,
        max_token_len=config.max_token_len
    )

    # 使用我们的扩展模型
    model = PI0PytorchWithGraph(pi0_config, enable_graph=config.enable_graph).to(device)

    # 启用梯度检查点 (如果支持)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("✅ 已启用梯度检查点")

    return model


def train_loop(config: SoftArmTrainConfig):
    """主训练循环 - 复用官方逻辑，最小化修改"""

    use_ddp, local_rank, device = setup_ddp()
    is_main = (not use_ddp) or (dist.get_rank() == 0)

    # 设置随机种子
    torch.manual_seed(config.seed + local_rank)
    np.random.seed(config.seed + local_rank)

    # 创建checkpoint目录
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # 初始化wandb (仅主进程)
    if is_main and config.wandb_enabled:
        wandb.init(
            name=config.exp_name,
            config=config.config_dict,
            project=config.project_name
        )

    # 构建数据集 - 使用我们的软体臂数据
    train_loader, data_config = build_soft_arm_datasets(config)

    # 构建模型 - 使用我们的图扩展模型
    model = create_soft_arm_model(config, device)

    # DDP包装
    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
        )

    # 加载预训练权重 (如果指定)
    if config.pytorch_weight_path and os.path.exists(config.pytorch_weight_path):
        model_path = os.path.join(config.pytorch_weight_path, "model.safetensors")
        if os.path.exists(model_path):
            safetensors.torch.load_model(
                (model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model),
                model_path
            )
            print(f"✅ 已加载预训练权重: {config.pytorch_weight_path}")

    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.peak_lr,
        betas=(config.b1, config.b2),
        weight_decay=config.weight_decay,
    )

    # 学习率调度器
    def lr_schedule(step: int):
        if step < config.warmup_steps:
            init_lr = config.peak_lr / (config.warmup_steps + 1)
            return init_lr + (config.peak_lr - init_lr) * step / config.warmup_steps
        # cosine decay
        progress = min(1.0, (step - config.warmup_steps) / max(1, config.decay_steps - config.warmup_steps))
        cos = 0.5 * (1 + np.cos(np.pi * progress))
        return config.decay_lr + (config.peak_lr - config.decay_lr) * cos

    # 训练循环
    model.train()
    global_step = 0
    start_time = time.time()
    infos = []

    if is_main:
        print(f"🚀 开始软体臂Graph VLA训练")
        print(f"   设备: {device}")
        print(f"   批量大小: {config.batch_size}")
        print(f"   训练步数: {config.num_train_steps}")
        print(f"   分布式: {use_ddp} (world_size={torch.distributed.get_world_size() if use_ddp else 1})")

    pbar = tqdm.tqdm(total=config.num_train_steps, desc="训练中", disable=not is_main) if is_main else None

    while global_step < config.num_train_steps:
        for observation, actions in train_loader:
            if global_step >= config.num_train_steps:
                break

            # 转移数据到设备
            # observation已经是OpenPi期望的对象格式
            observation.images = jax.tree.map(
                lambda x: x.to(device) if hasattr(x, 'to') else x,
                observation.images
            )
            actions = actions.to(torch.float32).to(device)

            # 提取图数据 (如果有)
            obs_dict = observation.to_dict()
            graph_data = obs_dict.get('graph_data', None)
            if graph_data is not None:
                graph_data = jax.tree.map(
                    lambda x: x.to(device) if hasattr(x, 'to') else x,
                    graph_data
                )

            # 更新学习率
            for pg in optimizer.param_groups:
                pg["lr"] = lr_schedule(global_step)

            # 前向传播 - 关键: 传递graph_data给我们的扩展模型
            losses = model(observation, actions, graph_data=graph_data)

            if isinstance(losses, (list, tuple)):
                losses = torch.stack(losses)
            elif not isinstance(losses, torch.Tensor):
                losses = torch.tensor(losses, device=device, dtype=torch.float32)

            loss = losses.mean()

            # 反向传播
            loss.backward()

            # 梯度裁剪
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.clip_gradient_norm)

            # 优化器步骤
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # 收集统计信息
            if is_main:
                infos.append({
                    "loss": loss.item(),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "grad_norm": float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm,
                })

            # 日志记录
            if is_main and (global_step % config.log_interval == 0):
                elapsed = time.time() - start_time
                avg_loss = sum(info["loss"] for info in infos) / len(infos)
                avg_lr = sum(info["learning_rate"] for info in infos) / len(infos)

                print(f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} time={elapsed:.1f}s")

                if config.wandb_enabled:
                    wandb.log({
                        "loss": avg_loss,
                        "learning_rate": avg_lr,
                        "step": global_step,
                    }, step=global_step)

                start_time = time.time()
                infos = []

            # 保存checkpoint
            if is_main and (global_step % config.save_interval == 0) and global_step > 0:
                ckpt_dir = os.path.join(config.checkpoint_dir, f"{global_step}")
                os.makedirs(ckpt_dir, exist_ok=True)

                # 保存模型
                model_to_save = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
                safetensors.torch.save_model(model_to_save, os.path.join(ckpt_dir, "model.safetensors"))

                # 保存优化器
                torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))

                # 保存元数据
                metadata = {
                    "global_step": global_step,
                    "config": config.config_dict,
                    "timestamp": time.time(),
                }
                torch.save(metadata, os.path.join(ckpt_dir, "metadata.pt"))

                print(f"✅ 已保存checkpoint: {ckpt_dir}")

            global_step += 1

            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                })

    # 训练完成
    if pbar is not None:
        pbar.close()

    if is_main and config.wandb_enabled:
        wandb.finish()

    if use_ddp:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()

    print("🎉 软体臂Graph VLA训练完成!")


def main():
    """主函数"""
    init_logging()

    # 解析命令行参数 (简化版)
    import argparse
    parser = argparse.ArgumentParser(description='软体臂Graph VLA训练')
    parser.add_argument('--config', type=str, default=None, help='配置文件路径')
    parser.add_argument('--resume', action='store_true', help='从checkpoint恢复训练')
    args = parser.parse_args()

    # 创建配置
    config = SoftArmTrainConfig(args.config)
    config.resume = args.resume

    # 开始训练
    train_loop(config)


if __name__ == "__main__":
    main()