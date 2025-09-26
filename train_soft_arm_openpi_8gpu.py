#!/usr/bin/env python3
"""
OpenPi软体臂8卡训练脚本
基于π₀.₅ DROID微调版本，适配软体臂数据

使用方法:
    单GPU: python train_soft_arm_openpi_8gpu.py
    8GPU:  torchrun --nproc_per_node=8 train_soft_arm_openpi_8gpu.py
"""

import os
import sys
import argparse
import json
from datetime import datetime
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from tqdm import tqdm

# 添加OpenPi路径
sys.path.append('/home/cx/AET_FOR_RL/vla/参考模型/openpi')
sys.path.append('/home/cx/AET_FOR_RL/vla')

from openpi_soft_arm_dataloader import create_soft_arm_dataloaders
from openpi.training import config as openpi_config

def init_distributed():
    """初始化分布式训练"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])

        print(f"🚀 初始化分布式训练: rank={rank}, world_size={world_size}, local_rank={local_rank}")

        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)

        return rank, world_size, local_rank
    else:
        print("🔧 单GPU训练模式")
        return 0, 1, 0

def get_soft_arm_config() -> Dict[str, Any]:
    """获取软体臂训练配置"""
    return {
        # 模型配置
        'model_name': 'pi05_droid',  # 使用DROID微调版本
        'pretrained_checkpoint': '~/.cache/openpi/checkpoints/pi05_droid',

        # 数据配置
        'batch_size': 8,  # 每GPU批量大小
        'num_workers': 4,
        'train_split': 0.9,
        'image_size': (224, 224),
        'action_chunk_size': 16,
        'max_sequence_length': 50,

        # 训练配置
        'num_epochs': 20,
        'learning_rate': 1e-4,  # 比预训练更小的学习率
        'weight_decay': 1e-5,
        'warmup_steps': 500,
        'save_interval': 1000,
        'eval_interval': 500,
        'log_interval': 50,

        # 软体臂专用
        'max_action_dim': 12,  # 支持最多6段×2参数
        'constraint_types': ['3DOF', '4DOF'],

        # 优化器
        'optimizer': 'adamw',
        'beta1': 0.9,
        'beta2': 0.95,
        'grad_clip_norm': 1.0,

        # 输出目录
        'output_dir': '/home/cx/AET_FOR_RL/vla/checkpoints/soft_arm_openpi',
        'experiment_name': f"soft_arm_pi05_droid_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    }

class SoftArmVLAModel(nn.Module):
    """软体臂VLA模型适配器"""

    def __init__(self, base_model, max_action_dim: int = 12):
        super().__init__()
        self.base_model = base_model
        self.max_action_dim = max_action_dim

        # 获取原始动作头的输入维度
        if hasattr(base_model, 'action_head'):
            original_dim = base_model.action_head.in_features
            self.action_head = nn.Linear(original_dim, max_action_dim)
        else:
            # 如果找不到动作头，创建一个简单的适配层
            self.action_head = nn.Linear(768, max_action_dim)  # 假设768维特征

        print(f"✅ 软体臂动作头: {self.action_head}")

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """前向传播"""
        # 使用基础模型提取特征
        features = self.base_model.encode(
            images=batch['image'],
            instructions=batch['instruction']
        )

        # 生成软体臂动作
        actions = self.action_head(features)

        return {
            'actions': actions,
            'features': features,
        }

def load_openpi_model(config: Dict[str, Any], device: torch.device):
    """加载OpenPi模型"""
    try:
        # 获取DROID配置
        openpi_cfg = openpi_config.get_config('pi05_droid')
        print(f"✅ 加载OpenPi配置: pi05_droid")

        # 创建基础模型
        from openpi.models import create_model
        base_model = create_model(openpi_cfg)

        # 加载预训练权重
        checkpoint_path = os.path.expanduser(config['pretrained_checkpoint'])
        if os.path.exists(checkpoint_path):
            print(f"🔽 加载预训练权重: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            base_model.load_state_dict(checkpoint['model'], strict=False)
            print("✅ 预训练权重加载成功")
        else:
            print(f"⚠️ 预训练权重不存在: {checkpoint_path}")
            print("   将使用随机初始化")

        # 创建软体臂适配模型
        model = SoftArmVLAModel(base_model, config['max_action_dim'])
        model = model.to(device)

        return model

    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        raise

def train_epoch(model: nn.Module,
                train_loader,
                optimizer: torch.optim.Optimizer,
                criterion: nn.Module,
                device: torch.device,
                epoch: int,
                config: Dict[str, Any],
                rank: int = 0) -> Dict[str, float]:
    """训练一个epoch"""

    model.train()
    total_loss = 0.0
    total_samples = 0

    if rank == 0:
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    else:
        pbar = train_loader

    for step, batch in enumerate(pbar):
        # 移动数据到GPU
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}

        optimizer.zero_grad()

        try:
            # 前向传播
            outputs = model(batch)
            predicted_actions = outputs['actions']
            target_actions = batch['actions']

            # 计算损失 (只考虑前几维有效动作)
            # 每个样本可能有不同的动作维度
            batch_size = predicted_actions.size(0)
            chunk_size = predicted_actions.size(1)

            loss = 0.0
            valid_samples = 0

            for i in range(batch_size):
                # 根据robot_config确定有效动作维度
                robot_config = batch['robot_config'][i]
                if '3DOF' in robot_config or '3dof' in robot_config.lower():
                    action_dim = 6  # 3段×2参数
                elif '4DOF' in robot_config or '4dof' in robot_config.lower():
                    action_dim = 8  # 4段×2参数
                else:
                    action_dim = 10  # 默认5段×2参数

                # 计算有效动作的MSE损失
                pred_valid = predicted_actions[i, :, :action_dim]
                target_valid = target_actions[i, :, :action_dim]

                sample_loss = criterion(pred_valid, target_valid)
                loss += sample_loss
                valid_samples += 1

            if valid_samples > 0:
                loss = loss / valid_samples
            else:
                loss = torch.tensor(0.0, device=device, requires_grad=True)

            # 反向传播
            loss.backward()

            # 梯度裁剪
            if config.get('grad_clip_norm'):
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip_norm'])

            optimizer.step()

            # 统计
            total_loss += loss.item()
            total_samples += batch_size

            if rank == 0 and step % config['log_interval'] == 0:
                pbar.set_postfix({
                    'loss': f'{loss.item():.6f}',
                    'avg_loss': f'{total_loss / (step + 1):.6f}'
                })

        except Exception as e:
            print(f"❌ 训练步骤失败: {e}")
            continue

    avg_loss = total_loss / len(train_loader)
    return {'train_loss': avg_loss}

def validate(model: nn.Module,
             val_loader,
             criterion: nn.Module,
             device: torch.device,
             rank: int = 0) -> Dict[str, float]:
    """验证模型"""

    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            try:
                outputs = model(batch)
                predicted_actions = outputs['actions']
                target_actions = batch['actions']

                # 同训练时的损失计算
                batch_size = predicted_actions.size(0)
                loss = 0.0
                valid_samples = 0

                for i in range(batch_size):
                    robot_config = batch['robot_config'][i]
                    if '3DOF' in robot_config or '3dof' in robot_config.lower():
                        action_dim = 6
                    elif '4DOF' in robot_config or '4dof' in robot_config.lower():
                        action_dim = 8
                    else:
                        action_dim = 10

                    pred_valid = predicted_actions[i, :, :action_dim]
                    target_valid = target_actions[i, :, :action_dim]

                    sample_loss = criterion(pred_valid, target_valid)
                    loss += sample_loss
                    valid_samples += 1

                if valid_samples > 0:
                    loss = loss / valid_samples

                total_loss += loss.item()
                total_samples += batch_size

            except Exception as e:
                if rank == 0:
                    print(f"⚠️ 验证步骤失败: {e}")
                continue

    avg_loss = total_loss / len(val_loader)
    return {'val_loss': avg_loss}

def save_checkpoint(model: nn.Module,
                   optimizer: torch.optim.Optimizer,
                   epoch: int,
                   metrics: Dict[str, float],
                   config: Dict[str, Any],
                   filepath: str):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': config,
    }

    torch.save(checkpoint, filepath)
    print(f"✅ 检查点已保存: {filepath}")

def main():
    parser = argparse.ArgumentParser(description='软体臂OpenPi训练')
    parser.add_argument('--config-override', type=str, help='配置覆盖JSON文件')
    args = parser.parse_args()

    # 初始化分布式训练
    rank, world_size, local_rank = init_distributed()
    device = torch.device(f'cuda:{local_rank}')

    # 获取配置
    config = get_soft_arm_config()
    if args.config_override:
        with open(args.config_override) as f:
            override_config = json.load(f)
        config.update(override_config)

    if rank == 0:
        print("🎯 软体臂OpenPi训练配置:")
        for key, value in config.items():
            print(f"   {key}: {value}")

    # 创建输出目录
    output_dir = os.path.join(config['output_dir'], config['experiment_name'])
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)

        # 保存配置
        with open(os.path.join(output_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)

    # 创建数据加载器
    if rank == 0:
        print("📊 创建数据加载器...")

    train_loader, val_loader = create_soft_arm_dataloaders(
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        train_split=config['train_split'],
        image_size=config['image_size'],
        action_chunk_size=config['action_chunk_size'],
        max_sequence_length=config['max_sequence_length']
    )

    # 分布式采样器
    if world_size > 1:
        train_sampler = DistributedSampler(train_loader.dataset)
        val_sampler = DistributedSampler(val_loader.dataset)

        train_loader = torch.utils.data.DataLoader(
            train_loader.dataset,
            batch_size=config['batch_size'],
            sampler=train_sampler,
            num_workers=config['num_workers'],
            pin_memory=True
        )

        val_loader = torch.utils.data.DataLoader(
            val_loader.dataset,
            batch_size=config['batch_size'],
            sampler=val_sampler,
            num_workers=config['num_workers'],
            pin_memory=True
        )

    # 创建模型
    if rank == 0:
        print("🤖 加载模型...")

    model = load_openpi_model(config, device)

    # 分布式模型
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # 优化器和损失函数
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        betas=(config['beta1'], config['beta2'])
    )

    criterion = nn.MSELoss()

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['num_epochs'] * len(train_loader)
    )

    # 训练循环
    best_val_loss = float('inf')

    for epoch in range(config['num_epochs']):
        if world_size > 1:
            train_sampler.set_epoch(epoch)

        # 训练
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch, config, rank
        )

        # 验证
        val_metrics = validate(model, val_loader, criterion, device, rank)

        # 学习率更新
        scheduler.step()

        if rank == 0:
            print(f"📊 Epoch {epoch}:")
            print(f"   训练损失: {train_metrics['train_loss']:.6f}")
            print(f"   验证损失: {val_metrics['val_loss']:.6f}")
            print(f"   学习率: {scheduler.get_last_lr()[0]:.2e}")

            # 保存最佳模型
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                best_model_path = os.path.join(output_dir, 'best_model.pth')
                save_checkpoint(model, optimizer, epoch, {**train_metrics, **val_metrics},
                              config, best_model_path)

            # 定期保存检查点
            if (epoch + 1) % 5 == 0:
                checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pth')
                save_checkpoint(model, optimizer, epoch, {**train_metrics, **val_metrics},
                              config, checkpoint_path)

    if rank == 0:
        print("🎉 训练完成!")
        print(f"   最佳验证损失: {best_val_loss:.6f}")
        print(f"   模型保存在: {output_dir}")

    # 清理分布式训练
    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()