#!/usr/bin/env python3
"""
OpenPi软体臂多卡训练脚本
基于OpenPi框架，支持8卡训练软体臂VLA模型

Usage:
    Single GPU:
        python train_soft_arm_openpi.py --exp_name soft_arm_test

    Multi-GPU (8卡):
        torchrun --standalone --nnodes=1 --nproc_per_node=8 train_soft_arm_openpi.py --exp_name soft_arm_8gpu
"""

import argparse
import os
import logging
import time
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import wandb

# 导入我们的数据适配器
from openpi_soft_arm_adapter import SoftArmOpenPiDataset, create_openpi_dataloader

def init_distributed():
    """初始化分布式训练"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        dist.init_process_group(backend='nccl')
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
    else:
        # 单GPU情况
        rank = 0
        world_size = 1
        local_rank = 0

    return rank, world_size, local_rank

def init_logging(rank: int):
    """初始化日志"""
    level = logging.INFO if rank == 0 else logging.WARNING
    logging.basicConfig(
        level=level,
        format=f"[Rank {rank}] %(asctime)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger(__name__)

class SoftArmVLAModel(nn.Module):
    """软体臂VLA模型，适配OpenPi架构"""

    def __init__(self,
                 max_segments: int = 5,
                 hidden_dim: int = 512,
                 vision_encoder_dim: int = 512,
                 language_encoder_dim: int = 512,
                 action_chunk_size: int = 16):
        super().__init__()

        self.max_segments = max_segments
        self.hidden_dim = hidden_dim
        self.action_chunk_size = action_chunk_size

        # 视觉编码器 - 处理224x224图像
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),       # 224 -> 112
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),                                            # 112 -> 56

            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),     # 56 -> 28
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),                                            # 28 -> 14

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),    # 14 -> 7
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.AdaptiveAvgPool2d((4, 4)),                               # -> 4x4

            nn.Flatten(),
            nn.Linear(256 * 4 * 4, vision_encoder_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # 语言编码器 - 简化版本
        self.language_encoder = nn.Sequential(
            nn.Linear(512, language_encoder_dim),  # 假设使用预训练embeddings
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # 多模态融合
        fusion_dim = vision_encoder_dim + language_encoder_dim
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1)
        )

        # 软体臂动作解码器 - 支持变长输出
        self.action_decoders = nn.ModuleDict({
            str(n_seg): nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_seg * 2 * action_chunk_size),  # n_seg段 × 2参数(α,β) × chunk_size
            )
            for n_seg in range(2, max_segments + 1)
        })

        # 段数分类器
        self.segment_classifier = nn.Linear(hidden_dim, max_segments - 1)  # 2,3,4,5段

    def forward(self,
                images: torch.Tensor,
                language_embeddings: torch.Tensor,
                target_segments: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            images: (B, T, H, W, C) 图像序列
            language_embeddings: (B, lang_dim) 语言嵌入
            target_segments: (B,) 目标段数
        """
        batch_size, seq_len = images.shape[:2]

        # 展平时间维度进行视觉编码
        images_flat = images.view(-1, *images.shape[2:])  # (B*T, H, W, C)
        images_flat = images_flat.permute(0, 3, 1, 2)     # (B*T, C, H, W)

        vision_features = self.vision_encoder(images_flat)  # (B*T, vision_dim)
        vision_features = vision_features.view(batch_size, seq_len, -1)  # (B, T, vision_dim)

        # 语言编码
        lang_features = self.language_encoder(language_embeddings)  # (B, lang_dim)

        # 扩展语言特征到时间维度
        lang_features = lang_features.unsqueeze(1).expand(-1, seq_len, -1)  # (B, T, lang_dim)

        # 多模态融合
        fused_features = torch.cat([vision_features, lang_features], dim=-1)  # (B, T, fusion_dim)
        fused_features = self.fusion_layer(fused_features)  # (B, T, hidden_dim)

        # 时间池化 - 使用最后一个时间步
        final_features = fused_features[:, -1]  # (B, hidden_dim)

        # 段数分类（如果没有提供target_segments）
        segment_logits = self.segment_classifier(final_features)  # (B, max_segments-1)

        if target_segments is not None:
            # 训练阶段：使用真实段数
            predicted_segments = target_segments
        else:
            # 推理阶段：使用预测段数
            predicted_segments = torch.argmax(segment_logits, dim=-1) + 2  # +2因为从2段开始

        # 根据段数生成动作
        actions = []
        for i in range(batch_size):
            n_seg = int(predicted_segments[i].item())
            decoder = self.action_decoders[str(n_seg)]
            action = decoder(final_features[i:i+1])  # (1, n_seg*2*chunk_size)
            action = action.view(1, n_seg * 2, self.action_chunk_size)  # (1, n_joints, chunk_size)
            actions.append(action)

        # 填充到最大尺寸
        max_joints = self.max_segments * 2
        padded_actions = torch.zeros(batch_size, max_joints, self.action_chunk_size,
                                   device=images.device, dtype=torch.float32)

        for i, action in enumerate(actions):
            n_joints = action.shape[1]
            padded_actions[i, :n_joints] = action[0]

        return {
            'actions': padded_actions,  # (B, max_joints, chunk_size)
            'segment_logits': segment_logits,  # (B, max_segments-1)
            'predicted_segments': predicted_segments,  # (B,)
            'features': final_features  # (B, hidden_dim)
        }

def simple_language_embedding(text: str, embedding_dim: int = 512) -> torch.Tensor:
    """简化的语言嵌入（用于演示）"""
    # 简单的字符级嵌入
    chars = [min(ord(c), 255) for c in text.lower()[:64]]
    chars += [0] * (64 - len(chars))  # 填充

    # 转换为嵌入向量（实际应用中应该使用预训练的语言模型）
    embedding = torch.zeros(embedding_dim)
    for i, char_code in enumerate(chars):
        if i < embedding_dim // 8:
            embedding[i*8:(i+1)*8] = torch.tensor([char_code / 255.0] * 8)

    return embedding

def train_epoch(model: nn.Module,
                dataloader: torch.utils.data.DataLoader,
                optimizer: torch.optim.Optimizer,
                criterion: nn.Module,
                device: torch.device,
                rank: int,
                logger: logging.Logger) -> float:
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()

        # 移动数据到设备
        images = batch['image'].to(device).float()  # (B, T, H, W, C)
        actions = batch['action'].to(device).float()  # (B, T, n_joints)
        segments = batch['segments'].to(device)      # (B,)

        # 生成语言嵌入
        language_embeddings = torch.stack([
            simple_language_embedding(desc) for desc in batch['language_instruction']
        ]).to(device)  # (B, embedding_dim)

        # 前向传播
        outputs = model(images, language_embeddings, target_segments=segments)
        predicted_actions = outputs['actions']  # (B, max_joints, chunk_size)

        # 计算损失 - 只对有效的关节计算
        batch_size = actions.shape[0]
        seq_len = actions.shape[1]

        # 重塑动作到正确的格式
        # actions: (B, T, n_joints) -> (B, n_joints, T) 然后截取chunk_size
        actions_reshaped = actions.transpose(1, 2)  # (B, n_joints, T)
        chunk_size = min(actions_reshaped.shape[2], model.action_chunk_size)
        actions_target = actions_reshaped[:, :, :chunk_size]  # (B, n_joints, chunk_size)

        # 填充到最大尺寸
        max_joints = predicted_actions.shape[1]
        actions_padded = torch.zeros_like(predicted_actions)
        for i in range(batch_size):
            n_joints = min(actions_target.shape[1], max_joints)
            actions_padded[i, :n_joints, :chunk_size] = actions_target[i, :n_joints]

        # L1损失（适合机器人控制）
        loss = criterion(predicted_actions, actions_padded)

        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if rank == 0 and batch_idx % 10 == 0:
            logger.info(f"Batch {batch_idx}: Loss = {loss.item():.6f}")

    return total_loss / num_batches if num_batches > 0 else 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, required=True, help='实验名称')
    parser.add_argument('--batch_size', type=int, default=4, help='每GPU的batch size')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--lr', type=float, default=3e-4, help='学习率')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='模型保存目录')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点路径')
    args = parser.parse_args()

    # 初始化分布式
    rank, world_size, local_rank = init_distributed()
    device = torch.device(f'cuda:{local_rank}')
    logger = init_logging(rank)

    if rank == 0:
        logger.info(f"🚀 开始软体臂VLA训练")
        logger.info(f"   实验名称: {args.exp_name}")
        logger.info(f"   设备数量: {world_size}")
        logger.info(f"   总batch size: {args.batch_size * world_size}")

    # 创建数据集
    dataset = SoftArmOpenPiDataset(batch_size=args.batch_size)

    # 分布式采样器
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank) if world_size > 1 else None

    # 数据加载器
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=4,
        pin_memory=True
    )

    # 创建模型
    model = SoftArmVLAModel().to(device)

    # DDP包装
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    # 优化器和损失函数
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.L1Loss()  # 适合机器人控制的损失

    # 创建保存目录
    save_dir = Path(args.save_dir) / args.exp_name
    if rank == 0:
        save_dir.mkdir(parents=True, exist_ok=True)

    # W&B初始化（仅主进程）
    if rank == 0:
        wandb.init(
            project='soft-arm-vla',
            name=args.exp_name,
            config={
                'batch_size': args.batch_size * world_size,
                'epochs': args.epochs,
                'learning_rate': args.lr,
                'world_size': world_size,
                'model': 'SoftArmVLA'
            }
        )

    # 训练循环
    best_loss = float('inf')

    for epoch in range(args.epochs):
        if sampler:
            sampler.set_epoch(epoch)

        start_time = time.time()
        avg_loss = train_epoch(model, dataloader, optimizer, criterion, device, rank, logger)
        epoch_time = time.time() - start_time

        scheduler.step()

        if rank == 0:
            logger.info(f"Epoch {epoch+1}/{args.epochs}: Loss = {avg_loss:.6f}, Time = {epoch_time:.2f}s")

            # W&B记录
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': avg_loss,
                'learning_rate': scheduler.get_last_lr()[0],
                'epoch_time': epoch_time
            })

            # 保存检查点
            if avg_loss < best_loss:
                best_loss = avg_loss
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict() if world_size > 1 else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': best_loss,
                    'config': {
                        'max_segments': 5,
                        'hidden_dim': 512,
                        'action_chunk_size': 16
                    }
                }
                torch.save(checkpoint, save_dir / 'best_model.pth')
                logger.info(f"✅ 保存最佳模型: loss = {best_loss:.6f}")

    if rank == 0:
        logger.info(f"🎉 训练完成! 最佳损失: {best_loss:.6f}")
        wandb.finish()

    # 清理分布式
    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()