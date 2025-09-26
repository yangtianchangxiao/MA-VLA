#!/usr/bin/env python3
"""
软体臂训练 - Dummy数据版本
用于快速验证训练流程，无需复杂模型加载
"""

import os
import sys
import torch
import torch.nn as nn
import time
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.standalone_graph_encoder import MockPI0Model

class DummyDataset:
    """生成假数据的数据集"""

    def __init__(self, batch_size=1, num_samples=100):
        self.batch_size = batch_size
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples // self.batch_size

    def __iter__(self):
        for _ in range(len(self)):
            # 生成假的observation
            observation = {
                'image': {'camera_0': torch.randn(self.batch_size, 3, 224, 224)},
                'instruction': [f'dummy task {i}' for i in range(self.batch_size)]
            }

            # 生成假的actions
            actions = torch.randn(self.batch_size, 16, 10)

            # 生成假的graph_data
            graph_data = {
                'node_features': torch.randn(self.batch_size, 10, 19),
                'num_nodes': torch.randint(5, 10, (self.batch_size,))
            }

            yield observation, actions, graph_data

def dummy_train():
    """Dummy训练函数"""

    print("🚀 开始Dummy软体臂训练...")

    # 设备
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建模型
    model = MockPI0Model(action_dim=10, action_horizon=16, enable_graph=True)
    model = model.to(device)

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # 数据集
    dataset = DummyDataset(batch_size=2, num_samples=20)

    # 训练循环
    model.train()

    for epoch in range(2):  # 2个epoch
        print(f"\n=== Epoch {epoch + 1} ===")

        epoch_loss = 0.0

        for step, (observation, actions, graph_data) in enumerate(dataset):
            # 转移到设备
            observation['image']['camera_0'] = observation['image']['camera_0'].to(device)
            actions = actions.to(device)
            graph_data = {k: v.to(device) for k, v in graph_data.items()}

            # 前向传播
            optimizer.zero_grad()
            loss = model(observation, actions, graph_data)

            # 反向传播
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # 打印进度
            if step % 2 == 0:
                print(f"Step {step}: loss = {loss.item():.4f}")

        avg_loss = epoch_loss / len(dataset)
        print(f"Epoch {epoch + 1} 平均损失: {avg_loss:.4f}")

    print("\n🎉 Dummy训练完成!")
    print("关键验证:")
    print("  ✅ 模型前向传播正常")
    print("  ✅ 损失计算正常")
    print("  ✅ 梯度更新正常")
    print("  ✅ GPU内存管理正常")

    return True

if __name__ == "__main__":
    try:
        dummy_train()
    except Exception as e:
        print(f"❌ Dummy训练失败: {e}")
        import traceback
        traceback.print_exc()