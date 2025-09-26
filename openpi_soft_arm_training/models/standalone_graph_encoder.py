#!/usr/bin/env python3
"""
独立图编码器 - 不依赖OpenPi复杂环境
用于快速测试和验证训练流程
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional

class StandaloneGraphEncoder(nn.Module):
    """独立图编码器 - 与OpenPi环境解耦"""

    def __init__(self, input_dim: int = 19, output_dim: int = 32):
        super().__init__()

        # 节点编码
        self.node_encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

        # 简单自注意力
        self.attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=4,
            batch_first=True
        )

        self.norm = nn.LayerNorm(output_dim)

    def forward(self, graph_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            graph_data: {
                'node_features': (B, N, 19),
                'num_nodes': (B,) - 有效节点数
            }
        Returns:
            graph_embedding: (B, 32) - 全局图表示
        """
        node_features = graph_data['node_features']  # (B, N, 19)
        num_nodes = graph_data.get('num_nodes', None)

        # 节点编码
        node_embeddings = self.node_encoder(node_features)  # (B, N, 32)

        # 自注意力
        attended, _ = self.attention(
            node_embeddings, node_embeddings, node_embeddings
        )  # (B, N, 32)

        # 残差 + 归一化
        node_embeddings = self.norm(attended + node_embeddings)

        # 全局池化 (考虑有效节点)
        if num_nodes is not None:
            # 创建mask
            B, N = node_embeddings.shape[:2]
            mask = torch.arange(N, device=node_embeddings.device)[None, :] < num_nodes[:, None]
            mask = mask.float().unsqueeze(-1)  # (B, N, 1)

            # 加权平均
            global_embedding = (node_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            # 简单平均池化
            global_embedding = node_embeddings.mean(dim=1)  # (B, 32)

        return global_embedding

class MockPI0Model(nn.Module):
    """模拟PI0模型 - 用于测试训练流程"""

    def __init__(self, action_dim: int = 10, action_horizon: int = 16, enable_graph: bool = True):
        super().__init__()

        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.enable_graph = enable_graph

        # 图编码器
        if enable_graph:
            self.graph_encoder = StandaloneGraphEncoder(19, 32)

        # 简化的视觉编码器
        self.visual_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(7),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512)
        )

        # 文本编码器 (简化)
        self.text_encoder = nn.Linear(384, 128)  # 假设文本特征维度

        # 融合层
        feature_dim = 512 + 128  # visual + text
        if enable_graph:
            feature_dim += 32  # + graph

        # 动作预测头
        self.action_head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, action_horizon * action_dim)
        )

        print(f"✅ Mock PI0模型创建完成 (图支持: {enable_graph})")

    def forward(self, observation, actions, graph_data=None):
        """前向传播"""
        batch_size = actions.shape[0]

        # 处理观察
        features = []

        # 视觉特征
        if 'image' in observation and 'camera_0' in observation['image']:
            visual_feat = self.visual_encoder(observation['image']['camera_0'])
            features.append(visual_feat)
        else:
            # 占位视觉特征
            features.append(torch.zeros(batch_size, 512, device=actions.device))

        # 文本特征 (简化处理)
        if 'instruction' in observation:
            # 简单的文本embedding (实际应该用transformer)
            text_feat = torch.randn(batch_size, 128, device=actions.device)
            features.append(text_feat)
        else:
            features.append(torch.zeros(batch_size, 128, device=actions.device))

        # 图特征
        if self.enable_graph and graph_data is not None:
            graph_feat = self.graph_encoder(graph_data)
            features.append(graph_feat)
        elif self.enable_graph:
            features.append(torch.zeros(batch_size, 32, device=actions.device))

        # 特征融合
        fused_features = torch.cat(features, dim=1)

        # 动作预测
        predicted_actions = self.action_head(fused_features)
        predicted_actions = predicted_actions.view(batch_size, self.action_horizon, self.action_dim)

        # 计算损失 (简单MSE)
        loss = F.mse_loss(predicted_actions, actions)

        return loss

def test_standalone_components():
    """测试独立组件"""
    print("🧪 测试独立图编码器...")

    # 测试图编码器
    encoder = StandaloneGraphEncoder(19, 32)
    graph_data = {
        'node_features': torch.randn(2, 10, 19),
        'num_nodes': torch.tensor([6, 8])
    }

    with torch.no_grad():
        output = encoder(graph_data)

    print(f"✅ 图编码器测试成功: {graph_data['node_features'].shape} → {output.shape}")

    # 测试Mock模型
    print("\n🧪 测试Mock PI0模型...")

    model = MockPI0Model(action_dim=10, action_horizon=16, enable_graph=True)

    # 模拟数据
    observation = {
        'image': {'camera_0': torch.randn(2, 3, 224, 224)},
        'instruction': ['task 1', 'task 2']
    }
    actions = torch.randn(2, 16, 10)

    with torch.no_grad():
        loss = model(observation, actions, graph_data)

    print(f"✅ Mock模型测试成功: loss = {loss.item():.4f}")

    # 测试GPU
    if torch.cuda.is_available():
        print("\n🧪 测试GPU兼容性...")
        device = torch.device('cuda:0')

        model = model.to(device)
        observation = {k: v.to(device) if isinstance(v, torch.Tensor) else
                      {k2: v2.to(device) if isinstance(v2, torch.Tensor) else v2 for k2, v2 in v.items()}
                      if isinstance(v, dict) else v
                      for k, v in observation.items()}
        actions = actions.to(device)
        graph_data = {k: v.to(device) for k, v in graph_data.items()}

        with torch.no_grad():
            loss = model(observation, actions, graph_data)

        print(f"✅ GPU测试成功: loss = {loss.item():.4f}, device = {loss.device}")

    return True

if __name__ == "__main__":
    test_standalone_components()
    print("\n🎉 独立组件测试完成!")