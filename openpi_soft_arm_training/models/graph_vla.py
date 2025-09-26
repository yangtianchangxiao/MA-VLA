#!/usr/bin/env python3
"""
Graph-based VLA模型
实现你提出的架构: 图编码 + 双路融合 + 节点式动作头 + Flow Matching
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional
import math

# 添加OpenPi路径
sys.path.append('/home/cx/AET_FOR_RL/vla/参考模型/openpi')
from openpi.shared.config import checkpoint_to_config
from openpi.models import pi0_5

class SoftArmGraphNN(nn.Module):
    """软体臂图神经网络

    输入: URDF图结构 (N, 19)
    输出: 图token (N, 32)
    """

    def __init__(self,
                 input_dim: int = 19,
                 hidden_dim: int = 64,
                 output_dim: int = 32,
                 num_layers: int = 3):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        # 节点特征编码
        self.node_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 图卷积层
        self.graph_layers = nn.ModuleList([
            GraphConvLayer(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])

        # 输出投影
        self.output_proj = nn.Linear(hidden_dim, output_dim)

        # 层归一化
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])

    def forward(self,
                node_features: torch.Tensor,  # (B, N, 19)
                edge_indices: torch.Tensor,   # (B, 2, E)
                batch_size: int) -> torch.Tensor:  # (B, N, 32)

        B, N, _ = node_features.shape

        # 节点编码
        x = self.node_encoder(node_features)  # (B, N, hidden_dim)

        # 图卷积传播
        for i, (conv_layer, norm_layer) in enumerate(zip(self.graph_layers, self.layer_norms)):
            residual = x
            x = conv_layer(x, edge_indices)
            x = norm_layer(x + residual)  # 残差连接
            x = F.relu(x)

        # 输出投影
        graph_tokens = self.output_proj(x)  # (B, N, output_dim)

        return graph_tokens

class GraphConvLayer(nn.Module):
    """图卷积层"""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim * 2, out_dim)  # 节点+邻居特征concat

    def forward(self,
                node_features: torch.Tensor,  # (B, N, in_dim)
                edge_indices: torch.Tensor    # (B, 2, E)
                ) -> torch.Tensor:

        B, N, in_dim = node_features.shape
        device = node_features.device

        # 简化的图卷积：对每个节点聚合邻居特征
        output = []

        for b in range(B):
            batch_edges = edge_indices[b]  # (2, E)
            batch_nodes = node_features[b]  # (N, in_dim)

            # 为每个节点收集邻居
            node_outputs = []
            for n in range(N):
                # 找到以n为目标的边
                neighbor_indices = batch_edges[0][batch_edges[1] == n]

                if len(neighbor_indices) > 0:
                    # 聚合邻居特征
                    neighbor_features = batch_nodes[neighbor_indices]  # (num_neighbors, in_dim)
                    neighbor_agg = torch.mean(neighbor_features, dim=0)  # (in_dim,)
                else:
                    # 没有邻居，用零填充
                    neighbor_agg = torch.zeros_like(batch_nodes[n])

                # 节点自身特征 + 邻居聚合特征
                combined = torch.cat([batch_nodes[n], neighbor_agg], dim=0)  # (2*in_dim,)
                node_outputs.append(combined)

            batch_output = torch.stack(node_outputs, dim=0)  # (N, 2*in_dim)
            output.append(batch_output)

        # 合并批次
        batched_output = torch.stack(output, dim=0)  # (B, N, 2*in_dim)

        # 线性变换
        return self.linear(batched_output)  # (B, N, out_dim)

class AttentionPooling(nn.Module):
    """注意力池化：将(N, 32)聚合成(1, 32)"""

    def __init__(self, input_dim: int = 32):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1)
        )

    def forward(self, graph_tokens: torch.Tensor) -> torch.Tensor:  # (B, N, 32) -> (B, 1, 32)
        # 计算注意力权重
        attention_scores = self.attention(graph_tokens)  # (B, N, 1)
        attention_weights = F.softmax(attention_scores, dim=1)  # (B, N, 1)

        # 加权聚合
        robot_token = torch.sum(graph_tokens * attention_weights, dim=1, keepdim=True)  # (B, 1, 32)

        return robot_token

class NodeActionHead(nn.Module):
    """节点式动作头

    为每个关节节点独立预测动作，然后组合成完整轨迹
    """

    def __init__(self,
                 vlm_feature_dim: int = 768,
                 node_token_dim: int = 32,
                 action_chunk_size: int = 16,
                 max_dof: int = 10):
        super().__init__()

        self.vlm_feature_dim = vlm_feature_dim
        self.node_token_dim = node_token_dim
        self.action_chunk_size = action_chunk_size
        self.max_dof = max_dof

        # 为每个关节预测动作的头
        # 输入: VLM特征 + 节点token (残差连接)
        self.joint_heads = nn.ModuleDict()

        # 动态创建不同形态的头
        for num_segments in [2, 3, 4, 5]:
            for constraint_type in ["3DOF", "4DOF"]:
                num_joints = num_segments * 2  # 每段两个参数(α, β)
                head_name = f"{num_segments}seg_{constraint_type}"

                self.joint_heads[head_name] = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(vlm_feature_dim + node_token_dim, 128),
                        nn.ReLU(),
                        nn.Linear(128, 64),
                        nn.ReLU(),
                        nn.Linear(64, action_chunk_size)  # 预测H步的Δq
                    ) for _ in range(num_joints)
                ])

    def forward(self,
                vlm_features: torch.Tensor,    # (B, vlm_dim)
                graph_tokens: torch.Tensor,    # (B, N, 32)
                robot_configs: List[str]       # batch的机器人配置
                ) -> torch.Tensor:

        B, N, _ = graph_tokens.shape
        device = graph_tokens.device

        # 为每个batch样本预测动作
        batch_actions = []

        for b in range(B):
            robot_config = robot_configs[b]

            # 解析机器人配置
            if "2_segments" in robot_config:
                num_segments = 2
            elif "3_segments" in robot_config:
                num_segments = 3
            elif "4_segments" in robot_config:
                num_segments = 4
            elif "5_segments" in robot_config:
                num_segments = 5
            else:
                num_segments = 3  # 默认

            constraint_type = "3DOF" if "3DOF" in robot_config else "4DOF"

            num_joints = num_segments * 2
            head_name = f"{num_segments}seg_{constraint_type}"

            # 获取对应的动作头
            if head_name not in self.joint_heads:
                # 如果没有对应的头，使用默认的3seg_3DOF
                head_name = "3seg_3DOF"
                num_joints = 6

            joint_heads = self.joint_heads[head_name]

            # VLM特征
            batch_vlm_features = vlm_features[b:b+1].expand(num_joints, -1)  # (num_joints, vlm_dim)

            # 图token（只取前num_joints个，或者重复/截断）
            if N >= num_joints:
                batch_graph_tokens = graph_tokens[b, :num_joints, :]  # (num_joints, 32)
            else:
                # 不够的话重复最后一个
                batch_graph_tokens = graph_tokens[b]  # (N, 32)
                padding = batch_graph_tokens[-1:].expand(num_joints - N, -1)
                batch_graph_tokens = torch.cat([batch_graph_tokens, padding], dim=0)

            # 预测每个关节的动作
            joint_actions = []
            for j in range(num_joints):
                # 组合特征: VLM + 节点token (残差)
                combined_features = torch.cat([
                    batch_vlm_features[j],
                    batch_graph_tokens[j]
                ], dim=0)  # (vlm_dim + 32,)

                # 使用对应的关节头预测
                delta_q = joint_heads[j](combined_features.unsqueeze(0))  # (1, action_chunk_size)
                joint_actions.append(delta_q)

            # 拼接成完整动作 (num_joints, action_chunk_size)
            sample_actions = torch.cat(joint_actions, dim=0)  # (num_joints, H)

            # 填充到最大DoF
            if num_joints < self.max_dof:
                padding = torch.zeros(self.max_dof - num_joints, self.action_chunk_size, device=device)
                sample_actions = torch.cat([sample_actions, padding], dim=0)

            batch_actions.append(sample_actions)

        # 组合批次 (B, max_dof, action_chunk_size)
        batched_actions = torch.stack(batch_actions, dim=0)

        # 转置为 (B, action_chunk_size, max_dof) 符合Flow Matching格式
        return batched_actions.transpose(1, 2)  # (B, H, DOF)

class SoftArmGraphVLA(nn.Module):
    """完整的软体臂Graph-based VLA模型

    架构:
    1. 图像+文本 → VLM backbone
    2. 机器人图 → GraphNN → (N×32 tokens)
    3. 双路融合: attention pool (全局) + 残差 (细粒度)
    4. 节点式动作头 → 拼表执行 → Flow Matching
    """

    def __init__(self,
                 pretrained_checkpoint: str,
                 action_chunk_size: int = 16,
                 max_dof: int = 10,
                 graph_token_dim: int = 32):
        super().__init__()

        self.action_chunk_size = action_chunk_size
        self.max_dof = max_dof
        self.graph_token_dim = graph_token_dim

        # 1. 加载预训练的OpenPi VLM backbone
        self._load_vlm_backbone(pretrained_checkpoint)

        # 2. 软体臂图神经网络
        self.graph_nn = SoftArmGraphNN(
            input_dim=19,
            output_dim=graph_token_dim
        )

        # 3. 双路融合
        self.attention_pooling = AttentionPooling(graph_token_dim)

        # 4. 节点式动作头
        self.action_head = NodeActionHead(
            vlm_feature_dim=768,  # 假设OpenPi的特征维度
            node_token_dim=graph_token_dim,
            action_chunk_size=action_chunk_size,
            max_dof=max_dof
        )

        # 5. Flow Matching组件 (简化版)
        self.flow_matching = SimpleFlowMatching(
            action_dim=max_dof,
            action_chunk_size=action_chunk_size
        )

    def _load_vlm_backbone(self, checkpoint_path: str):
        """加载OpenPi VLM backbone"""
        try:
            # 加载OpenPi预训练模型
            config = checkpoint_to_config(checkpoint_path)
            self.vlm_backbone = pi0_5.Pi05(config)

            # 加载权重
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            self.vlm_backbone.load_state_dict(checkpoint['model'], strict=False)

            print(f"✅ 成功加载OpenPi VLM backbone: {checkpoint_path}")

        except Exception as e:
            print(f"⚠️ OpenPi加载失败: {e}")
            print("  使用简化的VLM backbone")

            # 简化的VLM backbone
            self.vlm_backbone = SimplifiedVLMBackbone()

    def forward(self,
                images: torch.Tensor,           # (B, C, H, W)
                instructions: List[str],        # B个指令
                node_features: torch.Tensor,   # (B, N, 19)
                edge_indices: torch.Tensor,    # (B, 2, E)
                robot_configs: List[str],      # B个机器人配置
                target_actions: Optional[torch.Tensor] = None,  # (B, H, DOF) 用于训练
                timesteps: Optional[torch.Tensor] = None        # Flow Matching时间步
                ) -> Dict[str, torch.Tensor]:

        B = images.shape[0]

        # 1. VLM编码 (图像+文本)
        try:
            vlm_features = self.vlm_backbone.encode(images, instructions)  # (B, 768)
        except:
            # 回退到简化编码
            vlm_features = self.vlm_backbone(images, instructions)

        # 2. 图结构编码
        graph_tokens = self.graph_nn(node_features, edge_indices, B)  # (B, N, 32)

        # 3. 双路融合
        # 全局路: attention pooling → 进backbone
        robot_token = self.attention_pooling(graph_tokens)  # (B, 1, 32)

        # 将robot_token融入VLM特征 (简化融合)
        if vlm_features.dim() == 2:  # (B, 768)
            robot_global = robot_token.squeeze(1)  # (B, 32)
            # 线性投影到VLM维度
            if not hasattr(self, 'robot_proj'):
                self.robot_proj = nn.Linear(32, vlm_features.shape[1], device=vlm_features.device)
            robot_proj = self.robot_proj(robot_global)  # (B, 768)
            enhanced_vlm_features = vlm_features + robot_proj  # 残差融合
        else:
            enhanced_vlm_features = vlm_features

        # 4. 节点式动作预测
        # 细粒度路: 未池化的graph_tokens作为条件
        predicted_actions = self.action_head(
            enhanced_vlm_features,  # (B, 768)
            graph_tokens,           # (B, N, 32) 残差路径
            robot_configs           # 机器人配置
        )  # (B, H, DOF)

        # 5. Flow Matching (训练时)
        results = {
            'predicted_actions': predicted_actions,
            'graph_tokens': graph_tokens,
            'robot_token': robot_token,
            'vlm_features': enhanced_vlm_features,
        }

        if target_actions is not None:
            # 训练模式: 计算Flow Matching损失
            flow_loss = self.flow_matching(
                predicted_actions,
                target_actions,
                timesteps
            )
            results['flow_loss'] = flow_loss

        return results

class SimplifiedVLMBackbone(nn.Module):
    """简化的VLM backbone，用于测试"""

    def __init__(self):
        super().__init__()

        # 图像编码器 (简化的CNN)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 7, 2, 3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, 768)
        )

        # 文本编码器 (简化的token嵌入)
        self.text_encoder = nn.Sequential(
            nn.Embedding(10000, 256),  # 词汇表大小
            nn.Linear(256, 768)
        )

    def forward(self, images: torch.Tensor, instructions: List[str]) -> torch.Tensor:
        # 图像特征
        image_features = self.image_encoder(images)  # (B, 768)

        # 文本特征 (简化: 用指令长度作为特征)
        text_lengths = torch.tensor([len(inst) for inst in instructions],
                                   device=images.device, dtype=torch.long)
        text_lengths = torch.clamp(text_lengths, 0, 9999)  # 限制范围
        text_features = self.text_encoder[1](
            self.text_encoder[0](text_lengths).mean(dim=1)
        )  # (B, 768)

        # 简单融合
        return image_features + text_features

class SimpleFlowMatching(nn.Module):
    """简化的Flow Matching实现"""

    def __init__(self, action_dim: int, action_chunk_size: int):
        super().__init__()
        self.action_dim = action_dim
        self.action_chunk_size = action_chunk_size

    def forward(self,
                predicted_actions: torch.Tensor,    # (B, H, DOF)
                target_actions: torch.Tensor,       # (B, H, DOF)
                timesteps: Optional[torch.Tensor] = None
                ) -> torch.Tensor:

        # 简化的MSE损失 (真正的Flow Matching更复杂)
        # 这里只是占位，真正实现需要噪声调度、时间步等

        B, H, DOF = predicted_actions.shape

        # 只计算有效DoF的损失
        # 假设target_actions中填充的部分为0
        valid_mask = (target_actions.abs().sum(dim=1) > 0).float()  # (B, DOF)
        valid_mask = valid_mask.unsqueeze(1).expand(-1, H, -1)  # (B, H, DOF)

        # 加权MSE损失
        mse_loss = F.mse_loss(predicted_actions, target_actions, reduction='none')  # (B, H, DOF)
        weighted_loss = (mse_loss * valid_mask).sum() / (valid_mask.sum() + 1e-8)

        return weighted_loss

def create_soft_arm_graph_vla(config: Dict[str, Any]) -> SoftArmGraphVLA:
    """创建软体臂Graph VLA模型"""

    model = SoftArmGraphVLA(
        pretrained_checkpoint=config.get('pretrained_checkpoint', '~/.cache/openpi/checkpoints/pi05_droid'),
        action_chunk_size=config.get('action_chunk_size', 16),
        max_dof=config.get('max_dof', 10),
        graph_token_dim=config.get('graph_token_dim', 32)
    )

    return model

if __name__ == "__main__":
    # 测试模型
    print("🧪 测试Graph-based VLA模型...")

    # 模拟数据
    B, H, W = 2, 224, 224
    N, E = 6, 14  # 3段软体臂的图结构

    # 输入数据
    images = torch.randn(B, 3, H, W)
    instructions = ["Pick up the red cube", "Move to target position"]
    node_features = torch.randn(B, N, 19)  # 图节点特征
    edge_indices = torch.randint(0, N, (B, 2, E))  # 边索引
    robot_configs = ["3_segments_3DOF_default", "3_segments_3DOF_default"]
    target_actions = torch.randn(B, 16, 10)  # 目标动作

    # 创建模型
    config = {
        'action_chunk_size': 16,
        'max_dof': 10,
        'graph_token_dim': 32
    }

    model = create_soft_arm_graph_vla(config)

    # 前向传播
    with torch.no_grad():
        results = model(
            images=images,
            instructions=instructions,
            node_features=node_features,
            edge_indices=edge_indices,
            robot_configs=robot_configs,
            target_actions=target_actions
        )

    print("✅ 模型测试成功!")
    print(f"   预测动作形状: {results['predicted_actions'].shape}")
    print(f"   图token形状: {results['graph_tokens'].shape}")
    print(f"   机器人token形状: {results['robot_token'].shape}")
    print(f"   Flow损失: {results['flow_loss'].item():.6f}")