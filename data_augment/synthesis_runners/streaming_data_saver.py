#!/usr/bin/env python3
"""
流式数据保存器 - 支持robot graph和timestep数据的流式保存
避免内存积累，边生成边保存
"""

import os
import json
import random
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path


def generate_random_joint_limits(num_joints: int = 7) -> List[Tuple[float, float]]:
    """生成随机化的关节限制 - 正负分别随机化"""
    joint_limits = []
    for i in range(num_joints):
        lower_limit = -random.uniform(2.9, 3.07)  # 负方向限制
        upper_limit = random.uniform(2.9, 3.07)   # 正方向限制
        joint_limits.append((lower_limit, upper_limit))
    return joint_limits


def generate_chain_adjacency(num_nodes: int) -> List[List[int]]:
    """生成链式机器人的邻接矩阵"""
    adj = [[0] * num_nodes for _ in range(num_nodes)]
    for i in range(num_nodes - 1):
        adj[i][i + 1] = 1  # forward connection
        adj[i + 1][i] = 1  # backward connection
    return adj


def generate_robot_graph(morphology_data: Dict, joint_limits: List[Tuple[float, float]]) -> Dict:
    """生成robot graph结构"""
    # Node features: [lower_limit, upper_limit, velocity_limit, link_length, node_type]
    node_features = []

    # Base node (type=0)
    node_features.append([0.0, 0.0, 0.0, 0.0, 0])

    # Joint nodes
    for i, (lower, upper) in enumerate(joint_limits):
        # 获取link长度
        if 'dh_parameters' in morphology_data and len(morphology_data['dh_parameters']) > i:
            link_length = morphology_data['dh_parameters'][i][2]  # DH参数中的a参数
        else:
            link_length = 0.33  # 默认长度

        # 最后一个关节是end effector (type=2)，其他是intermediate (type=1)
        node_type = 2 if i == len(joint_limits) - 1 else 1

        node_features.append([lower, upper, 2.175, link_length, node_type])

    # 生成邻接矩阵 (base + joints)
    num_nodes = len(node_features)
    adjacency_matrix = generate_chain_adjacency(num_nodes)

    return {
        "num_nodes": num_nodes,
        "node_features": node_features,
        "adjacency_matrix": adjacency_matrix,
        "dh_parameters": morphology_data.get('dh_parameters', []),
        "morphology_type": morphology_data.get('type', 'unknown')
    }


class StreamingSynthesisDataSaver:
    """流式数据保存器 - 避免内存积累"""

    def __init__(self, output_dir: str, morphology_type: str):
        self.output_dir = Path(output_dir)
        self.morphology_type = morphology_type

        # 创建目录结构
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.graphs_dir = self.output_dir / "robot_graphs"
        self.graphs_dir.mkdir(exist_ok=True)

        # 轨迹数据文件
        self.trajectory_file = self.output_dir / "trajectory_data.parquet"
        self._trajectory_buffer = []
        self._buffer_size = 100  # 批量写入大小

        print(f"💾 StreamingSynthesisDataSaver initialized")
        print(f"   Output directory: {self.output_dir}")
        print(f"   Graphs directory: {self.graphs_dir}")

    def save_robot_graph(self, graph: Dict, graph_id: str):
        """立即保存单个robot graph"""
        graph_file = self.graphs_dir / f"{graph_id}_graph.json"

        with open(graph_file, 'w') as f:
            json.dump(graph, f, indent=2, default=self._json_serializer)

        print(f"   📊 Saved robot graph: {graph_id}")

    def append_timestep_data(self, timestep_data: Dict):
        """流式追加timestep数据"""
        self._trajectory_buffer.append(timestep_data)

        # 达到buffer大小时写入磁盘
        if len(self._trajectory_buffer) >= self._buffer_size:
            self._flush_trajectory_buffer()

    def _flush_trajectory_buffer(self):
        """将buffer中的数据写入parquet文件"""
        if not self._trajectory_buffer:
            return

        df = pd.DataFrame(self._trajectory_buffer)

        # 如果文件存在，追加；否则创建新文件
        if self.trajectory_file.exists():
            existing_df = pd.read_parquet(self.trajectory_file)
            df = pd.concat([existing_df, df], ignore_index=True)

        df.to_parquet(self.trajectory_file, index=False)

        print(f"   💾 Flushed {len(self._trajectory_buffer)} timesteps to {self.trajectory_file.name}")
        self._trajectory_buffer.clear()

    def finalize(self):
        """完成保存，清理剩余buffer"""
        if self._trajectory_buffer:
            self._flush_trajectory_buffer()

        # 保存元数据
        metadata = {
            'morphology_type': self.morphology_type,
            'total_graphs': len(list(self.graphs_dir.glob("*_graph.json"))),
            'trajectory_file': str(self.trajectory_file),
            'buffer_size_used': self._buffer_size
        }

        metadata_file = self.output_dir / "synthesis_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"   ✅ Synthesis data finalized: {metadata['total_graphs']} graphs saved")

    def _json_serializer(self, obj):
        """JSON序列化辅助函数"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        return obj