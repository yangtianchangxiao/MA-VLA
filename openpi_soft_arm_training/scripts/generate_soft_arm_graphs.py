#!/usr/bin/env python3
"""
软体臂图结构生成器
为不同段数的软体臂生成图结构数据
"""

import os
import numpy as np
from typing import Dict, List, Tuple, Any
import json

def generate_soft_arm_graph(num_segments: int, constraint_type: str = "3DOF") -> Dict[str, Any]:
    """
    生成软体臂图结构

    Args:
        num_segments: 软体臂段数 (2-5)
        constraint_type: 约束类型 ("3DOF" 或 "4DOF")

    Returns:
        图结构数据字典
    """

    # 软体臂的拓扑结构是线性链
    # 节点: 每段两个关节 (bending angle α, bending direction β)
    num_joints = num_segments * 2  # 每段两个PCC参数
    num_nodes = num_joints

    # 节点特征 (19维，与现有robot graph格式兼容)
    node_features = []

    for segment_idx in range(num_segments):
        for param_idx in range(2):  # α (bending) 和 β (direction)
            if param_idx == 0:  # α: bending angle
                joint_type = [0, 1, 0, 0, 0, 0]  # 连续弯曲关节
                axis = [0, 0, 1]  # Z轴弯曲
                limits = [-np.pi/2, np.pi/2, -1.0, 1.0]  # 角度范围 + 速度限制
            else:  # β: bending direction
                joint_type = [0, 0, 1, 0, 0, 0]  # 旋转关节
                axis = [0, 0, 1]  # Z轴旋转
                limits = [-np.pi, np.pi, -2.0, 2.0]  # 方向范围

            # 节点在软体臂上的位置 (沿Z轴分布)
            position = [0, 0, segment_idx * 0.1 + param_idx * 0.05]  # 10cm per segment
            orientation = [0, 0, 0]  # 初始方向

            # 组合成19D特征
            node_feature = joint_type + axis + position + orientation + limits
            assert len(node_feature) == 19, f"特征维度错误: {len(node_feature)}"

            node_features.append(node_feature)

    node_features = np.array(node_features, dtype=np.float32)  # (num_nodes, 19)

    # 边结构: 线性链连接 + 段内α-β耦合
    edges = []

    # 1. 段内耦合: α和β参数相互影响
    for segment_idx in range(num_segments):
        alpha_node = segment_idx * 2      # α节点
        beta_node = segment_idx * 2 + 1   # β节点

        # 双向连接
        edges.append([alpha_node, beta_node])
        edges.append([beta_node, alpha_node])

    # 2. 段间连接: 相邻段的影响
    for segment_idx in range(num_segments - 1):
        # 当前段的β与下一段的α连接 (弯曲传播)
        curr_beta = segment_idx * 2 + 1
        next_alpha = (segment_idx + 1) * 2

        # 双向连接
        edges.append([curr_beta, next_alpha])
        edges.append([next_alpha, curr_beta])

        # 相邻段的α也相互影响 (连续性约束)
        curr_alpha = segment_idx * 2
        edges.append([curr_alpha, next_alpha])
        edges.append([next_alpha, curr_alpha])

    # 转换为edge_indices格式 (2, num_edges)
    if edges:
        edge_indices = np.array(edges, dtype=np.int64).T  # (2, num_edges)
    else:
        edge_indices = np.zeros((2, 0), dtype=np.int64)

    # 图数据结构
    graph_data = {
        'node_features': node_features,        # (num_nodes, 19)
        'edge_indices': edge_indices,          # (2, num_edges)
        'num_nodes': num_nodes,
        'num_edges': edge_indices.shape[1],

        # 软体臂专用信息
        'num_segments': num_segments,
        'constraint_type': constraint_type,
        'joint_names': [f"segment_{i//2}_{'alpha' if i%2==0 else 'beta'}"
                       for i in range(num_nodes)],
        'segment_mapping': [(i//2, 'alpha' if i%2==0 else 'beta')
                           for i in range(num_nodes)],
    }

    return graph_data

def save_soft_arm_graphs(output_dir: str):
    """生成并保存所有软体臂图结构"""

    os.makedirs(output_dir, exist_ok=True)

    graphs_generated = []

    # 生成不同段数和约束类型的图
    for num_segments in [2, 3, 4, 5]:
        for constraint_type in ["3DOF", "4DOF"]:

            print(f"🔧 生成 {num_segments}段 {constraint_type} 软体臂图...")

            # 生成图数据
            graph_data = generate_soft_arm_graph(num_segments, constraint_type)

            # 文件名
            graph_filename = f"soft_arm_{num_segments}segments_{constraint_type}.npz"
            graph_path = os.path.join(output_dir, graph_filename)

            # 保存NPZ格式
            np.savez(graph_path, **graph_data)

            # 记录生成信息
            graph_info = {
                'filename': graph_filename,
                'path': graph_path,
                'num_segments': num_segments,
                'constraint_type': constraint_type,
                'num_nodes': graph_data['num_nodes'],
                'num_edges': graph_data['num_edges'],
                'node_features_shape': graph_data['node_features'].shape,
                'edge_indices_shape': graph_data['edge_indices'].shape,
            }

            graphs_generated.append(graph_info)

            print(f"✅ 保存: {graph_filename}")
            print(f"   节点数: {graph_data['num_nodes']}")
            print(f"   边数: {graph_data['num_edges']}")

    # 保存生成摘要
    summary = {
        'total_graphs': len(graphs_generated),
        'generation_time': str(np.datetime64('now')),
        'graphs': graphs_generated,
    }

    summary_path = os.path.join(output_dir, 'soft_arm_graphs_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ 生成完成! 总共 {len(graphs_generated)} 个图文件")
    print(f"📄 摘要保存: {summary_path}")

    return graphs_generated

def test_graph_structure(graph_path: str):
    """测试图结构的正确性"""
    print(f"\n🧪 测试图结构: {graph_path}")

    try:
        data = np.load(graph_path)

        print(f"✅ 文件加载成功")
        print(f"   包含字段: {list(data.keys())}")
        print(f"   节点特征形状: {data['node_features'].shape}")
        print(f"   边索引形状: {data['edge_indices'].shape}")
        print(f"   节点数: {data['num_nodes']}")
        print(f"   边数: {data['num_edges']}")

        # 检查数据范围
        node_features = data['node_features']
        print(f"   特征范围: [{node_features.min():.3f}, {node_features.max():.3f}]")

        # 检查边的有效性
        edge_indices = data['edge_indices']
        if edge_indices.shape[1] > 0:
            max_node_idx = edge_indices.max()
            print(f"   最大节点索引: {max_node_idx} (应该 < {data['num_nodes']})")

            if max_node_idx >= data['num_nodes']:
                print(f"❌ 边索引超出节点范围!")
            else:
                print(f"✅ 边索引有效")

        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def main():
    """主函数"""

    print("🚀 软体臂图结构生成器")
    print("=" * 40)

    # 输出目录
    output_dir = "/home/cx/AET_FOR_RL/vla/openpi_soft_arm_training/data/robot_graphs"

    # 生成图文件
    graphs = save_soft_arm_graphs(output_dir)

    # 测试生成的图
    print("\n🧪 测试生成的图结构...")
    for graph_info in graphs[:2]:  # 测试前两个
        test_graph_structure(graph_info['path'])

    print("\n🎉 软体臂图结构生成完成!")
    print("=" * 40)
    print(f"📂 输出目录: {output_dir}")
    print(f"📊 生成的图文件: {len(graphs)} 个")
    print("\n下一步:")
    print("1. 更新数据预处理脚本以使用新的图结构")
    print("2. 测试数据加载流程")
    print("3. 开始模型训练")

if __name__ == "__main__":
    main()