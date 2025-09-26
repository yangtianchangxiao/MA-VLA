#!/usr/bin/env python3
"""
软体臂数据预处理脚本
将分散的数据整理成统一格式，便于训练
"""

import os
import sys
import json
import h5py
import pickle
import yaml
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from tqdm import tqdm
import cv2

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append('/home/cx/AET_FOR_RL/vla')

def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def ensure_dir(path: str):
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)

class SoftArmDataProcessor:
    """软体臂数据处理器"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.raw_paths = config['raw_data']
        self.processed_paths = config['processed_data']

        # 确保输出目录存在
        root_dir = self.processed_paths['root']
        ensure_dir(root_dir)
        ensure_dir(os.path.join(root_dir, 'processed'))
        ensure_dir(os.path.join(root_dir, 'processed', 'image_cache'))

        self.episodes = []
        self.failed_episodes = []

    def process_all_data(self):
        """处理所有数据"""
        print("🚀 开始软体臂数据预处理")
        print("=" * 50)

        # 1. 处理软体臂关节数据
        self._process_soft_arm_data()

        # 2. 验证图像数据
        self._verify_image_data()

        # 3. 处理机器人图数据
        self._process_robot_graphs()

        # 4. 创建训练数据分割
        self._create_data_splits()

        # 5. 保存统一数据
        self._save_unified_data()

        # 6. 生成数据统计
        self._generate_statistics()

    def _process_soft_arm_data(self):
        """处理软体臂合成数据"""
        print("\n📊 处理软体臂关节数据...")

        # 处理3DOF数据
        dof3_dir = self.raw_paths['soft_arm_3dof']
        if os.path.exists(dof3_dir):
            self._process_morphology_data(dof3_dir, "3DOF")

        # 处理4DOF数据
        dof4_dir = self.raw_paths['soft_arm_4dof']
        if os.path.exists(dof4_dir):
            self._process_morphology_data(dof4_dir, "4DOF")

        print(f"✅ 总共处理了 {len(self.episodes)} 个episode")
        if self.failed_episodes:
            print(f"⚠️ 失败了 {len(self.failed_episodes)} 个episode")

    def _process_morphology_data(self, data_dir: str, constraint_type: str):
        """处理特定形态的数据"""

        for episode_dir in tqdm(os.listdir(data_dir), desc=f"Processing {constraint_type}"):
            if not episode_dir.startswith('episode_'):
                continue

            episode_path = os.path.join(data_dir, episode_dir)
            if not os.path.isdir(episode_path):
                continue

            episode_id = episode_dir
            original_episode_id = episode_id.split('episode_')[1].split('_')[0]

            # 处理每个段数配置
            for segment_dir in os.listdir(episode_path):
                if not segment_dir.endswith('_segments'):
                    continue

                segment_path = os.path.join(episode_path, segment_dir)
                if not os.path.isdir(segment_path):
                    continue

                try:
                    # 加载关节轨迹
                    trajectory_file = os.path.join(segment_path, 'joint_trajectory.npz')
                    config_file = os.path.join(segment_path, 'config.json')

                    if not os.path.exists(trajectory_file):
                        continue

                    # 加载数据
                    traj_data = np.load(trajectory_file)

                    # 基本信息
                    num_segments = int(segment_dir.split('_')[0])
                    robot_config = f"{segment_dir}_{constraint_type}"

                    # 加载配置
                    task_description = f"Complete manipulation task using {constraint_type} soft continuum arm with {segment_dir}"
                    if os.path.exists(config_file):
                        with open(config_file, 'r') as f:
                            config_data = json.load(f)
                            if 'task_description' in config_data:
                                task_description = str(config_data['task_description'])

                    # 提取轨迹数据
                    joint_positions = traj_data['joint_positions']  # (N, action_dim)
                    ee_positions = traj_data['end_effector_positions']  # (N, 3)
                    ee_orientations = traj_data['end_effector_orientations']  # (N, 3)
                    timestamps = traj_data['timestamps']
                    success_mask = traj_data['success_mask']

                    # 检查数据完整性
                    if len(joint_positions) != len(ee_positions) or len(joint_positions) == 0:
                        self.failed_episodes.append(f"{episode_id}/{segment_dir}: 数据长度不匹配")
                        continue

                    # 创建episode记录
                    episode_record = {
                        'episode_id': episode_id,
                        'original_episode_id': original_episode_id,
                        'robot_config': robot_config,
                        'constraint_type': constraint_type,
                        'num_segments': num_segments,
                        'action_dim': joint_positions.shape[1],
                        'sequence_length': len(joint_positions),
                        'task_description': task_description,

                        # 数据路径（相对于原始数据）
                        'trajectory_file': trajectory_file,
                        'config_file': config_file if os.path.exists(config_file) else None,

                        # 数据摘要
                        'joint_positions_shape': joint_positions.shape,
                        'ee_positions_shape': ee_positions.shape,
                        'success_rate': float(np.mean(success_mask)) if len(success_mask) > 0 else 0.0,
                        'temporal_smoothness': float(traj_data.get('temporal_smoothness', 0.0)),
                        'position_accuracy': float(traj_data.get('position_accuracy', 0.0)),

                        # 用于快速访问的缓存
                        'has_valid_images': False,  # 后续填充
                        'robot_graph_path': None,   # 后续填充
                    }

                    self.episodes.append(episode_record)

                except Exception as e:
                    self.failed_episodes.append(f"{episode_id}/{segment_dir}: {str(e)}")
                    continue

    def _verify_image_data(self):
        """验证图像数据可用性"""
        print("\n🖼️ 验证图像数据...")

        image_root = self.raw_paths['droid_images']
        valid_images = 0
        total_episodes = 0

        for episode in tqdm(self.episodes, desc="Checking images"):
            original_id = episode['original_episode_id']
            episode_image_dir = os.path.join(image_root, f"episode_{original_id}")

            total_episodes += 1

            if os.path.exists(episode_image_dir):
                # 检查相机视角
                camera_views = ['exterior_image_1_left', 'exterior_image_2_left', 'wrist_image_left']
                found_camera = False

                for camera in camera_views:
                    camera_dir = os.path.join(episode_image_dir, camera)
                    if os.path.exists(camera_dir):
                        images = [f for f in os.listdir(camera_dir) if f.endswith('.jpg')]
                        if len(images) > 0:
                            episode['image_camera'] = camera
                            episode['image_count'] = len(images)
                            episode['has_valid_images'] = True
                            found_camera = True
                            valid_images += 1
                            break

                if not found_camera:
                    episode['has_valid_images'] = False
            else:
                episode['has_valid_images'] = False

        print(f"✅ 图像数据: {valid_images}/{total_episodes} episodes有效")

    def _process_robot_graphs(self):
        """处理机器人图数据"""
        print("\n🕸️ 处理机器人图数据...")

        graph_root = self.raw_paths['robot_graphs']
        graph_cache = {}
        found_graphs = 0

        for episode in tqdm(self.episodes, desc="Loading robot graphs"):
            robot_config = episode['robot_config']
            num_segments = episode['num_segments']
            constraint_type = episode['constraint_type']

            # 查找对应的图文件
            # 软体臂图文件命名为 soft_arm_Nsegments_constraint.npz
            graph_file = os.path.join(graph_root, f"soft_arm_{num_segments}segments_{constraint_type}.npz")

            if os.path.exists(graph_file):
                episode['robot_graph_path'] = graph_file

                # 加载到缓存（避免重复加载）
                if graph_file not in graph_cache:
                    try:
                        graph_data = np.load(graph_file)
                        graph_cache[graph_file] = {
                            'node_features': graph_data['node_features'],  # (N, 19)
                            'edge_indices': graph_data['edge_indices'],    # (2, E)
                            'num_nodes': int(graph_data['num_nodes']),
                            'num_edges': int(graph_data['num_edges']),
                        }
                        found_graphs += 1
                    except Exception as e:
                        print(f"⚠️ 加载图失败: {graph_file}, 错误: {e}")
                        episode['robot_graph_path'] = None
            else:
                episode['robot_graph_path'] = None

        print(f"✅ 机器人图: 找到 {len(graph_cache)} 个唯一图文件")

        # 保存图缓存
        graph_cache_path = os.path.join(self.processed_paths['root'], 'processed', 'graph_cache.pkl')
        with open(graph_cache_path, 'wb') as f:
            pickle.dump(graph_cache, f)
        print(f"✅ 图缓存已保存: {graph_cache_path}")

    def _create_data_splits(self):
        """创建训练/验证数据分割"""
        print("\n📂 创建数据分割...")

        # 只使用有效数据
        valid_episodes = [ep for ep in self.episodes
                         if ep['has_valid_images'] and ep['robot_graph_path'] is not None]

        print(f"📊 有效episodes: {len(valid_episodes)}/{len(self.episodes)}")

        # 按原始episode分割，避免数据泄露
        unique_original_ids = list(set(ep['original_episode_id'] for ep in valid_episodes))
        unique_original_ids.sort()

        # 80-20分割
        split_idx = int(len(unique_original_ids) * 0.8)
        train_original_ids = set(unique_original_ids[:split_idx])
        val_original_ids = set(unique_original_ids[split_idx:])

        train_episodes = []
        val_episodes = []

        for ep in valid_episodes:
            if ep['original_episode_id'] in train_original_ids:
                train_episodes.append(ep)
            else:
                val_episodes.append(ep)

        print(f"📊 训练集: {len(train_episodes)} episodes")
        print(f"📊 验证集: {len(val_episodes)} episodes")

        # 保存分割信息
        train_split_path = os.path.join(self.processed_paths['root'], 'processed', 'train_episodes.json')
        val_split_path = os.path.join(self.processed_paths['root'], 'processed', 'val_episodes.json')

        with open(train_split_path, 'w') as f:
            json.dump(train_episodes, f, indent=2)

        with open(val_split_path, 'w') as f:
            json.dump(val_episodes, f, indent=2)

        print(f"✅ 训练分割已保存: {train_split_path}")
        print(f"✅ 验证分割已保存: {val_split_path}")

        self.train_episodes = train_episodes
        self.val_episodes = val_episodes

    def _save_unified_data(self):
        """保存统一的HDF5格式数据"""
        print("\n💾 保存统一数据格式...")

        h5_path = os.path.join(self.processed_paths['root'], 'processed', 'unified_episodes.h5')

        with h5py.File(h5_path, 'w') as h5f:
            # 创建组
            train_group = h5f.create_group('train')
            val_group = h5f.create_group('val')

            # 保存训练数据
            self._save_episodes_to_h5(train_group, self.train_episodes, 'train')

            # 保存验证数据
            self._save_episodes_to_h5(val_group, self.val_episodes, 'val')

        print(f"✅ 统一数据已保存: {h5_path}")

    def _save_episodes_to_h5(self, group: h5py.Group, episodes: List[Dict], split_name: str):
        """保存episodes到HDF5组"""

        for i, episode in enumerate(tqdm(episodes, desc=f"Saving {split_name} data")):
            ep_group = group.create_group(f"episode_{i}")

            # 保存元数据
            for key, value in episode.items():
                if isinstance(value, (str, int, float, bool)):
                    ep_group.attrs[key] = value
                elif isinstance(value, (list, tuple)) and len(value) > 0:
                    ep_group.attrs[key] = json.dumps(value)

            # 加载并保存轨迹数据
            try:
                traj_data = np.load(episode['trajectory_file'])
                ep_group.create_dataset('joint_positions', data=traj_data['joint_positions'])
                ep_group.create_dataset('ee_positions', data=traj_data['end_effector_positions'])
                ep_group.create_dataset('ee_orientations', data=traj_data['end_effector_orientations'])
                ep_group.create_dataset('timestamps', data=traj_data['timestamps'])
                ep_group.create_dataset('success_mask', data=traj_data['success_mask'])
            except Exception as e:
                print(f"⚠️ 保存轨迹数据失败: {episode['episode_id']}, 错误: {e}")

    def _generate_statistics(self):
        """生成数据统计信息"""
        print("\n📈 生成数据统计...")

        stats = {
            'total_episodes': len(self.episodes),
            'valid_episodes': len(self.train_episodes) + len(self.val_episodes),
            'train_episodes': len(self.train_episodes),
            'val_episodes': len(self.val_episodes),
            'failed_episodes': len(self.failed_episodes),

            'constraint_types': {},
            'segment_counts': {},
            'action_dimensions': {},
            'sequence_lengths': [],
            'success_rates': [],
        }

        # 统计各种维度
        valid_episodes = self.train_episodes + self.val_episodes

        for ep in valid_episodes:
            # 约束类型
            constraint = ep['constraint_type']
            stats['constraint_types'][constraint] = stats['constraint_types'].get(constraint, 0) + 1

            # 段数
            segments = ep['num_segments']
            stats['segment_counts'][segments] = stats['segment_counts'].get(segments, 0) + 1

            # 动作维度
            action_dim = ep['action_dim']
            stats['action_dimensions'][action_dim] = stats['action_dimensions'].get(action_dim, 0) + 1

            # 序列长度和成功率
            stats['sequence_lengths'].append(ep['sequence_length'])
            stats['success_rates'].append(ep['success_rate'])

        # 计算统计量
        if stats['sequence_lengths']:
            stats['avg_sequence_length'] = np.mean(stats['sequence_lengths'])
            stats['avg_success_rate'] = np.mean(stats['success_rates'])

        # 保存统计
        stats_path = os.path.join(self.processed_paths['root'], 'processed', 'statistics.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2, default=str)

        print(f"✅ 统计信息已保存: {stats_path}")

        # 打印关键统计
        print("\n📊 数据统计摘要:")
        print(f"  总episodes: {stats['total_episodes']}")
        print(f"  有效episodes: {stats['valid_episodes']}")
        print(f"  训练episodes: {stats['train_episodes']}")
        print(f"  验证episodes: {stats['val_episodes']}")
        print(f"  平均序列长度: {stats.get('avg_sequence_length', 0):.1f}")
        print(f"  平均成功率: {stats.get('avg_success_rate', 0):.3f}")
        print(f"  约束类型分布: {stats['constraint_types']}")
        print(f"  段数分布: {stats['segment_counts']}")

def main():
    """主函数"""

    # 加载配置
    config_path = os.path.join(PROJECT_ROOT, 'configs', 'data_paths.yaml')
    config = load_config(config_path)

    # 创建处理器
    processor = SoftArmDataProcessor(config)

    # 开始处理
    processor.process_all_data()

    print("\n🎉 数据预处理完成!")
    print("=" * 50)
    print("下一步:")
    print("1. 检查生成的统计信息")
    print("2. 运行训练脚本测试")
    print("3. 开始模型训练")

if __name__ == "__main__":
    main()