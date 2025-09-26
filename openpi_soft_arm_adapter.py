#!/usr/bin/env python3
"""
OpenPi软体臂数据适配器
将我们的软体臂合成数据适配为OpenPi可训练的格式，支持8卡训练
"""

import os
import json
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from typing import Dict, List, Iterator, Optional
import pandas as pd

class SoftArmOpenPiDataset:
    """软体臂数据集，适配OpenPi训练框架"""

    def __init__(self,
                 soft_arm_data_dir: str = "/home/cx/AET_FOR_RL/vla/synthesized_data/soft_arm_4dof_synthesis",
                 droid_images_dir: str = "/home/cx/AET_FOR_RL/vla/valid_original_data/droid_100/extracted_images",
                 droid_parquet_path: str = "/home/cx/AET_FOR_RL/vla/converted_data/droid_100_fixed/data/chunk-000/file-000.parquet",
                 task_descriptions_path: str = "/home/cx/AET_FOR_RL/vla/valid_original_data/droid_100/task_descriptions.json",
                 batch_size: int = 8,
                 action_chunk_size: int = 16,
                 max_sequence_length: int = 50):

        print(f"🔧 初始化软体臂OpenPi数据集")
        print(f"   软体臂数据: {soft_arm_data_dir}")
        print(f"   DROID图像: {droid_images_dir}")

        self.soft_arm_data_dir = Path(soft_arm_data_dir)
        self.droid_images_dir = Path(droid_images_dir)
        self.batch_size = batch_size
        self.action_chunk_size = action_chunk_size
        self.max_sequence_length = max_sequence_length

        # 加载DROID元数据
        self.droid_df = pd.read_parquet(droid_parquet_path)

        # 加载任务描述
        if os.path.exists(task_descriptions_path):
            with open(task_descriptions_path, 'r') as f:
                task_data = json.load(f)
                self.task_descriptions = {
                    str(ep): desc for ep, desc in zip(
                        task_data.get('valid_episode_list', []),
                        task_data.get('task_descriptions', [])
                    )
                }
        else:
            print(f"⚠️ 任务描述文件不存在，使用默认描述")
            self.task_descriptions = {}

        # 扫描软体臂配置
        self.samples = self._scan_soft_arm_data()
        print(f"✅ 数据集就绪: {len(self.samples)} 个样本")

    def _scan_soft_arm_data(self) -> List[Dict]:
        """扫描所有软体臂配置数据"""
        samples = []

        if not self.soft_arm_data_dir.exists():
            print(f"❌ 软体臂数据目录不存在: {self.soft_arm_data_dir}")
            return samples

        # 遍历所有episode
        for episode_dir in self.soft_arm_data_dir.glob("episode_*"):
            original_episode = int(episode_dir.name.split('_')[1])

            # 遍历所有配置（2,3,4,5段）
            for config_dir in episode_dir.glob("*_segments"):
                segments = int(config_dir.name.split('_')[0])

                # 检查必要文件
                traj_file = config_dir / "joint_trajectory.npz"
                config_file = config_dir / "config.json"

                if traj_file.exists() and config_file.exists():
                    samples.append({
                        'original_episode': original_episode,
                        'segments': segments,
                        'trajectory_path': traj_file,
                        'config_path': config_file,
                        'episode_dir': episode_dir.name,
                        'config_dir': config_dir.name
                    })

        return samples

    def _load_soft_arm_trajectory(self, traj_path: Path) -> Dict:
        """加载软体臂轨迹数据"""
        data = np.load(traj_path)
        return {
            'joint_positions': data['joint_positions'],           # (T, N) 关节角度
            'timestamps': data['timestamps'],                     # (T,) 时间戳
            'end_effector_positions': data['end_effector_positions'], # (T, 3) 末端位置
            'success_mask': data['success_mask'],                 # (T,) 成功掩码
            'constraint_type': str(data.get('constraint_type', '3DOF'))
        }

    def _load_droid_images(self, original_episode: int, num_frames: int) -> torch.Tensor:
        """加载对应episode的DROID图像"""
        episode_image_dir = self.droid_images_dir / f"episode_{original_episode:03d}"

        # 优先使用exterior_image_1_left（外部视角）
        camera_dirs = ['exterior_image_1_left', 'exterior_image_2_left', 'wrist_image_left']

        for camera_dir in camera_dirs:
            image_dir = episode_image_dir / camera_dir
            if image_dir.exists():
                break
        else:
            # 创建fallback图像
            print(f"⚠️ Episode {original_episode} 图像不存在，使用fallback")
            return self._create_fallback_images(num_frames)

        # 加载图像序列
        images = []
        image_files = sorted(image_dir.glob("frame_*.jpg"))

        for i in range(min(num_frames, len(image_files))):
            if i < len(image_files):
                img_path = image_files[i]
                img = Image.open(img_path).convert('RGB')
                img = img.resize((224, 224))  # OpenPi标准尺寸
                img_array = np.array(img) / 255.0
                images.append(img_array)
            else:
                # 重复最后一帧
                images.append(images[-1] if images else np.zeros((224, 224, 3)))

        return torch.tensor(np.stack(images), dtype=torch.float32)  # (T, H, W, C)

    def _create_fallback_images(self, num_frames: int) -> torch.Tensor:
        """创建fallback图像序列"""
        # 简单的结构化图像
        fallback = np.zeros((num_frames, 224, 224, 3))

        for t in range(num_frames):
            img = np.zeros((224, 224, 3))
            # 桌面 (棕色)
            img[150:, :] = [0.4, 0.3, 0.2]
            # 机械臂 (灰色)
            img[80:160, 50:150] = [0.5, 0.5, 0.5]
            # 物体 (红色，随时间移动)
            obj_x = int(100 + 20 * np.sin(t * 0.1))
            img[120:140, obj_x:obj_x+20] = [0.8, 0.2, 0.2]

            fallback[t] = img

        return torch.tensor(fallback, dtype=torch.float32)

    def _get_task_description(self, original_episode: int) -> str:
        """获取任务描述"""
        episode_key = str(original_episode)
        if episode_key in self.task_descriptions:
            return self.task_descriptions[episode_key]
        else:
            return "Complete the manipulation task with the soft continuum arm"

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        """获取单个训练样本，OpenPi格式"""
        sample = self.samples[idx]

        # 加载软体臂轨迹
        trajectory = self._load_soft_arm_trajectory(sample['trajectory_path'])
        joint_positions = trajectory['joint_positions']  # (T, N)

        # 限制序列长度
        seq_length = min(len(joint_positions), self.max_sequence_length)
        joint_positions = joint_positions[:seq_length]  # (seq_len, N)

        # 加载配置信息
        with open(sample['config_path'], 'r') as f:
            config = json.load(f)

        # 加载对应的DROID图像
        images = self._load_droid_images(sample['original_episode'], seq_length)  # (seq_len, H, W, C)

        # 获取任务描述
        task_description = self._get_task_description(sample['original_episode'])

        # 转换为OpenPi期望的格式
        return {
            # 视觉输入
            'image': images,  # (seq_len, H, W, C) - OpenPi期望格式

            # 语言输入
            'language_instruction': task_description,

            # 动作输出 - 重塑为action chunks
            'action': self._format_actions_for_openpi(joint_positions),  # (seq_len, action_dim)

            # 元数据
            'episode_id': sample['original_episode'],
            'segments': sample['segments'],
            'constraint_type': trajectory['constraint_type'],
            'robot_config': {
                'n_segments': config['n_segments'],
                'segment_lengths': config['segment_lengths'],
                'base_offset': config['base_offset'],
            }
        }

    def _format_actions_for_openpi(self, joint_positions: np.ndarray) -> torch.Tensor:
        """将软体臂关节角度格式化为OpenPi动作格式"""
        # joint_positions: (seq_len, n_joints)
        seq_len, n_joints = joint_positions.shape

        # OpenPi期望的动作chunk格式
        # 这里我们简单地将关节角度作为目标动作
        actions = torch.tensor(joint_positions, dtype=torch.float32)

        # 如果需要，可以在这里添加gripper动作维度
        # actions = torch.cat([actions, torch.zeros(seq_len, 1)], dim=-1)  # 添加gripper

        return actions

def create_openpi_dataloader(batch_size: int = 8) -> torch.utils.data.DataLoader:
    """创建OpenPi兼容的数据加载器"""
    dataset = SoftArmOpenPiDataset(batch_size=batch_size)

    def collate_fn(batch):
        """自定义collate函数，处理变长序列"""
        # 按segments分组，避免不同DOF的张量混合
        segments_to_batch = {}
        for item in batch:
            segments = item['segments']
            if segments not in segments_to_batch:
                segments_to_batch[segments] = []
            segments_to_batch[segments].append(item)

        # 使用最大的组
        largest_group = max(segments_to_batch.keys(), key=lambda k: len(segments_to_batch[k]))
        selected_batch = segments_to_batch[largest_group]

        # 标准collate
        collated = {}
        for key in selected_batch[0].keys():
            if key in ['image', 'action']:
                # 堆叠张量数据
                collated[key] = torch.stack([item[key] for item in selected_batch])
            elif key in ['episode_id', 'segments']:
                # 数值数据
                collated[key] = torch.tensor([item[key] for item in selected_batch])
            else:
                # 列表数据
                collated[key] = [item[key] for item in selected_batch]

        return collated

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )

def test_dataset():
    """测试数据集加载"""
    print("🧪 测试软体臂OpenPi数据集")

    dataset = SoftArmOpenPiDataset()
    if len(dataset) == 0:
        print("❌ 数据集为空")
        return

    # 测试单个样本
    sample = dataset[0]
    print(f"✅ 样本格式:")
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"   {key}: {value.shape} {value.dtype}")
        elif isinstance(value, dict):
            print(f"   {key}: dict with {len(value)} keys")
        else:
            print(f"   {key}: {type(value)} = {str(value)[:50]}...")

    # 测试数据加载器
    dataloader = create_openpi_dataloader(batch_size=2)
    batch = next(iter(dataloader))
    print(f"✅ Batch格式:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"   {key}: {value.shape} {value.dtype}")
        else:
            print(f"   {key}: {type(value)} length={len(value) if hasattr(value, '__len__') else 'N/A'}")

if __name__ == "__main__":
    test_dataset()