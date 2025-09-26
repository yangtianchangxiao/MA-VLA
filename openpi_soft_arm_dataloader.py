#!/usr/bin/env python3
"""
OpenPi软体臂数据加载器
将软体臂NPZ数据转换为OpenPi训练格式
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Any, Optional
import cv2
from pathlib import Path

class SoftArmOpenPiDataset(Dataset):
    """软体臂数据集，兼容OpenPi训练格式"""

    def __init__(self,
                 soft_arm_data_dir: str = "/home/cx/AET_FOR_RL/vla/synthesized_data",
                 droid_images_dir: str = "/home/cx/AET_FOR_RL/vla/valid_original_data/droid_100/extracted_images",
                 image_size: Tuple[int, int] = (224, 224),
                 max_sequence_length: int = 50,
                 action_chunk_size: int = 16):

        self.image_size = image_size
        self.max_sequence_length = max_sequence_length
        self.action_chunk_size = action_chunk_size

        # 加载所有数据集
        self.episodes = []

        # 3DOF数据 (更多数据)
        dof3_dir = os.path.join(soft_arm_data_dir, "soft_arm_morphology_synthesis")
        if os.path.exists(dof3_dir):
            self._load_episodes(dof3_dir, "3DOF", droid_images_dir)

        # 4DOF数据
        dof4_dir = os.path.join(soft_arm_data_dir, "soft_arm_4dof_synthesis")
        if os.path.exists(dof4_dir):
            self._load_episodes(dof4_dir, "4DOF", droid_images_dir)

        print(f"✅ 加载了 {len(self.episodes)} 个训练序列")

        # 统计信息
        if len(self.episodes) > 0:
            total_timesteps = sum(len(ep['action_chunk']) for ep in self.episodes)
            print(f"   总时间步: {total_timesteps}")
            print(f"   平均序列长度: {total_timesteps / len(self.episodes):.1f}")
        else:
            print("   没有找到有效数据，请检查数据路径")

    def _load_episodes(self, data_dir: str, constraint_type: str, image_dir: str):
        """加载指定目录下的所有episode数据"""

        # 处理分层目录结构: episode_xxx/n_segments/joint_trajectory.npz
        for episode_dir in os.listdir(data_dir):
            if not episode_dir.startswith('episode_'):
                continue

            episode_path = os.path.join(data_dir, episode_dir)
            if not os.path.isdir(episode_path):
                continue

            episode_id = episode_dir  # episode_000, episode_001, etc.

            # 遍历每个段数配置
            for segment_dir in os.listdir(episode_path):
                if not segment_dir.endswith('_segments'):
                    continue

                segment_path = os.path.join(episode_path, segment_dir)
                if not os.path.isdir(segment_path):
                    continue

                # 加载关节轨迹文件
                trajectory_file = os.path.join(segment_path, 'joint_trajectory.npz')
                config_file = os.path.join(segment_path, 'config.json')

                if not os.path.exists(trajectory_file):
                    continue

                try:
                    # 加载轨迹数据
                    data = np.load(trajectory_file)

                    # 加载配置信息
                    robot_config = f"{segment_dir}_{constraint_type}"
                    task_description = f"Complete manipulation task using {constraint_type} soft continuum arm with {segment_dir}"

                    if os.path.exists(config_file):
                        with open(config_file, 'r') as f:
                            config_data = json.load(f)
                            if 'task_description' in config_data:
                                task_description = str(config_data['task_description'])
                            robot_config = f"{segment_dir}_{constraint_type}_{config_data.get('robot_id', 'default')}"

                    # 提取数据
                    actions = data['joint_positions']  # 软体臂关节角度 (N, action_dim)
                    ee_positions = data['end_effector_positions']  # (N, 3)
                    ee_orientations = data.get('end_effector_orientations', None)  # (N, 3)

                    # 为每个时间步创建训练样本
                    for i in range(len(actions)):
                        # 找到对应的DROID图像
                        image_path = self._find_matching_image(
                            episode_id, i, image_dir, ee_positions[i]
                        )

                        if image_path is None:
                            continue

                        # 创建动作chunk (当前+未来动作)
                        action_chunk = self._create_action_chunk(actions, i)

                        episode_data = {
                            'episode_id': episode_id,
                            'robot_config': robot_config,
                            'constraint_type': constraint_type,
                            'timestep': i,
                            'image_path': image_path,
                            'task_description': task_description,
                            'action_chunk': action_chunk,
                            'ee_position': ee_positions[i],
                            'ee_orientation': ee_orientations[i] if ee_orientations is not None else None,
                        }

                        self.episodes.append(episode_data)

                except Exception as e:
                    print(f"⚠️ 加载episode失败: {trajectory_file}, 错误: {e}")
                    continue

    def _find_matching_image(self, episode_id: str, timestep: int,
                           image_dir: str, ee_pos: np.ndarray) -> Optional[str]:
        """找到匹配的DROID图像"""

        # 从episode_id提取原始DROID episode号
        if 'episode_' in episode_id:
            original_episode = episode_id.split('episode_')[1].split('_')[0]
        else:
            original_episode = episode_id.split('_')[0]

        # 查找图像文件
        episode_dir = os.path.join(image_dir, f"episode_{original_episode}")
        if not os.path.exists(episode_dir):
            return None

        # 优先选择外部相机视角
        camera_views = ['exterior_image_1_left', 'exterior_image_2_left', 'wrist_image_left']

        for camera in camera_views:
            camera_dir = os.path.join(episode_dir, camera)
            if os.path.exists(camera_dir):
                # 使用时间步匹配或就近选择
                images = sorted([f for f in os.listdir(camera_dir) if f.endswith('.jpg')])
                if images:
                    # 简单的时间步匹配
                    img_idx = min(timestep, len(images) - 1)
                    return os.path.join(camera_dir, images[img_idx])

        return None

    def _create_action_chunk(self, actions: np.ndarray, start_idx: int) -> np.ndarray:
        """创建动作chunk用于训练"""

        end_idx = min(start_idx + self.action_chunk_size, len(actions))
        action_chunk = actions[start_idx:end_idx]

        # 如果chunk不够长，用最后一个动作填充
        if len(action_chunk) < self.action_chunk_size:
            last_action = action_chunk[-1]
            padding = np.tile(last_action, (self.action_chunk_size - len(action_chunk), 1))
            action_chunk = np.concatenate([action_chunk, padding], axis=0)

        return action_chunk

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        episode = self.episodes[idx]

        # 加载图像
        image = self._load_and_preprocess_image(episode['image_path'])

        # 动作数据
        actions = torch.from_numpy(episode['action_chunk']).float()

        return {
            'image': image,
            'instruction': episode['task_description'],
            'actions': actions,
            'robot_config': episode['robot_config'],
            'constraint_type': episode['constraint_type'],
            'episode_id': episode['episode_id'],
            'timestep': episode['timestep'],
        }

    def _load_and_preprocess_image(self, image_path: str) -> torch.Tensor:
        """加载和预处理图像"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                # 创建占位图像
                image = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, self.image_size)

            # 转换为tensor并归一化
            image = torch.from_numpy(image).float() / 255.0
            image = image.permute(2, 0, 1)  # HWC -> CHW

            return image

        except Exception as e:
            print(f"⚠️ 图像加载失败: {image_path}, 错误: {e}")
            # 返回占位图像
            return torch.zeros(3, self.image_size[0], self.image_size[1])

def create_soft_arm_dataloaders(
    batch_size: int = 8,
    num_workers: int = 4,
    train_split: float = 0.8,
    **dataset_kwargs
) -> Tuple[DataLoader, DataLoader]:
    """创建训练和验证数据加载器"""

    dataset = SoftArmOpenPiDataset(**dataset_kwargs)

    # 分割训练和验证集
    train_size = int(len(dataset) * train_split)
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    print(f"✅ 数据分割: 训练 {len(train_dataset)}, 验证 {len(val_dataset)}")

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    return train_loader, val_loader

if __name__ == "__main__":
    # 测试数据加载器
    print("🧪 测试软体臂数据加载器...")

    dataset = SoftArmOpenPiDataset()

    if len(dataset) > 0:
        sample = dataset[0]
        print(f"✅ 样本数据形状:")
        print(f"   图像: {sample['image'].shape}")
        print(f"   动作: {sample['actions'].shape}")
        print(f"   指令: {sample['instruction'][:100]}...")
        print(f"   机器人配置: {sample['robot_config']}")

        # 测试数据加载器
        train_loader, val_loader = create_soft_arm_dataloaders(batch_size=4)
        print(f"✅ 数据加载器创建成功")
        print(f"   训练批次: {len(train_loader)}")
        print(f"   验证批次: {len(val_loader)}")

        # 测试一个批次
        batch = next(iter(train_loader))
        print(f"✅ 批次数据:")
        print(f"   图像批次: {batch['image'].shape}")
        print(f"   动作批次: {batch['actions'].shape}")
        print(f"   指令数量: {len(batch['instruction'])}")

    else:
        print("❌ 没有找到有效数据，请检查数据路径")