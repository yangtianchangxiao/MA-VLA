# Morphology Synthesis Standard Format (MSSF) v1.0

## 🎯 设计原则
- **数据复用**: 图像和任务描述在morphology variations间共享
- **训练友好**: 直接支持PyTorch DataLoader
- **LeRobot兼容**: 可无损转换为LeRobot格式
- **可扩展**: 支持新的morphology transformation类型

## 📊 数据结构

### 主文件: `morphology_dataset.json`
```json
{
  "format_version": "MSSF-v1.0",
  "dataset_name": "droid_100_morphology_synthesis", 
  "creation_timestamp": "2025-09-12T01:45:00Z",
  
  "metadata": {
    "total_variations": 869,
    "base_episodes": 46,
    "transformations": ["dof_modification", "link_scaling"],
    "average_expansion_ratio": 18.9
  },
  
  "base_episodes": {
    "0": {
      "task_description": "Put the marker in the pot",
      "image_sequence_path": "images/episode_0/",
      "image_format": "png", 
      "fps": 15,
      "total_frames": 312,
      "original_trajectory": "trajectories/episode_0_original.json"
    }
  },
  
  "morphology_variations": [
    {
      "variation_id": "dof_mod_ep000_var00",
      "base_episode": 0,
      "transformation": {
        "type": "dof_modification",
        "original_dof": 7,
        "target_dof": 5,
        "removed_joints": ["joint_2", "joint_4"],
        "quality_score": 0.967
      },
      "trajectory_path": "trajectories/dof_mod_ep000_var00.json",
      "morphology_description": "5-DOF Franka robot (removed elbow and wrist joints)"
    },
    {
      "variation_id": "link_scale_ep000_var00", 
      "base_episode": 0,
      "transformation": {
        "type": "link_scaling",
        "link_scales": [1.0, 0.85, 1.2, 0.9, 1.1, 0.8, 1.15],
        "quality_score": 0.906
      },
      "trajectory_path": "trajectories/link_scale_ep000_var00.json",
      "morphology_description": "Scaled Franka robot (link ratios: 1.0,0.85,1.2,0.9,1.1,0.8,1.15)"
    }
  ]
}
```

### 轨迹文件格式: `trajectories/variation_id.json`
```json
{
  "variation_id": "dof_mod_ep000_var00",
  "total_timesteps": 312,
  "timestep_dt": 0.0667,
  "joint_names": ["joint_0", "joint_1", "joint_3", "joint_5", "joint_6"],
  "trajectory": {
    "joint_positions": [
      [0.1, -0.2, 0.3, -1.5, 0.8],  // timestep 0
      [0.12, -0.18, 0.31, -1.48, 0.79] // timestep 1
    ],
    "joint_velocities": [...],
    "joint_accelerations": [...],
    "end_effector_poses": [...]
  }
}
```

## 🔧 Python API设计

### Dataset Class
```python
class MorphologySynthesisDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path="morphology_dataset.json"):
        self.data = self._load_dataset(dataset_path)
        self.variations = self.data["morphology_variations"]
        self.base_episodes = self.data["base_episodes"]
        
    def __len__(self):
        return len(self.variations)
        
    def __getitem__(self, idx):
        variation = self.variations[idx]
        base_episode = self.base_episodes[str(variation["base_episode"])]
        
        # Load shared images (efficient)
        images = self._load_image_sequence(base_episode["image_sequence_path"])
        
        # Load variation-specific trajectory
        trajectory = self._load_trajectory(variation["trajectory_path"])
        
        return {
            "images": images,
            "task_description": base_episode["task_description"],
            "morphology_description": variation["morphology_description"], 
            "joint_trajectory": trajectory["joint_positions"],
            "transformation": variation["transformation"],
            "variation_id": variation["variation_id"]
        }
```

## 🔄 LeRobot转换器接口

```python
def convert_to_lerobot(mssf_dataset_path, output_path):
    """将MSSF格式转换为LeRobot标准格式"""
    mssf_data = load_mssf(mssf_dataset_path)
    
    lerobot_episodes = []
    episode_idx = 0
    
    for variation in mssf_data["morphology_variations"]:
        base_ep = mssf_data["base_episodes"][str(variation["base_episode"])]
        
        # 为每个variation创建独立episode
        lerobot_episode = {
            "episode_index": episode_idx,
            "task": base_ep["task_description"],
            "morphology": variation["morphology_description"],
            "frames": []
        }
        
        trajectory = load_trajectory(variation["trajectory_path"])
        images = load_images(base_ep["image_sequence_path"])
        
        for frame_idx, (image, action) in enumerate(zip(images, trajectory)):
            lerobot_episode["frames"].append({
                "frame_index": frame_idx,
                "observation": {"image": image, "state": get_state(frame_idx)},
                "action": action,
                "timestamp": frame_idx * 0.0667
            })
        
        lerobot_episodes.append(lerobot_episode)
        episode_idx += 1
    
    save_lerobot_format(lerobot_episodes, output_path)
```

## ✅ 优势分析

1. **存储效率**: 46个base episodes的图像被869个variations共享
2. **训练友好**: 直接支持PyTorch DataLoader和batching
3. **元数据丰富**: 包含transformation详情和quality scores
4. **版本控制**: format_version支持未来升级
5. **可转换性**: 无损转换为LeRobot或其他格式

## 🚀 使用示例

```python
# 训练使用
dataset = MorphologySynthesisDataset("morphology_dataset.json")
dataloader = DataLoader(dataset, batch_size=6, shuffle=True)

for batch in dataloader:
    images = batch["images"]           # (B, T, C, H, W)  
    actions = batch["joint_trajectory"] # (B, T, DOF)
    task_desc = batch["task_description"]
    morph_desc = batch["morphology_description"]
    
    # VLA训练
    loss = model(images, task_desc, morph_desc, actions)

# LeRobot转换
convert_to_lerobot("morphology_dataset.json", "lerobot_output/")
```

---
**版本**: v1.0  
**状态**: 草案，待训练验证
**更新日志**: 
- v1.0: 初始设计，支持DOF和Link scaling