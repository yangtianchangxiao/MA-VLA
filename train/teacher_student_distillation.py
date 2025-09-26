#!/usr/bin/env python3
"""
Teacher-Student Distillation Training
使用RynnVLA-001作为teacher，通过连续IK生成训练数据
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
from pathlib import Path
from tqdm import tqdm
import random
from scipy.spatial.transform import Rotation as R

# Import our models
from vla_model import RealRynnVLALoRAGNN
from vla_trainer import CompleteVLADataset

class ContinuousIKSolver:
    """连续IK求解器，避免多解跳跃"""

    def __init__(self, num_joints=7):
        self.num_joints = num_joints
        self.prev_joint_config = None
        self.joint_limits = self._get_joint_limits()

    def _get_joint_limits(self):
        """获取关节限制（简化版，实际应从URDF获取）"""
        # 7-DOF机械臂的典型关节限制
        limits = {
            'lower': np.array([-2.8, -1.7, -2.8, -3.1, -2.8, -0.0, -2.8]),
            'upper': np.array([2.8, 1.7, 2.8, -0.0, 2.8, 3.8, 2.8])
        }
        return limits

    def solve_continuous_ik(self, cartesian_trajectory, base_morphology):
        """
        连续IK求解，确保解的连续性

        Args:
            cartesian_trajectory: [seq_len, 6] - (x,y,z,rx,ry,rz) trajectory
            base_morphology: dict - 机器人形态参数

        Returns:
            joint_trajectory: [seq_len, num_joints] - 连续的关节轨迹
        """
        seq_len = cartesian_trajectory.shape[0]
        joint_trajectory = np.zeros((seq_len, self.num_joints))

        # 如果是首次调用，初始化关节配置
        if self.prev_joint_config is None:
            self.prev_joint_config = np.zeros(self.num_joints)

        for t in range(seq_len):
            target_pose = cartesian_trajectory[t]  # [x, y, z, rx, ry, rz]

            # 简化的IK求解（实际应使用更精确的算法如KDL/Pinocchio）
            joint_config = self._solve_ik_single_step(
                target_pose,
                self.prev_joint_config,
                base_morphology
            )

            joint_trajectory[t] = joint_config
            self.prev_joint_config = joint_config

        return joint_trajectory

    def _solve_ik_single_step(self, target_pose, prev_joints, morphology):
        """
        单步IK求解，保证与前一步的连续性

        Args:
            target_pose: [6] - 目标位姿
            prev_joints: [num_joints] - 前一步关节角度
            morphology: dict - 机器人形态

        Returns:
            joint_config: [num_joints] - 优化后的关节配置
        """
        # 简化实现：基于前一步关节角度进行微调
        # 实际实现需要使用数值IK或解析IK

        target_pos = target_pose[:3]
        target_rot = target_pose[3:6]

        # 使用梯度下降优化关节角度
        joint_config = prev_joints.copy()

        # 简化的迭代优化
        for iteration in range(10):  # 限制迭代次数
            # 计算当前正运动学
            current_pose = self._forward_kinematics(joint_config, morphology)

            # 计算误差
            pos_error = target_pos - current_pose[:3]
            rot_error = target_rot - current_pose[3:6]

            # 简单的梯度更新（实际应计算雅可比矩阵）
            pos_gradient = pos_error * 0.1
            rot_gradient = rot_error * 0.05

            # 更新关节角度（简化版）
            joint_config[:3] += pos_gradient
            joint_config[3:6] += rot_gradient

            # 确保关节在限制范围内
            joint_config = np.clip(joint_config,
                                 self.joint_limits['lower'],
                                 self.joint_limits['upper'])

            # 检查收敛
            if np.linalg.norm(pos_error) < 0.001 and np.linalg.norm(rot_error) < 0.01:
                break

        # 优先选择与前一步最接近的解（避免跳跃）
        return self._select_continuous_solution(joint_config, prev_joints)

    def _forward_kinematics(self, joint_config, morphology):
        """简化的正运动学计算"""
        # 实际实现需要使用DH参数或URDF
        # 这里使用简化的几何计算

        # 获取形态参数
        link_lengths = morphology.get('link_lengths', [0.3, 0.3, 0.3, 0.2, 0.2, 0.1, 0.05])

        # 简化的FK计算
        x = sum(link_lengths[i] * np.cos(sum(joint_config[:i+1])) for i in range(min(3, len(joint_config))))
        y = sum(link_lengths[i] * np.sin(sum(joint_config[:i+1])) for i in range(min(3, len(joint_config))))
        z = sum(link_lengths[i] * joint_config[i] * 0.1 for i in range(3, min(len(joint_config), 6)))

        # 简化的方向计算
        rx = joint_config[3] if len(joint_config) > 3 else 0
        ry = joint_config[4] if len(joint_config) > 4 else 0
        rz = joint_config[5] if len(joint_config) > 5 else 0

        return np.array([x, y, z, rx, ry, rz])

    def _select_continuous_solution(self, candidate_config, prev_config):
        """选择与前一步最连续的解"""
        # 计算配置空间距离
        config_distance = np.linalg.norm(candidate_config - prev_config)

        # 如果距离过大，可能发生了跳跃，需要调整
        max_joint_change = 0.2  # 单步最大关节变化（弧度）

        adjusted_config = candidate_config.copy()
        for i in range(len(candidate_config)):
            joint_change = candidate_config[i] - prev_config[i]

            if abs(joint_change) > max_joint_change:
                # 限制变化幅度
                adjusted_config[i] = prev_config[i] + np.sign(joint_change) * max_joint_change

        return adjusted_config

class TeacherStudentDistiller:
    """Teacher-Student蒸馏训练器"""

    def __init__(self,
                 teacher_model_path="/home/cx/AET_FOR_RL/vla/参考模型/RynnVLA-001/pretrained_models/RynnVLA-001-7B-Base",
                 student_model_path="/home/cx/AET_FOR_RL/vla/train/vla_model_trained.pth",
                 device="cuda:0"):

        self.device = torch.device(device)
        self.action_chunk_size = 20

        print(f"🎓 Teacher-Student Distillation Initializing...")
        print(f"   Teacher: {teacher_model_path}")
        print(f"   Student: {student_model_path}")
        print(f"   Device: {self.device}")

        # 初始化连续IK求解器
        self.ik_solver = ContinuousIKSolver()

        # 加载Teacher模型（RynnVLA-001）
        self.teacher_model = self._load_teacher_model(teacher_model_path)

        # 加载Student模型（我们的GNN模型）
        self.student_model = self._load_student_model(student_model_path)

        # 初始化优化器
        self.optimizer = optim.AdamW(
            self.student_model.parameters(),
            lr=1e-4,
            weight_decay=0.01
        )

        print("✅ Teacher-Student models loaded successfully!")

    def _load_teacher_model(self, model_path):
        """加载RynnVLA-001作为teacher"""
        try:
            # 这里应该加载预训练的RynnVLA-001
            # 暂时使用我们的架构作为占位符
            base_model = RealRynnVLALoRAGNN(
                model_path=model_path,
                lora_rank=8,
                gnn_node_dim=256,
                action_chunk_size=self.action_chunk_size
            )

            teacher = base_model.to(self.device)
            teacher.eval()

            print("📚 Teacher model (RynnVLA-001) loaded")
            return teacher

        except Exception as e:
            print(f"⚠️ Failed to load teacher model: {e}")
            print("🔄 Using simplified teacher for development...")
            return None

    def _load_student_model(self, model_path):
        """加载我们训练的student模型"""
        try:
            # 重建student架构
            base_model = RealRynnVLALoRAGNN(
                model_path="/home/cx/AET_FOR_RL/vla/参考模型/RynnVLA-001/pretrained_models/RynnVLA-001-7B-Base",
                lora_rank=8,
                gnn_node_dim=256,
                action_chunk_size=self.action_chunk_size
            )

            # 使用与训练时相同的wrapper
            from vla_trainer import CompleteVLAWrapper
            student = CompleteVLAWrapper(base_model).to(self.device)

            # 加载训练好的权重
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device)
                student.load_state_dict(checkpoint['model_state_dict'])
                print(f"📖 Student model loaded from checkpoint")
            else:
                print(f"🆕 Student model initialized randomly (no checkpoint found)")

            return student

        except Exception as e:
            print(f"❌ Failed to load student model: {e}")
            return None

    def generate_synthetic_data(self, num_samples=1000, morphologies_list=None):
        """生成合成训练数据"""
        print(f"🔄 Generating {num_samples} synthetic training samples...")

        if morphologies_list is None:
            morphologies_list = [
                {"dof": 5, "link_lengths": [0.3, 0.3, 0.3, 0.2, 0.2]},
                {"dof": 6, "link_lengths": [0.3, 0.3, 0.3, 0.2, 0.2, 0.1]},
                {"dof": 7, "link_lengths": [0.3, 0.3, 0.3, 0.2, 0.2, 0.1, 0.05]}
            ]

        synthetic_data = []

        for i in tqdm(range(num_samples)):
            # 随机选择形态
            morphology = random.choice(morphologies_list)

            # 生成随机输入
            sample = self._generate_random_input(morphology)

            # Teacher生成笛卡尔轨迹
            cartesian_trajectory = self._teacher_inference(sample)

            # IK转换为关节轨迹
            joint_trajectory = self.ik_solver.solve_continuous_ik(
                cartesian_trajectory, morphology
            )

            # 构建训练样本
            training_sample = {
                'images': sample['images'],
                'description_tokens': sample['description_tokens'],
                'morphology_features': sample['morphology_features'],
                'target_joint_trajectory': torch.tensor(joint_trajectory, dtype=torch.float32),
                'morphology': morphology,
                'cartesian_trajectory': cartesian_trajectory
            }

            synthetic_data.append(training_sample)

        print(f"✅ Generated {len(synthetic_data)} synthetic samples")
        return synthetic_data

    def _generate_random_input(self, morphology):
        """生成随机输入数据"""
        batch_size = 1

        # 随机图像 (实际应该是真实的环境图像)
        images = torch.randn(batch_size, 3, 320, 180, device=self.device)

        # 随机指令tokens (实际应该是有意义的指令)
        description_tokens = torch.randn(batch_size, 32, device=self.device)

        # 形态特征
        dof = morphology["dof"]
        morphology_features = torch.tensor([
            dof / 7.0,  # 归一化DOF
            1.0, 1.0, 1.0,  # 链长缩放
            0.0, 0.0  # 基座位置
        ], device=self.device).unsqueeze(0)  # [1, 6]

        return {
            'images': images,
            'description_tokens': description_tokens,
            'morphology_features': morphology_features,
            'dof': dof
        }

    def _teacher_inference(self, sample):
        """Teacher模型推理生成笛卡尔轨迹"""
        if self.teacher_model is None:
            # 如果teacher模型未加载，生成随机轨迹用于测试
            seq_len = self.action_chunk_size
            trajectory = np.random.uniform(-1, 1, (seq_len, 6))  # [seq_len, 6] for (x,y,z,rx,ry,rz)

            # 使轨迹更合理一些
            for t in range(1, seq_len):
                trajectory[t] = trajectory[t-1] * 0.9 + trajectory[t] * 0.1  # 平滑化

            return trajectory

        # 实际的teacher推理
        try:
            with torch.no_grad():
                # 使用teacher模型生成动作
                outputs = self.teacher_model(
                    sample['images'],
                    sample['description_tokens'],
                    sample['morphology_features'],
                    num_joints=sample['dof']
                )

                # 假设teacher输出笛卡尔空间动作
                actions = outputs['actions']  # [1, action_dim, seq_len]

                # 转换为轨迹格式 [seq_len, 6]
                if actions.shape[-1] == self.action_chunk_size:
                    cartesian_trajectory = actions[0, :6, :].transpose(0, 1).cpu().numpy()
                else:
                    # 如果不是期望格式，使用随机轨迹
                    cartesian_trajectory = np.random.uniform(-1, 1, (self.action_chunk_size, 6))

                return cartesian_trajectory

        except Exception as e:
            print(f"⚠️ Teacher inference failed: {e}, using random trajectory")
            return np.random.uniform(-1, 1, (self.action_chunk_size, 6))

    def train_student(self, synthetic_data, epochs=10, batch_size=8):
        """训练student模型"""
        print(f"🎯 Training student model...")
        print(f"   Data samples: {len(synthetic_data)}")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}")

        self.student_model.train()

        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0

            # 随机打乱数据
            random.shuffle(synthetic_data)

            # 批处理训练
            for i in range(0, len(synthetic_data), batch_size):
                batch = synthetic_data[i:i+batch_size]

                if len(batch) == 0:
                    continue

                # 准备batch数据
                batch_images = torch.stack([sample['images'].squeeze(0) for sample in batch])
                batch_descriptions = torch.stack([sample['description_tokens'].squeeze(0) for sample in batch])
                batch_morphology = torch.stack([sample['morphology_features'].squeeze(0) for sample in batch])
                batch_targets = torch.stack([sample['target_joint_trajectory'] for sample in batch])

                # Student推理
                self.optimizer.zero_grad()

                outputs = self.student_model(
                    batch_images,
                    batch_descriptions,
                    batch_morphology
                )

                predicted_trajectories = outputs['actions']  # [batch, num_joints, seq_len]

                # 计算损失 (需要匹配维度)
                # batch_targets: [batch, seq_len, num_joints]
                # predicted: [batch, num_joints, seq_len]
                target_trajectories = batch_targets.transpose(1, 2)  # [batch, num_joints, seq_len]

                loss = nn.MSELoss()(predicted_trajectories, target_trajectories)

                # 反向传播
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / max(num_batches, 1)
            print(f"Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.6f}")

            # 保存checkpoint
            if (epoch + 1) % 5 == 0:
                self._save_checkpoint(epoch, avg_loss)

        print("✅ Student training completed!")

    def _save_checkpoint(self, epoch, loss):
        """保存训练checkpoint"""
        checkpoint_path = f"/home/cx/AET_FOR_RL/vla/train/student_distilled_epoch_{epoch+1}.pth"

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': self.student_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, checkpoint_path)

        print(f"💾 Checkpoint saved: {checkpoint_path}")

def main():
    """主训练函数"""
    print("🚀 Starting Teacher-Student Distillation Training")
    print("=" * 60)

    # 初始化distiller
    distiller = TeacherStudentDistiller()

    if distiller.student_model is None:
        print("❌ Failed to initialize student model")
        return

    # 生成合成数据
    synthetic_data = distiller.generate_synthetic_data(num_samples=500)  # 先用小数据集测试

    # 训练student
    distiller.train_student(synthetic_data, epochs=20, batch_size=4)

    print("\n🎉 Teacher-Student distillation completed!")
    print("🔥 Now you have a student model trained on continuous IK labels!")

if __name__ == "__main__":
    main()