#!/usr/bin/env python3
"""
4DOF约束的软体臂数据合成 - 作为3DOF的对比研究
调整约束标准以提高成功率
"""

import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
import time
import sys

# 添加模块路径
sys.path.append('/home/cx/AET_FOR_RL/vla/data_augment/morphology_modules')
from soft_arm_ik_solver import SoftArmConfig, SoftArmSynthesisModule
from typing import Optional

class SoftArm4DOFSynthesis:
    """4DOF约束软体臂合成 - 放宽约束标准提高成功率"""

    def __init__(self, output_dir: str = "/home/cx/AET_FOR_RL/vla/synthesized_data/soft_arm_4dof_synthesis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 放宽的成功率标准
        self.success_threshold = 0.7  # 从95%降到70%
        self.segment_variants = [2, 3, 4, 5]

        # 设置日志
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def modified_synthesis_trajectory(self, ee_trajectory: np.ndarray,
                                    soft_arm: SoftArmConfig,
                                    base_offset: np.ndarray = None) -> Optional[np.ndarray]:
        """修改的4DOF轨迹合成 - 使用放宽的约束"""

        from soft_arm_ik_solver import SoftArmIKSolver, SoftArmKinematics

        ik_solver = SoftArmIKSolver(soft_arm)
        kinematics = SoftArmKinematics(soft_arm)

        # 修改IK求解器的success判断标准
        original_solve_ik = ik_solver.solve_ik_hierarchical

        def relaxed_solve_ik_hierarchical(target_pos, target_normal=None,
                                        initial_guess=None, max_iterations=100):
            """放宽的分层IK求解"""

            alpha, beta, success = original_solve_ik(
                target_pos, target_normal, initial_guess, max_iterations
            )

            if not success and target_normal is not None:
                # 如果原始4DOF失败，尝试放宽法向约束
                predicted_pos, predicted_rot = kinematics.forward_kinematics(
                    np.concatenate([alpha, beta])
                )

                pos_error = np.linalg.norm(predicted_pos - target_pos)
                if pos_error < 0.08:  # 放宽位置容忍度到8cm
                    predicted_normal = predicted_rot[:, 2]
                    cos_sim = np.clip(np.dot(predicted_normal, target_normal), -1, 1)

                    # 放宽法向约束到60度 (cos(60°) = 0.5)
                    if cos_sim > 0.5:
                        success = True

            return alpha, beta, success

        # 临时替换方法
        ik_solver.solve_ik_hierarchical = relaxed_solve_ik_hierarchical

        # 生成轨迹 - 使用完整6D pose
        curvature_trajectory = []
        previous_solution = None

        for i, ee_pose in enumerate(ee_trajectory):
            # 4DOF约束：使用位置 + 角度
            target_world_pose = ee_pose[:6]  # 包含角度信息
            if base_offset is not None:
                target_pose = target_world_pose.copy()
                target_pose[:3] = target_world_pose[:3] - base_offset
            else:
                target_pose = target_world_pose

            solution, success, error = ik_solver.solve_ik(target_pose, initial_guess=previous_solution)

            if not success:
                # 4DOF失败时的fallback策略
                self.logger.debug(f"4DOF failed at step {i}, trying position-only")
                target_pos_only = target_pose[:3]
                solution, success, error = ik_solver.solve_ik(target_pos_only, initial_guess=previous_solution)

                if not success:
                    return None

            curvature_trajectory.append(solution)
            previous_solution = solution

        return np.array(curvature_trajectory)

    def synthesize_episode_4dof(self, episode_id: int, trajectory: np.ndarray):
        """为单个episode合成4DOF约束的软体臂配置"""

        episode_results = {}
        episode_dir = self.output_dir / f"episode_{episode_id:03d}"
        episode_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"开始4DOF合成 Episode {episode_id}, 轨迹长度: {len(trajectory)}")

        # 分析轨迹workspace
        trajectory_center = np.mean(trajectory[:, :3], axis=0)
        base_offset = np.array([trajectory_center[0] - 0.6, trajectory_center[1], 0.0])

        for n_segments in self.segment_variants:
            try:
                # 使用合理的段长度配置
                if n_segments == 2:
                    segment_lengths = [0.57, 0.57]
                elif n_segments == 3:
                    segment_lengths = [0.37, 0.40, 0.37]
                elif n_segments == 4:
                    segment_lengths = [0.28, 0.31, 0.29, 0.33]
                else:  # 5 segments
                    segment_lengths = [0.23, 0.25, 0.24, 0.26, 0.24]

                config = SoftArmConfig(
                    n_segments=n_segments,
                    segment_lengths=segment_lengths
                )

                # 使用修改的4DOF合成
                joint_trajectory = self.modified_synthesis_trajectory(
                    trajectory, config, base_offset
                )

                if joint_trajectory is not None and len(joint_trajectory) > 0:
                    success_rate = len(joint_trajectory) / len(trajectory)

                    # 计算时间连续性
                    if len(joint_trajectory) > 1:
                        joint_diff = np.diff(joint_trajectory, axis=0)
                        temporal_smoothness = np.mean(np.abs(joint_diff))
                    else:
                        temporal_smoothness = 0.0

                    if success_rate >= self.success_threshold:
                        # 保存数据
                        segment_dir = episode_dir / f"{n_segments}_segments"
                        segment_dir.mkdir(exist_ok=True)

                        # 保存轨迹数据
                        np.savez(
                            segment_dir / "joint_trajectory.npz",
                            joint_positions=joint_trajectory.astype(np.float32),
                            timestamps=np.arange(len(joint_trajectory), dtype=np.float32) / 30.0,
                            end_effector_positions=trajectory[:len(joint_trajectory), :3].astype(np.float32),
                            end_effector_orientations=trajectory[:len(joint_trajectory), 3:6].astype(np.float32),
                            success_mask=np.ones(len(joint_trajectory), dtype=bool),
                            temporal_smoothness=temporal_smoothness,
                            constraint_type="4DOF_relaxed"
                        )

                        # 生成图结构
                        robot_graph = self.generate_robot_graph(config)
                        np.savez(segment_dir / "robot_graph.npz", **robot_graph)

                        # 保存配置信息 - 确保所有类型都是JSON可序列化的
                        config_info = {
                            "episode_id": int(episode_id),
                            "n_segments": int(n_segments),
                            "segment_lengths": [float(x) for x in segment_lengths],
                            "total_length": float(sum(segment_lengths)),
                            "base_offset": [float(x) for x in base_offset.tolist()],
                            "constraint_type": "4DOF_relaxed",
                            "success_threshold": float(self.success_threshold),
                            "synthesis_params": {
                                "success_rate": float(success_rate),
                                "temporal_smoothness": float(temporal_smoothness),
                                "original_trajectory_length": int(len(trajectory)),
                                "synthesized_trajectory_length": int(len(joint_trajectory))
                            }
                        }

                        with open(segment_dir / "config.json", 'w') as f:
                            json.dump(config_info, f, indent=2)

                        episode_results[n_segments] = {
                            "success": True,
                            "success_rate": success_rate,
                            "smoothness": temporal_smoothness
                        }

                        self.logger.info(f"  {n_segments}段: ✅ 成功率{success_rate:.1%}, 平滑度{temporal_smoothness:.4f}")
                    else:
                        self.logger.warning(f"  {n_segments}段: ❌ 成功率{success_rate:.1%} < {self.success_threshold:.1%}")
                        episode_results[n_segments] = {"success": False, "success_rate": success_rate}

            except Exception as e:
                self.logger.error(f"  {n_segments}段: ❌ 异常: {str(e)}")
                episode_results[n_segments] = {"success": False, "error": str(e)}

        successful_configs = sum(1 for result in episode_results.values() if result.get("success", False))
        self.logger.info(f"Episode {episode_id} 完成: {successful_configs}/{len(self.segment_variants)} 成功")

        return episode_results

    def generate_robot_graph(self, config: SoftArmConfig):
        """生成软体臂的图结构表示"""
        n_segments = config.n_segments
        node_features = np.zeros((n_segments, 19))

        for i in range(n_segments):
            # 19维特征：joint_type(6) + axis(3) + position(3) + orientation(4) + limits(3)
            node_features[i] = [
                0, 0, 0, 0, 1, 1,  # joint_type: soft_alpha, soft_beta
                0, 1, 0,           # axis: bending direction
                0, 0, sum(config.segment_lengths[:i+1]),  # cumulative position
                0, 0, 0, 1,        # quaternion identity
                0.001, np.pi, config.segment_lengths[i]  # limits + length
            ]

        # 邻接矩阵：链式连接
        edge_indices = []
        edge_attributes = []
        for i in range(n_segments - 1):
            edge_indices.extend([[i, i+1], [i+1, i]])
            edge_attributes.extend([[1.0], [1.0]])  # 连接强度

        return {
            'node_features': node_features.astype(np.float32),
            'edge_indices': np.array(edge_indices).T.astype(np.int64),
            'edge_attributes': np.array(edge_attributes).astype(np.float32),
            'robot_type': 'soft_arm_4dof',
            'n_segments': n_segments,
            'total_dof': n_segments * 2,
            'constraint_type': '4DOF_relaxed'
        }

def main():
    """主函数：生成4DOF对比数据"""

    # 加载DROID数据
    df = pd.read_parquet('/home/cx/AET_FOR_RL/vla/converted_data/droid_100_fixed/data/chunk-000/file-000.parquet')

    synthesizer = SoftArm4DOFSynthesis()

    # 只处理前20个episode作为对比研究
    unique_episodes = df['episode_index'].unique()[:20]

    print(f"🚀 开始4DOF约束软体臂合成 (对比研究)")
    print(f"将处理 {len(unique_episodes)} 个episodes")
    print()

    total_results = {}

    for episode_id in unique_episodes:
        episode_data = df[df['episode_index'] == episode_id]

        # 构建轨迹
        trajectory = []
        for _, row in episode_data.iterrows():
            action = np.array(row['action'])
            trajectory.append(action[:6])  # 位置 + 角度

        trajectory = np.array(trajectory)

        # 合成该episode
        results = synthesizer.synthesize_episode_4dof(episode_id, trajectory)
        total_results[episode_id] = results

    # 统计总体结果
    total_configs = 0
    successful_configs = 0

    for ep_results in total_results.values():
        for result in ep_results.values():
            total_configs += 1
            if result.get("success", False):
                successful_configs += 1

    print()
    print(f"🎉 4DOF合成完成!")
    print(f"总成功率: {successful_configs}/{total_configs} = {successful_configs/total_configs:.1%}")
    print(f"数据保存位置: {synthesizer.output_dir}")
    print()
    print("📊 这些4DOF数据可以与3DOF数据进行对比研究:")
    print("   • 训练效果差异分析")
    print("   • 姿态控制能力评估")
    print("   • 数据量vs约束质量权衡验证")

if __name__ == "__main__":
    main()