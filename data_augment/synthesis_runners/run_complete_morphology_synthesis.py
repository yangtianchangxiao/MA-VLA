#!/usr/bin/env python3
"""
Complete Morphology Synthesis Runner
使用length augmentation + 随机机器人配置 + IK合成

正确的流程:
1. 加载length-augmented end-effector轨迹
2. 生成随机机器人配置
3. 用IK为每个机器人生成关节轨迹
4. 保存成功的(机器人, 关节轨迹, 图表示)组合
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import json
import time

# Add modules to path
sys.path.append('/home/cx/AET_FOR_RL/vla/data_augment/morphology_modules')

from robot_configuration_generator import RobotConfigurationGenerator, RobotConfig
from robot_graph_module import RobotGraphModule
from ik_reachability_validator import IKReachabilityValidator


class CompleteMorphologySynthesisRunner:
    """
    完整的形态学合成流程

    输入: Length-augmented end-effector轨迹
    输出: (随机机器人配置, 关节轨迹, 机器人图) 的训练数据
    """

    def __init__(self,
                 robots_per_trajectory: int = 3,
                 max_robot_attempts: int = 10,
                 success_rate_threshold: float = 0.5):
        """
        Args:
            robots_per_trajectory: 每个轨迹尝试生成多少个机器人
            max_robot_attempts: 最大尝试次数
            success_rate_threshold: 最低IK成功率阈值
        """
        self.robots_per_trajectory = robots_per_trajectory
        self.max_robot_attempts = max_robot_attempts
        self.success_rate_threshold = success_rate_threshold

        # 生成适合DROID轨迹尺寸的机器人
        self.robot_generator = RobotConfigurationGenerator(
            dof_range=(5, 7),  # 减少范围，增加成功率
            link_length_range=(0.08, 0.15),  # 更保守的尺寸
            base_height_range=(0.05, 0.2)
        )

        self.graph_module = RobotGraphModule()

        # 更宽松的IK参数
        self.ik_validator = IKReachabilityValidator(
            position_tolerance=0.05,   # 5cm tolerance
            orientation_tolerance=np.deg2rad(20),  # 20° tolerance
            max_iterations=50
        )

        print(f"🚀 CompleteMorphologySynthesisRunner:")
        print(f"   Target robots per trajectory: {robots_per_trajectory}")
        print(f"   Max attempts: {max_robot_attempts}")
        print(f"   Success threshold: {success_rate_threshold:.1%}")

    def extract_ee_trajectory_from_timesteps(self, timesteps_data: pd.DataFrame, episode_id: str) -> np.ndarray:
        """从timesteps数据中提取end-effector轨迹"""
        episode_steps = timesteps_data[timesteps_data['episode_index'] == episode_id]
        episode_steps = episode_steps.sort_values('step_index')

        trajectory = []
        for _, row in episode_steps.iterrows():
            # action是关节轨迹，我们需要从DROID原始数据重新提取end-effector
            # 简化：假设action的前6维是end-effector pose + gripper
            action = row['action']
            if len(action) >= 7:
                ee_step = action[:7]  # [x,y,z,rx,ry,rz,gripper]
            else:
                # 如果action不是end-effector格式，跳过
                print(f"⚠️ Episode {episode_id}: action shape {len(action)} not compatible")
                return None
            trajectory.append(ee_step)

        return np.array(trajectory)

    def load_droid_ee_trajectory(self, original_episode_id: int) -> np.ndarray:
        """从原始DROID数据加载end-effector轨迹"""
        # 直接从原始转换数据加载
        droid_path = "/home/cx/AET_FOR_RL/vla/converted_data/droid_100_fixed/data/chunk-000/file-000.parquet"
        df = pd.read_parquet(droid_path)

        episode_data = df[df['episode_index'] == original_episode_id]
        trajectory = []

        for _, row in episode_data.iterrows():
            ee_step = np.concatenate([
                row['observation.cartesian_position'],  # [x,y,z,rx,ry,rz]
                [row['action'][6] if len(row['action']) > 6 else 0.0]  # gripper
            ])
            trajectory.append(ee_step)

        return np.array(trajectory)

    def synthesize_robots_for_trajectory(self, ee_trajectory: np.ndarray,
                                       trajectory_info: Dict) -> List[Dict]:
        """为单个end-effector轨迹合成机器人配置"""

        print(f"\n🔄 Processing trajectory: {trajectory_info['episode_index']}")
        print(f"   Length: {len(ee_trajectory)} steps")
        print(f"   Scale factor: {trajectory_info.get('length_scale_factor', 1.0):.3f}")

        successful_robots = []
        attempts = 0

        while len(successful_robots) < self.robots_per_trajectory and attempts < self.max_robot_attempts:
            attempts += 1

            # 生成随机机器人
            robot_name = f"synth_robot_{trajectory_info['episode_index']}_{attempts:02d}"
            robot = self.robot_generator.generate_random_robot(robot_name)

            print(f"   🤖 Attempt {attempts}: {robot.name} (DOF={robot.dof}, reach={robot.total_reach:.2f}m)")

            # 测试IK可达性 (只测试几个点，节省时间)
            sample_size = min(5, len(ee_trajectory))
            sample_indices = np.linspace(0, len(ee_trajectory)-1, sample_size, dtype=int)

            is_reachable, success_rate, _ = self.ik_validator.validate_trajectory_reachability(
                robot, ee_trajectory[sample_indices], sample_points=sample_size
            )

            if is_reachable and success_rate >= self.success_rate_threshold:
                # 生成完整关节轨迹
                joint_trajectory = self.generate_joint_trajectory_with_retries(robot, ee_trajectory)

                if joint_trajectory is not None:
                    # 创建机器人图
                    robot_graph = self.graph_module.robot_to_graph_dict(robot)

                    result = {
                        'robot_config': robot,
                        'robot_graph': robot_graph,
                        'joint_trajectory': joint_trajectory,
                        'ee_trajectory': ee_trajectory,
                        'success_rate': success_rate,
                        'trajectory_info': trajectory_info
                    }

                    successful_robots.append(result)
                    print(f"     ✅ SUCCESS! Generated {len(joint_trajectory)} timesteps")
                else:
                    print(f"     ❌ Failed to generate complete trajectory")
            else:
                print(f"     ❌ Low reachability: {success_rate:.1%}")

        print(f"   📊 Result: {len(successful_robots)}/{self.robots_per_trajectory} robots in {attempts} attempts")
        return successful_robots

    def generate_joint_trajectory_with_retries(self, robot: RobotConfig,
                                             ee_trajectory: np.ndarray,
                                             max_retries: int = 3) -> np.ndarray:
        """用重试机制生成关节轨迹"""

        for retry in range(max_retries):
            joint_trajectory = []
            previous_joint_angles = None
            failed = False

            for t, ee_point in enumerate(ee_trajectory):
                # 转换为4x4变换矩阵
                position = ee_point[:3]
                rpy = ee_point[3:6]
                gripper = ee_point[6]

                # RPY到旋转矩阵
                r, p, y = rpy
                R_x = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
                R_y = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
                R_z = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
                R = R_z @ R_y @ R_x

                target_pose = np.eye(4)
                target_pose[:3, :3] = R
                target_pose[:3, 3] = position

                # IK求解
                ik_result = self.ik_validator.solve_ik_single_point(
                    robot, target_pose, initial_guess=previous_joint_angles
                )

                if not ik_result.success:
                    failed = True
                    break

                # 计算轴向旋转角度来完成6DOF控制
                joint_angles_with_axial = self.complete_6dof_with_axial_rotation(
                    robot, ik_result.joint_trajectory, target_pose
                )

                # 组合完整关节角度+gripper
                joint_step = np.concatenate([joint_angles_with_axial, [gripper]])
                joint_trajectory.append(joint_step)
                previous_joint_angles = ik_result.joint_trajectory

            if not failed:
                return np.array(joint_trajectory)

            # 重试时使用随机初始猜测
            print(f"     🔄 Retry {retry+1}/{max_retries} with random initial guess")

        return None

    def complete_6dof_with_axial_rotation(self, robot: RobotConfig,
                                        joint_angles_4dof: np.ndarray,
                                        target_pose: np.ndarray) -> np.ndarray:
        """
        完成6DOF控制：4DOF IK + 轴向旋转计算

        Args:
            robot: 机器人配置
            joint_angles_4dof: 4DOF IK求解的关节角度
            target_pose: 目标6DOF姿态

        Returns:
            完整的关节角度 (包含轴向旋转)
        """
        # 计算当前4DOF IK后的末端姿态
        dh_params = np.array([joint.dh_params for joint in robot.joints])
        current_pose = self.ik_validator.dh_forward_kinematics(dh_params, joint_angles_4dof)

        # 提取目标和当前的X轴方向 (轴向旋转影响X-Y方向)
        target_x = target_pose[:3, 0]  # 目标X轴方向
        current_x = current_pose[:3, 0]  # 当前X轴方向
        current_z = current_pose[:3, 2]  # 当前Z轴方向 (应该已经对齐)

        # 计算需要的轴向旋转角度
        # 将目标X轴投影到当前Z轴的垂直平面上
        target_x_proj = target_x - np.dot(target_x, current_z) * current_z
        current_x_proj = current_x - np.dot(current_x, current_z) * current_z

        # 归一化投影向量
        target_x_proj = target_x_proj / (np.linalg.norm(target_x_proj) + 1e-8)
        current_x_proj = current_x_proj / (np.linalg.norm(current_x_proj) + 1e-8)

        # 计算旋转角度 (绕Z轴)
        cos_angle = np.dot(current_x_proj, target_x_proj)
        sin_angle = np.dot(np.cross(current_x_proj, target_x_proj), current_z)
        axial_rotation = np.arctan2(sin_angle, cos_angle)

        # 如果机器人有轴向旋转关节，设置它；否则保持原值
        joint_angles_complete = joint_angles_4dof.copy()
        if len(joint_angles_complete) > 0:
            # 最后一个关节是轴向旋转关节
            joint_angles_complete[-1] = axial_rotation

        return joint_angles_complete

    def run_synthesis(self, length_augmented_dir: str, output_dir: str) -> Dict:
        """运行完整的形态学合成"""

        print(f"🚀 Complete Morphology Synthesis Pipeline")
        print(f"   📁 Input: {length_augmented_dir}")
        print(f"   💾 Output: {output_dir}")
        print("=" * 80)

        # 加载length-augmented数据
        episodes_path = os.path.join(length_augmented_dir, "episodes.parquet")
        timesteps_path = os.path.join(length_augmented_dir, "timesteps.parquet")

        episodes_df = pd.read_parquet(episodes_path)
        timesteps_df = pd.read_parquet(timesteps_path)

        print(f"📊 Loaded {len(episodes_df)} episodes, {len(timesteps_df)} timesteps")

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "robot_graphs"), exist_ok=True)

        # 处理每个episode
        all_results = []
        start_time = time.time()

        for _, episode_row in episodes_df.iterrows():
            episode_id = episode_row['episode_index']
            original_id = episode_row['original_episode_id']

            # 加载原始DROID end-effector轨迹 (更可靠)
            ee_trajectory = self.load_droid_ee_trajectory(original_id)

            if ee_trajectory is None:
                print(f"⚠️ Skip {episode_id}: failed to load trajectory")
                continue

            # 应用length scaling
            scale_factor = episode_row['length_scale_factor']
            if scale_factor != 1.0:
                # 简单的重采样
                original_length = len(ee_trajectory)
                new_length = int(original_length * scale_factor)
                if new_length > 10:  # 最少10步
                    indices = np.linspace(0, original_length-1, new_length)
                    ee_trajectory = np.array([ee_trajectory[int(i)] for i in indices])

            trajectory_info = episode_row.to_dict()

            # 为这个轨迹合成机器人
            episode_results = self.synthesize_robots_for_trajectory(ee_trajectory, trajectory_info)
            all_results.extend(episode_results)

        # 保存结果
        total_time = time.time() - start_time
        self.save_synthesis_results(all_results, output_dir, total_time)

        return {
            'total_episodes_processed': len(episodes_df),
            'total_robots_generated': len(all_results),
            'synthesis_time': total_time
        }

    def save_synthesis_results(self, results: List[Dict], output_dir: str, total_time: float):
        """保存合成结果"""

        timesteps_data = []
        episode_metadata = []

        for result in results:
            robot = result['robot_config']
            joint_traj = result['joint_trajectory']
            robot_graph = result['robot_graph']
            traj_info = result['trajectory_info']

            # 保存机器人图
            graph_filename = f"{robot.name}.json"
            graph_path = os.path.join(output_dir, "robot_graphs", graph_filename)
            self.graph_module.save_robot_graph(robot, graph_path)

            # 创建唯一episode ID
            unique_episode_id = f"{traj_info['episode_index']}_{robot.name}"

            # 转换为timesteps格式
            for t in range(len(joint_traj)):
                timestep = {
                    'episode_index': unique_episode_id,
                    'step_index': t,
                    'timestamp': t,
                    'robot_name': robot.name,
                    'original_episode_id': traj_info['original_episode_id'],
                    'augmentation_type': 'morphology_synthesis',
                    'action': joint_traj[t],  # 关节角度 + gripper
                    'is_first': (t == 0),
                    'is_last': (t == len(joint_traj) - 1),
                }
                timesteps_data.append(timestep)

            # Episode metadata
            episode_meta = {
                'episode_index': unique_episode_id,
                'robot_name': robot.name,
                'original_episode_id': traj_info['original_episode_id'],
                'base_episode_index': traj_info['episode_index'],
                'dof': robot.dof,
                'length': len(joint_traj),
                'length_scale_factor': traj_info['length_scale_factor'],
                'total_reach': robot.total_reach,
                'success_rate': result['success_rate'],
                'graph_file': f"robot_graphs/{graph_filename}"
            }
            episode_metadata.append(episode_meta)

        # 保存文件
        timesteps_path = os.path.join(output_dir, "timesteps.parquet")
        episodes_path = os.path.join(output_dir, "episodes.parquet")
        stats_path = os.path.join(output_dir, "synthesis_stats.json")

        pd.DataFrame(timesteps_data).to_parquet(timesteps_path, index=False)
        pd.DataFrame(episode_metadata).to_parquet(episodes_path, index=False)

        # 统计信息
        unique_episodes = len(set(r['trajectory_info']['original_episode_id'] for r in results)) if results else 1
        stats = {
            'total_robots': len(results),
            'total_timesteps': len(timesteps_data),
            'total_episodes': len(episode_metadata),
            'synthesis_time': total_time,
            'robots_per_original_episode': len(results) / unique_episodes,
            'average_success_rate': np.mean([r['success_rate'] for r in results]) if results else 0.0
        }

        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"\n🎉 Complete Morphology Synthesis Completed!")
        print(f"   📊 Total robots: {stats['total_robots']}")
        print(f"   📊 Total timesteps: {stats['total_timesteps']}")
        print(f"   📊 Average success rate: {stats['average_success_rate']:.1%}")
        print(f"   📊 Synthesis time: {total_time:.1f}s")
        print(f"   💾 Saved to: {output_dir}")


def main():
    """主函数"""
    runner = CompleteMorphologySynthesisRunner(
        robots_per_trajectory=2,  # 每个轨迹2个机器人
        max_robot_attempts=8,     # 最多尝试8次
        success_rate_threshold=0.4  # 40%成功率
    )

    length_augmented_dir = "/home/cx/AET_FOR_RL/vla/synthesized_data/length_augmented_droid"
    output_dir = "/home/cx/AET_FOR_RL/vla/synthesized_data/complete_morphology_synthesis"

    runner.run_synthesis(length_augmented_dir, output_dir)


if __name__ == "__main__":
    main()