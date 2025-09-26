#!/usr/bin/env python3
"""
测试4DOF约束（位置+Z轴法向）的软体臂IK求解效果
小规模测试：对比3DOF vs 4DOF约束的成功率和精度
"""

import numpy as np
import pandas as pd
import json
import sys
from pathlib import Path
import time

# 添加模块路径
sys.path.append('/home/cx/AET_FOR_RL/vla/data_augment/morphology_modules')
from soft_arm_ik_solver import SoftArmConfig, SoftArmIKSolver, SoftArmKinematics

def test_4dof_constraint():
    """测试4DOF约束效果"""
    print("🧪 4DOF约束效果测试")
    print("="*50)

    # 加载DROID测试数据
    df = pd.read_parquet('/home/cx/AET_FOR_RL/vla/converted_data/droid_100_fixed/data/chunk-000/file-000.parquet')

    # 选择2个不同的episode进行测试
    test_episodes = [0, 10]

    for episode_id in test_episodes:
        print(f"\n📊 测试 Episode {episode_id}")

        # 提取episode数据
        episode_data = df[df['episode_index'] == episode_id]
        trajectory = []
        for _, row in episode_data.iterrows():
            action = np.array(row['action'])
            trajectory.append(action[:6])  # 取前6维：位置+角度

        trajectory = np.array(trajectory)
        print(f"   轨迹长度: {len(trajectory)} 步")

        # 分析轨迹workspace
        trajectory_center = np.mean(trajectory[:, :3], axis=0)
        base_offset = np.array([trajectory_center[0] - 0.6, trajectory_center[1], 0.0])

        # 测试不同段数的软体臂配置
        for n_segments in [3, 4]:
            print(f"\n   🤖 {n_segments}段软体臂测试:")

            # 使用合理的段长度配置（参考已知有效配置）
            if n_segments == 3:
                segment_lengths = [0.37, 0.40, 0.37]  # 参考episode_000/3_segments
            else:
                segment_lengths = [0.28, 0.31, 0.29, 0.33]  # 估计合理值

            config = SoftArmConfig(
                n_segments=n_segments,
                segment_lengths=segment_lengths
            )

            # 测试3DOF约束（当前方案）
            success_3dof, time_3dof, pos_errors_3dof, normal_errors_3dof, avg_pos_3dof, avg_normal_3dof = test_constraint_mode(
                trajectory, config, base_offset, use_4dof=False
            )

            # 测试4DOF约束（修复方案）
            success_4dof, time_4dof, pos_errors_4dof, normal_errors_4dof, avg_pos_4dof, avg_normal_4dof = test_constraint_mode(
                trajectory, config, base_offset, use_4dof=True
            )

            # 结果对比
            print(f"      3DOF约束: 成功率={success_3dof:.1%}, 耗时={time_3dof:.2f}s")
            print(f"                位置误差={avg_pos_3dof:.6f}m, 法向误差=N/A")
            print(f"      4DOF约束: 成功率={success_4dof:.1%}, 耗时={time_4dof:.2f}s")
            print(f"                位置误差={avg_pos_4dof:.6f}m, 法向误差={avg_normal_4dof:.6f}")

            success_drop = success_3dof - success_4dof
            time_increase = (time_4dof - time_3dof) / time_3dof * 100

            if success_4dof >= 0.6:  # 60%成功率阈值
                print(f"      ✅ 4DOF可行: 成功率下降{success_drop:.1%}, 时间增加{time_increase:.1f}%")
                if avg_normal_4dof < 0.5:  # 法向误差<0.5表示cos相似度>0.5 (60度内)
                    print(f"         法向控制良好: 平均法向误差{avg_normal_4dof:.3f}")
                else:
                    print(f"         ⚠️  法向控制较差: 平均法向误差{avg_normal_4dof:.3f}")
            else:
                print(f"      ❌ 4DOF困难: 成功率下降{success_drop:.1%}, 时间增加{time_increase:.1f}%")

def test_constraint_mode(trajectory, config, base_offset, use_4dof=False):
    """测试特定约束模式 - 修正版：独立验证解的质量"""

    ik_solver = SoftArmIKSolver(config)
    kinematics = SoftArmKinematics(config)

    successes = 0
    pos_errors = []
    normal_errors = []
    start_time = time.time()

    # 只测试前20步，节省时间
    test_steps = min(20, len(trajectory))

    previous_solution = None

    for i in range(test_steps):
        ee_pose = trajectory[i]

        if use_4dof:
            # 4DOF约束：位置 + Z轴法向
            target_world_pose = ee_pose[:6]  # 包含角度
            target_pose = target_world_pose.copy()
            target_pose[:3] = target_world_pose[:3] - base_offset  # 位置偏移
            target_rpy = target_pose[3:6]
            target_pos = target_pose[:3]
        else:
            # 3DOF约束：仅位置
            target_world_position = ee_pose[:3]
            target_pos = target_world_position - base_offset
            target_pose = target_pos
            target_rpy = None

        try:
            solution, ik_success, ik_error = ik_solver.solve_ik(target_pose, initial_guess=previous_solution)

            # 不相信IK求解器的success，自己验证！
            predicted_pos, predicted_rot = kinematics.forward_kinematics(solution)

            # 验证位置精度
            pos_error = np.linalg.norm(predicted_pos - target_pos)
            pos_success = pos_error < 0.05  # 5cm容忍度

            if use_4dof and target_rpy is not None:
                # 验证法向精度
                target_rot = ik_solver.rpy_to_rotation_matrix(target_rpy)
                target_normal = target_rot[:, 2]  # Z轴法向
                predicted_normal = predicted_rot[:, 2]

                cos_similarity = np.clip(np.dot(predicted_normal, target_normal), -1, 1)
                normal_error = 1 - cos_similarity  # 范围[0,2]，0=完全对齐
                normal_success = cos_similarity > 0.866  # cos(30°) = 0.866，允许30度偏差

                real_success = pos_success and normal_success
                normal_errors.append(normal_error)
            else:
                real_success = pos_success
                normal_errors.append(0.0)  # 3DOF模式没有法向误差

            pos_errors.append(pos_error)

            if real_success:
                successes += 1
                previous_solution = solution

        except Exception as e:
            pos_errors.append(1.0)  # 异常时记录大误差
            normal_errors.append(2.0)

    end_time = time.time()

    success_rate = successes / test_steps
    total_time = end_time - start_time
    avg_pos_error = np.mean(pos_errors)
    avg_normal_error = np.mean(normal_errors)

    return success_rate, total_time, pos_errors, normal_errors, avg_pos_error, avg_normal_error

if __name__ == "__main__":
    test_4dof_constraint()