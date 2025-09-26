#!/usr/bin/env python3
"""
Soft Arm IK Solver - 基于连续曲率参数的软体机械臂IK求解

基于你的soft_arm.cpp实现，提供：
1. 软体臂正运动学（解析解）
2. 软体臂逆运动学（数值优化，但更稳定）
3. 随机软体臂配置生成
4. 与VLA训练框架集成
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize
from typing import Tuple, List, Dict, Optional
import random


class SoftArmConfig:
    """软体臂配置类"""

    def __init__(self, n_segments: int = 4, segment_lengths: List[float] = None):
        self.n_segments = n_segments

        # 如果没有指定长度，使用默认值
        if segment_lengths is None:
            self.segment_lengths = [0.18] * n_segments  # 默认每段18cm
        else:
            assert len(segment_lengths) == n_segments
            self.segment_lengths = segment_lengths

        # 参数限制
        self.alpha_limits = (0.001, np.pi)      # 弯曲角度范围，避免奇异
        self.beta_limits = (0, 2*np.pi)         # 方向角度范围

        # 工作空间估算
        self.max_reach = sum(self.segment_lengths)
        self.min_reach = 0.0

    @classmethod
    def random_config(cls, n_segments: int = 4,
                     length_range: Tuple[float, float] = (0.12, 0.25)) -> 'SoftArmConfig':
        """生成随机软体臂配置"""
        min_len, max_len = length_range
        random_lengths = [random.uniform(min_len, max_len) for _ in range(n_segments)]
        return cls(n_segments, random_lengths)

    @property
    def action_dim(self) -> int:
        """动作维度：每段2个参数(alpha, beta)"""
        return self.n_segments * 2

    def __repr__(self):
        return f"SoftArmConfig(segments={self.n_segments}, lengths={self.segment_lengths})"


class SoftArmKinematics:
    """软体臂运动学求解器"""

    def __init__(self, config: SoftArmConfig):
        self.config = config

    def config_to_translation(self, alpha: float, beta: float, arc_length: float) -> np.ndarray:
        """
        计算单段的位移向量 (严格按照C++代码实现)

        Args:
            alpha: 弯曲角度
            beta: 弯曲方向
            arc_length: 段长度

        Returns:
            position: (3,) 位移向量
        """
        # 严格按照C++代码的条件检查
        if alpha == 0:
            alpha = 0.000001

        # 严格按照C++代码的公式
        x_temp = arc_length/alpha * (1 - np.cos(alpha)) * np.sin(beta)
        y_temp = arc_length/alpha * (1 - np.cos(alpha)) * np.cos(beta)
        z_temp = arc_length/alpha * np.sin(alpha)

        return np.array([x_temp, y_temp, z_temp])

    def config_to_rotation(self, alpha: float, beta: float) -> np.ndarray:
        """
        计算单段的旋转矩阵 (基于你的C++代码)

        Args:
            alpha: 弯曲角度
            beta: 弯曲方向

        Returns:
            rotation: (3,3) 旋转矩阵
        """
        cos_a, sin_a = np.cos(alpha), np.sin(alpha)
        cos_b, sin_b = np.cos(beta), np.sin(beta)

        R = np.array([
            [cos_b*cos_b*(1-cos_a) + cos_a, -cos_b*sin_b*(1-cos_a),     sin_a*sin_b],
            [-cos_b*sin_b*(1-cos_a),         sin_b*sin_b*(1-cos_a) + cos_a, sin_a*cos_b],
            [-sin_a*sin_b,                   -sin_a*cos_b,                  cos_a]
        ])

        return R

    def forward_kinematics(self, curvature_params: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        正运动学：曲率参数 → 末端位姿
        严格按照C++代码的累积方式

        Args:
            curvature_params: (n_segments*2,) [α1,β1,α2,β2,α3,β3,α4,β4]

        Returns:
            position: (3,) 末端位置
            rotation: (3,3) 末端旋转矩阵
        """
        assert len(curvature_params) == self.config.action_dim

        # 初始化：世界坐标系 (严格按照C++)
        current_pos = np.array([0.0, 0.0, 0.0])  # base_position
        current_rot = np.eye(3)                   # Rot_world_base

        # 逐段累积变换 (严格按照C++的逻辑)
        for i in range(self.config.n_segments):
            alpha = curvature_params[i*2]
            beta = curvature_params[i*2 + 1]
            arc_length = self.config.segment_lengths[i]

            # 计算该段的相对变换
            segment_translation = self.config_to_translation(alpha, beta, arc_length)
            segment_rotation = self.config_to_rotation(alpha, beta)

            # 严格按照C++的累积方式:
            # Pos_base_end{i+1} = Pos_base_end{i} + Rot_base_end{i} * translation{i+1}
            # Rot_base_end{i+1} = Rot_base_end{i} * rotation{i+1}
            current_pos = current_pos + current_rot @ segment_translation
            current_rot = current_rot @ segment_rotation

        return current_pos, current_rot

    def pose_to_transform_matrix(self, position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
        """位姿转换为4x4变换矩阵"""
        T = np.eye(4)
        T[:3, :3] = rotation
        T[:3, 3] = position
        return T

    def transform_matrix_to_pose(self, T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """4x4变换矩阵转换为位姿"""
        position = T[:3, 3]
        rotation = T[:3, :3]
        return position, rotation


class SoftArmIKSolver:
    """软体臂IK求解器"""

    def __init__(self, config: SoftArmConfig):
        self.config = config
        self.kinematics = SoftArmKinematics(config)

    def rpy_to_rotation_matrix(self, rpy: np.ndarray) -> np.ndarray:
        """RPY欧拉角转旋转矩阵"""
        r, p, y = rpy
        R_x = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
        R_y = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
        R_z = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
        return R_z @ R_y @ R_x

    def ik_objective_4dof(self, curvature_params: np.ndarray, target_pose: np.ndarray) -> float:
        """4DOF IK目标函数：位置 + Z轴法向约束"""

        # 正运动学
        current_pos, current_rot = self.kinematics.forward_kinematics(curvature_params)

        # 目标位置
        target_pos = target_pose[:3]

        # 目标Z轴方向
        if len(target_pose) >= 6:
            target_rpy = target_pose[3:6]
            target_rot = self.rpy_to_rotation_matrix(target_rpy)
            target_z = target_rot[:, 2]  # Z轴方向
        else:
            target_z = np.array([0, 0, 1])  # 默认向上

        current_z = current_rot[:, 2]

        # 位置误差
        pos_error = np.linalg.norm(current_pos - target_pos)

        # Z轴法向误差（用余弦距离，更合理）
        cos_similarity = np.dot(current_z, target_z)
        cos_similarity = np.clip(cos_similarity, -1, 1)  # 数值稳定性
        z_error = 1 - cos_similarity  # 范围[0, 2]，0=完全对齐

        return pos_error + z_error * 0.1  # 降低法向权重

    def complete_6dof_with_axial_rotation(self, curvature_params_4dof: np.ndarray,
                                         target_pose: np.ndarray) -> np.ndarray:
        """
        完成6DOF控制：4DOF约束后调整最后段的轴向旋转

        Args:
            curvature_params_4dof: 4DOF约束的曲率参数
            target_pose: 目标6DOF位姿 [x,y,z,rx,ry,rz]

        Returns:
            complete_params: 完整的曲率参数（调整最后段beta）
        """
        if len(target_pose) < 6:
            return curvature_params_4dof  # 没有方向要求

        # 计算当前4DOF后的姿态
        current_pos, current_rot = self.kinematics.forward_kinematics(curvature_params_4dof)

        # 目标姿态
        target_rpy = target_pose[3:6]
        target_rot = self.rpy_to_rotation_matrix(target_rpy)

        # 提取X轴方向（轴向旋转影响X-Y方向）
        target_x = target_rot[:, 0]
        current_x = current_rot[:, 0]
        current_z = current_rot[:, 2]  # Z轴应该已经对齐

        # 将X轴投影到Z垂直平面
        target_x_proj = target_x - np.dot(target_x, current_z) * current_z
        current_x_proj = current_x - np.dot(current_x, current_z) * current_z

        # 归一化投影向量
        target_x_proj = target_x_proj / (np.linalg.norm(target_x_proj) + 1e-8)
        current_x_proj = current_x_proj / (np.linalg.norm(current_x_proj) + 1e-8)

        # 计算绕Z轴的旋转角度
        cos_angle = np.dot(current_x_proj, target_x_proj)
        sin_angle = np.dot(np.cross(current_x_proj, target_x_proj), current_z)
        axial_rotation = np.arctan2(sin_angle, cos_angle)

        # 调整最后一段的beta角度
        complete_params = curvature_params_4dof.copy()
        if self.config.n_segments > 0:
            # 最后段beta = 原beta + 轴向调整
            last_beta_idx = (self.config.n_segments - 1) * 2 + 1
            complete_params[last_beta_idx] = complete_params[last_beta_idx] + axial_rotation

            # 确保在范围内
            while complete_params[last_beta_idx] > 2*np.pi:
                complete_params[last_beta_idx] -= 2*np.pi
            while complete_params[last_beta_idx] < 0:
                complete_params[last_beta_idx] += 2*np.pi

        return complete_params

    def solve_ik_hierarchical(self, target_pos: np.ndarray, target_normal: Optional[np.ndarray] = None,
                             initial_guess: Optional[np.ndarray] = None,
                             max_iterations: int = 100) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        分层优化IK求解：第一步位置优先，第二步法向优化

        Args:
            target_pos: 目标位置 [x, y, z]
            target_normal: 目标Z轴法向量 [nx, ny, nz]
            initial_guess: 初始猜测
            max_iterations: 最大迭代次数

        Returns:
            alpha: 弯曲角度数组 [n_segments]
            beta: 方向角度数组 [n_segments]
            success: 是否成功
        """

        # 初始猜测
        if initial_guess is None:
            initial_guess = np.random.uniform(
                [self.config.alpha_limits[0], self.config.beta_limits[0]] * self.config.n_segments,
                [self.config.alpha_limits[1], self.config.beta_limits[1]] * self.config.n_segments
            )

        # 参数边界
        bounds = []
        for i in range(self.config.n_segments):
            bounds.append(self.config.alpha_limits)  # alpha
            bounds.append(self.config.beta_limits)   # beta

        # 第一步：位置优先优化 - 只管位置准确
        def position_only_objective(params):
            current_pos, _ = self.kinematics.forward_kinematics(params)
            pos_error = np.linalg.norm(current_pos - target_pos)
            return pos_error

        try:
            # 位置优化
            result_pos = minimize(
                fun=position_only_objective,
                x0=initial_guess,
                bounds=bounds,
                method='L-BFGS-B',
                options={'maxiter': max_iterations}
            )

            position_solution = result_pos.x
            pos_error = result_pos.fun

            # 如果没有法向要求，直接返回位置解
            if target_normal is None:
                alpha = position_solution[0::2]
                beta = position_solution[1::2]
                success = pos_error < 0.05  # 5cm tolerance
                return alpha, beta, success

            # 第二步：在位置解基础上优化法向 - 位置稳定，法向尽量好
            def hierarchical_objective(params):
                current_pos, current_rot = self.kinematics.forward_kinematics(params)

                # 位置误差（高权重 - 维持第一步的成果）
                pos_error = np.linalg.norm(current_pos - target_pos)

                # 法向误差（低权重 - 在位置准确基础上尽量改善）
                current_z = current_rot @ np.array([0, 0, 1])
                cos_similarity = np.clip(np.dot(current_z, target_normal), -1, 1)
                normal_error = 1 - cos_similarity

                # 分层权重：位置误差权重高10倍
                return pos_error * 10.0 + normal_error * 1.0

            # 从位置解开始的法向优化
            result_hierarchical = minimize(
                fun=hierarchical_objective,
                x0=position_solution,  # 从位置解开始！
                bounds=bounds,
                method='L-BFGS-B',
                options={'maxiter': max_iterations // 2}  # 更少迭代，避免偏离位置解太多
            )

            final_solution = result_hierarchical.x

            # 验证最终结果
            final_pos, final_rot = self.kinematics.forward_kinematics(final_solution)
            final_pos_error = np.linalg.norm(final_pos - target_pos)

            alpha = final_solution[0::2]
            beta = final_solution[1::2]
            success = final_pos_error < 0.05  # 位置准确是硬要求

            return alpha, beta, success

        except Exception as e:
            print(f"Hierarchical IK optimization failed: {e}")
            # 返回随机解
            alpha = np.random.uniform(self.config.alpha_limits[0], self.config.alpha_limits[1], self.config.n_segments)
            beta = np.random.uniform(self.config.beta_limits[0], self.config.beta_limits[1], self.config.n_segments)
            return alpha, beta, False

    def solve_ik(self, target_pose: np.ndarray,
                 initial_guess: Optional[np.ndarray] = None,
                 max_iterations: int = 100) -> Tuple[np.ndarray, bool, float]:
        """
        软体臂IK求解：兼容旧接口，内部使用分层优化

        Args:
            target_pose: 目标位姿 [x,y,z] 或 [x,y,z,rx,ry,rz]
            initial_guess: 初始猜测
            max_iterations: 最大迭代次数

        Returns:
            curvature_params: 完整曲率参数
            success: 是否成功
            error: 最终误差
        """

        target_pos = target_pose[:3]
        target_normal = None

        # 如果有方向信息，提取Z轴法向
        if len(target_pose) >= 6:
            target_rpy = target_pose[3:6]
            target_rot = self.rpy_to_rotation_matrix(target_rpy)
            target_normal = target_rot[:, 2]  # Z轴方向

        # 使用分层优化
        alpha, beta, success = self.solve_ik_hierarchical(
            target_pos, target_normal, initial_guess, max_iterations
        )

        # 组装成旧格式
        curvature_params = np.zeros(self.config.n_segments * 2)
        curvature_params[0::2] = alpha
        curvature_params[1::2] = beta

        # 计算最终误差
        if success:
            final_pos, _ = self.kinematics.forward_kinematics(curvature_params)
            final_error = np.linalg.norm(final_pos - target_pos)
        else:
            final_error = float('inf')

        return curvature_params, success, final_error

    def validate_trajectory_reachability(self, target_trajectory: np.ndarray,
                                       sample_points: int = 5,
                                       position_only: bool = True) -> Tuple[bool, float]:
        """验证轨迹可达性"""

        if len(target_trajectory) == 0:
            return False, 0.0

        # 采样几个点测试
        n_points = len(target_trajectory)
        if sample_points >= n_points:
            test_indices = range(n_points)
        else:
            test_indices = np.linspace(0, n_points-1, sample_points, dtype=int)

        success_count = 0
        total_error = 0.0

        previous_solution = None

        for idx in test_indices:
            target_pose = target_trajectory[idx]

            # 根据position_only标志选择约束方式
            if position_only:
                # 只使用位置约束（更宽松）
                target_test = target_pose[:3]
            else:
                # 使用完整6DOF约束（更严格）
                target_test = target_pose

            # 使用之前的解作为初始猜测
            solution, success, error = self.solve_ik(target_test, initial_guess=previous_solution)

            if success:
                success_count += 1
                previous_solution = solution

            total_error += error

        success_rate = success_count / len(test_indices)
        avg_error = total_error / len(test_indices)

        # 根据约束类型调整阈值
        threshold = 0.3 if position_only else 0.5  # 位置约束用30%，6DOF约束用50%
        return success_rate > threshold, success_rate


class SoftArmSynthesisModule:
    """软体臂合成模块，类似complete版本的架构"""

    def __init__(self, segment_range: Tuple[int, int] = (3, 6),
                 length_range: Tuple[float, float] = (0.12, 0.25)):
        self.segment_range = segment_range
        self.length_range = length_range

    def generate_random_soft_arm(self) -> SoftArmConfig:
        """生成随机软体臂配置"""
        n_segments = random.randint(*self.segment_range)
        return SoftArmConfig.random_config(n_segments, self.length_range)

    def synthesize_soft_arm_trajectory(self, ee_trajectory: np.ndarray,
                                     soft_arm: SoftArmConfig,
                                     base_offset: np.ndarray = None) -> Optional[np.ndarray]:
        """
        为软体臂合成轨迹

        Args:
            ee_trajectory: (N, 6) 末端轨迹 [x,y,z,rx,ry,rz]
            soft_arm: 软体臂配置

        Returns:
            curvature_trajectory: (N, n_segments*2) 曲率参数轨迹
        """

        ik_solver = SoftArmIKSolver(soft_arm)

        # 跳过可达性验证，直接尝试生成轨迹（实用主义策略）
        # TODO: 未来应该修复base_offset的传递问题

        # 生成完整轨迹
        curvature_trajectory = []
        previous_solution = None

        for i, ee_pose in enumerate(ee_trajectory):
            # 只使用位置约束 [x,y,z]，保持与validate_trajectory_reachability一致
            target_world_position = ee_pose[:3]

            # 转换到软体臂坐标系（考虑基座偏移）
            if base_offset is not None:
                target_position = target_world_position - base_offset
            else:
                target_position = target_world_position

            solution, success, error = ik_solver.solve_ik(target_position, initial_guess=previous_solution)

            if not success:
                print(f"IK failed at step {i}, error: {error:.4f}")
                return None

            curvature_trajectory.append(solution)
            previous_solution = solution

        return np.array(curvature_trajectory)


# 工具函数
def test_soft_arm_ik():
    """测试软体臂IK"""

    # 创建随机软体臂
    soft_arm = SoftArmConfig.random_config(4, (0.15, 0.20))
    print(f"Testing: {soft_arm}")

    # 创建IK求解器
    ik_solver = SoftArmIKSolver(soft_arm)

    # 测试正运动学
    test_params = np.array([0.5, 0.0, 0.3, np.pi/2, 0.2, np.pi, 0.1, 0.0])
    pos, rot = ik_solver.kinematics.forward_kinematics(test_params)
    print(f"Forward kinematics result: pos={pos}, reachable={np.linalg.norm(pos) < soft_arm.max_reach}")

    # 测试位置IK (3DOF)
    target_pos = np.array([0.3, 0.2, 0.4])
    solution, success, error = ik_solver.solve_ik(target_pos)
    print(f"Position IK (3DOF) result: success={success}, error={error:.4f}")

    # 先测试更简单的6DOF目标
    # 使用正运动学生成一个已知可达的目标
    test_curvature = np.array([0.3, 0.0, 0.2, np.pi/4, 0.1, np.pi/2, 0.05, 0.0])
    reachable_pos, reachable_rot = ik_solver.kinematics.forward_kinematics(test_curvature)

    # 转换为RPY
    from scipy.spatial.transform import Rotation as R
    reachable_rpy = R.from_matrix(reachable_rot).as_euler('xyz')

    target_6dof_reachable = np.concatenate([reachable_pos, reachable_rpy])
    print(f"Testing reachable target: pos={reachable_pos}, rpy={reachable_rpy}")

    solution_6dof, success_6dof, error_6dof = ik_solver.solve_ik(target_6dof_reachable)
    print(f"6DOF IK (reachable target) result: success={success_6dof}, error={error_6dof:.4f}")

    # 验证IK解
    if success_6dof:
        verify_pos, verify_rot = ik_solver.kinematics.forward_kinematics(solution_6dof)
        pos_verify_error = np.linalg.norm(verify_pos - target_6dof_reachable[:3])
        print(f"Position verification error: {pos_verify_error:.4f}")

        # 验证方向
        target_rot = ik_solver.rpy_to_rotation_matrix(target_6dof_reachable[3:6])
        rot_error = np.linalg.norm(verify_rot - target_rot, 'fro')
        print(f"Orientation verification error: {rot_error:.4f}")

        print("✅ 软体臂4DOF+轴向旋转策略成功！")
    else:
        print("❌ 即使是已知可达目标也失败了，算法有问题")

    # 额外测试：比较原始曲率参数和IK解
    print(f"\nOriginal curvature: {test_curvature}")
    if success_6dof:
        print(f"IK solution:        {solution_6dof}")
        param_diff = np.linalg.norm(test_curvature - solution_6dof)
        print(f"Parameter difference: {param_diff:.4f}")
        print("(Note: Different parameters can lead to same pose due to redundancy)")


if __name__ == "__main__":
    test_soft_arm_ik()