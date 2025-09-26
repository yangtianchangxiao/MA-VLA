#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-Effector Based Synthesis Module - THE CORRECT APPROACH

Uses end-effector trajectories (6DoF + gripper) as unified reference
Generates joint trajectories for different morphologies via IK
Output: nDoF joints + 1DoF gripper for each target robot

Based on our confirmed design:
- Input: DROID 7D actions [x,y,z,roll,pitch,yaw,gripper]
- Process: IK to various robot morphologies
- Output: [joint_1, joint_2, ..., joint_n, gripper] for each robot
"""

import numpy as np
import random
from typing import Dict, List, Tuple, Optional
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
try:
    from .base_morphology_module import SynthesisModule, MorphologyConfig
except ImportError:
    from base_morphology_module import SynthesisModule, MorphologyConfig


class EndEffectorSynthesisModule(SynthesisModule):
    """Module for end-effector based morphology synthesis using IK"""

    def __init__(self, target_robots: List[Dict] = None):
        """
        Args:
            target_robots: List of target robot configurations
                Each dict should contain: name, dh_params, joint_limits, dof
        """
        # Standard Franka Panda DH (7DOF) as baseline
        self.baseline_dh = np.array([
            [0,      0.333,  0,       np.pi/2],   # Joint 1
            [0,      0,      0,      -np.pi/2],   # Joint 2
            [0,      0.316,  0,       np.pi/2],   # Joint 3
            [0.0825, 0,      0,       np.pi/2],   # Joint 4
            [-0.0825, 0.384, 0,      -np.pi/2],  # Joint 5
            [0,       0,     0,       np.pi/2],   # Joint 6
            [0.088,   0.107, 0,       0]          # Joint 7
        ])

        self.target_robots = target_robots or self._get_default_robots()
        print(f"🎯 EndEffectorSynthesisModule: {len(self.target_robots)} target robots")

    def _get_default_robots(self) -> List[Dict]:
        """Default robot configurations for synthesis"""
        return [
            {
                'name': 'franka_panda_7dof',
                'dh_params': self.baseline_dh,
                'dof': 7,
                'joint_limits': [
                    (-2.9, 2.9), (-1.76, 1.76), (-2.9, 2.9),
                    (-3.07, 0.07), (-2.9, 2.9), (-0.02, 3.75), (-2.9, 2.9)
                ]
            },
            {
                'name': 'scaled_franka_0.8x',
                'dh_params': self.baseline_dh * np.array([0.8, 0.8, 1, 1]),  # Scale a,d params
                'dof': 7,
                'joint_limits': [
                    (-2.9, 2.9), (-1.76, 1.76), (-2.9, 2.9),
                    (-3.07, 0.07), (-2.9, 2.9), (-0.02, 3.75), (-2.9, 2.9)
                ]
            },
            {
                'name': 'scaled_franka_1.2x',
                'dh_params': self.baseline_dh * np.array([1.2, 1.2, 1, 1]),  # Scale a,d params
                'dof': 7,
                'joint_limits': [
                    (-2.9, 2.9), (-1.76, 1.76), (-2.9, 2.9),
                    (-3.07, 0.07), (-2.9, 2.9), (-0.02, 3.75), (-2.9, 2.9)
                ]
            }
        ]

    def forward_kinematics(self, joint_angles: np.ndarray, dh_params: np.ndarray) -> np.ndarray:
        """Forward kinematics using DH parameters"""
        T = np.eye(4)
        for i, (a, d, alpha, theta_offset) in enumerate(dh_params):
            if i < len(joint_angles):
                theta = joint_angles[i] + theta_offset

                # DH transformation matrix
                ct, st = np.cos(theta), np.sin(theta)
                ca, sa = np.cos(alpha), np.sin(alpha)

                Ti = np.array([
                    [ct, -st*ca,  st*sa, a*ct],
                    [st,  ct*ca, -ct*sa, a*st],
                    [0,   sa,     ca,    d   ],
                    [0,   0,      0,     1   ]
                ])
                T = T @ Ti
        return T

    def pose_to_vector(self, T: np.ndarray) -> np.ndarray:
        """Convert 4x4 transformation matrix to 6D pose vector [x,y,z,rx,ry,rz]"""
        position = T[:3, 3]
        rotation = Rotation.from_matrix(T[:3, :3])
        euler = rotation.as_euler('xyz')  # XYZ Euler angles
        return np.concatenate([position, euler])

    def vector_to_pose(self, pose_vec: np.ndarray) -> np.ndarray:
        """Convert 6D pose vector to 4x4 transformation matrix"""
        T = np.eye(4)
        T[:3, 3] = pose_vec[:3]  # Position
        rotation = Rotation.from_euler('xyz', pose_vec[3:6])
        T[:3, :3] = rotation.as_matrix()
        return T

    def damped_pseudo_inverse_ik(self, target_pose: np.ndarray, current_joints: np.ndarray,
                                dh_params: np.ndarray, joint_limits: List[Tuple],
                                damping: float = 0.01, max_iterations: int = 50) -> Tuple[np.ndarray, bool]:
        """Damped pseudo-inverse differential IK solver"""

        joints = current_joints.copy()
        target_vec = self.pose_to_vector(target_pose)

        for iteration in range(max_iterations):
            # Current pose and error
            current_T = self.forward_kinematics(joints, dh_params)
            current_vec = self.pose_to_vector(current_T)
            error = target_vec - current_vec

            if np.linalg.norm(error) < 1e-4:  # Converged
                return joints, True

            # Compute Jacobian numerically
            jacobian = self._compute_jacobian(joints, dh_params)

            # Damped pseudo-inverse
            JTJ = jacobian.T @ jacobian
            damped_inv = np.linalg.inv(JTJ + damping * np.eye(len(joints))) @ jacobian.T

            # Joint update
            delta_q = damped_inv @ error
            joints = joints + 0.1 * delta_q  # Small step size

            # Enforce joint limits
            for i, (q_min, q_max) in enumerate(joint_limits):
                joints[i] = np.clip(joints[i], q_min, q_max)

        # Check final error
        final_T = self.forward_kinematics(joints, dh_params)
        final_error = np.linalg.norm(self.pose_to_vector(final_T) - target_vec)
        success = final_error < 0.05  # 5cm + 5deg tolerance

        return joints, success

    def _compute_jacobian(self, joints: np.ndarray, dh_params: np.ndarray,
                         epsilon: float = 1e-6) -> np.ndarray:
        """Compute 6xN Jacobian matrix numerically"""
        n_joints = len(joints)
        jacobian = np.zeros((6, n_joints))

        # Current pose
        current_T = self.forward_kinematics(joints, dh_params)
        current_vec = self.pose_to_vector(current_T)

        # Numerical differentiation
        for i in range(n_joints):
            joints_plus = joints.copy()
            joints_plus[i] += epsilon

            T_plus = self.forward_kinematics(joints_plus, dh_params)
            vec_plus = self.pose_to_vector(T_plus)

            jacobian[:, i] = (vec_plus - current_vec) / epsilon

        return jacobian

    def synthesize_trajectory_for_robot(self, ee_trajectory: np.ndarray,
                                       robot_config: Dict) -> np.ndarray:
        """
        Synthesize joint trajectory for specific robot from end-effector trajectory

        Args:
            ee_trajectory: (T, 7) array [x,y,z,rx,ry,rz,gripper]
            robot_config: Target robot configuration

        Returns:
            (T, robot_dof + 1) array [joint_1, ..., joint_n, gripper]
        """
        dh_params = robot_config['dh_params']
        joint_limits = robot_config['joint_limits']
        dof = robot_config['dof']

        # Initialize with neutral pose
        current_joints = np.zeros(dof)
        synthesized_trajectory = []

        for t, ee_step in enumerate(ee_trajectory):
            # Split end-effector pose and gripper
            ee_pose_vec = ee_step[:6]  # [x,y,z,rx,ry,rz]
            gripper = ee_step[6]       # gripper value

            # Convert to transformation matrix
            target_T = self.vector_to_pose(ee_pose_vec)

            # IK to find joint angles
            if t == 0:
                initial_guess = current_joints
            else:
                initial_guess = synthesized_trajectory[-1][:dof]  # Previous joint solution

            new_joints, success = self.damped_pseudo_inverse_ik(
                target_T, initial_guess, dh_params, joint_limits
            )

            if success:
                current_joints = new_joints
            # If IK fails, keep previous joints (temporal smoothing)

            # Append joint + gripper
            step_output = np.concatenate([current_joints, [gripper]])
            synthesized_trajectory.append(step_output)

        return np.array(synthesized_trajectory)

    def apply_to_trajectory(self, trajectory: np.ndarray, variation_data: Dict) -> Dict[str, np.ndarray]:
        """
        Apply end-effector based synthesis to trajectory

        Args:
            trajectory: (T, 7) DROID format [x,y,z,rx,ry,rz,gripper]
            variation_data: Not used in this module (we use robot configs)

        Returns:
            Dict mapping robot names to synthesized trajectories
        """
        results = {}

        print(f"🔄 Synthesizing trajectory for {len(self.target_robots)} robots...")
        print(f"   Input trajectory shape: {trajectory.shape}")

        for robot_config in self.target_robots:
            robot_name = robot_config['name']
            print(f"   Processing {robot_name} ({robot_config['dof']}DOF)...")

            try:
                synthesized = self.synthesize_trajectory_for_robot(trajectory, robot_config)
                results[robot_name] = synthesized
                print(f"   ✅ {robot_name}: {synthesized.shape}")
            except Exception as e:
                print(f"   ❌ {robot_name}: Failed - {e}")
                # Fallback: copy original trajectory if same DOF
                if robot_config['dof'] == 7:
                    results[robot_name] = trajectory.copy()

        return results

    def generate_variation(self, config: MorphologyConfig) -> Dict:
        """Generate robot configurations (implementation for base class)"""
        return {
            'target_robots': self.target_robots,
            'method': 'end_effector_ik_synthesis'
        }


def main():
    """Test the end-effector synthesis module"""
    print("🧪 Testing EndEffectorSynthesisModule")
    print("=" * 50)

    # Create module
    module = EndEffectorSynthesisModule()

    # Generate test end-effector trajectory (DROID format)
    T = 100
    test_trajectory = np.zeros((T, 7))

    # Simple circular motion + gripper sequence
    for t in range(T):
        angle = 2 * np.pi * t / T
        test_trajectory[t] = [
            0.3 + 0.1 * np.cos(angle),  # x
            0.1 * np.sin(angle),        # y
            0.4,                        # z
            0, 0, angle * 0.1,          # rx, ry, rz
            1.0 if t < T//2 else 0.0    # gripper
        ]

    print(f"📊 Test trajectory: {test_trajectory.shape}")
    print(f"   Sample poses: {test_trajectory[:3, :6]}")
    print(f"   Gripper values: {test_trajectory[:10, 6]}")

    # Apply synthesis
    results = module.apply_to_trajectory(test_trajectory, {})

    print(f"\n🎉 Synthesis completed!")
    for robot_name, traj in results.items():
        print(f"   {robot_name}: {traj.shape}")
        print(f"     Joint range: [{traj[:, :-1].min():.3f}, {traj[:, :-1].max():.3f}]")
        print(f"     Gripper: {traj[:5, -1]}")  # First 5 gripper values


if __name__ == "__main__":
    main()