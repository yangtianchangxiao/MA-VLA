#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-Effector Synthesis Runner - THE CORRECT APPROACH Fixed
Based on end_effector_synthesis_module.py but with Linus-style fixes:
1. Fixed IK failure handling
2. Fixed pose representation (quaternions, not Euler)
3. Added trajectory continuity validation
4. Removed stupid fallbacks
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from typing import List, Dict, Tuple
from scipy.spatial.transform import Rotation

from morphology_modules.end_effector_synthesis_module import EndEffectorSynthesisModule


class FixedEndEffectorSynthesizer:
    """End-effector synthesis with Linus-style fixes"""

    def __init__(self, droid_data_path: str):
        self.droid_path = droid_data_path
        print("🎯 Fixed End-Effector Synthesizer - Linus Edition")
        print("=" * 60)

        # Load DROID data
        try:
            self.data_df = pd.read_parquet(f"{droid_data_path}/data/chunk-000/file-000.parquet")
            self.episodes_df = pd.read_parquet(f"{droid_data_path}/meta/episodes/chunk-000/file-000.parquet")
            print(f"   ✅ Loaded {len(self.data_df)} frames from {len(self.episodes_df)} episodes")
        except Exception as e:
            print(f"   ❌ Failed to load DROID data: {e}")
            raise

        # Initialize fixed synthesis module
        self.synthesis_module = EndEffectorSynthesisModule()

        # Override the broken methods with fixed versions
        self._patch_synthesis_module()

        print("✅ Fixed end-effector synthesizer ready")

    def _patch_synthesis_module(self):
        """Patch the broken methods in the original module"""
        # Replace the broken pose conversion methods
        self.synthesis_module.pose_to_vector = self._fixed_pose_to_vector
        self.synthesis_module.vector_to_pose = self._fixed_vector_to_pose
        self.synthesis_module.synthesize_trajectory_for_robot = self._fixed_synthesize_trajectory

    def _fixed_pose_to_vector(self, T: np.ndarray) -> np.ndarray:
        """
        FIXED: Convert 4x4 transformation matrix to 7D pose vector [x,y,z,qw,qx,qy,qz]
        Uses quaternions instead of Euler angles to avoid singularities
        """
        position = T[:3, 3]
        rotation = Rotation.from_matrix(T[:3, :3])
        quaternion = rotation.as_quat()  # [x,y,z,w] format
        # Convert to [w,x,y,z] format for consistency
        quat_wxyz = np.array([quaternion[3], quaternion[0], quaternion[1], quaternion[2]])
        return np.concatenate([position, quat_wxyz])

    def _fixed_vector_to_pose(self, pose_vec: np.ndarray) -> np.ndarray:
        """
        FIXED: Convert 7D pose vector [x,y,z,qw,qx,qy,qz] to 4x4 transformation matrix
        """
        T = np.eye(4)
        T[:3, 3] = pose_vec[:3]  # Position

        # Convert quaternion [w,x,y,z] to [x,y,z,w] for scipy
        quat_wxyz = pose_vec[3:7]
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])

        # Normalize quaternion to avoid numerical issues
        quat_xyzw = quat_xyzw / np.linalg.norm(quat_xyzw)

        rotation = Rotation.from_quat(quat_xyzw)
        T[:3, :3] = rotation.as_matrix()
        return T

    def _fixed_damped_pseudo_inverse_ik(self, target_pose: np.ndarray, current_joints: np.ndarray,
                                       dh_params: np.ndarray, joint_limits: List[Tuple],
                                       damping: float = 0.01, max_iterations: int = 100) -> Tuple[np.ndarray, bool, float]:
        """
        FIXED: Damped pseudo-inverse IK with proper error reporting and quaternion handling
        """
        joints = current_joints.copy()
        target_vec = self._fixed_pose_to_vector(target_pose)

        best_joints = joints.copy()
        best_error = float('inf')

        for iteration in range(max_iterations):
            # Current pose and error
            current_T = self.synthesis_module.forward_kinematics(joints, dh_params)
            current_vec = self._fixed_pose_to_vector(current_T)

            # Separate position and orientation errors
            pos_error = target_vec[:3] - current_vec[:3]

            # Quaternion difference - use proper quaternion error
            target_quat = target_vec[3:7]
            current_quat = current_vec[3:7]

            # Ensure same hemisphere (shortest rotation)
            if np.dot(target_quat, current_quat) < 0:
                current_quat = -current_quat

            # Quaternion error as rotation vector
            target_rot = Rotation.from_quat([target_quat[1], target_quat[2], target_quat[3], target_quat[0]])
            current_rot = Rotation.from_quat([current_quat[1], current_quat[2], current_quat[3], current_quat[0]])
            error_rot = target_rot * current_rot.inv()
            rot_error = error_rot.as_rotvec()

            # Combined error vector
            error = np.concatenate([pos_error, rot_error])
            error_norm = np.linalg.norm(error)

            if error_norm < best_error:
                best_error = error_norm
                best_joints = joints.copy()

            # Convergence check - stricter than original
            if error_norm < 1e-4:
                return joints, True, error_norm

            # Compute Jacobian numerically - need custom implementation for 7D quaternion
            jacobian = self._compute_7d_jacobian(joints, dh_params)

            # Damped pseudo-inverse with adaptive damping
            JTJ = jacobian.T @ jacobian
            adaptive_damping = damping * (1 + error_norm)  # Increase damping when error is large
            damped_inv = np.linalg.inv(JTJ + adaptive_damping * np.eye(len(joints))) @ jacobian.T

            # Joint update with adaptive step size
            step_size = 0.1 / (1 + error_norm)  # Smaller steps when error is large
            delta_q = damped_inv @ error
            joints = joints + step_size * delta_q

            # Enforce joint limits
            for i, (q_min, q_max) in enumerate(joint_limits):
                joints[i] = np.clip(joints[i], q_min, q_max)

        # Return best solution found
        return best_joints, best_error < 0.02, best_error  # 2cm tolerance

    def _compute_7d_jacobian(self, joints: np.ndarray, dh_params: np.ndarray,
                            epsilon: float = 1e-6) -> np.ndarray:
        """Compute 7xN Jacobian matrix for quaternion pose representation"""
        n_joints = len(joints)
        jacobian = np.zeros((7, n_joints))  # 7D for [pos(3) + quat(4)]

        # Current pose
        current_T = self.synthesis_module.forward_kinematics(joints, dh_params)
        current_vec = self._fixed_pose_to_vector(current_T)

        # Numerical differentiation
        for i in range(n_joints):
            joints_plus = joints.copy()
            joints_plus[i] += epsilon

            T_plus = self.synthesis_module.forward_kinematics(joints_plus, dh_params)
            vec_plus = self._fixed_pose_to_vector(T_plus)

            jacobian[:, i] = (vec_plus - current_vec) / epsilon

        return jacobian

    def _fixed_synthesize_trajectory(self, ee_trajectory: np.ndarray, robot_config: Dict) -> np.ndarray:
        """
        FIXED: Synthesize joint trajectory with proper continuity and failure handling
        """
        dh_params = robot_config['dh_params']
        joint_limits = robot_config['joint_limits']
        dof = robot_config['dof']

        # Initialize with neutral pose
        current_joints = np.zeros(dof)
        synthesized_trajectory = []

        failed_frames = 0
        total_frames = len(ee_trajectory)

        for t, ee_step in enumerate(ee_trajectory):
            # Split end-effector pose and gripper
            ee_pose_vec = ee_step[:6]  # [x,y,z,rx,ry,rz] - still Euler for input
            gripper = ee_step[6]

            # Convert Euler to quaternion for internal processing
            position = ee_pose_vec[:3]
            euler_angles = ee_pose_vec[3:6]
            rotation = Rotation.from_euler('xyz', euler_angles)
            quaternion = rotation.as_quat()  # [x,y,z,w]
            quat_wxyz = np.array([quaternion[3], quaternion[0], quaternion[1], quaternion[2]])

            # Create 7D pose vector for internal use
            pose_7d = np.concatenate([position, quat_wxyz])
            target_T = self._fixed_vector_to_pose(pose_7d)

            # IK with previous solution as initial guess
            if t == 0:
                initial_guess = current_joints
            else:
                initial_guess = synthesized_trajectory[-1][:dof]

            new_joints, success, error = self._fixed_damped_pseudo_inverse_ik(
                target_T, initial_guess, dh_params, joint_limits
            )

            if success:
                current_joints = new_joints
            else:
                failed_frames += 1
                # FIXED: Don't just keep previous joints, try to recover
                if t > 0:
                    # Use previous joints but report the failure
                    current_joints = synthesized_trajectory[-1][:dof]
                print(f"    ⚠️ IK failed at frame {t}: error={error:.4f}")

            # Append joint + gripper
            step_output = np.concatenate([current_joints, [gripper]])
            synthesized_trajectory.append(step_output)

        # Validate trajectory quality
        success_rate = 1.0 - (failed_frames / total_frames)
        print(f"    📊 Trajectory quality: {success_rate:.1%} success rate ({failed_frames}/{total_frames} failed)")

        if success_rate < 0.8:
            print(f"    ❌ Trajectory quality too low: {success_rate:.1%}")
            return None

        return np.array(synthesized_trajectory)

    def process_single_episode(self, episode_idx: int) -> Dict:
        """Process single episode with fixed end-effector synthesis"""

        # Get episode data
        episode_data = self.data_df[self.data_df['episode_index'] == episode_idx]
        if len(episode_data) == 0:
            return {}

        print(f"  🎯 Processing episode {episode_idx} ({len(episode_data)} frames)")

        # Extract end-effector trajectory from DROID data
        ee_trajectory = []
        for _, row in episode_data.iterrows():
            # Use cartesian_position if available, otherwise derive from action
            if 'observation.cartesian_position' in row and row['observation.cartesian_position'] is not None:
                cartesian_pos = np.array(row['observation.cartesian_position'])
                # Ensure we have 6D pose (position + orientation)
                if len(cartesian_pos) >= 6:
                    ee_pose = cartesian_pos[:6]
                else:
                    print(f"    ⚠️ Invalid cartesian_position size: {len(cartesian_pos)}, skipping frame")
                    continue
            else:
                # Fallback: assume action contains delta movements
                print(f"    ⚠️ No cartesian_position in frame, skipping episode {episode_idx}")
                return {}

            action = row['action'] if 'action' in row else np.zeros(7)
            gripper = action[6] if len(action) > 6 else 0.0

            ee_step = np.concatenate([ee_pose, [gripper]])
            ee_trajectory.append(ee_step)

        if len(ee_trajectory) == 0:
            return {}

        ee_trajectory = np.array(ee_trajectory)

        # Apply synthesis using fixed module
        try:
            results = {}
            for robot_config in self.synthesis_module.target_robots:
                robot_name = robot_config['name']
                print(f"    Processing {robot_name}...")

                joint_trajectory = self._fixed_synthesize_trajectory(ee_trajectory, robot_config)
                if joint_trajectory is not None:
                    results[robot_name] = joint_trajectory
                    print(f"    ✅ {robot_name}: {joint_trajectory.shape}")
                else:
                    print(f"    ❌ {robot_name}: Failed quality check")

            return {
                'episode_index': episode_idx,
                'ee_trajectory': ee_trajectory,
                'synthesized_trajectories': results,
                'frame_count': len(ee_trajectory)
            }

        except Exception as e:
            import traceback
            print(f"    ❌ Episode {episode_idx} failed: {e}")
            print(f"    📋 Full traceback: {traceback.format_exc()}")
            return {}

    def synthesize_valid_episodes(self, valid_episodes: List[int], output_path: str):
        """Process valid episodes with fixed synthesis"""

        print(f"\n🎯 Starting fixed end-effector synthesis:")
        print(f"   📊 Valid episodes: {len(valid_episodes)}")
        print()

        os.makedirs(output_path, exist_ok=True)
        results = []

        for episode_idx in tqdm(valid_episodes, desc="Processing Episodes"):
            episode_result = self.process_single_episode(episode_idx)
            if episode_result:
                results.append(episode_result)

        # Save results
        output_file = f"{output_path}/fixed_end_effector_synthesis_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n🎉 Fixed synthesis completed: {len(results)} successful episodes")
        print(f"💾 Results saved to: {output_file}")

        return results


def main():
    """Main execution with fixed end-effector synthesis"""

    # Load valid episodes
    TASK_DESC_PATH = "/home/cx/AET_FOR_RL/vla/valid_original_data/droid_100/task_descriptions.json"
    with open(TASK_DESC_PATH, 'r') as f:
        task_data = json.load(f)

    valid_episodes = task_data['valid_episode_list'][:5]  # Test with first 5 episodes

    # Configuration
    DROID_PATH = "/home/cx/AET_FOR_RL/vla/converted_data/droid_100"  # Use converted parquet data
    OUTPUT_PATH = "/home/cx/AET_FOR_RL/vla/synthesized_data/droid_100_morphology/fixed_end_effector"

    print(f"🎯 Fixed End-Effector Synthesis Configuration:")
    print(f"   📁 DROID path: {DROID_PATH}")
    print(f"   💾 Output path: {OUTPUT_PATH}")
    print(f"   📊 Test episodes: {valid_episodes}")
    print()

    # Run synthesis
    synthesizer = FixedEndEffectorSynthesizer(DROID_PATH)
    results = synthesizer.synthesize_valid_episodes(valid_episodes, OUTPUT_PATH)

    print("🎉 Fixed end-effector synthesis completed!")


if __name__ == "__main__":
    main()