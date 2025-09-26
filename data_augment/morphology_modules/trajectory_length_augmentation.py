#!/usr/bin/env python3
"""
Trajectory Length Augmentation Module
Linus-style: Simple, reliable, no special cases
"""

import numpy as np
from typing import Tuple, List, Optional
from scipy.interpolate import interp1d


class TrajectoryLengthAugmenter:
    """
    Simple trajectory length augmentation with workspace validation

    Philosophy: Keep it simple - just resample time axis with bounds checking
    """

    def __init__(self,
                 length_range: Tuple[float, float] = (0.5, 2.0),
                 workspace_bounds: Optional[np.ndarray] = None,
                 min_length: int = 10,
                 max_length: int = 1000):
        """
        Args:
            length_range: (min_factor, max_factor) for length scaling
            workspace_bounds: (2, 3) array [[x_min, y_min, z_min], [x_max, y_max, z_max]]
            min_length: Minimum trajectory length
            max_length: Maximum trajectory length
        """
        self.length_range = length_range
        self.min_length = min_length
        self.max_length = max_length

        # Default Franka workspace bounds if not provided
        if workspace_bounds is None:
            self.workspace_bounds = np.array([
                [0.2, -0.5, 0.1],   # [x_min, y_min, z_min]
                [0.8,  0.5, 0.8]    # [x_max, y_max, z_max]
            ])
        else:
            self.workspace_bounds = workspace_bounds

        print(f"🔄 TrajectoryLengthAugmenter:")
        print(f"   Length range: {length_range}")
        print(f"   Bounds: {min_length}-{max_length} timesteps")
        print(f"   Workspace: {self.workspace_bounds[0]} to {self.workspace_bounds[1]}")

    def is_in_workspace(self, positions: np.ndarray) -> np.ndarray:
        """
        Check if positions are within workspace bounds

        Args:
            positions: (T, 3) array of [x, y, z] positions

        Returns:
            (T,) boolean array indicating valid positions
        """
        lower_bound = self.workspace_bounds[0]  # [x_min, y_min, z_min]
        upper_bound = self.workspace_bounds[1]  # [x_max, y_max, z_max]

        # Check each dimension
        valid = np.all(
            (positions >= lower_bound) & (positions <= upper_bound),
            axis=1
        )
        return valid

    def resample_trajectory(self, trajectory: np.ndarray, new_length: int) -> np.ndarray:
        """
        Resample trajectory to new length using linear interpolation

        Args:
            trajectory: (T, 7) array [x,y,z,rx,ry,rz,gripper]
            new_length: Target trajectory length

        Returns:
            (new_length, 7) resampled trajectory
        """
        original_length = len(trajectory)

        # Original time indices
        original_t = np.linspace(0, 1, original_length)

        # New time indices
        new_t = np.linspace(0, 1, new_length)

        # Interpolate each dimension
        resampled = np.zeros((new_length, trajectory.shape[1]))

        for dim in range(trajectory.shape[1]):
            if dim < 6:  # Position and orientation - smooth interpolation
                interp_func = interp1d(original_t, trajectory[:, dim],
                                     kind='cubic', bounds_error=False,
                                     fill_value='extrapolate')
            else:  # Gripper - linear interpolation (discrete-ish)
                interp_func = interp1d(original_t, trajectory[:, dim],
                                     kind='linear', bounds_error=False,
                                     fill_value='extrapolate')

            resampled[:, dim] = interp_func(new_t)

        return resampled

    def generate_random_length(self, original_length: int) -> int:
        """Generate random new length within bounds"""
        # Random scaling factor
        scale_factor = np.random.uniform(self.length_range[0], self.length_range[1])

        # Apply scaling
        new_length = int(original_length * scale_factor)

        # Enforce bounds
        new_length = max(self.min_length, min(self.max_length, new_length))

        return new_length

    def augment_trajectory(self, trajectory: np.ndarray,
                          target_length: Optional[int] = None) -> Tuple[np.ndarray, dict]:
        """
        Augment trajectory with length variation and workspace validation

        Args:
            trajectory: (T, 7) array [x,y,z,rx,ry,rz,gripper]
            target_length: Specific target length (if None, use random)

        Returns:
            augmented_trajectory: (new_T, 7) array
            info: Dict with augmentation statistics
        """
        original_length = len(trajectory)

        # Determine new length
        if target_length is None:
            new_length = self.generate_random_length(original_length)
        else:
            new_length = max(self.min_length, min(self.max_length, target_length))

        # Resample trajectory
        augmented = self.resample_trajectory(trajectory, new_length)

        # Validate workspace constraints
        positions = augmented[:, :3]  # [x, y, z]
        valid_mask = self.is_in_workspace(positions)
        valid_ratio = np.mean(valid_mask)

        # Statistics
        info = {
            'original_length': original_length,
            'new_length': new_length,
            'scale_factor': new_length / original_length,
            'valid_ratio': valid_ratio,
            'workspace_violations': np.sum(~valid_mask),
            'success': valid_ratio > 0.9  # At least 90% must be in workspace
        }

        return augmented, info

    def batch_augment(self, trajectories: List[np.ndarray],
                     augmentations_per_traj: int = 3) -> Tuple[List[np.ndarray], List[dict]]:
        """
        Batch augment multiple trajectories

        Args:
            trajectories: List of (T_i, 7) trajectory arrays
            augmentations_per_traj: Number of augmentations per original trajectory

        Returns:
            augmented_trajectories: List of augmented trajectories
            infos: List of augmentation info dicts
        """
        augmented_trajectories = []
        infos = []

        for traj_idx, trajectory in enumerate(trajectories):
            print(f"🔄 Augmenting trajectory {traj_idx} (original length: {len(trajectory)})")

            for aug_idx in range(augmentations_per_traj):
                augmented, info = self.augment_trajectory(trajectory)

                if info['success']:
                    augmented_trajectories.append(augmented)
                    infos.append({**info, 'original_idx': traj_idx, 'augmentation_idx': aug_idx})
                    print(f"   ✅ Aug {aug_idx}: {info['original_length']} → {info['new_length']} "
                          f"(scale={info['scale_factor']:.2f}, valid={info['valid_ratio']:.1%})")
                else:
                    print(f"   ❌ Aug {aug_idx}: Failed workspace validation "
                          f"(valid={info['valid_ratio']:.1%})")

        success_rate = len(augmented_trajectories) / (len(trajectories) * augmentations_per_traj)
        print(f"\n📊 Batch augmentation summary:")
        print(f"   Total generated: {len(augmented_trajectories)}")
        print(f"   Success rate: {success_rate:.1%}")

        return augmented_trajectories, infos


def test_length_augmentation():
    """Test the length augmentation on real DROID data"""
    import pandas as pd

    # Load real data
    df = pd.read_parquet('/home/cx/AET_FOR_RL/vla/converted_data/droid_100_fixed/data/chunk-000/file-000.parquet')

    # Extract trajectories
    trajectories = []
    for ep_id in df['episode_index'].unique():
        ep_data = df[df['episode_index'] == ep_id]
        # Convert to end-effector trajectory format
        traj = []
        for _, row in ep_data.iterrows():
            ee_step = np.concatenate([
                row['observation.cartesian_position'],  # [x,y,z,rx,ry,rz]
                [row['action'][6]]  # gripper
            ])
            traj.append(ee_step)
        trajectories.append(np.array(traj))

    print(f"🔍 Loaded {len(trajectories)} real DROID trajectories")
    for i, traj in enumerate(trajectories):
        print(f"   Episode {i}: {len(traj)} timesteps")

    # Test augmentation
    augmenter = TrajectoryLengthAugmenter(
        length_range=(0.7, 1.5),  # 70% to 150% of original
        min_length=20,
        max_length=400
    )

    augmented_trajs, infos = augmenter.batch_augment(
        trajectories,
        augmentations_per_traj=5
    )

    # Show results
    print(f"\n🎉 Length augmentation test completed!")
    print(f"   Original: {len(trajectories)} trajectories")
    print(f"   Augmented: {len(augmented_trajs)} trajectories")

    if augmented_trajs:
        lengths = [len(traj) for traj in augmented_trajs]
        print(f"   Length range: {min(lengths)} - {max(lengths)} timesteps")
        print(f"   Average length: {np.mean(lengths):.1f} timesteps")


if __name__ == "__main__":
    test_length_augmentation()