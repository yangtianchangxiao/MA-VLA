#!/usr/bin/env python3
"""
Length-Augmented End-Effector Synthesis Runner
Combines trajectory length variation with morphology synthesis
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

# Add modules to path
sys.path.append('/home/cx/AET_FOR_RL/vla/data_augment/morphology_modules')

from end_effector_synthesis_module import EndEffectorSynthesisModule
from trajectory_length_augmentation import TrajectoryLengthAugmenter


class LengthAugmentedSynthesisRunner:
    """
    Combined length augmentation and morphology synthesis

    Pipeline:
    1. Load DROID data
    2. Augment trajectory lengths
    3. Synthesize for different robot morphologies
    4. Save augmented dataset
    """

    def __init__(self,
                 length_range: Tuple[float, float] = (0.7, 1.5),
                 augmentations_per_episode: int = 3,
                 target_robots: List[Dict] = None):
        """
        Args:
            length_range: (min_factor, max_factor) for length scaling
            augmentations_per_episode: How many length variants per episode
            target_robots: List of robot configurations
        """
        self.length_range = length_range
        self.augmentations_per_episode = augmentations_per_episode

        # Initialize modules
        self.length_augmenter = TrajectoryLengthAugmenter(
            length_range=length_range,
            min_length=20,
            max_length=500
        )

        # Use only working robots (no broken scaled versions)
        if target_robots is None:
            target_robots = [
                {
                    'name': 'franka_panda_7dof',
                    'dh_params': np.array([
                        [0,       0.333,  0,       np.pi/2],
                        [0,       0,      -0.316,  0],
                        [0,       0.316,  0,       np.pi/2],
                        [0.0825,  0,      0,       np.pi/2],
                        [-0.0825, 0.384,  0,      -np.pi/2],
                        [0,       0,      0,       np.pi/2],
                        [0.088,   0.107,  0,       0]
                    ]),
                    'dof': 7,
                    'joint_limits': [
                        (-2.9, 2.9), (-1.76, 1.76), (-2.9, 2.9),
                        (-3.07, 0.07), (-2.9, 2.9), (-0.02, 3.75), (-2.9, 2.9)
                    ]
                }
            ]

        self.morphology_synthesizer = EndEffectorSynthesisModule(
            target_robots=target_robots
        )

        print(f"🎯 LengthAugmentedSynthesisRunner:")
        print(f"   Length range: {length_range}")
        print(f"   Augmentations per episode: {augmentations_per_episode}")
        print(f"   Target robots: {len(target_robots)}")

    def extract_ee_trajectory_from_episode(self, episode_data: pd.DataFrame) -> np.ndarray:
        """Extract end-effector trajectory from episode data"""
        trajectory = []
        for _, row in episode_data.iterrows():
            # Use observation.cartesian_position as end-effector pose
            ee_step = np.concatenate([
                row['observation.cartesian_position'],  # [x,y,z,rx,ry,rz]
                [row['action'][6] if len(row['action']) > 6 else 0.0]  # gripper
            ])
            trajectory.append(ee_step)
        return np.array(trajectory)

    def process_single_episode(self, episode_id: int, episode_data: pd.DataFrame) -> List[Dict]:
        """Process a single episode with length augmentation and morphology synthesis"""
        print(f"\n📂 Processing Episode {episode_id} ({len(episode_data)} timesteps)")

        # Extract original end-effector trajectory
        original_ee_traj = self.extract_ee_trajectory_from_episode(episode_data)

        # Get original metadata
        original_metadata = {
            'episode_index': episode_id,
            'original_length': len(episode_data),
            'language_instruction': episode_data.iloc[0].get('language_instruction', ''),
            'source_file': episode_data.iloc[0].get('source_file', '')
        }

        results = []

        # 1. Process original trajectory (no length augmentation)
        print(f"   🔄 Original trajectory...")
        original_synthesis = self.morphology_synthesizer.apply_to_trajectory(original_ee_traj, {})

        for robot_name, robot_traj in original_synthesis.items():
            if robot_traj is not None:
                result = {
                    **original_metadata,
                    'augmentation_type': 'original',
                    'augmentation_idx': 0,
                    'robot_name': robot_name,
                    'trajectory': robot_traj,
                    'length_scale_factor': 1.0,
                    'final_length': len(robot_traj)
                }
                results.append(result)
                print(f"     ✅ {robot_name}: {len(robot_traj)} timesteps")

        # 2. Generate length-augmented trajectories
        print(f"   🔄 Length augmentation...")
        augmented_trajs, aug_infos = self.length_augmenter.batch_augment(
            [original_ee_traj],
            augmentations_per_traj=self.augmentations_per_episode
        )

        # 3. Synthesize morphologies for each augmented trajectory
        for aug_idx, (aug_traj, aug_info) in enumerate(zip(augmented_trajs, aug_infos)):
            print(f"   🔄 Augmented trajectory {aug_idx+1} "
                  f"({aug_info['original_length']} → {aug_info['new_length']} timesteps)...")

            aug_synthesis = self.morphology_synthesizer.apply_to_trajectory(aug_traj, {})

            for robot_name, robot_traj in aug_synthesis.items():
                if robot_traj is not None:
                    result = {
                        **original_metadata,
                        'augmentation_type': 'length_augmented',
                        'augmentation_idx': aug_idx + 1,
                        'robot_name': robot_name,
                        'trajectory': robot_traj,
                        'length_scale_factor': aug_info['scale_factor'],
                        'final_length': len(robot_traj)
                    }
                    results.append(result)
                    print(f"     ✅ {robot_name}: {len(robot_traj)} timesteps")

        return results

    def run_synthesis(self, input_path: str, output_dir: str) -> Dict:
        """Run complete length-augmented synthesis pipeline"""
        print(f"🚀 Length-Augmented Synthesis Pipeline")
        print(f"   📁 Input: {input_path}")
        print(f"   💾 Output: {output_dir}")
        print("=" * 60)

        # Load data
        df = pd.read_parquet(input_path)
        print(f"📊 Loaded {len(df)} timesteps from {df['episode_index'].nunique()} episodes")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Process each episode
        all_results = []
        statistics = {
            'original_episodes': 0,
            'augmented_trajectories': 0,
            'total_robot_trajectories': 0,
            'robots_tested': set(),
            'length_stats': []
        }

        for episode_id in sorted(df['episode_index'].unique()):
            episode_data = df[df['episode_index'] == episode_id]
            episode_results = self.process_single_episode(episode_id, episode_data)

            all_results.extend(episode_results)
            statistics['original_episodes'] += 1

            # Update stats
            for result in episode_results:
                statistics['robots_tested'].add(result['robot_name'])
                statistics['total_robot_trajectories'] += 1
                if result['augmentation_type'] == 'length_augmented':
                    statistics['augmented_trajectories'] += 1
                statistics['length_stats'].append({
                    'original_length': result['original_length'],
                    'final_length': result['final_length'],
                    'scale_factor': result['length_scale_factor']
                })

        # Convert trajectories to timestep format for training
        timesteps_data = []
        episode_metadata = []

        for result in all_results:
            trajectory = result['trajectory']
            traj_length = len(trajectory)

            # Create unique episode index for augmented data
            unique_episode_id = f"{result['episode_index']}_{result['augmentation_type']}_{result['augmentation_idx']}_{result['robot_name']}"

            # Convert trajectory to timesteps
            for t in range(traj_length):
                timestep = {
                    'episode_index': unique_episode_id,
                    'step_index': t,
                    'timestamp': t,
                    'robot_name': result['robot_name'],
                    'augmentation_type': result['augmentation_type'],
                    'original_episode_id': result['episode_index'],
                    'action': trajectory[t],  # Joint + gripper trajectory
                    'is_first': (t == 0),
                    'is_last': (t == traj_length - 1),
                }
                timesteps_data.append(timestep)

            # Episode metadata
            episode_meta = {
                'episode_index': unique_episode_id,
                'original_episode_id': result['episode_index'],
                'robot_name': result['robot_name'],
                'augmentation_type': result['augmentation_type'],
                'length': traj_length,
                'length_scale_factor': result['length_scale_factor'],
                'language_instruction': result['language_instruction'],
                'source_file': result.get('source_file', ''),
                'timestamp_start': 0,
                'timestamp_end': traj_length - 1,
            }
            episode_metadata.append(episode_meta)

        # Save results
        output_timesteps = os.path.join(output_dir, "timesteps.parquet")
        output_episodes = os.path.join(output_dir, "episodes.parquet")
        output_stats = os.path.join(output_dir, "synthesis_stats.json")

        pd.DataFrame(timesteps_data).to_parquet(output_timesteps, index=False)
        pd.DataFrame(episode_metadata).to_parquet(output_episodes, index=False)

        # Final statistics
        length_stats = statistics['length_stats']
        final_stats = {
            'original_episodes': statistics['original_episodes'],
            'total_synthesized_episodes': len(episode_metadata),
            'total_timesteps': len(timesteps_data),
            'robots_used': list(statistics['robots_tested']),
            'augmentation_multiplier': len(all_results) / statistics['original_episodes'],
            'length_distribution': {
                'min_length': min(s['final_length'] for s in length_stats),
                'max_length': max(s['final_length'] for s in length_stats),
                'avg_length': np.mean([s['final_length'] for s in length_stats]),
                'scale_factors': [s['scale_factor'] for s in length_stats]
            }
        }

        import json
        with open(output_stats, 'w') as f:
            json.dump(final_stats, f, indent=2, default=str)

        print(f"\n🎉 Length-Augmented Synthesis Completed!")
        print(f"   📊 Original episodes: {final_stats['original_episodes']}")
        print(f"   📊 Synthesized episodes: {final_stats['total_synthesized_episodes']}")
        print(f"   📊 Total timesteps: {final_stats['total_timesteps']}")
        print(f"   📊 Augmentation multiplier: {final_stats['augmentation_multiplier']:.1f}x")
        print(f"   📊 Length range: {final_stats['length_distribution']['min_length']}-{final_stats['length_distribution']['max_length']}")
        print(f"   💾 Saved to: {output_dir}")

        return final_stats


def main():
    """Main execution"""
    runner = LengthAugmentedSynthesisRunner(
        length_range=(0.8, 1.4),  # 80% to 140% length variation
        augmentations_per_episode=4  # 4 length variants per episode
    )

    input_path = "/home/cx/AET_FOR_RL/vla/converted_data/droid_100_fixed/data/chunk-000/file-000.parquet"
    output_dir = "/home/cx/AET_FOR_RL/vla/synthesized_data/length_augmented_droid"

    runner.run_synthesis(input_path, output_dir)


if __name__ == "__main__":
    main()