#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple End-Effector Synthesis - Linus Style
KISS principle: Keep It Simple, Stupid

Instead of fixing the broken quaternion mess, use the ORIGINAL module directly
with proper data validation. Sometimes the simplest fix is no fix at all.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import json
from tqdm import tqdm

from morphology_modules.end_effector_synthesis_module import EndEffectorSynthesisModule


def extract_droid_trajectory(episode_data):
    """Extract clean 7D trajectory from DROID data"""
    ee_trajectory = []

    for _, row in episode_data.iterrows():
        # Get 6D cartesian position
        cartesian_pos = np.array(row['observation.cartesian_position'])
        if len(cartesian_pos) != 6:
            continue

        # Get gripper from action
        action = row['action'] if 'action' in row else np.zeros(7)
        gripper = action[6] if len(action) > 6 else 0.0

        # Combine into 7D: [x,y,z,rx,ry,rz,gripper]
        ee_step = np.concatenate([cartesian_pos, [gripper]])
        ee_trajectory.append(ee_step)

    return np.array(ee_trajectory) if ee_trajectory else None


def test_synthesis_quality(original_trajectory, synthesized_results):
    """Simple quality check"""
    if not synthesized_results:
        return 0.0

    total_quality = 0.0
    valid_robots = 0

    for robot_name, joint_traj in synthesized_results.items():
        if joint_traj is not None and len(joint_traj) > 0:
            # Check trajectory smoothness (simple velocity check)
            if len(joint_traj) >= 2:
                velocities = np.diff(joint_traj[:, :-1], axis=0)  # Exclude gripper
                max_velocity = np.max(np.abs(velocities))

                # Reasonable velocity threshold (rad/timestep)
                if max_velocity < 1.0:  # 1 rad per timestep
                    total_quality += 1.0
                    valid_robots += 1
                    print(f"    ✅ {robot_name}: max_vel={max_velocity:.3f}")
                else:
                    print(f"    ❌ {robot_name}: max_vel={max_velocity:.3f} (too high)")
            else:
                print(f"    ❌ {robot_name}: trajectory too short")
        else:
            print(f"    ❌ {robot_name}: synthesis failed")

    return total_quality / len(synthesized_results) if synthesized_results else 0.0


def main():
    """Simple test of original end-effector synthesis"""

    print("🎯 Simple End-Effector Synthesis Test - Linus Style")
    print("=" * 60)
    print("Philosophy: Use the original module, just validate inputs properly")
    print()

    # Load data
    DROID_PATH = "/home/cx/AET_FOR_RL/vla/converted_data/droid_100_fixed"
    data_df = pd.read_parquet(f"{DROID_PATH}/data/chunk-000/file-000.parquet")

    # Test with first few episodes
    test_episodes = [0, 1, 2]

    # Initialize original synthesis module
    synthesis_module = EndEffectorSynthesisModule()

    results = []

    for episode_idx in test_episodes:
        print(f"\n📊 Testing episode {episode_idx}")

        # Extract episode data
        episode_data = data_df[data_df['episode_index'] == episode_idx]
        if len(episode_data) == 0:
            print(f"   ❌ No data for episode {episode_idx}")
            continue

        print(f"   📋 Episode has {len(episode_data)} frames")

        # Extract trajectory
        ee_trajectory = extract_droid_trajectory(episode_data)
        if ee_trajectory is None:
            print(f"   ❌ Failed to extract trajectory")
            continue

        print(f"   ✅ Extracted trajectory: {ee_trajectory.shape}")
        print(f"   📐 Position range: {ee_trajectory[:, :3].min(axis=0)} to {ee_trajectory[:, :3].max(axis=0)}")
        print(f"   🔄 Orientation range: {ee_trajectory[:, 3:6].min(axis=0)} to {ee_trajectory[:, 3:6].max(axis=0)}")
        print(f"   🤖 Gripper range: {ee_trajectory[:, 6].min()} to {ee_trajectory[:, 6].max()}")

        # Apply original synthesis (no modifications)
        try:
            print(f"   🔄 Running original synthesis...")
            synthesized_results = synthesis_module.apply_to_trajectory(ee_trajectory, {})

            # Test quality
            quality_score = test_synthesis_quality(ee_trajectory, synthesized_results)

            print(f"   📊 Quality score: {quality_score:.1%}")

            if quality_score > 0.5:  # At least 50% of robots succeeded
                results.append({
                    'episode_index': episode_idx,
                    'ee_trajectory': ee_trajectory,
                    'synthesized_results': synthesized_results,
                    'quality_score': quality_score,
                    'status': 'success'
                })
                print(f"   ✅ Episode {episode_idx}: SUCCESS")
            else:
                print(f"   ❌ Episode {episode_idx}: LOW QUALITY")

        except Exception as e:
            print(f"   ❌ Episode {episode_idx}: FAILED - {e}")

    # Summary
    print(f"\n🎉 Test Summary:")
    print(f"   📊 Episodes tested: {len(test_episodes)}")
    print(f"   ✅ Successful episodes: {len(results)}")

    if results:
        avg_quality = np.mean([r['quality_score'] for r in results])
        print(f"   📈 Average quality: {avg_quality:.1%}")

        # Save results
        output_dir = "/home/cx/AET_FOR_RL/vla/synthesized_data/droid_100_morphology/simple_end_effector"
        os.makedirs(output_dir, exist_ok=True)

        output_file = f"{output_dir}/simple_synthesis_test_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"   💾 Results saved: {output_file}")
    else:
        print(f"   ❌ No successful episodes")

    print(f"\n🏁 Test completed!")


if __name__ == "__main__":
    main()