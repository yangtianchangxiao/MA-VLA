#!/usr/bin/env python3
"""
Simple DROID TFRecord Parser - No RLDS dependency
Linus-style: Keep It Simple, Stupid
"""

import os
import glob
import numpy as np
import tensorflow as tf
from typing import Dict, List, Optional


def parse_float_list(feature):
    """Parse tensorflow float list feature"""
    if hasattr(feature, 'float_list') and feature.float_list.value:
        return np.array(feature.float_list.value, dtype=np.float32)
    return None


def parse_bytes_list(feature):
    """Parse tensorflow bytes list feature"""
    if hasattr(feature, 'bytes_list') and feature.bytes_list.value:
        return [item for item in feature.bytes_list.value]
    return None


def parse_int64_list(feature):
    """Parse tensorflow int64 list feature"""
    if hasattr(feature, 'int64_list') and feature.int64_list.value:
        return np.array(feature.int64_list.value, dtype=np.int64)
    return None


def extract_episode_from_tfrecord(record_bytes):
    """Extract complete episode trajectory from a single TFRecord"""
    try:
        # Parse the example
        example = tf.train.Example()
        example.ParseFromString(record_bytes)
        features = example.features.feature

        # Debug: print available features
        print(f"    Available features: {list(features.keys())[:10]}...")  # Show first 10

        # Extract trajectory sequences
        cartesian_trajectory = []
        joint_trajectory = []
        action_trajectory = []
        language_instructions = []

        # Find all timestep data by looking for indexed keys
        step_indices = set()
        for key in features.keys():
            if '/observation/' in key or '/action' in key:
                # Extract step index from key like "steps/observation/cartesian_position"
                parts = key.split('/')
                if len(parts) >= 2 and 'steps' in parts[0]:
                    step_indices.add(0)  # For now, extract first timestep of each sequence

        # If no steps structure, treat as single timestep (current behavior)
        if not step_indices:
            step_indices = {0}

        # Extract data for each timestep
        for step_idx in sorted(step_indices):
            timestep_data = {}

            # Look for observation data - exact field names from DROID
            if 'steps/observation/cartesian_position' in features:
                data = parse_float_list(features['steps/observation/cartesian_position'])
                if data is not None and len(data) >= 6:
                    timesteps_count = len(data) // 6
                    for t in range(timesteps_count):
                        cart_pos = data[t*6:(t+1)*6]
                        cartesian_trajectory.append(cart_pos)
                    print(f"    ✅ Found cartesian sequence: {len(cartesian_trajectory)} timesteps")

            if 'steps/observation/joint_position' in features:
                data = parse_float_list(features['steps/observation/joint_position'])
                if data is not None and len(data) >= 7:
                    timesteps_count = len(data) // 7
                    for t in range(timesteps_count):
                        joint_pos = data[t*7:(t+1)*7]
                        joint_trajectory.append(joint_pos)
                    print(f"    ✅ Found joint sequence: {len(joint_trajectory)} timesteps")

            # Look for action data - use action_dict/cartesian_position as action space
            if 'steps/action_dict/cartesian_position' in features:
                data = parse_float_list(features['steps/action_dict/cartesian_position'])
                if data is not None and len(data) >= 6:
                    timesteps_count = len(data) // 6
                    for t in range(timesteps_count):
                        action_6d = data[t*6:(t+1)*6]
                        # Add gripper from action_dict/gripper_position if available
                        if 'steps/action_dict/gripper_position' in features:
                            gripper_data = parse_float_list(features['steps/action_dict/gripper_position'])
                            if gripper_data is not None and t < len(gripper_data):
                                action_7d = np.concatenate([action_6d, [gripper_data[t]]])
                            else:
                                action_7d = np.concatenate([action_6d, [0.0]])
                        else:
                            action_7d = np.concatenate([action_6d, [0.0]])
                        action_trajectory.append(action_7d)
                    print(f"    ✅ Found action sequence: {len(action_trajectory)} timesteps")
            elif 'steps/action_dict/joint_position' in features:
                # Fallback to joint actions
                data = parse_float_list(features['steps/action_dict/joint_position'])
                if data is not None and len(data) >= 7:
                    timesteps_count = len(data) // 7
                    for t in range(timesteps_count):
                        action = data[t*7:(t+1)*7]
                        action_trajectory.append(action)
                    print(f"    ✅ Found joint action sequence: {len(action_trajectory)} timesteps")

            # Look for language instruction
            for key in features.keys():
                if 'language' in key.lower() or 'instruction' in key.lower() or 'task' in key.lower():
                    data = parse_bytes_list(features[key])
                    if data is not None and len(data) > 0:
                        try:
                            text = data[0].decode('utf-8')
                            if text.strip():  # Non-empty instruction
                                language_instructions.append(text)
                                print(f"    ✅ Found language: {text[:50]}...")
                        except:
                            pass
                    break

        # Return complete trajectory data
        if cartesian_trajectory and joint_trajectory:
            return {
                'cartesian_trajectory': cartesian_trajectory,
                'joint_trajectory': joint_trajectory,
                'action_trajectory': action_trajectory,
                'language_instruction': language_instructions[0] if language_instructions else "",
                'trajectory_length': len(cartesian_trajectory)
            }

        return None

    except Exception as e:
        print(f"    ❌ Failed to parse record: {e}")
        return None


def extract_droid_data_simple(input_dir: str, output_dir: str, max_files: int = 5, parquet_format: bool = False):
    """
    Simple DROID data extraction without complex dependencies

    Args:
        input_dir: Directory containing *.tfrecord* files
        output_dir: Where to save extracted data
        max_files: Maximum TFRecord files to process
    """
    print(f"🎯 Simple DROID Data Extraction")
    print(f"   📁 Input: {input_dir}")
    print(f"   💾 Output: {output_dir}")
    print(f"   📊 Max files: {max_files}")

    # Find TFRecord files
    pattern = os.path.join(input_dir, "*.tfrecord*")
    tfrecord_files = sorted(glob.glob(pattern))[:max_files]

    if not tfrecord_files:
        print(f"❌ No TFRecord files found at: {pattern}")
        return

    print(f"🔍 Found {len(tfrecord_files)} TFRecord files to process")

    os.makedirs(output_dir, exist_ok=True)

    all_episodes = []
    total_valid_samples = 0

    for file_idx, tfrecord_file in enumerate(tfrecord_files):
        print(f"\n📂 Processing file {file_idx+1}/{len(tfrecord_files)}: {os.path.basename(tfrecord_file)}")

        try:
            # Try both with and without compression
            for compression in [None, "GZIP"]:
                try:
                    dataset = tf.data.TFRecordDataset([tfrecord_file], compression_type=compression)
                    record_count = 0

                    for raw_record in dataset.take(10):  # Test first 10 records
                        episode_data = extract_episode_from_tfrecord(raw_record.numpy())
                        if episode_data:
                            episode_data['episode_index'] = len(all_episodes)
                            episode_data['source_file'] = os.path.basename(tfrecord_file)
                            all_episodes.append(episode_data)
                            # Count timesteps for trajectory data
                            if 'trajectory_length' in episode_data:
                                total_valid_samples += episode_data['trajectory_length']
                            else:
                                total_valid_samples += 1

                        record_count += 1
                        if record_count >= 10:  # Limit per file for testing
                            break

                    if record_count > 0:
                        print(f"    ✅ Successfully processed with compression={compression}")
                        break

                except Exception as e:
                    print(f"    ⚠️ Failed with compression={compression}: {e}")
                    continue

        except Exception as e:
            print(f"    ❌ Failed to process {tfrecord_file}: {e}")
            continue

    print(f"\n📊 Extraction Summary:")
    print(f"   📂 Files processed: {len(tfrecord_files)}")
    print(f"   📋 Valid episodes: {len(all_episodes)}")
    print(f"   🎯 Total samples: {total_valid_samples}")

    if all_episodes:
        if parquet_format:
            # Save in Parquet format compatible with existing synthesis runners
            save_as_parquet(all_episodes, output_dir)
        else:
            # Save as simple JSON
            import json
            output_file = os.path.join(output_dir, "extracted_droid_episodes.json")

            # Convert numpy arrays to lists for JSON serialization
            json_episodes = []
            for ep in all_episodes:
                json_ep = {}
                for k, v in ep.items():
                    if isinstance(v, np.ndarray):
                        json_ep[k] = v.tolist()
                    else:
                        json_ep[k] = v
                json_episodes.append(json_ep)

            with open(output_file, 'w') as f:
                json.dump(json_episodes, f, indent=2, default=str)

            print(f"   💾 Saved to: {output_file}")

        # Show sample data
        if all_episodes:
            sample = all_episodes[0]
            print(f"\n🔍 Sample episode data:")
            for k, v in sample.items():
                if isinstance(v, np.ndarray):
                    print(f"   {k}: shape={v.shape}, range=[{v.min():.3f}, {v.max():.3f}]")
                else:
                    print(f"   {k}: {v}")

    else:
        print(f"   ❌ No valid episodes extracted!")

    return len(all_episodes)


def save_as_parquet(episodes: List[Dict], output_dir: str):
    """Save episodes in Parquet format compatible with synthesis runners"""
    import pandas as pd
    import json

    # Create directory structure like the original
    data_dir = os.path.join(output_dir, "data", "chunk-000")
    episodes_dir = os.path.join(output_dir, "meta", "episodes", "chunk-000")

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(episodes_dir, exist_ok=True)

    # Convert episodes to timestep format
    timesteps = []
    episode_metadata = []

    for ep in episodes:
        episode_idx = ep['episode_index']

        # Handle new trajectory format
        if 'cartesian_trajectory' in ep:
            # New trajectory format
            cartesian_traj = ep['cartesian_trajectory']
            joint_traj = ep['joint_trajectory']
            action_traj = ep['action_trajectory']
            traj_length = len(cartesian_traj)

            # Create timestep for each trajectory point
            for t in range(traj_length):
                timestep = {
                    'episode_index': episode_idx,
                    'step_index': t,
                    'timestamp': t,
                    'observation.cartesian_position': cartesian_traj[t],
                    'observation.state': joint_traj[t],
                    'action': action_traj[t] if t < len(action_traj) else np.zeros(7),
                    'is_first': (t == 0),
                    'is_last': (t == traj_length - 1),
                }
                timesteps.append(timestep)

            # Episode metadata
            episode_meta = {
                'episode_index': episode_idx,
                'length': traj_length,
                'tasks': [ep.get('language_instruction', '')],
                'timestamp_start': 0,
                'timestamp_end': traj_length - 1,
            }
            episode_metadata.append(episode_meta)
        else:
            # Legacy single timestep format
            timestep = {
                'episode_index': episode_idx,
                'step_index': 0,
                'timestamp': 0,
                'observation.cartesian_position': ep['observation.cartesian_position'],
                'observation.state': ep['observation.state'],
                'action': ep['action'],
                'is_first': True,
                'is_last': True,
            }
            timesteps.append(timestep)

            # Episode metadata
            episode_meta = {
                'episode_index': episode_idx,
                'length': 1,
                'tasks': [ep.get('language_instruction', '')],
                'timestamp_start': 0,
                'timestamp_end': 0,
            }
            episode_metadata.append(episode_meta)

    # Save timestep data
    timesteps_df = pd.DataFrame(timesteps)
    timesteps_file = os.path.join(data_dir, "file-000.parquet")
    timesteps_df.to_parquet(timesteps_file, index=False)

    # Save episode metadata
    episodes_df = pd.DataFrame(episode_metadata)
    episodes_file = os.path.join(episodes_dir, "file-000.parquet")
    episodes_df.to_parquet(episodes_file, index=False)

    # Save stats
    stats = {
        "total_episodes": len(episodes),
        "total_timesteps": len(timesteps),
        "data_source": "real_droid_tfrecord_simple_parser",
    }

    stats_file = os.path.join(output_dir, "meta", "stats.json")
    os.makedirs(os.path.dirname(stats_file), exist_ok=True)
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"   💾 Parquet data saved to: {timesteps_file}")
    print(f"   💾 Episode metadata saved to: {episodes_file}")
    print(f"   💾 Stats saved to: {stats_file}")


def inspect_tfrecord_structure(tfrecord_file: str, max_records: int = 3):
    """Inspect the structure of a TFRecord file"""
    print(f"🔍 Inspecting TFRecord structure: {os.path.basename(tfrecord_file)}")

    for compression in [None, "GZIP"]:
        try:
            print(f"\n  📋 Trying compression: {compression}")
            dataset = tf.data.TFRecordDataset([tfrecord_file], compression_type=compression)

            for i, raw_record in enumerate(dataset.take(max_records)):
                print(f"\n  📄 Record {i+1}:")
                try:
                    example = tf.train.Example()
                    example.ParseFromString(raw_record.numpy())
                    features = example.features.feature

                    print(f"    📊 Total features: {len(features)}")

                    # Group features by type
                    float_features = []
                    bytes_features = []
                    int64_features = []

                    for key, feature in features.items():
                        if hasattr(feature, 'float_list') and feature.float_list.value:
                            float_features.append((key, len(feature.float_list.value)))
                        elif hasattr(feature, 'bytes_list') and feature.bytes_list.value:
                            bytes_features.append((key, len(feature.bytes_list.value)))
                        elif hasattr(feature, 'int64_list') and feature.int64_list.value:
                            int64_features.append((key, len(feature.int64_list.value)))

                    print(f"    🔢 Float features ({len(float_features)}):")
                    for name, size in float_features[:10]:  # Show first 10
                        print(f"      {name}: {size} values")

                    print(f"    📝 Bytes features ({len(bytes_features)}):")
                    for name, size in bytes_features[:10]:
                        print(f"      {name}: {size} items")

                    print(f"    🔢 Int64 features ({len(int64_features)}):")
                    for name, size in int64_features[:10]:
                        print(f"      {name}: {size} values")

                except Exception as e:
                    print(f"    ❌ Failed to parse record {i+1}: {e}")

            return True  # Successfully processed with this compression

        except Exception as e:
            print(f"  ❌ Failed with compression {compression}: {e}")

    return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Simple DROID TFRecord Parser')
    parser.add_argument('--input', default='/home/cx/AET_FOR_RL/vla/original_data/droid_100/1.0.0',
                       help='Input TFRecord directory')
    parser.add_argument('--output', default='/home/cx/AET_FOR_RL/vla/extracted_data/droid_simple',
                       help='Output directory')
    parser.add_argument('--inspect', action='store_true',
                       help='Just inspect TFRecord structure')
    parser.add_argument('--max-files', type=int, default=50,
                       help='Max TFRecord files to process')

    args = parser.parse_args()

    if args.inspect:
        # Find first TFRecord file and inspect it
        pattern = os.path.join(args.input, "*.tfrecord*")
        tfrecord_files = sorted(glob.glob(pattern))

        if tfrecord_files:
            inspect_tfrecord_structure(tfrecord_files[0])
        else:
            print(f"❌ No TFRecord files found at: {pattern}")
    else:
        # Extract data in Parquet format for compatibility
        extract_droid_data_simple(args.input, args.output, args.max_files, parquet_format=True)