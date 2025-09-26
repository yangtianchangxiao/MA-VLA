#!/usr/bin/env python3
"""
独立的DROID任务描述提取模块 - 支持官方TFRecord格式
"""
import os
import json
import numpy as np
import tensorflow as tf


class DROIDTaskExtractor:
    """DROID任务描述提取器 - 支持官方TFRecord格式"""

    def __init__(self, droid_data_path: str):
        self.droid_path = droid_data_path
        self.task_descriptions = {}
        self.total_episodes = 0

        self._extract_from_tfrecords()

        print(f"🗣️  DROID Task Extractor initialized (TFRecord format)")
        print(f"   Total episodes processed: {self.total_episodes}")
        print(f"   Valid task descriptions: {len(self.task_descriptions)}")

    def _extract_from_tfrecords(self):
        """从TFRecord文件中提取任务描述"""
        tfrecord_dir = f"{self.droid_path}/1.0.0"
        tfrecord_files = [f for f in os.listdir(tfrecord_dir) if 'tfrecord' in f and f.startswith('r2d2')]

        if not tfrecord_files:
            print(f"   ❌ No TFRecord files found in {tfrecord_dir}")
            return

        print(f"   📁 Found {len(tfrecord_files)} TFRecord files")

        episode_idx = 0

        for tfrecord_file in sorted(tfrecord_files):
            file_path = os.path.join(tfrecord_dir, tfrecord_file)
            print(f"   🔄 Processing {tfrecord_file}...")

            try:
                dataset = tf.data.TFRecordDataset([file_path])

                for raw_record in dataset:
                    try:
                        # Parse the TFRecord
                        example = tf.train.Example()
                        example.ParseFromString(raw_record.numpy())

                        # Extract language instructions
                        features = example.features.feature

                        # Try different language instruction fields
                        language_keys = [
                            'steps/language_instruction',
                            'steps/language_instruction_2',
                            'steps/language_instruction_3'
                        ]

                        task_description = None
                        for key in language_keys:
                            if key in features:
                                # Get first instruction from this episode
                                instruction_list = features[key].bytes_list.value
                                if instruction_list:
                                    task_description = instruction_list[0].decode('utf-8').strip()
                                    if task_description and task_description != 'nan':
                                        break

                        if task_description:
                            self.task_descriptions[episode_idx] = task_description

                        episode_idx += 1

                    except Exception as e:
                        print(f"     ⚠️ Failed to parse record: {str(e)[:100]}...")
                        continue

            except Exception as e:
                print(f"     ❌ Failed to read {tfrecord_file}: {e}")
                continue

        self.total_episodes = episode_idx
        print(f"   ✅ Processed {self.total_episodes} episodes total")
    
    
    def get_valid_episodes(self):
        """获取所有有效任务描述的episode列表"""
        return sorted(list(self.task_descriptions.keys()))
    
    def get_task_description(self, episode_idx: int):
        """获取指定episode的任务描述"""
        return self.task_descriptions.get(episode_idx)
    
    def save_task_descriptions(self, output_path: str = "droid_task_descriptions.json"):
        """保存任务描述到文件"""
        data = {
            'metadata': {
                'extractor': 'DROIDTaskExtractor_TFRecord',
                'source': 'DROID-100 official TFRecord files',
                'total_episodes': self.total_episodes,
                'valid_descriptions': len(self.task_descriptions)
            },
            'task_descriptions': {str(k): v for k, v in self.task_descriptions.items()},
            'valid_episode_list': self.get_valid_episodes()
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"📁 Task descriptions saved to: {output_path}")
        return data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Extract DROID task descriptions')
    parser.add_argument('--output', default='/home/cx/AET_FOR_RL/vla/train/data/droid_task_descriptions.json',
                       help='Output path for task descriptions JSON file')
    args = parser.parse_args()

    extractor = DROIDTaskExtractor('/home/cx/AET_FOR_RL/vla/original_data/droid_100')

    # 保存任务描述到指定路径
    extractor.save_task_descriptions(args.output)

    print(f"\n📊 Summary:")
    print(f"   Valid episodes: {len(extractor.get_valid_episodes())}")
    print(f"   Example episodes: {extractor.get_valid_episodes()[:10]}")