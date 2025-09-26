#!/usr/bin/env python3
"""
TFRecord图像提取器 - 从官方DROID TFRecord中提取图像数据
支持多摄像头视角和与cartesian_position数据对齐
"""
import os
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from PIL import Image
import cv2
from tqdm import tqdm
from typing import Dict, List, Tuple


class TFRecordImageExtractor:
    """从官方DROID TFRecord中提取图像数据"""

    def __init__(self, tfrecord_dir: str):
        self.tfrecord_dir = tfrecord_dir
        self.tfrecord_files = [f for f in os.listdir(tfrecord_dir)
                              if 'tfrecord' in f and f.startswith('r2d2')]

        print(f"🎥 TFRecord Image Extractor initialized")
        print(f"   Found {len(self.tfrecord_files)} TFRecord files")

    def extract_images_for_episodes(self, valid_episodes: List[int],
                                  output_dir: str,
                                  cameras: List[str] = None,
                                  max_frames_per_episode: int = 100) -> Dict:
        """为指定episodes提取图像数据"""

        if cameras is None:
            cameras = ['exterior_image_1_left', 'exterior_image_2_left', 'wrist_image_left']

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"🎯 提取图像数据:")
        print(f"   目标episodes: {len(valid_episodes)} 个")
        print(f"   摄像头: {cameras}")
        print(f"   输出目录: {output_dir}")

        extracted_data = {
            'metadata': {
                'extractor': 'TFRecordImageExtractor',
                'source': 'Official DROID TFRecord',
                'cameras': cameras,
                'total_episodes': len(valid_episodes),
                'max_frames_per_episode': max_frames_per_episode
            },
            'episodes': {},
            'statistics': {
                'successful_episodes': 0,
                'total_frames_extracted': 0,
                'failed_episodes': []
            }
        }

        episode_idx = 0
        valid_episode_set = set(valid_episodes)

        for tfrecord_file in tqdm(sorted(self.tfrecord_files), desc="Processing TFRecords"):
            file_path = os.path.join(self.tfrecord_dir, tfrecord_file)

            try:
                dataset = tf.data.TFRecordDataset([file_path])

                for raw_record in dataset:
                    if episode_idx not in valid_episode_set:
                        episode_idx += 1
                        continue

                    try:
                        episode_data = self._extract_episode_images(
                            raw_record, episode_idx, cameras,
                            output_path, max_frames_per_episode)

                        if episode_data['frame_count'] > 0:
                            extracted_data['episodes'][episode_idx] = episode_data
                            extracted_data['statistics']['successful_episodes'] += 1
                            extracted_data['statistics']['total_frames_extracted'] += episode_data['frame_count']
                        else:
                            extracted_data['statistics']['failed_episodes'].append(episode_idx)

                    except Exception as e:
                        print(f"     ❌ Episode {episode_idx} extraction failed: {e}")
                        extracted_data['statistics']['failed_episodes'].append(episode_idx)

                    episode_idx += 1

            except Exception as e:
                print(f"   ❌ Failed to process {tfrecord_file}: {e}")
                continue

        # 保存提取结果元数据
        summary_file = output_path / "extraction_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(extracted_data, f, indent=2, default=self._json_serializer)

        print(f"\n✅ 图像提取完成:")
        print(f"   成功episodes: {extracted_data['statistics']['successful_episodes']}")
        print(f"   总帧数: {extracted_data['statistics']['total_frames_extracted']}")
        print(f"   失败episodes: {len(extracted_data['statistics']['failed_episodes'])}")
        print(f"   摘要文件: {summary_file}")

        return extracted_data

    def _extract_episode_images(self, raw_record, episode_idx: int,
                               cameras: List[str], output_path: Path,
                               max_frames: int) -> Dict:
        """从单个episode提取图像数据"""

        # 创建episode目录
        episode_dir = output_path / f"episode_{episode_idx:03d}"
        episode_dir.mkdir(exist_ok=True)

        episode_data = {
            'episode_index': episode_idx,
            'cameras': {},
            'frame_count': 0,
            'episode_dir': str(episode_dir)
        }

        try:
            # 解析TFRecord
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())
            features = example.features.feature

            # 为每个摄像头创建目录
            for camera in cameras:
                camera_dir = episode_dir / camera
                camera_dir.mkdir(exist_ok=True)
                episode_data['cameras'][camera] = {
                    'directory': str(camera_dir),
                    'frames': []
                }

            # 提取图像序列
            frame_count = 0

            # 检查是否存在图像数据
            image_field_key = f'steps/observation/{cameras[0]}'
            if image_field_key not in features:
                print(f"     ⚠️ Episode {episode_idx}: 未找到图像字段 {image_field_key}")
                return episode_data

            # 获取图像数据
            for camera in cameras:
                image_field = f'steps/observation/{camera}'

                if image_field in features:
                    image_data_list = features[image_field].bytes_list.value

                    for frame_idx, image_bytes in enumerate(image_data_list[:max_frames]):
                        try:
                            # 解码JPEG图像
                            image_array = tf.image.decode_jpeg(image_bytes).numpy()

                            # 转换为PIL Image
                            pil_image = Image.fromarray(image_array)

                            # 保存图像
                            frame_filename = f"frame_{frame_idx:04d}.jpg"
                            frame_path = episode_dir / camera / frame_filename
                            pil_image.save(frame_path, 'JPEG', quality=90)

                            episode_data['cameras'][camera]['frames'].append({
                                'frame_index': frame_idx,
                                'filename': frame_filename,
                                'path': str(frame_path),
                                'shape': image_array.shape
                            })

                        except Exception as e:
                            print(f"     ⚠️ Frame {frame_idx} decode failed: {e}")
                            continue

                    frame_count = max(frame_count, len(episode_data['cameras'][camera]['frames']))

            episode_data['frame_count'] = frame_count

        except Exception as e:
            print(f"     ❌ Episode {episode_idx} parsing failed: {e}")

        return episode_data

    def create_frame_index(self, extraction_summary_path: str) -> Dict:
        """创建帧索引，用于快速查找特定timestep的图像"""

        with open(extraction_summary_path, 'r') as f:
            summary = json.load(f)

        frame_index = {}

        for episode_idx, episode_data in summary['episodes'].items():
            episode_idx = int(episode_idx)

            for frame_idx in range(episode_data['frame_count']):
                timestep_key = f"ep_{episode_idx:03d}_frame_{frame_idx:04d}"

                frame_index[timestep_key] = {
                    'episode_index': episode_idx,
                    'frame_index': frame_idx,
                    'image_paths': {}
                }

                for camera, camera_data in episode_data['cameras'].items():
                    if frame_idx < len(camera_data['frames']):
                        frame_info = camera_data['frames'][frame_idx]
                        frame_index[timestep_key]['image_paths'][camera] = frame_info['path']

        return frame_index

    def _json_serializer(self, obj):
        """JSON序列化辅助函数"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        return obj


def main():
    """测试TFRecord图像提取器"""
    import argparse

    parser = argparse.ArgumentParser(description='Extract images from DROID TFRecord')
    parser.add_argument('--input', default='/home/cx/AET_FOR_RL/vla/original_data/droid_100/1.0.0',
                       help='TFRecord directory')
    parser.add_argument('--output', default='/home/cx/AET_FOR_RL/vla/extracted_images',
                       help='Output directory for images')
    parser.add_argument('--episodes', default='/home/cx/AET_FOR_RL/vla/valid_original_data/droid_100/task_descriptions.json',
                       help='Valid episodes JSON file')
    parser.add_argument('--max-frames', type=int, default=50,
                       help='Maximum frames per episode')
    args = parser.parse_args()

    # 读取有效episodes
    with open(args.episodes, 'r') as f:
        task_data = json.load(f)
    valid_episodes = task_data['valid_episode_list']

    print(f"🎯 提取图像数据:")
    print(f"   TFRecord目录: {args.input}")
    print(f"   输出目录: {args.output}")
    print(f"   有效episodes: {len(valid_episodes)}")
    print(f"   最大帧数: {args.max_frames}")

    # 创建提取器
    extractor = TFRecordImageExtractor(args.input)

    # 提取图像
    result = extractor.extract_images_for_episodes(
        valid_episodes, args.output, max_frames_per_episode=args.max_frames)

    # 创建帧索引
    summary_path = os.path.join(args.output, "extraction_summary.json")
    frame_index = extractor.create_frame_index(summary_path)

    index_file = os.path.join(args.output, "frame_index.json")
    with open(index_file, 'w') as f:
        json.dump(frame_index, f, indent=2)

    print(f"   📋 帧索引: {index_file} ({len(frame_index)} entries)")
    print("🎉 图像提取完成!")


if __name__ == "__main__":
    main()