#!/usr/bin/env python3
"""
通用验证数据报告生成器
用于任何数据集的验证和质量检查
"""
import os
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class ValidationReportGenerator:
    """通用验证数据报告生成器"""

    def __init__(self, data_dir: str, dataset_name: str = "unknown"):
        self.data_dir = Path(data_dir)
        self.dataset_name = dataset_name
        self.report = {}

        print(f"🔍 Validation Report Generator initialized")
        print(f"   Dataset: {self.dataset_name}")
        print(f"   Data directory: {self.data_dir}")

    def analyze_task_descriptions(self, task_file: str = "task_descriptions.json") -> Dict:
        """分析任务描述文件"""
        task_path = self.data_dir / task_file

        if not task_path.exists():
            return {
                "file_exists": False,
                "total_episodes": 0,
                "valid_descriptions": 0,
                "valid_episode_list": []
            }

        try:
            with open(task_path, 'r') as f:
                task_data = json.load(f)

            return {
                "file_exists": True,
                "file_path": str(task_path),
                "total_episodes": task_data.get('metadata', {}).get('total_episodes', 0),
                "valid_descriptions": task_data.get('metadata', {}).get('valid_descriptions', 0),
                "valid_episode_list": task_data.get('valid_episode_list', []),
                "sample_tasks": list(task_data.get('task_descriptions', {}).items())[:5]
            }
        except Exception as e:
            return {
                "file_exists": True,
                "error": str(e),
                "total_episodes": 0,
                "valid_descriptions": 0,
                "valid_episode_list": []
            }

    def analyze_image_data(self, image_dir: str = "extracted_images") -> Dict:
        """分析图像数据目录"""
        image_path = self.data_dir / image_dir

        if not image_path.exists():
            return {
                "directory_exists": False,
                "total_episodes": 0,
                "total_images": 0,
                "episode_details": []
            }

        try:
            episode_details = []
            total_images = 0

            for item in os.listdir(image_path):
                if item.startswith('episode_'):
                    episode_dir = image_path / item
                    if episode_dir.is_dir():
                        episode_num = int(item.replace('episode_', ''))

                        # 统计图像文件
                        image_files = [f for f in os.listdir(episode_dir)
                                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                        image_count = len(image_files)
                        total_images += image_count

                        episode_details.append({
                            'episode': episode_num,
                            'images': image_count,
                            'has_images': image_count > 0
                        })

            # 按episode编号排序
            episode_details.sort(key=lambda x: x['episode'])

            return {
                "directory_exists": True,
                "directory_path": str(image_path),
                "total_episodes": len(episode_details),
                "total_images": total_images,
                "episode_details": episode_details,
                "episodes_with_images": len([ep for ep in episode_details if ep['has_images']]),
                "sample_episodes": episode_details[:10]  # 前10个作为示例
            }
        except Exception as e:
            return {
                "directory_exists": True,
                "error": str(e),
                "total_episodes": 0,
                "total_images": 0,
                "episode_details": []
            }

    def calculate_data_consistency(self, task_analysis: Dict, image_analysis: Dict) -> Dict:
        """计算数据一致性统计"""
        if not task_analysis.get('file_exists') or not image_analysis.get('directory_exists'):
            return {
                "consistency_check": False,
                "reason": "Missing task descriptions or image data"
            }

        valid_task_episodes = set(task_analysis.get('valid_episode_list', []))
        image_episodes = set([ep['episode'] for ep in image_analysis.get('episode_details', [])])

        # 计算交集
        usable_episodes = valid_task_episodes & image_episodes

        return {
            "consistency_check": True,
            "episodes_with_tasks": len(valid_task_episodes),
            "episodes_with_images": len(image_episodes),
            "episodes_with_both": len(usable_episodes),
            "data_match_rate": len(usable_episodes) / max(len(valid_task_episodes), 1),
            "usable_episode_list": sorted(list(usable_episodes)),
            "missing_images": sorted(list(valid_task_episodes - image_episodes)),
            "missing_tasks": sorted(list(image_episodes - valid_task_episodes))
        }

    def generate_quality_score(self, task_analysis: Dict, image_analysis: Dict, consistency: Dict) -> float:
        """生成数据质量评分 (0-1)"""
        score = 0.0

        # 任务描述质量 (40%)
        if task_analysis.get('file_exists'):
            task_ratio = task_analysis.get('valid_descriptions', 0) / max(task_analysis.get('total_episodes', 1), 1)
            score += 0.4 * task_ratio

        # 图像数据质量 (40%)
        if image_analysis.get('directory_exists'):
            image_ratio = image_analysis.get('episodes_with_images', 0) / max(image_analysis.get('total_episodes', 1), 1)
            score += 0.4 * image_ratio

        # 数据一致性 (20%)
        if consistency.get('consistency_check'):
            consistency_ratio = consistency.get('data_match_rate', 0)
            score += 0.2 * consistency_ratio

        return min(score, 1.0)

    def generate_report(self, output_file: str = "validation_report.json") -> Dict:
        """生成完整的验证报告"""
        print("📊 Generating validation report...")

        # 分析各组件
        task_analysis = self.analyze_task_descriptions()
        image_analysis = self.analyze_image_data()
        consistency = self.calculate_data_consistency(task_analysis, image_analysis)
        quality_score = self.generate_quality_score(task_analysis, image_analysis, consistency)

        # 构建报告
        self.report = {
            "validation_metadata": {
                "dataset_name": self.dataset_name,
                "data_directory": str(self.data_dir),
                "generation_timestamp": datetime.now().isoformat(),
                "generator_version": "1.0"
            },
            "validation_criteria": {
                "complete_task_descriptions": "Episodes must have meaningful task descriptions",
                "complete_image_data": "Episodes must have corresponding image frames",
                "data_consistency": "Task descriptions and image data must be aligned"
            },
            "task_descriptions": task_analysis,
            "image_data": image_analysis,
            "data_consistency": consistency,
            "quality_assessment": {
                "overall_score": round(quality_score, 3),
                "score_breakdown": {
                    "task_quality": f"{task_analysis.get('valid_descriptions', 0)}/{task_analysis.get('total_episodes', 0)} episodes",
                    "image_quality": f"{image_analysis.get('episodes_with_images', 0)}/{image_analysis.get('total_episodes', 0)} episodes",
                    "consistency": f"{consistency.get('episodes_with_both', 0)} usable episodes"
                },
                "recommendations": self._generate_recommendations(task_analysis, image_analysis, consistency)
            }
        }

        # 保存报告
        output_path = self.data_dir / output_file
        with open(output_path, 'w') as f:
            json.dump(self.report, f, indent=2, ensure_ascii=False)

        print(f"✅ Validation report saved to: {output_path}")
        self._print_summary()

        return self.report

    def _generate_recommendations(self, task_analysis: Dict, image_analysis: Dict, consistency: Dict) -> List[str]:
        """生成改进建议"""
        recommendations = []

        if task_analysis.get('valid_descriptions', 0) < task_analysis.get('total_episodes', 0):
            missing_tasks = task_analysis.get('total_episodes', 0) - task_analysis.get('valid_descriptions', 0)
            recommendations.append(f"Consider generating task descriptions for {missing_tasks} episodes without tasks")

        if image_analysis.get('episodes_with_images', 0) < image_analysis.get('total_episodes', 0):
            missing_images = image_analysis.get('total_episodes', 0) - image_analysis.get('episodes_with_images', 0)
            recommendations.append(f"Extract image data for {missing_images} episodes without images")

        if consistency.get('data_match_rate', 0) < 1.0:
            recommendations.append("Align task descriptions and image data for complete episode coverage")

        if not recommendations:
            recommendations.append("Data quality is excellent - ready for training!")

        return recommendations

    def _print_summary(self):
        """打印报告摘要"""
        print(f"\n📋 Validation Summary for {self.dataset_name}:")

        task_data = self.report.get('task_descriptions', {})
        image_data = self.report.get('image_data', {})
        consistency = self.report.get('data_consistency', {})
        quality = self.report.get('quality_assessment', {})

        print(f"   📝 Task Descriptions: {task_data.get('valid_descriptions', 0)}/{task_data.get('total_episodes', 0)} episodes")
        print(f"   🖼️  Image Data: {image_data.get('episodes_with_images', 0)}/{image_data.get('total_episodes', 0)} episodes")
        print(f"   🔗 Usable Episodes: {consistency.get('episodes_with_both', 0)} (both task + images)")
        print(f"   ⭐ Quality Score: {quality.get('overall_score', 0):.1%}")

        recommendations = quality.get('recommendations', [])
        if recommendations:
            print(f"   💡 Recommendations:")
            for rec in recommendations:
                print(f"      - {rec}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate validation report for dataset')
    parser.add_argument('data_dir', help='Directory containing the dataset to validate')
    parser.add_argument('--dataset-name', default='unknown', help='Name of the dataset')
    parser.add_argument('--output', default='validation_report.json', help='Output report filename')
    parser.add_argument('--task-file', default='task_descriptions.json', help='Task descriptions filename')
    parser.add_argument('--image-dir', default='extracted_images', help='Image data directory name')

    args = parser.parse_args()

    generator = ValidationReportGenerator(args.data_dir, args.dataset_name)
    report = generator.generate_report(args.output)