#!/usr/bin/env python3
"""
工作空间分析工具
深入挖掘：基于DROID末端轨迹分析，生成合适的软体臂配置

工作流程：
1. 分析DROID轨迹的空间分布
2. 基于空间需求生成软体臂配置
3. 验证配置的工作空间覆盖率
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import sys
import os
sys.path.append('/home/cx/AET_FOR_RL/vla/data_augment/morphology_modules')

from soft_arm_ik_solver import SoftArmConfig, SoftArmKinematics

class WorkspaceAnalyzer:
    """工作空间分析器"""

    def __init__(self):
        self.droid_path = "/home/cx/AET_FOR_RL/vla/converted_data/droid_100_fixed/data/chunk-000/file-000.parquet"

    def load_droid_workspace_data(self, max_episodes: int = 10) -> Dict:
        """加载DROID数据并分析工作空间"""
        print(f"📊 分析DROID数据的工作空间特征...")

        try:
            df = pd.read_parquet(self.droid_path)

            # 取前几个episode进行分析
            unique_episodes = df['episode_index'].unique()[:max_episodes]

            all_positions = []
            episode_stats = []

            for ep_id in unique_episodes:
                episode_data = df[df['episode_index'] == ep_id]

                # 提取位置数据
                positions = []
                for _, row in episode_data.iterrows():
                    pos = row['observation.cartesian_position'][:3]  # [x, y, z]
                    positions.append(pos)

                positions = np.array(positions)
                all_positions.extend(positions)

                # 计算episode统计
                ep_stats = {
                    'episode_id': ep_id,
                    'num_steps': len(positions),
                    'center': np.mean(positions, axis=0),
                    'range': np.max(positions, axis=0) - np.min(positions, axis=0),
                    'max_reach': np.max(np.linalg.norm(positions, axis=1)),
                    'min_reach': np.min(np.linalg.norm(positions, axis=1)),
                    'workspace_volume': np.prod(np.max(positions, axis=0) - np.min(positions, axis=0))
                }
                episode_stats.append(ep_stats)

            all_positions = np.array(all_positions)

            # 全局工作空间统计
            global_stats = {
                'total_points': len(all_positions),
                'overall_center': np.mean(all_positions, axis=0),
                'overall_range': np.max(all_positions, axis=0) - np.min(all_positions, axis=0),
                'max_reach_from_origin': np.max(np.linalg.norm(all_positions, axis=1)),
                'min_reach_from_origin': np.min(np.linalg.norm(all_positions, axis=1)),
                'position_std': np.std(all_positions, axis=0)
            }

            return {
                'all_positions': all_positions,
                'episode_stats': episode_stats,
                'global_stats': global_stats
            }

        except Exception as e:
            print(f"❌ 加载DROID数据失败: {e}")
            return None

    def analyze_workspace_requirements(self, workspace_data: Dict) -> Dict:
        """分析工作空间需求"""
        print("🔍 分析工作空间需求...")

        positions = workspace_data['all_positions']
        global_stats = workspace_data['global_stats']

        # 计算需要的机器人特征
        requirements = {
            # 基本范围要求
            'min_reach_required': global_stats['max_reach_from_origin'] * 1.1,  # 10%余量
            'workspace_center': global_stats['overall_center'],
            'workspace_dimensions': global_stats['overall_range'],

            # 精度要求
            'position_precision_required': np.mean(global_stats['position_std']),

            # 关节数量建议
            'suggested_segments': self._estimate_required_segments(positions),

            # 段长度建议
            'suggested_segment_length': self._estimate_segment_length(global_stats['max_reach_from_origin']),
        }

        return requirements

    def _estimate_required_segments(self, positions: np.ndarray) -> int:
        """基于轨迹复杂度估计需要的段数"""
        # 计算轨迹的"弯曲复杂度"
        # 简单启发式：基于位置变化的方差
        position_variance = np.var(positions, axis=0)
        complexity_score = np.sum(position_variance)

        # 启发式映射到段数
        if complexity_score < 0.01:
            return 3  # 简单轨迹
        elif complexity_score < 0.05:
            return 4  # 中等复杂度
        elif complexity_score < 0.1:
            return 5  # 复杂轨迹
        else:
            return 6  # 非常复杂

    def _estimate_segment_length(self, max_reach: float) -> float:
        """基于最大伸展距离估计段长度"""
        # 启发式：每段长度约为最大伸展的20-30%
        return max_reach * 0.25

    def generate_workspace_matched_configs(self, requirements: Dict, num_configs: int = 5) -> List[Dict]:
        """基于工作空间需求生成软体臂配置（包含基座位置）"""
        print(f"🤖 基于工作空间需求生成{num_configs}个软体臂配置...")

        configs = []
        base_segments = requirements['suggested_segments']
        base_length = requirements['suggested_segment_length']
        workspace_center = requirements['workspace_center']

        for i in range(num_configs):
            # 在建议值附近变化
            n_segments = max(3, min(6, base_segments + np.random.randint(-1, 2)))

            # 生成段长度，确保总伸展能覆盖工作空间
            segment_lengths = []
            for j in range(n_segments):
                # 在基础长度附近变化 ±20%
                length = base_length * (0.8 + 0.4 * np.random.random())
                segment_lengths.append(length)

            # 验证总伸展是否足够
            total_reach = sum(segment_lengths)
            required_reach = requirements['min_reach_required']

            if total_reach < required_reach:
                # 按比例缩放以满足伸展要求
                scale_factor = required_reach / total_reach * 1.05  # 5%额外余量
                segment_lengths = [l * scale_factor for l in segment_lengths]

            # 计算基座位置 - 让软体臂直接能到达DROID中心
            # 策略：基座位置使得软体臂直线伸展时恰好到达工作空间中心
            total_reach = sum(segment_lengths)

            # 方案1：基座在DROID中心下方，向上伸展
            base_offset = np.array([
                workspace_center[0],                    # X对齐
                workspace_center[1],                    # Y对齐
                workspace_center[2] - total_reach * 0.6  # Z下移，留60%伸展空间
            ])

            # 确保基座不在地下
            if base_offset[2] < 0:
                # 如果会在地下，改用水平伸展策略
                base_offset = np.array([
                    workspace_center[0] - total_reach * 0.5,  # X后移50%
                    workspace_center[1],                       # Y对齐
                    0.0                                        # 在地面
                ])

            config = SoftArmConfig(n_segments, segment_lengths)

            config_with_base = {
                'soft_arm_config': config,
                'base_position': base_offset,
                'estimated_workspace_center': workspace_center,
                'total_reach': sum(segment_lengths)
            }
            configs.append(config_with_base)

        return configs

    def validate_config_coverage(self, config_with_base: Dict, target_positions: np.ndarray,
                                sample_density: int = 2000) -> Dict:
        """验证软体臂配置的工作空间覆盖率（考虑基座偏移）"""
        config = config_with_base['soft_arm_config']
        base_position = config_with_base['base_position']

        print(f"✅ 验证配置覆盖率: {config}")
        print(f"   基座位置: {base_position}")

        kinematics = SoftArmKinematics(config)

        # 使用IK直接测试可达性
        from soft_arm_ik_solver import SoftArmIKSolver
        ik_solver = SoftArmIKSolver(config)

        coverage_count = 0
        total_tested = 0
        tolerance = 0.05  # 5cm容差

        # 测试每个目标点的可达性
        for target_pos in target_positions[::5]:  # 每5个点测试一次，提高密度
            total_tested += 1

            # 转换到软体臂坐标系
            target_relative = target_pos - base_position
            target_distance = np.linalg.norm(target_relative)

            # 先检查是否在理论伸展范围内
            if target_distance > config.max_reach:
                continue

            # 尝试IK求解
            try:
                curvature_params, success, error = ik_solver.solve_ik(target_relative)
                if success and error < tolerance:
                    coverage_count += 1
            except:
                continue

        coverage_rate = coverage_count / total_tested if total_tested > 0 else 0.0

        # 随机采样少量点用于可视化
        sampled_positions = []
        for _ in range(200):  # 减少采样数量
            alpha_params = np.random.uniform(0.1, np.pi*0.8, config.n_segments)
            beta_params = np.random.uniform(0, 2*np.pi, config.n_segments)

            curvature_params = []
            for i in range(config.n_segments):
                curvature_params.extend([alpha_params[i], beta_params[i]])

            try:
                pos, _ = kinematics.forward_kinematics(np.array(curvature_params))
                world_pos = pos + base_position
                sampled_positions.append(world_pos)
            except:
                continue

        sampled_positions = np.array(sampled_positions)

        return {
            'coverage_rate': coverage_rate,
            'reachable_points': len(sampled_positions),
            'sampled_workspace_range': np.max(sampled_positions, axis=0) - np.min(sampled_positions, axis=0),
            'max_reach_achieved': np.max(np.linalg.norm(sampled_positions - base_position, axis=1)),
            'base_position': base_position
        }

def main():
    """主函数：完整的工作空间分析流程"""
    print("🔍 深入挖掘：基于工作空间的软体臂配置生成")
    print("="*60)

    analyzer = WorkspaceAnalyzer()

    # 1. 分析DROID工作空间
    workspace_data = analyzer.load_droid_workspace_data(max_episodes=5)
    if workspace_data is None:
        return

    print(f"📊 DROID工作空间分析结果:")
    global_stats = workspace_data['global_stats']
    print(f"  总采样点: {global_stats['total_points']}")
    print(f"  中心位置: {global_stats['overall_center']}")
    print(f"  空间范围: {global_stats['overall_range']}")
    print(f"  最大伸展: {global_stats['max_reach_from_origin']:.3f}m")
    print(f"  最小伸展: {global_stats['min_reach_from_origin']:.3f}m")

    # 2. 分析需求
    requirements = analyzer.analyze_workspace_requirements(workspace_data)
    print(f"\n🎯 工作空间需求分析:")
    print(f"  建议段数: {requirements['suggested_segments']}")
    print(f"  建议段长度: {requirements['suggested_segment_length']:.3f}m")
    print(f"  需要伸展: {requirements['min_reach_required']:.3f}m")

    # 3. 生成匹配的配置
    configs = analyzer.generate_workspace_matched_configs(requirements, num_configs=3)

    # 4. 验证配置
    target_positions = workspace_data['all_positions']

    print(f"\n🤖 生成的软体臂配置验证:")
    for i, config_with_base in enumerate(configs):
        coverage_result = analyzer.validate_config_coverage(config_with_base, target_positions, sample_density=1000)

        config = config_with_base['soft_arm_config']
        base_pos = config_with_base['base_position']

        print(f"\n  配置 {i+1}: {config}")
        print(f"    基座位置: [{base_pos[0]:.3f}, {base_pos[1]:.3f}, {base_pos[2]:.3f}]")
        print(f"    工作空间覆盖率: {coverage_result['coverage_rate']:.1%}")
        print(f"    最大伸展: {coverage_result['max_reach_achieved']:.3f}m")
        print(f"    可达点数: {coverage_result['reachable_points']}")

        if coverage_result['coverage_rate'] > 0.7:
            print(f"    ✅ 配置良好，覆盖率 > 70%")
        elif coverage_result['coverage_rate'] > 0.5:
            print(f"    ⚠️ 配置可用，覆盖率 > 50%")
        else:
            print(f"    ❌ 配置不足，覆盖率 < 50%")

if __name__ == "__main__":
    main()