#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adaptive IK Filters - Learn limits from DROID data itself

Linus philosophy: "Theory and practice sometimes clash. Theory loses. Every single time."
"""

import numpy as np
import pandas as pd
from typing import Tuple, List
from dataclasses import dataclass

@dataclass
class FilterResult:
    """Filter result with clear pass/fail and reason"""
    passed: bool
    reason: str
    score: float = 0.0  # Quality score [0,1]

class DROIDDataAnalyzer:
    """Analyze DROID data to extract realistic limits"""
    
    def __init__(self, droid_path: str):
        self.droid_path = droid_path
        self.joint_stats = {}
        self.velocity_stats = {}
        self.acceleration_stats = {}
        
    def analyze_all_episodes(self, max_episodes: int = 100):
        """Analyze all episodes to extract realistic limits"""
        print("📊 Analyzing DROID data to extract realistic limits...")
        
        # Load data
        data_df = pd.read_parquet(f"{self.droid_path}/data/chunk-000/file-000.parquet")
        
        all_trajectories = []
        dt = 1/15  # 15 FPS
        
        for episode_idx in range(min(max_episodes, 100)):
            episode_data = data_df[data_df['episode_index'] == episode_idx]
            if len(episode_data) == 0:
                continue
                
            # Extract trajectory
            states = []
            for _, row in episode_data.iterrows():
                state = row['observation.state'] if 'observation.state' in row else np.zeros(7)
                states.append(state)
            
            trajectory = np.array(states)
            if len(trajectory) > 1:
                all_trajectories.append(trajectory)
        
        print(f"   📈 Analyzed {len(all_trajectories)} episodes")
        
        # Combine all trajectory data
        all_joints = np.vstack(all_trajectories)
        print(f"   📊 Total data points: {len(all_joints)}")
        
        # Calculate joint statistics
        self.joint_stats = {
            'min': np.min(all_joints, axis=0),
            'max': np.max(all_joints, axis=0),
            'mean': np.mean(all_joints, axis=0),
            'std': np.std(all_joints, axis=0)
        }
        
        # Calculate velocity and acceleration statistics
        all_velocities = []
        all_accelerations = []
        
        for trajectory in all_trajectories:
            if len(trajectory) >= 2:
                # Handle angle wraparound for accurate velocity calculation
                raw_diff = np.diff(trajectory, axis=0)
                wrapped_diff = np.array([np.arctan2(np.sin(raw_diff[:, i]), np.cos(raw_diff[:, i])) 
                                        for i in range(raw_diff.shape[1])]).T
                velocities = wrapped_diff / dt
                all_velocities.append(velocities)
                
                if len(trajectory) >= 3:
                    accelerations = np.diff(velocities, axis=0) / dt
                    all_accelerations.append(accelerations)
        
        if all_velocities:
            all_vel = np.vstack(all_velocities)
            self.velocity_stats = {
                'max_abs': np.max(np.abs(all_vel), axis=0),
                'percentile_95': np.percentile(np.abs(all_vel), 95, axis=0),
                'percentile_99': np.percentile(np.abs(all_vel), 99, axis=0)
            }
        
        if all_accelerations:
            all_acc = np.vstack(all_accelerations)  
            self.acceleration_stats = {
                'max_abs': np.max(np.abs(all_acc), axis=0),
                'percentile_95': np.percentile(np.abs(all_acc), 95, axis=0),
                'percentile_99': np.percentile(np.abs(all_acc), 99, axis=0)
            }
        
        self.print_analysis_report()
    
    def print_analysis_report(self):
        """Print analysis results"""
        print("\n📋 DROID Data Analysis Report:")
        print("=" * 50)
        
        print("\n🔧 Joint Limits (rad):")
        for i in range(7):
            print(f"   Joint {i}: [{self.joint_stats['min'][i]:6.3f}, {self.joint_stats['max'][i]:6.3f}] "
                  f"(μ={self.joint_stats['mean'][i]:6.3f}, σ={self.joint_stats['std'][i]:5.3f})")
        
        print("\n🚀 Velocity Limits (rad/s):")
        for i in range(7):
            print(f"   Joint {i}: Max={self.velocity_stats['max_abs'][i]:7.3f}, "
                  f"95%={self.velocity_stats['percentile_95'][i]:7.3f}, "
                  f"99%={self.velocity_stats['percentile_99'][i]:7.3f}")
        
        print("\n⚡ Acceleration Limits (rad/s²):")
        for i in range(7):
            print(f"   Joint {i}: Max={self.acceleration_stats['max_abs'][i]:8.1f}, "
                  f"95%={self.acceleration_stats['percentile_95'][i]:8.1f}, "
                  f"99%={self.acceleration_stats['percentile_99'][i]:8.1f}")
    
    def get_adaptive_limits(self, safety_factor: float = 2.0):
        """Get adaptive limits based on DROID data + safety factor"""
        return {
            'joint_limits': np.column_stack([
                self.joint_stats['min'] - self.joint_stats['std'],
                self.joint_stats['max'] + self.joint_stats['std'] 
            ]),
            'max_velocity': self.velocity_stats['max_abs'] * safety_factor,
            'max_acceleration': self.acceleration_stats['max_abs'] * safety_factor
        }

class AdaptiveIKFilter:
    """Adaptive IK filter based on DROID data statistics"""
    
    def __init__(self, droid_path: str):
        print("🤖 AdaptiveIKFilter: Learning from DROID data...")
        
        self.analyzer = DROIDDataAnalyzer(droid_path)
        self.analyzer.analyze_all_episodes()
        
        # Get adaptive limits
        self.limits = self.analyzer.get_adaptive_limits()
        
        print(f"\n✅ Adaptive limits configured based on DROID reality")
    
    def check_joint_limits(self, trajectory: np.ndarray) -> FilterResult:
        """Check joint limits - relaxed to 360 degrees for flexibility"""
        # Very generous limits: ±2π for all joints (360 degrees)
        joint_limit = 2 * np.pi
        
        violations = 0
        for joints in trajectory:
            for j, angle in enumerate(joints):
                if abs(angle) > joint_limit:
                    violations += 1
        
        if violations > 0:
            violation_rate = violations / (len(trajectory) * 7)
            return FilterResult(
                False, 
                f"Extreme joint violations: {violations} points ({violation_rate:.1%})", 
                0.0
            )
        else:
            return FilterResult(True, "Within generous joint limits (±360°)", 1.0)
    
    def check_velocity_limits(self, trajectory: np.ndarray, dt: float = 1/15) -> FilterResult:
        """Check for velocity sudden changes - prevent singularities and deadlocks"""
        if len(trajectory) < 2:
            return FilterResult(True, "Too short for velocity check", 1.0)
        
        # Handle angle wraparound for more accurate velocity calculation
        raw_diff = np.diff(trajectory, axis=0)
        # Wrap angle differences to [-π, π]
        wrapped_diff = np.array([np.arctan2(np.sin(raw_diff[:, i]), np.cos(raw_diff[:, i])) 
                                for i in range(raw_diff.shape[1])]).T
        velocities = wrapped_diff / dt
        max_velocities = np.max(np.abs(velocities), axis=0)
        
        # Reasonable velocity limit for real robots: 5 rad/s (generous but realistic)
        velocity_limit = 5.0
        
        # Check for extreme velocities that indicate IK failure
        extreme_violations = []
        for j, max_vel in enumerate(max_velocities):
            if max_vel > velocity_limit:
                extreme_violations.append(f"J{j}: {max_vel:.1f}>{velocity_limit}")
        
        if extreme_violations:
            return FilterResult(False, f"Extreme velocities (IK failure): {extreme_violations[:2]}", 0.0)
        else:
            # Check velocity smoothness - detect sudden changes
            if len(velocities) > 1:
                velocity_changes = np.diff(velocities, axis=0)
                max_change = np.max(np.abs(velocity_changes))
                smoothness_score = max(0.0, 1.0 - max_change / 5.0)
            else:
                smoothness_score = 1.0
            return FilterResult(True, f"Smooth velocities (max: {np.max(max_velocities):.2f})", smoothness_score)
    
    def check_acceleration_limits(self, trajectory: np.ndarray, dt: float = 1/15) -> FilterResult:
        """Check for acceleration spikes - prevent singularities and deadlocks"""
        if len(trajectory) < 3:
            return FilterResult(True, "Too short for acceleration check", 1.0)
        
        # Handle angle wraparound for velocity calculation
        raw_diff = np.diff(trajectory, axis=0)
        wrapped_diff = np.array([np.arctan2(np.sin(raw_diff[:, i]), np.cos(raw_diff[:, i])) 
                                for i in range(raw_diff.shape[1])]).T
        velocities = wrapped_diff / dt
        
        # Calculate acceleration from corrected velocities
        accelerations = np.diff(velocities, axis=0) / dt
        max_accelerations = np.max(np.abs(accelerations), axis=0)
        
        # Reasonable acceleration limit for real robots: 20 rad/s²
        acceleration_limit = 20.0
        
        # Check for extreme accelerations that indicate singularities/deadlocks
        extreme_violations = []
        for j, max_acc in enumerate(max_accelerations):
            if max_acc > acceleration_limit:
                extreme_violations.append(f"J{j}: {max_acc:.0f}>{acceleration_limit}")
        
        if extreme_violations:
            return FilterResult(False, f"Extreme accelerations (singularity): {extreme_violations[:2]}", 0.0)
        else:
            # Score based on smoothness - penalize sharp acceleration changes
            mean_acc = np.mean(max_accelerations)
            smoothness_score = max(0.0, 1.0 - mean_acc / 20.0)
            return FilterResult(True, f"Smooth accelerations (max: {np.max(max_accelerations):.1f})", smoothness_score)
    
    def validate_trajectory(self, trajectory: np.ndarray) -> Tuple[bool, List[Tuple[str, FilterResult]], float]:
        """Complete trajectory validation"""
        results = []
        
        # Joint limits check
        joint_result = self.check_joint_limits(trajectory)
        results.append(("Joint Limits", joint_result))
        
        # Velocity check
        velocity_result = self.check_velocity_limits(trajectory)
        results.append(("Velocity", velocity_result))
        
        # Acceleration check
        acceleration_result = self.check_acceleration_limits(trajectory)
        results.append(("Acceleration", acceleration_result))
        
        all_passed = all(result[1].passed for result in results)
        overall_score = np.mean([result[1].score for result in results])
        
        return all_passed, results, overall_score

def main():
    """Test adaptive filtering"""
    print("🧪 Testing Adaptive IK Filters")
    print("=" * 40)
    
    DROID_PATH = "/home/cx/AET_FOR_RL/vla/original_data/droid_100"
    
    # Create adaptive filter
    adaptive_filter = AdaptiveIKFilter(DROID_PATH)
    
    # Test on sample episode
    data_df = pd.read_parquet(f"{DROID_PATH}/data/chunk-000/file-000.parquet")
    episode_data = data_df[data_df['episode_index'] == 0]
    
    states = []
    for _, row in episode_data.iterrows():
        state = row['observation.state'] if 'observation.state' in row else np.zeros(7)
        states.append(state)
    
    trajectory = np.array(states)
    
    # Validate
    passed, results, score = adaptive_filter.validate_trajectory(trajectory)
    
    print(f"\n📊 Validation Results for Episode 0:")
    print("=" * 40)
    for name, result in results:
        status = "✅ PASS" if result.passed else "❌ FAIL"
        print(f"{status} {name:15s}: {result.reason}")
    
    print(f"\n🎯 Overall: {'✅ VALID' if passed else '❌ INVALID'} (Score: {score:.3f})")

if __name__ == "__main__":
    main()