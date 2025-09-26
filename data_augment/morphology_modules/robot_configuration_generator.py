#!/usr/bin/env python3
"""
Robot Configuration Generator
Random robot morphology generation for data augmentation

Linus-style: Generate diverse robot configs, test IK, keep what works
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import random


@dataclass
class JointConfig:
    """Single joint configuration"""
    joint_type: str  # 'revolute', 'prismatic'
    axis: np.ndarray  # 3D axis vector [x, y, z]
    limits: Tuple[float, float]  # (min, max)
    link_length: float  # length to next joint
    dh_params: np.ndarray  # [a, d, alpha, theta_offset]


@dataclass
class RobotConfig:
    """Complete robot configuration"""
    name: str
    joints: List[JointConfig]
    dof: int
    base_height: float
    total_reach: float


class RobotConfigurationGenerator:
    """
    Generate random robot configurations for morphology synthesis

    Philosophy:
    1. Start simple - basic kinematic chains
    2. Generate diverse but realistic configurations
    3. Validate with IK reachability
    4. Keep successful configs for training
    """

    def __init__(self,
                 dof_range: Tuple[int, int] = (5, 8),
                 link_length_range: Tuple[float, float] = (0.1, 0.4),
                 base_height_range: Tuple[float, float] = (0.2, 0.5)):
        """
        Args:
            dof_range: (min_dof, max_dof) for random robots
            link_length_range: (min_length, max_length) for links in meters
            base_height_range: (min_height, max_height) for base in meters
        """
        self.dof_range = dof_range
        self.link_length_range = link_length_range
        self.base_height_range = base_height_range

        # Common joint configurations for realistic robots
        self.common_axes = [
            np.array([0, 0, 1]),   # Z-axis (most common)
            np.array([1, 0, 0]),   # X-axis
            np.array([0, 1, 0]),   # Y-axis
        ]

        # Realistic joint limits (radians)
        self.joint_limits_options = [
            (-3.14, 3.14),    # Full rotation
            (-2.97, 2.97),    # Almost full (like Franka)
            (-1.57, 1.57),    # ±90 degrees
            (-2.09, 2.09),    # ±120 degrees
        ]

        print(f"🤖 RobotConfigurationGenerator:")
        print(f"   DOF range: {dof_range}")
        print(f"   Link length: {link_length_range}")
        print(f"   Base height: {base_height_range}")

    def generate_franka_like_joint(self, joint_index: int, dof: int) -> JointConfig:
        """Generate Franka-like joint configuration following real DH pattern"""

        joint_type = 'revolute'
        axis = np.array([0, 0, 1])  # All Z-axis like Franka

        # Random limits similar to Franka
        limits = random.choice(self.joint_limits_options)

        # Follow real Franka DH pattern: [0, -π/2, π/2, π/2, -π/2, π/2]
        # Scale all dimensions by a random factor for variety
        scale_factor = random.uniform(0.7, 1.3)

        if joint_index == 0:
            # Joint 1: Base rotation (like real Franka joint 1)
            a = 0
            d = random.uniform(0.25, 0.4) * scale_factor  # Base height
            alpha = 0  # Key: alpha=0 for base
            theta_offset = 0
            link_length = d

        elif joint_index == 1:
            # Joint 2: Shoulder (like real Franka joint 2)
            a = 0
            d = 0  # No Z-offset
            alpha = -np.pi/2  # Key: alpha=-π/2
            theta_offset = 0
            link_length = 0.05  # Small link

        elif joint_index == 2:
            # Joint 3: Upper arm (like real Franka joint 3)
            a = 0
            d = random.uniform(0.25, 0.4) * scale_factor  # Upper arm length
            alpha = np.pi/2  # Key: alpha=π/2
            theta_offset = 0
            link_length = d

        elif joint_index == 3:
            # Joint 4: Elbow (like real Franka joint 4)
            a = random.uniform(0.06, 0.12) * scale_factor  # Small X-offset
            d = 0
            alpha = np.pi/2  # Key: alpha=π/2
            theta_offset = 0
            link_length = a

        elif joint_index == 4:
            # Joint 5: Forearm (like real Franka joint 5)
            a = -random.uniform(0.06, 0.12) * scale_factor  # Negative X-offset
            d = random.uniform(0.25, 0.4) * scale_factor  # Forearm length
            alpha = -np.pi/2  # Key: alpha=-π/2
            theta_offset = 0
            link_length = d

        elif joint_index == dof - 1:
            # LAST JOINT: Always axial rotation for end-effector orientation control
            # This applies to both 6DOF (joint 5) and 7DOF (joint 6) robots
            a = random.uniform(0.08, 0.12) * scale_factor  # Small radial offset (like real Franka)
            d = random.uniform(0.08, 0.15) * scale_factor   # End-effector length
            alpha = 0  # Pure Z-axis rotation
            theta_offset = 0
            link_length = max(a, d)

        elif joint_index == dof - 2 and dof >= 7:
            # SECOND-TO-LAST JOINT: Wrist setup (only for 7DOF robots)
            # Joint 6 in 7DOF robots: sets up coordinate frame for final axial rotation
            a = 0
            d = 0
            alpha = np.pi/2  # Sets up coordinate frame for axial rotation
            theta_offset = 0
            link_length = 0.05  # Small wrist

        else:
            # MIDDLE JOINTS: Standard Franka-like joints
            # This should not happen with current DOF range (5-7), but just in case
            a = random.uniform(0.06, 0.12) * scale_factor
            d = random.uniform(0.15, 0.3) * scale_factor
            alpha = random.choice([np.pi/2, -np.pi/2])
            theta_offset = 0
            link_length = max(a, d)

        dh_params = np.array([a, d, alpha, theta_offset])

        return JointConfig(
            joint_type=joint_type,
            axis=axis,
            limits=limits,
            link_length=link_length,
            dh_params=dh_params
        )

    def generate_random_robot(self, robot_id: Optional[str] = None) -> RobotConfig:
        """Generate a complete random robot configuration"""

        # Random DOF
        dof = random.randint(*self.dof_range)

        # Random base height
        base_height = random.uniform(*self.base_height_range)

        # Generate Franka-like joints
        joints = []
        total_reach = 0

        for i in range(dof):
            joint = self.generate_franka_like_joint(i, dof)
            joints.append(joint)
            total_reach += joint.link_length

        # Robot name
        if robot_id is None:
            robot_id = f"random_robot_{dof}dof_{random.randint(1000, 9999)}"

        return RobotConfig(
            name=robot_id,
            joints=joints,
            dof=dof,
            base_height=base_height,
            total_reach=total_reach
        )

    def robot_to_dh_params(self, robot: RobotConfig) -> np.ndarray:
        """Convert robot config to DH parameter matrix"""
        dh_matrix = np.zeros((robot.dof, 4))

        for i, joint in enumerate(robot.joints):
            dh_matrix[i] = joint.dh_params

        return dh_matrix

    def robot_to_joint_limits(self, robot: RobotConfig) -> List[Tuple[float, float]]:
        """Convert robot config to joint limits list"""
        return [joint.limits for joint in robot.joints]

    def generate_robot_batch(self, batch_size: int = 10) -> List[RobotConfig]:
        """Generate a batch of random robots"""
        robots = []

        print(f"🔄 Generating {batch_size} random robots...")

        for i in range(batch_size):
            robot = self.generate_random_robot(f"batch_robot_{i:03d}")
            robots.append(robot)

            print(f"   ✅ {robot.name}: {robot.dof}DOF, reach={robot.total_reach:.2f}m")

        print(f"🎉 Generated {len(robots)} random robots")
        return robots

    def validate_robot_workspace(self, robot: RobotConfig,
                                target_positions: np.ndarray,
                                validation_samples: int = 100) -> Tuple[bool, float]:
        """
        Validate if robot can reach target positions

        Args:
            robot: Robot configuration to test
            target_positions: (N, 3) array of target positions to reach
            validation_samples: Number of random joint configs to test

        Returns:
            (is_valid, reachability_score)
        """
        # Simple reachability test based on workspace geometry

        # Maximum possible reach
        max_reach = robot.total_reach
        min_reach = max(0.1, robot.total_reach * 0.2)  # Minimum 20% of total reach

        # Check if target positions are within possible reach
        reachable_count = 0

        for pos in target_positions:
            distance_from_base = np.linalg.norm(pos[:2])  # XY distance from base
            height_from_base = abs(pos[2] - robot.base_height)

            total_distance = np.sqrt(distance_from_base**2 + height_from_base**2)

            # Simple reachability criterion
            if min_reach <= total_distance <= max_reach:
                reachable_count += 1

        reachability_score = reachable_count / len(target_positions)
        is_valid = reachability_score > 0.7  # At least 70% reachable

        return is_valid, reachability_score

    def filter_viable_robots(self, robots: List[RobotConfig],
                           target_trajectory: np.ndarray) -> List[Tuple[RobotConfig, float]]:
        """
        Filter robots that can potentially reach the target trajectory

        Args:
            robots: List of robot configurations to test
            target_trajectory: (T, 6) array of target end-effector poses

        Returns:
            List of (robot, reachability_score) for viable robots
        """
        viable_robots = []
        target_positions = target_trajectory[:, :3]  # Extract positions

        print(f"🔍 Filtering {len(robots)} robots for viability...")

        for robot in robots:
            is_valid, score = self.validate_robot_workspace(robot, target_positions)

            if is_valid:
                viable_robots.append((robot, score))
                print(f"   ✅ {robot.name}: reachability={score:.1%}")
            else:
                print(f"   ❌ {robot.name}: reachability={score:.1%} (too low)")

        print(f"📊 Viable robots: {len(viable_robots)}/{len(robots)}")
        return viable_robots


def test_robot_generation():
    """Test robot configuration generation"""

    print("🧪 Testing Robot Configuration Generator")
    print("=" * 50)

    # Create generator
    generator = RobotConfigurationGenerator(
        dof_range=(5, 7),
        link_length_range=(0.15, 0.35),
        base_height_range=(0.3, 0.4)
    )

    # Generate test robots
    robots = generator.generate_robot_batch(5)

    # Show detailed configs
    print()
    print("🔍 Detailed robot configurations:")
    for robot in robots[:2]:  # Show first 2 in detail
        print(f"\n{robot.name}:")
        print(f"   DOF: {robot.dof}")
        print(f"   Base height: {robot.base_height:.3f}m")
        print(f"   Total reach: {robot.total_reach:.3f}m")

        dh_matrix = generator.robot_to_dh_params(robot)
        limits = generator.robot_to_joint_limits(robot)

        print("   DH Parameters:")
        for i, (dh_row, joint_limits) in enumerate(zip(dh_matrix, limits)):
            print(f"     Joint {i+1}: a={dh_row[0]:.3f}, d={dh_row[1]:.3f}, "
                  f"alpha={dh_row[2]:.3f}, limits={joint_limits}")

    # Test workspace validation with dummy trajectory
    print()
    print("🔍 Testing workspace validation:")
    dummy_trajectory = np.random.uniform([0.2, -0.3, 0.2], [0.6, 0.3, 0.8], (20, 6))

    viable_robots = generator.filter_viable_robots(robots, dummy_trajectory)

    print(f"\n🎉 Test completed! {len(viable_robots)} viable robots found")


if __name__ == "__main__":
    test_robot_generation()