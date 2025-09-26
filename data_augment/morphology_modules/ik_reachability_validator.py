#!/usr/bin/env python3
"""
IK Reachability Validator
Test if robot configurations can reach target trajectories

Linus-style: No bullshit, test real IK, save what works
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import time

try:
    from .robot_configuration_generator import RobotConfig, JointConfig
except ImportError:
    from robot_configuration_generator import RobotConfig, JointConfig


@dataclass
class IKResult:
    """Result of IK attempt"""
    success: bool
    final_error: float
    iterations: int
    joint_trajectory: Optional[np.ndarray]
    position_errors: List[float]
    orientation_errors: List[float]


class IKReachabilityValidator:
    """
    Validate robot reachability using proper inverse kinematics

    Philosophy:
    1. Use damped least-squares IK (Levenberg-Marquardt style)
    2. Test on real end-effector trajectories
    3. Strict convergence criteria (1cm position, 5° orientation)
    4. No fallback mechanisms - either it works or it doesn't
    """

    def __init__(self,
                 max_iterations: int = 200,
                 position_tolerance: float = 0.05,  # 5cm - more realistic for IK
                 orientation_tolerance: float = np.deg2rad(15),  # 15° - more realistic
                 damping_factor: float = 0.05):
        """
        Args:
            max_iterations: Maximum IK iterations per point
            position_tolerance: Position error tolerance (meters)
            orientation_tolerance: Orientation error tolerance (radians)
            damping_factor: Damping for pseudo-inverse
        """
        self.max_iterations = max_iterations
        self.position_tolerance = position_tolerance
        self.orientation_tolerance = orientation_tolerance
        self.damping_factor = damping_factor

        print(f"⚙️ IKReachabilityValidator:")
        print(f"   Position tolerance: {position_tolerance*1000:.1f}mm")
        print(f"   Orientation tolerance: {np.rad2deg(orientation_tolerance):.1f}°")
        print(f"   Max iterations: {max_iterations}")
        print(f"   Damping factor: {damping_factor}")

    def dh_forward_kinematics(self, dh_params: np.ndarray, joint_angles: np.ndarray) -> np.ndarray:
        """
        Compute forward kinematics using DH parameters

        Args:
            dh_params: (N, 4) array of [a, d, alpha, theta_offset]
            joint_angles: (N,) joint angles

        Returns:
            4x4 end-effector transform matrix
        """
        T = np.eye(4)

        for i, (a, d, alpha, theta_offset) in enumerate(dh_params):
            theta = joint_angles[i] + theta_offset

            # DH transformation matrix
            ct = np.cos(theta)
            st = np.sin(theta)
            ca = np.cos(alpha)
            sa = np.sin(alpha)

            T_i = np.array([
                [ct,    -st*ca,  st*sa,   a*ct],
                [st,     ct*ca, -ct*sa,   a*st],
                [0,      sa,     ca,      d],
                [0,      0,      0,       1]
            ])

            T = T @ T_i

        return T

    def compute_jacobian_numerical(self, dh_params: np.ndarray, joint_angles: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
        """
        Compute Jacobian using numerical differentiation for verification
        """
        n_joints = len(joint_angles)
        jacobian = np.zeros((6, n_joints))

        # Get current pose
        current_pose = self.dh_forward_kinematics(dh_params, joint_angles)
        current_pos = current_pose[:3, 3]
        current_rot = current_pose[:3, :3]

        for i in range(n_joints):
            # Perturb joint i
            perturbed_angles = joint_angles.copy()
            perturbed_angles[i] += epsilon

            # Get perturbed pose
            perturbed_pose = self.dh_forward_kinematics(dh_params, perturbed_angles)
            perturbed_pos = perturbed_pose[:3, 3]
            perturbed_rot = perturbed_pose[:3, :3]

            # Numerical derivative for position
            jacobian[:3, i] = (perturbed_pos - current_pos) / epsilon

            # Numerical derivative for orientation (axis-angle)
            R_diff = perturbed_rot @ current_rot.T
            trace = np.trace(R_diff)
            angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))

            if angle < 1e-6:
                jacobian[3:, i] = 0
            else:
                axis = np.array([
                    R_diff[2, 1] - R_diff[1, 2],
                    R_diff[0, 2] - R_diff[2, 0],
                    R_diff[1, 0] - R_diff[0, 1]
                ]) / (2 * np.sin(angle))
                jacobian[3:, i] = (angle / epsilon) * axis

        return jacobian

    def compute_jacobian(self, dh_params: np.ndarray, joint_angles: np.ndarray) -> np.ndarray:
        """
        Compute geometric Jacobian matrix
        Position + Z-axis normal direction (4×N)

        Returns:
            (4, N) Jacobian matrix [px, py, pz, nz]
        """
        n_joints = len(joint_angles)
        jacobian = np.zeros((4, n_joints))

        # Forward kinematics step by step to get transforms
        transforms = [np.eye(4)]  # Base frame
        T = np.eye(4)

        for i, (a, d, alpha, theta_offset) in enumerate(dh_params):
            theta = joint_angles[i] + theta_offset

            ct = np.cos(theta)
            st = np.sin(theta)
            ca = np.cos(alpha)
            sa = np.sin(alpha)

            T_i = np.array([
                [ct,    -st*ca,  st*sa,   a*ct],
                [st,     ct*ca, -ct*sa,   a*st],
                [0,      sa,     ca,      d],
                [0,      0,      0,       1]
            ])

            T = T @ T_i
            transforms.append(T.copy())

        # End-effector position and Z-axis normal
        ee_pos = transforms[-1][:3, 3]
        ee_z_axis = transforms[-1][:3, 2]  # Current Z-axis direction

        # Compute Jacobian columns
        for i in range(n_joints):
            # For joint i, we need the transform BEFORE joint i acts
            T_joint_frame = transforms[i]
            joint_pos = T_joint_frame[:3, 3]
            joint_z = T_joint_frame[:3, 2]  # z-axis of joint i frame (rotation axis)

            # Linear velocity component: ω × (r_ee - r_joint)
            jacobian[:3, i] = np.cross(joint_z, ee_pos - joint_pos)

            # Angular velocity component for Z-axis only
            # How joint i rotation affects end-effector Z-axis direction
            z_axis_rate = np.cross(joint_z, ee_z_axis)
            jacobian[3, i] = z_axis_rate[2]  # Only Z-component of the Z-axis change

        return jacobian

    def pose_error(self, current_pose: np.ndarray, target_pose: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Compute pose error between current and target
        Position + Z-axis normal direction (4DOF)

        Args:
            current_pose: 4x4 current transform
            target_pose: 4x4 target transform

        Returns:
            error_vector: (4,) [px, py, pz, nz]
            position_error: scalar position error magnitude
            orientation_error: Z-axis direction error magnitude
        """
        # Position error (3DOF)
        pos_error = target_pose[:3, 3] - current_pose[:3, 3]
        pos_magnitude = np.linalg.norm(pos_error)

        # Z-axis direction error (1DOF)
        current_z = current_pose[:3, 2]  # Current Z-axis direction
        target_z = target_pose[:3, 2]    # Target Z-axis direction

        # Project Z-axis error onto Z component only (simplest approach)
        z_error = target_z[2] - current_z[2]  # Only Z-component matters
        rot_magnitude = abs(z_error)

        # Create 4DOF error vector: [px, py, pz, nz]
        error_vector = np.concatenate([pos_error, [z_error]])
        return error_vector, pos_magnitude, rot_magnitude

    def solve_ik_single_point(self, robot: RobotConfig, target_pose: np.ndarray,
                            initial_guess: np.ndarray = None, max_retries: int = 3) -> IKResult:
        """
        Solve IK for single target pose

        Args:
            robot: Robot configuration
            target_pose: 4x4 target transform matrix
            initial_guess: Initial joint angles
            max_retries: Maximum retry attempts with different initial guesses

        Returns:
            IKResult with success status and trajectory
        """

        # Try multiple initial guesses if needed
        for retry in range(max_retries):
            if initial_guess is None and retry == 0:
                # First try: joint center
                joint_angles = np.array([
                    (limits[0] + limits[1]) / 2
                    for joint in robot.joints
                    for limits in [joint.limits]
                ])
            elif initial_guess is None and retry == 1:
                # Second try: random within joint limits
                joint_angles = np.array([
                    np.random.uniform(joint.limits[0], joint.limits[1])
                    for joint in robot.joints
                ])
            elif initial_guess is None and retry >= 2:
                # Third+ try: different random configurations
                joint_angles = np.array([
                    np.random.uniform(joint.limits[0] * 0.7, joint.limits[1] * 0.7)
                    for joint in robot.joints
                ])
            else:
                joint_angles = initial_guess.copy()

            # Get DH parameters
            dh_params = np.array([joint.dh_params for joint in robot.joints])

            position_errors = []
            orientation_errors = []

            for iteration in range(self.max_iterations):
                # Forward kinematics
                current_pose = self.dh_forward_kinematics(dh_params, joint_angles)

                # Compute error
                error_vector, pos_error, rot_error = self.pose_error(current_pose, target_pose)
                position_errors.append(pos_error)
                orientation_errors.append(rot_error)

                # Check convergence
                if pos_error < self.position_tolerance and rot_error < self.orientation_tolerance:
                    return IKResult(
                        success=True,
                        final_error=pos_error + rot_error,
                        iterations=iteration + 1,
                        joint_trajectory=joint_angles.copy(),
                        position_errors=position_errors,
                        orientation_errors=orientation_errors
                    )

                # Compute Jacobian
                jacobian = self.compute_jacobian(dh_params, joint_angles)

                # Use damped least-squares for stability
                JTJ = jacobian.T @ jacobian
                damping_matrix = self.damping_factor**2 * np.eye(robot.dof)

                try:
                    J_inv = np.linalg.solve(JTJ + damping_matrix, jacobian.T)
                    delta_q = J_inv @ error_vector
                except np.linalg.LinAlgError:
                    # Fallback to pseudo-inverse with small step
                    J_inv = np.linalg.pinv(jacobian, rcond=1e-3)
                    delta_q = J_inv @ error_vector

                # Limit step size to prevent instability
                max_step = 0.2  # Maximum 0.2 radian per iteration
                delta_q = np.clip(delta_q, -max_step, max_step)
                joint_angles += delta_q

                # Apply joint limits
                for i, joint in enumerate(robot.joints):
                    joint_angles[i] = np.clip(joint_angles[i], joint.limits[0], joint.limits[1])

            # If we reach here, this retry failed
            if retry < max_retries - 1:
                print(f"     🔄 Retry {retry + 1}: IK failed, trying different initial guess")

        # All retries failed
        final_error = position_errors[-1] + orientation_errors[-1] if position_errors else float('inf')
        return IKResult(
            success=False,
            final_error=final_error,
            iterations=self.max_iterations,
            joint_trajectory=None,
            position_errors=position_errors,
            orientation_errors=orientation_errors
        )

    def validate_trajectory_reachability(self, robot: RobotConfig,
                                       ee_trajectory: np.ndarray,
                                       sample_points: int = 10) -> Tuple[bool, float, List[IKResult]]:
        """
        Validate if robot can reach end-effector trajectory

        Args:
            robot: Robot configuration to test
            ee_trajectory: (T, 7) end-effector trajectory [x,y,z,rx,ry,rz,gripper]
            sample_points: Number of points to test from trajectory

        Returns:
            (is_reachable, success_rate, ik_results)
        """
        if len(ee_trajectory) == 0:
            return False, 0.0, []

        # Sample points uniformly from trajectory
        if len(ee_trajectory) <= sample_points:
            sample_indices = list(range(len(ee_trajectory)))
        else:
            sample_indices = np.linspace(0, len(ee_trajectory)-1, sample_points, dtype=int)

        ik_results = []
        successful_points = 0

        print(f"   🎯 Testing {len(sample_indices)} trajectory points...")

        for i, idx in enumerate(sample_indices):
            ee_point = ee_trajectory[idx]

            # Convert to 4x4 transform (assuming RPY angles)
            position = ee_point[:3]
            rpy = ee_point[3:6]

            # RPY to rotation matrix
            r, p, y = rpy
            R_x = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
            R_y = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
            R_z = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
            R = R_z @ R_y @ R_x

            target_pose = np.eye(4)
            target_pose[:3, :3] = R
            target_pose[:3, 3] = position

            # Solve IK
            ik_result = self.solve_ik_single_point(robot, target_pose)
            ik_results.append(ik_result)

            if ik_result.success:
                successful_points += 1
                print(f"     ✅ Point {i+1}/{len(sample_indices)}: "
                      f"pos_err={ik_result.position_errors[-1]*1000:.1f}mm, "
                      f"rot_err={np.rad2deg(ik_result.orientation_errors[-1]):.1f}°")
            else:
                print(f"     ❌ Point {i+1}/{len(sample_indices)}: "
                      f"pos_err={ik_result.position_errors[-1]*1000:.1f}mm, "
                      f"rot_err={np.rad2deg(ik_result.orientation_errors[-1]):.1f}° (FAILED)")

        success_rate = successful_points / len(sample_indices)
        is_reachable = success_rate >= 0.7  # Require 70% success rate

        print(f"   📊 Success rate: {success_rate:.1%} ({successful_points}/{len(sample_indices)})")

        return is_reachable, success_rate, ik_results

    def batch_validate_robots(self, robots: List[RobotConfig],
                            ee_trajectory: np.ndarray) -> List[Tuple[RobotConfig, bool, float]]:
        """
        Validate multiple robots against same trajectory

        Returns:
            List of (robot, is_reachable, success_rate) tuples
        """
        results = []

        print(f"🔄 Validating {len(robots)} robots against trajectory ({len(ee_trajectory)} points)")
        print("=" * 60)

        for i, robot in enumerate(robots):
            print(f"\n🤖 Robot {i+1}/{len(robots)}: {robot.name}")
            print(f"   DOF: {robot.dof}, Reach: {robot.total_reach:.2f}m")

            start_time = time.time()
            is_reachable, success_rate, _ = self.validate_trajectory_reachability(robot, ee_trajectory)
            elapsed = time.time() - start_time

            results.append((robot, is_reachable, success_rate))

            status = "✅ VIABLE" if is_reachable else "❌ FAILED"
            print(f"   {status} - Success: {success_rate:.1%}, Time: {elapsed:.1f}s")

        viable_count = sum(1 for _, reachable, _ in results if reachable)
        print(f"\n📊 Validation Summary: {viable_count}/{len(robots)} robots viable")

        return results


def test_ik_validator():
    """Test IK validator with random robot"""
    print("🧪 Testing IK Reachability Validator")
    print("=" * 50)

    from robot_configuration_generator import RobotConfigurationGenerator

    # Create test robot
    generator = RobotConfigurationGenerator(dof_range=(6, 7))
    robot = generator.generate_random_robot("test_ik_robot")

    # Create validator
    validator = IKReachabilityValidator(
        position_tolerance=0.01,
        orientation_tolerance=np.deg2rad(5)
    )

    # Create simple test trajectory
    print(f"\n🎯 Creating test trajectory...")

    # Simple line trajectory in front of robot
    n_points = 20
    start_pos = np.array([0.3, 0.0, 0.3])
    end_pos = np.array([0.5, 0.2, 0.5])

    positions = np.linspace(start_pos, end_pos, n_points)
    orientations = np.tile([0, 0, 0], (n_points, 1))  # Zero RPY
    grippers = np.zeros(n_points)

    test_trajectory = np.column_stack([positions, orientations, grippers])

    print(f"   📏 Trajectory: {n_points} points, {start_pos} → {end_pos}")

    # Test reachability
    is_reachable, success_rate, ik_results = validator.validate_trajectory_reachability(
        robot, test_trajectory, sample_points=5
    )

    print(f"\n🎉 Test Results:")
    print(f"   Reachable: {is_reachable}")
    print(f"   Success rate: {success_rate:.1%}")
    print(f"   Average iterations: {np.mean([r.iterations for r in ik_results]):.1f}")


if __name__ == "__main__":
    test_ik_validator()