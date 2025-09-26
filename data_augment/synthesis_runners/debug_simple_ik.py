#!/usr/bin/env python3
"""Simple debug: test if basic IK works at all"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from ik_solvers.franka_droid_100_ik_solver import FrankaDroid100IKSolver

def test_basic_ik():
    print("🔍 Basic IK Test")
    print("=" * 40)

    ik_solver = FrankaDroid100IKSolver()

    # Test 1: Simple reachable position near robot base
    simple_position = np.array([0.5, 0.2, 0.6])  # 50cm forward, 20cm right, 60cm up
    target_pose = np.eye(4)
    target_pose[:3, 3] = simple_position

    print(f"Target position: {simple_position}")

    # Try IK
    initial_guess = np.array([0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0])
    joint_solution, success, error = ik_solver.inverse_kinematics(
        target_pose, initial_guess, position_only=True
    )

    print(f"IK Success: {success}")
    print(f"IK Error: {error:.4f}m")
    print(f"Solution joints: {joint_solution}")

    # Verify with FK
    if success:
        verify_pose = ik_solver.forward_kinematics(joint_solution)
        actual_position = verify_pose[:3, 3]
        verification_error = np.linalg.norm(actual_position - simple_position)
        print(f"Verification - Actual position: {actual_position}")
        print(f"Verification error: {verification_error:.6f}m")

    print()

    # Test 2: Try one of the DROID positions
    droid_position = np.array([0.79649401, -0.86190904, 0.53714529])
    target_pose[:3, 3] = droid_position

    print(f"DROID target position: {droid_position}")

    joint_solution, success, error = ik_solver.inverse_kinematics(
        target_pose, initial_guess, position_only=True
    )

    print(f"DROID IK Success: {success}")
    print(f"DROID IK Error: {error:.4f}m")

    if success:
        verify_pose = ik_solver.forward_kinematics(joint_solution)
        actual_position = verify_pose[:3, 3]
        verification_error = np.linalg.norm(actual_position - droid_position)
        print(f"DROID Verification - Actual position: {actual_position}")
        print(f"DROID Verification error: {verification_error:.6f}m")

if __name__ == "__main__":
    test_basic_ik()