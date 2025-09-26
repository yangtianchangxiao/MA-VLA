#!/usr/bin/env python3
"""Debug DH parameters and forward kinematics"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from ik_solvers.franka_droid_100_ik_solver import FrankaDroid100IKSolver

def debug_dh_and_fk():
    print("🔍 Debug DH Parameters & Forward Kinematics")
    print("=" * 50)

    ik_solver = FrankaDroid100IKSolver()

    print("DH Parameters:")
    print(ik_solver.standard_dh)
    print()

    print("Joint Limits:")
    for i, (min_limit, max_limit) in enumerate(ik_solver.joint_limits):
        print(f"  Joint {i}: [{min_limit:.3f}, {max_limit:.3f}]")
    print()

    # Test FK with neutral pose
    neutral_joints = np.array([0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0])
    print(f"Testing FK with neutral pose: {neutral_joints}")

    fk_result = ik_solver.forward_kinematics(neutral_joints)
    end_effector_pos = fk_result[:3, 3]
    print(f"End-effector position: {end_effector_pos}")
    print(f"Distance from origin: {np.linalg.norm(end_effector_pos):.3f}m")
    print()

    # Test FK with zero pose
    zero_joints = np.zeros(7)
    print(f"Testing FK with zero pose: {zero_joints}")

    fk_result = ik_solver.forward_kinematics(zero_joints)
    end_effector_pos = fk_result[:3, 3]
    print(f"End-effector position: {end_effector_pos}")
    print(f"Distance from origin: {np.linalg.norm(end_effector_pos):.3f}m")
    print()

    # Test a few more poses
    test_poses = [
        np.array([0.1, -0.5, 0.2, -1.5, 0.3, 2.0, 0.1]),
        np.array([0.5, -1.0, 0.5, -2.0, 1.0, 2.5, 0.5])
    ]

    for i, joints in enumerate(test_poses):
        print(f"Test pose {i+1}: {joints}")
        fk_result = ik_solver.forward_kinematics(joints)
        end_effector_pos = fk_result[:3, 3]
        print(f"  End-effector: {end_effector_pos}")
        print(f"  Distance: {np.linalg.norm(end_effector_pos):.3f}m")
        print()

if __name__ == "__main__":
    debug_dh_and_fk()