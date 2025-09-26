#!/usr/bin/env python3
"""
Robot Graph Representation Module
Convert robot configurations to graph format for GNN processing

Linus-style: Start simple, make it work, then optimize
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json

from robot_configuration_generator import RobotConfig, JointConfig


@dataclass
class NodeFeatures:
    """Node features for a single joint/link"""
    joint_index: int
    joint_type: str  # 'revolute', 'prismatic', 'base', 'end_effector'

    # Angle features (for revolute joints)
    sin_theta: float
    cos_theta: float
    theta_normalized: float  # current angle normalized to [-1, 1]

    # Joint constraints
    limit_min_norm: float  # normalized to [-1, 1]
    limit_max_norm: float  # normalized to [-1, 1]

    # Physical properties
    axis_x: float
    axis_y: float
    axis_z: float
    link_length: float

    # Graph structure
    depth_to_ee: int  # distance to end-effector in graph


@dataclass
class EdgeFeatures:
    """Edge features between joints"""
    from_joint: int
    to_joint: int
    is_parent_child: bool  # direct kinematic chain connection
    relative_transform: np.ndarray  # 4x4 SE(3) transform


class RobotGraphModule:
    """
    Convert robot configurations to graph representation

    Philosophy:
    1. Keep it simple - basic features first
    2. Each joint = one node
    3. Kinematic chain = edges
    4. Normalize everything to [-1, 1] range
    """

    def __init__(self):
        print(f"🧠 RobotGraphModule: Starting simple")

        # Feature normalization ranges
        self.angle_range = (-np.pi, np.pi)
        self.limit_range = (-3.5, 3.5)  # Most joint limits are within ±200°
        self.length_range = (0.05, 0.5)  # Link lengths 5cm to 50cm

    def normalize_value(self, value: float, value_range: Tuple[float, float]) -> float:
        """Normalize value to [-1, 1] range"""
        min_val, max_val = value_range
        normalized = 2.0 * (value - min_val) / (max_val - min_val) - 1.0
        return np.clip(normalized, -1.0, 1.0)

    def create_node_features(self, robot: RobotConfig, joint_angles: np.ndarray = None) -> List[NodeFeatures]:
        """Create node features for all joints in robot"""

        if joint_angles is None:
            # Use zero configuration if no angles provided
            joint_angles = np.zeros(robot.dof)

        if len(joint_angles) != robot.dof:
            raise ValueError(f"Expected {robot.dof} joint angles, got {len(joint_angles)}")

        nodes = []

        # Base node (always first)
        base_node = NodeFeatures(
            joint_index=0,
            joint_type='base',
            sin_theta=0.0,
            cos_theta=1.0,
            theta_normalized=0.0,
            limit_min_norm=0.0,
            limit_max_norm=0.0,
            axis_x=0.0, axis_y=0.0, axis_z=1.0,  # Base Z-up
            link_length=self.normalize_value(robot.base_height, self.length_range),
            depth_to_ee=robot.dof + 1
        )
        nodes.append(base_node)

        # Joint nodes
        for i, joint in enumerate(robot.joints):
            angle = joint_angles[i]

            # Angle features
            sin_theta = np.sin(angle)
            cos_theta = np.cos(angle)
            theta_norm = self.normalize_value(angle, self.angle_range)

            # Limit features
            limit_min_norm = self.normalize_value(joint.limits[0], self.limit_range)
            limit_max_norm = self.normalize_value(joint.limits[1], self.limit_range)

            # Physical features
            link_len_norm = self.normalize_value(joint.link_length, self.length_range)

            # Joint type encoding
            joint_type = joint.joint_type

            node = NodeFeatures(
                joint_index=i + 1,  # +1 because base is index 0
                joint_type=joint_type,
                sin_theta=sin_theta,
                cos_theta=cos_theta,
                theta_normalized=theta_norm,
                limit_min_norm=limit_min_norm,
                limit_max_norm=limit_max_norm,
                axis_x=joint.axis[0],
                axis_y=joint.axis[1],
                axis_z=joint.axis[2],
                link_length=link_len_norm,
                depth_to_ee=robot.dof - i
            )
            nodes.append(node)

        # End-effector node
        ee_node = NodeFeatures(
            joint_index=robot.dof + 1,
            joint_type='end_effector',
            sin_theta=0.0,
            cos_theta=1.0,
            theta_normalized=0.0,
            limit_min_norm=0.0,
            limit_max_norm=0.0,
            axis_x=0.0, axis_y=0.0, axis_z=1.0,
            link_length=0.0,  # No length for EE
            depth_to_ee=0
        )
        nodes.append(ee_node)

        return nodes

    def create_edge_features(self, robot: RobotConfig) -> List[EdgeFeatures]:
        """Create edge features for kinematic chain"""
        edges = []

        # Sequential kinematic chain connections
        # Base -> Joint1 -> Joint2 -> ... -> JointN -> EE
        for i in range(robot.dof + 1):  # +1 to include EE connection
            edge = EdgeFeatures(
                from_joint=i,
                to_joint=i + 1,
                is_parent_child=True,
                relative_transform=np.eye(4)  # Simplified for now
            )
            edges.append(edge)

        return edges

    def robot_to_graph_dict(self, robot: RobotConfig, joint_angles: np.ndarray = None) -> Dict:
        """Convert robot to graph dictionary format"""

        nodes = self.create_node_features(robot, joint_angles)
        edges = self.create_edge_features(robot)

        # Convert to arrays for GNN processing
        node_features = []
        for node in nodes:
            # Pack all features into single vector
            features = [
                node.sin_theta, node.cos_theta, node.theta_normalized,
                node.limit_min_norm, node.limit_max_norm,
                node.axis_x, node.axis_y, node.axis_z,
                node.link_length,
                float(node.depth_to_ee)
            ]
            node_features.append(features)

        # Edge indices for GNN
        edge_indices = []
        for edge in edges:
            edge_indices.append([edge.from_joint, edge.to_joint])

        # Joint type encoding
        type_encoding = {
            'base': 0,
            'revolute': 1,
            'prismatic': 2,
            'end_effector': 3
        }

        node_types = [type_encoding[node.joint_type] for node in nodes]

        graph_dict = {
            'robot_name': robot.name,
            'dof': robot.dof,
            'num_nodes': len(nodes),
            'node_features': np.array(node_features, dtype=np.float32),
            'node_types': np.array(node_types, dtype=np.int32),
            'edge_indices': np.array(edge_indices, dtype=np.int32),
            'total_reach': robot.total_reach,
            'base_height': robot.base_height
        }

        return graph_dict

    def save_robot_graph(self, robot: RobotConfig, output_path: str, joint_angles: np.ndarray = None):
        """Save robot graph to file"""
        graph_dict = self.robot_to_graph_dict(robot, joint_angles)

        # Convert numpy arrays to lists for JSON serialization
        serializable_dict = {}
        for key, value in graph_dict.items():
            if isinstance(value, np.ndarray):
                serializable_dict[key] = value.tolist()
            else:
                serializable_dict[key] = value

        with open(output_path, 'w') as f:
            json.dump(serializable_dict, f, indent=2)

        print(f"💾 Saved robot graph: {output_path}")

    def load_robot_graph(self, input_path: str) -> Dict:
        """Load robot graph from file"""
        with open(input_path, 'r') as f:
            graph_dict = json.load(f)

        # Convert lists back to numpy arrays
        graph_dict['node_features'] = np.array(graph_dict['node_features'], dtype=np.float32)
        graph_dict['node_types'] = np.array(graph_dict['node_types'], dtype=np.int32)
        graph_dict['edge_indices'] = np.array(graph_dict['edge_indices'], dtype=np.int32)

        return graph_dict

    def print_graph_summary(self, graph_dict: Dict):
        """Print summary of robot graph"""
        print(f"\n🤖 Robot Graph: {graph_dict['robot_name']}")
        print(f"   DOF: {graph_dict['dof']}")
        print(f"   Nodes: {graph_dict['num_nodes']}")
        print(f"   Edges: {len(graph_dict['edge_indices'])}")
        print(f"   Feature dim: {graph_dict['node_features'].shape[1]}")
        print(f"   Total reach: {graph_dict['total_reach']:.3f}m")

        print(f"   Node types: {np.bincount(graph_dict['node_types'])}")
        print(f"   Feature range: [{graph_dict['node_features'].min():.3f}, {graph_dict['node_features'].max():.3f}]")


def test_robot_graph_module():
    """Test the robot graph module"""
    print("🧪 Testing Robot Graph Module")
    print("=" * 50)

    # Import robot generator
    from robot_configuration_generator import RobotConfigurationGenerator

    # Create generator and robot
    generator = RobotConfigurationGenerator(
        dof_range=(5, 6),
        link_length_range=(0.2, 0.3)
    )

    robot = generator.generate_random_robot("test_robot")

    # Create graph module
    graph_module = RobotGraphModule()

    # Test with zero configuration
    print(f"\n🔄 Testing with zero configuration...")
    graph_dict = graph_module.robot_to_graph_dict(robot)
    graph_module.print_graph_summary(graph_dict)

    # Test with random configuration
    print(f"\n🔄 Testing with random joint angles...")
    random_angles = np.random.uniform(-1.0, 1.0, robot.dof)
    graph_dict_random = graph_module.robot_to_graph_dict(robot, random_angles)
    graph_module.print_graph_summary(graph_dict_random)

    # Test save/load
    print(f"\n🔄 Testing save/load...")
    test_path = "/tmp/test_robot_graph.json"
    graph_module.save_robot_graph(robot, test_path, random_angles)
    loaded_graph = graph_module.load_robot_graph(test_path)

    print(f"   ✅ Original shape: {graph_dict_random['node_features'].shape}")
    print(f"   ✅ Loaded shape: {loaded_graph['node_features'].shape}")
    print(f"   ✅ Data matches: {np.allclose(graph_dict_random['node_features'], loaded_graph['node_features'])}")

    print(f"\n🎉 Robot Graph Module test completed!")


if __name__ == "__main__":
    test_robot_graph_module()