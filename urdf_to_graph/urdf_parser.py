#!/usr/bin/env python3
"""
URDF to Graph Structure Converter
Based on the mature urdfpy approach for VLA multi-morphology training

Converts URDF files to graph representations suitable for GNN processing.
Supports both DGL and PyTorch Geometric formats.
"""

import numpy as np
import networkx as nx
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import json
import torch

try:
    from urdfpy import URDF
    HAS_URDFPY = True
except ImportError:
    print("⚠️ urdfpy not installed. Install with: pip install urdfpy")
    HAS_URDFPY = False

try:
    import dgl
    HAS_DGL = True
except ImportError:
    HAS_DGL = False

try:
    from torch_geometric.utils import from_networkx
    from torch_geometric.data import Data
    HAS_PYGEOMETRIC = True
except ImportError:
    HAS_PYGEOMETRIC = False


class URDFGraphConverter:
    """Convert URDF to graph structures for GNN training"""

    def __init__(self, skip_fixed_joints: bool = True):
        """
        Args:
            skip_fixed_joints: Whether to skip fixed joints in graph construction
        """
        self.skip_fixed_joints = skip_fixed_joints

    def parse_urdf_to_networkx(self, urdf_path: Union[str, Path]) -> nx.DiGraph:
        """Parse URDF file to NetworkX directed graph"""
        if not HAS_URDFPY:
            raise ImportError("urdfpy required for URDF parsing")

        robot = URDF.load(str(urdf_path))
        G = nx.DiGraph(name=robot.name)

        # Create joint name to index mapping
        joint_id = {}
        joint_idx = 0

        print(f"🤖 Parsing robot: {robot.name}")
        print(f"   Total joints: {len(robot.joints)}")

        # Add joints as nodes
        for joint in robot.joints:
            if self.skip_fixed_joints and joint.joint_type == "fixed":
                continue

            joint_id[joint.name] = joint_idx

            # Extract joint properties
            axis = np.array(joint.axis) if joint.axis is not None else np.zeros(3)

            # Safely extract origin properties
            if joint.origin is not None:
                xyz = np.array(getattr(joint.origin, 'xyz', [0.0, 0.0, 0.0]))
                rpy = np.array(getattr(joint.origin, 'rpy', [0.0, 0.0, 0.0]))
            else:
                xyz = np.zeros(3)
                rpy = np.zeros(3)

            # Extract joint limits
            lower = getattr(joint.limit, "lower", -np.pi) if joint.limit else -np.pi
            upper = getattr(joint.limit, "upper", np.pi) if joint.limit else np.pi
            velocity = getattr(joint.limit, "velocity", 1.0) if joint.limit else 1.0
            effort = getattr(joint.limit, "effort", 100.0) if joint.limit else 100.0

            # Joint type one-hot encoding
            joint_types = ["revolute", "continuous", "prismatic", "planar", "floating", "fixed"]
            type_onehot = [1.0 if joint.joint_type == jt else 0.0 for jt in joint_types]

            # Add node with rich features
            G.add_node(joint_idx,
                      name=joint.name,
                      joint_type=joint.joint_type,
                      type_onehot=np.array(type_onehot),
                      axis=axis,
                      xyz=xyz,
                      rpy=rpy,
                      lower=lower,
                      upper=upper,
                      velocity=velocity,
                      effort=effort,
                      parent_link=joint.parent,
                      child_link=joint.child)

            joint_idx += 1

        # Build parent-child relationships through shared links
        link_parent_joint = {j.child: j.name for j in robot.joints}

        for joint in robot.joints:
            if self.skip_fixed_joints and joint.joint_type == "fixed":
                continue

            parent_link = joint.parent
            if parent_link in link_parent_joint:
                parent_joint_name = link_parent_joint[parent_link]
                if parent_joint_name in joint_id:
                    parent_idx = joint_id[parent_joint_name]
                    child_idx = joint_id[joint.name]

                    # Add edge from parent to child joint
                    G.add_edge(parent_idx, child_idx,
                              via_link=parent_link,
                              parent_joint=parent_joint_name,
                              child_joint=joint.name)

        print(f"   Active joints: {len(joint_id)}")
        print(f"   Graph edges: {G.number_of_edges()}")

        return G

    def networkx_to_dgl(self, G: nx.DiGraph) -> 'dgl.DGLGraph':
        """Convert NetworkX graph to DGL format"""
        if not HAS_DGL:
            raise ImportError("dgl required for DGL conversion")

        # Extract node features
        node_attrs = ["type_onehot", "axis", "xyz", "rpy"]
        scalar_attrs = ["lower", "upper", "velocity", "effort"]

        # Convert to DGL
        dgl_graph = dgl.from_networkx(G, node_attrs=node_attrs + scalar_attrs)

        # Combine features into single tensor
        node_features = []
        for i in range(G.number_of_nodes()):
            features = []
            for attr in node_attrs:
                features.append(G.nodes[i][attr])
            for attr in scalar_attrs:
                features.append([G.nodes[i][attr]])
            node_features.append(np.concatenate(features))

        dgl_graph.ndata['x'] = torch.tensor(np.array(node_features), dtype=torch.float32)

        return dgl_graph

    def networkx_to_pygeometric(self, G: nx.DiGraph) -> 'torch_geometric.data.Data':
        """Convert NetworkX graph to PyTorch Geometric format"""
        if not HAS_PYGEOMETRIC:
            raise ImportError("torch_geometric required for PyG conversion")

        # Prepare node features
        node_features = []
        for i in range(G.number_of_nodes()):
            features = []
            # Type one-hot (6D)
            features.append(G.nodes[i]['type_onehot'])
            # Axis (3D)
            features.append(G.nodes[i]['axis'])
            # Position and orientation (6D)
            features.append(G.nodes[i]['xyz'])
            features.append(G.nodes[i]['rpy'])
            # Limits (4D)
            features.extend([
                [G.nodes[i]['lower']],
                [G.nodes[i]['upper']],
                [G.nodes[i]['velocity']],
                [G.nodes[i]['effort']]
            ])
            node_features.append(np.concatenate(features))

        # Create edge index
        edges = list(G.edges())
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        # Create data object
        data = Data(
            x=torch.tensor(np.array(node_features), dtype=torch.float32),
            edge_index=edge_index,
            num_nodes=G.number_of_nodes()
        )

        return data

    def save_graph_info(self, G: nx.DiGraph, output_path: Union[str, Path]):
        """Save graph structure information to JSON"""
        graph_info = {
            'robot_name': G.graph.get('name', 'unknown'),
            'num_joints': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'joints': [],
            'edges': []
        }

        # Save joint information
        for node_id in G.nodes():
            joint_info = {
                'id': node_id,
                'name': G.nodes[node_id]['name'],
                'type': G.nodes[node_id]['joint_type'],
                'parent_link': G.nodes[node_id]['parent_link'],
                'child_link': G.nodes[node_id]['child_link'],
                'limits': {
                    'lower': float(G.nodes[node_id]['lower']),
                    'upper': float(G.nodes[node_id]['upper']),
                    'velocity': float(G.nodes[node_id]['velocity']),
                    'effort': float(G.nodes[node_id]['effort'])
                }
            }
            graph_info['joints'].append(joint_info)

        # Save edge information
        for edge in G.edges():
            edge_info = {
                'parent': edge[0],
                'child': edge[1],
                'via_link': G.edges[edge]['via_link']
            }
            graph_info['edges'].append(edge_info)

        with open(output_path, 'w') as f:
            json.dump(graph_info, f, indent=2)

        print(f"📊 Graph info saved to: {output_path}")


def convert_maniskill_panda():
    """Convert ManiSkill Panda URDF to graph structure"""
    converter = URDFGraphConverter()

    # ManiSkill Panda URDF path
    urdf_path = "/home/cx/miniconda3/envs/ms3/lib/python3.10/site-packages/mani_skill/assets/robots/panda/panda_v2.urdf"

    if not Path(urdf_path).exists():
        print(f"❌ URDF not found: {urdf_path}")
        return

    try:
        # Parse to NetworkX
        G = converter.parse_urdf_to_networkx(urdf_path)

        # Save graph info
        converter.save_graph_info(G, "panda_graph_info.json")

        # Convert to different formats
        if HAS_DGL:
            dgl_graph = converter.networkx_to_dgl(G)
            print(f"✅ DGL graph created: {dgl_graph}")
            print(f"   Node features shape: {dgl_graph.ndata['x'].shape}")

        if HAS_PYGEOMETRIC:
            pyg_data = converter.networkx_to_pygeometric(G)
            print(f"✅ PyG data created: {pyg_data}")
            print(f"   Node features shape: {pyg_data.x.shape}")
            print(f"   Edge index shape: {pyg_data.edge_index.shape}")

        return G

    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return None


if __name__ == "__main__":
    print("🤖 URDF to Graph Converter")
    print("=" * 50)

    # Test with ManiSkill Panda
    graph = convert_maniskill_panda()

    if graph:
        print("\n🎉 Conversion successful!")
        print(f"   Robot: {graph.graph.get('name', 'unknown')}")
        print(f"   Joints: {graph.number_of_nodes()}")
        print(f"   Connections: {graph.number_of_edges()}")