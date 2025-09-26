#!/usr/bin/env python3
"""
Generate graph representations for all ManiSkill robots
Save them for use in VLA evaluation
"""

import os
import json
import torch
from pathlib import Path
from urdf_parser import URDFGraphConverter

def get_maniskill_robot_urdfs():
    """Get all robot URDF files from ManiSkill"""
    base_path = "/home/cx/miniconda3/envs/ms3/lib/python3.10/site-packages/mani_skill/assets/robots"

    # Priority robot arms for VLA evaluation
    priority_robots = [
        # Franka Panda variants
        "panda/panda_v2.urdf",
        "panda/panda_v3.urdf",
        "panda/panda_v2_gripper.urdf",
        "panda/mobile_panda_single_arm.urdf",
        "panda/mobile_panda_dual_arm.urdf",

        # XARM7
        "xarm7/xarm7.urdf",
        "xarm7/xarm7_ability_right_hand.urdf",
    ]

    # Find all robot URDFs
    all_urdfs = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.urdf'):
                rel_path = os.path.relpath(os.path.join(root, file), base_path)
                all_urdfs.append(rel_path)

    # Sort: priority first, then others
    priority_paths = [os.path.join(base_path, p) for p in priority_robots if os.path.exists(os.path.join(base_path, p))]
    other_paths = [os.path.join(base_path, p) for p in all_urdfs if os.path.join(base_path, p) not in priority_paths]

    return priority_paths + other_paths

def generate_robot_graph(urdf_path, output_dir):
    """Generate graph for single robot"""
    converter = URDFGraphConverter(skip_fixed_joints=True)

    try:
        # Parse URDF to graph
        G = converter.parse_urdf_to_networkx(urdf_path)

        # Get robot name from URDF path
        robot_name = Path(urdf_path).stem
        robot_folder = Path(urdf_path).parent.name
        full_name = f"{robot_folder}_{robot_name}"

        # Create output directory
        robot_output_dir = output_dir / full_name
        robot_output_dir.mkdir(exist_ok=True)

        # Save graph info
        info_path = robot_output_dir / "graph_info.json"
        converter.save_graph_info(G, info_path)

        # Convert to PyTorch Geometric and save
        try:
            pyg_data = converter.networkx_to_pygeometric(G)
            torch.save(pyg_data, robot_output_dir / "graph_pyg.pt")
            print(f"✅ {full_name}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, 19D features")
            return True, G.number_of_nodes(), G.number_of_edges()
        except Exception as e:
            print(f"⚠️ {full_name}: Graph created but PyG conversion failed: {e}")
            return True, G.number_of_nodes(), G.number_of_edges()

    except Exception as e:
        print(f"❌ {Path(urdf_path).stem}: Failed to parse URDF: {e}")
        return False, 0, 0

def main():
    """Generate graphs for all ManiSkill robots"""
    print("🤖 Generating graphs for all ManiSkill robots")
    print("=" * 60)

    # Setup output directory
    output_dir = Path("/home/cx/AET_FOR_RL/vla/urdf_to_graph/robot_graphs")
    output_dir.mkdir(exist_ok=True)

    # Get all robot URDFs
    robot_urdfs = get_maniskill_robot_urdfs()
    print(f"📁 Found {len(robot_urdfs)} robot URDF files")

    # Generate graphs
    results = []
    success_count = 0

    for i, urdf_path in enumerate(robot_urdfs[:15]):  # Limit to first 15 for now
        print(f"\n[{i+1:2d}/{min(15, len(robot_urdfs)):2d}] Processing: {Path(urdf_path).name}")

        success, nodes, edges = generate_robot_graph(urdf_path, output_dir)
        if success:
            success_count += 1

        results.append({
            'urdf_path': str(urdf_path),
            'robot_name': Path(urdf_path).stem,
            'success': success,
            'nodes': nodes,
            'edges': edges
        })

    # Save summary
    summary = {
        'total_processed': len(results),
        'successful': success_count,
        'failed': len(results) - success_count,
        'results': results
    }

    summary_path = output_dir / "generation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n🎉 Graph generation completed!")
    print(f"   Successful: {success_count}/{len(results)}")
    print(f"   Output directory: {output_dir}")
    print(f"   Summary saved: {summary_path}")

if __name__ == "__main__":
    main()