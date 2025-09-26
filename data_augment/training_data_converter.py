#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training Data Converter for VLA Training
Convert synthesis data to RynnVLA training format
"""

import os
import sys
import json
import numpy as np
from typing import List, Dict, Any
from pathlib import Path
import argparse

sys.path.append('/home/cx/AET_FOR_RL/vla/original_data/droid_100')

def load_synthesis_data(synthesis_path: str) -> List[Dict]:
    """Load all synthesis variations from chunk files"""
    variations = []
    
    for chunk_file in sorted(Path(synthesis_path).glob("*_chunk_*.json")):
        print(f"Loading {chunk_file}")
        with open(chunk_file, 'r') as f:
            chunk_data = json.load(f)
            variations.extend(chunk_data)
    
    return variations

def convert_to_training_format(synthesis_data: List[Dict], 
                               original_droid_path: str,
                               output_path: str,
                               transformation_type: str) -> Dict:
    """Convert synthesis data to RynnVLA training format"""
    
    training_stats = {
        "transformation_type": transformation_type,
        "total_episodes": len(synthesis_data),
        "episodes": []
    }
    
    for idx, variation in enumerate(synthesis_data):
        episode_idx = variation['episode_index']
        variation_idx = variation['variation_index']
        
        # Get original DROID image paths (preserved via episode mapping)
        droid_episode_path = f"{original_droid_path}/data/chunk-000/file-000.parquet"
        
        # Create training episode entry
        training_episode = {
            "episode_id": f"{transformation_type}_ep{episode_idx:03d}_var{variation_idx:02d}",
            "original_episode": episode_idx,
            "variation_type": transformation_type,
            "trajectory_length": len(variation['modified_trajectory']),
            "quality_score": variation.get('quality_score', 0.8),
            
            # Morphology info
            "morphology_config": variation['config'],
            "transformation_data": variation['variation_data'],
            
            # Action and state data
            "actions": variation['modified_trajectory'].tolist() if isinstance(variation['modified_trajectory'], np.ndarray) else variation['modified_trajectory'],
            "original_actions": variation['original_actions'].tolist() if isinstance(variation['original_actions'], np.ndarray) else variation['original_actions'],
            
            # Image data (reference to original DROID images)
            "image_data_source": "droid_100_original",
            "droid_episode_index": episode_idx,
            
            # Instruction (generic for now, can be enhanced)
            "instruction": f"Perform manipulation task with {transformation_type} morphology variation",
            
            # Required keys for RynnVLA
            "required_keys": [
                "action", 
                "timestamp", 
                "obs/front_image", 
                "obs/wrist_image", 
                "obs/state"
            ]
        }
        
        training_stats["episodes"].append(training_episode)
    
    # Save training statistics
    os.makedirs(output_path, exist_ok=True)
    output_file = f"{output_path}/{transformation_type}_training_stats.json"
    
    with open(output_file, 'w') as f:
        json.dump(training_stats, f, indent=2, default=str)
    
    print(f"✅ Converted {len(synthesis_data)} variations to training format")
    print(f"💾 Saved training stats to {output_file}")
    
    return training_stats

def create_merged_training_data(dof_stats: Dict, link_stats: Dict, output_path: str) -> str:
    """Merge DOF and Link scaling training data"""
    
    merged_stats = {
        "dataset_name": "droid_100_morphology_synthesis",
        "total_episodes": dof_stats["total_episodes"] + link_stats["total_episodes"],
        "transformation_types": ["dof_modification", "link_scaling"],
        "dof_episodes": dof_stats["total_episodes"],
        "link_episodes": link_stats["total_episodes"],
        "episodes": dof_stats["episodes"] + link_stats["episodes"]
    }
    
    output_file = f"{output_path}/merged_training_stats.json"
    with open(output_file, 'w') as f:
        json.dump(merged_stats, f, indent=2, default=str)
    
    print(f"🎯 Created merged training dataset: {merged_stats['total_episodes']} total episodes")
    print(f"   📊 DOF variations: {merged_stats['dof_episodes']}")
    print(f"   🔗 Link variations: {merged_stats['link_episodes']}")
    print(f"💾 Saved merged stats to {output_file}")
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description='Convert synthesis data to VLA training format')
    parser.add_argument('--dof_path', default='/home/cx/AET_FOR_RL/vla/synthesized_data/droid_100_morphology/dof_modification',
                        help='Path to DOF synthesis data')
    parser.add_argument('--link_path', default='/home/cx/AET_FOR_RL/vla/synthesized_data/droid_100_morphology/link_scaling',
                        help='Path to Link synthesis data')
    parser.add_argument('--droid_path', default='/home/cx/AET_FOR_RL/vla/original_data/droid_100',
                        help='Path to original DROID-100 data')
    parser.add_argument('--output', default='/home/cx/AET_FOR_RL/vla/training_data',
                        help='Output directory for training data')
    
    args = parser.parse_args()
    
    print("🚀 Converting Synthesis Data to VLA Training Format")
    print("=" * 60)
    
    # Load synthesis data
    print("\n📊 Loading DOF synthesis data...")
    dof_data = load_synthesis_data(args.dof_path)
    
    print(f"\n🔗 Loading Link synthesis data...")
    if os.path.exists(f"{args.link_path}/link_scaling_chunk_000.json"):
        link_data = load_synthesis_data(args.link_path)
    else:
        print("⚠️ Link synthesis not yet complete, using empty dataset")
        link_data = []
    
    # Convert to training format
    print(f"\n🔄 Converting to training format...")
    dof_stats = convert_to_training_format(dof_data, args.droid_path, args.output, "dof_modification")
    link_stats = convert_to_training_format(link_data, args.droid_path, args.output, "link_scaling")
    
    # Create merged dataset
    print(f"\n🎯 Creating merged training dataset...")
    merged_file = create_merged_training_data(dof_stats, link_stats, args.output)
    
    print(f"\n✅ Training data conversion completed!")
    print(f"📁 Training data ready at: {args.output}")
    print(f"📋 Merged stats file: {merged_file}")

if __name__ == "__main__":
    main()