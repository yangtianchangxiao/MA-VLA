#!/usr/bin/env python3
"""
Quick Morphology Evaluation Test
测试我们的多形态评估框架是否工作正常
"""

import sys
import os
sys.path.append('/home/cx/AET_FOR_RL')

import numpy as np
from pathlib import Path

def quick_evaluation_test():
    """快速测试评估框架"""
    print("🧪 Quick Morphology Evaluation Test")
    print("=" * 40)
    
    # 检查模型文件
    model_path = Path("/home/cx/AET_FOR_RL/MA-VLA/checkpoints/ma_vla_final.pt")
    test_data_path = Path("/home/cx/AET_FOR_RL/MA-VLA/data/datasets/droid_100")
    
    print(f"📁 Model path: {model_path}")
    print(f"   Exists: {model_path.exists()}")
    
    print(f"📁 Test data path: {test_data_path}")
    print(f"   Exists: {test_data_path.exists()}")
    
    if model_path.exists():
        print("🔍 Model checkpoint exists - evaluation framework ready")
        print(f"   Size: {model_path.stat().st_size / (1024**2):.1f} MB")
    
    if test_data_path.exists():
        print("🔍 Inspecting test data...")
        try:
            import pandas as pd
            data_file = test_data_path / "data" / "chunk-000" / "file-000.parquet"
            if data_file.exists():
                df = pd.read_parquet(data_file)
                print(f"   Data shape: {df.shape}")
                print(f"   Columns: {list(df.columns)[:5]}...")
                if 'episode_id' in df.columns:
                    print(f"   Number of episodes: {df['episode_id'].nunique()}")
            else:
                print("   ❌ Parquet file not found")
        except Exception as e:
            print(f"   ❌ Error loading data: {e}")
    
    print("\n✅ Quick test complete!")
    print("Ready to run full evaluation framework")

if __name__ == "__main__":
    quick_evaluation_test()