#!/usr/bin/env python3
"""
测试加载我们训练好的VLA模型
"""

import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

# Add the training directory to path so we can import our model
sys.path.append('/home/cx/AET_FOR_RL/vla/train')
sys.path.append('/home/cx/AET_FOR_RL/vla')
sys.path.append('/home/cx/AET_FOR_RL/MA-VLA/src')

import torch
import numpy as np
from pathlib import Path

def test_model_loading():
    """测试模型加载"""
    print("🧪 Testing VLA Model Loading")
    print("=" * 40)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Device: {device}")
    
    # 尝试导入我们的模型
    try:
        from ma_vla_core import MA_VLA_Agent, RobotConfig
        print("✅ Successfully imported MA_VLA_Agent")
    except Exception as e:
        print(f"❌ Failed to import model: {e}")
        return None
        
    # 尝试加载checkpoint
    model_path = "/home/cx/AET_FOR_RL/MA-VLA/checkpoints/ma_vla_final.pt"
    
    try:
        print("🔍 Loading checkpoint...")
        checkpoint = torch.load(model_path, map_location='cpu')
        print(f"✅ Checkpoint loaded")
        
        if isinstance(checkpoint, dict):
            print(f"   Keys: {list(checkpoint.keys())}")
            
            # 尝试提取模型状态
            if 'model_state_dict' in checkpoint:
                model_state = checkpoint['model_state_dict']
                print("   Found model_state_dict")
            elif 'model' in checkpoint:
                model_state = checkpoint['model']
                print("   Found model")
            else:
                # 可能整个checkpoint就是state dict
                model_state = checkpoint
                print("   Using checkpoint as state dict")
                
            # 查看模型参数的一些key
            model_keys = list(model_state.keys())[:10]
            print(f"   Model parameter keys (first 10): {model_keys}")
            
            # 检查checkpoint中保存的配置
            print("\n🤖 Attempting to create model instance...")
            
            # 从checkpoint获取配置
            max_dof = checkpoint.get('max_dof', 14)
            vision_language_dim = checkpoint.get('vision_language_dim', 512)
            
            print(f"   max_dof: {max_dof}")
            print(f"   vision_language_dim: {vision_language_dim}")
            
            # 尝试创建模型
            try:
                model = MA_VLA_Agent(
                    max_dof=max_dof,
                    observation_dim=vision_language_dim,
                    hidden_dim=256  # 默认值
                )
                
                # 加载state_dict
                if 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'], strict=False)
                else:
                    model.load_state_dict(model_state, strict=False)
                    
                model.to(device)
                model.eval()
                
                print("✅ Model loaded successfully!")
                print(f"   Model on device: {next(model.parameters()).device}")
                
                # 测试一个简单的forward pass
                print("\n🧪 Testing forward pass...")
                
                # Create MA-VLA compatible inputs
                dummy_observations = torch.randn(vision_language_dim).to(device)
                
                # Create a robot config for testing (Franka Panda 7-DOF)
                robot_config = RobotConfig(
                    name="Franka_Panda",
                    dof=7,
                    joint_types=["revolute"] * 7,
                    joint_limits=[(-2.8973, 2.8973), (-1.7628, 1.7628), (-2.8973, 2.8973), 
                                 (-3.0718, -0.0698), (-2.8973, 2.8973), (-0.0175, 3.7525), 
                                 (-2.8973, 2.8973)],
                    link_lengths=[0.333, 0.316, 0.384, 0.088, 0.107, 0.103, 0.0]
                )
                
                try:
                    with torch.no_grad():
                        output = model(dummy_observations, robot_config)
                    print(f"✅ Forward pass successful!")
                    print(f"   Actions shape: {output['actions'].shape}")
                    print(f"   Values shape: {output['values'].shape}")
                    print(f"   Output keys: {list(output.keys())}")
                    return model
                    
                except Exception as e:
                    print(f"❌ Forward pass failed: {e}")
                    return None
                    
            except Exception as e:
                print(f"❌ Failed to create/load model: {e}")
                print("   This might be due to architecture mismatch or missing parameters")
                return None
                
        else:
            print("❌ Checkpoint is not a dictionary")
            return None
            
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        return None

def main():
    """主函数"""
    model = test_model_loading()
    
    if model is not None:
        print("\n🎉 SUCCESS!")
        print("Model is loaded and ready for evaluation!")
    else:
        print("\n🚨 FAILED!")
        print("Need to investigate model architecture or checkpoint format")

if __name__ == "__main__":
    main()