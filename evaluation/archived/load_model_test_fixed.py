#!/usr/bin/env python3
"""
修复设备放置问题的模型加载测试
"""

import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

# Add paths
sys.path.append('/home/cx/AET_FOR_RL/vla/train')
sys.path.append('/home/cx/AET_FOR_RL/vla')
sys.path.append('/home/cx/AET_FOR_RL/MA-VLA/src')

import torch
import numpy as np
from pathlib import Path

def fix_model_device_placement(model, device):
    """确保模型的所有组件都在正确的设备上"""
    print("🔧 Fixing device placement issues...")
    
    # 递归地将所有子模块移到设备上
    model = model.to(device)
    
    # 检查并修复embedding层
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Embedding):
            print(f"   Moving embedding {name} to {device}")
            module = module.to(device)
    
    # 确保所有参数和缓冲区都在正确的设备上
    for name, param in model.named_parameters():
        if param.device != device:
            print(f"   Moving parameter {name} to {device}")
            param.data = param.data.to(device)
    
    for name, buffer in model.named_buffers():
        if buffer.device != device:
            print(f"   Moving buffer {name} to {device}")
            buffer.data = buffer.data.to(device)
            
    return model

def test_model_loading_with_fix():
    """测试模型加载并修复设备问题"""
    print("🧪 Testing VLA Model Loading (with device fix)")
    print("=" * 50)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Device: {device}")
    
    # 导入模型
    try:
        from ma_vla_core import MA_VLA_Agent, RobotConfig
        print("✅ Successfully imported MA_VLA_Agent")
    except Exception as e:
        print(f"❌ Failed to import model: {e}")
        return None
        
    # 加载checkpoint
    model_path = "/home/cx/AET_FOR_RL/MA-VLA/checkpoints/ma_vla_final.pt"
    
    try:
        print("\n📦 Loading checkpoint...")
        checkpoint = torch.load(model_path, map_location='cpu')
        print(f"✅ Checkpoint loaded")
        print(f"   Keys: {list(checkpoint.keys())}")
        
        # 获取配置
        max_dof = checkpoint.get('max_dof', 14)
        vision_language_dim = checkpoint.get('vision_language_dim', 512)
        
        print(f"   max_dof: {max_dof}")
        print(f"   vision_language_dim: {vision_language_dim}")
        
        # 创建模型
        print("\n🤖 Creating model instance...")
        model = MA_VLA_Agent(
            max_dof=max_dof,
            observation_dim=vision_language_dim,
            hidden_dim=256
        )
        
        # 加载权重
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            print("✅ Loaded state_dict")
        
        # 修复设备放置问题
        model = fix_model_device_placement(model, device)
        model.eval()
        
        print(f"✅ Model ready on {device}")
        
        # 测试forward pass
        print("\n🧪 Testing forward pass...")
        
        # 创建测试输入 - 确保所有输入都在正确的设备上
        dummy_observations = torch.randn(vision_language_dim).to(device)
        
        # 创建机器人配置
        robot_config = RobotConfig(
            name="Franka_Panda",
            dof=7,
            joint_types=["revolute"] * 7,
            joint_limits=[(-2.8973, 2.8973), (-1.7628, 1.7628), (-2.8973, 2.8973), 
                         (-3.0718, -0.0698), (-2.8973, 2.8973), (-0.0175, 3.7525), 
                         (-2.8973, 2.8973)],
            link_lengths=[0.333, 0.316, 0.384, 0.088, 0.107, 0.103, 0.0]
        )
        
        # 修改MA-VLA模型的forward方法中的张量创建，确保使用正确的设备
        # 这里我们创建一个包装函数
        def safe_forward(model, observations, robot_config):
            """安全的forward pass，确保所有张量在同一设备"""
            # 保存原始的tensor创建函数
            original_tensor = torch.tensor
            original_zeros = torch.zeros
            
            # 创建设备感知的tensor创建函数
            def device_aware_tensor(*args, **kwargs):
                if 'device' not in kwargs:
                    kwargs['device'] = device
                return original_tensor(*args, **kwargs)
            
            def device_aware_zeros(*args, **kwargs):
                if 'device' not in kwargs:
                    kwargs['device'] = device
                return original_zeros(*args, **kwargs)
            
            # 临时替换
            torch.tensor = device_aware_tensor
            torch.zeros = device_aware_zeros
            
            try:
                with torch.no_grad():
                    output = model(observations, robot_config)
                return output
            finally:
                # 恢复原始函数
                torch.tensor = original_tensor
                torch.zeros = original_zeros
        
        try:
            output = safe_forward(model, dummy_observations, robot_config)
            print(f"✅ Forward pass successful!")
            print(f"   Actions shape: {output['actions'].shape}")
            print(f"   Values shape: {output['values'].shape}")
            print(f"   Actions device: {output['actions'].device}")
            
            # 验证输出
            actions = output['actions'].cpu().numpy()
            print(f"   Action range: [{actions.min():.3f}, {actions.max():.3f}]")
            
            return model, robot_config
            
        except Exception as e:
            print(f"❌ Forward pass failed: {e}")
            print(f"   Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            return None, None
            
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None, None

def test_multi_morphology_inference(model, base_config):
    """测试多形态推理"""
    if model is None:
        return
        
    print("\n🤖 Testing Multi-Morphology Inference")
    print("=" * 50)
    
    from ma_vla_core import RobotConfig  # Import here as well
    device = next(model.parameters()).device
    
    # 测试不同的形态配置
    morphology_configs = [
        ("5-DOF", 5, [0.3, 0.3, 0.3, 0.1, 0.1]),
        ("7-DOF", 7, [0.333, 0.316, 0.384, 0.088, 0.107, 0.103, 0.0]),
        ("8-DOF", 8, [0.3, 0.3, 0.3, 0.2, 0.2, 0.1, 0.1, 0.05])
    ]
    
    for name, dof, lengths in morphology_configs:
        print(f"\n📊 Testing {name}...")
        
        config = RobotConfig(
            name=name,
            dof=dof,
            joint_types=["revolute"] * dof,
            joint_limits=[(-3.14, 3.14)] * dof,
            link_lengths=lengths
        )
        
        # 创建观察
        observations = torch.randn(512).to(device)
        
        try:
            # 使用monkey patching确保设备一致性
            original_tensor = torch.tensor
            torch.tensor = lambda *args, **kwargs: original_tensor(*args, **{**kwargs, 'device': device})
            
            with torch.no_grad():
                output = model(observations, config)
            
            torch.tensor = original_tensor
            
            print(f"   ✅ Success! Actions shape: {output['actions'].shape}")
            actions = output['actions'].cpu().numpy()
            print(f"   Action stats: mean={actions.mean():.3f}, std={actions.std():.3f}")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")

def main():
    """主函数"""
    print("🚀 MA-VLA Model Loading and Testing")
    print("This time with proper device handling!")
    print("=" * 50)
    
    # 测试模型加载
    model, robot_config = test_model_loading_with_fix()
    
    if model is not None:
        print("\n✅ Model loaded successfully!")
        
        # 测试多形态推理
        test_multi_morphology_inference(model, robot_config)
        
        print("\n🎉 SUCCESS!")
        print("Model is ready for real evaluation!")
        print("Next step: Run inference on actual DROID-100 test data")
    else:
        print("\n❌ Model loading failed")
        print("Need to debug the device placement issues further")

if __name__ == "__main__":
    main()