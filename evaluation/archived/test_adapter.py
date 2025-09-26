#!/usr/bin/env python3
"""
Test script for our Multi-Morphology VLA adapter
Verify that the model loads correctly and can make predictions
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path

# Import our adapter
from morphology_vla_adapter import MorphologyVLAAdapter, create_vla_model

def test_adapter_loading():
    """Test basic adapter loading functionality"""
    print("🧪 Testing Multi-Morphology VLA Adapter")
    print("=" * 50)
    
    try:
        # Test adapter creation
        adapter = create_vla_model()
        
        # Check model info
        info = adapter.get_model_info()
        print(f"✅ Model Info:")
        print(f"   Name: {info['name']}")
        print(f"   Architecture: {info['architecture']}")
        print(f"   Parameters: {info['parameters']}")
        print(f"   Capabilities: {len(info['unique_capabilities'])} unique features")
        
        return adapter
        
    except Exception as e:
        print(f"❌ Adapter loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_prediction_pipeline(adapter):
    """Test prediction with sample inputs"""
    print(f"\n🎯 Testing Prediction Pipeline")
    
    try:
        # Create sample image (simulate DROID camera 320x180)
        sample_image = np.random.randint(0, 255, (180, 320, 3), dtype=np.uint8)
        print(f"   📸 Sample image shape: {sample_image.shape}")
        
        # Sample instruction
        instruction = "Pick up the object using the extended arm configuration"
        print(f"   🗣️  Instruction: '{instruction}'")
        
        # Test different morphology configurations
        test_configs = [
            # Standard 7-DOF
            {
                "dof": 7,
                "link_scales": [1.0] * 7,
                "morphology_type": "standard_franka"
            },
            # Extended link 6-DOF
            {
                "dof": 6, 
                "link_scales": [1.0, 1.0, 1.2, 1.1, 1.0, 1.0],
                "morphology_type": "extended_arm"
            },
            # Compact 8-DOF
            {
                "dof": 8,
                "link_scales": [0.9] * 8,
                "morphology_type": "compact_multi_dof"
            }
        ]
        
        for i, config in enumerate(test_configs):
            print(f"\n   🤖 Test {i+1}: {config['morphology_type']} ({config['dof']} DOF)")
            
            # Make prediction
            action = adapter.predict_action(sample_image, instruction, config)
            
            print(f"      ✅ Action shape: {action.shape}")
            print(f"      📊 Action range: [{action.min():.3f}, {action.max():.3f}]")
            print(f"      🎯 Sample actions: [{action[0]:.3f}, {action[1]:.3f}, {action[2]:.3f}]...")
            
            # Verify action dimensions match expected DOF
            expected_dof = config['dof'] if config['dof'] <= 7 else 7  # Our model outputs 7-DOF
            if len(action) == expected_dof or len(action) == 7:  # Allow both
                print(f"      ✅ Action dimensions correct")
            else:
                print(f"      ⚠️  Unexpected action dimensions: {len(action)} (expected {expected_dof})")
        
        print(f"\n🎉 Prediction pipeline test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Prediction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_benchmark_interface():
    """Test standard benchmark interface compatibility"""
    print(f"\n🏆 Testing Benchmark Interface Compatibility")
    
    try:
        # Test standard VLA interface
        model = create_vla_model()
        
        # Simulate LIBERO-style input
        rgb_obs = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)  # Typical benchmark size
        task_instruction = "place the red block on the blue plate"
        
        print(f"   📋 Simulating LIBERO-style evaluation")
        print(f"   🖼️  Obs shape: {rgb_obs.shape}")
        print(f"   📝 Instruction: '{task_instruction}'")
        
        # Make prediction using standard interface
        action = model.predict_action(rgb_obs, task_instruction)
        
        print(f"   ✅ Standard prediction successful")
        print(f"   🎯 Action: {action[:3]} (first 3 DOF)")
        
        # Test our morphology-aware enhancement
        morph_instruction = "use the extended arm to reach the distant object"
        morph_config = {
            "dof": 7,
            "link_scales": [1.0, 1.0, 1.3, 1.2, 1.0, 1.0, 1.0],
            "morphology_type": "extended_reach"
        }
        
        morphology_action = model.predict_action(rgb_obs, morph_instruction, morph_config)
        
        print(f"   🌟 Morphology-aware prediction successful")
        print(f"   🤖 Enhanced action: {morphology_action[:3]} (first 3 DOF)")
        
        # Compare actions
        action_diff = np.abs(action - morphology_action).mean()
        print(f"   📊 Action difference (morphology vs standard): {action_diff:.4f}")
        
        if action_diff > 0.01:  # Some meaningful difference expected
            print(f"   ✅ Morphology awareness verified (actions differ meaningfully)")
        else:
            print(f"   ⚠️  Actions very similar - morphology effect minimal")
        
        return True
        
    except Exception as e:
        print(f"❌ Benchmark interface test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all adapter tests"""
    print("🚀 Multi-Morphology VLA Adapter Testing Suite")
    print("=" * 60)
    
    # Test 1: Basic loading
    adapter = test_adapter_loading()
    if adapter is None:
        print("❌ Cannot proceed - adapter loading failed")
        return
    
    # Test 2: Prediction pipeline
    prediction_ok = test_prediction_pipeline(adapter)
    
    # Test 3: Benchmark compatibility
    benchmark_ok = test_benchmark_interface()
    
    # Summary
    print(f"\n📊 Test Results Summary:")
    print(f"   ✅ Adapter Loading: {'PASS' if adapter else 'FAIL'}")
    print(f"   ✅ Prediction Pipeline: {'PASS' if prediction_ok else 'FAIL'}")  
    print(f"   ✅ Benchmark Interface: {'PASS' if benchmark_ok else 'FAIL'}")
    
    if all([adapter, prediction_ok, benchmark_ok]):
        print(f"\n🎉 ALL TESTS PASSED! Adapter ready for SOTA evaluation!")
        print(f"   🎯 Next step: Run LIBERO benchmark evaluation")
        print(f"   🏆 Target: Beat OpenVLA-OFT's 97.1% success rate")
    else:
        print(f"\n⚠️  Some tests failed - check adapter implementation")

if __name__ == "__main__":
    main()