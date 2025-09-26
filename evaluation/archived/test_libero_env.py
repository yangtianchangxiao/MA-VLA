#!/usr/bin/env python3
"""
Test LIBERO Environment Creation
Check if we can create and interact with LIBERO tasks
"""

import numpy as np
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
import os

def test_libero_environment():
    """Test basic LIBERO environment functionality"""
    print("🧪 Testing LIBERO Environment Creation")
    print("=" * 50)
    
    try:
        # Get available benchmarks
        benchmark_dict = benchmark.get_benchmark_dict()
        print(f"✅ Available benchmarks: {list(benchmark_dict.keys())}")
        
        # Choose a simple benchmark to start with
        task_suite_name = "libero_10"  # Start with small suite
        if task_suite_name not in benchmark_dict:
            print(f"⚠️  {task_suite_name} not available, trying libero_spatial")
            task_suite_name = "libero_spatial"
        
        task_suite = benchmark_dict[task_suite_name]()
        print(f"✅ Created task suite: {task_suite_name}")
        
        # Get first task
        task_id = 0
        task = task_suite.get_task(task_id)
        task_name = task.name
        task_description = task.language
        
        print(f"📋 Task {task_id}: {task_name}")
        print(f"🗣️  Description: '{task_description}'")
        
        # Get task BDDL file
        from libero.libero import get_libero_path
        task_bddl_file = os.path.join(
            get_libero_path("bddl_files"), 
            task.problem_folder, 
            task.bddl_file
        )
        print(f"📄 BDDL file: {task_bddl_file}")
        
        # Create environment
        env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": 128,
            "camera_widths": 128,
            "has_renderer": False,  # Offscreen rendering
            "has_offscreen_renderer": True,
        }
        
        print(f"🏗️  Creating environment...")
        env = OffScreenRenderEnv(**env_args)
        print(f"✅ Environment created successfully!")
        
        # Test environment reset and step
        print(f"🔄 Testing environment reset...")
        env.seed(0)
        obs = env.reset()
        print(f"✅ Environment reset successful")
        print(f"📊 Observation keys: {list(obs.keys())}")
        
        if 'agentview_rgb' in obs:
            rgb_shape = obs['agentview_rgb'].shape
            print(f"📸 RGB observation shape: {rgb_shape}")
        
        # Test action step
        print(f"🎮 Testing dummy action step...")
        dummy_action = np.zeros(7)  # 7-DOF action for Franka
        obs, reward, done, info = env.step(dummy_action)
        
        print(f"✅ Action step successful")
        print(f"🏆 Reward: {reward}")
        print(f"✅ Done: {done}")
        print(f"ℹ️  Info keys: {list(info.keys()) if isinstance(info, dict) else type(info)}")
        
        # Close environment
        env.close()
        print(f"🔒 Environment closed successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ LIBERO environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_task_suite_info():
    """Test getting information about all task suites"""
    print(f"\n📚 Testing Task Suite Information")
    
    try:
        benchmark_dict = benchmark.get_benchmark_dict()
        
        for suite_name, suite_class in benchmark_dict.items():
            print(f"\n🔍 {suite_name}:")
            try:
                suite = suite_class()
                num_tasks = suite.n_tasks
                print(f"   📊 Number of tasks: {num_tasks}")
                
                # Get first task info
                if num_tasks > 0:
                    task = suite.get_task(0)
                    print(f"   📝 Sample task: {task.name}")
                    print(f"   💬 Sample description: '{task.language}'")
                    
            except Exception as e:
                print(f"   ❌ Failed to load {suite_name}: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Task suite info test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 LIBERO Environment Testing")
    print("=" * 60)
    
    # Test 1: Basic environment functionality
    env_test_passed = test_libero_environment()
    
    # Test 2: Task suite information
    suite_test_passed = test_task_suite_info()
    
    print(f"\n📊 Test Results:")
    print(f"   🏗️  Environment Test: {'PASS' if env_test_passed else 'FAIL'}")
    print(f"   📚 Task Suite Test: {'PASS' if suite_test_passed else 'FAIL'}")
    
    if env_test_passed and suite_test_passed:
        print(f"\n🎉 All LIBERO tests passed! Environment is ready for VLA evaluation.")
        print(f"💡 Next: Create VLA-LIBERO interface adapter")
    else:
        print(f"\n⚠️  Some tests failed. Check the error messages above.")