#!/usr/bin/env python3
"""
测试修改后的合成脚本 - 只对1个episode生成少量变种来验证流程
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import json
from run_link_scaling_synthesis import LinkScalingSynthesizer
from run_dof_modification_synthesis import DOFModificationSynthesizer


def test_synthesis():
    """测试两个合成脚本"""

    # Load valid episodes
    TASK_DESC_PATH = "/home/cx/AET_FOR_RL/vla/valid_original_data/droid_100/task_descriptions.json"
    with open(TASK_DESC_PATH, 'r') as f:
        task_data = json.load(f)

    valid_episodes = task_data['valid_episode_list'][:3]  # 只测试前3个episodes

    print(f"🧪 Testing synthesis with episodes: {valid_episodes}")
    print("=" * 50)

    # 使用转换后的数据路径
    DROID_PATH = "/home/cx/AET_FOR_RL/vla/converted_data/droid_100"

    # Test 1: Link Scaling
    print("\n🔗 Testing Link Scaling Synthesis...")
    try:
        link_synthesizer = LinkScalingSynthesizer(DROID_PATH)
        link_variations = link_synthesizer.synthesize_valid_episodes_streaming(valid_episodes, 2)  # 2 variations per episode
        print(f"✅ Link scaling test successful: {link_variations} variations generated")
    except Exception as e:
        print(f"❌ Link scaling test failed: {e}")
        return False

    # Test 2: DOF Modification
    print("\n🔧 Testing DOF Modification Synthesis...")
    try:
        dof_synthesizer = DOFModificationSynthesizer(DROID_PATH)
        dof_variations = dof_synthesizer.synthesize_valid_episodes_streaming(valid_episodes, 2)  # 2 variations per episode
        print(f"✅ DOF modification test successful: {dof_variations} variations generated")
    except Exception as e:
        print(f"❌ DOF modification test failed: {e}")
        return False

    print(f"\n🎉 All tests passed!")
    print(f"   📊 Link scaling variations: {link_variations}")
    print(f"   🔧 DOF modification variations: {dof_variations}")
    print(f"   📈 Total variations: {link_variations + dof_variations}")

    return True


if __name__ == "__main__":
    test_synthesis()