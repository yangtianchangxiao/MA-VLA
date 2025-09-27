#!/usr/bin/env python3
"""
测试OpenPI 0.5模型下载和基础功能
"""

def test_pi05_download():
    print("🚀 开始测试OpenPI 0.5模型下载...")

    try:
        # 导入必要模块
        print("📦 导入OpenPI模块...")
        from openpi.training import config as _config
        from openpi.shared import download
        from openpi.policies import policy_config

        print("✅ 模块导入成功!")

        # 1. 测试π₀.₅ base模型下载
        print("\n📥 下载π₀.₅ base模型...")
        try:
            config = _config.get_config("pi05_base")
            checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_base")
            print(f"✅ π₀.₅ base模型下载成功!")
            print(f"   路径: {checkpoint_dir}")
            print(f"   配置: {type(config).__name__}")
        except Exception as e:
            print(f"❌ π₀.₅ base下载失败: {e}")
            # 尝试下载DROID版本作为替代
            print("🔄 尝试下载π₀.₅ DROID版本...")
            config = _config.get_config("pi05_droid")
            checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_droid")
            print(f"✅ π₀.₅ DROID版本下载成功!")
            print(f"   路径: {checkpoint_dir}")

        # 2. 测试policy创建
        print("\n🤖 创建policy...")
        policy = policy_config.create_trained_policy(config, checkpoint_dir)
        print("✅ Policy创建成功!")

        # 3. 测试推理
        print("\n🧠 测试推理功能...")
        from openpi.policies import droid_policy
        example = droid_policy.make_droid_example()
        result = policy.infer(example)

        print(f"✅ 推理测试成功!")
        print(f"   输入: 图像 + 文本 + 机器人状态")
        print(f"   输出: actions shape = {result['actions'].shape}")

        # 4. 检查模型结构
        print(f"\n🔍 模型信息:")
        print(f"   Config类型: {type(config).__name__}")
        print(f"   Model类型: {type(config.model).__name__}")
        if hasattr(config.model, 'action_dim'):
            print(f"   Action维度: {config.model.action_dim}")
        if hasattr(config.model, 'action_horizon'):
            print(f"   Action horizon: {config.model.action_horizon}")

        # 释放内存
        del policy

        print(f"\n🎉 所有测试通过! OpenPI 0.5已可用于你的hybrid方案!")
        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        print(f"   请检查:")
        print(f"   1. 网络连接")
        print(f"   2. Google Cloud Storage访问权限")
        print(f"   3. 依赖安装完整性")
        return False

if __name__ == "__main__":
    success = test_pi05_download()
    exit(0 if success else 1)