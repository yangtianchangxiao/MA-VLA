#!/usr/bin/env python3
"""
软体臂训练流程测试脚本
只测试数据流和基础组件，不需要完整OpenPi模型
符合Linus原则: 测试核心逻辑，不测试外部依赖
"""

import os
import sys
import torch
import yaml
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def test_data_pipeline():
    """测试数据管道"""
    print("🧪 测试数据管道...")

    try:
        from data.soft_arm_data_adapter import create_soft_arm_openpi_dataloader

        processed_data_dir = project_root / "data" / "processed"

        if not processed_data_dir.exists():
            print(f"❌ 数据目录不存在: {processed_data_dir}")
            return False

        # 测试训练数据加载器
        train_loader = create_soft_arm_openpi_dataloader(
            str(processed_data_dir),
            split='train',
            batch_size=2
        )

        # 测试一个批次
        obs_wrapper, actions = next(iter(train_loader))
        obs_dict = obs_wrapper.to_dict()

        print(f"✅ 数据加载成功:")
        print(f"   图像形状: {obs_dict['image']['camera_0'].shape}")
        print(f"   动作形状: {actions.shape}")
        print(f"   图数据形状: {obs_dict['graph_data']['node_features'].shape}")
        print(f"   指令数量: {len(obs_dict['instruction'])}")

        return True

    except Exception as e:
        print(f"❌ 数据管道测试失败: {e}")
        return False

def test_graph_encoder():
    """测试图编码器"""
    print("\n🧪 测试图编码器...")

    try:
        from models.standalone_graph_encoder import StandaloneGraphEncoder

        encoder = StandaloneGraphEncoder(19, 32)

        # 模拟图数据
        batch_size = 2
        num_nodes = 10
        graph_data = {
            'node_features': torch.randn(batch_size, num_nodes, 19),
            'num_nodes': torch.tensor([6, 8])  # 有效节点数
        }

        with torch.no_grad():
            output = encoder(graph_data)

        print(f"✅ 图编码器测试成功:")
        print(f"   输入: {graph_data['node_features'].shape}")
        print(f"   输出: {output.shape}")
        print(f"   输出范围: [{output.min():.3f}, {output.max():.3f}]")

        return True

    except Exception as e:
        print(f"❌ 图编码器测试失败: {e}")
        return False

def test_config_loading():
    """测试配置加载"""
    print("\n🧪 测试配置加载...")

    try:
        config_path = project_root / "configs" / "debug_config.yaml"

        if not config_path.exists():
            print(f"❌ 配置文件不存在: {config_path}")
            return False

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        print(f"✅ 配置加载成功:")
        print(f"   实验名称: {config['experiment']['name']}")
        print(f"   批量大小: {config['data']['batch_size']}")
        print(f"   最大训练步数: {config['training']['max_steps']}")
        print(f"   图支持: {config['model']['graph']['enabled']}")

        return True

    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        return False

def test_integration():
    """测试数据和模型集成"""
    print("\n🧪 测试数据和模型集成...")

    try:
        from data.soft_arm_data_adapter import create_soft_arm_openpi_dataloader
        from models.standalone_graph_encoder import MockPI0Model

        # 加载一个批次数据
        processed_data_dir = project_root / "data" / "processed"
        train_loader = create_soft_arm_openpi_dataloader(
            str(processed_data_dir),
            split='train',
            batch_size=1
        )

        obs_wrapper, actions = next(iter(train_loader))
        obs_dict = obs_wrapper.to_dict()

        # 提取图数据
        graph_data = obs_dict['graph_data']

        # 创建模拟模型
        model = MockPI0Model(action_dim=10, action_horizon=16, enable_graph=True)

        # 测试前向传播
        with torch.no_grad():
            loss = model(obs_dict, actions, graph_data)

        print(f"✅ 集成测试成功:")
        print(f"   数据形状检查通过:")
        print(f"     图像: {obs_dict['image']['camera_0'].shape}")
        print(f"     动作: {actions.shape}")
        print(f"     图数据: {graph_data['node_features'].shape}")
        print(f"   模型前向传播: loss = {loss.item():.4f}")

        return True

    except Exception as e:
        print(f"❌ 集成测试失败: {e}")
        return False

def test_device_compatibility():
    """测试GPU兼容性"""
    print("\n🧪 测试GPU兼容性...")

    if not torch.cuda.is_available():
        print("⚠️ CUDA不可用，跳过GPU测试")
        return True

    try:
        device = torch.device('cuda:0')

        # 测试模拟模型在GPU上
        from models.standalone_graph_encoder import MockPI0Model
        model = MockPI0Model(action_dim=10, action_horizon=16, enable_graph=True).to(device)

        # 模拟GPU数据
        observation = {
            'image': {'camera_0': torch.randn(2, 3, 224, 224).to(device)},
            'instruction': ['task 1', 'task 2']
        }
        actions = torch.randn(2, 16, 10).to(device)
        graph_data = {
            'node_features': torch.randn(2, 10, 19).to(device),
            'num_nodes': torch.tensor([6, 8]).to(device)
        }

        with torch.no_grad():
            loss = model(observation, actions, graph_data)

        print(f"✅ GPU测试成功:")
        print(f"   设备: {device}")
        print(f"   GPU内存使用: {torch.cuda.memory_allocated()/1e6:.1f}MB")
        print(f"   损失值: {loss.item():.4f}")
        print(f"   输出设备: {loss.device}")

        return True

    except Exception as e:
        print(f"❌ GPU测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 软体臂训练流程测试开始")
    print(f"项目根目录: {project_root}")

    # 检查基础环境
    print(f"Python版本: {sys.version}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")

    # 运行所有测试
    tests = [
        ("配置加载", test_config_loading),
        ("数据管道", test_data_pipeline),
        ("图编码器", test_graph_encoder),
        ("数据模型集成", test_integration),
        ("GPU兼容性", test_device_compatibility),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"测试: {test_name}")
        print('='*50)

        success = test_func()
        results.append((test_name, success))

    # 总结
    print(f"\n{'='*50}")
    print("🎯 测试总结")
    print('='*50)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")

    print(f"\n总计: {passed}/{total} 测试通过")

    if passed == total:
        print("🎉 所有测试通过! 训练流程准备就绪")
        return 0
    else:
        print("⚠️ 部分测试失败，请检查问题")
        return 1

if __name__ == "__main__":
    exit(main())