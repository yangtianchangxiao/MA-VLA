#!/usr/bin/env python3
"""
OpenPi软体臂训练配置
基于π₀.₅模型，适配软体臂数据和8卡训练
"""

import dataclasses
from typing import Dict, Any, Optional
import ml_collections

from openpi.training.config import TrainConfig, DataConfig, ModelConfig, OptimizerConfig

def get_soft_arm_config() -> ml_collections.ConfigDict:
    """获取软体臂专用的OpenPi训练配置"""

    # 基于pi05_droid配置进行修改
    config = ml_collections.ConfigDict({
        # 实验配置
        'exp_name': 'soft_arm_pi05',
        'description': 'π₀.₅ model fine-tuned on soft continuum arm data',

        # 模型配置 - 基于π₀.₅
        'model_config': ml_collections.ConfigDict({
            'model_name': 'pi05',
            'pretrained_checkpoint': 'gs://openpi-assets/checkpoints/pi05_base',

            # 视觉编码器
            'vision_encoder': ml_collections.ConfigDict({
                'image_size': 224,
                'patch_size': 16,
                'embed_dim': 768,
                'num_layers': 12,
                'num_heads': 12,
            }),

            # 语言编码器
            'language_encoder': ml_collections.ConfigDict({
                'vocab_size': 32000,
                'embed_dim': 768,
                'num_layers': 12,
                'num_heads': 12,
                'max_seq_len': 128,
            }),

            # 动作解码器 - 适配软体臂
            'action_decoder': ml_collections.ConfigDict({
                'action_dim': 10,  # 最大5段×2参数 = 10维
                'chunk_size': 16,  # 动作预测长度
                'use_flow_matching': True,  # π₀.₅特性
                'num_flow_steps': 10,
            }),

            # 软体臂专用配置
            'soft_arm_config': ml_collections.ConfigDict({
                'max_segments': 5,
                'min_segments': 2,
                'joint_type': 'continuous',  # 连续体关节
                'constraint_types': ['3DOF', '4DOF'],  # 支持的约束类型
            }),
        }),

        # 数据配置
        'data_config': ml_collections.ConfigDict({
            'dataset_name': 'soft_arm_synthesis',
            'data_dir': '/home/cx/AET_FOR_RL/vla/synthesized_data',
            'image_dir': '/home/cx/AET_FOR_RL/vla/valid_original_data/droid_100/extracted_images',

            # 数据加载
            'batch_size': 16,  # 每GPU批量大小
            'num_workers': 8,
            'prefetch_factor': 2,
            'pin_memory': True,

            # 数据增强
            'image_augmentation': ml_collections.ConfigDict({
                'random_crop': True,
                'color_jitter': True,
                'horizontal_flip': False,  # 机器人数据不适合翻转
                'normalize': True,
            }),

            # 序列配置
            'sequence_length': 50,
            'action_chunk_size': 16,
            'language_conditioning': True,

            # 软体臂数据专用
            'soft_arm_data': ml_collections.ConfigDict({
                'include_3dof': True,
                'include_4dof': True,
                'constraint_sampling': 'balanced',  # balanced, 3dof_heavy, 4dof_heavy
                'segment_sampling': 'uniform',      # uniform, weighted
            }),
        }),

        # 优化器配置
        'optimizer_config': ml_collections.ConfigDict({
            'learning_rate': 3e-4,
            'warmup_steps': 1000,
            'weight_decay': 0.01,
            'beta1': 0.9,
            'beta2': 0.95,
            'grad_clip_norm': 1.0,

            # 学习率调度
            'lr_schedule': 'cosine',
            'lr_decay_steps': 50000,
            'min_lr_ratio': 0.1,
        }),

        # 训练配置
        'training_config': ml_collections.ConfigDict({
            'num_epochs': 50,
            'save_interval': 5000,
            'eval_interval': 1000,
            'log_interval': 100,
            'max_steps': 100000,

            # 分布式训练
            'use_fsdp': True,
            'fsdp_devices': 8,  # 8卡训练
            'gradient_accumulation_steps': 1,

            # 混合精度
            'use_amp': True,
            'amp_dtype': 'bfloat16',

            # 正则化
            'dropout_rate': 0.1,
            'layer_norm_eps': 1e-6,
        }),

        # 评估配置
        'eval_config': ml_collections.ConfigDict({
            'eval_batch_size': 8,
            'eval_steps': 100,
            'metrics': ['action_mse', 'action_l1', 'segment_accuracy'],

            # 软体臂专用评估
            'soft_arm_metrics': ml_collections.ConfigDict({
                'position_error_threshold': 0.05,  # 5cm
                'orientation_error_threshold': 0.5,  # cos similarity
                'evaluate_by_segments': True,
                'evaluate_by_constraint': True,
            }),
        }),

        # 检查点配置
        'checkpoint_config': ml_collections.ConfigDict({
            'save_dir': '/home/cx/AET_FOR_RL/vla/checkpoints/soft_arm_pi05',
            'save_best': True,
            'save_latest': True,
            'keep_top_k': 3,
            'monitor_metric': 'eval/action_l1',
            'monitor_mode': 'min',
        }),

        # 日志配置
        'logging_config': ml_collections.ConfigDict({
            'use_wandb': True,
            'wandb_project': 'soft-arm-pi05',
            'wandb_entity': 'your-wandb-entity',
            'log_gradients': False,
            'log_activations': False,

            # 可视化
            'log_images': True,
            'log_videos': False,
            'log_trajectory_plots': True,
        }),

        # 硬件配置
        'hardware_config': ml_collections.ConfigDict({
            'device': 'cuda',
            'num_gpus': 8,
            'gpu_memory_limit': None,  # 使用全部GPU内存
            'num_cpu_threads': 32,
        }),

        # 调试配置
        'debug_config': ml_collections.ConfigDict({
            'debug_mode': False,
            'profile_training': False,
            'detect_anomaly': False,
            'log_device_placement': False,
        }),
    })

    return config

def get_soft_arm_data_config() -> Dict[str, Any]:
    """获取软体臂数据配置"""
    return {
        'data_sources': {
            'soft_arm_4dof': {
                'path': '/home/cx/AET_FOR_RL/vla/synthesized_data/soft_arm_4dof_synthesis',
                'episodes': 20,
                'configs_per_episode': 4,
                'constraint_type': '4DOF_relaxed',
                'weight': 0.3,  # 4DOF数据权重较低（数据少）
            },
            'soft_arm_3dof': {
                'path': '/home/cx/AET_FOR_RL/vla/synthesized_data/soft_arm_morphology_synthesis',
                'episodes': 46,
                'configs_per_episode': 4,
                'constraint_type': '3DOF',
                'weight': 0.7,  # 3DOF数据权重较高（数据多）
            },
        },
        'image_sources': {
            'droid_extracted': {
                'path': '/home/cx/AET_FOR_RL/vla/valid_original_data/droid_100/extracted_images',
                'camera_views': ['exterior_image_1_left', 'exterior_image_2_left', 'wrist_image_left'],
                'preferred_view': 'exterior_image_1_left',
                'fallback_enabled': True,
            },
        },
        'task_descriptions': {
            'path': '/home/cx/AET_FOR_RL/vla/valid_original_data/droid_100/task_descriptions.json',
            'fallback_description': 'Complete the manipulation task with the soft continuum arm',
            'max_length': 128,
        },
    }

def validate_config(config: ml_collections.ConfigDict) -> bool:
    """验证配置的有效性"""
    required_keys = [
        'model_config', 'data_config', 'optimizer_config',
        'training_config', 'checkpoint_config'
    ]

    for key in required_keys:
        if key not in config:
            print(f"❌ 缺少必要配置: {key}")
            return False

    # 检查GPU数量
    if config.training_config.fsdp_devices > 8:
        print("⚠️ GPU数量超过8张，请调整fsdp_devices")
        return False

    # 检查批量大小
    total_batch_size = config.data_config.batch_size * config.training_config.fsdp_devices
    if total_batch_size > 128:
        print(f"⚠️ 总批量大小 {total_batch_size} 可能过大，建议调整")

    # 检查数据路径
    import os
    data_paths = [
        config.data_config.data_dir,
        config.data_config.image_dir,
    ]

    for path in data_paths:
        if not os.path.exists(path):
            print(f"❌ 数据路径不存在: {path}")
            return False

    print("✅ 配置验证通过")
    return True

def create_training_script_template() -> str:
    """创建训练脚本模板"""
    return '''#!/usr/bin/env python3
"""
OpenPi软体臂训练脚本
基于π₀.₅模型的8卡分布式训练
"""

import os
import sys
sys.path.append('/home/cx/AET_FOR_RL/vla')

from openpi_soft_arm_config import get_soft_arm_config, validate_config
from openpi.training import config as openpi_config

def main():
    # 获取配置
    config = get_soft_arm_config()

    # 验证配置
    if not validate_config(config):
        print("❌ 配置验证失败")
        sys.exit(1)

    print("🚀 开始OpenPi软体臂训练")
    print(f"   模型: {config.model_config.model_name}")
    print(f"   GPU数量: {config.training_config.fsdp_devices}")
    print(f"   总批量大小: {config.data_config.batch_size * config.training_config.fsdp_devices}")

    # 注册配置到OpenPi系统
    openpi_config.CONFIG_MAP['soft_arm_pi05'] = config

    # 导入并运行训练
    from openpi.scripts.train_pytorch import main as train_main
    train_main()

if __name__ == "__main__":
    main()
'''

if __name__ == "__main__":
    # 测试配置
    config = get_soft_arm_config()
    data_config = get_soft_arm_data_config()

    print("🧪 测试软体臂OpenPi配置")
    print(f"✅ 配置创建成功")
    print(f"   模型: {config.model_config.model_name}")
    print(f"   GPU数量: {config.training_config.fsdp_devices}")
    print(f"   批量大小: {config.data_config.batch_size}")
    print(f"   检查点目录: {config.checkpoint_config.save_dir}")

    print(f"✅ 数据配置:")
    for name, source in data_config['data_sources'].items():
        print(f"   {name}: {source['episodes']} episodes, weight={source['weight']}")

    # 验证配置
    validate_config(config)

    # 创建训练脚本
    script_content = create_training_script_template()
    with open('/home/cx/AET_FOR_RL/vla/train_soft_arm_pi05.py', 'w') as f:
        f.write(script_content)

    print("📄 训练脚本已创建: train_soft_arm_pi05.py")