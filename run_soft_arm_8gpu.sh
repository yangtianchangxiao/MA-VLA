#!/bin/bash
# 软体臂VLA 8卡训练启动脚本
# 基于OpenPi框架的多GPU分布式训练

set -e

echo "🚀 软体臂VLA 8卡训练启动"
echo "========================================"

# 检查CUDA设备
if ! nvidia-smi > /dev/null 2>&1; then
    echo "❌ 没有检测到NVIDIA GPU"
    exit 1
fi

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "✅ 检测到 $GPU_COUNT 张GPU"

if [ $GPU_COUNT -lt 8 ]; then
    echo "⚠️ GPU数量不足8张，将使用所有可用GPU ($GPU_COUNT张)"
    NPROC_PER_NODE=$GPU_COUNT
else
    NPROC_PER_NODE=8
fi

# 环境设置
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4
export TORCH_DISTRIBUTED_DEBUG=DETAIL  # 调试分布式问题

# 激活conda环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate AET_FOR_RL

echo "🔧 环境配置:"
echo "   Conda环境: AET_FOR_RL"
echo "   GPU数量: $NPROC_PER_NODE"
echo "   工作目录: $(pwd)"
echo ""

# 检查数据文件
echo "📊 数据检查:"
SOFT_ARM_DATA="/home/cx/AET_FOR_RL/vla/synthesized_data/soft_arm_4dof_synthesis"
DROID_IMAGES="/home/cx/AET_FOR_RL/vla/valid_original_data/droid_100/extracted_images"

if [ -d "$SOFT_ARM_DATA" ]; then
    SOFT_ARM_COUNT=$(find $SOFT_ARM_DATA -name "joint_trajectory.npz" | wc -l)
    echo "   软体臂数据: ✅ $SOFT_ARM_COUNT 个配置"
else
    echo "   软体臂数据: ❌ 不存在"
    exit 1
fi

if [ -d "$DROID_IMAGES" ]; then
    IMAGE_COUNT=$(find $DROID_IMAGES -name "*.jpg" | wc -l)
    echo "   DROID图像: ✅ $IMAGE_COUNT 张图像"
else
    echo "   DROID图像: ❌ 不存在"
    exit 1
fi

# 创建实验目录
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXP_NAME="soft_arm_${NPROC_PER_NODE}gpu_${TIMESTAMP}"
CHECKPOINT_DIR="./checkpoints/$EXP_NAME"
mkdir -p "$CHECKPOINT_DIR"

echo "   实验名称: $EXP_NAME"
echo "   检查点目录: $CHECKPOINT_DIR"
echo ""

# 训练参数
BATCH_SIZE=4  # 每GPU的batch size
EPOCHS=20
LEARNING_RATE=3e-4

echo "📋 训练参数:"
echo "   每GPU批量大小: $BATCH_SIZE"
echo "   总批量大小: $((BATCH_SIZE * NPROC_PER_NODE))"
echo "   训练轮数: $EPOCHS"
echo "   学习率: $LEARNING_RATE"
echo ""

# 测试数据加载（可选）
if [ "$1" = "--test-data" ]; then
    echo "🧪 测试数据加载..."
    python openpi_soft_arm_adapter.py
    echo "数据测试完成，按任意键继续训练或Ctrl+C退出..."
    read -n 1
fi

echo "🚀 开始多GPU训练..."
echo "========================================"

# 启动分布式训练
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$NPROC_PER_NODE \
    train_soft_arm_openpi.py \
    --exp_name "$EXP_NAME" \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LEARNING_RATE \
    --save_dir "./checkpoints"

echo ""
echo "🎉 训练完成!"
echo "========================================"
echo "📁 检查点位置: $CHECKPOINT_DIR"
echo "📊 查看训练日志: wandb项目 'soft-arm-vla'"

# 显示最佳模型信息
if [ -f "$CHECKPOINT_DIR/best_model.pth" ]; then
    echo "✅ 最佳模型已保存"
    MODEL_SIZE=$(du -h "$CHECKPOINT_DIR/best_model.pth" | cut -f1)
    echo "   模型大小: $MODEL_SIZE"

    # 显示模型信息
    python -c "
import torch
checkpoint = torch.load('$CHECKPOINT_DIR/best_model.pth', map_location='cpu')
print(f'   最佳损失: {checkpoint[\"loss\"]:.6f}')
print(f'   训练轮数: {checkpoint[\"epoch\"]+1}')
print(f'   模型配置: {checkpoint[\"config\"]}')
"
else
    echo "⚠️ 未找到最佳模型文件"
fi

echo ""
echo "💡 下一步:"
echo "   1. 评估模型: python evaluate_soft_arm_model.py --model $CHECKPOINT_DIR/best_model.pth"
echo "   2. 可视化结果: tensorboard --logdir runs/"
echo "   3. 生成预测: python predict_soft_arm.py --model $CHECKPOINT_DIR/best_model.pth"