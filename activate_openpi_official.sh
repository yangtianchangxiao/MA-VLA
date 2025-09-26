#!/bin/bash
# OpenPi官方环境激活脚本

OPENPI_DIR="/home/cx/AET_FOR_RL/vla/参考模型/openpi"
cd "$OPENPI_DIR"

echo "🚀 OpenPi 0.5 环境激活"
echo "   工作目录: $(pwd)"
echo "   使用UV虚拟环境"

# 设置环境变量
export PATH="$HOME/.local/bin:$PATH"
export OPENPI_DATA_HOME=~/.cache/openpi
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

echo "   数据缓存: $OPENPI_DATA_HOME (12GB已下载)"
echo "   可用GPU: $CUDA_VISIBLE_DEVICES"
echo ""

echo "💡 使用方法:"
echo "   运行命令: uv run python <script.py>"
echo "   安装包: uv pip install <package>"
echo "   查看环境: uv pip list"
echo ""

echo "🎯 训练命令示例:"
echo "   单GPU: uv run python scripts/train_pytorch.py <config>"
echo "   8GPU:  uv run torchrun --nproc_per_node=8 scripts/train_pytorch.py <config>"
echo ""

echo "📋 可用配置:"
uv run python -c "from openpi.training import config; print('   pi05_droid: π₀.₅ DROID微调模型')"

echo ""
echo "✅ 已下载模型:"
echo "   π₀.₅ Base: gs://openpi-assets/checkpoints/pi05_base"
echo "   π₀.₅ DROID: gs://openpi-assets/checkpoints/pi05_droid"
echo ""
echo "🎯 下一步: 创建软体臂数据适配器和训练脚本"