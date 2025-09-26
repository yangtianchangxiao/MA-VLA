#!/bin/bash
# 软体臂Graph VLA训练启动脚本
# 符合Linus原则: 一个脚本解决所有训练需求

# 设置环境
export PYTHONPATH="/home/cx/AET_FOR_RL/vla/openpi_soft_arm_training:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"  # 8个GPU
export PATH="$HOME/.local/bin:$PATH"  # UV路径

# 进入OpenPi目录使用UV环境
cd /home/cx/AET_FOR_RL/vla/参考模型/openpi

echo "🚀 软体臂Graph VLA训练启动"
echo "   时间: $(date)"
echo "   GPU数量: $(nvidia-smi -L | wc -l 2>/dev/null || echo '未知')"
echo "   Python路径: $PYTHONPATH"
echo "   UV环境: $(pwd)"

# 检查数据
echo "📊 检查训练数据..."
if [ ! -d "/home/cx/AET_FOR_RL/vla/openpi_soft_arm_training/data/processed" ]; then
    echo "❌ 数据目录不存在，请先运行数据预处理"
    exit 1
fi

echo "✅ 数据检查完成"

# 根据参数选择训练模式
case "$1" in
    "single")
        echo "🖥️  单GPU训练模式 (使用UV环境)"
        uv run python /home/cx/AET_FOR_RL/vla/openpi_soft_arm_training/scripts/train_soft_arm.py
        ;;
    "multi" | "")
        echo "🖥️  8GPU分布式训练模式 (使用UV环境)"
        uv run torchrun \
            --standalone \
            --nnodes=1 \
            --nproc_per_node=8 \
            /home/cx/AET_FOR_RL/vla/openpi_soft_arm_training/scripts/train_soft_arm.py
        ;;
    "debug")
        echo "🐛 调试模式 (单GPU, 小批量, UV环境)"
        CUDA_VISIBLE_DEVICES="0" uv run python /home/cx/AET_FOR_RL/vla/openpi_soft_arm_training/scripts/train_soft_arm.py --config /home/cx/AET_FOR_RL/vla/openpi_soft_arm_training/configs/debug_config.yaml
        ;;
    "resume")
        echo "🔄 恢复训练模式 (使用UV环境)"
        uv run torchrun \
            --standalone \
            --nnodes=1 \
            --nproc_per_node=8 \
            /home/cx/AET_FOR_RL/vla/openpi_soft_arm_training/scripts/train_soft_arm.py --resume
        ;;
    *)
        echo "使用方法:"
        echo "  $0 single    # 单GPU训练"
        echo "  $0 multi     # 8GPU分布式训练 (默认)"
        echo "  $0 debug     # 调试模式"
        echo "  $0 resume    # 恢复训练"
        exit 1
        ;;
esac

echo "🎉 训练脚本执行完成!"