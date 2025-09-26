#!/bin/bash
# 4DOF约束软体臂数据合成运行脚本
# 作为3DOF的对比研究数据生成

set -e  # 遇到错误立即退出

# 环境设置
export CONDA_ENV="AET_FOR_RL"
export WORKSPACE="/home/cx/AET_FOR_RL/vla"
export DATA_OUTPUT_DIR="/home/cx/AET_FOR_RL/vla/synthesized_data/soft_arm_4dof_synthesis"

echo "🚀 4DOF软体臂约束合成启动"
echo "工作目录: $WORKSPACE"
echo "输出目录: $DATA_OUTPUT_DIR"
echo "=========================================="

# 检查conda环境
if ! conda info --envs | grep -q "$CONDA_ENV"; then
    echo "❌ 错误: conda环境 '$CONDA_ENV' 不存在"
    exit 1
fi

# 激活环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV

# 检查Python脚本存在
SYNTHESIS_SCRIPT="$WORKSPACE/run_soft_arm_synthesis_4dof.py"
if [ ! -f "$SYNTHESIS_SCRIPT" ]; then
    echo "❌ 错误: 合成脚本不存在: $SYNTHESIS_SCRIPT"
    exit 1
fi

# 检查DROID数据
DROID_DATA="/home/cx/AET_FOR_RL/vla/converted_data/droid_100_fixed/data/chunk-000/file-000.parquet"
if [ ! -f "$DROID_DATA" ]; then
    echo "❌ 错误: DROID数据不存在: $DROID_DATA"
    exit 1
fi

# 创建输出目录
mkdir -p "$DATA_OUTPUT_DIR"

# 进入工作目录
cd "$WORKSPACE"

echo "✅ 环境检查完成，开始4DOF合成..."
echo "预计处理时间: 10-15分钟 (20个episodes)"
echo ""

# 在tmux中运行（如果需要后台运行）
if [ "$1" = "--tmux" ]; then
    echo "📺 在tmux会话中启动合成..."
    tmux new-session -d -s "soft_arm_4dof_synthesis" \
        "cd $WORKSPACE && conda activate $CONDA_ENV && python $SYNTHESIS_SCRIPT"
    echo "✅ tmux会话已启动: soft_arm_4dof_synthesis"
    echo "查看进度: tmux attach -t soft_arm_4dof_synthesis"
else
    # 直接运行
    echo "🔧 直接运行合成脚本..."
    python "$SYNTHESIS_SCRIPT"
fi

echo ""
echo "🎉 4DOF软体臂合成完成!"
echo "数据位置: $DATA_OUTPUT_DIR"
echo "数据格式: NPZ (joint trajectories) + JSON (configs) + NPZ (robot graphs)"

# 生成数据统计
if [ -d "$DATA_OUTPUT_DIR" ]; then
    echo ""
    echo "📊 生成统计:"
    echo "Episodes: $(find $DATA_OUTPUT_DIR -maxdepth 1 -type d -name 'episode_*' | wc -l)"
    echo "Configurations: $(find $DATA_OUTPUT_DIR -name 'config.json' | wc -l)"
    echo "Trajectories: $(find $DATA_OUTPUT_DIR -name 'joint_trajectory.npz' | wc -l)"
    echo "Robot Graphs: $(find $DATA_OUTPUT_DIR -name 'robot_graph.npz' | wc -l)"
    echo ""
    echo "总数据大小: $(du -sh $DATA_OUTPUT_DIR 2>/dev/null | cut -f1)"
fi

echo "=========================================="
echo "💡 使用说明:"
echo "  • 对比研究: 与3DOF数据进行训练效果对比"
echo "  • 数据兼容: 与3DOF数据格式完全兼容"
echo "  • VLA训练: 可直接用于VLA模型训练"