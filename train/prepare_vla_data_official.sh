#!/bin/bash
# VLA训练数据准备脚本 - 官方DROID数据版本
# 处理官方DROID TFRecord格式数据

set -e  # 遇到错误就退出

echo "🚀 VLA训练数据准备开始 (官方DROID版本)"
echo "================================================"

# 激活conda环境
source /home/cx/miniconda3/etc/profile.d/conda.sh
conda activate AET_FOR_RL

# 设置路径
SCRIPT_DIR="/home/cx/AET_FOR_RL/vla/train"
MODULES_DIR="$SCRIPT_DIR/modules"
OUTPUT_DIR="$SCRIPT_DIR/data"

# 官方DROID数据路径
OFFICIAL_DROID_DIR="/home/cx/AET_FOR_RL/vla/original_data/droid_100"
CONVERTED_DATA_DIR="/home/cx/AET_FOR_RL/vla/converted_data/droid_100_fixed"
VALID_DATA_DIR="/home/cx/AET_FOR_RL/vla/valid_original_data/droid_100"

cd "$SCRIPT_DIR"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"
mkdir -p "$VALID_DATA_DIR"

echo ""
echo "🔍 Phase 1: 检查官方DROID数据"
echo "----------------------------------------"

if [ ! -d "$OFFICIAL_DROID_DIR/1.0.0" ]; then
    echo "❌ 官方DROID数据不存在: $OFFICIAL_DROID_DIR/1.0.0"
    echo "💡 请先下载: gsutil -m cp -r gs://gresearch/robotics/droid_100 $OFFICIAL_DROID_DIR"
    exit 1
fi

TFRECORD_COUNT=$(find "$OFFICIAL_DROID_DIR/1.0.0" -name "*.tfrecord*" | wc -l)
echo "✅ 发现 $TFRECORD_COUNT 个TFRecord文件"

echo ""
echo "📝 Phase 2: 提取任务描述 (TFRecord格式)"
echo "----------------------------------------"

echo "   🔄 从官方TFRecord提取task descriptions..."
python "$MODULES_DIR/task_extractor.py" --output "$VALID_DATA_DIR/task_descriptions.json"

# 检查提取结果
if [ -f "$VALID_DATA_DIR/task_descriptions.json" ]; then
    VALID_EPISODES=$(python -c "
import json
with open('$VALID_DATA_DIR/task_descriptions.json', 'r') as f:
    data = json.load(f)
print(len(data['valid_episode_list']))
")
    echo "✅ 任务描述提取完成: $VALID_EPISODES 个有效episodes"
else
    echo "❌ 任务描述提取失败"
    exit 1
fi

echo ""
echo "🎥 Phase 3: 图像提取 (TFRecord格式)"
echo "----------------------------------------"

echo "   🔄 从TFRecord提取图像数据..."
IMAGES_DIR="$VALID_DATA_DIR/extracted_images"

if [ ! -d "$IMAGES_DIR" ] || [ -z "$(ls -A "$IMAGES_DIR" 2>/dev/null)" ]; then
    echo "   📸 提取图像数据 (3个摄像头视角)..."
    python "$MODULES_DIR/tfrecord_image_extractor.py" \
        --input "$OFFICIAL_DROID_DIR/1.0.0" \
        --output "$IMAGES_DIR" \
        --episodes "$VALID_DATA_DIR/task_descriptions.json" \
        --max-frames 50
else
    echo "   ✅ 图像数据已提取"
fi

# 验证图像提取结果
if [ -f "$IMAGES_DIR/extraction_summary.json" ]; then
    EXTRACTED_EPISODES=$(python -c "
import json
with open('$IMAGES_DIR/extraction_summary.json', 'r') as f:
    data = json.load(f)
print(data['statistics']['successful_episodes'])
")
    TOTAL_FRAMES=$(python -c "
import json
with open('$IMAGES_DIR/extraction_summary.json', 'r') as f:
    data = json.load(f)
print(data['statistics']['total_frames_extracted'])
")
    echo "✅ 图像提取完成: $EXTRACTED_EPISODES episodes, $TOTAL_FRAMES 帧"
else
    echo "⚠️ 图像提取可能有问题，但可继续数据转换"
fi

echo ""
echo "🔄 Phase 4: TFRecord → Parquet 转换"
echo "----------------------------------------"

echo "   🔄 使用简化解析器提取真实TFRecord数据..."

if [ ! -d "$CONVERTED_DATA_DIR" ]; then
    echo "   📊 提取真实DROID轨迹数据..."
    python "$MODULES_DIR/simple_droid_parser.py" \
        --input "$OFFICIAL_DROID_DIR/1.0.0" \
        --output "$CONVERTED_DATA_DIR"
else
    echo "   ✅ 转换后的数据已存在"
fi

# 验证转换结果
if [ -f "$CONVERTED_DATA_DIR/data/chunk-000/file-000.parquet" ]; then
    TIMESTEP_COUNT=$(python -c "
import pandas as pd
df = pd.read_parquet('$CONVERTED_DATA_DIR/data/chunk-000/file-000.parquet')
print(len(df))
")
    echo "✅ Parquet转换完成: $TIMESTEP_COUNT timesteps"

    # 验证关键字段
    HAS_CARTESIAN=$(python -c "
import pandas as pd
df = pd.read_parquet('$CONVERTED_DATA_DIR/data/chunk-000/file-000.parquet')
print('observation.cartesian_position' in df.columns)
")

    if [ "$HAS_CARTESIAN" = "True" ]; then
        echo "✅ 关键字段验证通过: observation.cartesian_position"
    else
        echo "❌ 缺少关键字段: observation.cartesian_position"
        exit 1
    fi
else
    echo "❌ Parquet转换失败"
    exit 1
fi

echo ""
echo "🎯 Phase 5: 数据验证与统计"
echo "----------------------------------------"

# 统计数据
TOTAL_EPISODES=$(python -c "
import pandas as pd
df = pd.read_parquet('$CONVERTED_DATA_DIR/data/chunk-000/file-000.parquet')
print(df['episode_index'].nunique())
")

EPISODES_WITH_CARTESIAN=$(python -c "
import pandas as pd
df = pd.read_parquet('$CONVERTED_DATA_DIR/data/chunk-000/file-000.parquet')
valid_df = df[df['observation.cartesian_position'].notna()]
print(valid_df['episode_index'].nunique())
")

echo "📊 数据统计:"
echo "   总episodes: $TOTAL_EPISODES"
echo "   有效episodes (task descriptions): $VALID_EPISODES"
echo "   有cartesian_position的episodes: $EPISODES_WITH_CARTESIAN"
echo "   总timesteps: $TIMESTEP_COUNT"

echo ""
echo "📋 Phase 6: 生成验证报告"
echo "----------------------------------------"

# 创建综合验证报告
VALIDATION_REPORT="$OUTPUT_DIR/droid_official_validation_report.json"

python -c "
import json
import pandas as pd
from datetime import datetime

# 读取数据
with open('$VALID_DATA_DIR/task_descriptions.json', 'r') as f:
    task_data = json.load(f)

df = pd.read_parquet('$CONVERTED_DATA_DIR/data/chunk-000/file-000.parquet')

# 生成验证报告
report = {
    'validation_timestamp': datetime.now().isoformat(),
    'data_source': 'Official DROID TFRecord',
    'conversion_status': 'completed',
    'statistics': {
        'total_episodes': int(df['episode_index'].nunique()),
        'total_timesteps': len(df),
        'valid_task_episodes': len(task_data['valid_episode_list']),
        'episodes_with_cartesian_position': int(df[df['observation.cartesian_position'].notna()]['episode_index'].nunique())
    },
    'data_quality': {
        'has_task_descriptions': True,
        'has_cartesian_position': 'observation.cartesian_position' in df.columns,
        'has_joint_states': 'observation.state' in df.columns,
        'has_actions': 'action' in df.columns
    },
    'file_paths': {
        'task_descriptions': '$VALID_DATA_DIR/task_descriptions.json',
        'converted_parquet': '$CONVERTED_DATA_DIR/data/chunk-000/file-000.parquet',
        'episode_metadata': '$CONVERTED_DATA_DIR/meta/episodes/chunk-000/file-000.parquet'
    },
    'ready_for_synthesis': True
}

with open('$VALIDATION_REPORT', 'w') as f:
    json.dump(report, f, indent=2)

print('Validation report saved to: $VALIDATION_REPORT')
"

# 创建训练准备就绪标志
READY_STATUS="$OUTPUT_DIR/training_ready.json"
echo "{
    \"status\": \"ready\",
    \"data_source\": \"official_droid_tfrecord\",
    \"usable_episodes\": $VALID_EPISODES,
    \"total_timesteps\": $TIMESTEP_COUNT,
    \"timestamp\": \"$(date)\",
    \"synthesis_ready\": true
}" > "$READY_STATUS"

echo "✅ 训练准备状态已保存: $READY_STATUS"

echo ""
echo "🎉 VLA训练数据准备完成!"
echo "========================================="
echo ""
echo "📊 数据概览:"
echo "   官方DROID数据: $OFFICIAL_DROID_DIR/1.0.0"
echo "   任务描述: $VALID_DATA_DIR/task_descriptions.json"
echo "   转换后数据: $CONVERTED_DATA_DIR"
echo "   验证报告: $VALIDATION_REPORT"
echo "   准备状态: $READY_STATUS"
echo ""
echo "🚀 可以进行的下一步操作:"
echo "   1. 运行morphology synthesis: cd data_augment/synthesis_runners"
echo "   2. Link scaling: python run_link_scaling_synthesis.py"
echo "   3. DOF modification: python run_dof_modification_synthesis.py"
echo "   4. 训练VLA模型: python vla_trainer.py"