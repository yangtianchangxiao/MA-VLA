#!/bin/bash
# 软体臂OpenPi训练启动脚本

echo "🚀 启动软体臂OpenPi训练"
echo "=================================="

# 激活OpenPi环境
OPENPI_DIR="/home/cx/AET_FOR_RL/vla/参考模型/openpi"
cd "$OPENPI_DIR"

# 设置环境变量
export PATH="$HOME/.local/bin:$PATH"
export OPENPI_DATA_HOME=~/.cache/openpi
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

echo "✅ 环境配置:"
echo "   OpenPi目录: $OPENPI_DIR"
echo "   数据缓存: $OPENPI_DATA_HOME"
echo "   可用GPU: $CUDA_VISIBLE_DEVICES"
echo "   UV版本: $(uv --version)"

# 检查GPU状态
echo ""
echo "📊 GPU状态检查:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits

echo ""
echo "🎯 训练选项:"
echo "   1) 单GPU训练 (GPU 0)"
echo "   2) 8GPU分布式训练"
echo "   3) 测试数据加载器"
echo "   4) 验证环境配置"

read -p "请选择 (1-4): " choice

case $choice in
    1)
        echo "🔧 启动单GPU训练..."
        export CUDA_VISIBLE_DEVICES=0
        uv run python /home/cx/AET_FOR_RL/vla/train_soft_arm_openpi_8gpu.py
        ;;
    2)
        echo "🚄 启动8GPU分布式训练..."
        uv run torchrun --nproc_per_node=8 /home/cx/AET_FOR_RL/vla/train_soft_arm_openpi_8gpu.py
        ;;
    3)
        echo "🧪 测试数据加载器..."
        uv run python /home/cx/AET_FOR_RL/vla/openpi_soft_arm_dataloader.py
        ;;
    4)
        echo "✅ 验证环境配置..."
        uv run python -c "
import torch
import sys
sys.path.append('/home/cx/AET_FOR_RL/vla/参考模型/openpi')

print('🔍 环境验证:')
print(f'   PyTorch: {torch.__version__}')
print(f'   CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'   GPU数量: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'   GPU {i}: {torch.cuda.get_device_name(i)}')

try:
    import openpi
    print('✅ OpenPi导入成功')

    from openpi.training import config
    available_configs = list(config.CONFIG_MAP.keys())
    print(f'✅ 可用配置: {available_configs}')

    if 'pi05_droid' in available_configs:
        droid_config = config.get_config('pi05_droid')
        print('✅ DROID配置加载成功')

except Exception as e:
    print(f'❌ OpenPi验证失败: {e}')

# 测试软体臂数据
try:
    sys.path.append('/home/cx/AET_FOR_RL/vla')
    from openpi_soft_arm_dataloader import SoftArmOpenPiDataset

    dataset = SoftArmOpenPiDataset()
    print(f'✅ 软体臂数据集: {len(dataset)} 样本')

    if len(dataset) > 0:
        sample = dataset[0]
        print(f'   样本图像形状: {sample[\"image\"].shape}')
        print(f'   样本动作形状: {sample[\"actions\"].shape}')
        print(f'   机器人配置: {sample[\"robot_config\"]}')

except Exception as e:
    print(f'❌ 数据集验证失败: {e}')
"
        ;;
    *)
        echo "❌ 无效选择"
        exit 1
        ;;
esac

echo ""
echo "✅ 完成!"