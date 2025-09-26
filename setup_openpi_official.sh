#!/bin/bash
# OpenPi 0.5 官方UV安装脚本
# 完全按照官方README的方法安装

set -e

echo "🚀 OpenPi 0.5 官方安装开始"
echo "============================================"

OPENPI_DIR="/home/cx/AET_FOR_RL/vla/参考模型/openpi"
WORK_DIR="/home/cx/AET_FOR_RL/vla"

echo "✅ OpenPi源码目录: $OPENPI_DIR"
echo "✅ 工作目录: $WORK_DIR"

# 检查UV是否安装
if ! command -v uv &> /dev/null; then
    echo "📦 安装UV包管理器..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source ~/.cargo/env
    export PATH="$HOME/.cargo/bin:$PATH"
fi

echo "✅ UV版本: $(uv --version)"

# 进入OpenPi目录
cd "$OPENPI_DIR"

# 检查submodules
echo "🔄 更新git submodules..."
git submodule update --init --recursive

echo "📦 使用UV安装OpenPi环境..."
echo "   这可能需要几分钟..."

# 按官方README安装
export GIT_LFS_SKIP_SMUDGE=1
uv sync

echo "⚙️ 安装OpenPi包..."
uv pip install -e .

echo "🧪 验证安装..."
# 使用UV运行验证脚本
uv run python -c "
import sys
print('✅ Python版本:', sys.version)

try:
    import jax
    print('✅ JAX版本:', jax.__version__)
    print('✅ JAX设备:', jax.devices())
except Exception as e:
    print('⚠️ JAX导入失败:', e)

try:
    import torch
    print('✅ PyTorch版本:', torch.__version__)
    print('✅ CUDA可用:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('✅ GPU数量:', torch.cuda.device_count())
except Exception as e:
    print('⚠️ PyTorch导入失败:', e)

try:
    import openpi
    print('✅ OpenPi导入成功')
except Exception as e:
    print('❌ OpenPi导入失败:', e)
"

echo "📥 测试模型配置和下载..."
uv run python -c "
try:
    from openpi.training import config as _config
    from openpi.shared import download

    print('✅ 配置模块导入成功')

    # 获取所有可用配置
    configs = list(_config.CONFIG_MAP.keys())
    print('✅ 可用配置:', configs)

    # 测试pi05配置
    if 'pi05_droid' in configs:
        config = _config.get_config('pi05_droid')
        print('✅ Pi 0.5 DROID配置获取成功')
        print('   模型路径:', config.pretrained_checkpoint if hasattr(config, 'pretrained_checkpoint') else 'N/A')

    print('\\n🔽 模型将在首次使用时自动下载到: ~/.cache/openpi')

except Exception as e:
    print(f'❌ 配置测试失败: {e}')
    import traceback
    traceback.print_exc()
"

# 创建快速激活脚本
ACTIVATE_SCRIPT="$WORK_DIR/activate_openpi_official.sh"
cat > "$ACTIVATE_SCRIPT" << 'EOF'
#!/bin/bash
# OpenPi官方环境激活脚本

OPENPI_DIR="/home/cx/AET_FOR_RL/vla/参考模型/openpi"
cd "$OPENPI_DIR"

echo "🚀 OpenPi 0.5 环境激活"
echo "   工作目录: $(pwd)"
echo "   使用UV虚拟环境"

# 设置环境变量
export OPENPI_DATA_HOME=~/.cache/openpi
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

echo "   数据缓存: $OPENPI_DATA_HOME"
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
uv run python -c "from openpi.training import config; print('  ', list(config.CONFIG_MAP.keys()))"
EOF

chmod +x "$ACTIVATE_SCRIPT"

# 测试下载π₀.₅模型
echo ""
echo "📥 测试π₀.₅模型下载..."
uv run python -c "
from openpi.shared import download
import os

# 测试下载pi05_base模型
try:
    print('🔽 下载π₀.₅ base模型...')
    checkpoint_dir = download.maybe_download('gs://openpi-assets/checkpoints/pi05_base')
    print(f'✅ 模型下载成功: {checkpoint_dir}')

    # 检查文件大小
    if os.path.exists(checkpoint_dir):
        import subprocess
        result = subprocess.run(['du', '-sh', checkpoint_dir], capture_output=True, text=True)
        if result.returncode == 0:
            print(f'   模型大小: {result.stdout.strip().split()[0]}')

    # 测试pi05_droid模型
    print('🔽 下载π₀.₅ DROID模型...')
    droid_checkpoint = download.maybe_download('gs://openpi-assets/checkpoints/pi05_droid')
    print(f'✅ DROID模型下载成功: {droid_checkpoint}')

except Exception as e:
    print(f'❌ 模型下载失败: {e}')
    print('   这可能是网络问题，模型会在训练时自动重试下载')
"

cd "$WORK_DIR"

echo ""
echo "🎉 OpenPi 0.5 环境配置完成!"
echo "============================================"
echo ""
echo "📋 环境信息:"
echo "   安装方式: UV (官方推荐)"
echo "   OpenPi目录: $OPENPI_DIR"
echo "   激活脚本: $ACTIVATE_SCRIPT"
echo "   数据缓存: ~/.cache/openpi"
echo ""
echo "🚀 使用方法:"
echo "   1. 激活环境: source $ACTIVATE_SCRIPT"
echo "   2. 运行脚本: uv run python <script.py>"
echo "   3. 查看配置: uv run python -c \"from openpi.training import config; print(list(config.CONFIG_MAP.keys()))\""
echo ""
echo "📊 支持的模型 (已下载):"
echo "   - π₀.₅ Base: 基础模型，适合微调"
echo "   - π₀.₅ DROID: 在DROID数据上微调的模型"
echo ""
echo "💡 下一步:"
echo "   1. source $ACTIVATE_SCRIPT"
echo "   2. 创建软体臂数据适配器"
echo "   3. 配置训练参数"
echo "   4. 开始8卡训练!"

# 保存环境信息
cat > "$WORK_DIR/openpi_official_info.json" << EOF
{
    "installation_method": "uv",
    "openpi_dir": "$OPENPI_DIR",
    "activate_script": "$ACTIVATE_SCRIPT",
    "data_cache": "~/.cache/openpi",
    "installation_time": "$(date -Iseconds)",
    "models_downloaded": [
        "pi05_base",
        "pi05_droid"
    ],
    "usage": {
        "run_script": "uv run python <script.py>",
        "install_package": "uv pip install <package>",
        "training_single": "uv run python scripts/train_pytorch.py <config>",
        "training_multi": "uv run torchrun --nproc_per_node=8 scripts/train_pytorch.py <config>"
    }
}
EOF

echo "📄 环境信息已保存: $WORK_DIR/openpi_official_info.json"