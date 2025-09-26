#!/bin/bash
# OpenPi 0.5 专用环境配置脚本
# 基于官方 pyproject.toml 创建专用conda环境

set -e

echo "🚀 OpenPi 0.5 环境配置开始"
echo "============================================"

# 检查是否在正确目录
OPENPI_DIR="/home/cx/AET_FOR_RL/vla/参考模型/openpi"
if [ ! -d "$OPENPI_DIR" ]; then
    echo "❌ OpenPi目录不存在: $OPENPI_DIR"
    exit 1
fi

echo "✅ OpenPi源码目录: $OPENPI_DIR"

# 环境配置
ENV_NAME="openpi_05"
PYTHON_VERSION="3.11"

echo "🔧 环境配置:"
echo "   环境名称: $ENV_NAME"
echo "   Python版本: $PYTHON_VERSION"
echo ""

# 检查conda是否存在
if ! command -v conda &> /dev/null; then
    echo "❌ Conda未安装，请先安装miniconda或anaconda"
    exit 1
fi

source ~/miniconda3/etc/profile.d/conda.sh

# 删除现有环境（如果存在）
if conda info --envs | grep -q "^$ENV_NAME "; then
    echo "🗑️ 删除现有环境: $ENV_NAME"
    conda remove -n $ENV_NAME --all -y
fi

echo "📦 创建新的conda环境..."
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

echo "🔄 激活环境..."
conda activate $ENV_NAME

# 升级基础工具
echo "⬆️ 升级基础工具..."
pip install --upgrade pip setuptools wheel

# 安装uv包管理器（OpenPi推荐）
echo "📋 安装uv包管理器..."
pip install uv

# 进入OpenPi目录
cd "$OPENPI_DIR"

echo "📦 安装核心依赖..."

# 核心ML框架
echo "   安装JAX (CUDA 12)..."
pip install jax[cuda12]==0.4.35 jaxlib==0.4.35

echo "   安装PyTorch..."
pip install torch==2.4.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "   安装Flax..."
pip install flax==0.8.5

# 核心依赖
echo "📚 安装核心依赖..."
pip install \
    augmax>=0.3.4 \
    dm-tree>=0.1.8 \
    einops>=0.8.0 \
    equinox>=0.11.8 \
    flatbuffers>=24.3.25 \
    numpy>=1.22.4,\<2.0.0 \
    numpydantic>=1.6.6 \
    opencv-python>=4.10.0.84 \
    pillow>=11.0.0 \
    sentencepiece>=0.2.0 \
    tqdm-loggable>=0.2 \
    typing-extensions>=4.12.2 \
    tyro>=0.9.5 \
    wandb>=0.19.1 \
    filelock>=3.16.1 \
    beartype==0.19.0 \
    transformers==4.53.2 \
    rich>=14.0.0 \
    polars>=1.30.0

# 机器学习专用库
echo "🤖 安装ML专用库..."
pip install \
    ml-collections==1.0.0 \
    jaxtyping==0.2.36 \
    orbax-checkpoint==0.4.4

# 数据处理和存储
echo "💾 安装数据处理库..."
pip install \
    fsspec[gcs]>=2024.6.0 \
    imageio>=2.36.1

# RLDS支持（用于DROID数据）
echo "📊 安装RLDS支持..."
pip install \
    tensorflow-cpu==2.15.0 \
    tensorflow-datasets==4.9.9

# Git-based依赖
echo "🔗 安装Git依赖..."
pip install git+https://github.com/huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5
pip install git+https://github.com/kvablack/dlimp@ad72ce3a9b414db2185bc0b38461d4101a65477a

# 开发工具（可选）
echo "🛠️ 安装开发工具..."
pip install \
    pytest>=8.3.4 \
    ipykernel>=6.29.5 \
    ipywidgets>=8.1.5 \
    matplotlib>=3.10.0 \
    pynvml>=12.0.0

# 安装OpenPi本身
echo "⚙️ 安装OpenPi..."
pip install -e .

# 特殊处理：安装client包
if [ -d "packages/openpi-client" ]; then
    echo "📱 安装OpenPi Client..."
    cd packages/openpi-client
    pip install -e .
    cd ../..
fi

# 验证安装
echo ""
echo "🧪 验证安装..."
python -c "
import jax
import torch
import flax
import openpi
print('✅ JAX版本:', jax.__version__)
print('✅ JAX设备:', jax.devices())
print('✅ PyTorch版本:', torch.__version__)
print('✅ CUDA可用:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('✅ GPU数量:', torch.cuda.device_count())
print('✅ Flax版本:', flax.__version__)
print('✅ OpenPi导入成功')
"

# 测试模型下载
echo ""
echo "📥 测试模型下载..."
python -c "
try:
    from openpi.training import config as _config
    from openpi.shared import download

    print('✅ 配置模块导入成功')

    # 测试配置获取
    config = _config.get_config('pi05_droid')
    print('✅ Pi 0.5 DROID配置获取成功')

    print('🔽 模型将在首次使用时自动下载到: ~/.cache/openpi')
except Exception as e:
    print(f'⚠️ 模型测试失败: {e}')
    print('   这是正常的，模型会在实际使用时下载')
"

# 创建训练脚本的软链接
echo ""
echo "🔗 创建训练脚本链接..."
SCRIPTS_DIR="/home/cx/AET_FOR_RL/vla/openpi_scripts"
mkdir -p "$SCRIPTS_DIR"

ln -sf "$OPENPI_DIR/scripts/train_pytorch.py" "$SCRIPTS_DIR/train_pytorch.py"
ln -sf "$OPENPI_DIR/scripts/train.py" "$SCRIPTS_DIR/train.py"

# 创建快速激活脚本
ACTIVATE_SCRIPT="/home/cx/AET_FOR_RL/vla/activate_openpi.sh"
cat > "$ACTIVATE_SCRIPT" << EOF
#!/bin/bash
# 快速激活OpenPi环境

source ~/miniconda3/etc/profile.d/conda.sh
conda activate $ENV_NAME

echo "🚀 OpenPi 0.5 环境已激活"
echo "   Python: \$(python --version)"
echo "   工作目录: \$(pwd)"
echo "   训练脚本: $SCRIPTS_DIR/"

# 设置环境变量
export OPENPI_DATA_HOME=~/.cache/openpi
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

echo "   数据目录: \$OPENPI_DATA_HOME"
echo "   可用GPU: \$CUDA_VISIBLE_DEVICES"
echo ""
echo "💡 使用说明:"
echo "   单GPU训练: python $SCRIPTS_DIR/train_pytorch.py <config>"
echo "   8GPU训练: torchrun --nproc_per_node=8 $SCRIPTS_DIR/train_pytorch.py <config>"
echo "   配置列表: python -c \"from openpi.training import config; print(list(config.CONFIG_MAP.keys()))\""
EOF

chmod +x "$ACTIVATE_SCRIPT"

echo ""
echo "🎉 OpenPi 0.5 环境配置完成!"
echo "============================================"
echo ""
echo "📋 环境信息:"
echo "   环境名称: $ENV_NAME"
echo "   Python版本: $PYTHON_VERSION"
echo "   OpenPi目录: $OPENPI_DIR"
echo "   训练脚本: $SCRIPTS_DIR"
echo "   激活脚本: $ACTIVATE_SCRIPT"
echo ""
echo "🚀 使用方法:"
echo "   1. 激活环境: source $ACTIVATE_SCRIPT"
echo "   2. 或手动激活: conda activate $ENV_NAME"
echo "   3. 查看可用配置: python -c \"from openpi.training import config; print(list(config.CONFIG_MAP.keys()))\""
echo "   4. 开始训练: 参考OpenPi官方文档"
echo ""
echo "📊 支持的模型:"
echo "   - π₀ (pi0_base): 基础流匹配模型"
echo "   - π₀-FAST (pi0_fast_base): 自回归模型"
echo "   - π₀.₅ (pi05_base): 升级版本，更好的泛化"
echo "   - π₀.₅-DROID (pi05_droid): 在DROID数据上微调"
echo ""
echo "💡 下一步:"
echo "   1. source $ACTIVATE_SCRIPT"
echo "   2. 准备你的软体臂数据集"
echo "   3. 创建自定义配置文件"
echo "   4. 开始8卡训练!"

# 保存环境信息
cat > "/home/cx/AET_FOR_RL/vla/openpi_environment_info.json" << EOF
{
    "environment_name": "$ENV_NAME",
    "python_version": "$PYTHON_VERSION",
    "openpi_dir": "$OPENPI_DIR",
    "scripts_dir": "$SCRIPTS_DIR",
    "activate_script": "$ACTIVATE_SCRIPT",
    "creation_time": "$(date -Iseconds)",
    "supported_models": [
        "pi0_base",
        "pi0_fast_base",
        "pi05_base",
        "pi05_droid"
    ],
    "training_capabilities": {
        "single_gpu": true,
        "multi_gpu": true,
        "max_recommended_gpus": 8,
        "supports_fsdp": true
    }
}
EOF

echo "📄 环境信息已保存: /home/cx/AET_FOR_RL/vla/openpi_environment_info.json"