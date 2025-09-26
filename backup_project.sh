#!/bin/bash
# 项目备份脚本

echo "🚀 OpenPi软体臂VLA项目备份"
echo "=========================="

PROJECT_DIR="openpi_soft_arm_training"
BACKUP_DIR="vla_backup_$(date +%Y%m%d_%H%M%S)"

echo "📁 创建备份目录: $BACKUP_DIR"
mkdir -p "$BACKUP_DIR"

echo "📋 备份核心代码文件..."
# 复制核心代码和配置
rsync -av --exclude='.venv' \
         --exclude='__pycache__' \
         --exclude='*.pyc' \
         --exclude='checkpoints' \
         --exclude='debug_checkpoints' \
         --exclude='outputs' \
         --exclude='logs' \
         --exclude='data/processed' \
         "$PROJECT_DIR/" "$BACKUP_DIR/"

echo "📊 备份统计:"
echo "  - 备份大小: $(du -sh $BACKUP_DIR | cut -f1)"
echo "  - Python文件: $(find $BACKUP_DIR -name "*.py" | wc -l)"
echo "  - 配置文件: $(find $BACKUP_DIR -name "*.yaml" -o -name "*.json" | wc -l)"

echo "📦 创建压缩包..."
tar -czf "${BACKUP_DIR}.tar.gz" "$BACKUP_DIR"
echo "  - 压缩包: ${BACKUP_DIR}.tar.gz ($(du -sh ${BACKUP_DIR}.tar.gz | cut -f1))"

echo "✅ 备份完成！"
echo ""
echo "🔗 推荐上传方式:"
echo "  1. GitHub (推荐): git init && git add . && git commit -m 'Initial commit'"
echo "  2. 阿里云OSS: 上传 ${BACKUP_DIR}.tar.gz"
echo "  3. 百度网盘: 上传 ${BACKUP_DIR}.tar.gz"
echo ""
echo "🚀 新平台恢复命令:"
echo "  tar -xzf ${BACKUP_DIR}.tar.gz"
echo "  cd ${BACKUP_DIR}"
echo "  # 重新安装依赖: pip install -r requirements.txt"