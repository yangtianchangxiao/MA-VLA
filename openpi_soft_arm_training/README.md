# OpenPi Soft Arm Training

专用于软体臂VLA模型训练的独立项目。

## 项目结构

```
openpi_soft_arm_training/
├── data/                  # 数据管理
│   ├── soft_arm/         # 软体臂关节数据
│   ├── images/           # DROID图像数据
│   ├── robot_graphs/     # 机器人图结构
│   └── processed/        # 预处理后数据
├── models/               # 模型实现
│   ├── graph_vla.py     # Graph-based VLA主模型
│   ├── graph_nn.py      # Graph Neural Network
│   └── flow_matching.py # Flow Matching组件
├── configs/              # 训练配置
│   └── soft_arm_config.yaml
├── scripts/              # 训练脚本
│   ├── train.py         # 主训练脚本
│   ├── eval.py          # 评估脚本
│   └── data_prep.py     # 数据预处理
├── outputs/              # 训练输出
├── logs/                # 日志文件
└── checkpoints/         # 模型检查点
```

## 核心架构

基于你提出的Graph-based VLA设计：

1. **图编码**: URDF → Graph (19D节点) → GraphNN → (N×32 tokens)
2. **双路融合**: Attention Pool (全局) + 残差连接 (细粒度)
3. **节点式动作头**: 每个关节独立预测，再组合成完整动作
4. **Flow Matching**: 在拼接后的 [H×DOF] 空间做时序建模

## 数据来源

- 软体臂合成数据: 37,557个训练样本
- DROID-100图像: 6,900张真实机器人图像
- 机器人图: 2-5段软体臂的图结构数据

## 使用方法

```bash
# 激活环境
source /home/cx/AET_FOR_RL/vla/activate_openpi_official.sh

# 预处理数据
python scripts/data_prep.py

# 开始训练
python scripts/train.py --config configs/soft_arm_config.yaml

# 8GPU训练
torchrun --nproc_per_node=8 scripts/train.py --config configs/soft_arm_config.yaml
```