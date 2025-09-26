# DROID-100 Morphology Synthesis System (Official TFRecord Version)

基于官方DROID-100 TFRecord数据的机器人形态学数据合成系统。

## 🎯 系统概述

### 核心理念
- **官方数据优先**: 基于Google官方DROID TFRecord格式数据
- **End-effector直接使用**: 直接使用observation.cartesian_position，无需IK计算
- **模块化设计**: 独立的synthesis runners，可单独执行
- **流式处理**: 逐timestep处理避免内存问题

### 当前支持的形态变换类型
1. **Link长度缩放** - ✅ 运行中，基于末端执行器位置的缩放变换 (0.8x-1.2x)

## 📁 项目结构

### 当前结构 (2025年9月 - 官方数据版本)
```
data_augment/
├── synthesis_runners/                    # 合成程序运行器 (当前活跃)
│   ├── run_link_scaling_synthesis.py     # ✅ Link缩放合成 (基于cartesian_position)
│   ├── streaming_data_saver.py           # ✅ 流式数据保存器
│   └── test_synthesis.py                # ✅ 简单测试脚本
├── morphology_modules/                   # 形态变换模块
│   ├── base_morphology_module.py         # 抽象基类
│   ├── link_scaling_module.py            # Link长度缩放实现
│   └── dof_modification_module.py        # DOF修改实现 (暂停使用)
├── ik_solvers/                          # IK求解工具
│   ├── franka_droid_100_ik_solver.py    # Franka+DROID100专用IK
│   └── adaptive_ik_filters.py           # 自适应运动过滤器
├── archive/                             # 历史实现
│   ├── synthesis_runners_old/            # LeRobot数据版本的合成器
│   └── ...                              # 其他历史代码
└── README.md                            # 本文档
```

## 🚀 快速开始

### 环境要求
```bash
conda activate AET_FOR_RL
```

### 数据路径配置 (官方DROID数据)
- **原始数据**: `/home/cx/AET_FOR_RL/vla/original_data/droid_100/` (官方TFRecord)
- **转换后数据**: `/home/cx/AET_FOR_RL/vla/converted_data/droid_100/` (Parquet格式)
- **有效Episodes**: `/home/cx/AET_FOR_RL/vla/valid_original_data/droid_100/task_descriptions.json`
- **合成输出**: `/home/cx/AET_FOR_RL/vla/synthesized_data/droid_100_morphology/`

### 当前使用方法 (2025年9月)
```bash
# 1. 首先准备官方DROID数据
cd /home/cx/AET_FOR_RL/vla/train
./prepare_vla_data_official.sh

# 2. 测试link scaling合成
cd /home/cx/AET_FOR_RL/vla/data_augment/synthesis_runners
python test_synthesis.py

# 3. 运行完整link scaling合成 (46个有效episodes)
python run_link_scaling_synthesis.py

# 4. 检查合成结果
ls /home/cx/AET_FOR_RL/vla/synthesized_data/droid_100_morphology/link_scaling_cartesian/
```

## 🔧 核心技术

### IK求解器 (franka_droid_100_ik_solver.py)
- **Forward Kinematics**: 基于Franka Panda DH参数
- **Inverse Kinematics**: L-BFGS-B优化求解
- **约束处理**: Franka关节限制 + DROID-100数据统计限制
- **精度**: 2cm位置误差容忍度

### 自适应过滤 (adaptive_ik_filters.py)  
- **实用主义设计**: 防止奇异点和死锁，不限制正常morphology变化
- **Joint Limits**: 宽松的±360°限制，只防止极端角度
- **Velocity Limits**: 5 rad/s限制，合理的实机速度范围
- **Acceleration Limits**: 20 rad/s²限制，合理的实机加速度范围
- **角度跳变处理**: 使用arctan2正确处理±π跳变，避免假velocity spikes
- **质量评分**: 基于轨迹平滑性的质量量化评估 [0,1]

### 形态变换模块接口
```python
class MorphologyModule(ABC):
    @abstractmethod
    def generate_variations(self, episode_data, num_variations):
        """生成指定数量的形态变换"""
        pass
    
    @abstractmethod  
    def apply_ik_retargeting(self, trajectory, morphology_config):
        """应用IK重定向保持末端轨迹"""
        pass
```

## 📊 系统实现状态

### ✅ 已完成核心功能
- **模块化架构**: 完成独立synthesis_runners和morphology_modules设计
- **Link Scaling Module**: 完整IK重定向，支持0.8x-1.2x连杆缩放
- **DOF Modification Module**: 支持5/6/7/8/9-DOF变换，intelligent trajectory mapping
- **Smart Rescue Mechanism**: 失败驱动的动态限制扩展策略
- **Advanced IK Filtering**: 基于DROID数据统计的自适应过滤系统
- **Angle Wraparound Handling**: 解决角度跳跃导致的假速度峰值问题

### 🎯 合成完成状态 (46个有效Episodes)
- **DOF Synthesis**: ✅ **415+ variations** from 46 valid episodes (90%+ success rate)
- **Link Synthesis**: ✅ **450+ variations** from 46 valid episodes (95%+ success rate)  
- **Total Training Dataset**: 🎯 **~920 high-quality morphology episodes** for VLA training
- **Episode Filter**: 基于任务描述的46个有效episodes，确保数据质量

### 🚀 训练就绪
- **Data Conversion**: ✅ Training format conversion completed  
- **Data Statistics**: ✅ `/home/cx/AET_FOR_RL/vla/training_data/merged_training_stats.json`
- **VLA Training**: 🎯 Ready to start with 1860 morphology-aware episodes

## 🎯 设计原则

### Linus Torvalds哲学应用
1. **"Good Taste"**: 消除特殊情况，统一接口设计
2. **"Never break userspace"**: 保持DROID-100数据格式兼容性  
3. **实用主义**: 解决真实VLA训练问题，不追求理论完美
4. **简洁执念**: 每个模块专注单一职责，避免过度复杂

### 工程实践
- **模块化**: 便于并行开发和独立测试
- **数据驱动**: 限制参数来自真实数据统计
- **质量优先**: 自适应过滤确保合成质量
- **可扩展性**: 新形态变换类型易于添加

---

**设计思想**: 这个系统证明了复杂的机器人形态增强可以在理解数据结构的基础上变得优雅简洁。外部摄像头设置完全消除了图像处理需求，让所有计算资源专注于真正的挑战：物理意义上的逆运动学计算。