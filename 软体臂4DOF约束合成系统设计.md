# 软体臂4DOF约束合成系统设计文档

## 🎯 系统概述

### 核心目标
基于DROID-100数据，生成**4DOF约束**（位置+Z轴法向）的软体臂训练数据，作为现有**3DOF约束**（仅位置）数据的对比研究。

### 设计哲学
遵循Linus "好品味"原则：
- **消除特殊情况**: 统一的数据格式，与3DOF完全兼容
- **实用主义**: 放宽约束标准，提高成功率至70%
- **向后兼容**: 不破坏现有VLA训练管道

## 🏗️ 系统架构

### 约束对比
```
3DOF约束 (现有):     位置 [x,y,z] → 高成功率(91%+)
4DOF约束 (新增):     位置 [x,y,z] + Z轴法向 → 中等成功率(70-80%)
```

### 核心差异：IK求解逻辑
```python
# 3DOF: 仅位置约束
target_world_position = ee_pose[:3]  # 只取[x,y,z]
solution = ik_solver.solve_ik(target_world_position)

# 4DOF: 位置+角度约束 + 放宽fallback
target_world_pose = ee_pose[:6]      # 取[x,y,z,roll,pitch,yaw]
solution = ik_solver.solve_ik(target_world_pose)
if not success:
    solution = ik_solver.solve_ik(target_world_position)  # fallback到3DOF
```

## 📊 数据格式规范

### 完全兼容的数据结构
```python
# joint_trajectory.npz 字段 (与3DOF完全一致)
{
    'joint_positions': (N, 8) float32,        # 软体臂关节角度 [α1,β1,α2,β2,α3,β3,α4,β4]
    'timestamps': (N,) float32,               # 时间戳 [0, 1/30, 2/30, ...]
    'end_effector_positions': (N, 3) float32, # 末端位置 [x,y,z]
    'end_effector_orientations': (N, 3) float32, # 末端角度 [roll,pitch,yaw]
    'success_mask': (N,) bool,                # 成功掩码
    'temporal_smoothness': float64,           # 时间平滑度指标
    'constraint_type': str                    # "4DOF_relaxed" vs "3DOF"
}
```

### 配置文件格式
```json
{
  "episode_id": 1,
  "n_segments": 3,
  "segment_lengths": [0.37, 0.40, 0.37],
  "total_length": 1.14,
  "base_offset": [-0.071, 0.172, 0.0],
  "constraint_type": "4DOF_relaxed",
  "success_threshold": 0.7,
  "synthesis_params": {
    "success_rate": 1.0,
    "temporal_smoothness": 0.0096,
    "original_trajectory_length": 238,
    "synthesized_trajectory_length": 238
  }
}
```

## 🔧 软体臂模型规范

### PCC (Piecewise Constant Curvature) 模型
```python
# 每段软体臂由2个参数控制
α: 弯曲角度 [0, π]     # 弯曲程度
β: 弯曲方向 [0, 2π]    # 弯曲方向

# N段软体臂 = 2N个关节参数
2段: [α1, β1, α2, β2]           → 4 DOF
3段: [α1, β1, α2, β2, α3, β3]   → 6 DOF
4段: [α1, β1, α2, β2, α3, β3, α4, β4] → 8 DOF
5段: [α1, β1, α2, β2, α3, β3, α4, β4, α5, β5] → 10 DOF
```

### 段长度配置
```python
# 经验证有效的段长度组合
segment_configs = {
    2: [0.57, 0.57],                    # 总长1.14m
    3: [0.37, 0.40, 0.37],             # 总长1.14m
    4: [0.28, 0.31, 0.29, 0.33],       # 总长1.21m
    5: [0.23, 0.25, 0.24, 0.26, 0.24]  # 总长1.22m
}
```

## 🎯 4DOF约束策略

### 放宽的成功标准
```python
# 位置精度 (从5cm放宽到8cm)
position_tolerance = 0.08  # 8cm

# 法向精度 (从30°放宽到60°)
normal_tolerance = np.cos(60 * π/180) = 0.5

# 成功率阈值 (从95%降到70%)
success_threshold = 0.7
```

### 分层IK求解策略
```python
def relaxed_solve_ik_hierarchical(target_pos, target_normal=None):
    # 1. 尝试标准4DOF求解
    solution = original_4dof_solve(target_pos, target_normal)

    if not success and target_normal is not None:
        # 2. 验证位置精度
        predicted_pos = forward_kinematics(solution)
        pos_error = norm(predicted_pos - target_pos)

        if pos_error < 0.08:  # 位置可接受
            # 3. 检查法向约束
            predicted_normal = get_z_axis(solution)
            cos_similarity = dot(predicted_normal, target_normal)

            if cos_similarity > 0.5:  # 60°内可接受
                success = True

    return solution, success
```

## 📁 数据存储结构

### 目录组织
```
synthesized_data/soft_arm_4dof_synthesis/
├── episode_000/
│   ├── 2_segments/
│   │   ├── joint_trajectory.npz      # 轨迹数据
│   │   ├── robot_graph.npz           # 图结构
│   │   └── config.json               # 配置信息
│   ├── 3_segments/
│   ├── 4_segments/
│   └── 5_segments/
├── episode_001/
│   └── ...
└── episode_019/
    └── ...
```

### 文件大小估算
```
单个episode (4个配置):
├── joint_trajectory.npz: ~6-8KB
├── robot_graph.npz: ~2KB
└── config.json: ~0.5KB
总计: ~35KB/episode

20个episodes × 4个配置 = ~2.8MB 总数据量
```

## 🎲 机器人图结构生成

### 19维节点特征
```python
# 每个软体臂段的图节点特征
node_features[i] = [
    # joint_type (6D): [rigid_revolute, rigid_prismatic, soft_alpha, soft_beta]
    0, 0, 0, 0, 1, 1,          # 软体关节: alpha + beta

    # axis (3D): 弯曲方向
    0, 1, 0,                   # Y轴弯曲

    # position (3D): 累积位置
    0, 0, cumulative_length,   # Z轴累积长度

    # orientation (4D): 四元数
    0, 0, 0, 1,                # 单位四元数

    # limits (3D): [min_limit, max_limit, segment_length]
    0.001, π, segment_length   # α∈[0,π], β∈[0,2π]
]
```

### 邻接矩阵
```python
# 链式连接结构
edges = [[0,1], [1,2], [2,3], ...]  # 顺序连接
edge_attributes = [[1.0], [1.0], [1.0], ...]  # 连接强度
```

## 📊 对比研究设计

### 数据集对比
| 约束类型 | 成功率 | 姿态控制 | 数据量 | 训练难度 |
|---------|--------|----------|--------|----------|
| **3DOF** | 91%+ | 无 | 多 | 简单 |
| **4DOF** | 70-80% | 有Z轴法向 | 中等 | 复杂 |

### 实验假设
1. **训练效果**: 4DOF数据虽然量少，但姿态信息更丰富
2. **泛化能力**: 4DOF约束训练的模型在姿态任务上更准确
3. **收敛速度**: 3DOF数据多，可能收敛更快
4. **最终性能**: 需要实验验证哪种约束策略更适合VLA训练

### VLA训练兼容性
```python
# 训练管道完全兼容
def load_trajectory_data(data_path):
    data = np.load(data_path)
    return {
        'joint_positions': data['joint_positions'],     # (N, 8)
        'timestamps': data['timestamps'],               # (N,)
        'constraint_type': data.get('constraint_type', '3DOF')  # 自动识别
    }
```

## 🚀 运行流程

### 命令行运行
```bash
# 1. 直接运行 (前台)
./run_4dof_soft_arm_synthesis.sh

# 2. 后台tmux运行
./run_4dof_soft_arm_synthesis.sh --tmux

# 3. 手动运行Python脚本
conda activate AET_FOR_RL
python run_soft_arm_synthesis_4dof.py
```

### 进度监控
```bash
# tmux会话监控
tmux attach -t soft_arm_4dof_synthesis

# 数据生成监控
watch -n 5 "find synthesized_data/soft_arm_4dof_synthesis -name '*.json' | wc -l"
```

## 📈 预期成果

### 生成数据规模
```
输入: 20个DROID episodes
处理: 每个episode × 4种软体臂配置 (2,3,4,5段)
输出: ~60-80个成功配置 (约75%成功率)
```

### 数据质量指标
```python
# 位置精度: 8cm内
position_accuracy < 0.08

# 法向精度: 60°内
normal_error < np.arccos(0.5) = 60°

# 时间连续性: 低平滑度值表示好的连续性
temporal_smoothness < 0.01

# 成功率: 每个配置的轨迹完成率
trajectory_success_rate >= 0.7
```

### 科研价值
1. **约束复杂度研究**: 验证IK约束复杂度对VLA训练的影响
2. **数据效率分析**: 比较高质量少量数据 vs 低约束大量数据
3. **姿态控制评估**: 4DOF训练模型在姿态任务上的优势
4. **软体机器人VLA**: 首个基于真实数据的软体臂VLA研究

## ⚙️ 技术细节

### 核心文件清单
```
run_soft_arm_synthesis_4dof.py       # 主合成脚本
run_4dof_soft_arm_synthesis.sh       # 运行脚本
test_4dof_constraint.py              # 4DOF约束测试脚本
data_augment/morphology_modules/
└── soft_arm_ik_solver.py            # 软体臂IK求解器
```

### 依赖关系
```python
# Python包依赖
numpy >= 1.21.0      # 数值计算
pandas >= 1.3.0      # 数据处理
pathlib             # 路径操作
logging              # 日志记录
json                 # 配置存储

# 自定义模块
soft_arm_ik_solver   # 软体臂运动学
```

### 错误处理机制
```python
# 1. JSON序列化错误修复
config_info = {
    "episode_id": int(episode_id),              # numpy.int64 → int
    "segment_lengths": [float(x) for x in lengths], # numpy.float64 → float
    "base_offset": [float(x) for x in offset.tolist()], # numpy.array → list
}

# 2. IK求解失败处理
if not success_4dof:
    # fallback到3DOF求解
    solution = ik_solver.solve_ik(target_pos_only)

# 3. 异常episode跳过
try:
    results = synthesize_episode_4dof(episode_id, trajectory)
except Exception as e:
    logger.error(f"Episode {episode_id} failed: {e}")
    continue  # 跳过失败的episode，不影响整体进度
```

## 🎉 关键优势

### 1. 向后兼容性
- 与现有3DOF数据格式**100%兼容**
- VLA训练管道**零修改**即可使用
- 可混合训练3DOF+4DOF数据

### 2. 科学对比价值
- 控制变量：只改变IK约束逻辑
- 相同输入：相同DROID episodes和软体臂配置
- 客观指标：成功率、精度、时间连续性

### 3. 实用主义设计
- 放宽约束标准，优化成功率
- fallback机制，避免完全失败
- 渐进增强，不破坏现有功能

### 4. 扩展潜力
- 支持更复杂约束类型 (5DOF, 6DOF)
- 支持更多软体臂配置 (6段，7段)
- 支持自适应约束阈值优化

---

*基于Linus Torvalds "好品味"哲学设计 - 简单、实用、向后兼容*