# 形态学合成程序目录

## 当前可用程序

### 生产级程序
1. **run_complete_morphology_synthesis.py** ✅
   - 4DOF IK + 轴向旋转的完整6DOF形态学合成
   - 输入: length-augmented数据
   - 输出: 多机器人配置的关节轨迹
   - 状态: 生产就绪，98%成功率

### 实验级程序
2. **run_end_effector_synthesis.py**
   - End-effector基础合成
   - 状态: 实验中

3. **run_length_augmented_synthesis.py**
   - Length augmentation预处理
   - 输出到: `synthesized_data/length_augmented_droid/`
   - 状态: 可用

### 调试工具
4. **debug_dh_params.py** - DH参数调试
5. **debug_simple_ik.py** - IK求解调试
6. **test_synthesis.py** - 合成测试

### 工具模块
7. **streaming_data_saver.py** - 数据保存工具

## 已归档程序
- **run_link_scaling_synthesis.py** → `archive/deprecated_synthesis_runners/`
- **run_dof_modification_synthesis.py** → `archive/deprecated_synthesis_runners/`
- **run_random_robot_synthesis.py** → `archive/deprecated_synthesis_runners/`

*原因: 实现逻辑错误，不符合基于原型的相对变化需求*

## 推荐工作流程
```bash
# 1. Length augmentation预处理
python run_length_augmented_synthesis.py

# 2. 完整形态学合成 (4DOF IK + 轴向旋转)
python run_complete_morphology_synthesis.py
```

---
*更新时间: 2025-09-20*
*状态: 清理完成，保留核心程序*