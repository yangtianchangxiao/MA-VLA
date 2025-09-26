# LIBERO评估环境配置 TODO

## 🎯 目标
为VLA模型创建LIBERO标准化评估环境，实现模型在标准benchmark上的性能测试。

## 📋 任务清单

### 1. 环境创建
- [ ] 创建Python 3.8.13 conda环境
  ```bash
  conda create -n libero_eval python=3.8.13
  conda activate libero_eval
  ```

### 2. PyTorch安装 (LIBERO要求)
- [ ] 安装指定版本PyTorch
  ```bash
  pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 \
    --extra-index-url https://download.pytorch.org/whl/cu113
  ```

### 3. LIBERO安装
- [ ] 安装robosuite依赖
  ```bash
  pip install robosuite
  ```
- [ ] 克隆并安装LIBERO
  ```bash
  git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
  cd LIBERO
  pip install -r requirements.txt
  pip install -e .
  ```

### 4. RynnVLA依赖兼容性配置
- [ ] 安装PyTorch 1.11兼容版本的依赖
  ```bash
  # 基于/home/cx/AET_FOR_RL/vla/参考模型/RynnVLA-001/requirements.txt
  # 但需要调整版本以兼容PyTorch 1.11
  pip install transformers==4.21.0    # 替代4.43.0
  pip install safetensors==0.2.8      # 替代0.4.2  
  pip install accelerate==0.20.0      # 替代0.33.0
  # 其他依赖保持原版本或测试兼容性
  ```

### 5. VLA模型兼容性测试
- [ ] 测试我们的VLA模型在PyTorch 1.11环境中的加载
  ```python
  # 测试加载trained model: /home/cx/AET_FOR_RL/vla/train/vla_model_trained.pth
  from vla_model import RealRynnVLALoRAGNN
  model = RealRynnVLALoRAGNN()
  # 测试是否能正常加载和推理
  ```

### 6. VLA→LIBERO适配器开发
- [ ] 创建适配器类
  ```python
  class VLAtoLIBEROAdapter:
      def __init__(self, vla_model_path):
          # 加载我们的VLA模型
      def step(self, obs, task_description):
          # 转换LIBERO观测→VLA输入格式
          # 输出7-DOF关节控制信号
  ```

### 7. 标准化评估
- [ ] 在LIBERO benchmark任务上测试VLA模型
- [ ] 对比其他VLA模型性能
- [ ] 生成评估报告

## ⚠️ 潜在问题
1. **版本兼容性**: PyTorch 1.11可能不支持我们用PyTorch 2.5训练的模型特性
2. **依赖冲突**: transformers/safetensors等可能在老版本PyTorch下有限制
3. **CUDA版本**: cu113 vs cu121可能需要重新编译某些包

## 📅 执行时机
- 等Link scaling synthesis完成
- VLA训练完成后
- 有完整的trained model后再进行评估环境配置

## 📂 相关文件
- VLA模型: `/home/cx/AET_FOR_RL/vla/train/vla_model_trained.pth`
- RynnVLA要求: `/home/cx/AET_FOR_RL/vla/参考模型/RynnVLA-001/requirements.txt`
- 训练代码: `/home/cx/AET_FOR_RL/vla/train/`

---
**创建时间**: 2025-09-12  
**状态**: 等待VLA训练完成后执行