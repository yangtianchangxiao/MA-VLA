# VLA Training System

Vision-Language-Action model with morphology awareness and multi-joint cooperation.

## 🎯 Overview

A complete VLA training pipeline that combines:
- **Real DROID images** (external camera, 320×180 RGB)
- **Morphology-aware descriptions** (text descriptions of robot variations)
- **Graph Neural Networks** (multi-joint cooperation)
- **LoRA adaptation** (parameter-efficient fine-tuning)

## 📁 Project Structure

```
/home/cx/AET_FOR_RL/vla/train/
├── vla_trainer.py              # Main training script
├── vla_model.py                # RynnVLA + LoRA + GNN architecture
├── vla_model_trained.pth       # Trained model (3.14GB)
└── data/                       # Training data (6.0M)
    ├── droid_unified_morphology.json     # 13 augmented episodes
    └── extracted_droid_images/           # 60 real DROID images
        ├── episode_0/ (30 images)
        └── episode_2/ (30 images)
```

## 🚀 Training Results

```
✅ Training completed successfully:
├── Final loss: 0.117099
├── Epochs: 10
├── GPU memory: 3.59GB (stable)
├── Total samples: 390
└── Morphology variations: 8
```

## 🔧 Architecture

**Model**: RynnVLA-7B base + LoRA (rank 32) + GNN
**Parameters**: 440.64M total (LoRA: 0.26M, GNN: 7.35M)
**Training strategy**: Frozen backbone + trainable adaptation layers

## 💡 Key Innovation

**External Camera Insight**: Fixed external cameras provide same images for all morphology variations, eliminating need for image transformation.

## 🧪 Usage

```python
# Train new model
python vla_trainer.py

# Load trained model
import torch
from vla_model import RealRynnVLALoRAGNN

model = RealRynnVLALoRAGNN()
checkpoint = torch.load('vla_model_trained.pth')
model.load_state_dict(checkpoint['model_state_dict'])
```

## 📊 Data

- **Real images**: 60 DROID frames extracted from episodes 0 and 2
- **Morphology data**: 13 augmented variations from 3 original episodes
- **Training samples**: 390 total (4.3x augmentation multiplier)

Built with [RynnVLA-001](https://github.com/GenEmbedded/RynnVLA) and DROID-100 dataset.