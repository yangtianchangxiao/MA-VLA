# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

**Required Conda Environment:** `conda activate AET_FOR_RL`

This is a Vision-Language-Action (VLA) model project focused on multi-morphology robotics control with end-effector synthesis.

## Key Commands

### Training
```bash
# Train VLA model with morphology data
python train/vla_trainer.py

# Extract DROID images and prepare training data
python extract_all_droid_episodes.py
python create_filtered_morphology_data.py
```

### Data Synthesis
```bash
# End-effector based synthesis (correct approach)
python data_augment/synthesis_runners/run_end_effector_synthesis.py

# Generate robot graphs for GNN
python urdf_to_graph/generate_all_robot_graphs.py

# Legacy morphology synthesis (deprecated)
python data_augment/synthesis_runners/run_link_scaling_synthesis.py
python data_augment/synthesis_runners/run_dof_modification_synthesis.py
```

### Evaluation
```bash
# ManiSkill environment evaluation
python evaluation/maniskill_vla_evaluation.py

# Real DROID data evaluation
python evaluation/real_droid_evaluation.py
```

## Architecture Overview

This project implements a Graph-to-Graph VLA architecture with multi-morphology support:

### Core Components

1. **End-Effector Synthesis** (`data_augment/morphology_modules/end_effector_synthesis_module.py`)
   - **Input**: DROID 7D actions [x,y,z,roll,pitch,yaw,gripper]
   - **Process**: IK synthesis for different robot morphologies
   - **Output**: Joint trajectories [joint_1, ..., joint_n, gripper] for each target robot
   - **Key Insight**: Uses 6DoF end-effector + 1DoF gripper as unified reference trajectory

2. **URDF-to-Graph Conversion** (`urdf_to_graph/`)
   - Converts robot URDF models to 19D node feature graphs for GNN processing
   - Features: joint_type(6D) + axis(3D) + position(3D) + orientation(3D) + limits(4D)
   - Generated graphs stored in `urdf_to_graph/robot_graphs/`

3. **VLA Model** (`train/vla_model.py`)
   - Graph-based multi-agent architecture where each joint is an independent agent
   - Coordinates through GNN to produce adaptive-dimension actions
   - Supports variable DoF robots (5DoF to 9DoF)

4. **Training System** (`train/vla_trainer.py`)
   - Complete VLA training with real DROID images + morphology descriptions
   - Handles task instruction extraction from DROID metadata
   - **Critical**: Only 46/100 DROID episodes have valid task instructions

### Data Flow

```
DROID-100 Dataset → End-Effector Trajectories (6DoF + Gripper)
       ↓
IK Synthesis → Multiple Robot Morphologies (nDoF + Gripper)
       ↓
VLA Training → (Images + Language Instructions + Robot Graphs) → Joint Actions
       ↓
ManiSkill Evaluation → Multi-Morphology Performance Testing
```

## Key Technical Details

### DROID Data Handling
- **Task Instructions**: Located in `/meta/episodes/` not main data files
- **Format**: `tasks` field is numpy.ndarray, take `tasks[0]`
- **Quality Issue**: 54% of episodes have empty task descriptions
- **Valid Episodes**: Only 46/100 episodes have meaningful language instructions

### Morphology Synthesis
- **Correct Approach**: End-effector based synthesis using IK retargeting
- **Legacy Approach**: Joint-space morphology variations (deprecated)
- **Target Robots**: Franka variants, scaled versions (0.8x, 1.2x), reduced DoF (5DoF, 6DoF)

### Graph Neural Network
- Each robot joint represented as graph node with 19D features
- Message passing between joints for coordinated control
- Variable output dimensions based on target robot DoF

## Important Data Locations

- **Original DROID**: `original_data/droid_100/`
- **Synthesized Data**: `synthesized_data/droid_100_morphology/`
- **Robot Graphs**: `urdf_to_graph/robot_graphs/`
- **Training Data**: `training_data/`

## Common Pitfalls

1. **Empty Task Instructions**: Always filter DROID episodes with `if task_str.strip():`
2. **Numpy Array Tasks**: DROID tasks are numpy arrays, use `str(tasks[0])`
3. **IK Convergence**: End-effector synthesis may fail for extreme morphologies
4. **Graph Compatibility**: Robot graphs must match expected 19D node features
5. **Environment Setup**: Must use `AET_FOR_RL` conda environment for all operations

## Project Status

- ✅ **End-effector synthesis**: Implemented and working
- ✅ **Robot graph generation**: 10/15 ManiSkill robots supported
- ✅ **VLA training**: Complete pipeline with real DROID images
- ✅ **Multi-morphology evaluation**: ManiSkill integration ready
- 🚧 **Scale-up**: Processing full DROID-100 dataset (100 episodes → 400+ morphology variations)

## Data Quality Notes

- **Training Scale**: 18,400 samples (46 valid episodes × 8 morphologies × ~50 timesteps)
- **Success Rate**: 93.5% for morphology synthesis with adaptive filtering
- **Image Quality**: Real DROID images, not synthetic renders
- **Language Quality**: Authentic manipulation task descriptions where available

The architecture prioritizes practical robotics applications over theoretical completeness, following a "make it work, then make it better" philosophy.