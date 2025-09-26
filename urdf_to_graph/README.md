# URDF to Graph Structure Converter

Convert URDF robot models to graph representations for GNN-based VLA training.

## 🎯 Purpose

This module converts URDF files (like those used in ManiSkill) into graph structures suitable for our Graph-to-Graph VLA architecture. Each robot joint becomes a graph node with rich features, enabling multi-morphology robot understanding.

## 🚀 Quick Start

### Installation
```bash
# Install core dependencies
pip install urdfpy networkx torch

# Optional: Install GNN frameworks
pip install dgl torch-geometric
```

### Basic Usage
```python
from urdf_parser import URDFGraphConverter

# Create converter
converter = URDFGraphConverter()

# Parse URDF to NetworkX graph
graph = converter.parse_urdf_to_networkx("path/to/robot.urdf")

# Convert to GNN formats
dgl_graph = converter.networkx_to_dgl(graph)        # DGL format
pyg_data = converter.networkx_to_pygeometric(graph) # PyTorch Geometric format
```

### Test with ManiSkill Panda
```bash
cd /home/cx/AET_FOR_RL/vla/urdf_to_graph
python urdf_parser.py
```

## 📊 Graph Structure

### Node Features (per joint)
- **Joint Type** (6D): One-hot encoding [revolute, continuous, prismatic, planar, floating, fixed]
- **Axis** (3D): Joint rotation/translation axis
- **Position** (3D): Joint origin xyz coordinates
- **Orientation** (3D): Joint origin rpy angles
- **Limits** (4D): [lower, upper, velocity, effort] limits

**Total: 19D node features**

### Edge Structure
- **Directed edges**: Parent joint → Child joint
- **Edge attributes**: Connection through intermediate link
- **Topology**: Tree structure matching robot kinematic chain

## 🔧 Integration with VLA

### Current GNN Architecture
```python
# In vla_model.py - update SimpleRobotGraph
class EnhancedRobotGraph(nn.Module):
    def __init__(self, node_dim=19, hidden_dim=64):  # Match 19D features
        super().__init__()
        self.node_projection = nn.Linear(node_dim, hidden_dim)
        # ... rest of GNN layers
```

### Usage in Training
```python
# Generate robot graph from URDF
urdf_path = "path/to/robot.urdf"
robot_graph = converter.parse_urdf_to_networkx(urdf_path)
dgl_graph = converter.networkx_to_dgl(robot_graph)

# Feed to VLA model
outputs = vla_model(images, language, dgl_graph)
```

## 📁 Files

- `urdf_parser.py` - Main conversion logic
- `requirements.txt` - Dependencies
- `README.md` - This documentation

## 🎯 Benefits

1. **Universal Compatibility**: Works with any URDF file (ManiSkill, ROS, custom)
2. **Rich Features**: 19D node features capture joint properties and limits
3. **Multiple Formats**: Supports both DGL and PyTorch Geometric
4. **Plug-and-Play**: Drop-in replacement for current graph generation

## 🔄 Next Steps

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Test conversion**: Run `python urdf_parser.py`
3. **Integrate with VLA**: Update `vla_model.py` to use 19D node features
4. **Extend training**: Use URDF graphs instead of manual DH parameters

This replaces the problematic DH parameter approach with a robust URDF-based solution!