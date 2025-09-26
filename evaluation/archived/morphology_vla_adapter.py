#!/usr/bin/env python3
"""
Adapter to interface our Multi-Morphology VLA model with SOTA benchmarks
Converts our model output format to standard VLA benchmark expectations
"""

import torch
import numpy as np
import sys
sys.path.append('/home/cx/AET_FOR_RL/vla/train')

from vla_model import RealRynnVLALoRAGNN
from vla_trainer import CompleteVLADataset

class MorphologyVLAAdapter:
    """Adapter for our morphology-aware VLA model to work with SOTA benchmarks"""
    
    def __init__(self, model_path="/home/cx/AET_FOR_RL/vla/train/vla_model_trained.pth"):
        print("🔄 Loading Multi-Morphology VLA Model...")
        
        # Load our trained model
        self.model = self._load_model(model_path)
        
        # Morphology capabilities
        self.supported_dofs = [5, 6, 7, 8, 9]
        self.link_scale_range = (0.8, 1.2)
        
        print("✅ Multi-Morphology VLA Model loaded")
    
    def _load_model(self, model_path):
        """Load our trained model with proper architecture"""
        import torch.nn as nn
        from pathlib import Path
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize base model architecture
        base_model = RealRynnVLALoRAGNN(
            model_path="../RynnVLA-001/pretrained_models/RynnVLA-001-7B-Base",
            lora_rank=32,
            gnn_node_dim=256
        )
        
        # Create complete wrapper (same as training)
        class CompleteVLAWrapper(nn.Module):
            """Complete VLA model with real vision + language + morphology + action"""
            
            def __init__(self, base_model):
                super().__init__()
                self.base_model = base_model
                hidden_size = base_model.hidden_size
                
                # Vision encoder for real DROID images (320x180)
                self.vision_encoder = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),      # -> 160x90
                    nn.ReLU(),
                    nn.BatchNorm2d(64),
                    nn.MaxPool2d(2),                                           # -> 80x45
                    nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),    # -> 40x23  
                    nn.ReLU(),
                    nn.BatchNorm2d(128),
                    nn.AdaptiveAvgPool2d((8, 8)),                              # -> 8x8
                    nn.Flatten(),
                    nn.Linear(128 * 8 * 8, 2048),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(2048, hidden_size)
                )
                
                # Language encoder for morphology descriptions
                self.language_encoder = nn.Sequential(
                    nn.Linear(32, 512),   # Token sequence to embedding
                    nn.ReLU(),
                    nn.Linear(512, hidden_size)
                )
                
                # Morphology encoder
                self.morphology_encoder = nn.Sequential(
                    nn.Linear(6, 256),
                    nn.ReLU(),
                    nn.BatchNorm1d(256),
                    nn.Linear(256, 512),
                    nn.ReLU(),
                    nn.Linear(512, hidden_size)
                )
                
                # Multimodal fusion
                self.fusion_attention = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
                self.fusion_norm = nn.LayerNorm(hidden_size)
                self.fusion_mlp = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size * 2),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_size * 2, hidden_size)
                )
            
            def forward(self, images, description_tokens, morphology_features):
                batch_size = images.shape[0]
                
                # 1. Encode modalities (same as training)
                vision_features = self.vision_encoder(images)          # (B, hidden_size)
                language_features = self.language_encoder(description_tokens.float())  # (B, hidden_size)
                morphology_enc = self.morphology_encoder(morphology_features)  # (B, hidden_size)
                
                # 2. Multimodal attention fusion (same as training)
                # Stack modalities for attention
                multimodal_input = torch.stack([vision_features, language_features, morphology_enc], dim=1)  # (B, 3, hidden_size)
                
                # Self-attention across modalities
                attended, _ = self.fusion_attention(multimodal_input, multimodal_input, multimodal_input)
                
                # Residual connection and norm
                fused = self.fusion_norm(attended + multimodal_input)
                
                # MLP and final pooling
                enhanced = self.fusion_mlp(fused)
                final_features = enhanced.mean(dim=1)  # Average across modalities (B, hidden_size)
                
                # 3. Process through base model GNN (same as training)
                return self._process_through_gnn(final_features)
            
            def _process_through_gnn(self, fused_features):
                """Process through LoRA + GNN architecture"""
                batch_size = fused_features.shape[0]
                
                # Apply LoRA adaptation
                adapted = fused_features + self.base_model.lora_projection(fused_features.unsqueeze(1)).squeeze(1)
                
                # Final norm
                normalized = self.base_model.final_norm(adapted.unsqueeze(1)).squeeze(1)
                
                # Convert to joint nodes
                node_features = self.base_model.to_joint_nodes(normalized)
                node_features = node_features.reshape(batch_size * 7, -1)
                
                # GNN processing - use robot_graph instead of gnn_layers
                joint_nodes = node_features.reshape(batch_size, 7, -1)
                updated_nodes = self.base_model.robot_graph(joint_nodes)
                
                # Action decoding - use graph_decoder instead of action_head
                actions = self.base_model.graph_decoder(updated_nodes)
                
                return {'actions': actions}
        
        # Create model instance
        model = CompleteVLAWrapper(base_model)
        
        # Load trained weights
        if Path(model_path).exists():
            print(f"   📥 Loading weights from {model_path}")
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"   ✅ Loaded model with training loss: {checkpoint['loss']:.6f}")
        else:
            print(f"   ⚠️  Model file not found at {model_path}, using untrained weights")
        
        model = model.to(device)
        model.eval()
        
        return model
    
    def predict_action(self, image, instruction, morphology_config=None):
        """
        Standard VLA interface for SOTA benchmarks
        
        Args:
            image: RGB observation image (H, W, C) or (C, H, W) tensor
            instruction: Natural language instruction  
            morphology_config: Our unique morphology specification
        
        Returns:
            action: 7-DOF action vector (adapted for current morphology)
        """
        with torch.no_grad():
            # Process inputs through our multi-morphology model
            if morphology_config is None:
                # Default 7-DOF Franka configuration
                morphology_config = {
                    "dof": 7,
                    "link_scales": [1.0] * 7,
                    "morphology_type": "standard_franka"
                }
            
            # Convert image to proper tensor format
            if isinstance(image, np.ndarray):
                if len(image.shape) == 3 and image.shape[2] == 3:  # (H, W, C)
                    image = torch.from_numpy(image).permute(2, 0, 1).float()  # -> (C, H, W)
                elif len(image.shape) == 3 and image.shape[0] == 3:  # (C, H, W)
                    image = torch.from_numpy(image).float()
                else:
                    raise ValueError(f"Unexpected image shape: {image.shape}")
            elif isinstance(image, torch.Tensor):
                if len(image.shape) == 3 and image.shape[2] == 3:  # (H, W, C)
                    image = image.permute(2, 0, 1).float()  # -> (C, H, W)
                elif len(image.shape) == 3 and image.shape[0] == 3:  # (C, H, W)
                    image = image.float()
                else:
                    raise ValueError(f"Unexpected image tensor shape: {image.shape}")
            
            # Add batch dimension and ensure device placement
            if len(image.shape) == 3:
                image = image.unsqueeze(0)  # Add batch dimension -> (1, C, H, W)
            
            device = next(self.model.parameters()).device
            image = image.to(device)
            
            # Resize image to expected input size (320x180)
            import torch.nn.functional as F
            if image.shape[2:] != (180, 320):
                image = F.interpolate(image, size=(180, 320), mode='bilinear', align_corners=False)
            
            # Normalize image to [0,1] range if needed
            if image.max() > 1.0:
                image = image / 255.0
            
            # Process instruction into tokens (same as training)
            description_tokens = self._tokenize_description(instruction)
            description_tokens = description_tokens.unsqueeze(0).to(device)  # Add batch dimension
            
            # Our model's unique morphology processing (6-dimensional)
            morphology_features = self._encode_morphology(morphology_config)
            morphology_features = morphology_features.unsqueeze(0).to(device)  # Add batch dimension
            
            # Forward pass through GNN VLA (same signature as training)
            output = self.model(
                images=image,
                description_tokens=description_tokens,
                morphology_features=morphology_features
            )
            
            action = output['actions'].cpu().numpy().flatten()
            
            # Adapt action dimensions if needed
            action = self._adapt_action_dimensions(action, morphology_config)
            
            return action
    
    def _tokenize_description(self, description):
        """Simple tokenization for morphology-aware descriptions (same as training)"""
        # Convert to lowercase and limit length
        text = description.lower()[:64]
        # Convert to ASCII codes, limit to reasonable vocab range
        tokens = [min(ord(c), 999) for c in text]
        
        # Pad to fixed length
        max_len = 32
        if len(tokens) < max_len:
            tokens.extend([0] * (max_len - len(tokens)))
        else:
            tokens = tokens[:max_len]
        
        return torch.tensor(tokens, dtype=torch.long)
    
    def _encode_morphology(self, config):
        """Encode morphology configuration for our model (6-dimensional)"""
        # Get link scales (first 3 links are most important for our model)
        link_scales = config.get("link_scales", [1.0] * 7)
        
        # Pad or truncate to ensure we have at least 3 elements
        if len(link_scales) < 3:
            link_scales.extend([1.0] * (3 - len(link_scales)))
        
        # Base position (default zeros)
        base_pos = [0.0, 0.0, 0.0]  # Base position not used in evaluation, set to default
        
        # Create 6D feature vector: [link1_scale, link2_scale, link3_scale, base_x, base_y, base_z_rotation]
        morphology_features = torch.tensor([
            link_scales[0],  # Link 1 scale
            link_scales[1],  # Link 2 scale  
            link_scales[2],  # Link 3 scale
            base_pos[0],     # Base X position (always 0 for evaluation)
            base_pos[1],     # Base Y position (always 0 for evaluation)
            0.0,             # Base Z rotation (always 0 for evaluation)
        ], dtype=torch.float32)
        
        return morphology_features
    
    def _adapt_action_dimensions(self, action, morphology_config):
        """Adapt action dimensions based on morphology"""
        target_dof = morphology_config.get("dof", 7)
        
        if len(action) == target_dof:
            return action
        elif len(action) > target_dof:
            # Reduce dimensions (e.g., 7-DOF -> 6-DOF)
            return action[:target_dof]
        else:
            # Expand dimensions (e.g., 7-DOF -> 8-DOF) 
            expanded = np.zeros(target_dof)
            expanded[:len(action)] = action
            return expanded
    
    def get_model_info(self):
        """Return model information for benchmark reporting"""
        return {
            "name": "RynnVLA-LoRA-GNN-Morphology",
            "architecture": "RynnVLA + LoRA + Graph Neural Networks",
            "parameters": "171.83M trainable / 276.28M total",
            "unique_capabilities": [
                "Multi-morphology awareness",
                "DOF adaptation (5-9 DOF)",
                "Link length scaling (0.8x-1.2x)", 
                "GNN joint cooperation",
                "IK-retargeted trajectories"
            ],
            "training_data": "DROID-100 + 1870 morphology variations"
        }

# Standard interface functions for benchmark integration
def create_vla_model():
    """Create VLA model instance for benchmark evaluation"""
    return MorphologyVLAAdapter()

def evaluate_on_task(model, task_config):
    """Evaluate model on specific benchmark task"""
    # Implementation depends on specific benchmark protocol
    pass
