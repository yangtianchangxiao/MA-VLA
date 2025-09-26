#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base Morphology Module - Abstract interface for all morphology transformations

Extracted from modular_synthesis_system.py to support independent module development.
"""

import numpy as np
from typing import Dict
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class MorphologyConfig:
    """Configuration for a single morphology variation"""
    name: str
    link_scales: list  # Scale factors for each link
    base_position: np.ndarray  # [x,y,z] calculated via IK to preserve end-effector trajectory
    base_orientation: np.ndarray  # [rx,ry,rz] fixed, unchanged (利用7-DOF冗余性)
    dof_modification: dict = None  # Future DOF changes
    camera_params: dict = None  # Future camera adjustments
    
    def __post_init__(self):
        self.base_position = np.array(self.base_position)
        self.base_orientation = np.array(self.base_orientation)


class SynthesisModule(ABC):
    """Abstract base class for synthesis modules"""
    
    @abstractmethod
    def generate_variation(self, config: MorphologyConfig) -> Dict:
        """Generate variation data based on config"""
        pass
    
    @abstractmethod
    def apply_to_trajectory(self, trajectory: np.ndarray, variation_data: Dict) -> np.ndarray:
        """Apply variation to trajectory data"""
        pass