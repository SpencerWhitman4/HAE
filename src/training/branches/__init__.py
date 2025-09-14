#!/usr/bin/env python3

"""
Branches分支训练器 - 三条并行训练分支
"""

from .hierarchical_trainer import HierarchicalTrainer
from .ablation_trainer import AblationTrainer  
from .baseline_trainer import BaselineTrainer

__all__ = [
    'HierarchicalTrainer',
    'AblationTrainer',
    'BaselineTrainer'
]

__version__ = "1.0.0"
