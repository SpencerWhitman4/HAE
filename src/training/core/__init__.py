#!/usr/bin/env python3

"""
Core基础设施模块 - 导出核心组件
"""

from .base_trainer import (
    TrainingStage,
    TrainingResult, 
    BaseTrainer,
    TrainerFactory,
    TrainingProgressCallback
)

from .environment_factory import (
    EnvironmentFactory,
    EnvironmentTransitionManager
)

from .model_transfer import (
    ModelTransferManager
)

from .training_pipeline import (
    TrainingPipeline,
    PipelineConfig,
    PipelineResult,
    PipelineStatus
)

from .session_manager import (
    SessionManager,
    create_session_manager
)

from .visualization_manager import (
    TrainingVisualizationManager,
    create_visualization_manager
)

__all__ = [
    # 基础训练组件
    'TrainingStage',
    'TrainingResult',
    'BaseTrainer', 
    'TrainerFactory',
    'TrainingProgressCallback',
    
    # 环境管理
    'EnvironmentFactory',
    'EnvironmentTransitionManager',
    
    # 模型迁移
    'ModelTransferManager',
    
    # 训练流水线
    'TrainingPipeline',
    'PipelineConfig',
    'PipelineResult',
    'PipelineStatus',
    
    # 会话管理
    'SessionManager',
    'create_session_manager',
    
    # 可视化管理
    'TrainingVisualizationManager',
    'create_visualization_manager'
]

# 版本信息
__version__ = "1.0.0"

# 模块信息
__author__ = "HAUAV Team"
__description__ = "Core infrastructure for HAUAV training pipeline"

from .base_trainer import BaseTrainer, TrainingStage, TrainingResult
from .environment_factory import EnvironmentFactory
from .model_transfer import ModelTransferManager
from .training_pipeline import TrainingPipeline, PipelineStatus

__all__ = [
    'BaseTrainer',
    'TrainingStage', 
    'TrainingResult',
    'EnvironmentFactory',
    'ModelTransferManager',
    'TrainingPipeline',
    'PipelineStatus'
]
