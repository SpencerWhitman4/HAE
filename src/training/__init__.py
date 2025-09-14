#!/usr/bin/env python3

"""
新训练架构 - 四层架构统一入口

架构设计:
- core/: 基础设施层 (抽象基类、环境工厂、模型迁移、训练流水线)
- foundation/: 基座模型层 (BaseFlightTrainer)
- branches/: 分支训练层 (Hierarchical、Ablation、Baseline)
- orchestration/: 编排层 (TrainingOrchestrator)

使用方式:
1. 快速开始: 使用TrainingOrchestrator运行完整流水线
2. 模块化使用: 单独使用各层组件进行定制化训练
"""

# 核心基础设施
from .core import (
    # 基础训练组件
    TrainingStage, TrainingResult, BaseTrainer, TrainerFactory, TrainingProgressCallback,
    # 环境管理
    EnvironmentFactory, EnvironmentTransitionManager,
    # 模型迁移
    ModelTransferManager,
    # 训练流水线
    TrainingPipeline, PipelineConfig, PipelineResult, PipelineStatus
)

# 基座模型训练
from .foundation import BaseFlightTrainer, BaseFlightModel

# 分支训练器
from .branches import HierarchicalTrainer, AblationTrainer, BaselineTrainer

# 训练编排器
from .orchestration import TrainingOrchestrator

# 便捷配置类
from .orchestration.training_orchestrator import OrchestrationConfig, OrchestrationResult

__all__ = [
    # === 核心基础设施 ===
    'TrainingStage',
    'TrainingResult', 
    'BaseTrainer',
    'TrainerFactory',
    'TrainingProgressCallback',
    'EnvironmentFactory',
    'EnvironmentTransitionManager', 
    'ModelTransferManager',
    'TrainingPipeline',
    'PipelineConfig',
    'PipelineResult',
    'PipelineStatus',
    
    # === 基座模型层 ===
    'BaseFlightTrainer',
    'BaseFlightModel',
    
    # === 分支训练层 ===
    'HierarchicalTrainer',
    'AblationTrainer', 
    'BaselineTrainer',
    
    # === 编排层 ===
    'TrainingOrchestrator',
    'OrchestrationConfig',
    'OrchestrationResult'
]

# 版本信息
__version__ = "1.0.0"
__author__ = "HAUAV Team"
__description__ = "新四层训练架构 - 基座模型驱动的分层训练系统"

# 架构说明
ARCHITECTURE_INFO = {
    'design_pattern': '四层架构 (Foundation → Branches)',
    'core_concept': '基座模型驱动的三分支并行训练',
    'training_stages': [
        'Foundation: BaseFlightAviary → 基座模型 (model.zip)',
        'Hierarchical: HA-UAV完整分层训练',
        'Ablation: B组消融实验 (B1/B2/B3)',
        'Baseline: SB3基线对比 (PPO/SAC/TD3)'
    ],
    'key_features': [
        '统一基座模型迁移',
        '环境工厂模式',
        '模块化训练器设计',
        '流水线式训练编排',
        '自动对比分析'
    ]
}


def get_architecture_info() -> dict:
    """获取架构信息"""
    return ARCHITECTURE_INFO


def create_quick_config(experiment_name: str = "hauav_training") -> OrchestrationConfig:
    """
    创建快速配置
    
    Args:
        experiment_name: 实验名称
        
    Returns:
        预配置的编排器配置
    """
    return OrchestrationConfig(
        experiment_name=experiment_name,
        output_dir="./experiments",
        
        # 启用所有训练阶段
        enable_foundation=True,
        enable_hierarchical=True, 
        enable_ablation=True,
        enable_baseline=True,
        
        # 基础配置
        enable_parallel_branches=False,  # 串行训练更稳定
        enable_cross_evaluation=True,
        
        # Foundation配置 (较短的基座训练)
        foundation_config={
            'total_timesteps': 50000,
            'enable_curriculum': True,
            'hover_training_steps': 15000,
            'flight_training_steps': 35000
        },
        
        # Hierarchical配置
        hierarchical_config={
            'total_timesteps': 100000,
            'high_level_update_frequency': 5,
            'future_horizon': 5
        },
        
        # Ablation配置
        ablation_config={
            'total_timesteps': 75000,
            'ablation_types': ['B1', 'B2', 'B3']
        },
        
        # Baseline配置  
        baseline_config={
            'total_timesteps': 100000,
            'algorithms': ['ppo', 'sac']
        }
    )


def run_complete_training(config: OrchestrationConfig = None) -> OrchestrationResult:
    """
    运行完整训练流程的便捷函数
    
    Args:
        config: 编排器配置，默认使用快速配置
        
    Returns:
        训练结果
    """
    if config is None:
        config = create_quick_config()
    
    orchestrator = TrainingOrchestrator(config)
    return orchestrator.run_complete_training()


# 使用示例
USAGE_EXAMPLE = '''
# === 使用示例 ===

# 1. 快速开始 - 运行完整训练流程
from src.training_new import run_complete_training, create_quick_config

config = create_quick_config("my_experiment")
result = run_complete_training(config)

if result.success:
    print("训练完成!")
    print(result.final_comparison_report)
else:
    print(f"训练失败: {result.error_message}")

# 2. 自定义配置
from src.training_new import OrchestrationConfig, TrainingOrchestrator

config = OrchestrationConfig(
    experiment_name="custom_experiment",
    enable_ablation=False,  # 跳过消融实验
    foundation_config={'total_timesteps': 30000}
)

orchestrator = TrainingOrchestrator(config)
result = orchestrator.run_complete_training()

# 3. 单独使用训练器
from src.training_new import BaseFlightTrainer, EnvironmentFactory

env_factory = EnvironmentFactory()
trainer = BaseFlightTrainer(
    config={'total_timesteps': 50000},
    env_factory=env_factory
)
foundation_result = trainer.train()

# 4. 流水线模式
from src.training_new import TrainingPipeline, PipelineConfig

pipeline_config = PipelineConfig(
    foundation_config={'total_timesteps': 30000},
    branches_config={'hierarchical': {'total_timesteps': 50000}}
)

pipeline = TrainingPipeline(pipeline_config)
pipeline_result = pipeline.run_pipeline()
'''


def print_usage_example():
    """打印使用示例"""
    print(USAGE_EXAMPLE)
