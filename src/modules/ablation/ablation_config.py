#!/usr/bin/env python3

"""
B组消融实验配置管理
==================

基于HA-UAV架构文档设计的B组消融实验配置系统。
验证分层架构的必要性，包括：
- B1: 高层直接控制（移除低层策略）
- B2: 扁平化架构（移除高层策略）
- B3: 单步子目标（退化分层）

设计原则：
1. 最大化复用现有基础设施
2. 配置驱动的模块替换机制
3. 数据流适配策略
4. 接口兼容性保证
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)


@dataclass
class AblationConfig:
    """B组消融实验配置"""
    
    # 实验基本信息
    experiment_group: str = "baseline"  # "B1", "B2", "B3", "baseline"
    experiment_name: str = ""
    description: str = ""
    
    # 分层架构控制
    use_hierarchical: bool = True
    use_high_level_policy: bool = True
    use_low_level_policy: bool = True
    
    # 高层策略配置
    high_level_direct_control: bool = False  # B1: 高层直接输出4D控制
    high_level_update_frequency: int = 5     # τ=5步周期
    subgoal_horizon: int = 5                 # T=5步子目标序列
    history_length: int = 20                 # K=20步历史
    
    # 低层策略配置
    low_level_bypass: bool = False           # B1: 跳过低层策略
    flat_policy_input_dim: int = 86          # B2: 扁平策略输入维度
    
    # 观测处理配置
    observation_processing: str = "hierarchical"  # "hierarchical", "flat", "direct"
    high_level_obs_dim: int = 28             # 高层单步观测维度
    low_level_obs_dim: int = 64              # 低层观测维度
    
    # 网络架构配置
    policy_type: str = "HierarchicalPolicy"  # "DirectControlPolicy", "FlatPolicy", "SingleStepHierarchicalPolicy"
    high_level_features_dim: int = 256
    low_level_features_dim: int = 128
    
    # 训练配置
    buffer_adaptation: bool = True           # 是否适配buffer
    use_hierarchical_gae: bool = True        # 是否使用分层GAE
    joint_training: bool = True              # 是否联合训练
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AblationConfig':
        """从字典创建配置"""
        return cls(**config_dict)


def create_b1_config() -> AblationConfig:
    """
    B1实验配置：高层直接控制
    
    数据流：86维obs → StateManager → K×28历史序列 → DirectControlPolicy → 4D控制
    特点：保持高层策略，移除低层策略，高层直接输出控制命令
    """
    return AblationConfig(
        experiment_group="B1",
        experiment_name="B1_Direct_Control",
        description="高层策略直接输出4D控制，跳过低层策略",
        
        # 分层架构：保持高层，移除低层
        use_hierarchical=True,
        use_high_level_policy=True,
        use_low_level_policy=False,
        
        # B1特有配置
        high_level_direct_control=True,
        low_level_bypass=True,
        
        # 观测处理：使用高层历史序列
        observation_processing="direct",
        high_level_update_frequency=5,  # 保持τ=5步更新
        history_length=20,              # 保持K=20步历史
        
        # 网络架构
        policy_type="DirectControlPolicy",
        high_level_features_dim=256,
        
        # 训练配置
        buffer_adaptation=True,         # 需要适配buffer（低层数据置零）
        use_hierarchical_gae=False,     # 只有高层GAE
        joint_training=False            # 只训练高层
    )


def create_b2_config() -> AblationConfig:
    """
    B2实验配置：扁平化架构
    
    数据流：86维obs → StateManager → 86维原始obs → FlatPolicy → 4D控制
    特点：移除高层策略，使用扁平化网络直接从原始观测到控制
    """
    return AblationConfig(
        experiment_group="B2",
        experiment_name="B2_Flat_Policy",
        description="扁平化策略，直接从86维观测到4D控制",
        
        # 分层架构：移除高层，保持低层
        use_hierarchical=False,
        use_high_level_policy=False,
        use_low_level_policy=True,
        
        # B2特有配置
        high_level_direct_control=False,
        low_level_bypass=False,
        
        # 观测处理：使用原始观测
        observation_processing="flat",
        flat_policy_input_dim=86,
        
        # 网络架构
        policy_type="FlatPolicy",
        low_level_features_dim=128,
        
        # 训练配置
        buffer_adaptation=True,         # 需要适配buffer（高层数据置零）
        use_hierarchical_gae=False,     # 只有低层GAE
        joint_training=False            # 只训练扁平策略
    )


def create_b3_config() -> AblationConfig:
    """
    B3实验配置：单步子目标
    
    数据流：86维obs → StateManager → 64维低层obs + 单步子目标(重复5次) → Policy → 4D控制
    特点：保持分层结构，但退化为单步子目标
    """
    return AblationConfig(
        experiment_group="B3",
        experiment_name="B3_Single_Step_Subgoal",
        description="单步子目标的退化分层架构",
        
        # 分层架构：保持双层，但退化
        use_hierarchical=True,
        use_high_level_policy=True,
        use_low_level_policy=True,
        
        # B3特有配置
        high_level_direct_control=False,
        low_level_bypass=False,
        subgoal_horizon=1,              # 关键：单步子目标
        
        # 观测处理：保持分层
        observation_processing="hierarchical",
        high_level_update_frequency=5,   # 保持τ=5步更新
        history_length=20,               # 保持K=20步历史
        
        # 网络架构
        policy_type="SingleStepHierarchicalPolicy",
        high_level_features_dim=256,
        low_level_features_dim=128,
        
        # 训练配置
        buffer_adaptation=True,          # 需要适配buffer（单步扩展）
        use_hierarchical_gae=True,       # 保持分层GAE
        joint_training=True              # 联合训练
    )


def create_baseline_config() -> AblationConfig:
    """
    基线配置：完整HA-UAV架构
    
    数据流：完整的分层架构，用于对比
    """
    return AblationConfig(
        experiment_group="baseline",
        experiment_name="Baseline_HA_UAV",
        description="完整的HA-UAV分层架构",
        
        # 完整分层架构
        use_hierarchical=True,
        use_high_level_policy=True,
        use_low_level_policy=True,
        
        # 标准配置
        high_level_direct_control=False,
        low_level_bypass=False,
        high_level_update_frequency=5,
        subgoal_horizon=5,
        history_length=20,
        
        # 观测处理
        observation_processing="hierarchical",
        
        # 网络架构
        policy_type="HierarchicalPolicy",
        high_level_features_dim=256,
        low_level_features_dim=128,
        
        # 训练配置
        buffer_adaptation=False,         # 不需要适配
        use_hierarchical_gae=True,       # 完整分层GAE
        joint_training=True              # 联合训练
    )


class AblationConfigManager:
    """B组消融实验配置管理器"""
    
    def __init__(self):
        self._configs = {
            "B1": create_b1_config,
            "B2": create_b2_config,
            "B3": create_b3_config,
            "baseline": create_baseline_config
        }
        
        logger.info("B组消融实验配置管理器初始化完成")
    
    def get_config(self, experiment_id: str) -> AblationConfig:
        """获取指定实验的配置"""
        if experiment_id not in self._configs:
            raise ValueError(f"未知的实验ID: {experiment_id}，支持的实验: {list(self._configs.keys())}")
        
        config = self._configs[experiment_id]()
        logger.info(f"加载实验配置: {experiment_id}")
        return config
    
    def list_experiments(self) -> Dict[str, str]:
        """列出所有可用的实验"""
        experiments = {}
        for exp_id in self._configs:
            config = self._configs[exp_id]()
            experiments[exp_id] = config.description
        return experiments
    
    def validate_config(self, config: AblationConfig) -> bool:
        """验证配置的有效性"""
        # 检查B1配置的一致性
        if config.experiment_group == "B1":
            if not (config.high_level_direct_control and config.low_level_bypass):
                logger.error("B1配置错误：应该启用high_level_direct_control和low_level_bypass")
                return False
        
        # 检查B2配置的一致性
        elif config.experiment_group == "B2":
            if config.use_hierarchical or config.use_high_level_policy:
                logger.error("B2配置错误：应该禁用分层策略")
                return False
        
        # 检查B3配置的一致性
        elif config.experiment_group == "B3":
            if config.subgoal_horizon != 1:
                logger.error("B3配置错误：subgoal_horizon应该为1")
                return False
        
        logger.info(f"配置验证通过: {config.experiment_group}")
        return True


# 全局配置管理器实例
ablation_config_manager = AblationConfigManager()


# 便捷函数
def get_ablation_config(experiment_id: str) -> AblationConfig:
    """获取消融实验配置"""
    return ablation_config_manager.get_config(experiment_id)


def list_ablation_experiments() -> Dict[str, str]:
    """列出所有消融实验"""
    return ablation_config_manager.list_experiments()
