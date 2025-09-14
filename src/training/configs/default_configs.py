#!/usr/bin/env python3

"""
默认训练配置
"""

# 基座模型训练配置
FOUNDATION_CONFIG = {
    'total_timesteps': 100000,
    'batch_size': 256,
    'learning_rate': 3e-4,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'max_grad_norm': 0.5,
    
    # 课程学习
    'enable_curriculum': True,
    'hover_training_steps': 25000,
    'flight_training_steps': 75000,
    
    # 环境配置
    'drone_model': 'CF2X',
    'physics': 'PYB',
    'gui_training': False,
    'max_episode_steps': 1000,
    
    # 网络结构
    'hidden_dims': [256, 256, 128]
}

# 分层训练配置
HIERARCHICAL_CONFIG = {
    'total_timesteps': 200000,
    'batch_size': 256,
    'learning_rate': 3e-4,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    
    # 分层特定参数
    'high_level_update_frequency': 5,
    'future_horizon': 5,
    'enable_intrinsic_motivation': True,
    
    # 损失权重
    'low_level_loss_weight': 1.0,
    'high_level_loss_weight': 0.5,
    'intrinsic_loss_weight': 0.1,
    
    # 环境配置
    'drone_model': 'CF2X',
    'physics': 'PYB',
    'gui_training': False,
    'max_episode_steps': 1000
}

# 消融实验配置
ABLATION_CONFIG = {
    'total_timesteps': 150000,
    'batch_size': 256,
    'learning_rate': 3e-4,
    'gamma': 0.99,
    
    # 消融实验类型
    'ablation_types': ['B1', 'B2', 'B3'],
    
    # 环境配置
    'drone_model': 'CF2X',
    'physics': 'PYB',
    'gui_training': False,
    'max_episode_steps': 1000
}

# 基线对比配置
BASELINE_CONFIG = {
    'total_timesteps': 200000,
    'eval_freq': 10000,
    'eval_episodes': 10,
    
    # 支持的算法
    'algorithms': ['ppo', 'sac', 'td3'],
    'use_pretrained_init': True,
    
    # PPO配置
    'ppo_learning_rate': 3e-4,
    'ppo_n_steps': 2048,
    'ppo_batch_size': 64,
    'ppo_n_epochs': 10,
    'clip_range': 0.2,
    'ent_coef': 0.0,
    'vf_coef': 0.5,
    
    # SAC配置
    'sac_learning_rate': 3e-4,
    'sac_buffer_size': 1000000,
    'sac_learning_starts': 100,
    'sac_batch_size': 256,
    'sac_tau': 0.005,
    'sac_train_freq': 1,
    'sac_gradient_steps': 1,
    
    # TD3配置
    'td3_learning_rate': 3e-4,
    'td3_buffer_size': 1000000,
    'td3_learning_starts': 100,
    'td3_batch_size': 100,
    'td3_tau': 0.005,
    'td3_train_freq': 1,
    'td3_gradient_steps': 1,
    'td3_policy_delay': 2,
    
    # 通用参数
    'gamma': 0.99,
    
    # 环境配置
    'drone_model': 'CF2X',
    'physics': 'PYB',
    'gui_training': False,
    'max_episode_steps': 1000
}

# 迁移学习配置
TRANSFER_CONFIG = {
    'transfer_mode': 'partial',  # 'full', 'partial', 'bias_only'
    'freeze_encoder': True,
    'freeze_actor': False
}

# 预设配置集合
PRESET_CONFIGS = {
    'quick_test': {
        'foundation': {**FOUNDATION_CONFIG, 'total_timesteps': 10000, 'hover_training_steps': 3000, 'flight_training_steps': 7000},
        'hierarchical': {**HIERARCHICAL_CONFIG, 'total_timesteps': 20000},
        'ablation': {**ABLATION_CONFIG, 'total_timesteps': 15000, 'ablation_types': ['B1', 'B2']},
        'baseline': {**BASELINE_CONFIG, 'total_timesteps': 20000, 'algorithms': ['ppo']},
        'transfer': TRANSFER_CONFIG
    },
    
    'standard': {
        'foundation': FOUNDATION_CONFIG,
        'hierarchical': HIERARCHICAL_CONFIG,
        'ablation': ABLATION_CONFIG,
        'baseline': BASELINE_CONFIG,
        'transfer': TRANSFER_CONFIG
    },
    
    'extended': {
        'foundation': {**FOUNDATION_CONFIG, 'total_timesteps': 200000, 'hover_training_steps': 50000, 'flight_training_steps': 150000},
        'hierarchical': {**HIERARCHICAL_CONFIG, 'total_timesteps': 400000},
        'ablation': {**ABLATION_CONFIG, 'total_timesteps': 300000},
        'baseline': {**BASELINE_CONFIG, 'total_timesteps': 400000, 'algorithms': ['ppo', 'sac', 'td3']},
        'transfer': TRANSFER_CONFIG
    }
}


def get_preset_config(preset_name: str = 'standard') -> dict:
    """
    获取预设配置
    
    Args:
        preset_name: 预设名称 ('quick_test', 'standard', 'extended')
        
    Returns:
        配置字典
    """
    if preset_name not in PRESET_CONFIGS:
        raise ValueError(f"未知的预设配置: {preset_name}. 可用: {list(PRESET_CONFIGS.keys())}")
    
    return PRESET_CONFIGS[preset_name]


def merge_configs(base_config: dict, override_config: dict) -> dict:
    """
    合并配置字典
    
    Args:
        base_config: 基础配置
        override_config: 覆盖配置
        
    Returns:
        合并后的配置
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged
