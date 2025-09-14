#!/usr/bin/env python3

"""
默认训练配置文件
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
    'environment_type': 'baseflight',
    'drone_model': 'CF2X',
    'physics': 'PYB',
    'gui_training': False,
    'gui_evaluation': False,
    'max_episode_steps': 1000,
    
    # 网络结构
    'hidden_dims': [256, 256, 128],
    
    # 评估和保存
    'evaluation_frequency': 10000,
    'save_frequency': 10000,
    'final_eval_episodes': 100,
    
    # 功能开关
    'enable_tensorboard': True,
    'enable_trajectory_recording': True,
    'enable_visualization': True
}

# 分层训练配置
HIERARCHICAL_CONFIG = {
    'total_timesteps': 200000,
    'batch_size': 128,
    'learning_rate': 1e-4,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    
    # 分层特定参数
    'high_level_update_frequency': 10,
    'future_horizon': 20,
    'hierarchical_buffer_size': 50000,
    
    # 环境配置
    'environment_type': 'hauav',
    'drone_model': 'CF2X',
    'physics': 'PYB',
    'gui_training': False,
    'gui_evaluation': False,
    'max_episode_steps': 1500,
    
    # 网络结构
    'high_level_dims': [256, 128],
    'low_level_dims': [128, 128],
    
    # 评估和保存
    'evaluation_frequency': 20000,
    'save_frequency': 20000,
    'final_eval_episodes': 50,
    
    # 功能开关
    'enable_tensorboard': True,
    'enable_trajectory_recording': True,
    'enable_visualization': True
}

# 消融实验配置
ABLATION_CONFIG = {
    'total_timesteps': 100000,
    'batch_size': 128,
    'learning_rate': 1e-4,
    
    # 消融特定参数
    'ablation_components': {
        'B1': {'disable_high_level': True},
        'B2': {'disable_future_prediction': True},
        'B3': {'disable_hierarchical_buffer': True}
    },
    
    # 环境配置
    'environment_type': 'hauav',
    'drone_model': 'CF2X',
    'physics': 'PYB',
    'gui_training': False,
    'gui_evaluation': False,
    'max_episode_steps': 1500,
    
    # 评估和保存
    'evaluation_frequency': 10000,
    'save_frequency': 10000,
    'final_eval_episodes': 50,
    
    # 功能开关
    'enable_tensorboard': True,
    'enable_trajectory_recording': True,
    'enable_visualization': True
}

# 基线算法配置
BASELINE_CONFIG = {
    'total_timesteps': 100000,
    'batch_size': 256,
    'learning_rate': 3e-4,
    
    # 基线算法选项
    'algorithms': {
        'ppo': {
            'algorithm': 'PPO',
            'clip_range': 0.2,
            'ent_coef': 0.01
        },
        'sac': {
            'algorithm': 'SAC',
            'learning_rate': 3e-4,
            'buffer_size': 100000
        },
        'dqn': {
            'algorithm': 'DQN',
            'learning_rate': 1e-4,
            'buffer_size': 50000
        }
    },
    
    # 环境配置
    'environment_type': 'hauav',
    'drone_model': 'CF2X',
    'physics': 'PYB',
    'gui_training': False,
    'gui_evaluation': False,
    'max_episode_steps': 1000,
    
    # 评估和保存
    'evaluation_frequency': 10000,
    'save_frequency': 10000,
    'final_eval_episodes': 100,
    
    # 功能开关
    'enable_tensorboard': True,
    'enable_trajectory_recording': True,
    'enable_visualization': True
}
