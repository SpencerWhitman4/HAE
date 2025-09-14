#!/usr/bin/env python3

"""
环境工厂 - 统一环境创建和管理，复用现有environment_manager
"""

import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path
import sys

# 添加项目路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from .base_trainer import TrainingStage

# 复用现有环境组件
from src.envs.HAUAV_Aviary import HAUAVAviary, HAUAVConfig
from src.envs.BaseFlightAviary import BaseFlightAviary, BaseFlightConfig
from src.utils.enums import DroneModel, Physics

import gymnasium as gym
import numpy as np

logger = logging.getLogger(__name__)


class BaselineWrapper(gym.Wrapper):
    """基线算法环境包装器 - 为SB3等基线库提供兼容接口"""
    
    def __init__(self, env, agent_type: str = "sb3"):
        super().__init__(env)
        self.agent_type = agent_type
        self.logger = logging.getLogger(__name__)
        
        # 确保观测和动作空间格式正确
        self._setup_spaces()
        
    def _setup_spaces(self):
        """设置观测和动作空间"""
        # 确保观测空间是Box类型
        if hasattr(self.env, 'observation_space'):
            self.observation_space = self.env.observation_space
        else:
            # 默认86维观测空间
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(86,), dtype=np.float32
            )
        
        # 确保动作空间是Box类型
        if hasattr(self.env, 'action_space'):
            self.action_space = self.env.action_space
        else:
            # 默认4维动作空间 [vx, vy, vz, yaw_rate]
            self.action_space = gym.spaces.Box(
                low=np.array([-1.0, -1.0, -1.0, -1.0]),
                high=np.array([1.0, 1.0, 1.0, 1.0]),
                dtype=np.float32
            )
    
    def reset(self, **kwargs):
        """重置环境"""
        obs, info = self.env.reset(**kwargs)
        
        # 处理多智能体到单智能体的转换
        if isinstance(obs, (list, np.ndarray)) and obs.ndim > 1:
            obs = obs[0]  # 取第一个智能体的观测
        
        # 确保观测是正确的numpy数组格式
        obs = np.array(obs, dtype=np.float32)
        
        return obs, info
    
    def step(self, action):
        """执行动作"""
        # 确保动作是正确的格式
        action = np.array(action, dtype=np.float32)
        
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 处理多智能体到单智能体的转换
        if isinstance(obs, (list, np.ndarray)) and obs.ndim > 1:
            obs = obs[0]
        if isinstance(reward, (list, np.ndarray)):
            reward = reward[0] if len(reward) > 0 else 0.0
        if isinstance(terminated, (list, np.ndarray)):
            terminated = terminated[0] if len(terminated) > 0 else False
        if isinstance(truncated, (list, np.ndarray)):
            truncated = truncated[0] if len(truncated) > 0 else False
        
        # 确保数据类型正确
        obs = np.array(obs, dtype=np.float32)
        reward = float(reward)
        terminated = bool(terminated)
        truncated = bool(truncated)
        
        return obs, reward, terminated, truncated, info
    
    def close(self):
        """关闭环境"""
        if hasattr(self.env, 'close'):
            self.env.close()


logger = logging.getLogger(__name__)


class EnvironmentFactory:
    """
    环境工厂 - 基于现有environment_manager重构
    
    负责为不同训练阶段创建相应的环境实例
    """
    
    def __init__(self):
        self.logger = logger
        self._env_cache = {}  # 环境缓存
    
    def create_environment(self, 
                          stage: TrainingStage, 
                          config: Dict[str, Any],
                          mode: str = "train",
                          **kwargs) -> Union[HAUAVAviary, BaseFlightAviary]:
        """
        创建环境实例
        
        Args:
            stage: 训练阶段
            config: 环境配置
            mode: 模式 ("train" 或 "eval")
            **kwargs: 额外参数
            
        Returns:
            环境实例
        """
        cache_key = f"{stage.value}_{mode}"
        
        # 检查缓存
        if cache_key in self._env_cache:
            return self._env_cache[cache_key]
        
        # 创建环境
        if stage == TrainingStage.FOUNDATION:
            env = self._create_baseflight_env(config, mode, **kwargs)
        
        elif stage == TrainingStage.HIERARCHICAL:
            env = self._create_hauav_env(config, mode, **kwargs)
        
        elif stage == TrainingStage.ABLATION:
            env = self._create_ablation_env(config, mode, **kwargs)
        
        elif stage == TrainingStage.BASELINE:
            env = self._create_baseline_env(config, mode, **kwargs)
        
        else:
            raise ValueError(f"不支持的训练阶段: {stage}")
        
        # 缓存环境
        self._env_cache[cache_key] = env
        
        self.logger.info(f"创建 {stage.value} 环境完成 (模式: {mode})")
        return env
    
    def _create_baseflight_env(self, 
                              config: Dict[str, Any], 
                              mode: str,
                              **kwargs) -> BaseFlightAviary:
        """创建BaseFlightAviary环境"""
        
        # 基于现有配置创建BaseFlightConfig
        env_config = BaseFlightConfig()
        
        # 更新配置参数
        env_config.MAX_EPISODE_STEPS = config.get('max_episode_steps', 1000)
        env_config.hover_training_steps = config.get('hover_training_steps', 250)
        env_config.flight_training_steps = config.get('flight_training_steps', 750)
        env_config.curriculum_learning = config.get('enable_curriculum', True)
        env_config.ENABLE_TRAJECTORY_RECORDING = config.get('enable_trajectory_recording', False)
        
        # 创建BaseFlightAviary - 使用正确的参数
        drone_model_map = {
            'CF2X': DroneModel.CF2X,
            'CF2P': DroneModel.CF2P
        }
        
        physics_map = {
            'PYB': Physics.PYB,
            'DYN': Physics.DYN
        }
        
        env = BaseFlightAviary(
            drone_model=drone_model_map.get(config.get('drone_model', 'CF2X'), DroneModel.CF2X),
            physics=physics_map.get(config.get('physics', 'PYB'), Physics.PYB),
            gui=config.get('gui_training', False) if mode == "train" else config.get('gui_evaluation', False),
            record=config.get('enable_trajectory_recording', False),
            config=env_config
        )
        
        # 轨迹记录配置
        if config.get('enable_trajectory_recording', False):
            self._setup_trajectory_recording(env, mode, config)
        
        return env
    
    def _create_hauav_env(self, 
                         config: Dict[str, Any], 
                         mode: str,
                         **kwargs) -> HAUAVAviary:
        """创建HAUAVAviary环境"""
        
        # 基于现有配置创建HAUAVConfig
        env_config = HAUAVConfig()
        
        # 更新配置参数
        env_config.MAX_EPISODE_STEPS = config.get('max_episode_steps', 1000)
        env_config.EXPLORATION_THRESHOLD = config.get('exploration_threshold', 0.95)
        env_config.COLLISION_THRESHOLD = config.get('collision_threshold', 0.2)
        env_config.SAFETY_THRESHOLD = config.get('safety_threshold', 0.5)
        env_config.ENABLE_TRAJECTORY_RECORDING = config.get('enable_trajectory_recording', False)
        
        # 创建HAUAVAviary - 使用正确的参数
        drone_model_map = {
            'CF2X': DroneModel.CF2X,
            'CF2P': DroneModel.CF2P
        }
        
        physics_map = {
            'PYB': Physics.PYB,
            'DYN': Physics.DYN
        }
        
        env = HAUAVAviary(
            drone_model=drone_model_map.get(config.get('drone_model', 'CF2X'), DroneModel.CF2X),
            physics=physics_map.get(config.get('physics', 'PYB'), Physics.PYB),
            gui=config.get('gui_training', False) if mode == "train" else config.get('gui_evaluation', False),
            record=config.get('enable_trajectory_recording', False),
            config=env_config,
            obstacles=True,  # HAUAV需要障碍物
            enable_tensorboard=config.get('enable_tensorboard', False),
            tensorboard_mode=mode
        )
        
        # 轨迹记录配置
        if config.get('enable_trajectory_recording', False):
            self._setup_trajectory_recording(env, mode, config)
        
        return env
    
    def _create_ablation_env(self, 
                            config: Dict[str, Any], 
                            mode: str,
                            **kwargs) -> HAUAVAviary:
        """创建消融实验环境 - 基于HAUAVAviary"""
        
        # 消融实验使用HAUAVAviary作为基础环境
        # 具体的消融逻辑在模型层处理
        ablation_type = kwargs.get('ablation_type', 'B1')
        
        # 根据消融类型调整配置
        if ablation_type == 'B1':  # 高层直接控制
            config['high_level_update_frequency'] = config.get('high_level_update_frequency', 5)
        elif ablation_type == 'B2':  # 扁平化
            config['high_level_update_frequency'] = 1  # 每步更新
        elif ablation_type == 'B3':  # 单步分层
            config['future_horizon'] = 1  # 单步子目标
        
        return self._create_hauav_env(config, mode, **kwargs)
    
    def _create_baseline_env(self, 
                            config: Dict[str, Any], 
                            mode: str,
                            **kwargs) -> BaselineWrapper:
        """创建基线环境 - 包装现有环境"""
        
        # 基线算法使用HAUAVAviary + BaselineWrapper
        base_env = self._create_hauav_env(config, mode, **kwargs)
        
        # 应用BaselineWrapper（复用现有实现）
        algorithm = kwargs.get('algorithm', 'ppo')
        wrapped_env = BaselineWrapper(base_env, agent_type="sb3")
        
        return wrapped_env
    
    def _setup_trajectory_recording(self, 
                                   env, 
                                   mode: str, 
                                   config: Dict[str, Any]) -> None:
        """设置轨迹记录 - 如果环境支持的话"""
        
        # 检查环境是否支持轨迹记录
        if hasattr(env, 'setup_trajectory_recording'):
            trajectory_dir = config.get('trajectory_dir', './logs/trajectories')
            env.setup_trajectory_recording(
                save_dir=f"{trajectory_dir}/{mode}",
                save_frequency=config.get('trajectory_save_frequency', 100)
            )
        else:
            # 环境不支持轨迹记录，跳过
            pass
    
    def get_environment_info(self, stage: TrainingStage) -> Dict[str, Any]:
        """获取环境信息"""
        
        info_map = {
            TrainingStage.FOUNDATION: {
                'env_type': 'BaseFlightAviary',
                'observation_space': 'Box(86,)',
                'action_space': 'Box(4,)',
                'features': ['悬停训练', '基础飞行', '课程学习'],
                'expected_performance': '悬停稳定性 > 95%'
            },
            TrainingStage.HIERARCHICAL: {
                'env_type': 'HAUAVAviary', 
                'observation_space': 'Box(86,)',
                'action_space': 'Box(4,)',
                'features': ['分层决策', '86维雷达观测', '子目标序列'],
                'expected_performance': '导航成功率 > 80%'
            },
            TrainingStage.ABLATION: {
                'env_type': 'HAUAVAviary (Modified)',
                'observation_space': 'Box(86,)',
                'action_space': 'Box(4,)',
                'features': ['消融变体', 'B1/B2/B3实验'],
                'expected_performance': '验证分层架构必要性'
            },
            TrainingStage.BASELINE: {
                'env_type': 'HAUAVAviary + BaselineWrapper',
                'observation_space': 'Box(86,)',
                'action_space': 'Box(4,)', 
                'features': ['SB3兼容', 'PPO/SAC/TD3'],
                'expected_performance': '基线性能对比'
            }
        }
        
        return info_map.get(stage, {})
    
    def validate_environment(self, env, stage: TrainingStage) -> bool:
        """验证环境配置正确性"""
        
        try:
            # 基本验证
            obs, _ = env.reset()
            assert obs.shape == (86,), f"观测维度错误: {obs.shape}"
            
            # 动作空间验证
            action = env.action_space.sample()
            obs_next, reward, terminated, truncated, info = env.step(action)
            
            self.logger.info(f"{stage.value} 环境验证通过")
            return True
            
        except Exception as e:
            self.logger.error(f"{stage.value} 环境验证失败: {e}")
            return False
    
    def cleanup_environments(self) -> None:
        """清理所有环境"""
        for cache_key, env in self._env_cache.items():
            try:
                env.close()
                self.logger.debug(f"关闭环境: {cache_key}")
            except Exception as e:
                self.logger.error(f"关闭环境 {cache_key} 时发生错误: {e}")
        
        self._env_cache.clear()
        self.logger.info("所有环境已清理")


class EnvironmentTransitionManager:
    """环境转换管理器 - 处理不同训练阶段间的环境切换"""
    
    def __init__(self, factory: EnvironmentFactory):
        self.factory = factory
        self.logger = logging.getLogger(__name__)
    
    def get_transition_mapping(self, 
                              from_stage: TrainingStage, 
                              to_stage: TrainingStage) -> Dict[str, Any]:
        """
        获取阶段间转换映射
        
        Args:
            from_stage: 源阶段
            to_stage: 目标阶段
            
        Returns:
            转换映射信息
        """
        
        transition_map = {
            (TrainingStage.FOUNDATION, TrainingStage.HIERARCHICAL): {
                'env_change': 'BaseFlightAviary -> HAUAVAviary',
                'observation_compatibility': 'Full (86dim)',
                'action_compatibility': 'Full (4dim)',
                'weight_transfer': 'Low-level control weights',
                'notes': '基础飞行控制 -> 分层决策'
            },
            
            (TrainingStage.FOUNDATION, TrainingStage.ABLATION): {
                'env_change': 'BaseFlightAviary -> HAUAVAviary (Modified)',
                'observation_compatibility': 'Full (86dim)',
                'action_compatibility': 'Full (4dim)',
                'weight_transfer': 'Depends on ablation type',
                'notes': '基础飞行控制 -> 消融变体'
            },
            
            (TrainingStage.FOUNDATION, TrainingStage.BASELINE): {
                'env_change': 'BaseFlightAviary -> HAUAVAviary + Wrapper',
                'observation_compatibility': 'Full (86dim)',
                'action_compatibility': 'Full (4dim)',
                'weight_transfer': 'Initialization bias',
                'notes': '基础飞行控制 -> SB3基线'
            }
        }
        
        key = (from_stage, to_stage)
        return transition_map.get(key, {'error': f'不支持的转换: {key}'})
    
    def validate_transition(self, 
                           from_stage: TrainingStage, 
                           to_stage: TrainingStage) -> bool:
        """验证阶段转换的有效性"""
        
        # Foundation必须是起点
        if from_stage != TrainingStage.FOUNDATION:
            self.logger.warning(f"非标准转换: {from_stage} -> {to_stage}")
        
        # 获取转换映射
        mapping = self.get_transition_mapping(from_stage, to_stage)
        
        if 'error' in mapping:
            self.logger.error(mapping['error'])
            return False
        
        self.logger.info(f"转换验证通过: {mapping['env_change']}")
        return True
