#!/usr/bin/env python3

"""
训练适配器 - 解决HAComponentsManager与训练系统的接口匹配问题
"""

import torch
import numpy as np
from typing import Dict, Any, Tuple, Optional, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class TrainingAdapter:
    """训练适配器 - 处理接口匹配和数据格式转换"""
    
    def __init__(self, ha_components=None):
        """
        初始化训练适配器
        
        Args:
            ha_components: HAComponentsManager实例，可选
        """
        self.ha_components = ha_components
        self.logger = logger
        
    def adapt_observation_format(self, obs_batch: np.ndarray) -> np.ndarray:
        """
        适配观测格式：从BaseRLAviary的(NUM_DRONES, OBS_DIM)转换为单智能体格式
        
        Args:
            obs_batch: (NUM_DRONES, OBS_DIM) 格式的观测批次
            
        Returns:
            适配后的单智能体观测
        """
        if obs_batch.ndim > 1 and obs_batch.shape[0] > 0:
            # 取第一个无人机的观测
            return obs_batch[0]
        return obs_batch
    
    def adapt_action_format(self, action: np.ndarray) -> np.ndarray:
        """
        适配动作格式：确保动作为正确的4维向量
        
        Args:
            action: 预测的动作
            
        Returns:
            适配后的动作
        """
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        
        action = np.array(action).flatten()
        
        # 确保是4维动作
        if len(action) < 4:
            action = np.pad(action, (0, 4 - len(action)), mode='constant')
        elif len(action) > 4:
            action = action[:4]
            
        return action
    
    def adapt_step_output(self, env_output: Tuple) -> Tuple:
        """
        适配环境step输出：处理多智能体到单智能体的转换
        
        Args:
            env_output: (obs_batch, reward_batch, terminated_batch, truncated_batch, info)
            
        Returns:
            适配后的单智能体输出
        """
        obs_batch, reward_batch, terminated_batch, truncated_batch, info = env_output
        
        # 适配观测
        obs = self.adapt_observation_format(obs_batch)
        
        # 适配奖励、终止标志等
        if isinstance(reward_batch, (list, np.ndarray)) and len(reward_batch) > 0:
            reward = reward_batch[0] if hasattr(reward_batch, '__len__') and len(reward_batch) > 0 else reward_batch
        else:
            reward = float(reward_batch)
            
        if isinstance(terminated_batch, (list, np.ndarray)) and len(terminated_batch) > 0:
            terminated = terminated_batch[0] if hasattr(terminated_batch, '__len__') and len(terminated_batch) > 0 else terminated_batch
        else:
            terminated = bool(terminated_batch)
            
        if isinstance(truncated_batch, (list, np.ndarray)) and len(truncated_batch) > 0:
            truncated = truncated_batch[0] if hasattr(truncated_batch, '__len__') and len(truncated_batch) > 0 else truncated_batch
        else:
            truncated = bool(truncated_batch)
        
        return obs, reward, terminated, truncated, info
    
    def safe_predict(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        安全的预测方法，处理各种可能的错误
        
        Args:
            obs: 观测向量
            deterministic: 是否确定性预测
            
        Returns:
            预测的动作
        """
        try:
            # 确保观测格式正确
            obs = self.adapt_observation_format(obs)
            
            # 调用HAComponentsManager的predict方法
            action = self.ha_components.predict(obs, deterministic=deterministic)
            
            # 适配动作格式
            return self.adapt_action_format(action)
            
        except Exception as e:
            self.logger.warning(f"预测失败，使用默认动作: {e}")
            # 返回安全的默认动作（悬停）
            return np.array([0.0, 0.0, 0.0, 0.0])
    
    def safe_train_step(self, env) -> Dict[str, Any]:
        """
        安全的训练步骤，处理HAComponentsManager的train_step调用
        
        Args:
            env: 环境实例
            
        Returns:
            训练步骤统计信息
        """
        try:
            # 调用HAComponentsManager的train_step
            if hasattr(self.ha_components, 'train_step'):
                return self.ha_components.train_step(env)
            else:
                # 如果没有train_step方法，实现基本的训练逻辑
                return self._basic_train_step(env)
                
        except Exception as e:
            self.logger.error(f"训练步骤失败: {e}")
            return {
                'total_steps': 0,
                'episodes': 0,
                'mean_reward': 0.0,
                'error': str(e)
            }
    
    def _basic_train_step(self, env) -> Dict[str, Any]:
        """
        基本的训练步骤实现（如果HAComponentsManager没有train_step方法）
        
        Args:
            env: 环境实例
            
        Returns:
            基本的统计信息
        """
        # 执行一个episode的数据收集
        obs_batch, _ = env.reset()
        obs = self.adapt_observation_format(obs_batch)
        
        episode_reward = 0
        step_count = 0
        
        for step in range(100):  # 限制步数
            action = self.safe_predict(obs)
            env_output = env.step(action)
            obs, reward, terminated, truncated, info = self.adapt_step_output(env_output)
            
            episode_reward += reward
            step_count += 1
            
            if terminated or truncated:
                break
        
        return {
            'total_steps': step_count,
            'episodes': 1,
            'mean_reward': episode_reward,
            'episode_length': step_count
        }
    
    def safe_evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        安全的动作评估，处理evaluate_actions的格式问题
        
        Args:
            obs: 观测张量
            actions: 动作张量
            
        Returns:
            (values, log_probs, entropy) - 确保格式一致性
        """
        try:
            if hasattr(self.ha_components, 'policy') and hasattr(self.ha_components.policy, 'evaluate_actions'):
                values, log_probs, entropy = self.ha_components.policy.evaluate_actions(obs, actions)
                
                # 确保返回值格式正确
                if values.dim() == 1:
                    values = values.unsqueeze(-1)  # [batch_size] -> [batch_size, 1]
                if log_probs.dim() == 2 and log_probs.size(1) == 1:
                    log_probs = log_probs.squeeze(-1)  # [batch_size, 1] -> [batch_size]
                if entropy.dim() == 2 and entropy.size(1) == 1:
                    entropy = entropy.squeeze(-1)  # [batch_size, 1] -> [batch_size]
                
                return values, log_probs, entropy
            else:
                # 创建标准化的虚拟返回值
                batch_size = obs.size(0)
                device = obs.device
                
                values = torch.zeros(batch_size, 1, device=device, dtype=torch.float32)
                log_probs = torch.zeros(batch_size, device=device, dtype=torch.float32)
                entropy = torch.zeros(batch_size, device=device, dtype=torch.float32)
                
                return values, log_probs, entropy
                
        except Exception as e:
            self.logger.warning(f"动作评估失败，使用默认值: {e}")
            batch_size = obs.size(0)
            device = obs.device
            
            values = torch.zeros(batch_size, 1, device=device, dtype=torch.float32)
            log_probs = torch.zeros(batch_size, device=device, dtype=torch.float32)
            entropy = torch.zeros(batch_size, device=device, dtype=torch.float32)
            
            return values, log_probs, entropy
    
    def create_smart_lr_schedule(self, initial_lr: float = 3e-4, schedule_type: str = "linear") -> callable:
        """
        创建智能的学习率调度器，解决lr_schedule问题
        
        Args:
            initial_lr: 初始学习率
            schedule_type: 调度类型 ("linear", "exponential", "cosine")
            
        Returns:
            学习率调度函数
        """
        def lr_schedule(progress_remaining: float) -> float:
            """
            智能学习率调度
            
            Args:
                progress_remaining: 剩余训练进度 (1.0 -> 0.0)
                
            Returns:
                当前学习率倍数
            """
            if schedule_type == "linear":
                # 线性衰减
                return progress_remaining
            elif schedule_type == "exponential":
                # 指数衰减
                return progress_remaining ** 2
            elif schedule_type == "cosine":
                # 余弦退火
                import math
                return 0.5 * (1 + math.cos(math.pi * (1 - progress_remaining)))
            else:
                # 默认常数学习率
                return 1.0
        
        self.logger.info(f"创建{schedule_type}学习率调度器，初始学习率: {initial_lr}")
        return lr_schedule
    
    def validate_ha_components(self) -> Dict[str, bool]:
        """
        验证HAComponentsManager的关键组件
        
        Returns:
            验证结果字典
        """
        validation = {
            'has_predict': hasattr(self.ha_components, 'predict'),
            'has_train_step': hasattr(self.ha_components, 'train_step'),
            'has_policy': hasattr(self.ha_components, 'policy'),
            'has_initialize_components': hasattr(self.ha_components, 'initialize_components'),
            'has_save_model': hasattr(self.ha_components, 'save_model'),
            'has_load_model': hasattr(self.ha_components, 'load_model')
        }
        
        if validation['has_policy']:
            validation['policy_has_evaluate_actions'] = hasattr(self.ha_components.policy, 'evaluate_actions')
        else:
            validation['policy_has_evaluate_actions'] = False
            
        return validation


def create_training_adapter(ha_components) -> TrainingAdapter:
    """
    创建训练适配器的便捷函数
    
    Args:
        ha_components: HAComponentsManager实例
        
    Returns:
        TrainingAdapter实例
    """
    return TrainingAdapter(ha_components)
