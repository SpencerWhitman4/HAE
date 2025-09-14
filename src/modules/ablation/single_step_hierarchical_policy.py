#!/usr/bin/env python3

"""
B3消融实验：单步分层策略
===================

数据流：86维obs → StateManager → 64维低层obs + 单步子目标(重复5次) → Policy → 4D控制

设计特点：
1. 保持分层结构，但退化为单步子目标
2. 高层策略仍然每τ=5步更新，但只生成单步子目标
3. 低层策略接收单步子目标（重复5次填充）
4. 验证多步子目标序列的必要性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List, Optional
import logging
from stable_baselines3.common.distributions import Distribution

from .ablation_config import AblationConfig

logger = logging.getLogger(__name__)


class SingleStepHighLevelNetwork(nn.Module):
    """
    B3单步高层策略网络
    
    输入：K×28维历史序列观测
    输出：单步子目标（将重复5次给低层）
    """
    
    def __init__(self, config: AblationConfig):
        super().__init__()
        self.config = config
        
        # 输入维度：K×28历史序列
        history_input_dim = config.history_length * config.high_level_obs_dim  # 20×28=560
        
        # 特征提取层
        self.feature_extractor = nn.Sequential(
            nn.Linear(history_input_dim, config.high_level_features_dim),
            nn.ReLU(),
            nn.Linear(config.high_level_features_dim, config.high_level_features_dim),
            nn.ReLU(),
            nn.Linear(config.high_level_features_dim, config.high_level_features_dim // 2),
            nn.ReLU()
        )
        
        # 单步子目标输出层（3维：目标位置偏移）
        feature_dim = config.high_level_features_dim // 2
        self.subgoal_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # 3维子目标：[dx, dy, dz]
        )
        
        # 高层价值函数头
        self.high_value_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # 子目标限制
        self.subgoal_bounds = {
            'position_offset': 2.0  # 位置偏移限制 [m]
        }
        
        logger.info(f"SingleStepHighLevelNetwork初始化完成，输入维度: {history_input_dim}")
    
    def forward(self, history_obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            history_obs: [batch_size, K*28] 历史观测序列
            
        Returns:
            single_subgoal: [batch_size, 3] 单步子目标
            high_value: [batch_size, 1] 高层状态价值
        """
        # 特征提取
        features = self.feature_extractor(history_obs)
        
        # 单步子目标输出
        subgoal_raw = self.subgoal_head(features)
        single_subgoal = torch.tanh(subgoal_raw) * self.subgoal_bounds['position_offset']
        
        # 高层价值估计
        high_value = self.high_value_head(features)
        
        return single_subgoal, high_value


class SingleStepLowLevelNetwork(nn.Module):
    """
    B3低层策略网络（与原始低层策略相同，但接收重复的单步子目标）
    
    输入：64维低层观测 + 5×3维子目标序列（实际是单步重复）
    输出：4维控制命令
    """
    
    def __init__(self, config: AblationConfig):
        super().__init__()
        self.config = config
        
        # 输入维度：64维低层观测 + 5×3维子目标序列
        low_obs_dim = config.low_level_obs_dim  # 64
        subgoal_seq_dim = config.subgoal_horizon * 3  # 5×3=15 (但B3中实际是1×3重复5次)
        input_dim = low_obs_dim + subgoal_seq_dim  # 64+15=79
        
        # 特征提取层
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, config.low_level_features_dim),
            nn.ReLU(),
            nn.Linear(config.low_level_features_dim, config.low_level_features_dim),
            nn.ReLU(),
            nn.Linear(config.low_level_features_dim, config.low_level_features_dim // 2),
            nn.ReLU()
        )
        
        # 控制输出层
        feature_dim = config.low_level_features_dim // 2
        self.control_head = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 4)  # 4维控制输出
        )
        
        # 低层价值函数头
        self.low_value_head = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        # 控制命令限制
        self.control_bounds = {
            'v_linear': 1.0,    # 线速度限制 [m/s]
            'omega_z': 1.0      # 角速度限制 [rad/s]
        }
        
        logger.info(f"SingleStepLowLevelNetwork初始化完成，输入维度: {input_dim}")
    
    def forward(self, low_obs: torch.Tensor, subgoal_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            low_obs: [batch_size, 64] 低层观测
            subgoal_seq: [batch_size, 15] 子目标序列（B3中是单步重复）
            
        Returns:
            control_output: [batch_size, 4] 控制命令
            low_value: [batch_size, 1] 低层状态价值
        """
        # 拼接输入
        combined_input = torch.cat([low_obs, subgoal_seq], dim=1)
        
        # 特征提取
        features = self.feature_extractor(combined_input)
        
        # 控制输出
        control_raw = self.control_head(features)
        control_output = self._apply_control_bounds(control_raw)
        
        # 低层价值估计
        low_value = self.low_value_head(features)
        
        return control_output, low_value
    
    def _apply_control_bounds(self, control_raw: torch.Tensor) -> torch.Tensor:
        """应用控制命令限制"""
        # 分离线速度和角速度
        v_linear = control_raw[:, :3]  # [vx, vy, vz]
        omega_z = control_raw[:, 3:4]  # [ω_z]
        
        # 应用限制
        v_linear = torch.tanh(v_linear) * self.control_bounds['v_linear']
        omega_z = torch.tanh(omega_z) * self.control_bounds['omega_z']
        
        return torch.cat([v_linear, omega_z], dim=1)


class SingleStepHierarchicalPolicy:
    """
    B3消融实验：单步分层策略
    
    保持分层结构，但高层策略只生成单步子目标
    """
    
    def __init__(self, config: AblationConfig, device: str = "cpu"):
        self.config = config
        self.device = device
        
        # 验证配置
        assert config.experiment_group == "B3", f"配置错误：期望B3，得到{config.experiment_group}"
        assert config.subgoal_horizon == 1, f"B3配置subgoal_horizon应为1，得到{config.subgoal_horizon}"
        
        # 创建网络
        self.high_level_network = SingleStepHighLevelNetwork(config).to(device)
        self.low_level_network = SingleStepLowLevelNetwork(config).to(device)
        
        # 高层更新计数器
        self.step_counter = 0
        self.update_frequency = config.high_level_update_frequency
        
        # 缓存的单步子目标
        self.cached_single_subgoal = None
        
        logger.info("SingleStepHierarchicalPolicy初始化完成")
    
    def predict(self, history_obs: np.ndarray, low_obs: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, Dict]:
        """
        预测控制动作
        
        Args:
            history_obs: [K*28] 历史观测序列
            low_obs: [64] 低层观测
            deterministic: 是否确定性预测
            
        Returns:
            action: [4] 控制命令
            info: 预测信息
        """
        # 转换为tensor
        history_tensor = torch.FloatTensor(history_obs).unsqueeze(0).to(self.device)
        low_obs_tensor = torch.FloatTensor(low_obs).unsqueeze(0).to(self.device)
        
        # 检查是否需要更新高层策略
        if self.step_counter % self.update_frequency == 0:
            with torch.no_grad():
                single_subgoal, high_value = self.high_level_network(history_tensor)
                self.cached_single_subgoal = single_subgoal.cpu().numpy()[0]
        
        # 生成重复的子目标序列（单步子目标重复5次）
        repeated_subgoal = np.tile(self.cached_single_subgoal, (self.config.subgoal_horizon,))  # [15]
        subgoal_tensor = torch.FloatTensor(repeated_subgoal).unsqueeze(0).to(self.device)
        
        # 低层策略生成控制命令
        with torch.no_grad():
            control_output, low_value = self.low_level_network(low_obs_tensor, subgoal_tensor)
            action = control_output.cpu().numpy()[0]
        
        self.step_counter += 1
        
        info = {
            'policy_type': 'SingleStepHierarchical',
            'step_counter': self.step_counter,
            'update_cycle': self.step_counter % self.update_frequency,
            'high_level_active': True,
            'low_level_active': True,
            'single_subgoal': self.cached_single_subgoal.copy(),
            'repeated_subgoal_seq': repeated_subgoal.copy()
        }
        
        return action, info
    
    def predict_values(self, history_obs: np.ndarray, low_obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """预测状态价值"""
        history_tensor = torch.FloatTensor(history_obs).unsqueeze(0).to(self.device)
        low_obs_tensor = torch.FloatTensor(low_obs).unsqueeze(0).to(self.device)
        
        # 生成重复的子目标序列
        if self.cached_single_subgoal is not None:
            repeated_subgoal = np.tile(self.cached_single_subgoal, (self.config.subgoal_horizon,))
            subgoal_tensor = torch.FloatTensor(repeated_subgoal).unsqueeze(0).to(self.device)
        else:
            # 使用零子目标
            repeated_subgoal = np.zeros(self.config.subgoal_horizon * 3)
            subgoal_tensor = torch.FloatTensor(repeated_subgoal).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            _, high_value = self.high_level_network(history_tensor)
            _, low_value = self.low_level_network(low_obs_tensor, subgoal_tensor)
            
            return high_value.cpu().numpy()[0], low_value.cpu().numpy()[0]
    
    def forward_training(self, history_obs: torch.Tensor, low_obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        训练时的前向传播
        
        Args:
            history_obs: [batch_size, K*28] 历史观测批次
            low_obs: [batch_size, 64] 低层观测批次
            
        Returns:
            training_outputs: 训练输出字典
        """
        # 高层前向传播
        single_subgoal, high_value = self.high_level_network(history_obs)
        
        # 生成重复的子目标序列
        batch_size = single_subgoal.shape[0]
        repeated_subgoal = single_subgoal.unsqueeze(1).repeat(1, self.config.subgoal_horizon, 1)  # [batch, 5, 3]
        repeated_subgoal = repeated_subgoal.view(batch_size, -1)  # [batch, 15]
        
        # 低层前向传播
        control_output, low_value = self.low_level_network(low_obs, repeated_subgoal)
        
        # 计算log概率（用于PPO）
        # 高层子目标分布
        high_std = torch.ones_like(single_subgoal) * 0.1
        high_dist = torch.distributions.Normal(single_subgoal, high_std)
        high_log_probs = high_dist.log_prob(single_subgoal).sum(dim=1)
        
        # 低层控制分布
        low_std = torch.ones_like(control_output) * 0.1
        low_dist = torch.distributions.Normal(control_output, low_std)
        low_log_probs = low_dist.log_prob(control_output).sum(dim=1)
        
        return {
            'single_subgoal': single_subgoal,
            'repeated_subgoal': repeated_subgoal,
            'control_output': control_output,
            'high_value': high_value,
            'low_value': low_value,
            'high_log_probs': high_log_probs,
            'low_log_probs': low_log_probs
        }
    
    def evaluate_actions(self, history_obs: torch.Tensor, low_obs: torch.Tensor, 
                        high_actions: torch.Tensor, low_actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        评估动作（用于PPO训练）
        
        Args:
            history_obs: [batch_size, K*28] 历史观测
            low_obs: [batch_size, 64] 低层观测
            high_actions: [batch_size, 3] 高层动作（单步子目标）
            low_actions: [batch_size, 4] 低层动作（控制命令）
            
        Returns:
            evaluation_results: 评估结果字典
        """
        # 前向传播
        single_subgoal, high_value = self.high_level_network(history_obs)
        
        # 生成重复的子目标序列
        batch_size = single_subgoal.shape[0]
        repeated_subgoal = single_subgoal.unsqueeze(1).repeat(1, self.config.subgoal_horizon, 1)
        repeated_subgoal = repeated_subgoal.view(batch_size, -1)
        
        control_output, low_value = self.low_level_network(low_obs, repeated_subgoal)
        
        # 高层动作评估
        high_std = torch.ones_like(single_subgoal) * 0.1
        high_dist = torch.distributions.Normal(single_subgoal, high_std)
        high_log_probs = high_dist.log_prob(high_actions).sum(dim=1)
        high_entropy = high_dist.entropy().sum(dim=1)
        
        # 低层动作评估
        low_std = torch.ones_like(control_output) * 0.1
        low_dist = torch.distributions.Normal(control_output, low_std)
        low_log_probs = low_dist.log_prob(low_actions).sum(dim=1)
        low_entropy = low_dist.entropy().sum(dim=1)
        
        return {
            'high_values': high_value,
            'low_values': low_value,
            'high_log_probs': high_log_probs,
            'low_log_probs': low_log_probs,
            'high_entropy': high_entropy,
            'low_entropy': low_entropy
        }
    
    def get_policy_state(self) -> Dict:
        """获取策略状态（用于保存/恢复）"""
        return {
            'high_level_state_dict': self.high_level_network.state_dict(),
            'low_level_state_dict': self.low_level_network.state_dict(),
            'step_counter': self.step_counter,
            'cached_single_subgoal': self.cached_single_subgoal,
            'config': self.config.to_dict()
        }
    
    def load_policy_state(self, state: Dict):
        """加载策略状态"""
        self.high_level_network.load_state_dict(state['high_level_state_dict'])
        self.low_level_network.load_state_dict(state['low_level_state_dict'])
        self.step_counter = state['step_counter']
        self.cached_single_subgoal = state['cached_single_subgoal']
        logger.info("SingleStepHierarchicalPolicy状态加载完成")
    
    def reset(self):
        """重置策略状态"""
        self.step_counter = 0
        self.cached_single_subgoal = None
        logger.debug("SingleStepHierarchicalPolicy状态重置")


def create_single_step_hierarchical_policy(config: AblationConfig, device: str = "cpu") -> SingleStepHierarchicalPolicy:
    """
    创建B3单步分层策略
    
    Args:
        config: B3实验配置
        device: 计算设备
        
    Returns:
        SingleStepHierarchicalPolicy实例
    """
    if config.experiment_group != "B3":
        raise ValueError(f"配置错误：create_single_step_hierarchical_policy需要B3配置，得到{config.experiment_group}")
    
    return SingleStepHierarchicalPolicy(config, device)
