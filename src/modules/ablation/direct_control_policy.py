#!/usr/bin/env python3

"""
B1消融实验：直接控制策略
====================

数据流：86维obs → StateManager → K×28历史序列 → DirectControlPolicy → 4D控制

设计特点：
1. 保持高层策略，移除低层策略
2. 高层策略直接输出4D控制命令
3. 保持τ=5步更新周期和K=20步历史
4. 跳过低层策略处理
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


class DirectControlPolicyNetwork(nn.Module):
    """
    B1直接控制策略网络
    
    输入：K×28维历史序列观测
    输出：4维控制命令 [vx, vy, vz, ω_z]
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
        
        # 直接控制输出层
        # 输出4维控制：[vx, vy, vz, ω_z]
        feature_dim = config.high_level_features_dim // 2
        self.control_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4)  # 4维控制输出
        )
        
        # 价值函数头
        self.value_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # 控制命令限制
        self.control_bounds = {
            'v_linear': 1.0,    # 线速度限制 [m/s]
            'omega_z': 1.0      # 角速度限制 [rad/s]
        }
        
        logger.info(f"DirectControlPolicyNetwork初始化完成，输入维度: {history_input_dim}")
    
    def forward(self, history_obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            history_obs: [batch_size, K*28] 历史观测序列
            
        Returns:
            control_output: [batch_size, 4] 控制命令
            value: [batch_size, 1] 状态价值
        """
        # 特征提取
        features = self.feature_extractor(history_obs)
        
        # 控制输出
        control_raw = self.control_head(features)
        
        # 应用控制限制
        control_output = self._apply_control_bounds(control_raw)
        
        # 价值估计
        value = self.value_head(features)
        
        return control_output, value
    
    def _apply_control_bounds(self, control_raw: torch.Tensor) -> torch.Tensor:
        """应用控制命令限制"""
        # 分离线速度和角速度
        v_linear = control_raw[:, :3]  # [vx, vy, vz]
        omega_z = control_raw[:, 3:4]  # [ω_z]
        
        # 应用限制
        v_linear = torch.tanh(v_linear) * self.control_bounds['v_linear']
        omega_z = torch.tanh(omega_z) * self.control_bounds['omega_z']
        
        return torch.cat([v_linear, omega_z], dim=1)


class DirectControlPolicy:
    """
    B1消融实验：直接控制策略
    
    替代原始的分层策略，直接从历史观测生成控制命令
    """
    
    def __init__(self, config: AblationConfig, device: str = "cpu"):
        self.config = config
        self.device = device
        
        # 验证配置
        assert config.experiment_group == "B1", f"配置错误：期望B1，得到{config.experiment_group}"
        assert config.high_level_direct_control, "B1配置应启用high_level_direct_control"
        assert config.low_level_bypass, "B1配置应启用low_level_bypass"
        
        # 创建网络
        self.network = DirectControlPolicyNetwork(config).to(device)
        
        # 高层更新计数器
        self.step_counter = 0
        self.update_frequency = config.high_level_update_frequency
        
        # 缓存的控制命令（用于τ=5步周期）
        self.cached_control = None
        
        logger.info("DirectControlPolicy初始化完成")
    
    def predict(self, history_obs: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, Dict]:
        """
        预测控制动作
        
        Args:
            history_obs: [K*28] 历史观测序列
            deterministic: 是否确定性预测
            
        Returns:
            action: [4] 控制命令
            info: 预测信息
        """
        # 转换为tensor
        obs_tensor = torch.FloatTensor(history_obs).unsqueeze(0).to(self.device)
        
        # 检查是否需要更新控制命令
        if self.step_counter % self.update_frequency == 0:
            with torch.no_grad():
                control_output, value = self.network(obs_tensor)
                self.cached_control = control_output.cpu().numpy()[0]
        
        self.step_counter += 1
        
        # 返回缓存的控制命令
        action = self.cached_control.copy()
        
        info = {
            'policy_type': 'DirectControl',
            'step_counter': self.step_counter,
            'update_cycle': self.step_counter % self.update_frequency,
            'high_level_active': True,
            'low_level_active': False
        }
        
        return action, info
    
    def predict_values(self, history_obs: np.ndarray) -> np.ndarray:
        """预测状态价值"""
        obs_tensor = torch.FloatTensor(history_obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            _, value = self.network(obs_tensor)
            return value.cpu().numpy()[0]
    
    def forward_training(self, history_obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        训练时的前向传播
        
        Args:
            history_obs: [batch_size, K*28] 历史观测批次
            
        Returns:
            control_logits: [batch_size, 4] 控制命令logits
            control_values: [batch_size, 1] 状态价值
            log_probs: [batch_size] 动作log概率
        """
        control_output, value = self.network(history_obs)
        
        # 为了与PPO兼容，需要计算log概率
        # 这里使用高斯分布假设
        std = torch.ones_like(control_output) * 0.1  # 固定标准差
        dist = torch.distributions.Normal(control_output, std)
        log_probs = dist.log_prob(control_output).sum(dim=1)
        
        return control_output, value, log_probs
    
    def evaluate_actions(self, history_obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        评估动作（用于PPO训练）
        
        Args:
            history_obs: [batch_size, K*28] 观测
            actions: [batch_size, 4] 动作
            
        Returns:
            values: [batch_size, 1] 状态价值
            log_probs: [batch_size] 动作log概率
            entropy: [batch_size] 动作熵
        """
        control_output, values = self.network(history_obs)
        
        # 计算动作分布
        std = torch.ones_like(control_output) * 0.1
        dist = torch.distributions.Normal(control_output, std)
        
        # 计算log概率和熵
        log_probs = dist.log_prob(actions).sum(dim=1)
        entropy = dist.entropy().sum(dim=1)
        
        return values, log_probs, entropy
    
    def get_policy_state(self) -> Dict:
        """获取策略状态（用于保存/恢复）"""
        return {
            'network_state_dict': self.network.state_dict(),
            'step_counter': self.step_counter,
            'cached_control': self.cached_control,
            'config': self.config.to_dict()
        }
    
    def load_policy_state(self, state: Dict):
        """加载策略状态"""
        self.network.load_state_dict(state['network_state_dict'])
        self.step_counter = state['step_counter']
        self.cached_control = state['cached_control']
        logger.info("DirectControlPolicy状态加载完成")
    
    def reset(self):
        """重置策略状态"""
        self.step_counter = 0
        self.cached_control = None
        logger.debug("DirectControlPolicy状态重置")


def create_direct_control_policy(config: AblationConfig, device: str = "cpu") -> DirectControlPolicy:
    """
    创建B1直接控制策略
    
    Args:
        config: B1实验配置
        device: 计算设备
        
    Returns:
        DirectControlPolicy实例
    """
    if config.experiment_group != "B1":
        raise ValueError(f"配置错误：create_direct_control_policy需要B1配置，得到{config.experiment_group}")
    
    return DirectControlPolicy(config, device)
