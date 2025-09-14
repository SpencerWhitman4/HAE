#!/usr/bin/env python3

"""
B2消融实验：扁平策略
================

数据流：86维obs → StateManager → 86维原始obs → FlatPolicy → 4D控制

设计特点：
1. 移除高层策略，保持低层策略形式
2. 直接从86维原始观测到4D控制
3. 不使用分层结构和历史序列
4. 单一网络端到端学习
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


class FlatPolicyNetwork(nn.Module):
    """
    B2扁平策略网络
    
    输入：86维原始观测
    输出：4维控制命令 [vx, vy, vz, ω_z]
    """
    
    def __init__(self, config: AblationConfig):
        super().__init__()
        self.config = config
        
        # 输入维度：86维原始观测
        input_dim = config.flat_policy_input_dim  # 86
        
        # 特征提取层（更深的网络补偿分层结构的缺失）
        self.feature_extractor = nn.Sequential(
            # 第一层：原始观测编码
            nn.Linear(input_dim, config.low_level_features_dim * 2),  # 86 -> 256
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # 第二层：特征提取
            nn.Linear(config.low_level_features_dim * 2, config.low_level_features_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # 第三层：特征压缩
            nn.Linear(config.low_level_features_dim * 2, config.low_level_features_dim),
            nn.ReLU(),
            
            # 第四层：特征精炼
            nn.Linear(config.low_level_features_dim, config.low_level_features_dim),
            nn.ReLU()
        )
        
        # 控制输出层
        # 输出4维控制：[vx, vy, vz, ω_z]
        self.control_head = nn.Sequential(
            nn.Linear(config.low_level_features_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4)  # 4维控制输出
        )
        
        # 价值函数头
        self.value_head = nn.Sequential(
            nn.Linear(config.low_level_features_dim, 64),
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
        
        logger.info(f"FlatPolicyNetwork初始化完成，输入维度: {input_dim}")
    
    def forward(self, raw_obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            raw_obs: [batch_size, 86] 原始观测
            
        Returns:
            control_output: [batch_size, 4] 控制命令
            value: [batch_size, 1] 状态价值
        """
        # 特征提取
        features = self.feature_extractor(raw_obs)
        
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


class FlatPolicy:
    """
    B2消融实验：扁平策略
    
    直接从原始观测生成控制命令，不使用分层结构
    """
    
    def __init__(self, config: AblationConfig, device: str = "cpu"):
        self.config = config
        self.device = device
        
        # 验证配置
        assert config.experiment_group == "B2", f"配置错误：期望B2，得到{config.experiment_group}"
        assert not config.use_hierarchical, "B2配置应禁用use_hierarchical"
        assert not config.use_high_level_policy, "B2配置应禁用use_high_level_policy"
        
        # 创建网络
        self.network = FlatPolicyNetwork(config).to(device)
        
        # 步数计数器（用于记录）
        self.step_counter = 0
        
        logger.info("FlatPolicy初始化完成")
    
    def predict(self, raw_obs: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, Dict]:
        """
        预测控制动作
        
        Args:
            raw_obs: [86] 原始观测
            deterministic: 是否确定性预测
            
        Returns:
            action: [4] 控制命令
            info: 预测信息
        """
        # 转换为tensor
        obs_tensor = torch.FloatTensor(raw_obs).unsqueeze(0).to(self.device)
        
        # 前向传播
        with torch.no_grad():
            control_output, value = self.network(obs_tensor)
            action = control_output.cpu().numpy()[0]
        
        self.step_counter += 1
        
        info = {
            'policy_type': 'Flat',
            'step_counter': self.step_counter,
            'high_level_active': False,
            'low_level_active': True,
            'input_dim': raw_obs.shape[0],
            'output_dim': action.shape[0]
        }
        
        return action, info
    
    def predict_values(self, raw_obs: np.ndarray) -> np.ndarray:
        """预测状态价值"""
        obs_tensor = torch.FloatTensor(raw_obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            _, value = self.network(obs_tensor)
            return value.cpu().numpy()[0]
    
    def forward_training(self, raw_obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        训练时的前向传播
        
        Args:
            raw_obs: [batch_size, 86] 原始观测批次
            
        Returns:
            control_logits: [batch_size, 4] 控制命令logits
            control_values: [batch_size, 1] 状态价值
            log_probs: [batch_size] 动作log概率
        """
        control_output, value = self.network(raw_obs)
        
        # 为了与PPO兼容，计算log概率
        # 使用高斯分布假设
        std = torch.ones_like(control_output) * 0.1  # 固定标准差
        dist = torch.distributions.Normal(control_output, std)
        log_probs = dist.log_prob(control_output).sum(dim=1)
        
        return control_output, value, log_probs
    
    def evaluate_actions(self, raw_obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        评估动作（用于PPO训练）
        
        Args:
            raw_obs: [batch_size, 86] 观测
            actions: [batch_size, 4] 动作
            
        Returns:
            values: [batch_size, 1] 状态价值
            log_probs: [batch_size] 动作log概率
            entropy: [batch_size] 动作熵
        """
        control_output, values = self.network(raw_obs)
        
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
            'config': self.config.to_dict()
        }
    
    def load_policy_state(self, state: Dict):
        """加载策略状态"""
        self.network.load_state_dict(state['network_state_dict'])
        self.step_counter = state['step_counter']
        logger.info("FlatPolicy状态加载完成")
    
    def reset(self):
        """重置策略状态"""
        self.step_counter = 0
        logger.debug("FlatPolicy状态重置")
    
    def get_feature_visualization(self, raw_obs: np.ndarray) -> Dict[str, np.ndarray]:
        """
        获取特征可视化（用于分析）
        
        Args:
            raw_obs: [86] 原始观测
            
        Returns:
            features: 各层特征字典
        """
        obs_tensor = torch.FloatTensor(raw_obs).unsqueeze(0).to(self.device)
        
        features = {}
        with torch.no_grad():
            # 逐层前向传播
            x = obs_tensor
            for i, layer in enumerate(self.network.feature_extractor):
                if isinstance(layer, nn.Linear):
                    x = layer(x)
                    features[f'layer_{i}_linear'] = x.cpu().numpy()[0]
                elif isinstance(layer, nn.ReLU):
                    x = layer(x)
                    features[f'layer_{i}_relu'] = x.cpu().numpy()[0]
                elif isinstance(layer, nn.Dropout):
                    x = layer(x)
            
            # 控制输出
            control_output, value = self.network(obs_tensor)
            features['control_output'] = control_output.cpu().numpy()[0]
            features['value'] = value.cpu().numpy()[0]
        
        return features


def create_flat_policy(config: AblationConfig, device: str = "cpu") -> FlatPolicy:
    """
    创建B2扁平策略
    
    Args:
        config: B2实验配置
        device: 计算设备
        
    Returns:
        FlatPolicy实例
    """
    if config.experiment_group != "B2":
        raise ValueError(f"配置错误：create_flat_policy需要B2配置，得到{config.experiment_group}")
    
    return FlatPolicy(config, device)
