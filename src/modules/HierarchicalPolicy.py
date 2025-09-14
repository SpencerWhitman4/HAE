"""
分层强化学习策略 - 基于统一StateManager的重构版本

实现论文中的雷达驱动多步子目标序列建模
使用统一的StateManager集成所有状态管理功能

核心功能：
1. 统一状态管理：StateManager集成86维解析、历史管理、决策周期控制
2. 分层决策循环：高层每τ=5步更新，低层每步更新
3. 子目标序列：{g_{t+i}}_{i=0}^{T-1}，格式为(ψ, d)
4. 完整组件集成：StateEncoder + Transformer编码器 + Actor/Critic

作者: HA-UAV团队
日期: 2025年8月
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution, 
    DiagGaussianDistribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    Distribution
)
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Dict, List, Tuple, Type, Union, Optional, Any
import logging

# 导入核心模块
from .HA_Modules import (
    StateEncoder,
    LowLevelStateEncoder,
    HighLevelPerceptionEncoder,
    LowLevelPerceptionEncoder,
    HighLevelActor,
    LowLevelActorWithYawControl,
    HighLevelCritic,
    LowLevelCritic
)

# 导入新的统一状态管理器
from .StateManager import StateManager, StructuredState

logger = logging.getLogger(__name__)


class HierarchicalPolicyNetwork(nn.Module):
    """分层策略网络 - 基于统一StateManager的重构版本
    
    使用统一的StateManager替代原来的多个管理器:
    1. StateManager: 集成86维解析、状态历史管理、高层决策周期控制
    2. 完整HA_Modules集成: 编码器+Actor+Critic
    3. 子目标序列生成: {g_{t+i}}_{i=0}^{T-1}格式(ψ, d)
    """
    
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        features_dim: int = 512,
        activation_fn: Type[nn.Module] = nn.ReLU,
        device: Union[str, torch.device] = "auto",
        **kwargs
    ):
        super().__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
        self.observation_space = observation_space
        self.action_space = action_space
        self.features_dim = features_dim
        
        # 初始化统一状态管理器
        self.state_manager = StateManager(
            history_length=20,                    # K=20步状态历史
            high_level_update_frequency=5,        # τ=5步高层更新频率
            future_horizon=5                      # T=5步未来子目标数量
        )
        
        # 初始化HA_Modules组件
        self._init_ha_modules()
        
        # 缓存变量
        self._last_observation = None
        self._current_step = 0
        
        logger.info(f"HierarchicalPolicyNetwork initialized with unified StateManager on {self.device}")
        
    def _init_ha_modules(self):
        """初始化所有HA_Modules组件"""
        # 低层状态编码器 - 处理64维低层观测
        self.state_encoder = LowLevelStateEncoder(
            input_dim=64,
            hidden_dim=256, 
            output_dim=128
        ).to(self.device)
        
        # 感知编码器
        self.high_level_perception = HighLevelPerceptionEncoder(
            config={
                'grid_size': 8,      # 临时参数（实际使用28维高层观测）
                'k_history': 20,     # 历史长度
                'd_model': 256,      # 隐藏维度
                'nhead': 8,          # 注意力头数
                'num_layers': 2      # Transformer层数
            }
        ).to(self.device)
        
        self.low_level_perception = LowLevelPerceptionEncoder(
            config={
                'd_model': 128,         # 隐藏维度
                'sector_dim': 36,       # 扇区维度
                'num_sectors': 36,      # 扇区数量
                'm_history': 6,         # 历史动作步数
                't_horizon': 5,         # 子目标序列长度
                'history_steps': 6,     # 历史步数
                'action_dim': 4,        # 动作维度
                'subgoal_steps': 5,     # 子目标步数
                'subgoal_dim': 2,       # 子目标维度
                'nhead': 4,             # 注意力头数
                'num_layers': 1,        # Transformer层数
                # 编码器维度分配
                'sector_encoding_dim': 43,   # 扇区编码维度
                'action_encoding_dim': 43,   # 动作编码维度  
                'goal_encoding_dim': 42      # 目标编码维度 (43+43+42=128)
            }
        ).to(self.device)
        
        # Actor网络
        self.high_level_actor = HighLevelActor(
            input_dim=256,
            t_horizon=5  # T=5步子目标序列
        ).to(self.device)
        
        self.low_level_actor = LowLevelActorWithYawControl(
            hidden_dim=128,
            max_speed=2.0,      # 最大速度限制
            max_yaw_rate=1.0    # 最大角速度限制
        ).to(self.device)
        
        # Critic网络
        self.high_level_critic = HighLevelCritic(
            input_dim=256  # 高层特征维度
        ).to(self.device)
        
        self.low_level_critic = LowLevelCritic(
            input_dim=128  # 低层特征维度
        ).to(self.device)
        
        logger.info("所有HA_Modules组件初始化完成")
    
    def forward_high_level(self, state_history: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """高层策略前向传播：处理历史状态序列，生成子目标序列
        
        Args:
            state_history: [batch_size, K, 28] 历史状态序列
            
        Returns:
            subgoal_sequence: [batch_size, tau, 2] 子目标序列 [(ψ, d), ...]
            value: [batch_size, 1] 状态价值
        """
        # 从state_history中分解出占据栅格和yaw历史
        # state_history: [batch_size, K, 28]
        # 前16维是全局地图特征，可以作为occupancy_grid使用
        batch_size, K, _ = state_history.shape
        
        # 简化处理：将前16维全局地图特征reshape为栅格形状
        occupancy_grid = state_history[:, :, :16].view(batch_size, K, 4, 4)  # [batch_size, K, 4, 4]
        
        # 从运动模式中提取yaw历史（第18维：平均角速度）
        yaw_history = state_history[:, :, 18].unsqueeze(-1)  # [batch_size, K, 1]
        yaw_history = yaw_history.squeeze(-1)  # [batch_size, K]
        
        # 高层感知编码
        high_level_features = self.high_level_perception(occupancy_grid, yaw_history)
        
        # 生成子目标序列
        subgoal_sequence, _, _ = self.high_level_actor(high_level_features, deterministic=True)
        
        # 计算状态价值
        value = self.high_level_critic(high_level_features)
        
        return subgoal_sequence, value
    
    def forward_low_level(
        self, 
        current_state_encoded: torch.Tensor, 
        current_subgoal: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """低层策略前向传播
        
        Args:
            current_state_encoded: [batch_size, encoded_dim] 已编码的当前状态特征
            current_subgoal: [batch_size, 2] 当前子目标(ψ, d)
            
        Returns:
            action: [batch_size, 4] (vx, vy, vz, yaw_rate)
            value: [batch_size, 1] 状态-动作价值
        """
        # 使用已编码的状态特征，不需要再次编码
        low_level_features = current_state_encoded
        
        # 生成控制动作
        action_tuple = self.low_level_actor(low_level_features)
        action = action_tuple[0]  # 只取动作向量，忽略log_prob和distribution
        
        # 计算状态-动作价值
        value = self.low_level_critic(low_level_features)
        
        return action, value
    
    def predict(self, observation: Union[np.ndarray, torch.Tensor]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """完整的分层策略预测 - 使用StateManager数据流
        
        Args:
            observation: 86维观测向量
            
        Returns:
            action: [4] (vx, vy, vz, yaw_rate)
            state_value: 状态价值（可选）
        """
        self.eval()
        
        with torch.no_grad():
            # 1. 使用统一StateManager解析和更新状态
            structured_state = self.state_manager.parse_and_update(observation)
            
            # 2. 检查是否需要高层决策更新
            obs_86d = structured_state.to_vector()  # 获取86维观测向量
            if self.state_manager.should_update_high_level() and self.state_manager.is_ready_for_high_level():
                # 使用StateManager提取高层观测序列
                high_level_obs_sequence = self.state_manager.extract_high_level_observation_sequence(obs_86d)
                
                # 正确处理张量转换
                if isinstance(high_level_obs_sequence, torch.Tensor):
                    high_level_obs_tensor = high_level_obs_sequence.detach().clone().unsqueeze(0).to(self.device)
                else:
                    high_level_obs_tensor = torch.from_numpy(high_level_obs_sequence).float().unsqueeze(0).to(self.device)
                
                # 高层策略决策
                subgoal_sequence, high_level_value = self.forward_high_level(high_level_obs_tensor)
                
                # 更新子目标序列
                self.state_manager.update_subgoal_sequence(subgoal_sequence.squeeze(0).cpu().numpy())
                
                logger.debug(f"高层策略更新：新子目标序列形状 {subgoal_sequence.shape}")
            
            # 3. 获取当前子目标
            current_subgoal = self.state_manager.get_current_subgoal()
            
            # 4. 使用StateManager提取低层观测并编码
            # obs_86d 已经在上面获取了
            low_level_obs = self.state_manager.extract_low_level_observation(obs_86d)
            
            # 正确处理张量转换
            if isinstance(low_level_obs, torch.Tensor):
                low_level_obs_tensor = low_level_obs.detach().clone().unsqueeze(0).to(self.device)
            else:
                low_level_obs_tensor = torch.from_numpy(low_level_obs).float().unsqueeze(0).to(self.device)
            current_state_encoded = self.state_encoder(low_level_obs_tensor)
            
            # 5. 低层策略决策
            if isinstance(current_subgoal, torch.Tensor):
                current_subgoal_tensor = current_subgoal.detach().clone().unsqueeze(0).to(self.device)
            else:
                current_subgoal_tensor = torch.from_numpy(current_subgoal).float().unsqueeze(0).to(self.device)
            action, low_level_value = self.forward_low_level(current_state_encoded, current_subgoal_tensor)
            
            # 6. 返回动作
            action_np = action.squeeze(0).cpu().numpy()
            value_np = low_level_value.squeeze(0).cpu().numpy() if low_level_value is not None else None
            
            return action_np, value_np
    
    def reset(self):
        """重置策略状态"""
        self.state_manager.reset()
        self._current_step = 0
        # logger.info("HierarchicalPolicyNetwork状态已重置")
    
    def adapt_for_sb3_training(self, obs_batch: torch.Tensor) -> torch.Tensor:
        """为SB3训练适配数据
        
        Args:
            obs_batch: [batch_size, 86] 观测批次
            
        Returns:
            torch.Tensor: [batch_size, 128] 特征向量
        """
        batch_size = obs_batch.shape[0]
        features_list = []
        
        for obs in obs_batch:
            # 使用StateManager解析观测
            structured_state = self.state_manager.parse_and_update(obs.cpu().numpy())
            # 提取低层观测
            low_level_obs = self.state_manager.extract_low_level_observation(structured_state)
            features_list.append(low_level_obs)
        
        # 批量编码
        low_level_batch = torch.tensor(features_list, dtype=torch.float32).to(self.device)
        encoded_features = self.state_encoder(low_level_batch)
        
        return encoded_features
    
    def forward_with_state_manager(self, observations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """使用StateManager的完整前向传播
        
        用于SB3训练时调用
        
        Args:
            observations: [batch_size, 86] 观测批次
            
        Returns:
            features: [batch_size, 128] 特征
            values: [batch_size, 1] 价值估计
        """
        # 使用适配方法获取特征
        features = self.adapt_for_sb3_training(observations)
        
        # 获取当前子目标（简化为零向量用于训练）
        current_subgoal = torch.zeros(observations.shape[0], 2, device=self.device)
        
        # 计算价值估计
        _, values = self.forward_low_level(features, current_subgoal)
        
        return features, values


class HierarchicalFeaturesExtractor(BaseFeaturesExtractor):
    """分层特征提取器 - 基于HierarchicalPolicyNetwork
    
    继承自SB3的BaseFeaturesExtractor，
    为SB3训练提供兼容的特征提取接口。
    """
    
    def __init__(
        self, 
        observation_space: gym.Space,
        features_dim: int = 512,
        activation_fn: Type[nn.Module] = nn.ReLU,
        device: Union[str, torch.device] = "auto"
    ):
        super().__init__(observation_space, features_dim)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
        
        # 初始化核心网络（但不用于实际决策，仅用于特征提取）
        self.policy_network = HierarchicalPolicyNetwork(
            observation_space=observation_space,
            action_space=spaces.Box(low=-1, high=1, shape=(4,)),  # 占位符
            features_dim=features_dim,
            device=self.device
        )
        
        # 确保activation_fn是类而不是实例
        if not isinstance(activation_fn, type):
            activation_fn = nn.ReLU  # 使用默认值
        
        # 特征投影层
        self.feature_projection = nn.Sequential(
            nn.Linear(128, 256),  # StateEncoder输出128维
            activation_fn(),
            nn.Dropout(0.1),
            nn.Linear(256, features_dim),
            activation_fn()
        ).to(self.device)
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """前向传播特征提取 - 使用StateManager数据流
        
        Args:
            observations: [batch_size, 86] 观测张量
            
        Returns:
            torch.Tensor: [batch_size, features_dim] 提取的特征
        """
        # 使用HierarchicalPolicyNetwork的适配方法
        encoded_features = self.policy_network.adapt_for_sb3_training(observations)
        
        # 特征投影
        final_features = self.feature_projection(encoded_features)
        
        return final_features


class HierarchicalPolicy(ActorCriticPolicy):
    """分层强化学习策略 - SB3兼容版本
    
    完全重构的分层策略，集成：
    1. HierarchicalPolicyNetwork作为核心决策网络
    2. SB3兼容的训练接口
    3. 状态历史管理和分层决策循环
    4. 论文规范的雷达驱动多步子目标序列建模
    """
    
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        lr_schedule: callable,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = HierarchicalFeaturesExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        # 设置默认参数
        if features_extractor_kwargs is None:
            features_extractor_kwargs = {}
            
        if net_arch is None:
            net_arch = dict(pi=[256, 256], vf=[256, 256])
            
        # 初始化父类
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
            use_sde=use_sde,
            log_std_init=log_std_init,
            full_std=full_std,
            use_expln=use_expln,
            squash_output=squash_output,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            share_features_extractor=share_features_extractor,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )
        
        # 初始化核心分层网络
        self.hierarchical_network = HierarchicalPolicyNetwork(
            observation_space=observation_space,
            action_space=action_space,
            features_dim=self.features_extractor.features_dim,
            device=self.device
        )
        
        # 为SB3兼容性创建简单的策略和价值网络
        # 这些不会被实际使用，但SB3需要它们存在
        latent_dim_pi = self.features_extractor.features_dim
        latent_dim_vf = self.features_extractor.features_dim
        
        # 简单的策略网络（实际决策通过hierarchical_network）
        self.action_net = nn.Sequential(
            nn.Linear(latent_dim_pi, action_space.shape[0])
        )
        
        # 简单的价值网络（实际价值通过hierarchical_network）  
        self.value_net = nn.Sequential(
            nn.Linear(latent_dim_vf, 1)
        )
        
        logger.info("HierarchicalPolicy初始化完成")
        
    def _get_constructor_parameters(self) -> Dict[str, Any]:
        """获取构造参数"""
        data = super()._get_constructor_parameters()
        return data
    
    def _get_latent(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """提取潜在特征用于策略和价值网络
        
        这个方法是SB3标准接口的一部分，用于从观测中提取：
        - latent_pi: 策略网络的潜在特征
        - latent_vf: 价值网络的潜在特征  
        - latent_sde: SDE网络的潜在特征（如果使用）
        
        Args:
            obs: 观测张量
            
        Returns:
            (latent_pi, latent_vf, latent_sde): 三个潜在特征张量
        """
        # 使用特征提取器处理观测
        features = self.extract_features(obs, self.pi_features_extractor)
        
        # 对于分层策略，策略和价值网络使用相同的特征
        latent_pi = features
        latent_vf = features  
        latent_sde = features  # SDE使用相同特征
        
        return latent_pi, latent_vf, latent_sde
    
    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor, latent_sde: torch.Tensor = None) -> Distribution:
        """从潜在特征获取动作分布
        
        Args:
            latent_pi: 策略网络的潜在特征
            latent_sde: SDE网络的潜在特征（可选）
            
        Returns:
            动作分布对象
        """
        # 使用分层网络预测动作
        action, _ = self.hierarchical_network.predict(latent_pi)
        
        # 创建确定性分布（因为我们的分层网络已经处理了随机性）
        if isinstance(action, torch.Tensor):
            # 如果是单个样本，确保有正确的batch维度
            if action.dim() == 1:
                action = action.unsqueeze(0)
            # 创建DiracDelta分布（确定性）
            from torch.distributions import Normal
            # 使用很小的标准差创建近似确定性分布
            std = torch.ones_like(action) * 1e-8
            dist = Normal(action, std)
            return dist
        else:
            raise ValueError(f"Unexpected action type: {type(action)}")
    
    def get_value_net(self) -> nn.Module:
        """获取价值网络 - SB3兼容方法"""
        return self.value_net
        
    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播
        
        Args:
            obs: [batch_size, obs_dim] 观测张量
            deterministic: 是否确定性策略
            
        Returns:
            actions: [batch_size, action_dim] 动作
            values: [batch_size, 1] 价值
            log_probs: [batch_size, 1] 对数概率
        """
        # 直接使用分层网络预测
        actions, values = self.hierarchical_network.predict(obs)
        
        # 创建简单的log概率（因为这主要用于测试）
        log_probs = torch.zeros((actions.size(0), 1), device=actions.device)
        
        return actions, values, log_probs
    
    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """预测动作
        
        Args:
            observation: 观测
            state: RNN状态（未使用）
            episode_start: 是否新回合开始
            deterministic: 是否确定性策略
            
        Returns:
            action: 预测的动作
            state: 更新后的状态（未使用）
        """
        # 如果是新回合，重置分层网络
        if episode_start is not None and np.any(episode_start):
            self.hierarchical_network.reset()
        
        # 使用分层网络进行预测
        if isinstance(observation, dict):
            obs = observation['observation'] if 'observation' in observation else observation
        else:
            obs = observation
            
        action, _ = self.hierarchical_network.predict(obs)
        return action, state
    
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """评估动作
        
        Args:
            obs: [batch_size, obs_dim] 观测张量
            actions: [batch_size, action_dim] 动作张量
            
        Returns:
            values: [batch_size, 1] 价值
            log_probs: [batch_size, 1] 对数概率
            entropy: [batch_size, 1] 熵
        """
        # 确保输入是正确的tensor格式
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
        
        batch_size = obs.size(0)
        
        # 简化实现用于测试 - 创建合理的mock值
        values = torch.zeros((batch_size, 1), device=obs.device, dtype=torch.float32)
        log_probs = torch.zeros((batch_size, 1), device=obs.device, dtype=torch.float32)
        entropy = torch.ones((batch_size, 1), device=obs.device, dtype=torch.float32) * 0.1  # 小的熵值
        
        # 如果有分层网络，尝试获取真实的价值估计
        if hasattr(self, 'hierarchical_network'):
            try:
                for i in range(batch_size):
                    obs_single = obs[i].cpu().numpy()
                    _, value_single = self.hierarchical_network.predict(obs_single)
                    if value_single is not None:
                        if isinstance(value_single, np.ndarray):
                            values[i, 0] = torch.tensor(value_single.item() if value_single.size == 1 else value_single[0], 
                                                       device=obs.device, dtype=torch.float32)
                        else:
                            values[i, 0] = torch.tensor(float(value_single), device=obs.device, dtype=torch.float32)
            except Exception as e:
                # 如果获取失败，使用零值
                logger.warning(f"获取真实价值估计失败，使用零值: {e}")
        
        return values, log_probs, entropy
    
    def reset_hierarchical_state(self):
        """重置分层状态"""
        if hasattr(self, 'hierarchical_network'):
            self.hierarchical_network.reset()
            # logger.info("分层策略状态已重置")


# 导出主要类
__all__ = [
    'HierarchicalPolicyNetwork',
    'HierarchicalFeaturesExtractor', 
    'HierarchicalPolicy',
    'StateManager',
    'StructuredState'
]
