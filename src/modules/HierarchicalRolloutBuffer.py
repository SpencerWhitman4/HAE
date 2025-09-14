"""
分层强化学习滚动缓冲区 - 统一数据流版本

核心设计原则：
1. 直接使用StateManager的观测提取方法
2. 与HierarchicalPolicy的数据流完全匹配
3. 支持分层GAE优势估计
4. 统一的批处理采样机制

数据流架构：
86维观测 → StateManager → 64维低层观测 + 28维×K高层观测 → 分层缓冲区存储
"""

import warnings
from typing import Dict, Generator, List, Optional, Union, NamedTuple
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.vec_env import VecNormalize
import logging

# 导入StateManager
from .StateManager import StateManager

logger = logging.getLogger(__name__)


class HierarchicalRolloutBufferSamples(NamedTuple):
    """分层滚动缓冲区样本数据结构"""
    # 标准PPO数据
    observations: torch.Tensor          # [batch_size, 86] 原始观测
    actions: torch.Tensor              # [batch_size, 4] 执行动作
    old_values: torch.Tensor           # [batch_size] 旧价值估计
    old_log_prob: torch.Tensor         # [batch_size] 旧动作概率
    advantages: torch.Tensor           # [batch_size] 优势估计
    returns: torch.Tensor              # [batch_size] 回报
    
    # 分层特有数据
    low_level_observations: torch.Tensor    # [batch_size, 64] 低层观测
    high_level_observations: torch.Tensor   # [batch_size, K, 28] 高层观测序列
    high_level_actions: torch.Tensor        # [batch_size, 10] 高层动作(5×2子目标)
    low_level_actions: torch.Tensor         # [batch_size, 4] 低层动作
    
    # 分层价值和优势
    high_level_values: torch.Tensor         # [batch_size] 高层价值
    low_level_values: torch.Tensor          # [batch_size] 低层价值
    high_level_advantages: torch.Tensor     # [batch_size] 高层优势
    low_level_advantages: torch.Tensor      # [batch_size] 低层优势
    high_level_returns: torch.Tensor        # [batch_size] 高层回报
    low_level_returns: torch.Tensor         # [batch_size] 低层回报
    
    # 更新掩码
    high_level_update_mask: torch.Tensor    # [batch_size] 高层更新掩码
    low_level_update_mask: torch.Tensor     # [batch_size] 低层更新掩码


class HierarchicalRolloutBuffer(RolloutBuffer):
    """分层强化学习滚动缓冲区 - 统一数据流版本"""
    
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[torch.device, str] = "auto",
        gae_lambda: float = 0.95,
        gamma: float = 0.99,
        n_envs: int = 1,
        hierarchical_config: Optional[Dict] = None
    ):
        """初始化分层滚动缓冲区"""
        
        # 验证观测空间
        assert isinstance(observation_space, spaces.Box), "观测空间必须是Box类型"
        assert observation_space.shape == (86,), f"观测空间维度必须是(86,)，实际为{observation_space.shape}"
        
        # 分层配置
        self.hierarchical_config = hierarchical_config or {}
        self.high_level_update_freq = self.hierarchical_config.get('high_level_update_freq', 5)
        self.history_length = self.hierarchical_config.get('history_length', 5)
        
        # 初始化StateManager用于数据转换
        self.state_manager = StateManager(
            history_length=self.history_length,
            high_level_update_frequency=self.high_level_update_freq,
            future_horizon=5
        )
        
        # 计算分层观测维度
        self.low_level_obs_dim = 64      # StateManager.extract_low_level_observation
        self.high_level_obs_dim = 28     # StateManager.extract_high_level_observation_sequence每步维度
        self.high_level_seq_length = self.history_length  # K步序列长度
        self.high_action_dim = 10        # 5个子目标 × 2维(ψ,d)
        self.low_action_dim = action_space.shape[0]  # 4维控制动作
        
        # 调用父类构造函数
        super().__init__(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            gae_lambda=gae_lambda,
            gamma=gamma,
            n_envs=n_envs
        )
        
        # 父类初始化完成后，初始化分层存储数组
        self._init_hierarchical_arrays()
        
        logger.debug(f"HierarchicalRolloutBuffer初始化完成: "
                   f"低层观测{self.low_level_obs_dim}维, "
                   f"高层观测{self.high_level_seq_length}×{self.high_level_obs_dim}维")
    
    def _init_hierarchical_arrays(self):
        """初始化分层存储数组"""
        # 低层观测 [buffer_size, n_envs, 64]
        self.low_level_observations = np.zeros(
            (self.buffer_size, self.n_envs, self.low_level_obs_dim), dtype=np.float32
        )
        
        # 高层观测序列 [buffer_size, n_envs, K, 28] 
        self.high_level_observations = np.zeros(
            (self.buffer_size, self.n_envs, self.high_level_seq_length, self.high_level_obs_dim), 
            dtype=np.float32
        )
        
        # 分层动作
        self.high_level_actions = np.zeros(
            (self.buffer_size, self.n_envs, self.high_action_dim), dtype=np.float32
        )
        self.low_level_actions = np.zeros(
            (self.buffer_size, self.n_envs, self.low_action_dim), dtype=np.float32
        )
        
        # 分层价值估计
        self.high_level_values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.low_level_values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        
        # 分层优势和回报
        self.high_level_advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.low_level_advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.high_level_returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.low_level_returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        
        # 更新掩码
        self.high_level_update_mask = np.zeros((self.buffer_size, self.n_envs), dtype=np.bool_)
        self.low_level_update_mask = np.ones((self.buffer_size, self.n_envs), dtype=np.bool_)  # 低层每步更新
        
        logger.debug("分层存储数组初始化完成")
    
    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: torch.Tensor,
        log_prob: torch.Tensor,
        # 分层特有参数 - 直接来自HierarchicalPolicy
        high_level_action: Optional[np.ndarray] = None,
        high_level_value: Optional[torch.Tensor] = None,
        low_level_value: Optional[torch.Tensor] = None,
        should_update_high_level: bool = False,
    ) -> None:
        """添加一步经验到缓冲区
        
        Args:
            obs: [n_envs, 86] 原始观测
            action: [n_envs, 4] 执行的动作(低层)
            reward: [n_envs] 环境奖励
            episode_start: [n_envs] 是否为新轮次开始
            value: [n_envs] 综合价值估计
            log_prob: [n_envs] 动作对数概率
            
            high_level_action: [n_envs, 10] 高层动作(5×2子目标)，可为None
            high_level_value: [n_envs] 高层价值估计
            low_level_value: [n_envs] 低层价值估计  
            should_update_high_level: 当前步是否更新高层
        """
        # 调用父类add方法存储基础数据
        super().add(obs, action, reward, episode_start, value, log_prob)
        
        # 使用StateManager转换观测数据
        obs_tensor = torch.from_numpy(obs).float()
        
        # 提取低层观测 [n_envs, 64]
        low_level_obs = self.state_manager.extract_low_level_observation(obs_tensor)
        
        # 提取高层观测序列 [n_envs, K, 28]
        high_level_obs = self.state_manager.extract_high_level_observation_sequence(
            obs_tensor, K=self.high_level_seq_length
        )
        
        # 获取当前位置
        current_pos = (self.pos - 1) % self.buffer_size  # 父类add已经更新了pos
        
        # 存储分层观测数据
        self.low_level_observations[current_pos] = low_level_obs.numpy()
        self.high_level_observations[current_pos] = high_level_obs.numpy()
        
        # 存储分层动作
        self.low_level_actions[current_pos] = action.copy()
        if high_level_action is not None:
            self.high_level_actions[current_pos] = high_level_action.copy()
        
        # 存储分层价值
        if high_level_value is not None:
            if hasattr(high_level_value, 'detach'):
                self.high_level_values[current_pos] = high_level_value.detach().cpu().numpy().flatten()
            else:
                self.high_level_values[current_pos] = np.array(high_level_value).flatten()
                
        if low_level_value is not None:
            if hasattr(low_level_value, 'detach'):
                self.low_level_values[current_pos] = low_level_value.detach().cpu().numpy().flatten()
            else:
                self.low_level_values[current_pos] = np.array(low_level_value).flatten()
        else:
            if hasattr(value, 'detach'):
                self.low_level_values[current_pos] = value.detach().cpu().numpy().flatten()
            else:
                self.low_level_values[current_pos] = np.array(value).flatten()
        
        # 存储更新掩码
        self.high_level_update_mask[current_pos] = should_update_high_level
        self.low_level_update_mask[current_pos] = True  # 低层每步更新
        
        logger.debug(f"添加分层经验到缓冲区位置: {current_pos}")
    
    def compute_returns_and_advantage(self, last_values: torch.Tensor, dones: np.ndarray) -> None:
        """计算回报和优势函数 - 分层版本"""
        # 调用父类方法计算综合优势
        super().compute_returns_and_advantage(last_values, dones)
        
        # 计算分层优势
        self._compute_hierarchical_gae_advantages(last_values, dones)
        
        logger.debug("分层GAE优势计算完成")
    
    def _compute_hierarchical_gae_advantages(self, last_values: torch.Tensor, dones: np.ndarray):
        """计算分层GAE优势估计"""
        # 分解最后价值为高层和低层
        last_high_values = last_values * 0.4  # 假设高层占40%
        last_low_values = last_values * 0.6   # 低层占60%
        
        # 高层GAE - 使用高层更新掩码
        self._compute_masked_gae(
            values=self.high_level_values,
            last_values=last_high_values.detach().cpu().numpy() if hasattr(last_high_values, 'detach') else np.array(last_high_values),
            rewards=self.rewards * 0.4,  # 高层获得40%的奖励
            dones=dones,
            update_mask=self.high_level_update_mask,
            advantages_out=self.high_level_advantages,
            returns_out=self.high_level_returns,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda
        )
        
        # 低层GAE - 每步都更新
        self._compute_masked_gae(
            values=self.low_level_values,
            last_values=last_low_values.detach().cpu().numpy() if hasattr(last_low_values, 'detach') else np.array(last_low_values),
            rewards=self.rewards * 0.6,  # 低层获得60%的奖励
            dones=dones,
            update_mask=self.low_level_update_mask,
            advantages_out=self.low_level_advantages,
            returns_out=self.low_level_returns,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda
        )
        
        logger.debug("分层GAE计算完成")
    
    def _compute_masked_gae(
        self, 
        values: np.ndarray,
        last_values: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        update_mask: np.ndarray,
        advantages_out: np.ndarray,
        returns_out: np.ndarray,
        gamma: float,
        gae_lambda: float
    ):
        """带掩码的GAE计算"""
        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
                next_mask = np.ones_like(dones, dtype=np.bool_)
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = values[step + 1]
                next_mask = update_mask[step + 1]
            
            # 只在更新步骤计算优势
            current_mask = update_mask[step]
            
            delta = rewards[step] + gamma * next_values * next_non_terminal - values[step]
            last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam * next_mask.astype(float)
            
            # 应用掩码
            advantages_out[step] = last_gae_lam * current_mask.astype(float)
        
        # 计算回报
        returns_out[:] = advantages_out + values
    
    def get(self, batch_size: Optional[int] = None) -> Generator[HierarchicalRolloutBufferSamples, None, None]:
        """获取训练批次数据 - 分层版本"""
        assert self.full, "缓冲区必须已满才能采样"
        
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        start_idx = 0
        
        while start_idx < len(indices):
            # 计算批次大小
            if batch_size is None:
                batch_indices = indices[start_idx:]
                start_idx = len(indices)
            else:
                batch_indices = indices[start_idx : start_idx + batch_size]
                start_idx += batch_size
            
            yield self._get_hierarchical_samples(batch_indices)
        
        logger.debug(f"分层批次数据获取完成，总样本数: {len(indices)}")
    
    def _get_hierarchical_samples(self, batch_indices: np.ndarray) -> HierarchicalRolloutBufferSamples:
        """获取分层样本数据"""
        # 计算2D索引
        env_indices = batch_indices % self.n_envs
        time_indices = batch_indices // self.n_envs
        
        # 基础数据
        base_data = {
            'observations': self.to_torch(self.observations[time_indices, env_indices]),
            'actions': self.to_torch(self.actions[time_indices, env_indices]),
            'old_values': self.to_torch(self.values[time_indices, env_indices].flatten()),
            'old_log_prob': self.to_torch(self.log_probs[time_indices, env_indices].flatten()),
            'advantages': self.to_torch(self.advantages[time_indices, env_indices].flatten()),
            'returns': self.to_torch(self.returns[time_indices, env_indices].flatten()),
        }
        
        # 分层观测数据
        low_level_obs = self.to_torch(self.low_level_observations[time_indices, env_indices])
        high_level_obs = self.to_torch(self.high_level_observations[time_indices, env_indices])
        
        # 分层动作数据
        high_level_actions = self.to_torch(self.high_level_actions[time_indices, env_indices])
        low_level_actions = self.to_torch(self.low_level_actions[time_indices, env_indices])
        
        # 分层价值和优势
        hierarchical_data = {
            'low_level_observations': low_level_obs,
            'high_level_observations': high_level_obs,
            'high_level_actions': high_level_actions,
            'low_level_actions': low_level_actions,
            'high_level_values': self.to_torch(self.high_level_values[time_indices, env_indices].flatten()),
            'low_level_values': self.to_torch(self.low_level_values[time_indices, env_indices].flatten()),
            'high_level_advantages': self.to_torch(self.high_level_advantages[time_indices, env_indices].flatten()),
            'low_level_advantages': self.to_torch(self.low_level_advantages[time_indices, env_indices].flatten()),
            'high_level_returns': self.to_torch(self.high_level_returns[time_indices, env_indices].flatten()),
            'low_level_returns': self.to_torch(self.low_level_returns[time_indices, env_indices].flatten()),
            'high_level_update_mask': self.to_torch(self.high_level_update_mask[time_indices, env_indices].flatten().astype(np.float32)),
            'low_level_update_mask': self.to_torch(self.low_level_update_mask[time_indices, env_indices].flatten().astype(np.float32)),
        }
        
        # 合并数据
        base_data.update(hierarchical_data)
        
        return HierarchicalRolloutBufferSamples(**base_data)
    
    def reset(self) -> None:
        """重置缓冲区"""
        super().reset()
        
        # 重置分层数组（如果已初始化）
        if hasattr(self, 'low_level_observations'):
            self.low_level_observations.fill(0)
            self.high_level_observations.fill(0)
            self.high_level_actions.fill(0)
            self.low_level_actions.fill(0)
            self.high_level_values.fill(0)
            self.low_level_values.fill(0)
            self.high_level_advantages.fill(0)
            self.low_level_advantages.fill(0)
            self.high_level_returns.fill(0)
            self.low_level_returns.fill(0)
            self.high_level_update_mask.fill(False)
            self.low_level_update_mask.fill(True)
        
        # 重置StateManager（如果已初始化）
        if hasattr(self, 'state_manager'):
            try:
                self.state_manager.reset()
            except Exception as e:
                logger.warning(f"StateManager重置失败: {e}")
        
        # logger.info("分层滚动缓冲区已重置")
    
    def get_hierarchical_statistics(self) -> Dict[str, float]:
        """获取分层训练统计信息"""
        if not self.full:
            return {}
        
        # 基于更新掩码的统计
        high_level_active_steps = np.sum(self.high_level_update_mask)
        low_level_active_steps = np.sum(self.low_level_update_mask)
        
        stats = {
            # 高层统计
            'high_level_update_ratio': float(high_level_active_steps) / (self.buffer_size * self.n_envs),
            'high_level_value_mean': float(np.mean(self.high_level_values[self.high_level_update_mask])) if high_level_active_steps > 0 else 0.0,
            'high_level_advantage_mean': float(np.mean(self.high_level_advantages[self.high_level_update_mask])) if high_level_active_steps > 0 else 0.0,
            'high_level_advantage_std': float(np.std(self.high_level_advantages[self.high_level_update_mask])) if high_level_active_steps > 0 else 0.0,
            
            # 低层统计  
            'low_level_update_ratio': float(low_level_active_steps) / (self.buffer_size * self.n_envs),
            'low_level_value_mean': float(np.mean(self.low_level_values[self.low_level_update_mask])) if low_level_active_steps > 0 else 0.0,
            'low_level_advantage_mean': float(np.mean(self.low_level_advantages[self.low_level_update_mask])) if low_level_active_steps > 0 else 0.0,
            'low_level_advantage_std': float(np.std(self.low_level_advantages[self.low_level_update_mask])) if low_level_active_steps > 0 else 0.0,
            
            # 数据维度信息
            'low_level_obs_dim': self.low_level_obs_dim,
            'high_level_obs_seq_length': self.high_level_seq_length,
            'high_level_obs_dim': self.high_level_obs_dim,
        }
        
        return stats
    
    def extract_training_data_for_policy(self, policy_network) -> Dict[str, torch.Tensor]:
        """为HierarchicalPolicy提取训练数据
        
        Returns:
            dict: 包含所有训练所需数据的字典
        """
        assert self.full, "缓冲区必须已满"
        
        # 扁平化所有数据
        batch_size = self.buffer_size * self.n_envs
        
        training_data = {
            # 原始观测 - 86维
            'observations': torch.from_numpy(self.observations.reshape(batch_size, -1)).float(),
            
            # 分层观测
            'low_level_observations': torch.from_numpy(self.low_level_observations.reshape(batch_size, -1)).float(),
            'high_level_observations': torch.from_numpy(self.high_level_observations.reshape(batch_size, self.high_level_seq_length, -1)).float(),
            
            # 分层动作
            'high_level_actions': torch.from_numpy(self.high_level_actions.reshape(batch_size, -1)).float(),
            'low_level_actions': torch.from_numpy(self.low_level_actions.reshape(batch_size, -1)).float(),
            
            # 分层价值和优势
            'high_level_values': torch.from_numpy(self.high_level_values.reshape(batch_size)).float(),
            'low_level_values': torch.from_numpy(self.low_level_values.reshape(batch_size)).float(),
            'high_level_advantages': torch.from_numpy(self.high_level_advantages.reshape(batch_size)).float(),
            'low_level_advantages': torch.from_numpy(self.low_level_advantages.reshape(batch_size)).float(),
            'high_level_returns': torch.from_numpy(self.high_level_returns.reshape(batch_size)).float(),
            'low_level_returns': torch.from_numpy(self.low_level_returns.reshape(batch_size)).float(),
            
            # 更新掩码
            'high_level_update_mask': torch.from_numpy(self.high_level_update_mask.reshape(batch_size)).float(),
            'low_level_update_mask': torch.from_numpy(self.low_level_update_mask.reshape(batch_size)).float(),
        }
        
        return training_data


def create_hierarchical_rollout_buffer(
    buffer_size: int,
    observation_space: spaces.Space,
    action_space: spaces.Space,
    device: Union[torch.device, str] = "auto",
    gae_lambda: float = 0.95,
    gamma: float = 0.99,
    n_envs: int = 1,
    hierarchical_config: Optional[Dict] = None
) -> HierarchicalRolloutBuffer:
    """创建分层滚动缓冲区的便捷函数"""
    
    logger.info(f"创建分层滚动缓冲区:")
    logger.info(f"  缓冲区大小: {buffer_size}")
    logger.info(f"  观测空间: {observation_space}")
    logger.info(f"  动作空间: {action_space}")
    logger.info(f"  GAE lambda: {gae_lambda}")
    logger.info(f"  折扣因子: {gamma}")
    logger.info(f"  并行环境数: {n_envs}")
    
    return HierarchicalRolloutBuffer(
        buffer_size=buffer_size,
        observation_space=observation_space,
        action_space=action_space,
        device=device,
        gae_lambda=gae_lambda,
        gamma=gamma,
        n_envs=n_envs,
        hierarchical_config=hierarchical_config
    )


# 导出主要类和函数
__all__ = [
    'HierarchicalRolloutBufferSamples',
    'HierarchicalRolloutBuffer', 
    'create_hierarchical_rollout_buffer'
]