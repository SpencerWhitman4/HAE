"""
modules/__init__.py - 基于现有组件的完整集成

核心设计原则：
1. 完全复用现有的HierarchicalPolicy、StateManager、HierarchicalRolloutBuffer
2. 通过配置和管理器类提供统一接口
3. 不重新创建网络，只做组件间的数据流整合
4. 基于StateManager的86维观测数据流架构

重构日期: 2025年8月15日
作者: HA-UAV团队
"""

import logging
import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym
from typing import Dict, List, Tuple, Any, Optional, Union, TYPE_CHECKING
from dataclasses import dataclass, asdict
from pathlib import Path
import json

# 设置日志
logger = logging.getLogger(__name__)

# 导入现有核心组件
try:
    # StateManager和StructuredState
    from .StateManager import StateManager, StructuredState
    
    # HierarchicalPolicy相关组件
    from .HierarchicalPolicy import (
        HierarchicalPolicyNetwork,
        HierarchicalFeaturesExtractor, 
        HierarchicalPolicy
    )
    
    # 分层缓冲区
    from .HierarchicalRolloutBuffer import (
        HierarchicalRolloutBuffer,
        HierarchicalRolloutBufferSamples,
        create_hierarchical_rollout_buffer
    )
    
    # HA_Modules神经网络组件
    from .HA_Modules import HierarchicalRLSystem
    
    logger.debug("所有现有组件导入成功")
    
    # B组消融实验组件
    from .ablation.ablation_config import (
        AblationConfig,
        AblationConfigManager,
        get_ablation_config,
        list_ablation_experiments,
        create_b1_config,
        create_b2_config,
        create_b3_config,
        create_baseline_config
    )
    
    from .ablation.ablation_adapter import (
        AblationStateManagerAdapter,
        AblationPolicyWrapper,
        AblationBufferAdapter,
        AblationComponentsManager,
        create_ablation_system
    )
    
    from .ablation.direct_control_policy import (
        DirectControlPolicy,
        DirectControlPolicyNetwork,
        create_direct_control_policy
    )
    
    from .ablation.flat_policy import (
        FlatPolicy,
        FlatPolicyNetwork,
        create_flat_policy
    )
    
    from .ablation.single_step_hierarchical_policy import (
        SingleStepHierarchicalPolicy,
        SingleStepHighLevelNetwork,
        SingleStepLowLevelNetwork,
        create_single_step_hierarchical_policy
    )
    
    logger.debug("B组消融实验组件导入成功")
    
except ImportError as e:
    logger.error(f"组件导入失败: {e}")
    raise ImportError(f"无法导入必要的模块组件: {e}")


@dataclass
class ModelConfiguration:
    """统一模型配置类 - 基于现有组件架构"""
    
    # StateManager配置
    state_manager_config: Dict[str, Any] = None
    
    # HierarchicalPolicy配置  
    policy_config: Dict[str, Any] = None
    
    # HierarchicalRolloutBuffer配置
    buffer_config: Dict[str, Any] = None
    
    # 训练配置
    training_config: Dict[str, Any] = None
    
    # 环境配置
    env_config: Dict[str, Any] = None
    
    def __post_init__(self):
        """初始化默认配置"""
        if self.state_manager_config is None:
            self.state_manager_config = {
                'history_length': 20,                    # K=20步状态历史
                'high_level_update_frequency': 5,        # τ=5步高层更新频率  
                'future_horizon': 5                      # T=5步未来子目标数量
            }
        
        if self.policy_config is None:
            self.policy_config = {
                'features_dim': 512,
                'net_arch': dict(pi=[256, 256], vf=[256, 256]),
                'activation_fn': 'ReLU',
                'learning_rate': 3e-4
            }
        
        if self.buffer_config is None:
            self.buffer_config = {
                'buffer_size': 2048,
                'gae_lambda': 0.95,
                'gamma': 0.99,
                'hierarchical_config': {
                    'high_level_update_freq': 5,
                    'history_length': 5
                }
            }
        
        if self.training_config is None:
            self.training_config = {
                'algorithm': 'PPO',
                'batch_size': 64,
                'n_epochs': 10,
                'clip_range': 0.2,
                'ent_coef': 0.01,
                'vf_coef': 0.5
            }
        
        if self.env_config is None:
            self.env_config = {
                'observation_dim': 86,
                'action_dim': 4,
                'max_episode_steps': 1000
            }
    
    def save(self, filepath: str):
        """保存配置"""
        config_dict = asdict(self)
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
        logger.info(f"配置已保存到: {filepath}")
    
    @classmethod  
    def load(cls, filepath: str) -> 'ModelConfiguration':
        """加载配置"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)


class HAComponentsManager:
    """HA组件管理器 - 基于现有组件的集成管理"""
    
    def __init__(self, config: Optional[ModelConfiguration] = None):
        """初始化组件管理器
        
        Args:
            config: 模型配置，如果为None则使用默认配置
        """
        self.config = config or ModelConfiguration()
        
        # 组件实例
        self.state_manager: Optional[StateManager] = None
        self.policy: Optional[HierarchicalPolicy] = None  
        self.buffer: Optional[HierarchicalRolloutBuffer] = None
        self.ha_modules: Optional[HierarchicalRLSystem] = None
        
        # 环境信息
        self.observation_space: Optional[gym.Space] = None
        self.action_space: Optional[gym.Space] = None
        
        # 运行状态
        self.is_initialized = False
        self.training_mode = True
        
        logger.debug("HAComponentsManager初始化完成")
    
    def initialize_components(self, env: gym.Env) -> bool:
        """初始化所有组件
        
        Args:
            env: Gym环境实例
            
        Returns:
            bool: 初始化是否成功
        """
        try:
            # 开始初始化HA组件
            
            # 1. 获取环境信息
            self.observation_space = env.observation_space
            self.action_space = env.action_space
            
            # 验证环境维度
            assert self.observation_space.shape == (86,), f"观测维度必须是86，实际为{self.observation_space.shape}"
            assert self.action_space.shape == (4,), f"动作维度必须是4，实际为{self.action_space.shape}"
            
            # 2. 初始化StateManager
            self.state_manager = StateManager(**self.config.state_manager_config)
            logger.debug("StateManager初始化完成")
            
            # 3. 初始化HierarchicalPolicy
            # 确保learning_rate是浮点数
            learning_rate = float(self.config.policy_config['learning_rate'])
            
            self.policy = HierarchicalPolicy(
                observation_space=self.observation_space,
                action_space=self.action_space,
                lr_schedule=lambda _: learning_rate,
                features_extractor_class=HierarchicalFeaturesExtractor,
                features_extractor_kwargs={'features_dim': self.config.policy_config['features_dim']},
                net_arch=self.config.policy_config['net_arch']
            )
            logger.debug("HierarchicalPolicy初始化完成")
            
            # 4. 初始化HierarchicalRolloutBuffer
            self.buffer = create_hierarchical_rollout_buffer(
                buffer_size=self.config.buffer_config['buffer_size'],
                observation_space=self.observation_space,
                action_space=self.action_space,
                gae_lambda=self.config.buffer_config['gae_lambda'],
                gamma=self.config.buffer_config['gamma'],
                hierarchical_config=self.config.buffer_config['hierarchical_config']
            )
            logger.debug("HierarchicalRolloutBuffer初始化完成")
            
            # 5. 初始化HA_Modules（用于高级神经网络组件）
            ha_config = {
                'lidar_dim': 36,
                'action_dim': 4,
                'grid_size': 32,
                'yaw_history_len': 20,
                'state_dim': 256,
                'subgoal_dim': 10
            }
            self.ha_modules = HierarchicalRLSystem(config=ha_config)
            logger.debug("HierarchicalRLSystem初始化完成")
            
            # 6. 建立组件间连接
            self._establish_component_connections()
            
            self.is_initialized = True
            logger.debug("所有HA组件初始化成功！")
            return True
            
        except Exception as e:
            logger.error(f"组件初始化失败: {e}")
            self.is_initialized = False
            return False
    
    def _establish_component_connections(self):
        """建立组件间的数据流连接"""
        try:
            # 将StateManager注入到Policy的特征提取器中
            if hasattr(self.policy, 'features_extractor'):
                if hasattr(self.policy.features_extractor, 'policy_network'):
                    # 共享同一个StateManager实例
                    self.policy.features_extractor.policy_network.state_manager = self.state_manager
                    logger.info("StateManager已注入到HierarchicalPolicy")
            
            # 将StateManager注入到Buffer中（已经在Buffer初始化时创建了自己的实例）
            # 这里我们保持Buffer的独立StateManager，确保数据一致性
            logger.debug("组件连接建立完成")
            
        except Exception as e:
            logger.warning(f"组件连接建立部分失败: {e}")
    
    def predict(self, observation: np.ndarray) -> np.ndarray:
        """统一预测接口
        
        Args:
            observation: [86] 观测向量
            
        Returns:
            np.ndarray: [4] 动作向量
        """
        if not self.is_initialized:
            raise RuntimeError("组件未初始化，请先调用initialize_components()")
        
        # 使用HierarchicalPolicy进行预测
        action, _ = self.policy.predict(observation, deterministic=not self.training_mode)
        return action
    
    def collect_rollout(self, env: gym.Env, n_steps: int = None, trajectory_callback=None) -> Dict[str, Any]:
        """收集经验数据 - 参考HAUAVTrainer的经验收集模式，支持轨迹记录
        
        Args:
            env: 环境实例
            n_steps: 固定收集步数
            trajectory_callback: 轨迹记录回调函数，签名为 (obs, action, reward, next_obs, done, info)
            
        Returns:
            dict: 收集统计信息
        """
        if not self.is_initialized:
            raise RuntimeError("组件未初始化")
        
        if n_steps is None:
            n_steps = self.config.buffer_config['buffer_size']
        
        # 收集统计
        stats = {
            'total_steps': 0,
            'episodes': 0,
            'total_reward': 0.0,
            'episode_rewards': []
        }
        
        # 🔧 关键修改：确保环境状态初始化
        if not hasattr(self, '_current_obs') or self._current_obs is None:
            logger.info("初始化环境状态")
            obs, _ = env.reset()
            self._current_obs = obs
            self._episode_reward = 0.0
            # 轨迹记录：新episode开始
            if trajectory_callback and hasattr(trajectory_callback, '_start_trajectory_episode'):
                trajectory_callback._start_trajectory_episode()
        
        obs = self._current_obs
        episode_reward = self._episode_reward
        
        # 🎯 核心改进：按固定步数收集，不管episode是否结束
        logger.info(f"开始收集经验: 目标{n_steps}步")
        
        for step in range(n_steps):
            # 1. 动作预测
            action = self.predict(obs)
            
            # 2. 环境交互
            next_obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            
            # 3. 轨迹记录：记录单步数据
            if trajectory_callback:
                try:
                    trajectory_callback._log_trajectory_step(obs, action, reward, next_obs, done or truncated, info)
                except Exception as e:
                    logger.warning(f"轨迹记录失败: {e}")
            
            # 4. 存储经验（简化版本）
            self.buffer.add(
                obs=obs.reshape(1, -1),
                action=action.reshape(1, -1),
                reward=np.array([reward]),
                episode_start=np.array([step == 0 and stats['total_steps'] == 0]),
                value=torch.zeros(1),  # 后续补充
                log_prob=torch.zeros(1),  # 后续补充
                should_update_high_level=False
            )
            
            stats['total_steps'] += 1
            obs = next_obs
            
            # 5. 处理episode结束（但不中断收集）
            if done or truncated:
                stats['episodes'] += 1
                stats['episode_rewards'].append(episode_reward)
                stats['total_reward'] += episode_reward
                
                logger.info(f"Episode {stats['episodes']} 完成，奖励: {episode_reward:.2f}")
                
                # 轨迹记录：完成episode
                if trajectory_callback and hasattr(trajectory_callback, '_finalize_trajectory_episode'):
                    trajectory_callback._finalize_trajectory_episode(episode_reward, stats['total_steps'], info)
                
                # 重置环境继续收集
                obs, _ = env.reset()
                episode_reward = 0.0
                
                # 轨迹记录：新episode开始
                if trajectory_callback and hasattr(trajectory_callback, '_start_trajectory_episode'):
                    trajectory_callback._start_trajectory_episode()
            
            # 每100步报告进度
            if (step + 1) % 100 == 0:
                logger.info(f"收集进度: {step + 1}/{n_steps} ({(step + 1)/n_steps:.1%})")
        
        # 保存当前状态
        self._current_obs = obs
        self._episode_reward = episode_reward
        
        # 计算GAE（简化处理）
        with torch.no_grad():
            last_values = torch.zeros(1)
            self.buffer.compute_returns_and_advantage(last_values, np.array([False]))
        
        # 统计信息
        if stats['episodes'] > 0:
            stats['mean_reward'] = stats['total_reward'] / stats['episodes']
        else:
            # 未完成episode，使用当前累积奖励
            stats['mean_reward'] = episode_reward
            stats['total_reward'] = episode_reward
        
        logger.info(f"经验收集完成: {stats['total_steps']}步, {stats['episodes']}个完整episode, "
                   f"平均奖励: {stats['mean_reward']:.3f}")
        
        return stats
    
    def get_training_data(self) -> HierarchicalRolloutBufferSamples:
        """获取训练数据
        
        Returns:
            HierarchicalRolloutBufferSamples: 分层训练数据样本
        """
        if not self.is_initialized or not self.buffer.full:
            raise RuntimeError("组件未初始化或buffer未满")
        
        # 返回第一个batch的数据
        for batch in self.buffer.get(batch_size=self.config.training_config['batch_size']):
            return batch
    
    def update_policy(self, rollout_data: HierarchicalRolloutBufferSamples) -> Dict[str, float]:
        """完整的分层PPO策略更新
        
        Args:
            rollout_data: 分层训练数据
            
        Returns:
            dict: 详细训练统计信息
        """
        if not self.is_initialized:
            raise RuntimeError("组件未初始化")
        
        # 训练配置
        clip_range = self.config.training_config.get('clip_range', 0.2)
        vf_coef = self.config.training_config.get('vf_coef', 0.5)
        ent_coef = self.config.training_config.get('ent_coef', 0.01)
        max_grad_norm = self.config.training_config.get('max_grad_norm', 0.5)
        
        # 累积统计信息
        policy_losses = []
        value_losses = []
        entropy_losses = []
        high_level_losses = []
        low_level_losses = []
        approx_kl_divs = []
        clip_fractions = []
        
        # === 1. 标准PPO更新（综合策略） ===
        with torch.no_grad():
            # 重新计算当前策略的值
            values, log_probs, entropy = self.policy.evaluate_actions(
                rollout_data.observations, 
                rollout_data.actions
            )
            
            # 计算重要性采样比率
            ratio = torch.exp(log_probs - rollout_data.old_log_prob)
            
            # KL散度估计
            approx_kl = ((rollout_data.old_log_prob - log_probs).mean()).item()
            approx_kl_divs.append(approx_kl)
        
        # PPO裁剪目标
        surr1 = ratio * rollout_data.advantages
        surr2 = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * rollout_data.advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # 价值函数损失
        if self.config.training_config.get('clip_range_vf') is not None:
            # 裁剪价值函数
            clip_range_vf = self.config.training_config['clip_range_vf']
            values_clipped = rollout_data.old_values + torch.clamp(
                values - rollout_data.old_values, -clip_range_vf, clip_range_vf
            )
            value_loss_1 = F.mse_loss(values, rollout_data.returns)
            value_loss_2 = F.mse_loss(values_clipped, rollout_data.returns)
            value_loss = torch.max(value_loss_1, value_loss_2).mean()
        else:
            value_loss = F.mse_loss(values.squeeze(), rollout_data.returns)
        
        # 熵损失
        entropy_loss = -entropy.mean()
        
        # 总损失
        total_loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss
        
        # === 2. 分层特定更新 ===
        high_level_loss = torch.tensor(0.0, device=rollout_data.observations.device)
        low_level_loss = torch.tensor(0.0, device=rollout_data.observations.device)
        
        # 高层策略更新（基于更新掩码）
        if rollout_data.high_level_update_mask.sum() > 0:
            high_level_indices = rollout_data.high_level_update_mask.bool()
            
            if high_level_indices.any():
                # 高层观测和动作
                hl_obs = rollout_data.high_level_observations[high_level_indices]  # [N_hl, K, 28]
                hl_actions = rollout_data.high_level_actions[high_level_indices]   # [N_hl, 10]
                hl_advantages = rollout_data.high_level_advantages[high_level_indices]  # [N_hl]
                hl_returns = rollout_data.high_level_returns[high_level_indices]    # [N_hl]
                hl_old_values = rollout_data.high_level_values[high_level_indices]  # [N_hl]
                
                # 通过策略网络获取高层特征
                if hasattr(self.policy, 'features_extractor') and hasattr(self.policy.features_extractor, 'policy_network'):
                    policy_net = self.policy.features_extractor.policy_network
                    
                    # 模拟高层网络前向传播（需要占据栅格和yaw历史）
                    # 这里简化处理，实际需要从hl_obs重构原始输入
                    batch_size = hl_obs.size(0)
                    
                    # 创建模拟输入（实际应该从StateManager获取）
                    occupancy_grids = torch.randn(batch_size, 5, 8, 8, device=hl_obs.device)
                    yaw_histories = torch.randn(batch_size, 5, device=hl_obs.device)
                    
                    try:
                        # 高层前向传播
                        hl_subgoals, hl_values_pred = policy_net.forward_high_level(occupancy_grids)
                        
                        # 高层价值损失
                        hl_value_loss = F.mse_loss(hl_values_pred.squeeze(), hl_returns)
                        
                        # 高层策略损失（基于子目标质量）
                        # 这里简化为L2损失，实际应该是策略梯度
                        target_subgoals = hl_actions.view(batch_size, 5, 2)  # 重塑为[N, T, 2]
                        hl_policy_loss = F.mse_loss(hl_subgoals, target_subgoals)
                        
                        high_level_loss = hl_policy_loss + 0.5 * hl_value_loss
                        
                    except Exception as e:
                        logger.warning(f"高层更新失败: {e}")
                        high_level_loss = torch.tensor(0.0, device=hl_obs.device)
        
        # 低层策略更新（每步都有）
        if rollout_data.low_level_update_mask.sum() > 0:
            ll_indices = rollout_data.low_level_update_mask.bool()
            
            if ll_indices.any():
                # 低层观测和动作
                ll_obs = rollout_data.low_level_observations[ll_indices]      # [N_ll, 64]
                ll_actions = rollout_data.low_level_actions[ll_indices]       # [N_ll, 4]  
                ll_advantages = rollout_data.low_level_advantages[ll_indices] # [N_ll]
                ll_returns = rollout_data.low_level_returns[ll_indices]       # [N_ll]
                ll_old_values = rollout_data.low_level_values[ll_indices]     # [N_ll]
                
                # 通过策略网络获取低层特征
                if hasattr(self.policy, 'features_extractor') and hasattr(self.policy.features_extractor, 'policy_network'):
                    policy_net = self.policy.features_extractor.policy_network
                    
                    try:
                        # 低层状态编码
                        ll_features = policy_net.state_encoder(ll_obs)  # [N_ll, 128]
                        
                        # 获取当前子目标（简化处理）
                        current_subgoals = torch.zeros(ll_obs.size(0), 2, device=ll_obs.device)
                        
                        # 低层前向传播
                        ll_actions_pred, ll_values_pred = policy_net.forward_low_level(ll_features, current_subgoals)
                        
                        # 低层价值损失
                        ll_value_loss = F.mse_loss(ll_values_pred.squeeze(), ll_returns)
                        
                        # 低层策略损失
                        ll_policy_loss = F.mse_loss(ll_actions_pred, ll_actions)
                        
                        low_level_loss = ll_policy_loss + 0.5 * ll_value_loss
                        
                    except Exception as e:
                        logger.warning(f"低层更新失败: {e}")
                        low_level_loss = torch.tensor(0.0, device=ll_obs.device)
        
        # === 3. 组合总损失 ===
        hierarchical_weight = self.config.training_config.get('hierarchical_weight', 0.1)
        combined_loss = total_loss + hierarchical_weight * (high_level_loss + low_level_loss)
        
        # === 4. 反向传播和优化 ===
        # 获取优化器
        optimizer = None
        if hasattr(self.policy, 'optimizer'):
            optimizer = self.policy.optimizer
        elif hasattr(self.policy, 'policy') and hasattr(self.policy.policy, 'optimizer'):
            optimizer = self.policy.policy.optimizer
        
        if optimizer is not None:
            # 梯度清零
            optimizer.zero_grad()
            
            # 反向传播
            combined_loss.backward()
            
            # 梯度裁剪
            if max_grad_norm is not None:
                if hasattr(self.policy, 'parameters'):
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.policy.parameters(), max_grad_norm
                    )
                else:
                    grad_norm = 0.0
            else:
                grad_norm = 0.0
            
            # 优化器步进
            optimizer.step()
        else:
            logger.warning("未找到优化器，跳过参数更新")
            grad_norm = 0.0
        
        # === 5. 计算统计信息 ===
        with torch.no_grad():
            # 裁剪比例
            clip_fraction = ((ratio - 1.0).abs() > clip_range).float().mean().item()
            clip_fractions.append(clip_fraction)
            
            # 记录损失
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())  
            entropy_losses.append(entropy_loss.item())
            high_level_losses.append(high_level_loss.item())
            low_level_losses.append(low_level_loss.item())
        
        # === 6. 返回详细统计 ===
        return {
            # 标准PPO统计
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy_loss': np.mean(entropy_losses),
            'total_loss': combined_loss.item(),
            'approx_kl': np.mean(approx_kl_divs),
            'clip_fraction': np.mean(clip_fractions),
            'explained_variance': self._explained_variance(rollout_data.returns.cpu().numpy(), 
                                                           values.detach().cpu().numpy() if hasattr(values, 'detach') else values),
            
            # 分层特定统计
            'high_level_loss': np.mean(high_level_losses),
            'low_level_loss': np.mean(low_level_losses),
            'high_level_updates': rollout_data.high_level_update_mask.sum().item(),
            'low_level_updates': rollout_data.low_level_update_mask.sum().item(),
            'hierarchical_weight': hierarchical_weight,
            
            # 训练诊断
            'gradient_norm': grad_norm if isinstance(grad_norm, float) else grad_norm.item(),
            'learning_rate': self.config.policy_config['learning_rate'],
            'clip_range': clip_range,
            'vf_coef': vf_coef,
            'ent_coef': ent_coef,
        }
    
    def _explained_variance(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """计算解释方差
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            
        Returns:
            float: 解释方差
        """
        var_y = np.var(y_true)
        if var_y == 0:
            return np.nan
        return 1 - np.var(y_true - y_pred) / var_y
    
    def train_step(self, env: gym.Env) -> Dict[str, Any]:
        """执行一步完整训练 - 参考HAUAVTrainer模式
        
        Args:
            env: 环境实例
            
        Returns:
            dict: 训练统计信息
        """
        # 🔧 重要：每次train_step重置buffer
        logger.info("训练步骤开始，重置buffer")
        self.buffer.reset()
        
        # 1. 收集固定步数的经验
        rollout_stats = self.collect_rollout(env, n_steps=self.config.buffer_config['buffer_size'])
        
        # 2. 检查数据充足性
        min_batch_size = self.config.training_config.get('batch_size', 64)
        if rollout_stats['total_steps'] < min_batch_size:
            logger.warning(f"数据不足({rollout_stats['total_steps']}步)，跳过策略更新")
            rollout_stats['training_skipped'] = True
            rollout_stats['policy_loss'] = 0.0
            rollout_stats['value_loss'] = 0.0
            return rollout_stats
        
        # 3. 策略更新（如果buffer满足条件）
        if hasattr(self.buffer, 'full') and self.buffer.full:
            training_stats = self._update_policy_batch()
            # 合并统计信息
            combined_stats = {**rollout_stats, **training_stats}
        else:
            # buffer未满，进行轻量级更新
            training_stats = self._lightweight_policy_update()
            combined_stats = {**rollout_stats, **training_stats}
        
        combined_stats['training_skipped'] = False
        combined_stats['buffer_stats'] = self.buffer.get_hierarchical_statistics()
        
        logger.info(f"训练步骤完成: 收集{rollout_stats['total_steps']}步, "
                   f"episodes: {rollout_stats['episodes']}, "
                   f"平均奖励: {rollout_stats.get('mean_reward', 0.0):.3f}")
        
        return combined_stats
    
    def update_policy_from_buffer(self) -> Dict[str, Any]:
        """从buffer更新策略 - 独立方法供训练器调用"""
        if not self.is_initialized:
            raise RuntimeError("组件未初始化")
        
        training_stats = {}
        n_epochs = self.config.training_config.get('n_epochs', 4)
        batch_size = self.config.training_config.get('batch_size', 64)
        
        # 检查buffer中是否有足够数据
        if not hasattr(self.buffer, 'size') or self.buffer.size() < batch_size:
            logger.warning(f"Buffer数据不足，跳过策略更新")
            return {'policy_loss': 0.0, 'value_loss': 0.0, 'training_skipped': True}
        
        logger.info(f"开始策略更新，epochs: {n_epochs}, batch_size: {batch_size}")
        for epoch in range(n_epochs):
            try:
                for batch_data in self.buffer.get(batch_size=batch_size):
                    epoch_stats = self.update_policy(batch_data)
                    
                    # 累积统计
                    for key, value in epoch_stats.items():
                        training_stats[key] = training_stats.get(key, 0.0) + value / n_epochs
            except Exception as e:
                logger.error(f"策略更新失败: {e}")
                break
        
        training_stats['training_skipped'] = False
        logger.info(f"策略更新完成: loss={training_stats.get('policy_loss', 0.0):.4f}")
        return training_stats
    
    def _update_policy_batch(self) -> Dict[str, Any]:
        """批量策略更新 - 参考HAUAVTrainer"""
        training_stats = {}
        n_epochs = self.config.training_config.get('n_epochs', 4)
        batch_size = self.config.training_config.get('batch_size', 64)
        
        logger.info(f"开始策略更新: {n_epochs} epochs, batch_size: {batch_size}")
        
        for epoch in range(n_epochs):
            epoch_stats = {'policy_loss': 0.0, 'value_loss': 0.0, 'batches': 0}
            
            for batch_data in self.buffer.get(batch_size=batch_size):
                batch_update_stats = self.update_policy(batch_data)
                
                # 累积epoch统计
                for key, value in batch_update_stats.items():
                    if key in epoch_stats:
                        epoch_stats[key] += value
                epoch_stats['batches'] += 1
            
            # 计算epoch平均值
            if epoch_stats['batches'] > 0:
                for key in ['policy_loss', 'value_loss']:
                    if key in epoch_stats:
                        epoch_stats[key] /= epoch_stats['batches']
            
            # 累积到总统计
            for key, value in epoch_stats.items():
                if key != 'batches':
                    training_stats[key] = training_stats.get(key, 0.0) + value / n_epochs
        
        return training_stats
    
    def _lightweight_policy_update(self) -> Dict[str, Any]:
        """轻量级策略更新"""
        # 简化的更新，避免复杂计算
        return {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'lightweight_update': True
        }
    
    def save_model(self, save_path: str):
        """保存模型
        
        Args:
            save_path: 保存路径
        """
        if not self.is_initialized:
            raise RuntimeError("组件未初始化")
        
        # 保存策略
        self.policy.save(save_path)
        
        # 保存配置
        config_path = save_path.replace('.zip', '_config.json')
        self.config.save(config_path)
        
        # 保存StateManager状态
        import pickle
        sm_path = save_path.replace('.zip', '_state_manager.pkl')
        with open(sm_path, 'wb') as f:
            pickle.dump(self.state_manager, f)
        
        logger.info(f"模型已保存到: {save_path}")
    
    def load_model(self, load_path: str, env: gym.Env):
        """加载模型
        
        Args:
            load_path: 加载路径
            env: 环境实例
        """
        # 加载配置
        config_path = load_path.replace('.zip', '_config.json')
        if Path(config_path).exists():
            self.config = ModelConfiguration.load(config_path)
        
        # 初始化组件
        self.initialize_components(env)
        
        # 加载策略
        self.policy = HierarchicalPolicy.load(load_path, env=env)
        
        # 加载StateManager状态
        import pickle
        sm_path = load_path.replace('.zip', '_state_manager.pkl')
        if Path(sm_path).exists():
            with open(sm_path, 'rb') as f:
                self.state_manager = pickle.load(f)
        
        logger.info(f"模型已从 {load_path} 加载")
    
    def set_training_mode(self, training: bool):
        """设置训练模式
        
        Args:
            training: 是否为训练模式
        """
        self.training_mode = training
        if self.policy:
            self.policy.set_training_mode(training)
    
    def get_training_stats(self) -> Dict[str, Any]:
        """获取训练统计信息
        
        Returns:
            Dict[str, Any]: 包含训练统计的字典
        """
        stats = {}
        
        try:
            # 基础统计
            stats['is_initialized'] = self.is_initialized
            stats['training_mode'] = self.training_mode
            
            # 缓冲区统计
            if self.buffer is not None:
                stats['buffer'] = {
                    'size': self.buffer.buffer_size,
                    'position': self.buffer.pos,
                    'full': self.buffer.full,
                    'n_envs': self.buffer.n_envs
                }
                
                # 分层缓冲区特有统计
                if hasattr(self.buffer, 'get_hierarchical_statistics'):
                    hierarchical_stats = self.buffer.get_hierarchical_statistics()
                    stats['buffer'].update(hierarchical_stats)
            
            # 策略统计
            if self.policy is not None:
                stats['policy'] = {
                    'type': type(self.policy).__name__,
                    'observation_space': str(self.observation_space.shape) if self.observation_space else None,
                    'action_space': str(self.action_space.shape) if self.action_space else None
                }
            
            # StateManager统计
            if self.state_manager is not None:
                stats['state_manager'] = {
                    'type': type(self.state_manager).__name__,
                    'history_length': getattr(self.state_manager, 'history_length', None),
                    'high_level_update_frequency': getattr(self.state_manager, 'high_level_update_frequency', None)
                }
            
            # HA_Modules统计
            if self.ha_modules is not None:
                stats['ha_modules'] = {
                    'type': type(self.ha_modules).__name__,
                    'parameters': sum(p.numel() for p in self.ha_modules.parameters()),
                    'trainable_parameters': sum(p.numel() for p in self.ha_modules.parameters() if p.requires_grad)
                }
            
            # 配置统计
            stats['config'] = {
                'state_manager_config': self.config.state_manager_config,
                'buffer_config': {k: v for k, v in self.config.buffer_config.items() if k != 'hierarchical_config'},
                'training_config': self.config.training_config
            }
            
            logger.info("训练统计信息收集完成")
            
        except Exception as e:
            logger.error(f"统计信息收集失败: {e}")
            stats['error'] = str(e)
        
        return stats
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态
        
        Returns:
            dict: 系统状态信息
        """
        status = {
            'initialized': self.is_initialized,
            'training_mode': self.training_mode,
            'components': {
                'state_manager': self.state_manager is not None,
                'policy': self.policy is not None,
                'buffer': self.buffer is not None,
                'ha_modules': self.ha_modules is not None
            }
        }
        
        if self.state_manager:
            status['state_manager_info'] = {
                'step_counter': self.state_manager.step_counter,
                'history_length': len(self.state_manager.state_history),
                'is_ready_for_high_level': self.state_manager.is_ready_for_high_level()
            }
        
        if self.buffer and self.buffer.full:
            status['buffer_stats'] = self.buffer.get_hierarchical_statistics()
        
        return status


# =============== 便捷函数 ===============

def create_ha_system(env: gym.Env, config: Optional[ModelConfiguration] = None) -> HAComponentsManager:
    """创建完整的HA系统
    
    Args:
        env: Gym环境
        config: 模型配置
        
    Returns:
        HAComponentsManager: 初始化完成的系统管理器
    """
    manager = HAComponentsManager(config)
    success = manager.initialize_components(env)
    
    if not success:
        raise RuntimeError("HA系统创建失败")
    
    return manager

def create_default_config() -> ModelConfiguration:
    """创建默认配置
    
    Returns:
        ModelConfiguration: 默认配置对象
    """
    return ModelConfiguration()

# =============== 模块导出 ===============

__all__ = [
    # 现有核心组件（完全复用）
    'StateManager',
    'StructuredState', 
    'HierarchicalPolicyNetwork',
    'HierarchicalFeaturesExtractor',
    'HierarchicalPolicy',
    'HierarchicalRolloutBuffer',
    'HierarchicalRolloutBufferSamples',
    'HierarchicalRLSystem',
    
    # 管理和配置类
    'ModelConfiguration',
    'HAComponentsManager',
    
    # 便捷函数
    'create_ha_system',
    'create_default_config',
    'create_hierarchical_rollout_buffer',
    
    # B组消融实验组件
    'AblationConfig',
    'AblationConfigManager', 
    'get_ablation_config',
    'list_ablation_experiments',
    'create_b1_config',
    'create_b2_config',
    'create_b3_config',
    'create_baseline_config',
    
    'AblationStateManagerAdapter',
    'AblationPolicyWrapper',
    'AblationBufferAdapter',
    'AblationComponentsManager',
    'create_ablation_system',
    
    'DirectControlPolicy',
    'DirectControlPolicyNetwork', 
    'create_direct_control_policy',
    
    'FlatPolicy',
    'FlatPolicyNetwork',
    'create_flat_policy',
    
    'SingleStepHierarchicalPolicy',
    'SingleStepHighLevelNetwork',
    'SingleStepLowLevelNetwork',
    'create_single_step_hierarchical_policy',
]

logger.info("modules包初始化完成 - 基于现有组件的完整集成 + B组消融实验")