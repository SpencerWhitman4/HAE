#!/usr/bin/env python3

"""
基座模型训练器 - 基于BaseFlightAviary训练悬停+飞行基础模型
"""

import logging
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import sys

# 添加项目路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入核心组件
from ..core.base_trainer import BaseTrainer, TrainingStage, TrainingResult
from ..core.environment_factory import EnvironmentFactory

# 复用现有组件
from src.envs.BaseFlightAviary import BaseFlightAviary, BaseFlightConfig
from src.utils.Logger import Logger
from src.utils.utils import sync

logger = logging.getLogger(__name__)


class BaseFlightModel(nn.Module):
    """
    基座飞行模型 - 学习悬停和基础飞行控制
    
    输入: 86维激光雷达观测
    输出: 4维控制指令 [thrust, roll, pitch, yaw_rate]
    """
    
    def __init__(self, 
                 obs_dim: int = 86,
                 action_dim: int = 4,
                 hidden_dims: List[int] = [256, 256, 128],
                 activation: str = "relu"):
        
        super(BaseFlightModel, self).__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # 特征提取器
        layers = []
        in_dim = obs_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU() if activation == "relu" else nn.Tanh(),
                nn.Dropout(0.1)
            ])
            in_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Actor网络 - 控制策略
        self.actor = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()  # 输出范围[-1, 1]
        )
        
        # Critic网络 - 价值估计  
        self.critic = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # 参数初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """参数初始化"""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=0.5)
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            obs: 观测张量 [batch_size, obs_dim]
            
        Returns:
            (action, value): 动作和价值估计
        """
        features = self.encoder(obs)
        action = self.actor(features)
        value = self.critic(features)
        
        return action, value
    
    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """获取动作"""
        with torch.no_grad():
            action, _ = self.forward(obs)
            
            if not deterministic:
                # 添加探索噪声
                noise = torch.normal(0, 0.1, size=action.shape)
                action = action + noise
                action = torch.clamp(action, -1.0, 1.0)
            
            return action
    
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """获取价值估计"""
        with torch.no_grad():
            features = self.encoder(obs)
            value = self.critic(features)
            return value


class BaseFlightTrainer(BaseTrainer):
    """
    基座模型训练器
    
    训练BaseFlightAviary环境中的悬停+基础飞行控制策略
    作为后续分层训练的统一基座
    """
    
    def __init__(self, 
                 config: Dict[str, Any],
                 foundation_model_path: Optional[Path] = None,
                 output_dir: str = "./models"):
        
        super().__init__(
            stage=TrainingStage.FOUNDATION,
            config=config,
            experiment_name="BaseFlightModel",
            stage_variant=None
        )
        
        self.foundation_model_path = foundation_model_path
        self.output_dir = output_dir
        self.env_factory = EnvironmentFactory()
        
        # 训练配置
        self.total_timesteps = config.get('total_timesteps', 100000)
        self.batch_size = config.get('batch_size', 256)
        self.learning_rate = config.get('learning_rate', 3e-4)
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.clip_range = config.get('clip_range', 0.2)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        
        # 课程学习配置
        self.hover_training_steps = config.get('hover_training_steps', 25000)
        self.flight_training_steps = config.get('flight_training_steps', 75000)
        self.enable_curriculum = config.get('enable_curriculum', True)
        
        # 模型和训练组件
        self.model = None
        self.optimizer = None
        self.env = None
        
        # 训练统计
        self.training_stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'hover_success_rate': [],
            'flight_success_rate': [],
            'value_losses': [],
            'policy_losses': []
        }
        
        # Episode计数和轨迹记录
        self.episode_count = 0
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
        self.trajectory_episode_started = False
        
        # 当前训练阶段
        self.current_phase = "hover"  # "hover" or "flight"
        self.phase_start_step = 0
        
        # 初始化progress_callbacks
        self.progress_callbacks = []
    
    def setup(self) -> bool:
        """设置训练器"""
        try:
            # 初始化会话管理（包含可视化）
            session_info = self.initialize_session(
                enable_trajectory=self.config.get('enable_trajectory', True),
                enable_tensorboard=self.config.get('enable_tensorboard', True),
                enable_visualization=self.config.get('enable_visualization', True),
                enable_rich_display=self.config.get('enable_rich_display', True)
            )
            
            self.logger.info(f"会话初始化完成: {session_info['session_dir']}")
            
            # 初始化TensorBoard Writer
            self.tensorboard_writer = None
            if self.session_manager and self.session_manager.feature_flags.get('tensorboard', False):
                try:
                    from torch.utils.tensorboard import SummaryWriter
                    tensorboard_path = self.session_manager.data_managers['tensorboard']['train']
                    self.tensorboard_writer = SummaryWriter(log_dir=str(tensorboard_path))
                    self.logger.info(f"✅ TensorBoard Writer 初始化完成: {tensorboard_path}")
                except ImportError:
                    self.logger.warning("TensorBoard不可用，跳过TensorBoard记录")
                except Exception as e:
                    self.logger.warning(f"TensorBoard初始化失败: {e}")
            
            # 初始化轨迹管理器
            self.trajectory_manager = None
            if self.session_manager and self.session_manager.feature_flags.get('trajectory', False):
                try:
                    trajectory_managers = self.session_manager.data_managers.get('trajectory', {})
                    self.trajectory_manager = trajectory_managers.get('train')
                    if self.trajectory_manager:
                        self.logger.info("✅ 轨迹管理器 初始化完成")
                except Exception as e:
                    self.logger.warning(f"轨迹管理器初始化失败: {e}")
            
            # 创建环境
            env_config = {
                'drone_model': self.config.get('drone_model', 'CF2X'),
                'physics': self.config.get('physics', 'PYB'),
                'gui_training': self.config.get('gui_training', False),
                'max_episode_steps': self.config.get('max_episode_steps', 1000),
                'hover_training_steps': self.hover_training_steps,
                'flight_training_steps': self.flight_training_steps,
                'enable_curriculum': self.enable_curriculum
            }
            
            self.env = self.env_factory.create_environment(
                stage=self.stage,
                config=env_config,
                mode="train"
            )
            
            # 创建模型
            self.model = BaseFlightModel(
                obs_dim=86,
                action_dim=4,
                hidden_dims=self.config.get('hidden_dims', [256, 256, 128])
            )
            
            # 创建优化器
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                eps=1e-5
            )
            
            # 🔄 尝试加载之前的模型进行续训
            self._try_resume_training()
            
            self.write_stage_message("基座模型训练器设置完成")
            return True
            
        except Exception as e:
            self.logger.error(f"训练器设置失败: {e}")
            if self.visualization_manager:
                self.visualization_manager.write_message(f"设置失败: {e}", "ERROR")
            return False
    
    def _execute_training(self) -> Dict[str, Any]:
        """执行基座模型训练逻辑 - 修复版本，确保回调正常工作"""
        evaluation_frequency = self.config.get('evaluation_frequency', 10000)
        checkpoint_frequency = self.config.get('checkpoint_frequency', 50000)
        save_frequency = self.config.get('save_frequency', 10000)  # 添加定期保存频率
        
        # 调试模式下更频繁的保存和评估
        if self.config.get('debug', False):
            checkpoint_frequency = min(checkpoint_frequency, 500)  # 调试模式下每500步保存检查点
            evaluation_frequency = min(evaluation_frequency, 500)  # 调试模式下每500步评估
            save_frequency = min(save_frequency, 200)  # 调试模式下每200步保存模型
        
        # 🎯 优化学习参数：使用配置中的buffer_size
        # 从配置中获取buffer_size，如果没有则使用n_steps或默认值
        buffer_size = self.config.get('buffer_size', self.config.get('n_steps', 2048))
        effective_batch_size = min(self.batch_size, 64)  # 限制批次大小为64，提高训练频率
        learn_interval = min(buffer_size, max(effective_batch_size, buffer_size))  # 使用配置的buffer_size作为学习间隔
        
        training_stats = []
        start_time = time.time()
        self.logger.info(f"开始基座模型训练: {self.total_timesteps:,} 步 (学习间隔: {learn_interval} 步)")
        self.logger.info(f"📊 检查点频率: {checkpoint_frequency}, 评估频率: {evaluation_frequency}, 保存频率: {save_frequency}")
        
        # 初始化训练循环
        obs, _ = self.env.reset()
        episode_reward = 0
        episode_length = 0
        episode_count = 0
        rollout_buffer = []
        
        # 🎬 启动第一个episode的轨迹记录
        self._start_trajectory_episode(episode_count)
        
        # 🔧 检查可视化管理器状态
        if self.visualization_manager:
            self.logger.info(f"✅ 可视化管理器已启用: {type(self.visualization_manager).__name__}")
        else:
            self.logger.warning("❌ 可视化管理器未启用，进度条将不显示")
        
        # 🔧 立即调用第一次回调，确保进度条启动
        if self.visualization_manager:
            initial_metrics = {
                'episode': 0,
                'total_reward': 0.0,
                'exploration_rate': 1.0,
                'policy_loss': 0.0,
                'value_loss': 0.0,
                'phase': self.current_phase
            }
            self.visualization_manager.on_step(0, initial_metrics)
            self.logger.info("✅ 基座模型初始回调已触发，进度条应已启动")
        
        while self.current_step < self.total_timesteps:
            try:
                # 🔧 合理的批次收集：确保有足够经验进行训练
                steps_to_collect = min(learn_interval, self.total_timesteps - self.current_step)
                
                self.logger.info(f"开始收集经验: {steps_to_collect} 步 (当前步数: {self.current_step})")
                
                # 限时收集经验，避免无限卡住
                collect_start = time.time()
                steps_collected = 0
                max_collect_time = 60.0  # 最多60秒收集时间
                
                try:
                    # 课程学习阶段切换
                    if self.enable_curriculum:
                        self._update_curriculum_phase(self.current_step)
                    
                    # 收集经验 - 添加超时保护
                    for step_idx in range(steps_to_collect):
                        # 超时检查
                        if time.time() - collect_start > max_collect_time:
                            self.logger.warning(f"经验收集超时 ({max_collect_time}s)，已收集 {steps_collected}/{steps_to_collect} 步")
                            break
                        # 收集经验
                        with torch.no_grad():
                            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                            action, value = self.model(obs_tensor)
                            action_np = action.squeeze(0).cpu().numpy()
                        
                        # 环境交互
                        next_obs, reward, terminated, truncated, info = self.env.step(action_np)
                        done = terminated or truncated
                        
                        # 🎬 记录轨迹数据
                        self._log_trajectory_step(obs, action_np, reward, next_obs, done, info)
                        
                        # 存储经验
                        rollout_buffer.append({
                            'obs': obs.copy(),
                            'action': action_np.copy(),
                            'reward': reward,
                            'value': value.item(),
                            'done': done
                        })
                        
                        # 更新状态
                        obs = next_obs
                        episode_reward += reward
                        episode_length += 1
                        steps_collected += 1
                        
                        # 每隔一定步数更新一次进度，避免频繁更新
                        if step_idx % 100 == 0:
                            temp_metrics = {
                                'episode': episode_count,
                                'total_reward': episode_reward,
                                'exploration_rate': max(0.0, 1.0 - (self.current_step + steps_collected) / self.total_timesteps),
                                'phase': self.current_phase
                            }
                            if self.visualization_manager:
                                self.visualization_manager.on_step(self.current_step + steps_collected, temp_metrics)
                        
                        # 处理回合结束
                        if done:
                            # 🎬 结束轨迹记录
                            self._finalize_trajectory_episode(episode_reward, episode_length, info)
                            self._handle_episode_end(episode_reward, episode_length, info)
                            
                            # 重置环境并开始新episode
                            obs, _ = self.env.reset()
                            episode_count += 1
                            episode_reward = 0
                            episode_length = 0
                            
                            # 🎬 启动新episode的轨迹记录
                            self._start_trajectory_episode(episode_count)
                        
                        # 超时保护 - 如果单步时间过长，记录并继续
                        if time.time() - collect_start > 10.0 and step_idx < 10:
                            self.logger.warning(f"经验收集步骤 {step_idx} 耗时过长，可能存在环境问题")
                    
                    collect_time = time.time() - collect_start
                    
                    if collect_time > 3.0:  # 降低警告阈值
                        self.logger.warning(f"经验收集耗时过长: {collect_time:.1f}s")
                    
                    # 在收集完经验后进行训练
                    if len(rollout_buffer) >= effective_batch_size:
                        self._train_step(rollout_buffer)
                        rollout_buffer = []
                    
                except Exception as e:
                    self.logger.error(f"经验收集失败: {e}")
                    # 即使失败也要更新进度，确保训练继续
                    if steps_collected == 0:
                        steps_collected = steps_to_collect  # 如果没有收集到步数，使用预期值
                
                # 🎯 立即更新步数和触发回调
                self.current_step += steps_collected
                
                # 🔧 核心：立即触发回调，确保进度条更新
                step_metrics = {
                    'episode': episode_count,
                    'total_reward': episode_reward,
                    'exploration_rate': max(0.0, 1.0 - self.current_step / self.total_timesteps),
                    'policy_loss': np.mean(self.training_stats['policy_losses'][-10:]) if self.training_stats['policy_losses'] else 0.0,
                    'value_loss': np.mean(self.training_stats['value_losses'][-10:]) if self.training_stats['value_losses'] else 0.0,
                    'phase': self.current_phase
                }
                
                # 📊 TensorBoard记录
                self._log_to_tensorboard(step_metrics)
                
                # 调用可视化管理器回调
                if self.visualization_manager:
                    self.visualization_manager.on_step(self.current_step, step_metrics)
                    self.logger.info(f"✅ 回调已触发: {self.current_step}/{self.total_timesteps} ({self.current_step/self.total_timesteps:.1%})")
                
                # 调用训练器回调
                self.on_step_callback(self.current_step, step_metrics)
                
                # Episode回调
                if episode_count > 0 and episode_count != getattr(self, '_last_episode_count', 0):
                    self.on_episode_callback(episode_count, step_metrics)
                    self._last_episode_count = episode_count
                
                # 定期评估（改为范围触发，更可靠）
                steps_since_last_eval = self.current_step - getattr(self, '_last_eval_step', 0)
                if steps_since_last_eval >= evaluation_frequency and self.current_step > 0:
                    self.logger.info(f"🎯 开始定期评估 (步数: {self.current_step}, 距上次评估: {steps_since_last_eval} 步)")
                    try:
                        eval_results = self._perform_evaluation()
                        self.on_evaluation_callback(eval_results)
                        self._last_eval_step = self.current_step  # 记录最后评估步数
                        self.logger.info(f"✅ 评估完成 - 平均奖励: {eval_results.get('mean_reward', 0):.2f}")
                    except Exception as e:
                        self.logger.warning(f"❌ 评估失败: {e}")
                elif self.current_step > 0 and steps_since_last_eval >= evaluation_frequency * 0.8:
                    # 调试信息：接近评估点时提示
                    self.logger.debug(f"🔔 接近评估点: 还需 {evaluation_frequency - steps_since_last_eval} 步触发评估")
                
                # 定期保存检查点（也使用范围触发）
                steps_since_last_checkpoint = self.current_step - getattr(self, '_last_checkpoint_step', 0)
                if steps_since_last_checkpoint >= checkpoint_frequency and self.current_step > 0:
                    self.logger.info(f"💾 保存检查点 (步数: {self.current_step}, 距上次保存: {steps_since_last_checkpoint} 步)")
                    try:
                        self.on_checkpoint_callback(self.current_step)
                        self._last_checkpoint_step = self.current_step  # 记录最后检查点步数
                        self.logger.info(f"✅ 检查点已保存")
                    except Exception as e:
                        self.logger.warning(f"❌ 保存检查点失败: {e}")
                
                # 定期保存模型（使用范围触发）
                steps_since_last_save = self.current_step - getattr(self, '_last_save_step', 0)
                if steps_since_last_save >= save_frequency and self.current_step > 0:
                    self.logger.info(f"💾 定期保存模型 (步数: {self.current_step}, 距上次保存: {steps_since_last_save} 步)")
                    try:
                        if self.session_manager:
                            save_path = self.session_manager.get_model_save_path(f"model_step_{self.current_step}.zip")
                            if self.save_model(save_path):
                                self._last_save_step = self.current_step  # 记录最后保存步数
                                if self.visualization_manager:
                                    self.visualization_manager.on_model_save("定期", path=str(save_path))
                                self.logger.info(f"✅ 模型已保存: {save_path}")
                            else:
                                self.logger.warning(f"❌ 模型保存失败")
                    except Exception as e:
                        self.logger.warning(f"❌ 定期保存模型失败: {e}")
                
                # 进度日志
                recent_rewards = self.training_stats['episode_rewards'][-10:] if self.training_stats['episode_rewards'] else [0.0]
                avg_reward = np.mean(recent_rewards)
                self.logger.info(f"基座模型训练进度: {self.current_step:,}/{self.total_timesteps:,} "
                               f"({self.current_step/self.total_timesteps:.1%}) | "
                               f"阶段: {self.current_phase} | "
                               f"平均奖励: {avg_reward:.3f} | "
                               f"Episodes: {episode_count}")
                
                # 记录统计
                batch_stats = {
                    'steps': steps_collected,
                    'total_steps': self.current_step,
                    'phase': self.current_phase,
                    'avg_reward': avg_reward
                }
                training_stats.append(batch_stats)
                
            except Exception as e:
                self.logger.error(f"训练步骤失败: {e}")
                import traceback
                traceback.print_exc()
                # 即使出错也要更新进度，避免卡住
                self.current_step += steps_collected  # 使用实际收集的步数
                if self.visualization_manager:
                    error_metrics = {
                        'episode': episode_count,
                        'total_reward': 0.0,
                        'exploration_rate': 0.0,
                        'policy_loss': 0.0,
                        'value_loss': 0.0,
                        'phase': self.current_phase
                    }
                    self.visualization_manager.on_step(self.current_step, error_metrics)
                break
        
        # 最终训练步骤
        if rollout_buffer:
            self._train_step(rollout_buffer)
        
        # 汇总训练统计
        elapsed_time = time.time() - start_time
        
        # 最终评估
        final_metrics = self._final_evaluation()
        final_metrics.update({
            'training_time': elapsed_time,
            'training_completion_rate': self.current_step / self.total_timesteps if self.total_timesteps > 0 else 0.0,
            'training_batches': len(training_stats),
            'total_episodes': episode_count
        })
        
        self.logger.info(f"基座模型训练完成: 完成步数={self.current_step}, 最终奖励={final_metrics.get('avg_episode_reward', 0.0):.3f}")
        
        return final_metrics
    
    def _update_curriculum_phase(self, step: int) -> None:
        """更新课程学习阶段"""
        
        if step < self.hover_training_steps and self.current_phase != "hover":
            self.current_phase = "hover"
            self.phase_start_step = step
            self.logger.info(f"切换到悬停训练阶段 (步骤 {step})")
            
        elif step >= self.hover_training_steps and self.current_phase != "flight":
            self.current_phase = "flight" 
            self.phase_start_step = step
            self.logger.info(f"切换到飞行训练阶段 (步骤 {step})")
    
    def _train_step(self, rollout_buffer: List[Dict]) -> None:
        """执行一步训练"""
        
        # 转换为张量
        batch_obs = torch.FloatTensor([exp['obs'] for exp in rollout_buffer])
        batch_actions = torch.FloatTensor([exp['action'] for exp in rollout_buffer])
        batch_rewards = torch.FloatTensor([exp['reward'] for exp in rollout_buffer])
        batch_values = torch.FloatTensor([exp['value'] for exp in rollout_buffer])
        batch_dones = torch.FloatTensor([exp['done'] for exp in rollout_buffer])
        
        # 计算GAE优势
        advantages = self._compute_gae(batch_rewards, batch_values, batch_dones)
        returns = advantages + batch_values
        
        # 归一化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 前向传播
        pred_actions, pred_values = self.model(batch_obs)
        
        # 计算损失
        value_loss = nn.MSELoss()(pred_values.squeeze(), returns)
        
        # PPO策略损失 (简化版)
        action_diff = (pred_actions - batch_actions).pow(2).mean()
        policy_loss = action_diff - 0.01 * advantages.mean()  # 简化的策略损失
        
        total_loss = value_loss + policy_loss
        
        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        # 记录统计
        self.training_stats['value_losses'].append(value_loss.item())
        self.training_stats['policy_losses'].append(policy_loss.item())
    
    def _compute_gae(self, 
                    rewards: torch.Tensor, 
                    values: torch.Tensor, 
                    dones: torch.Tensor) -> torch.Tensor:
        """计算GAE优势"""
        
        advantages = torch.zeros_like(rewards)
        gae = 0
        
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[i + 1] * (1 - dones[i])
            
            delta = rewards[i] + self.gamma * next_value - values[i]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[i]) * gae
            advantages[i] = gae
        
        return advantages
    
    def _handle_episode_end(self, 
                           episode_reward: float, 
                           episode_length: int,
                           info: Dict[str, Any]) -> None:
        """处理回合结束"""
        
        self.training_stats['episode_rewards'].append(episode_reward)
        self.training_stats['episode_lengths'].append(episode_length)
        
        # 根据当前阶段记录成功率
        if self.current_phase == "hover":
            success = info.get('hover_success', False)
            self.training_stats['hover_success_rate'].append(1.0 if success else 0.0)
        else:
            success = info.get('flight_success', False)
            self.training_stats['flight_success_rate'].append(1.0 if success else 0.0)
    
    def _report_progress(self, step: int, episode_count: int) -> None:
        """报告训练进度"""
        
        if len(self.training_stats['episode_rewards']) > 0:
            recent_rewards = self.training_stats['episode_rewards'][-10:]
            avg_reward = np.mean(recent_rewards)
            
            if self.current_phase == "hover" and self.training_stats['hover_success_rate']:
                recent_success = self.training_stats['hover_success_rate'][-10:]
                success_rate = np.mean(recent_success)
            elif self.current_phase == "flight" and self.training_stats['flight_success_rate']:
                recent_success = self.training_stats['flight_success_rate'][-10:]
                success_rate = np.mean(recent_success)
            else:
                success_rate = 0.0
            
            progress_data = {
                'step': step,
                'episode': episode_count,
                'phase': self.current_phase,
                'avg_reward': avg_reward,
                'success_rate': success_rate,
                'progress': step / self.total_timesteps
            }
            
            self.logger.info(
                f"步骤 {step}/{self.total_timesteps} | "
                f"阶段: {self.current_phase} | "
                f"平均奖励: {avg_reward:.2f} | "
                f"成功率: {success_rate:.2%}"
            )
            
            # 调用进度回调
            for callback in self.progress_callbacks:
                callback(self.stage, progress_data)
    
    def _perform_evaluation(self) -> Dict[str, float]:
        """执行评估"""
        num_eval_episodes = self.config.get('eval_episodes', 10)
        
        if self.visualization_manager:
            self.visualization_manager.on_evaluation_start(num_eval_episodes)
        
        try:
            eval_results = self.evaluate(num_eval_episodes)
            
            # 评估结果分析和日志
            success_rate = eval_results.get('success_rate', 0.0)
            mean_reward = eval_results.get('mean_reward', 0.0)
            
            if success_rate > 0.8:
                self.logger.info(f"🎉 优秀性能: 成功率 {success_rate:.2%}, 平均奖励 {mean_reward:.2f}")
            elif success_rate > 0.5:
                self.logger.info(f"✅ 良好性能: 成功率 {success_rate:.2%}, 平均奖励 {mean_reward:.2f}")
            else:
                self.logger.warning(f"⚠️ 需要改进: 成功率 {success_rate:.2%}, 平均奖励 {mean_reward:.2f}")
            
            self.write_stage_message(f"评估完成: 成功率 {success_rate:.1%}")
            return eval_results
            
        except Exception as e:
            self.logger.error(f"评估失败: {e}")
            return {'mean_reward': 0.0, 'success_rate': 0.0, 'error': str(e)}
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, Any]:
        """评估方法 - 修复版本，添加超时和强制终止机制"""
        if not self.model or not self.env:
            return {'mean_reward': 0.0, 'success_rate': 0.0, 'error': 'Model or environment not initialized'}
        
        episode_rewards = []
        episode_lengths = []
        success_count = 0
        
        # 设置为评估模式
        self.model.eval()
        
        try:
            # 🔧 减少评估episode数，避免长时间评估
            eval_episodes = min(num_episodes, 5)
            self.logger.info(f"开始评估: {eval_episodes} episodes")
            
            for episode in range(eval_episodes):
                try:
                    obs, info = self.env.reset()
                    episode_reward = 0
                    episode_length = 0
                    done = False
                    
                    # 🔧 减少最大步数，避免episode过长
                    max_steps = 150  # 固定150步最大限制
                    
                    while not done and episode_length < max_steps:
                        # 使用模型预测动作
                        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                        action = self.model.get_action(obs_tensor, deterministic=True)
                        action_np = action.squeeze(0).cpu().numpy()
                        
                        obs, reward, terminated, truncated, info = self.env.step(action_np)
                        
                        episode_reward += reward
                        episode_length += 1
                        done = terminated or truncated
                        
                        # 🔧 增强的成功条件判断
                        if self._is_success(terminated, truncated, info, episode_length):
                            success_count += 1
                            break
                    
                    episode_rewards.append(episode_reward)
                    episode_lengths.append(episode_length)
                    self.logger.info(f"评估 episode {episode + 1}: 奖励={episode_reward:.2f}, 步数={episode_length}")
                    
                except Exception as e:
                    self.logger.error(f"评估 episode {episode + 1} 失败: {e}")
                    episode_rewards.append(0.0)
                    episode_lengths.append(0)
                    continue
            
            # 恢复训练模式
            self.model.train()
            
            # 确保有数据
            if not episode_rewards:
                episode_rewards = [0.0]
                episode_lengths = [0]
            
            return {
                'mean_reward': float(np.mean(episode_rewards)),
                'std_reward': float(np.std(episode_rewards)),
                'success_rate': success_count / eval_episodes if eval_episodes > 0 else 0.0,
                'mean_episode_length': float(np.mean(episode_lengths)),
                'episodes_evaluated': eval_episodes
            }
            
        except Exception as e:
            self.model.train()
            self.logger.error(f"评估过程中出错: {e}")
            return {'mean_reward': 0.0, 'success_rate': 0.0, 'error': str(e)}
    
    def _is_success(self, terminated: bool, truncated: bool, info: dict, episode_length: int) -> bool:
        """判断成功条件"""
        if not terminated:
            return False
        
        # 1. 显式成功标志
        if info.get('hover_success', False) or info.get('flight_success', False):
            return True
        
        # 2. 基于阶段的成功判断
        if self.current_phase == "hover":
            # 悬停成功：稳定时间 > 阈值
            if info.get('stable_time', 0) > 100:
                return True
        elif self.current_phase == "flight":
            # 飞行成功：距离目标 < 阈值
            if info.get('distance_to_target', float('inf')) < 0.5:
                return True
        
        # 3. 基于奖励阈值的成功判断
        total_reward = info.get('total_reward', 0.0)
        if total_reward > 50:  # 自定义奖励阈值
            return True
        
        # 4. 基于episode长度的成功判断（避免早期终止）
        if episode_length > 800:  # 长期存活
            return True
            
        return False
    
    def save_model(self, path: Path, metadata: Optional[Dict] = None) -> bool:
        """保存基座模型"""
        if not self.model:
            self.logger.error("模型未初始化，无法保存")
            return False
        
        try:
            # 保存PyTorch模型，包含训练进度
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
                'current_step': getattr(self, 'current_step', 0),  # 保存当前训练步数
                'current_phase': getattr(self, 'current_phase', 'hover'),  # 保存当前训练阶段
                'config': self.config,
                'training_stats': self.training_stats,
                'metadata': metadata or {}
            }, str(path))
            
            self.logger.info(f"✅ 基座模型已保存: {path} (步数: {getattr(self, 'current_step', 0)})")
            return True
        except Exception as e:
            self.logger.error(f"模型保存失败: {e}")
            return False
    
    def load_model(self, path: Path) -> bool:
        """加载基座模型"""
        if not self.model:
            self.logger.error("模型未初始化，无法加载")
            return False
        
        try:
            # 修复PyTorch 2.6的weights_only问题
            checkpoint = torch.load(str(path), weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            if self.optimizer and 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # 恢复训练进度
            if 'current_step' in checkpoint:
                self.current_step = checkpoint['current_step']
                self.logger.info(f"续训起始步数: {self.current_step}")
            
            # 恢复训练阶段
            if 'current_phase' in checkpoint:
                self.current_phase = checkpoint['current_phase']
                self.logger.info(f"续训起始阶段: {self.current_phase}")
            
            self.logger.info(f"✅ 基座模型已加载: {path}")
            return True
        except Exception as e:
            self.logger.error(f"模型加载失败: {e}")
            return False
    
    def _try_resume_training(self):
        """尝试加载之前的模型进行续训"""
        if not self.config.get('resume_training', True):
            self.logger.info("续训功能已禁用，从零开始训练")
            return
        
        # 查找可加载的模型
        model_paths_to_try = []
        
        # 1. 搜索所有历史foundation训练会话
        logs_dir = Path("logs")
        if logs_dir.exists():
            # 查找所有foundation训练目录
            foundation_dirs = list(logs_dir.glob("train_foundation_*"))
            foundation_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)  # 按修改时间倒序
            
            self.logger.info(f"🔍 发现 {len(foundation_dirs)} 个foundation训练会话")
            
            # 遍历查找可用模型
            for session_dir in foundation_dirs:
                model_dir = session_dir / "Model"
                if not model_dir.exists():
                    continue
                
                # 查找最佳模型
                best_model_path = model_dir / "best_model_foundation.zip"
                if best_model_path.exists():
                    model_paths_to_try.append(("最佳模型", best_model_path))
                    self.logger.debug(f"找到最佳模型: {best_model_path}")
                
                # 查找检查点
                try:
                    checkpoints = list(model_dir.glob("checkpoint_*.zip"))
                    if checkpoints:
                        # 按文件名中的步数排序，取最新的
                        latest_checkpoint = max(checkpoints, key=lambda p: int(p.stem.split('_')[1]))
                        model_paths_to_try.append(("检查点", latest_checkpoint))
                        self.logger.debug(f"找到检查点: {latest_checkpoint}")
                except Exception as e:
                    self.logger.debug(f"解析检查点失败: {e}")
                
                # 如果已经找到足够多的候选模型，就不继续搜索了
                if len(model_paths_to_try) >= 3:
                    break
        
        # 优先级排序：最新的最佳模型 > 最新的检查点
        if model_paths_to_try:
            self.logger.info(f"📦 找到 {len(model_paths_to_try)} 个可加载的模型")
            
            # 尝试加载模型
            for model_type, model_path in model_paths_to_try:
                try:
                    if self.load_model(model_path):
                        self.logger.info(f"🔄 续训已启动，加载{model_type}: {model_path}")
                        self.write_stage_message(f"续训模式 - 加载{model_type}")
                        return
                except Exception as e:
                    self.logger.warning(f"加载{model_type}失败: {e}")
        
        self.logger.info("🆕 未找到可加载的模型，从零开始训练")
        self.write_stage_message("新训练 - 从零开始")
    
    def on_step_callback(self, step: int, metrics: Dict[str, Any]):
        """步骤回调"""
        super().on_step_callback(step, metrics)
        
        # 更新可视化管理器
        if self.visualization_manager:
            step_metrics = {
                'step': step,
                'mean_reward': metrics.get('total_reward', 0.0),
                'episode_length': metrics.get('episode_length', 0),
                'exploration_rate': metrics.get('exploration_rate', 0.0),
                'loss': metrics.get('value_loss', 0.0),
                'phase': metrics.get('phase', self.current_phase)
            }
            
            try:
                self.visualization_manager.on_step(step, step_metrics)
            except AttributeError:
                self.visualization_manager.update_metrics(step_metrics)
    
    def on_episode_callback(self, episode: int, metrics: Dict[str, Any]):
        """Episode回调"""
        self.current_episode = episode
        
        episode_reward = metrics.get('total_reward', 0.0)
        if episode_reward > self.best_reward:
            self.best_reward = episode_reward
            
            # 保存最佳模型
            if self.session_manager:
                best_model_path = self.session_manager.get_model_save_path("best_model")
                self.save_model(best_model_path)
                
                if self.visualization_manager:
                    self.visualization_manager.on_model_save(
                        "最佳", 
                        self.best_reward, 
                        str(best_model_path)
                    )
        
        if self.visualization_manager:
            episode_info = {
                'episode': episode,
                'mean_reward': episode_reward,
                'episode_length': metrics.get('episode_length', 0),
                'is_best': episode_reward > self.best_reward,
                'phase': self.current_phase
            }
            try:
                self.visualization_manager.on_episode_end(episode, episode_info)
            except AttributeError:
                self.visualization_manager.update_metrics(episode_info)
    
    def on_evaluation_callback(self, eval_results: Dict[str, float]):
        """评估回调"""
        self.evaluation_history.append(eval_results)
        
        # 📊 记录评估结果到TensorBoard
        self._log_evaluation_to_tensorboard(eval_results)
        
        if self.visualization_manager:
            try:
                self.visualization_manager.on_evaluation_end(eval_results)
            except AttributeError:
                self.visualization_manager.update_metrics(eval_results)
    
    def on_checkpoint_callback(self, step: int):
        """检查点回调"""
        if self.session_manager:
            checkpoint_path = self.session_manager.get_model_save_path(f"checkpoint_{step}.zip")
            if self.save_model(checkpoint_path):
                if self.visualization_manager:
                    self.visualization_manager.write_stage_message(f"检查点已保存: step {step}")
            else:
                self.logger.error(f"检查点保存失败: {checkpoint_path}")
    
    def write_stage_message(self, message: str):
        """写入阶段消息"""
        if self.visualization_manager:
            self.visualization_manager.write_stage_message(message)
        else:
            self.logger.info(message)
    
    def _final_evaluation(self) -> Dict[str, Any]:
        """最终评估 - 增强版本，包含悬停质量和飞行质量详细分析"""
        eval_episodes = 5  # 减少评估episode数，避免长时间卡住
        eval_rewards = []
        hover_successes = 0
        flight_successes = 0
        
        # 详细质量指标
        hover_quality_scores = []
        flight_quality_scores = []
        position_stability_scores = []
        velocity_smoothness_scores = []
        
        # 设置为评估模式
        self.model.eval()
        
        try:
            self.logger.info(f"开始最终评估: {eval_episodes} episodes")
            self.logger.info("=" * 60)
            self.logger.info("🎯 基座模型评估 - 悬停与飞行质量分析")
            self.logger.info("=" * 60)
            
            for episode in range(eval_episodes):
                self.logger.info(f"\n📋 Episode {episode + 1}/{eval_episodes} 开始评估")
                
                try:
                    obs, _ = self.env.reset()
                    episode_reward = 0
                    episode_success = False
                    
                    # 质量评估指标
                    positions = []
                    velocities = []
                    hover_quality = 0.0
                    flight_quality = 0.0
                    
                    # 🔧 减少最大步数，避免episode过长
                    max_steps = min(self.config.get('max_episode_steps', 1000), 200)  # 最多200步
                    
                    for step in range(max_steps):
                        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                        action = self.model.get_action(obs_tensor, deterministic=True)
                        action_np = action.squeeze(0).cpu().numpy()
                        
                        obs, reward, terminated, truncated, info = self.env.step(action_np)
                        episode_reward += reward
                        
                        # 收集位置和速度数据用于质量分析
                        if hasattr(self.env, 'pos') and hasattr(self.env, 'vel'):
                            positions.append(self.env.pos[0].copy())  # 取第一个无人机的位置
                            velocities.append(self.env.vel[0].copy())  # 取第一个无人机的速度
                        
                        # 🔧 强化终止条件
                        if terminated or truncated:
                            hover_quality = info.get('hover_quality', 0.0)
                            flight_quality = info.get('flight_quality', 0.0)
                            
                            if info.get('hover_success', False):
                                hover_successes += 1
                                episode_success = True
                                self.logger.info(f"  ✅ 悬停成功! 悬停质量: {hover_quality:.3f}")
                            if info.get('flight_success', False):
                                flight_successes += 1
                                episode_success = True
                                self.logger.info(f"  ✅ 飞行成功! 飞行质量: {flight_quality:.3f}")
                            break
                        
                        # 🔧 添加早期成功判断
                        if step > 100 and episode_reward > 50:  # 100步后如果奖励足够高就认为成功
                            episode_success = True
                            # 估算质量分数
                            hover_quality = min(episode_reward / 100.0, 1.0)
                            flight_quality = min(episode_reward / 200.0, 1.0)
                            self.logger.info(f"  ⭐ 早期完成! 估算悬停质量: {hover_quality:.3f}, 飞行质量: {flight_quality:.3f}")
                            break
                    
                    # 计算详细质量指标
                    if len(positions) > 10:  # 确保有足够数据
                        position_stability = self._calculate_position_stability(positions)
                        velocity_smoothness = self._calculate_velocity_smoothness(velocities)
                        position_stability_scores.append(position_stability)
                        velocity_smoothness_scores.append(velocity_smoothness)
                        
                        self.logger.info(f"  📊 位置稳定性: {position_stability:.3f}")
                        self.logger.info(f"  📊 速度平滑性: {velocity_smoothness:.3f}")
                    else:
                        position_stability_scores.append(0.0)
                        velocity_smoothness_scores.append(0.0)
                    
                    hover_quality_scores.append(hover_quality)
                    flight_quality_scores.append(flight_quality)
                    eval_rewards.append(episode_reward)
                    
                    self.logger.info(f"  📈 Episode {episode + 1} 完成:")
                    self.logger.info(f"    奖励: {episode_reward:.2f}")
                    self.logger.info(f"    步数: {step + 1}")
                    self.logger.info(f"    成功: {'是' if episode_success else '否'}")
                    self.logger.info(f"    悬停质量: {hover_quality:.3f}")
                    self.logger.info(f"    飞行质量: {flight_quality:.3f}")
                    
                except Exception as e:
                    self.logger.error(f"评估 episode {episode + 1} 失败: {e}")
                    eval_rewards.append(0.0)  # 添加默认奖励避免空列表
                    hover_quality_scores.append(0.0)
                    flight_quality_scores.append(0.0)
                    position_stability_scores.append(0.0)
                    velocity_smoothness_scores.append(0.0)
                    continue
                    
        except Exception as e:
            self.logger.error(f"最终评估失败: {e}")
            # 确保有默认值
            if not eval_rewards:
                eval_rewards = [0.0]
                hover_quality_scores = [0.0]
                flight_quality_scores = [0.0]
                position_stability_scores = [0.0]
                velocity_smoothness_scores = [0.0]
        finally:
            # 恢复训练模式
            self.model.train()
            
        # 打印详细评估总结
        self._print_evaluation_summary(
            eval_rewards, hover_quality_scores, flight_quality_scores,
            position_stability_scores, velocity_smoothness_scores,
            hover_successes, flight_successes, eval_episodes
        )
        
        # 计算统计指标
        final_metrics = {
            'avg_episode_reward': float(np.mean(eval_rewards)) if eval_rewards else 0.0,
            'std_episode_reward': float(np.std(eval_rewards)) if eval_rewards else 0.0,
            'hover_success_rate': hover_successes / eval_episodes if eval_episodes > 0 else 0.0,
            'flight_success_rate': flight_successes / eval_episodes if eval_episodes > 0 else 0.0,
            'avg_hover_quality': float(np.mean(hover_quality_scores)) if hover_quality_scores else 0.0,
            'avg_flight_quality': float(np.mean(flight_quality_scores)) if flight_quality_scores else 0.0,
            'avg_position_stability': float(np.mean(position_stability_scores)) if position_stability_scores else 0.0,
            'avg_velocity_smoothness': float(np.mean(velocity_smoothness_scores)) if velocity_smoothness_scores else 0.0,
            'total_episodes_trained': len(self.training_stats['episode_rewards']),
            'final_avg_reward': float(np.mean(self.training_stats['episode_rewards'][-10:])) if self.training_stats['episode_rewards'] else 0.0,
            'eval_episodes_completed': len(eval_rewards)
        }
        
        return final_metrics
    
    def _calculate_position_stability(self, positions: List[np.ndarray]) -> float:
        """计算位置稳定性分数 (0-1，越高越稳定)"""
        if len(positions) < 2:
            return 0.0
        
        positions_array = np.array(positions)
        
        # 计算位置变化的标准差
        position_std = np.std(positions_array, axis=0)
        avg_std = np.mean(position_std)
        
        # 转换为0-1分数，较小的标准差表示更好的稳定性
        stability_score = max(0.0, min(1.0, 1.0 - avg_std / 2.0))
        
        return stability_score
    
    def _calculate_velocity_smoothness(self, velocities: List[np.ndarray]) -> float:
        """计算速度平滑性分数 (0-1，越高越平滑)"""
        if len(velocities) < 3:
            return 0.0
        
        velocities_array = np.array(velocities)
        
        # 计算速度变化的加速度
        accelerations = np.diff(velocities_array, axis=0)
        acceleration_magnitude = np.linalg.norm(accelerations, axis=1)
        
        # 计算平均加速度幅度
        avg_acceleration = np.mean(acceleration_magnitude)
        
        # 转换为0-1分数，较小的加速度变化表示更平滑的运动
        smoothness_score = max(0.0, min(1.0, 1.0 - avg_acceleration / 10.0))
        
        return smoothness_score
    
    def _print_evaluation_summary(self, eval_rewards: List[float], 
                                hover_quality_scores: List[float],
                                flight_quality_scores: List[float],
                                position_stability_scores: List[float],
                                velocity_smoothness_scores: List[float],
                                hover_successes: int, flight_successes: int, 
                                total_episodes: int):
        """打印详细的评估总结"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("🎯 基座模型评估总结报告")
        self.logger.info("=" * 80)
        
        # 基本统计
        avg_reward = np.mean(eval_rewards) if eval_rewards else 0.0
        std_reward = np.std(eval_rewards) if eval_rewards else 0.0
        
        self.logger.info(f"\n📊 基本性能指标:")
        self.logger.info(f"  总 Episodes: {total_episodes}")
        self.logger.info(f"  平均奖励: {avg_reward:.2f} ± {std_reward:.2f}")
        self.logger.info(f"  奖励范围: [{min(eval_rewards):.2f}, {max(eval_rewards):.2f}]")
        
        # 成功率统计
        hover_success_rate = hover_successes / total_episodes * 100
        flight_success_rate = flight_successes / total_episodes * 100
        
        self.logger.info(f"\n🎯 任务成功率:")
        self.logger.info(f"  悬停成功率: {hover_success_rate:.1f}% ({hover_successes}/{total_episodes})")
        self.logger.info(f"  飞行成功率: {flight_success_rate:.1f}% ({flight_successes}/{total_episodes})")
        
        # 质量分析
        avg_hover_quality = np.mean(hover_quality_scores) if hover_quality_scores else 0.0
        avg_flight_quality = np.mean(flight_quality_scores) if flight_quality_scores else 0.0
        avg_position_stability = np.mean(position_stability_scores) if position_stability_scores else 0.0
        avg_velocity_smoothness = np.mean(velocity_smoothness_scores) if velocity_smoothness_scores else 0.0
        
        self.logger.info(f"\n⭐ 飞行质量分析:")
        self.logger.info(f"  悬停质量: {avg_hover_quality:.3f}/1.000")
        self.logger.info(f"  飞行质量: {avg_flight_quality:.3f}/1.000")
        self.logger.info(f"  位置稳定性: {avg_position_stability:.3f}/1.000")
        self.logger.info(f"  速度平滑性: {avg_velocity_smoothness:.3f}/1.000")
        
        # 质量等级评估
        overall_quality = (avg_hover_quality + avg_flight_quality + 
                          avg_position_stability + avg_velocity_smoothness) / 4.0
        
        if overall_quality >= 0.8:
            quality_level = "优秀 🌟"
        elif overall_quality >= 0.6:
            quality_level = "良好 ✅"
        elif overall_quality >= 0.4:
            quality_level = "一般 ⚠️"
        else:
            quality_level = "需要改进 ❌"
            
        self.logger.info(f"\n🏆 综合质量评级:")
        self.logger.info(f"  总体质量分数: {overall_quality:.3f}/1.000")
        self.logger.info(f"  质量等级: {quality_level}")
        
        # 详细分析和建议
        self.logger.info(f"\n📋 详细分析:")
        
        if avg_hover_quality < 0.5:
            self.logger.info("  ⚠️  悬停质量较低，建议增加悬停训练时间")
        elif avg_hover_quality >= 0.8:
            self.logger.info("  ✅ 悬停能力优秀")
            
        if avg_flight_quality < 0.5:
            self.logger.info("  ⚠️  飞行质量较低，建议增加飞行路径训练")
        elif avg_flight_quality >= 0.8:
            self.logger.info("  ✅ 飞行能力优秀")
            
        if avg_position_stability < 0.6:
            self.logger.info("  ⚠️  位置稳定性需要提升，建议调整PID参数")
        elif avg_position_stability >= 0.8:
            self.logger.info("  ✅ 位置控制稳定")
            
        if avg_velocity_smoothness < 0.6:
            self.logger.info("  ⚠️  速度变化过于剧烈，建议优化动作空间")
        elif avg_velocity_smoothness >= 0.8:
            self.logger.info("  ✅ 运动平滑自然")
        
        self.logger.info("=" * 80)
        self.logger.info("✅ 基座模型评估完成")
        self.logger.info("=" * 80 + "\n")
    
    def _log_to_tensorboard(self, metrics: Dict[str, Any]):
        """记录指标到TensorBoard"""
        if not self.tensorboard_writer:
            return
        
        try:
            step = self.current_step
            
            # 训练指标
            if 'total_reward' in metrics:
                self.tensorboard_writer.add_scalar('Train/EpisodeReward', metrics['total_reward'], step)
            
            if 'exploration_rate' in metrics:
                self.tensorboard_writer.add_scalar('Train/ExplorationRate', metrics['exploration_rate'], step)
            
            if 'policy_loss' in metrics and metrics['policy_loss'] > 0:
                self.tensorboard_writer.add_scalar('Train/PolicyLoss', metrics['policy_loss'], step)
            
            if 'value_loss' in metrics and metrics['value_loss'] > 0:
                self.tensorboard_writer.add_scalar('Train/ValueLoss', metrics['value_loss'], step)
            
            # 训练阶段
            phase_mapping = {'hover': 0, 'flight': 1}
            if 'phase' in metrics and metrics['phase'] in phase_mapping:
                self.tensorboard_writer.add_scalar('Train/TrainingPhase', phase_mapping[metrics['phase']], step)
            
            # 统计信息
            if self.training_stats['episode_rewards']:
                avg_reward = np.mean(self.training_stats['episode_rewards'][-10:])
                self.tensorboard_writer.add_scalar('Train/AvgReward_10ep', avg_reward, step)
            
            if self.training_stats['episode_lengths']:
                avg_length = np.mean(self.training_stats['episode_lengths'][-10:])
                self.tensorboard_writer.add_scalar('Train/AvgEpisodeLength', avg_length, step)
            
            # 成功率
            if self.training_stats['hover_success_rate']:
                self.tensorboard_writer.add_scalar('Train/HoverSuccessRate', 
                                                 self.training_stats['hover_success_rate'][-1], step)
            
            if self.training_stats['flight_success_rate']:
                self.tensorboard_writer.add_scalar('Train/FlightSuccessRate', 
                                                 self.training_stats['flight_success_rate'][-1], step)
            
            # 定期flush
            if step % 100 == 0:
                self.tensorboard_writer.flush()
                
        except Exception as e:
            # 不影响训练进程，只记录警告
            self.logger.warning(f"TensorBoard记录失败: {e}")
    
    def _log_evaluation_to_tensorboard(self, eval_results: Dict[str, Any]):
        """记录评估结果到TensorBoard"""
        if not self.tensorboard_writer:
            return
        
        try:
            step = self.current_step
            
            # 评估指标
            if 'avg_reward' in eval_results:
                self.tensorboard_writer.add_scalar('Eval/AvgReward', eval_results['avg_reward'], step)
            
            if 'success_rate' in eval_results:
                self.tensorboard_writer.add_scalar('Eval/SuccessRate', eval_results['success_rate'], step)
            
            if 'hover_quality' in eval_results:
                self.tensorboard_writer.add_scalar('Eval/HoverQuality', eval_results['hover_quality'], step)
            
            if 'flight_quality' in eval_results:
                self.tensorboard_writer.add_scalar('Eval/FlightQuality', eval_results['flight_quality'], step)
            
            if 'position_stability' in eval_results:
                self.tensorboard_writer.add_scalar('Eval/PositionStability', eval_results['position_stability'], step)
            
            if 'velocity_smoothness' in eval_results:
                self.tensorboard_writer.add_scalar('Eval/VelocitySmoothness', eval_results['velocity_smoothness'], step)
            
            self.tensorboard_writer.flush()
            
        except Exception as e:
            self.logger.warning(f"评估TensorBoard记录失败: {e}")
    
    def cleanup(self):
        """清理资源"""
        if self.tensorboard_writer:
            try:
                self.tensorboard_writer.close()
                self.logger.info("✅ TensorBoard Writer 已关闭")
            except Exception as e:
                self.logger.warning(f"关闭TensorBoard Writer失败: {e}")
        
        super().cleanup() if hasattr(super(), 'cleanup') else None
    
    def _start_trajectory_episode(self, episode_num: int):
        """开始新episode的轨迹记录"""
        if self.trajectory_manager:
            try:
                self.trajectory_manager.start_new_episode(episode_num)
                self.trajectory_episode_started = True
                self.current_episode_reward = 0.0
                self.current_episode_length = 0
            except Exception as e:
                self.logger.warning(f"启动轨迹记录失败: {e}")
                self.trajectory_episode_started = False
    
    def _log_trajectory_step(self, obs, action, reward, next_obs, done, info):
        """记录单步轨迹数据"""
        if not self.trajectory_manager or not self.trajectory_episode_started:
            return
        
        try:
            # 🔧 正确解析轨迹数据：从环境的get_trajectory_step_data方法获取准确数据
            trajectory_data = None
            if hasattr(self.env, 'get_trajectory_step_data'):
                try:
                    trajectory_data = self.env.get_trajectory_step_data(drone_idx=0)
                except Exception as e:
                    # 如果参数名不匹配，尝试无参数调用
                    try:
                        trajectory_data = self.env.get_trajectory_step_data()
                    except:
                        trajectory_data = None
            
            if trajectory_data is not None:
                # 使用环境提供的准确数据
                current_position = trajectory_data['current_position'].tolist() if isinstance(trajectory_data['current_position'], np.ndarray) else list(trajectory_data['current_position'])
                current_velocity = trajectory_data['current_velocity'].tolist() if isinstance(trajectory_data['current_velocity'], np.ndarray) else list(trajectory_data['current_velocity'])
                target_action = trajectory_data.get('target_velocity', [0.0, 0.0, 0.0, 0.0])
                rpm_action = trajectory_data.get('rpm_action', [0.0, 0.0, 0.0, 0.0])
                
                # 调试模式下打印实际数据
                if self.config.get('debug', False) and self.current_episode_length < 5:
                    self.logger.info(f"=== 轨迹记录调试 (步骤 {self.current_episode_length}) ===")
                    self.logger.info(f"实际位置: {current_position}")
                    self.logger.info(f"实际速度: {current_velocity}")
                    self.logger.info(f"目标动作: {target_action}")
                    self.logger.info(f"RPM输出: {rpm_action}")
                    if isinstance(info, dict) and 'direct_state' in info:
                        direct_state = info['direct_state']
                        if isinstance(direct_state, dict) and 'current_velocity' in direct_state:
                            self.logger.info(f"直接状态速度: {direct_state['current_velocity']}")
            else:
                # 后备方案：从观测数据解析（基于gym-pybullet-drones的KIN观测格式）
                if isinstance(obs, np.ndarray) and len(obs) >= 12:
                    # KIN观测格式：[x, y, z, qx, qy, qz, qw, roll, pitch, yaw, vx, vy, vz, wx, wy, wz, ...]
                    current_position = obs[:3].tolist()      # 位置 [0:3]
                    current_velocity = obs[10:13].tolist()   # 速度 [10:13]
                else:
                    current_position = [0.0, 0.0, 0.0]
                    current_velocity = [0.0, 0.0, 0.0]
                
                # 从info中获取控制器数据
                target_action = [0.0, 0.0, 0.0, 0.0]
                rpm_action = [0.0, 0.0, 0.0, 0.0]
                
                if isinstance(info, dict):
                    # 检查direct_state中的速度信息
                    if 'direct_state' in info and isinstance(info['direct_state'], dict):
                        direct_state = info['direct_state']
                        if 'current_velocity' in direct_state:
                            vel_tensor = direct_state['current_velocity']
                            if hasattr(vel_tensor, 'tolist'):
                                current_velocity = vel_tensor.tolist()
                            elif hasattr(vel_tensor, '__len__') and len(vel_tensor) >= 3:
                                current_velocity = list(vel_tensor[:3])
                    
                    # 获取控制相关数据
                    if 'incremental_control' in info and isinstance(info['incremental_control'], dict):
                        target_data = info['incremental_control'].get('drone_0_target', [0.0, 0.0, 0.0, 0.0])
                        if hasattr(target_data, '__len__') and len(target_data) >= 4:
                            target_action = list(target_data[:4])
            
            # 确保所有数据都是float类型
            current_position = [float(x) for x in current_position[:3]]
            current_velocity = [float(x) for x in current_velocity[:3]]
            target_velocity = target_action[:3] if target_action and len(target_action) >= 3 else [0.0, 0.0, 0.0]
            
            # 构建轨迹数据
            step_data = {
                'step': float(self.current_episode_length),
                'current_position': current_position,
                'current_velocity': current_velocity,
                'target_velocity': target_velocity,
                'model_action': action.tolist() if isinstance(action, np.ndarray) else list(action),
                'rpm_action': rpm_action.tolist() if isinstance(rpm_action, np.ndarray) else list(rpm_action) if rpm_action else [0.0, 0.0, 0.0, 0.0],
                'reward': float(reward),
                'exploration_rate': float(max(0.0, 1.0 - self.current_step / self.total_timesteps)),
                'done': bool(done)  # 确保是布尔值
            }
            
            self.trajectory_manager.log_step(step_data)
            self.current_episode_length += 1
            self.current_episode_reward += reward
            
        except Exception as e:
            self.logger.warning(f"轨迹记录失败: {e}")
            # 在调试模式下打印更多信息
            if self.config.get('debug', False):
                self.logger.warning(f"obs shape: {obs.shape if isinstance(obs, np.ndarray) else type(obs)}")
                self.logger.warning(f"info keys: {list(info.keys()) if isinstance(info, dict) else type(info)}")
                import traceback
                self.logger.warning(f"轨迹记录详细错误: {traceback.format_exc()}")
    
    def _finalize_trajectory_episode(self, episode_reward: float, episode_length: int, info: Dict[str, Any]):
        """完成episode的轨迹记录"""
        if not self.trajectory_manager or not self.trajectory_episode_started:
            return
        
        try:
            # 确定终止原因
            termination_reason = "unknown"
            if info.get('collision', False):
                termination_reason = "collision"
            elif info.get('timeout', False):
                termination_reason = "timeout"
            elif info.get('hover_success', False):
                termination_reason = "hover_success"
            elif info.get('flight_success', False):
                termination_reason = "flight_success"
            elif info.get('out_of_bounds', False):
                termination_reason = "out_of_bounds"
            else:
                termination_reason = "normal"
            
            # 最终exploration rate
            final_exploration_rate = max(0.0, 1.0 - self.current_step / self.total_timesteps)
            
            self.trajectory_manager.finalize_episode(
                termination_reason=termination_reason,
                final_exploration_rate=final_exploration_rate,
                total_reward=episode_reward
            )
            
            self.trajectory_episode_started = False
            
        except Exception as e:
            self.logger.warning(f"完成轨迹记录失败: {e}")
            self.trajectory_episode_started = False

    
def create_baseflight_trainer(config: Dict[str, Any], 
                             foundation_model_path: Optional[Path] = None) -> BaseFlightTrainer:
    """创建基座模型训练器的便捷函数"""
    return BaseFlightTrainer(config, foundation_model_path)


if __name__ == "__main__":
    # 简单测试
    test_config = {
        'total_timesteps': 10000,
        'hover_training_steps': 5000,
        'flight_training_steps': 5000,
        'evaluation_frequency': 2000,
        'eval_episodes': 5,
        'enable_trajectory': True,
        'enable_tensorboard': True,
        'enable_visualization': True,
        'enable_rich_display': True,
        'enable_curriculum': True
    }
    
    trainer = create_baseflight_trainer(test_config)
    
    print("开始测试基座模型训练器...")
    if trainer.setup():
        print("✅ 训练器设置成功")
        # 这里可以添加简短的训练测试
    else:
        print("❌ 训练器设置失败")
