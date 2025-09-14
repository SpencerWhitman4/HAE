#!/usr/bin/env python3

"""
分层训练器 - HA-UAV完整分层决策训练
集成智能会话管理和可视化系统
"""

import sys
import time
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

# 添加项目路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.training.core.base_trainer import BaseTrainer, TrainingStage, TrainingResult
from src.training.core.environment_factory import EnvironmentFactory
from src.training.core.training_adapter import create_training_adapter


class HierarchicalTrainer(BaseTrainer):
    """HA-UAV分层系统训练器 - 完整集成"""
    
    def __init__(self, 
                 config: Dict[str, Any], 
                 foundation_model_path: Optional[Path] = None,
                 output_dir: str = "./models"):
        super().__init__(
            stage=TrainingStage.HIERARCHICAL,
            config=config,
            experiment_name="HA-UAV",
            stage_variant=None
        )
        
        self.foundation_model_path = foundation_model_path
        self.output_dir = output_dir
        self.env_factory = EnvironmentFactory()
        self.env = None
        self.ha_components = None
        self.training_adapter = None
        
        # 初始化progress_callbacks
        self.progress_callbacks = []
        
        # 轨迹记录相关变量
        self.trajectory_manager = None
        self.trajectory_episode_started = False
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
        self.current_exploration_rate = 1.0
        
        # 更新配置以包含基座模型信息
        if foundation_model_path:
            self.config['foundation_model_path'] = str(foundation_model_path)
    
    def setup(self) -> bool:
        """设置HA-UAV环境和分层策略"""
        try:
            # 初始化会话管理（包含可视化）
            session_info = self.initialize_session(
                enable_trajectory=self.config.get('enable_trajectory', True),
                enable_tensorboard=self.config.get('enable_tensorboard', True),
                enable_visualization=self.config.get('enable_visualization', True),
                enable_rich_display=self.config.get('enable_rich_display', True)
            )
            
            self.logger.info(f"会话初始化完成: {session_info['session_dir']}")
            
            # 创建HAUAVAviary环境
            self.env = self.env_factory.create_environment(
                TrainingStage.HIERARCHICAL,
                self.config
            )
            self.logger.info("HAUAVAviary环境创建完成")
            
            # 初始化HA组件管理器
            from src.modules import HAComponentsManager, ModelConfiguration
            
            model_config = ModelConfiguration()
            # 从config中更新组件配置
            if 'ha_config' in self.config:
                for key, value in self.config['ha_config'].items():
                    if hasattr(model_config, key):
                        setattr(model_config, key, value)
            
            # 🎯 创建HAComponentsManager（核心组件）
            try:
                self.ha_components = HAComponentsManager(model_config)
                success = self.ha_components.initialize_components(self.env)
                if success:
                    self.logger.info("✅ HAComponentsManager初始化成功")
                else:
                    self.logger.error("❌ HAComponentsManager初始化失败")
                    return False
            except Exception as e:
                self.logger.error(f"HAComponentsManager创建失败: {e}")
                return False
            
            # 🎯 基座模型权重迁移（如果存在）
            if self.foundation_model_path and self.foundation_model_path.exists():
                try:
                    self._transfer_foundation_weights()
                    self.logger.info(f"✅ 基座模型权重迁移完成: {self.foundation_model_path}")
                except Exception as e:
                    self.logger.warning(f"权重迁移失败，继续训练: {e}")
            
            # 废弃TrainingAdapter，直接使用HAComponentsManager
            self.training_adapter = None  # 明确标记为废弃
            
            self.write_stage_message("HA-UAV分层训练器设置完成")
            return True
                
        except Exception as e:
            self.logger.error(f"分层训练器设置失败: {e}")
            if self.visualization_manager:
                self.visualization_manager.write_message(f"设置失败: {e}", "ERROR")
            return False
    
    def _transfer_foundation_weights(self):
        """实现基座模型到分层模型的权重迁移"""
        from stable_baselines3 import PPO
        
        try:
            # 加载基座PPO模型
            foundation_model = PPO.load(str(self.foundation_model_path))
            foundation_weights = foundation_model.policy.state_dict()
            
            # 迁移到分层策略的共享层
            if self.ha_components and self.ha_components.policy:
                hierarchical_weights = self.ha_components.policy.state_dict()
                
                # 权重映射规则（需要根据实际网络结构调整）
                transferred_count = 0
                for key, value in foundation_weights.items():
                    if key in hierarchical_weights and value.shape == hierarchical_weights[key].shape:
                        hierarchical_weights[key] = value.clone()
                        transferred_count += 1
                        self.logger.debug(f"权重迁移: {key}")
                
                # 加载迁移后的权重
                self.ha_components.policy.load_state_dict(hierarchical_weights)
                self.logger.info(f"成功迁移 {transferred_count} 个权重参数")
                
        except Exception as e:
            self.logger.error(f"权重迁移失败: {e}")
            raise
    
    def _execute_training(self) -> Dict[str, Any]:
        """执行分层训练逻辑 - 修复版本，确保回调正常工作"""
        total_timesteps = self.config.get('total_timesteps', 100000)
        evaluation_frequency = self.config.get('evaluation_frequency', 10000)
        checkpoint_frequency = self.config.get('checkpoint_frequency', 50000)
        
        # 🎯 关键：使用很小的学习间隔，确保回调及时触发
        learn_interval = min(100, total_timesteps // 20)  # 很小的间隔
        
        training_stats = []
        start_time = time.time()
        self.logger.info(f"开始HA-UAV分层训练: {total_timesteps:,} 步 (学习间隔: {learn_interval} 步)")
        
        # 🔧 立即调用第一次回调，确保进度条启动
        if self.visualization_manager:
            initial_metrics = {
                'episode': 0,
                'total_reward': 0.0,
                'exploration_rate': 1.0,
                'policy_loss': 0.0,
                'value_loss': 0.0
            }
            self.visualization_manager.on_step(0, initial_metrics)
            self.logger.info("✅ 初始回调已触发，进度条应已启动")
        
        while self.current_step < total_timesteps:
            try:
                if self.ha_components is None:
                    self.logger.error("HAComponentsManager未初始化，退出训练")
                    break
                
                # 🔧 使用超小批次，避免长时间卡住
                steps_to_collect = min(learn_interval, total_timesteps - self.current_step, 50)  # 最多50步
                
                self.logger.info(f"开始收集经验: {steps_to_collect} 步 (当前步数: {self.current_step})")
                
                # 限时收集经验，避免无限卡住
                collect_start = time.time()
                timeout = 10.0  # 10秒超时
                
                try:
                    # 🎯 关键修复：将轨迹记录集成到collect_rollout中
                    rollout_stats = self.ha_components.collect_rollout(
                        self.env, 
                        n_steps=steps_to_collect,
                        trajectory_callback=self  # 传递self作为轨迹记录回调
                    )
                    collect_time = time.time() - collect_start
                    
                    if collect_time > 5.0:
                        self.logger.warning(f"经验收集耗时过长: {collect_time:.1f}s")
                    
                except Exception as e:
                    self.logger.error(f"经验收集失败: {e}")
                    # 即使失败也要更新进度
                    rollout_stats = {
                        'total_steps': steps_to_collect,
                        'episodes': 0,
                        'mean_reward': 0.0
                    }
                
                # 🎯 立即更新步数和触发回调
                steps_collected = rollout_stats.get('total_steps', steps_to_collect)
                self.current_step += steps_collected
                episodes_this_batch = rollout_stats.get('episodes', 0)
                
                # 🔧 核心：立即触发回调，确保进度条更新
                step_metrics = {
                    'episode': episodes_this_batch,
                    'total_reward': rollout_stats.get('mean_reward', 0.0),
                    'exploration_rate': max(0.0, 1.0 - self.current_step / total_timesteps),
                    'policy_loss': 0.0,
                    'value_loss': 0.0
                }
                
                # 调用可视化管理器回调
                if self.visualization_manager:
                    self.visualization_manager.on_step(self.current_step, step_metrics)
                    self.logger.info(f"✅ 回调已触发: {self.current_step}/{total_timesteps} ({self.current_step/total_timesteps:.1%})")
                
                # 调用训练器回调
                self.on_step_callback(self.current_step, step_metrics)
                
                # 尝试策略更新（快速跳过如果条件不满足）
                try:
                    if hasattr(self.ha_components, 'buffer'):
                        # 正确获取缓冲区大小：使用pos属性
                        buffer_pos = getattr(self.ha_components.buffer, 'pos', 0)
                        buffer_full = getattr(self.ha_components.buffer, 'full', False)
                        # 动态调整最小批次大小，适应短期训练
                        min_size = min(self.config.get('batch_size', 32), total_timesteps // 2)
                        min_size = max(min_size, 10)  # 至少10步才进行更新
                        
                        # 缓冲区大小为pos（如果未满）或buffer_size（如果已满）
                        effective_buffer_size = buffer_pos if not buffer_full else self.ha_components.buffer.buffer_size
                        
                        if effective_buffer_size >= min_size:
                            self.logger.info(f"执行策略更新: buffer_size={effective_buffer_size}")
                            training_stats_batch = self.ha_components.update_policy_from_buffer()
                            step_metrics.update({
                                'policy_loss': training_stats_batch.get('policy_loss', 0.0),
                                'value_loss': training_stats_batch.get('value_loss', 0.0)
                            })
                            
                            # 再次回调包含训练指标
                            if self.visualization_manager:
                                self.visualization_manager.on_step(self.current_step, step_metrics)
                        else:
                            self.logger.info(f"跳过策略更新: buffer_size={effective_buffer_size} < {min_size}")
                except Exception as e:
                    self.logger.warning(f"策略更新失败: {e}")
                
                # Episode回调
                if episodes_this_batch > 0:
                    self.on_episode_callback(episodes_this_batch, rollout_stats)
                
                # 定期评估（快速跳过）
                if self.current_step % evaluation_frequency == 0 and self.current_step > 0:
                    self.logger.info(f"开始定期评估 (步数: {self.current_step})")
                    try:
                        eval_results = self._perform_evaluation()
                        self.on_evaluation_callback(eval_results)
                    except Exception as e:
                        self.logger.warning(f"评估失败: {e}")
                
                # 定期保存检查点
                if self.current_step % checkpoint_frequency == 0 and self.current_step > 0:
                    try:
                        self.on_checkpoint_callback(self.current_step)
                    except Exception as e:
                        self.logger.warning(f"保存检查点失败: {e}")
                
                # 进度日志
                recent_reward = rollout_stats.get('mean_reward', 0.0)
                self.logger.info(f"训练进度: {self.current_step:,}/{total_timesteps:,} "
                               f"({self.current_step/total_timesteps:.1%}) | "
                               f"奖励: {recent_reward:.3f} | "
                               f"Episodes: {episodes_this_batch}")
                
                # 记录统计
                training_stats.append(rollout_stats)
                
            except Exception as e:
                self.logger.error(f"训练步骤失败: {e}")
                import traceback
                traceback.print_exc()
                # 即使出错也要更新进度，避免卡住
                self.current_step += learn_interval
                if self.visualization_manager:
                    error_metrics = {
                        'episode': 0,
                        'total_reward': 0.0,
                        'exploration_rate': 0.0,
                        'policy_loss': 0.0,
                        'value_loss': 0.0
                    }
                    self.visualization_manager.on_step(self.current_step, error_metrics)
                break
        
        # 汇总训练统计
        elapsed_time = time.time() - start_time
        total_reward = sum(stats.get('mean_reward', 0.0) for stats in training_stats)
        total_episodes = sum(stats.get('episodes', 0) for stats in training_stats)
        
        final_stats = {
            'final_reward': total_reward / len(training_stats) if training_stats else 0.0,
            'training_time': elapsed_time,
            'training_completion_rate': self.current_step / total_timesteps if total_timesteps > 0 else 0.0,
            'training_batches': len(training_stats),
            'total_episodes': total_episodes
        }
        
        self.logger.info(f"HA-UAV分层训练完成: 完成步数={self.current_step}, 最终奖励={final_stats['final_reward']:.3f}")
        
        return final_stats
    
    def _perform_evaluation(self) -> Dict[str, float]:
        """执行评估 - 委托给HAComponentsManager并增强成功率判断"""
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
            # 不再调用 on_evaluation_callback 避免无限递归
            return eval_results
            
        except Exception as e:
            self.logger.error(f"评估失败: {e}")
            return {'mean_reward': 0.0, 'success_rate': 0.0, 'error': str(e)}
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, Any]:
        """评估方法 - 委托给HAComponentsManager并增强成功率判断"""
        self.logger.info(f"📋 开始评估: {num_episodes} Episodes")
        
        if not self.ha_components or not self.env:
            self.logger.error("❌ 组件或环境未初始化")
            return {'mean_reward': 0.0, 'success_rate': 0.0, 'error': 'Components not initialized'}
        
        episode_rewards = []
        episode_lengths = []
        success_count = 0
        
        # 设置为评估模式
        self.logger.info("🔧 设置为评估模式")
        self.ha_components.set_training_mode(False)
        
        try:
            for episode in range(num_episodes):
                self.logger.info(f"🎯 开始Episode {episode + 1}/{num_episodes}")
                obs_batch, info = self.env.reset()
                self.logger.info(f"✅ 环境重置完成，观测形状: {obs_batch.shape}")
                
                # 适配观测格式：从(NUM_DRONES, OBS_DIM)取第一个无人机
                if obs_batch.ndim > 1 and obs_batch.shape[0] > 0:
                    obs = obs_batch[0]  # [86] 单智能体观测
                else:
                    obs = obs_batch
                
                episode_reward = 0
                episode_length = 0
                done = False
                max_steps = 100  # 减少最大步数防止卡住
                
                self.logger.info(f"🔄 开始Episode {episode + 1} 执行循环，最大步数: {max_steps}")
                
                while not done and episode_length < max_steps:
                    # 🎯 直接使用HAComponentsManager的预测方法
                    action = self.ha_components.predict(obs)
                    
                    obs_batch, reward, terminated, truncated, info = self.env.step(action.reshape(1, -1))
                    
                    # 适配观测格式
                    if obs_batch.ndim > 1 and obs_batch.shape[0] > 0:
                        obs = obs_batch[0]
                    else:
                        obs = obs_batch
                    
                    episode_reward += reward
                    episode_length += 1
                    done = terminated or truncated
                    
                    # 每10步输出一次进度
                    if episode_length % 10 == 0:
                        self.logger.info(f"  📊 Episode {episode + 1} 步数: {episode_length}, 奖励: {episode_reward:.3f}")
                    
                    # 🔧 增强的成功条件判断
                    if self._is_success(terminated, truncated, info, episode_length):
                        success_count += 1
                        break
                
                self.logger.info(f"✅ Episode {episode + 1} 完成: 奖励={episode_reward:.3f}, 步数={episode_length}, 完成原因={'终止' if terminated else '截断' if truncated else '超时'}")
                
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
            
            # 恢复训练模式
            self.logger.info("🔧 恢复训练模式")
            self.ha_components.set_training_mode(True)
            
            self.logger.info(f"📈 评估统计: 平均奖励={np.mean(episode_rewards):.3f}, 成功率={success_count}/{num_episodes}")
            
            return {
                'mean_reward': float(np.mean(episode_rewards)),
                'std_reward': float(np.std(episode_rewards)),
                'success_rate': success_count / num_episodes,
                'mean_episode_length': float(np.mean(episode_lengths)),
                'episodes_evaluated': num_episodes
            }
            
        except Exception as e:
            self.ha_components.set_training_mode(True)
            self.logger.error(f"评估过程中出错: {e}")
            return {'mean_reward': 0.0, 'success_rate': 0.0, 'error': str(e)}
    
    def _is_success(self, terminated: bool, truncated: bool, info: dict, episode_length: int) -> bool:
        """增强的成功条件判断"""
        if not terminated:
            return False
        
        # 1. 显式成功标志
        if info.get('navigation_success', False):
            return True
        
        # 2. 探索完成
        if info.get('exploration_completed', False):
            return True
            
        # 3. 基于探索率的成功判断
        exploration_rate = info.get('exploration_rate', 0.0)
        if exploration_rate > 0.8:  # 80%以上探索率
            return True
        
        # 4. 基于奖励阈值的成功判断
        total_reward = info.get('total_reward', 0.0)
        if total_reward > 100:  # 自定义奖励阈值
            return True
        
        # 5. 基于episode长度的成功判断（避免早期终止）
        if episode_length > 500:  # 长期存活
            return True
            
        return False
        
        try:
            for episode in range(num_episodes):
                # HAUAVAviary.reset返回 (NUM_DRONES, OBS_DIM) 格式
                obs_batch, info = self.env.reset()
                # 使用适配器处理观测格式
                obs = self.training_adapter.adapt_observation_format(obs_batch)
                
                episode_reward = 0
                episode_length = 0
                done = False
                
                while not done:
                    # 使用适配器进行安全预测
                    action = self.training_adapter.safe_predict(obs, deterministic=True)
                    
                    # HAUAVAviary.step返回多智能体格式
                    env_output = self.env.step(action)
                    obs, reward, terminated, truncated, info = self.training_adapter.adapt_step_output(env_output)
                    
                    episode_reward += reward
                    episode_length += 1
                    done = terminated or truncated
                    
                    # 判断成功条件
                    if terminated and info.get('navigation_success', False):
                        success_count += 1
                
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
            
            return {
                'mean_reward': float(np.mean(episode_rewards)),
                'std_reward': float(np.std(episode_rewards)),
                'success_rate': success_count / num_episodes,
                'mean_episode_length': float(np.mean(episode_lengths)),
                'episodes_evaluated': num_episodes
            }
            
        except Exception as e:
            self.logger.error(f"评估过程中出错: {e}")
            return {
                'mean_reward': 0.0,
                'std_reward': 0.0,
                'success_rate': 0.0,
                'mean_episode_length': 0.0,
                'episodes_evaluated': 0,
                'error': str(e)
            }
    
    def _perform_final_evaluation(self) -> Dict[str, Any]:
        """重写最终评估，添加详细调试信息"""
        self.logger.info("🚁 开始最终评估...")
        
        try:
            final_eval_episodes = self.config.get('final_eval_episodes', 3)  # 减少到3个Episode
            self.logger.info(f"📊 最终评估配置: {final_eval_episodes} Episodes")
            
            if self.visualization_manager:
                self.visualization_manager.write_stage_message("🏗️ 开始最终评估...")
            
            self.logger.info("📋 调用evaluate方法...")
            final_eval_results = self.evaluate(final_eval_episodes)
            self.logger.info(f"📈 评估结果: {final_eval_results}")
            
            # 添加final_前缀区分最终评估
            final_results = {}
            for key, value in final_eval_results.items():
                final_results[f'final_{key}'] = value
            
            self.logger.info(f"🏗️ 最终评估完成 - 奖励: {final_eval_results.get('mean_reward', 0.0):.3f}")
            
            if self.visualization_manager:
                self.visualization_manager.write_stage_message(
                    f"🏗️ 最终评估完成 - 奖励: {final_eval_results.get('mean_reward', 0.0):.3f}"
                )
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"❌ 最终评估失败: {e}")
            import traceback
            traceback.print_exc()
            return {'final_evaluation_error': str(e)}
    
    def save_model(self, path: Path, metadata: Optional[Dict] = None) -> bool:
        """保存HA-UAV模型"""
        if not self.ha_components:
            self.logger.error("HA组件未初始化，无法保存模型")
            return False
        
        try:
            self.ha_components.save_model(str(path))
            self.logger.info(f"HA-UAV模型已保存: {path}")
            return True
        except Exception as e:
            self.logger.error(f"模型保存失败: {e}")
            return False
    
    def load_model(self, path: Path) -> bool:
        """加载HA-UAV模型"""
        if not self.ha_components or not self.env:
            self.logger.error("组件未初始化，无法加载模型")
            return False
        
        try:
            self.ha_components.load_model(str(path), self.env)
            self.logger.info(f"HA-UAV模型已加载: {path}")
            return True
        except Exception as e:
            self.logger.error(f"模型加载失败: {e}")
            return False


    def on_step_callback(self, step: int, metrics: Dict[str, Any]):
        """步骤回调 - 更新可视化和指标"""
        super().on_step_callback(step, metrics)
        
        # 更新可视化管理器
        if self.visualization_manager:
            # 构建标准化的步骤指标
            step_metrics = {
                'step': step,
                'mean_reward': metrics.get('mean_reward', 0.0),
                'episode_length': metrics.get('episode_length', 0),
                'buffer_size': metrics.get('buffer_size', 0),
                'exploration_rate': metrics.get('exploration_rate', 0.0),
                'loss': metrics.get('loss', 0.0),
                'learning_rate': metrics.get('learning_rate', 0.0)
            }
            
            # 使用正确的方法名
            try:
                self.visualization_manager.on_step(step, step_metrics)
            except AttributeError:
                # 如果方法不存在，尝试更新指标
                self.visualization_manager.update_metrics(step_metrics)
    
    def on_episode_callback(self, episode: int, metrics: Dict[str, Any]):
        """Episode回调"""
        self.current_episode = episode
        
        episode_reward = metrics.get('mean_reward', 0.0)
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
                'is_best': episode_reward > self.best_reward
            }
            # 使用正确的方法名
            try:
                self.visualization_manager.on_episode_end(episode, episode_info)
            except AttributeError:
                self.visualization_manager.update_metrics(episode_info)
    
    def on_evaluation_callback(self, eval_results: Dict[str, float]):
        """评估回调"""
        self.evaluation_history.append(eval_results)
        
        if self.visualization_manager:
            try:
                self.visualization_manager.on_evaluation_end(eval_results)
            except AttributeError:
                self.visualization_manager.update_metrics(eval_results)
    
    def on_checkpoint_callback(self, step: int):
        """检查点回调"""
        if self.session_manager:
            checkpoint_path = self.session_manager.get_model_save_path(f"checkpoint_{step}")
            self.save_model(checkpoint_path)
            
            if self.visualization_manager:
                self.visualization_manager.write_stage_message(f"检查点已保存: step {step}")
    
    def write_stage_message(self, message: str):
        """写入阶段消息"""
        if self.visualization_manager:
            self.visualization_manager.write_stage_message(message)
        else:
            self.logger.info(message)
    
    def write_message(self, message: str, msg_type: str = "INFO"):
        """写入消息"""
        if self.visualization_manager:
            self.visualization_manager.write_message(message, msg_type)
        else:
            if msg_type == "ERROR":
                self.logger.error(message)
            elif msg_type == "WARNING":
                self.logger.warning(message)
            else:
                self.logger.info(message)
    
    def _start_trajectory_episode(self, episode_num: int):
        """开始新episode的轨迹记录 - 与基础训练器一致"""
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
        """记录单步轨迹数据 - 复制基础训练器的实现"""
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
                    self.logger.info(f"=== 分层轨迹记录调试 (步骤 {self.current_episode_length}) ===")
                    self.logger.info(f"实际位置: {current_position}")
                    self.logger.info(f"实际速度: {current_velocity}")
                    self.logger.info(f"目标动作: {target_action}")
                    self.logger.info(f"RPM输出: {rpm_action}")
                    if isinstance(info, dict) and 'direct_state' in info:
                        direct_state = info['direct_state']
                        if isinstance(direct_state, dict) and 'current_velocity' in direct_state:
                            self.logger.info(f"直接状态速度: {direct_state['current_velocity']}")
            else:
                # 后备方案：从观测数据解析（基于HAUAVAviary的86维观测格式）
                if isinstance(obs, np.ndarray) and len(obs) >= 12:
                    # 86维观测格式的位置和速度信息
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
                    
                    # 获取分层控制相关数据
                    if 'hierarchical_control' in info and isinstance(info['hierarchical_control'], dict):
                        target_data = info['hierarchical_control'].get('high_level_target', [0.0, 0.0, 0.0, 0.0])
                        if hasattr(target_data, '__len__') and len(target_data) >= 4:
                            target_action = list(target_data[:4])
                    elif 'incremental_control' in info and isinstance(info['incremental_control'], dict):
                        target_data = info['incremental_control'].get('drone_0_target', [0.0, 0.0, 0.0, 0.0])
                        if hasattr(target_data, '__len__') and len(target_data) >= 4:
                            target_action = list(target_data[:4])
            
            # 确保所有数据都是float类型
            current_position = [float(x) for x in current_position[:3]]
            current_velocity = [float(x) for x in current_velocity[:3]]
            target_velocity = target_action[:3] if target_action and len(target_action) >= 3 else [0.0, 0.0, 0.0]
            
            # 构建轨迹数据 - 适配HAUAVAviary
            step_data = {
                'step': float(self.current_episode_length),
                'current_position': current_position,
                'current_velocity': current_velocity,
                'target_velocity': target_velocity,
                'model_action': action.tolist() if isinstance(action, np.ndarray) else list(action),
                'rpm_action': rpm_action.tolist() if isinstance(rpm_action, np.ndarray) else list(rpm_action) if rpm_action else [0.0, 0.0, 0.0, 0.0],
                'exploration_rate': float(getattr(self, 'current_exploration_rate', 1.0))
            }
            
            self.trajectory_manager.log_step(step_data)
            self.current_episode_length += 1
            self.current_episode_reward += reward
            
        except Exception as e:
            self.logger.warning(f"分层轨迹记录失败: {e}")
            # 在调试模式下打印更多信息
            if self.config.get('debug', False):
                self.logger.warning(f"obs shape: {obs.shape if isinstance(obs, np.ndarray) else type(obs)}")
                self.logger.warning(f"info keys: {list(info.keys()) if isinstance(info, dict) else type(info)}")
                import traceback
                self.logger.warning(f"分层轨迹记录详细错误: {traceback.format_exc()}")

    def _finalize_trajectory_episode(self, episode_reward: float, episode_length: int, info: Dict[str, Any]):
        """完成episode的轨迹记录 - 与基础训练器一致"""
        if not self.trajectory_manager or not self.trajectory_episode_started:
            return
        
        try:
            # 确定终止原因 - 适配分层训练
            termination_reason = "unknown"
            if info.get('collision', False):
                termination_reason = "collision"
            elif info.get('timeout', False):
                termination_reason = "timeout"
            elif info.get('navigation_success', False):
                termination_reason = "navigation_success"
            elif info.get('exploration_completed', False):
                termination_reason = "exploration_completed"
            elif info.get('out_of_bounds', False):
                termination_reason = "out_of_bounds"
            else:
                termination_reason = "completed"
            
            self.trajectory_manager.finalize_episode(
                termination_reason=termination_reason,
                final_exploration_rate=float(getattr(self, 'current_exploration_rate', 1.0)),
                total_reward=episode_reward
            )
            
            self.trajectory_episode_started = False
            
        except Exception as e:
            self.logger.warning(f"完成分层轨迹记录失败: {e}")
            self.trajectory_episode_started = False

def create_hierarchical_trainer(config: Dict[str, Any], 
                               foundation_model_path: Optional[Path] = None) -> HierarchicalTrainer:
    """创建分层训练器的便捷函数"""
    return HierarchicalTrainer(config, foundation_model_path)


if __name__ == "__main__":
    # 简单测试
    test_config = {
        'total_timesteps': 10000,
        'evaluation_frequency': 2000,
        'eval_episodes': 5,
        'enable_trajectory': True,
        'enable_tensorboard': True,
        'enable_visualization': True,
        'enable_rich_display': True
    }
    
    trainer = create_hierarchical_trainer(test_config)
    
    print("开始测试分层训练器...")
    if trainer.setup():
        print("✅ 训练器设置成功")
        # 这里可以添加简短的训练测试
    else:
        print("❌ 训练器设置失败")
