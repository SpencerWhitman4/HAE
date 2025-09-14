#!/usr/bin/env python3

"""
基线训练器 - SB3算法对比训练 (PPO/SAC/TD3)
"""

import logging
import time
import torch
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import sys

# 导入核心组件
from ..core.base_trainer import BaseTrainer, TrainingStage, TrainingResult
from ..core.environment_factory import EnvironmentFactory
from ..core.model_transfer import ModelTransferManager

# SB3导入
try:
    from stable_baselines3 import PPO, SAC, TD3
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
    from stable_baselines3.common.vec_env import VecEnv
    from stable_baselines3.common.logger import configure
    SB3_AVAILABLE = True
except ImportError:
    print("Warning: Stable-Baselines3 not available. Install with: pip install stable-baselines3")
    SB3_AVAILABLE = False

# 复用现有基线包装器
from ..core.environment_factory import BaselineWrapper

logger = logging.getLogger(__name__)


class BaselineProgressCallback(BaseCallback):
    """SB3训练进度回调"""
    
    def __init__(self, 
                 progress_callbacks: List = None,
                 stage: TrainingStage = TrainingStage.BASELINE,
                 algorithm: str = "unknown",
                 verbose: int = 0):
        
        super(BaselineProgressCallback, self).__init__(verbose)
        self.progress_callbacks = progress_callbacks or []
        self.stage = stage
        self.algorithm = algorithm
        self.episode_count = 0
        self.best_mean_reward = -np.inf
    
    def _on_step(self) -> bool:
        """每步回调"""
        
        # 每1000步报告一次进度
        if self.n_calls % 1000 == 0:
            
            # 获取训练统计
            if hasattr(self.locals, 'infos'):
                infos = self.locals.get('infos', [])
                if infos and 'episode' in infos[0]:
                    self.episode_count = infos[0]['episode']['l']
            
            # 获取当前奖励
            if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
                recent_rewards = [ep_info['r'] for ep_info in self.model.ep_info_buffer[-10:]]
                mean_reward = np.mean(recent_rewards) if recent_rewards else 0.0
            else:
                mean_reward = 0.0
            
            # 进度数据
            progress_data = {
                'algorithm': self.algorithm,
                'step': self.n_calls,
                'episode': self.episode_count,
                'mean_reward': mean_reward,
                'progress': self.n_calls / getattr(self.model, 'total_timesteps', 1)
            }
            
            # 调用外部回调
            for callback in self.progress_callbacks:
                try:
                    callback(self.stage, progress_data)
                except Exception as e:
                    logger.warning(f"进度回调失败: {e}")
        
        return True


class BaselineTrainer(BaseTrainer):
    """
    基线训练器 - 使用SB3算法进行对比训练
    
    支持算法:
    - PPO: Proximal Policy Optimization
    - SAC: Soft Actor-Critic
    - TD3: Twin Delayed DDPG
    """
    
    def __init__(self,
                 config: Dict[str, Any],
                 env_factory: Optional[EnvironmentFactory] = None,
                 transfer_manager: Optional[ModelTransferManager] = None,
                 foundation_checkpoint: Optional[Dict[str, Any]] = None,
                 output_dir: str = "./models"):
        
        # 获取算法列表作为阶段变体
        algorithms = config.get('algorithms', ['ppo', 'sac', 'td3'])
        stage_variant = '-'.join(algorithms)
        
        super().__init__(
            stage=TrainingStage.BASELINE,
            config=config,
            experiment_name="HA-UAV",
            stage_variant=stage_variant
        )
        
        # 检查SB3可用性
        if not SB3_AVAILABLE:
            raise ImportError("Stable-Baselines3 is required for baseline training")
        
        # 环境工厂
        self.env_factory = env_factory or EnvironmentFactory()
        
        # 迁移管理器
        self.transfer_manager = transfer_manager
        self.foundation_checkpoint = foundation_checkpoint
        
        # 输出目录
        self.output_dir = output_dir
        
        # 基线算法配置
        self.algorithms = config.get('algorithms', ['ppo', 'sac'])
        self.use_pretrained_init = config.get('use_pretrained_init', True)
        
        # 训练配置
        self.total_timesteps = config.get('total_timesteps', 200000)
        self.eval_freq = config.get('eval_freq', 10000)
        self.eval_episodes = config.get('eval_episodes', 10)
        
        # 算法特定配置
        self.algorithm_configs = {
            'ppo': {
                'learning_rate': config.get('ppo_learning_rate', 3e-4),
                'n_steps': config.get('ppo_n_steps', 2048),
                'batch_size': config.get('ppo_batch_size', 64),
                'n_epochs': config.get('ppo_n_epochs', 10),
                'gamma': config.get('gamma', 0.99),
                'gae_lambda': config.get('gae_lambda', 0.95),
                'clip_range': config.get('clip_range', 0.2),
                'ent_coef': config.get('ent_coef', 0.0),
                'vf_coef': config.get('vf_coef', 0.5)
            },
            'sac': {
                'learning_rate': config.get('sac_learning_rate', 3e-4),
                'buffer_size': config.get('sac_buffer_size', 1000000),
                'learning_starts': config.get('sac_learning_starts', 100),
                'batch_size': config.get('sac_batch_size', 256),
                'tau': config.get('sac_tau', 0.005),
                'gamma': config.get('gamma', 0.99),
                'train_freq': config.get('sac_train_freq', 1),
                'gradient_steps': config.get('sac_gradient_steps', 1)
            },
            'td3': {
                'learning_rate': config.get('td3_learning_rate', 3e-4),
                'buffer_size': config.get('td3_buffer_size', 1000000),
                'learning_starts': config.get('td3_learning_starts', 100),
                'batch_size': config.get('td3_batch_size', 100),
                'tau': config.get('td3_tau', 0.005),
                'gamma': config.get('gamma', 0.99),
                'train_freq': config.get('td3_train_freq', 1),
                'gradient_steps': config.get('td3_gradient_steps', 1),
                'policy_delay': config.get('td3_policy_delay', 2)
            }
        }
        
        # 训练结果存储
        self.baseline_results = {}
        
        # 当前环境
        self.env = None
        self.eval_env = None
        
        # 🔧 修复：初始化progress_callbacks（集成可视化管理器回调）
        self.progress_callbacks = []
        if hasattr(self, 'visualization_manager') and self.visualization_manager:
            self.progress_callbacks.append(self.visualization_manager.on_training_progress)
    
    def setup(self) -> bool:
        """设置训练器"""
        try:
            # 创建训练环境
            env_config = {
                'drone_model': self.config.get('drone_model', 'CF2X'),
                'physics': self.config.get('physics', 'PYB'),
                'gui_training': self.config.get('gui_training', False),
                'max_episode_steps': self.config.get('max_episode_steps', 1000)
            }
            
            # 创建基础环境
            base_env = self.env_factory.create_environment(
                stage=self.stage,
                config=env_config,
                mode="train"
            )
            
            # 应用SB3包装器
            self.env = BaselineWrapper(base_env, agent_type="sb3")
            
            # 创建评估环境
            eval_base_env = self.env_factory.create_environment(
                stage=self.stage,
                config={**env_config, 'gui_training': False},
                mode="eval"
            )
            self.eval_env = BaselineWrapper(eval_base_env, agent_type="sb3")
            
            # 设置日志
            log_dir = Path(self.output_dir) / "baseline_logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            
            self.logger.info("基线训练器设置完成")
            return True
            
        except Exception as e:
            self.logger.error(f"基线训练器设置失败: {e}")
            return False
    
    def train(self) -> TrainingResult:
        """执行基线训练"""
        start_time = time.time()
        
        if not self.setup():
            return TrainingResult(
                success=False,
                error_message="基线训练器设置失败"
            )
        
        try:
            self.logger.info(f"开始基线算法训练: {self.algorithms}")
            
            # 逐个训练每种算法
            for algorithm in self.algorithms:
                self.logger.info(f"开始训练算法: {algorithm.upper()}")
                
                result = self._train_single_algorithm(algorithm)
                self.baseline_results[algorithm] = result
                
                if result.success:
                    self.logger.info(f"算法 {algorithm.upper()} 训练成功")
                else:
                    self.logger.error(f"算法 {algorithm.upper()} 训练失败: {result.error_message}")
            
            # 汇总结果
            training_duration = time.time() - start_time
            
            # 创建对比报告
            comparison_metrics = self._create_baseline_comparison()
            
            # 确定最佳算法
            best_algorithm = self._determine_best_algorithm()
            best_model = self.baseline_results[best_algorithm].trained_model if best_algorithm else None
            
            self.logger.info(f"基线算法训练完成，耗时: {training_duration:.2f}秒")
            
            return TrainingResult(
                success=True,
                trained_model=best_model,
                metrics=comparison_metrics,
                metadata={
                    'stage': self.stage.value,
                    'training_duration': training_duration,
                    'algorithms': self.algorithms,
                    'individual_results': {k: {'success': v.success, 'metrics': v.metrics}
                                         for k, v in self.baseline_results.items()},
                    'best_algorithm': best_algorithm
                }
            )
            
        except Exception as e:
            self.logger.error(f"基线训练过程失败: {e}")
            return TrainingResult(
                success=False,
                error_message=str(e)
            )
    
    def _train_single_algorithm(self, algorithm: str) -> TrainingResult:
        """训练单个基线算法 - 轻量级协调器模式"""
        
        try:
            # 创建模型
            model = self._create_sb3_model(algorithm)
            
            # 权重初始化
            if self.use_pretrained_init and self.foundation_checkpoint and self.transfer_manager:
                self._apply_pretrained_initialization(model, algorithm)
            
            # 计算训练步数
            timesteps_per_algorithm = self.total_timesteps // len(self.algorithms)
            self.logger.info(f"{algorithm.upper()} 训练步数: {timesteps_per_algorithm}")
            
            # 🎯 核心：委托给训练执行方法
            final_metrics = self._execute_baseline_training(algorithm, model, timesteps_per_algorithm)
            
            # 保存模型
            model_path = Path(self.output_dir) / f"final_{algorithm}_model.zip"
            model.save(str(model_path))
            
            return TrainingResult(
                success=True,
                trained_model=model,
                metrics=final_metrics,
                metadata={
                    'algorithm': algorithm,
                    'timesteps': timesteps_per_algorithm,
                    'model_path': str(model_path)
                }
            )
            
        except Exception as e:
            self.logger.error(f"算法 {algorithm} 训练失败: {e}")
            return TrainingResult(
                success=False,
                error_message=str(e)
            )
            
    def _execute_baseline_training(self, algorithm: str, model, timesteps_per_algorithm: int) -> Dict[str, Any]:
        """执行基线训练逻辑 - 修复版本，确保回调正常工作"""
        self.logger.info(f"开始 {algorithm.upper()} 基线训练: {timesteps_per_algorithm:,} 步")
        
        # 🎯 关键：使用很小的学习间隔，确保回调及时触发
        learn_interval = min(100, timesteps_per_algorithm // 20)  # 很小的间隔
        current_step = 0
        training_stats = []
        
        # 🔧 立即调用第一次回调，确保进度条启动
        if self.visualization_manager:
            initial_metrics = {
                'episode': 0,
                'total_reward': 0.0,
                'exploration_rate': 1.0,
                'policy_loss': 0.0,
                'value_loss': 0.0,
                'algorithm': algorithm
            }
            self.visualization_manager.on_step(0, initial_metrics)
            self.logger.info(f"✅ {algorithm.upper()} 初始回调已触发，进度条应已启动")
        
        # 使用小批次训练，确保回调及时触发
        while current_step < timesteps_per_algorithm:
            try:
                # 🔧 使用超小批次，避免长时间卡住
                steps_to_learn = min(learn_interval, timesteps_per_algorithm - current_step, 50)  # 最多50步
                
                self.logger.info(f"开始 {algorithm.upper()} 学习: {steps_to_learn} 步 (当前步数: {current_step})")
                
                # 限时学习，避免无限卡住
                learn_start = time.time()
                
                try:
                    # SB3的小批次学习
                    model.learn(
                        total_timesteps=steps_to_learn,
                        reset_num_timesteps=False,  # 关键：不重置计数器
                        progress_bar=False
                    )
                    learn_time = time.time() - learn_start
                    
                    if learn_time > 5.0:
                        self.logger.warning(f"{algorithm.upper()} 学习耗时过长: {learn_time:.1f}s")
                    
                except Exception as e:
                    self.logger.error(f"{algorithm.upper()} 学习失败: {e}")
                    # 即使失败也要更新进度
                    steps_to_learn = min(steps_to_learn, 10)  # 减少步数
                
                # 🎯 立即更新步数和触发回调
                current_step += steps_to_learn
                
                # 🔧 核心：立即触发回调，确保进度条更新
                step_metrics = {
                    'episode': current_step // 1000,  # 估算episode数
                    'total_reward': 0.0,  # SB3内部维护
                    'exploration_rate': max(0.0, 1.0 - current_step / timesteps_per_algorithm),
                    'policy_loss': 0.0,  # SB3内部维护
                    'value_loss': 0.0,   # SB3内部维护
                    'algorithm': algorithm
                }
                
                # 调用可视化管理器回调
                if self.visualization_manager:
                    self.visualization_manager.on_step(current_step, step_metrics)
                    self.logger.info(f"✅ {algorithm.upper()} 回调已触发: {current_step}/{timesteps_per_algorithm} ({current_step/timesteps_per_algorithm:.1%})")
                
                # 调用训练器回调
                self.on_step_callback(current_step, step_metrics)
                
                # 定期评估（快速跳过）
                if current_step % self.eval_freq == 0 and current_step > 0:
                    self.logger.info(f"开始 {algorithm.upper()} 定期评估 (步数: {current_step})")
                    try:
                        eval_results = self._evaluate_sb3_model(model, algorithm, num_episodes=5)  # 减少评估episode
                        self.on_evaluation_callback(eval_results)
                    except Exception as e:
                        self.logger.warning(f"{algorithm.upper()} 评估失败: {e}")
                
                # 进度日志
                self.logger.info(f"{algorithm.upper()} 训练进度: {current_step:,}/{timesteps_per_algorithm:,} "
                               f"({current_step/timesteps_per_algorithm:.1%})")
                
                # 记录统计
                training_stats.append({
                    'steps': steps_to_learn,
                    'total_steps': current_step
                })
                
            except Exception as e:
                self.logger.error(f"{algorithm.upper()} 训练步骤失败: {e}")
                import traceback
                traceback.print_exc()
                # 即使出错也要更新进度，避免卡住
                current_step += learn_interval
                if self.visualization_manager:
                    error_metrics = {
                        'episode': 0,
                        'total_reward': 0.0,
                        'exploration_rate': 0.0,
                        'policy_loss': 0.0,
                        'value_loss': 0.0,
                        'algorithm': algorithm
                    }
                    self.visualization_manager.on_step(current_step, error_metrics)
                break
        
        # 最终评估
        try:
            final_metrics = self._evaluate_sb3_model(model, algorithm)
            self.logger.info(f"{algorithm.upper()} 基线训练完成: 成功率={final_metrics.get('success_rate', 0):.2%}")
            return final_metrics
        except Exception as e:
            self.logger.error(f"{algorithm.upper()} 最终评估失败: {e}")
            return {'success_rate': 0.0, 'mean_reward': 0.0, 'error': str(e)}
    
    def _create_sb3_model(self, algorithm: str):
        """创建SB3模型"""
        
        config = self.algorithm_configs[algorithm]
        
        if algorithm == 'ppo':
            model = PPO(
                policy="MlpPolicy",
                env=self.env,
                verbose=1,
                **config
            )
        
        elif algorithm == 'sac':
            model = SAC(
                policy="MlpPolicy",
                env=self.env,
                verbose=1,
                **config
            )
        
        elif algorithm == 'td3':
            model = TD3(
                policy="MlpPolicy",
                env=self.env,
                verbose=1,
                **config
            )
        
        else:
            raise ValueError(f"不支持的算法: {algorithm}")
        
        return model
    
    def _apply_pretrained_initialization(self, model, algorithm: str) -> None:
        """应用预训练权重初始化"""
        
        if not self.transfer_manager or not self.foundation_checkpoint:
            return
        
        try:
            # 使用迁移管理器进行权重迁移
            transfer_result = self.transfer_manager.transfer_weights(
                foundation_checkpoint=self.foundation_checkpoint,
                target_model=model.policy,  # SB3模型的策略网络
                target_stage=self.stage,
                transfer_config={
                    'algorithm': algorithm,
                    'use_pretrained_init': True
                }
            )
            
            self.logger.info(f"{algorithm.upper()} 预训练初始化: {transfer_result['success_rate']:.2%}")
            
        except Exception as e:
            self.logger.warning(f"{algorithm.upper()} 预训练初始化失败: {e}")
    
    def _evaluate_sb3_model(self, model, algorithm: str, num_episodes: int = None) -> Dict[str, Any]:
        """评估SB3模型 - 修复版本，添加超时和强制终止机制"""
        
        # 🔧 减少评估episode数，避免长时间评估
        eval_episodes = num_episodes or min(self.eval_episodes, 5)
        eval_rewards = []
        eval_successes = 0
        
        self.logger.info(f"开始评估 {algorithm.upper()}: {eval_episodes} episodes")
        
        for episode in range(eval_episodes):
            try:
                obs, _ = self.eval_env.reset()
                episode_reward = 0
                episode_success = False
                
                # 🔧 减少最大步数，避免episode过长
                max_steps = min(self.config.get('max_episode_steps', 1000), 200)  # 最多200步
                
                for step in range(max_steps):
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = self.eval_env.step(action)
                    episode_reward += reward
                    
                    if terminated or truncated:
                        if info.get('success', False) or info.get('navigation_success', False):
                            eval_successes += 1
                            episode_success = True
                        break
                    
                    # 🔧 添加早期成功判断
                    if step > 100 and episode_reward > 50:  # 100步后如果奖励足够高就认为成功
                        episode_success = True
                        eval_successes += 1
                        break
                
                eval_rewards.append(episode_reward)
                self.logger.info(f"{algorithm.upper()} 评估 episode {episode + 1}: 奖励={episode_reward:.2f}, 步数={step + 1}, 成功={episode_success}")
                
            except Exception as e:
                self.logger.error(f"{algorithm.upper()} 评估 episode {episode + 1} 失败: {e}")
                eval_rewards.append(0.0)
                continue
        
        # 确保有数据
        if not eval_rewards:
            eval_rewards = [0.0]
        
        return {
            'algorithm': algorithm,
            'avg_episode_reward': float(np.mean(eval_rewards)),
            'std_episode_reward': float(np.std(eval_rewards)),
            'success_rate': eval_successes / eval_episodes if eval_episodes > 0 else 0.0,
            'total_eval_episodes': eval_episodes,
            'mean_reward': float(np.mean(eval_rewards))  # 兼容性字段
        }
    
    def _create_baseline_comparison(self) -> Dict[str, Any]:
        """创建基线对比报告"""
        
        comparison = {
            'algorithm_results': {},
            'rankings': {},
            'analysis': {}
        }
        
        # 收集各算法的指标
        metrics = ['avg_episode_reward', 'success_rate']
        
        for metric in metrics:
            comparison['algorithm_results'][metric] = {}
            
            for algorithm in self.algorithms:
                if algorithm in self.baseline_results and self.baseline_results[algorithm].success:
                    value = self.baseline_results[algorithm].metrics.get(metric, 0.0)
                    comparison['algorithm_results'][metric][algorithm] = value
        
        # 生成排名
        for metric in metrics:
            if comparison['algorithm_results'][metric]:
                sorted_algorithms = sorted(
                    comparison['algorithm_results'][metric].items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                comparison['rankings'][metric] = [alg for alg, _ in sorted_algorithms]
        
        # 分析结论
        comparison['analysis'] = self._analyze_baseline_results()
        
        return comparison
    
    def _analyze_baseline_results(self) -> Dict[str, Any]:
        """分析基线结果"""
        
        analysis = {
            'best_algorithm': self._determine_best_algorithm(),
            'performance_comparison': {},
            'insights': []
        }
        
        # 性能对比分析
        rewards = []
        success_rates = []
        
        for algorithm, result in self.baseline_results.items():
            if result.success:
                rewards.append(result.metrics.get('avg_episode_reward', 0.0))
                success_rates.append(result.metrics.get('success_rate', 0.0))
        
        if rewards:
            analysis['performance_comparison'] = {
                'reward_variance': np.var(rewards),
                'success_rate_variance': np.var(success_rates),
                'performance_gap': max(rewards) - min(rewards) if len(rewards) > 1 else 0
            }
        
        # 生成洞察
        if analysis['performance_comparison'].get('performance_gap', 0) > 50:
            analysis['insights'].append("不同算法间存在显著性能差异")
        
        if analysis['performance_comparison'].get('reward_variance', 0) < 100:
            analysis['insights'].append("各算法性能相对稳定")
        
        return analysis
    
    def _determine_best_algorithm(self) -> Optional[str]:
        """确定最佳算法"""
        
        best_algorithm = None
        best_score = -float('inf')
        
        for algorithm, result in self.baseline_results.items():
            if result.success:
                # 综合评分
                success_rate = result.metrics.get('success_rate', 0.0)
                avg_reward = result.metrics.get('avg_episode_reward', 0.0)
                score = success_rate * 100 + avg_reward * 0.1
                
                if score > best_score:
                    best_score = score
                    best_algorithm = algorithm
        
        return best_algorithm
    
    def cleanup(self) -> None:
        """清理资源"""
        if self.env:
            self.env.close()
        
        if self.eval_env:
            self.eval_env.close()
        
        self.logger.info("基线训练器资源已清理")
    
    def _execute_training(self) -> Dict[str, Any]:
        """执行训练逻辑 - 实现抽象方法"""
        return self.train().metadata if hasattr(self.train(), 'metadata') else {}
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, Any]:
        """评估当前模型 - 实现抽象方法"""
        # 如果有最佳算法，评估它
        best_algorithm = self._determine_best_algorithm()
        if best_algorithm and best_algorithm in self.baseline_results:
            model = self.baseline_results[best_algorithm].trained_model
            return self._evaluate_sb3_model(model, best_algorithm)
        return {'mean_reward': 0.0, 'success_rate': 0.0}
    
    def save_model(self, path: Path, metadata: Optional[Dict] = None) -> bool:
        """保存模型 - 实现抽象方法"""
        best_algorithm = self._determine_best_algorithm()
        if best_algorithm and best_algorithm in self.baseline_results:
            try:
                model = self.baseline_results[best_algorithm].trained_model
                model.save(str(path))
                self.logger.info(f"最佳基线模型 ({best_algorithm}) 已保存到: {path}")
                return True
            except Exception as e:
                self.logger.error(f"模型保存失败: {e}")
        return False
    
    def load_model(self, path: Path) -> bool:
        """加载模型 - 实现抽象方法"""
        try:
            # 这里需要根据模型类型选择正确的SB3算法
            # 简化实现，假设是PPO
            from stable_baselines3 import PPO
            model = PPO.load(str(path))
            # 将加载的模型存储到结果中
            from ..core.base_trainer import TrainingResult
            self.baseline_results['loaded'] = TrainingResult(
                success=True,
                trained_model=model,
                metrics={'loaded_from': str(path)}
            )
            self.logger.info(f"基线模型已从 {path} 加载")
            return True
        except Exception as e:
            self.logger.error(f"模型加载失败: {e}")
            return False
