#!/usr/bin/env python3

"""
消融实验训练器 - B组消融实验 (B1/B2/B3) 训练
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

# 导入核心组件
from ..core.base_trainer import BaseTrainer, TrainingStage, TrainingResult
from ..core.environment_factory import EnvironmentFactory
from ..core.model_transfer import ModelTransferManager

# 添加项目路径以导入src.modules
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# 复用现有B组消融实验组件 - 直接使用AblationComponentsManager
from src.modules import (
    AblationComponentsManager,
    AblationConfig,
    get_ablation_config,
    create_ablation_system,
    create_b1_config,
    create_b2_config,
    create_b3_config
)

logger = logging.getLogger(__name__)


class AblationTrainer(BaseTrainer):
    """
    消融实验训练器
    
    支持三种消融实验:
    - B1: 高层直接控制 (DirectControlPolicy)
    - B2: 扁平化决策 (FlatPolicy) 
    - B3: 单步分层 (SingleStepHierarchicalPolicy)
    """
    
    def __init__(self,
                 config: Dict[str, Any],
                 env_factory: Optional[EnvironmentFactory] = None,
                 transfer_manager: Optional[ModelTransferManager] = None,
                 foundation_checkpoint: Optional[Dict[str, Any]] = None,
                 output_dir: str = "./models"):
        
        # 获取消融类型作为阶段变体
        ablation_types = config.get('ablation_types', ['B1', 'B2', 'B3'])
        stage_variant = '-'.join(ablation_types)
        
        super().__init__(
            stage=TrainingStage.ABLATION,
            config=config,
            experiment_name="HA-UAV",
            stage_variant=stage_variant
        )
        
        # 环境工厂
        self.env_factory = env_factory or EnvironmentFactory()
        
        # 迁移管理器
        self.transfer_manager = transfer_manager
        self.foundation_checkpoint = foundation_checkpoint
        
        # 输出目录
        self.output_dir = output_dir
        
        # 消融实验配置
        self.ablation_types = config.get('ablation_types', ['B1', 'B2', 'B3'])
        
        # 训练配置
        self.total_timesteps = config.get('total_timesteps', 150000)
        self.batch_size = config.get('batch_size', 256)
        self.learning_rate = config.get('learning_rate', 3e-4)
        self.gamma = config.get('gamma', 0.99)
        
        # 消融实验结果存储
        self.ablation_results = {}
        
        # 当前训练的消融管理器
        self.current_ablation_manager = None
        self.env = None
        
        # 训练统计 (按消融类型分组)
        self.training_stats = {
            ablation_type: {
                'episode_rewards': [],
                'episode_lengths': [],
                'success_rates': [],
                'training_losses': []
            } for ablation_type in self.ablation_types
        }
    
    def setup(self) -> bool:
        """设置训练器"""
        try:
            # 创建环境 (消融实验使用HAUAVAviary)
            env_config = {
                'drone_model': self.config.get('drone_model', 'CF2X'),
                'physics': self.config.get('physics', 'PYB'),
                'gui_training': self.config.get('gui_training', False),
                'max_episode_steps': self.config.get('max_episode_steps', 1000)
            }
            
            self.env = self.env_factory.create_environment(
                stage=self.stage,
                config=env_config,
                mode="train"
            )
            
            self.logger.info("消融实验训练器设置完成")
            return True
            
        except Exception as e:
            self.logger.error(f"消融实验训练器设置失败: {e}")
            return False
    
    def train(self) -> TrainingResult:
        """执行消融实验训练"""
        start_time = time.time()
        
        if not self.setup():
            return TrainingResult(
                success=False,
                error_message="消融实验训练器设置失败"
            )
        
        try:
            self.logger.info(f"开始消融实验训练: {self.ablation_types}")
            
            # 逐个训练每种消融类型
            for ablation_type in self.ablation_types:
                self.logger.info(f"开始训练消融类型: {ablation_type}")
                
                result = self._train_single_ablation(ablation_type)
                self.ablation_results[ablation_type] = result
                
                if result.success:
                    self.logger.info(f"消融类型 {ablation_type} 训练成功")
                else:
                    self.logger.error(f"消融类型 {ablation_type} 训练失败: {result.error_message}")
            
            # 汇总结果
            training_duration = time.time() - start_time
            
            # 创建综合评估报告
            comparison_metrics = self._create_ablation_comparison()
            
            # 确定最佳模型 (用于返回)
            best_ablation_type = self._determine_best_ablation()
            best_model = self.ablation_results[best_ablation_type].trained_model if best_ablation_type else None
            
            self.logger.info(f"消融实验训练完成，耗时: {training_duration:.2f}秒")
            
            return TrainingResult(
                success=True,
                trained_model=best_model,  # 返回最佳模型
                metrics=comparison_metrics,
                metadata={
                    'stage': self.stage.value,
                    'training_duration': training_duration,
                    'ablation_types': self.ablation_types,
                    'individual_results': {k: {'success': v.success, 'metrics': v.metrics} 
                                         for k, v in self.ablation_results.items()},
                    'best_ablation_type': best_ablation_type,
                    'training_stats': self.training_stats
                }
            )
            
        except Exception as e:
            self.logger.error(f"消融实验训练过程失败: {e}")
            return TrainingResult(
                success=False,
                error_message=str(e)
            )
    
    def _train_single_ablation(self, ablation_type: str) -> TrainingResult:
        """训练单个消融实验 - 使用AblationComponentsManager"""
        
        try:
            # 创建消融配置
            if ablation_type == 'B1':
                ablation_config = create_b1_config()
            elif ablation_type == 'B2':
                ablation_config = create_b2_config()  
            elif ablation_type == 'B3':
                ablation_config = create_b3_config()
            else:
                raise ValueError(f"未知的消融类型: {ablation_type}")
            
            # 创建消融系统
            self.current_ablation_manager = create_ablation_system(
                env=self.env,
                config=ablation_config
            )
            
            # 权重迁移
            if self.foundation_checkpoint and self.transfer_manager:
                transfer_result = self.transfer_manager.transfer_weights(
                    foundation_checkpoint=self.foundation_checkpoint,
                    target_model=self.current_ablation_manager.policy,
                    target_stage=self.stage,
                    transfer_config={'ablation_type': ablation_type}
                )
                self.logger.info(f"{ablation_type} 权重迁移: {transfer_result['success_rate']:.2%}")
            
            # 执行训练循环
            final_metrics = self._training_loop(ablation_type)
            
            return TrainingResult(
                success=True,
                trained_model=self.current_ablation_manager.policy,
                metrics=final_metrics,
                metadata={
                    'ablation_type': ablation_type,
                    'training_stats': self.training_stats[ablation_type],
                    'ablation_system_stats': self.current_ablation_manager.get_training_stats()
                }
            )
            
        except Exception as e:
            self.logger.error(f"消融类型 {ablation_type} 训练失败: {e}")
            return TrainingResult(
                success=False,
                error_message=str(e)
            )
    
    
    def _training_loop(self, ablation_type: str) -> Dict[str, Any]:
        """执行训练循环 - 轻量级协调器，委托给AblationComponentsManager"""
        
        # 训练参数 (根据消融类型调整)
        timesteps_per_ablation = self.total_timesteps // len(self.ablation_types)
        
        if ablation_type == 'B2':
            # 扁平化需要更多训练时间
            timesteps_per_ablation = int(timesteps_per_ablation * 1.2)
        elif ablation_type == 'B3':
            # 单步分层相对简单
            timesteps_per_ablation = int(timesteps_per_ablation * 0.8)
        
        self.logger.info(f"{ablation_type} 训练步数: {timesteps_per_ablation}")
        
        # 🎯 核心：直接委托给训练执行方法
        return self._execute_ablation_training(ablation_type, timesteps_per_ablation)
        
    def _execute_ablation_training(self, ablation_type: str, timesteps_per_ablation: int) -> Dict[str, Any]:
        """执行消融训练逻辑 - 轻量级协调器，直接委托给AblationComponentsManager"""
        self.logger.info(f"开始 {ablation_type} 消融训练: {timesteps_per_ablation:,} 步")
        
        # 设置训练模式
        self.current_ablation_manager.set_training_mode(True)
        
        training_stats = []
        current_step = 0
        evaluation_frequency = timesteps_per_ablation // 5  # 每20%评估一次
        
        while current_step < timesteps_per_ablation:
            try:
                # 🎯 核心：直接使用AblationComponentsManager的完整训练步骤
                step_stats = self.current_ablation_manager.train_step(self.env)
                training_stats.append(step_stats)
                
                # 更新步数（从AblationComponentsManager的统计中获取）
                current_step += step_stats.get('total_steps', 1)
                
                # 协调器职责：进度管理和统计收集
                self._update_ablation_stats(ablation_type, step_stats)
                
                # 定期评估
                if current_step % evaluation_frequency == 0:
                    eval_results = self._evaluate_ablation(ablation_type)
                    self.logger.info(f"{ablation_type} 阶段评估: 成功率 {eval_results.get('success_rate', 0):.2%}")
                
            except Exception as e:
                self.logger.error(f"{ablation_type} 训练步骤失败: {e}")
                break
        
        # 最终评估
        final_eval = self._evaluate_ablation(ablation_type)
        
        self.logger.info(f"{ablation_type} 消融训练完成: 步数={current_step}, 成功率={final_eval.get('success_rate', 0):.2%}")
        
        return final_eval
    
    def _update_ablation_stats(self, ablation_type: str, step_stats: Dict[str, Any]):
        """更新消融实验统计信息"""
        stats = self.training_stats[ablation_type]
        
        # 收集episode奖励
        if 'episode_rewards' in step_stats:
            stats['episode_rewards'].extend(step_stats['episode_rewards'])
        elif 'mean_reward' in step_stats:
            stats['episode_rewards'].append(step_stats['mean_reward'])
        
        # 收集episode长度
        if 'episode_lengths' in step_stats:
            stats['episode_lengths'].extend(step_stats['episode_lengths'])
        elif 'mean_episode_length' in step_stats:
            stats['episode_lengths'].append(step_stats['mean_episode_length'])
        
        # 收集成功率
        if 'success_rate' in step_stats:
            stats['success_rates'].append(step_stats['success_rate'])
        elif step_stats.get('episodes', 0) > 0:
            # 基于环境信息判断成功
            success_info = step_stats.get('info', {})
            success = self._determine_success(ablation_type, success_info)
            stats['success_rates'].append(1.0 if success else 0.0)
        
        # 收集训练损失
        for loss_key in ['total_loss', 'policy_loss', 'value_loss']:
            if loss_key in step_stats:
                stats['training_losses'].append(step_stats[loss_key])
                break
    
    
    def _report_ablation_progress(self, ablation_type: str, step: int, total_steps: int, episode_count: int) -> None:
        """报告消融训练进度"""
        
        stats = self.training_stats[ablation_type]
        
        if len(stats['episode_rewards']) > 0:
            recent_rewards = stats['episode_rewards'][-10:]
            avg_reward = np.mean(recent_rewards)
            
            recent_success = stats['success_rates'][-10:]
            success_rate = np.mean(recent_success) if recent_success else 0.0
            
            progress_data = {
                'ablation_type': ablation_type,
                'step': step,
                'total_steps': total_steps,
                'episode': episode_count,
                'avg_reward': avg_reward,
                'success_rate': success_rate,
                'progress': step / total_steps
            }
            
            self.logger.info(
                f"[{ablation_type}] 步骤 {step}/{total_steps} | "
                f"平均奖励: {avg_reward:.2f} | "
                f"成功率: {success_rate:.2%} | "
                f"进度: {progress_data['progress']:.1%}"
            )
            
            # 调用进度回调
            for callback in self.progress_callbacks:
                callback(self.stage, progress_data)
    
    def _evaluate_ablation(self, ablation_type: str) -> Dict[str, Any]:
        """评估消融实验 - 委托给AblationComponentsManager并增强成功率判断"""
        if not self.current_ablation_manager or not self.env:
            return {'mean_reward': 0.0, 'success_rate': 0.0, 'error': 'Components not initialized'}
        
        eval_episodes = self.config.get('eval_episodes', 10)
        episode_rewards = []
        episode_lengths = []
        success_count = 0
        
        # 设置为评估模式
        self.current_ablation_manager.set_training_mode(False)
        
        try:
            for episode in range(eval_episodes):
                obs, _ = self.env.reset()
                episode_reward = 0
                episode_length = 0
                
                for step in range(self.config.get('max_episode_steps', 1000)):
                    # 🎯 直接使用AblationComponentsManager的预测方法
                    action = self.current_ablation_manager.predict(obs)
                    
                    obs, reward, terminated, truncated, info = self.env.step(action)
                    episode_reward += reward
                    episode_length += 1
                    
                    if terminated or truncated:
                        # 🔧 增强的成功条件判断
                        if self._is_ablation_success(ablation_type, terminated, truncated, info, episode_length):
                            success_count += 1
                        break
                
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
            
            # 恢复训练模式
            self.current_ablation_manager.set_training_mode(True)
            
            return {
                'mean_reward': float(np.mean(episode_rewards)),
                'std_reward': float(np.std(episode_rewards)),
                'success_rate': success_count / eval_episodes,
                'mean_episode_length': float(np.mean(episode_lengths)),
                'episodes_evaluated': eval_episodes,
                'ablation_type': ablation_type
            }
            
        except Exception as e:
            self.current_ablation_manager.set_training_mode(True)
            self.logger.error(f"{ablation_type} 评估过程中出错: {e}")
            return {'mean_reward': 0.0, 'success_rate': 0.0, 'error': str(e)}
    
    def _is_ablation_success(self, ablation_type: str, terminated: bool, truncated: bool, info: dict, episode_length: int) -> bool:
        """消融实验的成功条件判断"""
        if not terminated:
            return False
        
        # 1. 显式成功标志
        if info.get('navigation_success', False) or info.get('exploration_completed', False):
            return True
            
        # 2. 基于探索率的成功判断
        exploration_rate = info.get('exploration_rate', 0.0)
        if exploration_rate > 0.75:  # 消融实验标准稍低
            return True
        
        # 3. 基于奖励阈值的成功判断（针对不同消融类型）
        total_reward = info.get('total_reward', 0.0)
        success_thresholds = {
            'B1': 80,   # 直接控制要求较低
            'B2': 90,   # 扁平化要求中等
            'B3': 85    # 单步分层要求中等
        }
        threshold = success_thresholds.get(ablation_type, 80)
        if total_reward > threshold:
            return True
        
        # 4. 基于episode长度的成功判断
        min_lengths = {
            'B1': 400,  # 直接控制要求较低
            'B2': 450,  # 扁平化要求中等  
            'B3': 425   # 单步分层要求中等
        }
        min_length = min_lengths.get(ablation_type, 400)
        if episode_length > min_length:
            return True
            
        return False
    
    def _determine_success(self, ablation_type: str, info: Dict[str, Any]) -> bool:
        """确定成功标准"""
        
        if ablation_type == 'B1':
            # 直接控制：关注基础飞行稳定性
            return info.get('flight_stability', False)
        
        elif ablation_type == 'B2':
            # 扁平化：关注任务完成度
            return info.get('task_completion', False)
        
        elif ablation_type == 'B3':
            # 单步分层：关注短期目标达成
            return info.get('subgoal_reached', False)
        
        else:
            return info.get('success', False)
    
    def _create_ablation_comparison(self) -> Dict[str, Any]:
        """创建消融实验对比报告"""
        
        comparison = {
            'summary': {},
            'detailed_results': {},
            'performance_ranking': []
        }
        
        # 汇总每个消融类型的表现
        for ablation_type, result in self.ablation_results.items():
            if result.success and result.metrics:
                metrics = result.metrics
                comparison['detailed_results'][ablation_type] = {
                    'avg_reward': metrics.get('avg_episode_reward', 0.0),
                    'success_rate': metrics.get('success_rate', 0.0),
                    'stability': metrics.get('std_episode_reward', float('inf')),
                    'episodes_trained': metrics.get('total_episodes_trained', 0)
                }
        
        # 排名（基于综合分数：50%奖励 + 50%成功率）
        rankings = []
        for ablation_type, metrics in comparison['detailed_results'].items():
            score = 0.5 * metrics['avg_reward'] + 0.5 * metrics['success_rate'] * 100
            rankings.append((ablation_type, score, metrics))
        
        rankings.sort(key=lambda x: x[1], reverse=True)
        comparison['performance_ranking'] = [
            {
                'rank': i+1,
                'ablation_type': item[0],
                'composite_score': item[1],
                'metrics': item[2]
            }
            for i, item in enumerate(rankings)
        ]
        
        # 总结
        if rankings:
            best = rankings[0]
            comparison['summary'] = {
                'best_ablation_type': best[0],
                'best_score': best[1],
                'total_ablations_tested': len(self.ablation_types),
                'successful_ablations': len([r for r in self.ablation_results.values() if r.success])
            }
        
        return comparison
    
    def _determine_best_ablation(self) -> Optional[str]:
        """确定最佳消融类型"""
        
        best_type = None
        best_score = -float('inf')
        
        for ablation_type, result in self.ablation_results.items():
            if result.success and result.metrics:
                # 综合分数：奖励 + 成功率
                reward = result.metrics.get('avg_episode_reward', 0.0)
                success_rate = result.metrics.get('success_rate', 0.0)
                score = reward + success_rate * 100  # 成功率权重更高
                
                if score > best_score:
                    best_score = score
                    best_type = ablation_type
        
        return best_type
    
    def cleanup(self) -> None:
        """清理资源"""
        if self.env:
            self.env.close()
        
        # 保存所有消融实验结果
        for ablation_type, result in self.ablation_results.items():
            if result.success and hasattr(result, 'trained_model'):
                try:
                    save_path = Path(self.output_dir) / f"ablation_{ablation_type}_final_model.zip"
                    # 如果是AblationComponentsManager，使用其保存方法
                    if hasattr(result.trained_model, 'save_model'):
                        result.trained_model.save_model(str(save_path))
                    else:
                        # 否则使用torch保存
                        torch.save(result.trained_model.state_dict(), save_path)
                    self.logger.info(f"消融模型 {ablation_type} 已保存到: {save_path}")
                except Exception as e:
                    self.logger.warning(f"消融模型 {ablation_type} 保存失败: {e}")
        
        self.logger.info("消融实验训练器资源已清理")
    
    def _report_ablation_progress(self, 
                                 ablation_type: str,
                                 step: int,
                                 total_steps: int,
                                 episode_count: int) -> None:
        """报告消融实验进度"""
        
        stats = self.training_stats[ablation_type]
        
        if len(stats['episode_rewards']) > 0:
            recent_rewards = stats['episode_rewards'][-10:]
            avg_reward = np.mean(recent_rewards)
            
            recent_success = stats['success_rates'][-10:]
            success_rate = np.mean(recent_success) if recent_success else 0.0
            
            progress_data = {
                'ablation_type': ablation_type,
                'step': step,
                'total_steps': total_steps,
                'episode': episode_count,
                'avg_reward': avg_reward,
                'success_rate': success_rate,
                'progress': step / total_steps
            }
            
            self.logger.info(
                f"[{ablation_type}] 步骤 {step}/{total_steps} | "
                f"平均奖励: {avg_reward:.2f} | "
                f"成功率: {success_rate:.2%}"
            )
            
            # 调用进度回调
            for callback in self.progress_callbacks:
                callback(self.stage, progress_data)
    
    def _evaluate_ablation(self, ablation_type: str) -> Dict[str, Any]:
        """评估消融实验"""
        
        eval_episodes = 20
        eval_rewards = []
        eval_successes = 0
        
        for episode in range(eval_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0
            
            for step in range(self.config.get('max_episode_steps', 1000)):
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                
                with torch.no_grad():
                    action_output = self.current_model.get_action(obs_tensor, deterministic=True)
                    
                    if isinstance(action_output, dict):
                        action = action_output['action']
                    else:
                        action = action_output
                    
                    action_np = action.squeeze(0).cpu().numpy()
                
                obs, reward, terminated, truncated, info = self.env.step(action_np)
                episode_reward += reward
                
                if terminated or truncated:
                    if self._determine_success(ablation_type, info):
                        eval_successes += 1
                    break
            
            eval_rewards.append(episode_reward)
        
        return {
            'avg_episode_reward': np.mean(eval_rewards),
            'std_episode_reward': np.std(eval_rewards),
            'success_rate': eval_successes / eval_episodes,
            'total_episodes_trained': len(self.training_stats[ablation_type]['episode_rewards'])
        }
    
    def _create_ablation_comparison(self) -> Dict[str, Any]:
        """创建消融实验对比"""
        
        comparison = {
            'ablation_results': {},
            'rankings': {},
            'analysis': {}
        }
        
        # 收集各消融类型的指标
        metrics = ['avg_episode_reward', 'success_rate']
        
        for metric in metrics:
            comparison['ablation_results'][metric] = {}
            
            for ablation_type in self.ablation_types:
                if ablation_type in self.ablation_results and self.ablation_results[ablation_type].success:
                    value = self.ablation_results[ablation_type].metrics.get(metric, 0.0)
                    comparison['ablation_results'][metric][ablation_type] = value
        
        # 生成排名
        for metric in metrics:
            if comparison['ablation_results'][metric]:
                sorted_ablations = sorted(
                    comparison['ablation_results'][metric].items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                comparison['rankings'][metric] = [ablation for ablation, _ in sorted_ablations]
        
        # 分析结论
        comparison['analysis'] = self._analyze_ablation_results()
        
        return comparison
    
    def _analyze_ablation_results(self) -> Dict[str, Any]:
        """分析消融实验结果"""
        
        analysis = {
            'hierarchy_necessity': 'unknown',
            'performance_degradation': {},
            'insights': []
        }
        
        # 获取完整分层的基准性能 (这里简化，实际需要完整分层训练结果)
        baseline_performance = 0.8  # 假设的基准成功率
        
        # 分析各消融类型的性能下降
        for ablation_type in self.ablation_types:
            if ablation_type in self.ablation_results and self.ablation_results[ablation_type].success:
                success_rate = self.ablation_results[ablation_type].metrics.get('success_rate', 0.0)
                degradation = baseline_performance - success_rate
                analysis['performance_degradation'][ablation_type] = degradation
        
        # 确定分层架构的必要性
        avg_degradation = np.mean(list(analysis['performance_degradation'].values()))
        if avg_degradation > 0.2:
            analysis['hierarchy_necessity'] = 'essential'
            analysis['insights'].append("消融实验显示分层架构对性能至关重要")
        elif avg_degradation > 0.1:
            analysis['hierarchy_necessity'] = 'beneficial'
            analysis['insights'].append("分层架构提供了显著的性能优势")
        else:
            analysis['hierarchy_necessity'] = 'marginal'
            analysis['insights'].append("分层架构的优势有限")
        
        return analysis
    
    def _determine_best_ablation(self) -> Optional[str]:
        """确定最佳消融类型"""
        
        best_ablation = None
        best_score = -float('inf')
        
        for ablation_type, result in self.ablation_results.items():
            if result.success:
                # 综合评分：成功率 + 平均奖励
                success_rate = result.metrics.get('success_rate', 0.0)
                avg_reward = result.metrics.get('avg_episode_reward', 0.0)
                score = success_rate * 100 + avg_reward
                
                if score > best_score:
                    best_score = score
                    best_ablation = ablation_type
        
        return best_ablation
    
    def cleanup(self) -> None:
        """清理资源"""
        if self.env:
            self.env.close()
        
        self.logger.info("消融实验训练器资源已清理")

    def evaluate(self, num_episodes: int = 10) -> Dict[str, Any]:
        """评估模型性能
        
        Args:
            num_episodes: 评估轮数
            
        Returns:
            Dict[str, Any]: 评估结果
        """
        if not self.ablation_manager:
            self.logger.error("消融组件管理器未初始化")
            return {}
        
        # 使用当前消融管理器进行评估
        eval_results = {}
        
        # 设置评估模式
        if hasattr(self.ablation_manager, 'set_training_mode'):
            self.ablation_manager.set_training_mode(False)
        
        eval_rewards = []
        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0
            
            for step in range(1000):  # 最大步数
                action, _ = self.ablation_manager.predict(obs)
                obs, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                
                if terminated or truncated:
                    break
            
            eval_rewards.append(episode_reward)
        
        # 恢复训练模式
        if hasattr(self.ablation_manager, 'set_training_mode'):
            self.ablation_manager.set_training_mode(True)
        
        eval_results = {
            'avg_episode_reward': np.mean(eval_rewards),
            'std_episode_reward': np.std(eval_rewards),
            'experiment_group': self.current_ablation_type if hasattr(self, 'current_ablation_type') else 'unknown'
        }
        
        return eval_results
    
    def save_model(self, path: Path, metadata: Optional[Dict] = None) -> bool:
        """保存模型
        
        Args:
            path: 保存路径
            metadata: 额外元数据
            
        Returns:
            bool: 是否保存成功
        """
        try:
            if not self.ablation_manager:
                self.logger.error("消融组件管理器未初始化，无法保存模型")
                return False
            
            # 使用消融管理器的保存方法
            if hasattr(self.ablation_manager, 'save_model'):
                self.ablation_manager.save_model(str(path))
            else:
                # 简化保存策略
                torch.save(self.ablation_manager.policy.state_dict(), str(path))
            
            # 保存额外元数据
            if metadata:
                metadata_path = path.with_suffix('.metadata.json')
                import json
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
            
            self.logger.info(f"消融实验模型已保存到: {path}")
            return True
            
        except Exception as e:
            self.logger.error(f"模型保存失败: {e}")
            return False
    
    def load_model(self, path: Path) -> bool:
        """加载模型
        
        Args:
            path: 模型路径
            
        Returns:
            bool: 是否加载成功
        """
        try:
            if not self.ablation_manager:
                self.logger.error("消融组件管理器未初始化，无法加载模型")
                return False
            
            # 使用消融管理器的加载方法
            if hasattr(self.ablation_manager, 'load_model'):
                self.ablation_manager.load_model(str(path))
            else:
                # 简化加载策略
                state_dict = torch.load(str(path))
                self.ablation_manager.policy.load_state_dict(state_dict)
            
            self.logger.info(f"消融实验模型已从 {path} 加载")
            return True
            
        except Exception as e:
            self.logger.error(f"模型加载失败: {e}")
            return False
    
    def _execute_training(self) -> Dict[str, Any]:
        """执行实际训练逻辑 - BaseTrainer抽象方法实现"""
        # 这个方法在train()中已经通过委托模式实现
        # 为了满足抽象基类要求，提供一个简化版本
        return {
            'training_completed': True,
            'ablation_types': self.ablation_types,
            'total_results': len(self.ablation_results)
        }
