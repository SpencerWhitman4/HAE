#!/usr/bin/env python3

"""
训练可视化管理器 - 智能的阶段感知可视化系统
"""

import logging
from typing import Dict, Any, Optional
from tqdm import tqdm

# 可选的colorama支持
try:
    import colorama
    from colorama import Fore, Style
    colorama.init()
    COLORAMA_AVAILABLE = True
except ImportError:
    # 如果colorama不可用，定义空的颜色常量
    class _MockColor:
        def __getattr__(self, name): return ""
    Fore = _MockColor()
    Style = _MockColor()
    COLORAMA_AVAILABLE = False

from src.training.core.base_trainer import TrainingStage


class TrainingVisualizationManager:
    """
    智能训练可视化管理器
    
    根据不同训练阶段提供定制化的可视化体验:
    - Foundation: 蓝色主题，基础飞行图标
    - Hierarchical: 绿色主题，分层架构图标
    - Ablation: 黄色主题，实验分析图标
    - Baseline: 紫色主题，基准对比图标
    """
    
    # 阶段可视化配置
    STAGE_THEMES = {
        TrainingStage.FOUNDATION: {
            'name': 'Foundation Training',
            'icon': '🏗️',
            'color': Fore.CYAN,
            'bar_color': 'blue',
            'accent_color': Fore.BLUE,
            'description': '基座模型训练'
        },
        TrainingStage.HIERARCHICAL: {
            'name': 'Hierarchical Training',
            'icon': '🚁',
            'color': Fore.GREEN,
            'bar_color': 'green', 
            'accent_color': Fore.LIGHTGREEN_EX,
            'description': 'HA-UAV分层训练'
        },
        TrainingStage.ABLATION: {
            'name': 'Ablation Study',
            'icon': '🔬',
            'color': Fore.YELLOW,
            'bar_color': 'yellow',
            'accent_color': Fore.LIGHTYELLOW_EX,
            'description': '消融实验研究'
        },
        TrainingStage.BASELINE: {
            'name': 'Baseline Training',
            'icon': '📊',
            'color': Fore.MAGENTA,
            'bar_color': 'magenta',
            'accent_color': Fore.LIGHTMAGENTA_EX,
            'description': '基线算法训练'
        }
    }
    
    def __init__(self,
                 total_steps: int,
                 training_stage: TrainingStage,
                 experiment_name: str = "HA-UAV",
                 stage_variant: Optional[str] = None,
                 evaluation_frequency: int = 10000,
                 update_frequency: int = 1,
                 enable_rich_display: bool = True):
        """
        初始化可视化管理器
        
        Args:
            total_steps: 总训练步数
            training_stage: 训练阶段
            experiment_name: 实验名称
            stage_variant: 阶段变体（如B1/B2/B3、ppo/sac等）
            evaluation_frequency: 评估频率
            update_frequency: 进度条更新频率
            enable_rich_display: 是否启用丰富显示
        """
        self.total_steps = total_steps
        self.training_stage = training_stage
        self.experiment_name = experiment_name
        self.stage_variant = stage_variant
        self.evaluation_frequency = evaluation_frequency
        self.update_frequency = update_frequency
        self.enable_rich_display = enable_rich_display
        
        # 获取阶段主题
        self.theme = self.STAGE_THEMES.get(training_stage, self.STAGE_THEMES[TrainingStage.HIERARCHICAL])
        
        # 构建显示名称
        display_name = f"{self.theme['icon']} {experiment_name} {self.theme['description']}"
        if stage_variant:
            display_name += f" ({stage_variant})"
        
        # 创建进度条
        self.progress_bar = self._create_progress_bar(display_name)
        
        # 训练指标追踪
        self.training_metrics = {
            'current_step': 0,
            'episode': 0,
            'total_reward': 0.0,
            'mean_reward': 0.0,
            'exploration_rate': 0.0,
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'total_loss': 0.0,
            'best_score': float('-inf'),
            'success_rate': 0.0,
            'last_update_step': 0,
            'episodes_completed': 0,
            'learning_rate': 0.0,
            'entropy': 0.0
        }
        
        # 评估相关
        self.evaluation_history = []
        self.current_evaluation = None
        
        self.logger = logging.getLogger(__name__)
        
        # 调试信息
        self.logger.info(f"✅ 可视化管理器已初始化: {display_name}")
        self.logger.info(f"📊 总步数: {total_steps}, 更新频率: {update_frequency}")
    
    def _create_progress_bar(self, display_name: str) -> tqdm:
        """创建定制化进度条"""
        if self.enable_rich_display:
            bar_format = (
                f"{self.theme['color']}{display_name}{Style.RESET_ALL} "
                f"|{{bar:50}}| {{percentage:3.0f}}% "
                f"[{{elapsed}}<{{remaining}}] {{postfix}}"
            )
            ncols = 160
        else:
            bar_format = '{l_bar}{bar:30}{r_bar}'
            ncols = 120
        
        return tqdm(
            total=self.total_steps,
            desc=f"{self.theme['color']}{display_name}{Style.RESET_ALL}",
            ncols=ncols,
            bar_format=bar_format,
            colour=self.theme['bar_color'],
            dynamic_ncols=True,
            smoothing=0.1,
            miniters=max(1, self.update_frequency),  # 确保至少为1
            maxinterval=2.0,  # 最大更新间隔2秒
            mininterval=0.1   # 最小更新间隔0.1秒
        )
    
    def on_training_start(self, config_summary: Dict[str, Any]):
        """训练开始回调 - 显示配置信息"""
        if not self.enable_rich_display:
            return
        
        # 打印训练开始横幅
        banner_color = self.theme['color']
        accent_color = self.theme['accent_color']
        
        print(f"\n{banner_color}{'='*120}{Style.RESET_ALL}")
        print(f"{banner_color}{self.theme['icon']} {self.experiment_name} - {self.theme['description']}{Style.RESET_ALL}")
        if self.stage_variant:
            print(f"{accent_color}变体配置: {self.stage_variant}{Style.RESET_ALL}")
        print(f"{banner_color}{'='*120}{Style.RESET_ALL}")
        
        # 显示配置信息
        self._display_config_table(config_summary)
        print(f"{banner_color}{'='*120}{Style.RESET_ALL}\n")
    
    def _display_config_table(self, config_summary: Dict[str, Any]):
        """显示配置表格"""
        print(f"{self.theme['accent_color']}📋 训练配置:{Style.RESET_ALL}")
        
        config_items = [
            ("总训练步数", f"{config_summary.get('total_timesteps', self.total_steps):,}"),
            ("评估频率", f"{config_summary.get('evaluation_frequency', self.evaluation_frequency):,}"),
            ("缓冲区大小", str(config_summary.get('buffer_size', 'N/A'))),
            ("学习率", str(config_summary.get('learning_rate', 'N/A'))),
            ("批次大小", str(config_summary.get('batch_size', 'N/A'))),
            ("网络架构", str(config_summary.get('network_arch', 'N/A'))),
        ]
        
        # 阶段特定配置
        if self.training_stage == TrainingStage.HIERARCHICAL:
            config_items.extend([
                ("高层更新频率", str(config_summary.get('high_level_update_frequency', 'N/A'))),
                ("未来视野", str(config_summary.get('future_horizon', 'N/A'))),
            ])
        elif self.training_stage == TrainingStage.ABLATION:
            config_items.append(("消融类型", str(config_summary.get('ablation_type', 'N/A'))))
        elif self.training_stage == TrainingStage.BASELINE:
            config_items.append(("基线算法", str(config_summary.get('algorithm', 'N/A'))))
        
        # 功能开关
        feature_items = [
            ("TensorBoard", config_summary.get('tensorboard_enabled', False)),
            ("轨迹记录", config_summary.get('trajectory_enabled', False)),
            ("基座模型", config_summary.get('foundation_model_path') is not None),
        ]
        
        # 显示配置项
        for key, value in config_items:
            status_color = Fore.WHITE if str(value) != 'N/A' else Fore.LIGHTBLACK_EX
            print(f"  📊 {key}: {status_color}{value}{Style.RESET_ALL}")
        
        # 显示功能状态
        for key, enabled in feature_items:
            status_color = Fore.GREEN if enabled else Fore.RED
            status_text = "✅" if enabled else "❌"
            print(f"  {status_text} {key}: {status_color}{enabled}{Style.RESET_ALL}")
    
    def on_step(self, step: int, metrics: Dict[str, Any]):
        """每步训练回调 - 标准化指标格式"""
        # 调试信息 - 仅在关键步骤记录
        if step == 0:
            self.logger.info(f"🎯 第一次回调: step={step}, metrics={list(metrics.keys())}")
        
        # 标准化指标格式
        standardized_metrics = self._standardize_metrics(metrics)
        
        # 更新指标
        self.training_metrics.update(standardized_metrics)
        self.training_metrics['current_step'] = step
        
        # 计算更新步数
        steps_to_update = step - self.training_metrics['last_update_step']
        
        # 按频率更新进度条 - 对于初始步骤或达到更新频率时更新
        if steps_to_update >= self.update_frequency or step == 0:
            self.progress_bar.update(steps_to_update)
            self.training_metrics['last_update_step'] = step
            
            # 调试信息
            if step == 0:
                self.logger.info(f"📊 进度条更新: steps_to_update={steps_to_update}, progress={self.progress_bar.n}/{self.progress_bar.total}")
            
            # 更新状态显示
            if self.enable_rich_display:
                status_info = self._format_training_status()
                self.progress_bar.set_postfix_str(status_info)
            
            self.progress_bar.refresh()
    
    def on_episode_end(self, episode: int, episode_metrics: Dict[str, Any]):
        """Episode结束回调 - 标准化指标格式"""
        # 标准化episode指标
        standardized_metrics = self._standardize_metrics(episode_metrics)
        
        self.training_metrics.update(standardized_metrics)
        self.training_metrics['episode'] = episode
        self.training_metrics['episodes_completed'] += 1
        
        # 更新显示
        if self.enable_rich_display:
            status_info = self._format_training_status()
            self.progress_bar.set_postfix_str(status_info)
            self.progress_bar.refresh()
    
    def on_evaluation_start(self, num_episodes: int):
        """评估开始回调"""
        eval_msg = (
            f"{self.theme['accent_color']}🔍 评估开始 "
            f"({num_episodes} episodes){Style.RESET_ALL}"
        )
        self.progress_bar.write(eval_msg)
        
        self.current_evaluation = {
            'start_step': self.training_metrics['current_step'],
            'num_episodes': num_episodes
        }
    
    def on_evaluation_end(self, eval_metrics: Dict[str, float]):
        """评估结束回调"""
        # 记录评估历史
        eval_record = {
            'step': self.training_metrics['current_step'],
            'metrics': eval_metrics.copy()
        }
        self.evaluation_history.append(eval_record)
        
        # 更新最佳分数
        if 'mean_reward' in eval_metrics:
            if eval_metrics['mean_reward'] > self.training_metrics['best_score']:
                self.training_metrics['best_score'] = eval_metrics['mean_reward']
        
        # 更新成功率
        if 'success_rate' in eval_metrics:
            self.training_metrics['success_rate'] = eval_metrics['success_rate']
        
        # 显示评估结果
        eval_result_msg = self._format_evaluation_results(eval_metrics)
        self.progress_bar.write(eval_result_msg)
        
        self.current_evaluation = None
    
    def _format_evaluation_results(self, eval_metrics: Dict[str, float]) -> str:
        """格式化评估结果"""
        parts = [f"{self.theme['color']}📊 评估结果:{Style.RESET_ALL}"]
        
        metrics_display = [
            ("奖励", eval_metrics.get('mean_reward'), ".3f", Fore.YELLOW),
            ("成功率", eval_metrics.get('success_rate'), ".3f", Fore.GREEN),
            ("Episode长度", eval_metrics.get('mean_episode_length'), ".1f", Fore.CYAN),
            ("标准差", eval_metrics.get('std_reward'), ".3f", Fore.WHITE)
        ]
        
        # 添加基座模型特定的质量指标
        if 'avg_hover_quality' in eval_metrics:
            metrics_display.extend([
                ("悬停质量", eval_metrics.get('avg_hover_quality'), ".3f", Fore.MAGENTA),
                ("飞行质量", eval_metrics.get('avg_flight_quality'), ".3f", Fore.BLUE),
                ("位置稳定性", eval_metrics.get('avg_position_stability'), ".3f", Fore.LIGHTGREEN_EX),
                ("速度平滑性", eval_metrics.get('avg_velocity_smoothness'), ".3f", Fore.LIGHTCYAN_EX)
            ])
        
        for name, value, fmt, color in metrics_display:
            if value is not None:
                if fmt and isinstance(value, (int, float)) and not isinstance(value, str):
                    # 转换 numpy 类型为原生 Python 类型
                    if hasattr(value, 'item'):
                        value = value.item()
                    formatted_value = f"{float(value):{fmt}}"
                else:
                    formatted_value = str(value)
                parts.append(f"{color}{name}:{formatted_value}{Style.RESET_ALL}")
        
        return " | ".join(parts)
    
    def on_model_save(self, model_type: str, score: Optional[float] = None, path: Optional[str] = None):
        """模型保存回调"""
        save_icon = "💾"
        if model_type == "最佳":
            save_icon = "🏆"
        elif model_type == "检查点":
            save_icon = "📋"
        elif model_type == "最终":
            save_icon = "🎯"
        
        if score is not None:
            save_msg = (
                f"{self.theme['accent_color']}{save_icon} {model_type}模型已保存 "
                f"(分数: {score:.3f}){Style.RESET_ALL}"
            )
        else:
            save_msg = f"{self.theme['accent_color']}{save_icon} {model_type}模型已保存{Style.RESET_ALL}"
        
        if path:
            save_msg += f" - {Fore.LIGHTBLACK_EX}{path}{Style.RESET_ALL}"
        
        self.progress_bar.write(save_msg)
    
    def on_checkpoint_save(self, step: int, path: str):
        """检查点保存回调"""
        checkpoint_msg = (
            f"{self.theme['accent_color']}📋 检查点已保存 "
            f"(步骤: {step:,}) - {Fore.LIGHTBLACK_EX}{path}{Style.RESET_ALL}"
        )
        self.progress_bar.write(checkpoint_msg)
    
    def on_training_end(self, final_stats: Dict[str, Any]):
        """训练结束回调"""
        # 确保进度条完成
        remaining_steps = self.total_steps - self.progress_bar.n
        if remaining_steps > 0:
            self.progress_bar.update(remaining_steps)
        
        self.progress_bar.close()
        
        if not self.enable_rich_display:
            return
        
        # 显示训练完成横幅
        banner_color = self.theme['color']
        print(f"\n{banner_color}{'='*120}{Style.RESET_ALL}")
        print(f"{banner_color}🎉 {self.theme['description']}完成!{Style.RESET_ALL}")
        if self.stage_variant:
            print(f"{self.theme['accent_color']}变体: {self.stage_variant}{Style.RESET_ALL}")
        print(f"{banner_color}{'='*120}{Style.RESET_ALL}")
        
        # 显示最终统计
        self._display_final_statistics(final_stats)
        print(f"{banner_color}{'='*120}{Style.RESET_ALL}")
    
    def _display_final_statistics(self, final_stats: Dict[str, Any]):
        """显示最终统计信息"""
        print(f"{self.theme['accent_color']}📈 最终统计:{Style.RESET_ALL}")
        
        stats_items = [
            ("🏆 最佳分数", final_stats.get('best_score', self.training_metrics['best_score']), ".3f", Fore.YELLOW),
            ("✅ 最终成功率", final_stats.get('final_success_rate', self.training_metrics['success_rate']), ".3f", Fore.GREEN),
            ("📊 总Episodes", final_stats.get('total_episodes', self.training_metrics['episodes_completed']), "", Fore.WHITE),
            ("⏱️ 训练时长", final_stats.get('training_time', 'N/A'), "", Fore.CYAN),
        ]
        
        if 'final_reward' in final_stats:
            stats_items.insert(1, ("📈 最终奖励", final_stats['final_reward'], ".3f", Fore.LIGHTGREEN_EX))
        
        for name, value, fmt, color in stats_items:
            if value != 'N/A' and value is not None:
                if fmt and isinstance(value, (int, float)) and not isinstance(value, str):
                    # 转换 numpy 类型为原生 Python 类型
                    if hasattr(value, 'item'):
                        value = value.item()
                    display_value = f"{float(value):{fmt}}"
                else:
                    display_value = str(value)
                print(f"  {name}: {color}{display_value}{Style.RESET_ALL}")
        
        # 显示模型路径
        if 'model_path' in final_stats:
            print(f"  💾 模型路径: {Fore.BLUE}{final_stats['model_path']}{Style.RESET_ALL}")
        
        # 显示评估历史摘要
        if self.evaluation_history:
            print(f"\n{self.theme['accent_color']}📊 评估历史摘要:{Style.RESET_ALL}")
            print(f"  🔢 评估次数: {Fore.WHITE}{len(self.evaluation_history)}{Style.RESET_ALL}")
            
            if len(self.evaluation_history) > 1:
                rewards = [eval_rec['metrics'].get('mean_reward', 0) for eval_rec in self.evaluation_history]
                improvement = rewards[-1] - rewards[0] if rewards else 0
                color = Fore.GREEN if improvement > 0 else Fore.RED
                print(f"  📈 性能改进: {color}{improvement:+.3f}{Style.RESET_ALL}")
    
    def _format_training_status(self) -> str:
        """格式化训练状态信息"""
        # 距离下次评估的步数
        steps_to_eval = self.evaluation_frequency - (self.training_metrics['current_step'] % self.evaluation_frequency)
        
        # 构建状态字符串
        status_parts = [
            f"{Fore.WHITE}Ep:{int(self.training_metrics['episode']):>4d}{Style.RESET_ALL}",
            f"{Fore.YELLOW}R:{self.training_metrics.get('mean_reward', 0.0):>6.2f}{Style.RESET_ALL}",
        ]
        
        # 添加探索率（如果可用）
        if self.training_metrics.get('exploration_rate', 0.0) > 0:
            status_parts.append(f"{Fore.CYAN}Exp:{self.training_metrics['exploration_rate']:>5.3f}{Style.RESET_ALL}")
        
        # 添加损失信息
        if self.training_metrics.get('policy_loss', 0.0) != 0.0:
            status_parts.append(f"{Fore.RED}PL:{self.training_metrics['policy_loss']:>7.4f}{Style.RESET_ALL}")
        
        if self.training_metrics.get('value_loss', 0.0) != 0.0:
            status_parts.append(f"{Fore.BLUE}VL:{self.training_metrics['value_loss']:>7.4f}{Style.RESET_ALL}")
        
        # 添加性能指标
        status_parts.extend([
            f"{Fore.GREEN}Best:{self.training_metrics['best_score']:>6.2f}{Style.RESET_ALL}",
            f"{Fore.MAGENTA}Succ:{self.training_metrics['success_rate']:>5.3f}{Style.RESET_ALL}",
            f"{Fore.WHITE}EvalIn:{steps_to_eval:>5d}{Style.RESET_ALL}"
        ])
        
        return " | ".join(status_parts)
    
    def write_message(self, message: str, level: str = "INFO"):
        """写入自定义消息"""
        level_colors = {
            "DEBUG": Fore.LIGHTBLACK_EX,
            "INFO": Fore.WHITE,
            "WARNING": Fore.YELLOW,
            "ERROR": Fore.RED,
            "SUCCESS": Fore.GREEN
        }
        
        color = level_colors.get(level.upper(), Fore.WHITE)
        colored_message = f"{color}{message}{Style.RESET_ALL}"
        self.progress_bar.write(colored_message)
    
    def write_stage_message(self, message: str):
        """写入阶段特定消息"""
        stage_message = f"{self.theme['color']}{self.theme['icon']} {message}{Style.RESET_ALL}"
        self.progress_bar.write(stage_message)
    
    def update_metrics(self, metrics: Dict[str, Any]):
        """手动更新训练指标 - 使用标准化格式"""
        standardized_metrics = self._standardize_metrics(metrics)
        self.training_metrics.update(standardized_metrics)
        
        if self.enable_rich_display:
            status_info = self._format_training_status()
            self.progress_bar.set_postfix_str(status_info)
            self.progress_bar.refresh()
    
    def _standardize_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        标准化训练指标格式
        
        Args:
            metrics: 原始指标字典
            
        Returns:
            标准化后的指标字典
        """
        standardized = {}
        
        # 标准指标映射
        metric_mapping = {
            # 奖励相关
            'reward': ['reward', 'mean_reward', 'episode_reward', 'total_reward'],
            'episode_length': ['episode_length', 'length', 'steps', 'episode_steps'],
            'success_rate': ['success_rate', 'success', 'is_success'],
            
            # 训练相关
            'loss': ['loss', 'total_loss', 'policy_loss', 'value_loss'],
            'entropy': ['entropy', 'entropy_loss'],
            'learning_rate': ['learning_rate', 'lr'],
            'buffer_size': ['buffer_size', 'replay_buffer_size'],
            
            # 探索相关
            'exploration_rate': ['exploration_rate', 'epsilon', 'exploration'],
            'clip_fraction': ['clip_fraction', 'clip_frac'],
            
            # 性能相关
            'fps': ['fps', 'steps_per_second'],
            'explained_variance': ['explained_variance', 'explained_var']
        }
        
        # 执行映射
        for standard_key, possible_keys in metric_mapping.items():
            for key in possible_keys:
                if key in metrics:
                    try:
                        # 确保数值类型
                        value = float(metrics[key])
                        standardized[standard_key] = value
                        break
                    except (ValueError, TypeError):
                        continue
        
        # 保留其他未映射的指标
        for key, value in metrics.items():
            if key not in standardized:
                try:
                    # 尝试转换为数值
                    if isinstance(value, (int, float)):
                        standardized[key] = float(value)
                    elif isinstance(value, str) and value.replace('.', '').replace('-', '').isdigit():
                        standardized[key] = float(value)
                except:
                    # 保留非数值指标
                    standardized[key] = value
        
        return standardized
    
    def get_evaluation_history(self) -> list:
        """获取评估历史"""
        return self.evaluation_history.copy()
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """获取当前训练指标"""
        return self.training_metrics.copy()
    
    def close(self):
        """关闭可视化管理器"""
        if hasattr(self, 'progress_bar') and self.progress_bar:
            self.progress_bar.close()


# 便捷工厂函数
def create_visualization_manager(total_steps: int,
                               training_stage: TrainingStage,
                               experiment_name: str = "HA-UAV",
                               stage_variant: Optional[str] = None,
                               **kwargs) -> TrainingVisualizationManager:
    """创建可视化管理器的便捷函数"""
    return TrainingVisualizationManager(
        total_steps=total_steps,
        training_stage=training_stage,
        experiment_name=experiment_name,
        stage_variant=stage_variant,
        **kwargs
    )
