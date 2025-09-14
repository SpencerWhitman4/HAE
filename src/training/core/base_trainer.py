#!/usr/bin/env python3

"""
基础训练器抽象类 - 集成智能会话管理和可视化系统
"""

import time
import logging
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

# 延迟导入避免循环依赖
SessionManager = None
TrainingVisualizationManager = None


class TrainingStage(Enum):
    """训练阶段枚举"""
    FOUNDATION = "foundation"
    HIERARCHICAL = "hierarchical"
    ABLATION = "ablation"
    BASELINE = "baseline"


@dataclass
class TrainingResult:
    """标准化的训练结果"""
    stage: TrainingStage
    success: bool
    total_steps: int
    total_episodes: int
    final_reward: float
    best_reward: float
    training_time: float
    model_path: Optional[Path]
    metrics: Dict[str, Any]
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'stage': self.stage.value,
            'success': self.success,
            'total_steps': self.total_steps,
            'total_episodes': self.total_episodes,
            'final_reward': self.final_reward,
            'best_reward': self.best_reward,
            'training_time': self.training_time,
            'model_path': str(self.model_path) if self.model_path else None,
            'metrics': self.metrics,
            'error_message': self.error_message
        }


class BaseTrainer(ABC):
    """
    基础训练器抽象类 - 集成智能会话管理和阶段感知可视化
    
    提供统一的训练接口，自动集成：
    - SessionManager: 智能目录管理和数据保存
    - VisualizationManager: 阶段感知的实时可视化
    """
    
    def __init__(self, 
                 stage: TrainingStage, 
                 config: Dict[str, Any],
                 experiment_name: str = "HA-UAV",
                 stage_variant: Optional[str] = None):
        """
        初始化基础训练器
        
        Args:
            stage: 训练阶段
            config: 训练配置
            experiment_name: 实验名称
            stage_variant: 阶段变体（如B1/B2/B3、ppo/sac等）
        """
        self.stage = stage
        self.config = config
        self.experiment_name = experiment_name
        self.stage_variant = stage_variant
        self.logger = logging.getLogger(f"{__name__}.{stage.value}")
        
        # 管理器实例
        self.session_manager = None
        self.visualization_manager = None
        
        # 训练状态
        self.training_start_time = None
        self.best_reward = float('-inf')
        self.current_step = 0
        self.current_episode = 0
        
        # 性能追踪
        self.training_metrics = {}
        self.evaluation_history = []
    
    def initialize_session(self, 
                          enable_trajectory: bool = True,
                          enable_tensorboard: bool = True,
                          enable_visualization: bool = True,
                          enable_pointcloud: bool = False,
                          enable_analysis: bool = False,
                          enable_rich_display: bool = True):
        """
        初始化会话管理器和可视化管理器
        
        Args:
            enable_*: 各种功能模块的启用标志
            enable_rich_display: 是否启用丰富的可视化显示
        """
        # 延迟导入
        global SessionManager, TrainingVisualizationManager
        if SessionManager is None:
            from .session_manager import SessionManager
        if TrainingVisualizationManager is None:
            from .visualization_manager import TrainingVisualizationManager
        
        # 初始化会话管理器
        self.session_manager = SessionManager(
            experiment_name=self.experiment_name,
            training_stage=self.stage,
            config=self.config,
            stage_variant=self.stage_variant,
            enable_trajectory=enable_trajectory,
            enable_tensorboard=enable_tensorboard,
            enable_visualization=enable_visualization,
            enable_pointcloud=enable_pointcloud,
            enable_analysis=enable_analysis
        )
        
        # 初始化可视化管理器
        if enable_visualization:
            total_steps = self.config.get('total_timesteps', 100000)
            eval_freq = self.config.get('evaluation_frequency', 10000)
            
            self.visualization_manager = TrainingVisualizationManager(
                total_steps=total_steps,
                training_stage=self.stage,
                experiment_name=self.experiment_name,
                stage_variant=self.stage_variant,
                evaluation_frequency=eval_freq,
                enable_rich_display=enable_rich_display
            )
        
        self.logger.info(f"✅ {self.stage.value} 训练器初始化完成")
        
        # 返回会话信息供子类使用
        return self.session_manager.get_session_info()
    
    @abstractmethod
    def setup(self) -> bool:
        """设置训练环境和模型 - 子类必须实现"""
        pass
    
    def train(self) -> TrainingResult:
        """
        执行训练过程 - 集成完整的会话管理和可视化
        
        Returns:
            TrainingResult: 标准化的训练结果
        """
        
        # 如果未初始化，使用默认设置
        if not self.session_manager:
            self.initialize_session()
        
        # 🔧 首先设置训练器（调用子类的setup方法）
        if not self.setup():
            return TrainingResult(
                stage=self.stage,
                success=False,
                total_steps=0,
                total_episodes=0,
                final_reward=0.0,
                best_reward=0.0,
                training_time=0.0,
                model_path=Path(""),
                metrics={},
                error_message="训练器设置失败"
            )
        
        self.training_start_time = time.time()
        
        try:
            # 训练开始回调
            if self.visualization_manager:
                config_summary = self._get_config_summary()
                self.visualization_manager.on_training_start(config_summary)
            
            # 执行实际训练
            training_metrics = self._execute_training()
            
            # 计算训练时间
            training_time = time.time() - self.training_start_time
            
            # 保存最终模型
            model_path = self._save_final_model()
            
            # 最终评估
            final_eval = self._perform_final_evaluation()
            training_metrics.update(final_eval)
            
            # 创建训练结果
            result = TrainingResult(
                stage=self.stage,
                success=True,
                total_steps=self.current_step,
                total_episodes=self.current_episode,
                final_reward=training_metrics.get('final_mean_reward', training_metrics.get('avg_episode_reward', 0.0)),
                best_reward=self.best_reward,
                training_time=training_time,
                model_path=model_path,
                metrics=training_metrics
            )
            
            # 保存训练结果
            if self.session_manager:
                self.session_manager.save_training_results(training_metrics)
            
            # 训练结束回调
            if self.visualization_manager:
                final_stats = {
                    'best_score': self.best_reward,
                    'final_reward': training_metrics.get('final_mean_reward', training_metrics.get('avg_episode_reward', 0.0)),
                    'total_episodes': self.current_episode,
                    'training_time': f"{training_time:.1f}s",
                    'model_path': str(model_path),
                    'final_success_rate': training_metrics.get('final_success_rate', training_metrics.get('success_rate', 0.0))
                }
                self.visualization_manager.on_training_end(final_stats)
            
            self.logger.info(f"✅ {self.stage.value} 训练成功完成")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ {self.stage.value} 训练失败: {e}")
            
            # 训练失败的结果
            return TrainingResult(
                stage=self.stage,
                success=False,
                total_steps=self.current_step,
                total_episodes=self.current_episode,
                final_reward=0.0,
                best_reward=self.best_reward,
                training_time=time.time() - self.training_start_time if self.training_start_time else 0.0,
                model_path=Path(""),
                metrics={'error': str(e)},
                error_message=str(e)
            )
        finally:
            # 清理资源
            self.cleanup()
    
    @abstractmethod
    def _execute_training(self) -> Dict[str, Any]:
        """执行实际训练逻辑 - 子类必须实现"""
        pass
    
    @abstractmethod
    def evaluate(self, num_episodes: int = 100) -> Dict[str, float]:
        """评估模型性能 - 子类必须实现"""
        pass
    
    @abstractmethod
    def save_model(self, path: Path) -> bool:
        """保存训练模型 - 子类必须实现"""
        pass
    
    @abstractmethod
    def load_model(self, path: Path) -> bool:
        """加载预训练模型 - 子类必须实现"""
        pass
    
    def _perform_final_evaluation(self) -> Dict[str, Any]:
        """执行最终评估"""
        try:
            final_eval_episodes = self.config.get('final_eval_episodes', 5)  # 减少为5
            
            if self.visualization_manager:
                self.visualization_manager.write_stage_message("开始最终评估...")
            
            final_eval_results = self.evaluate(final_eval_episodes)
            
            # 添加final_前缀区分最终评估
            final_results = {}
            for key, value in final_eval_results.items():
                final_results[f'final_{key}'] = value
            
            if self.visualization_manager:
                self.visualization_manager.write_stage_message(
                    f"最终评估完成 - 奖励: {final_eval_results.get('mean_reward', 0.0):.3f}"
                )
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"最终评估失败: {e}")
            return {'final_evaluation_error': str(e)}
    
    def _save_final_model(self) -> Path:
        """保存最终模型"""
        if self.session_manager:
            model_path = self.session_manager.get_model_save_path("final_model")
            self.save_model(model_path)
            
            if self.visualization_manager:
                self.visualization_manager.on_model_save(
                    "最终", 
                    self.best_reward, 
                    str(model_path)
                )
            
            return model_path
        else:
            # 默认路径
            model_path = Path(f"{self.stage.value}_final_model.zip")
            self.save_model(model_path)
            return model_path
    
    def _get_config_summary(self) -> Dict[str, Any]:
        """获取配置摘要用于显示"""
        summary = {
            'total_timesteps': self.config.get('total_timesteps', 100000),
            'evaluation_frequency': self.config.get('evaluation_frequency', 10000),
            'learning_rate': self.config.get('learning_rate', 3e-4),
            'batch_size': self.config.get('batch_size', 128),
            'foundation_model_path': self.config.get('foundation_model_path') is not None
        }
        
        # 根据训练阶段设置不同的配置信息
        if self.stage == TrainingStage.FOUNDATION:
            # 基座模型使用PPO参数
            summary.update({
                'buffer_size': self.config.get('buffer_size', self.config.get('n_steps', 2048)),
                'network_arch': self.config.get('network_arch', 'PPO-MLP'),
            })
        elif self.stage == TrainingStage.HIERARCHICAL:
            # 分层模型参数
            summary.update({
                'buffer_size': self.config.get('buffer_size', 100000),
                'network_arch': self.config.get('network_arch', 'HA-UAV-Hierarchical'),
                'high_level_update_frequency': self.config.get('high_level_update_frequency', 10),
                'future_horizon': self.config.get('future_horizon', 5)
            })
        elif self.stage == TrainingStage.ABLATION:
            # 消融研究参数
            summary.update({
                'buffer_size': self.config.get('buffer_size', 100000),
                'network_arch': self.config.get('network_arch', 'HA-UAV-Ablation'),
                'high_level_update_frequency': self.config.get('high_level_update_frequency', 10),
                'future_horizon': self.config.get('future_horizon', 5),
                'ablation_type': self.stage_variant or self.config.get('ablation_type', 'N/A')
            })
        elif self.stage == TrainingStage.BASELINE:
            # 基线算法参数
            summary.update({
                'buffer_size': self.config.get('buffer_size', 1000000),
                'network_arch': self.config.get('network_arch', 'SAC/TD3-MLP'),
                'algorithm': self.stage_variant or self.config.get('algorithm', 'N/A')
            })
        else:
            # 默认参数
            summary.update({
                'buffer_size': self.config.get('buffer_size', 'N/A'),
                'network_arch': self.config.get('network_arch', 'N/A'),
            })
        
        # 添加会话管理器的功能状态
        if self.session_manager:
            feature_flags = self.session_manager.feature_flags
            summary.update({
                'tensorboard_enabled': feature_flags.get('tensorboard', False),
                'trajectory_enabled': feature_flags.get('trajectory', False)
            })
        
        return summary
    
    # === 回调接口 ===
    
    def on_step_callback(self, step: int, metrics: Dict[str, Any]):
        """步骤回调 - 更新可视化和指标"""
        self.current_step = step
        self.training_metrics.update(metrics)
        
        if self.visualization_manager:
            self.visualization_manager.on_step(step, metrics)
    
    def on_episode_callback(self, episode: int, metrics: Dict[str, Any]):
        """Episode回调"""
        self.current_episode = episode
        self.training_metrics.update(metrics)
        
        if self.visualization_manager:
            self.visualization_manager.on_episode_end(episode, metrics)
    
    def on_evaluation_callback(self, eval_metrics: Dict[str, float]):
        """评估回调"""
        # 记录评估历史
        eval_record = {
            'step': self.current_step,
            'episode': self.current_episode,
            'metrics': eval_metrics.copy()
        }
        self.evaluation_history.append(eval_record)
        
        # 更新最佳奖励
        if 'mean_reward' in eval_metrics:
            if eval_metrics['mean_reward'] > self.best_reward:
                self.best_reward = eval_metrics['mean_reward']
                
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
            self.visualization_manager.on_evaluation_end(eval_metrics)
    
    def on_checkpoint_callback(self, step: int):
        """检查点保存回调"""
        if self.session_manager:
            checkpoint_path = self.session_manager.get_checkpoint_path(step)
            
            try:
                self.save_model(checkpoint_path)
                
                if self.visualization_manager:
                    self.visualization_manager.on_checkpoint_save(step, str(checkpoint_path))
                
                self.logger.info(f"检查点已保存: {checkpoint_path}")
                
            except Exception as e:
                self.logger.error(f"检查点保存失败: {e}")
    
    # === 数据链路接口 ===
    
    def get_session_paths(self) -> Dict[str, Path]:
        """获取会话路径"""
        if self.session_manager:
            return self.session_manager.get_session_paths()
        return {}
    
    def get_tensorboard_log_dir(self, mode: str = "train") -> Optional[Path]:
        """获取TensorBoard日志目录"""
        if self.session_manager:
            return self.session_manager.get_tensorboard_log_dir(mode)
        return None
    
    def get_trajectory_manager(self, mode: str = "train"):
        """获取轨迹管理器"""
        if self.session_manager:
            return self.session_manager.get_trajectory_manager(mode)
        return None
    
    def save_analysis_data(self, data: Dict[str, Any], filename: str):
        """保存分析数据"""
        if self.session_manager:
            self.session_manager.save_analysis_data(data, filename)
    
    def save_visualization_data(self, data: Dict[str, Any], filename: str):
        """保存可视化数据"""
        if self.session_manager:
            self.session_manager.save_visualization_data(data, filename)
    
    def write_message(self, message: str, level: str = "INFO"):
        """写入消息到可视化界面"""
        if self.visualization_manager:
            self.visualization_manager.write_message(message, level)
        else:
            getattr(self.logger, level.lower(), self.logger.info)(message)
    
    def write_stage_message(self, message: str):
        """写入阶段特定消息"""
        if self.visualization_manager:
            self.visualization_manager.write_stage_message(message)
        else:
            self.logger.info(f"[{self.stage.value}] {message}")
    
    # === 资源管理 ===
    
    def cleanup(self):
        """清理资源"""
        if self.session_manager:
            self.session_manager.close()
        
        if self.visualization_manager:
            self.visualization_manager.close()
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.cleanup()


# 便捷工厂函数
class TrainerFactory:
    """训练器工厂类"""
    
    _trainers = {}
    
    @classmethod
    def register(cls, stage: TrainingStage, trainer_class):
        """注册训练器类"""
        cls._trainers[stage] = trainer_class
    
    @classmethod
    def create(cls, stage: TrainingStage, config: Dict[str, Any], **kwargs):
        """创建训练器实例"""
        if stage not in cls._trainers:
            raise ValueError(f"未注册的训练阶段: {stage}")
        
        return cls._trainers[stage](config=config, **kwargs)


# 进度回调类
class TrainingProgressCallback:
    """训练进度回调接口"""
    
    def __init__(self, trainer: BaseTrainer):
        self.trainer = trainer
    
    def on_step(self, step: int, metrics: Dict[str, Any]):
        """步骤回调"""
        self.trainer.on_step_callback(step, metrics)
    
    def on_episode_end(self, episode: int, metrics: Dict[str, Any]):
        """Episode结束回调"""
        self.trainer.on_episode_callback(episode, metrics)
    
    def on_evaluation_end(self, eval_metrics: Dict[str, float]):
        """评估结束回调"""
        self.trainer.on_evaluation_callback(eval_metrics)
    
    def on_checkpoint_save(self, step: int):
        """检查点保存回调"""
        self.trainer.on_checkpoint_callback(step)
    
    @abstractmethod
    def evaluate(self, num_episodes: int = 10) -> Dict[str, Any]:
        """
        评估模型性能
        
        Args:
            num_episodes: 评估轮数
            
        Returns:
            Dict[str, Any]: 评估结果
        """
        pass
    
    @abstractmethod
    def save_model(self, path: Path, metadata: Optional[Dict] = None) -> bool:
        """
        保存模型
        
        Args:
            path: 保存路径
            metadata: 额外元数据
            
        Returns:
            bool: 是否保存成功
        """
        pass
    
    @abstractmethod
    def load_model(self, path: Path) -> bool:
        """
        加载模型
        
        Args:
            path: 模型路径
            
        Returns:
            bool: 是否加载成功
        """
        pass
    
    def cleanup(self) -> None:
        """清理资源"""
        try:
            if self.env is not None:
                self.env.close()
            self.logger.info(f"{self.stage.value} 训练器资源清理完成")
        except Exception as e:
            self.logger.error(f"资源清理时发生错误: {e}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """获取训练摘要"""
        training_time = time.time() - self.start_time if self.start_time else 0
        
        return {
            'stage': self.stage.value,
            'current_step': self.current_step,
            'current_episode': self.current_episode,
            'best_score': self.best_score,
            'training_time': training_time,
            'is_initialized': self.is_initialized,
            'metrics': self.training_metrics.copy()
        }
    
    def log_progress(self, step: int, metrics: Dict[str, Any]) -> None:
        """记录训练进度"""
        self.current_step = step
        self.training_metrics.update(metrics)
        
        # 改为基于间隔的日志记录
        steps_since_last_log = step - getattr(self, '_last_log_step', 0)
        if steps_since_last_log >= 10000 and step > 0:  # 每10k步记录一次
            self._last_log_step = step
            self.logger.info(
                f"[{self.stage.value}] Step {step}: "
                f"Episode {self.current_episode}, "
                f"Best Score {self.best_score:.3f}"
            )
    
    def _validate_config(self) -> bool:
        """验证配置有效性"""
        required_keys = ['total_steps', 'eval_freq', 'save_freq']
        
        for key in required_keys:
            if key not in self.config:
                self.logger.error(f"配置缺少必需参数: {key}")
                return False
        
        return True
    
    def _create_session_paths(self) -> Dict[str, Path]:
        """创建会话路径（复用现有session_manager）"""
        if self.session_manager:
            return self.session_manager.get_session_paths()
        else:
            # 简单路径创建
            base_path = Path("./logs") / f"{self.stage.value}_session"
            base_path.mkdir(parents=True, exist_ok=True)
            
            return {
                'session': base_path,
                'models': base_path / 'models',
                'results': base_path / 'results',
                'plots': base_path / 'plots'
            }


class TrainerFactory:
    """训练器工厂 - 根据阶段创建相应的训练器"""
    
    @staticmethod
    def create_trainer(stage: TrainingStage, 
                      config: Dict[str, Any], 
                      session_manager=None, 
                      **kwargs) -> BaseTrainer:
        """
        创建训练器实例
        
        Args:
            stage: 训练阶段
            config: 配置
            session_manager: 会话管理器
            **kwargs: 额外参数
            
        Returns:
            BaseTrainer: 对应的训练器实例
        """
        if stage == TrainingStage.FOUNDATION:
            from ..foundation.baseflight_trainer import BaseFlightTrainer
            return BaseFlightTrainer(stage, config, session_manager, **kwargs)
            
        elif stage == TrainingStage.HIERARCHICAL:
            from ..branches.hierarchical_trainer import HierarchicalTrainer
            return HierarchicalTrainer(stage, config, session_manager, **kwargs)
            
        elif stage == TrainingStage.ABLATION:
            from ..branches.ablation_trainer import AblationTrainer
            return AblationTrainer(stage, config, session_manager, **kwargs)
            
        elif stage == TrainingStage.BASELINE:
            from ..branches.baseline_trainer import BaselineTrainer
            return BaselineTrainer(stage, config, session_manager, **kwargs)
            
        else:
            raise ValueError(f"不支持的训练阶段: {stage}")


class TrainingProgressCallback:
    """训练进度回调 - 复用现有visualization_manager的概念"""
    
    def __init__(self, trainer: BaseTrainer):
        self.trainer = trainer
        self.logger = trainer.logger
    
    def on_training_start(self) -> None:
        """训练开始回调"""
        self.logger.info(f"开始 {self.trainer.stage.value} 阶段训练")
        self.trainer.start_time = time.time()
    
    def on_step(self, step: int, metrics: Dict[str, Any]) -> None:
        """步骤回调"""
        self.trainer.log_progress(step, metrics)
    
    def on_episode_end(self, episode: int, reward: float) -> None:
        """Episode结束回调"""
        self.trainer.current_episode = episode
        if reward > self.trainer.best_score:
            self.trainer.best_score = reward
    
    def on_evaluation(self, eval_results: Dict[str, Any]) -> None:
        """评估回调"""
        self.logger.info(f"评估结果: {eval_results}")
    
    def on_training_end(self, results: TrainingResult) -> None:
        """训练结束回调"""
        self.logger.info(
            f"{self.trainer.stage.value} 训练完成: "
            f"成功={results.success}, "
            f"总步数={results.total_steps}, "
            f"最佳奖励={results.best_reward:.3f}"
        )
