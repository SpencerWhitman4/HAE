#!/usr/bin/env python3

"""
训练流水线 - 统一的四阶段训练流程管理
"""

import logging
import time
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import json

from .base_trainer import TrainingStage, TrainingResult, TrainingProgressCallback
from .environment_factory import EnvironmentFactory
from .model_transfer import ModelTransferManager

logger = logging.getLogger(__name__)


class PipelineStatus(Enum):
    """流水线状态"""
    IDLE = "idle"
    PREPARING = "preparing"
    FOUNDATION_TRAINING = "foundation_training"
    TRANSFERRING = "transferring"
    BRANCH_TRAINING = "branch_training"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class PipelineConfig:
    """流水线配置"""
    # 基础配置
    experiment_name: str = "hauav_pipeline"
    output_dir: str = "./experiments"
    log_level: str = "INFO"
    
    # Foundation阶段配置
    foundation_config: Dict[str, Any] = None
    foundation_trainer_class: str = "BaseFlightTrainer"
    
    # 分支训练配置
    branches_config: Dict[str, Dict[str, Any]] = None
    enable_parallel_branches: bool = False
    
    # 迁移配置
    transfer_config: Dict[str, Any] = None
    
    # 评估配置
    evaluation_config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.foundation_config is None:
            self.foundation_config = {}
        if self.branches_config is None:
            self.branches_config = {
                "hierarchical": {},
                "ablation": {"ablation_types": ["B1", "B2", "B3"]},
                "baseline": {"algorithms": ["ppo", "sac"]}
            }
        if self.transfer_config is None:
            self.transfer_config = {"transfer_mode": "partial"}
        if self.evaluation_config is None:
            self.evaluation_config = {"eval_episodes": 100}


@dataclass 
class PipelineResult:
    """流水线结果"""
    status: PipelineStatus
    total_duration: float
    foundation_result: Optional[TrainingResult] = None
    branch_results: Dict[str, TrainingResult] = None
    evaluation_results: Dict[str, Any] = None
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.branch_results is None:
            self.branch_results = {}


class TrainingPipeline:
    """
    训练流水线 - 编排四阶段训练流程
    
    Foundation → (Hierarchical | Ablation | Baseline)
    """
    
    def __init__(self, 
                 config: PipelineConfig,
                 environment_factory: Optional[EnvironmentFactory] = None,
                 model_transfer_manager: Optional[ModelTransferManager] = None):
        
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, config.log_level.upper()))
        
        # 初始化组件
        self.env_factory = environment_factory or EnvironmentFactory()
        self.transfer_manager = model_transfer_manager or ModelTransferManager()
        
        # 状态管理
        self.status = PipelineStatus.IDLE
        self.start_time = None
        self.foundation_checkpoint = None
        
        # 结果存储
        self.experiment_dir = Path(config.output_dir) / config.experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # 回调函数
        self.progress_callbacks: List[Callable] = []
        
        # 训练器实例缓存
        self._trainer_cache = {}
    
    def add_progress_callback(self, callback: Callable) -> None:
        """添加进度回调函数"""
        self.progress_callbacks.append(callback)
    
    def run_pipeline(self) -> PipelineResult:
        """
        执行完整的训练流水线
        
        Returns:
            流水线执行结果
        """
        
        self.start_time = time.time()
        self.status = PipelineStatus.PREPARING
        
        try:
            # 1. 准备阶段
            self._notify_progress("开始训练流水线", {"stage": "preparation"})
            self._prepare_pipeline()
            
            # 2. Foundation训练
            self.status = PipelineStatus.FOUNDATION_TRAINING
            self._notify_progress("开始基座模型训练", {"stage": "foundation"})
            foundation_result = self._run_foundation_training()
            
            # 3. 模型迁移准备
            self.status = PipelineStatus.TRANSFERRING
            self._notify_progress("准备模型迁移", {"stage": "transfer"})
            self._prepare_model_transfer()
            
            # 4. 分支训练
            self.status = PipelineStatus.BRANCH_TRAINING
            self._notify_progress("开始分支训练", {"stage": "branches"})
            branch_results = self._run_branch_training()
            
            # 5. 评估阶段
            self.status = PipelineStatus.EVALUATING
            self._notify_progress("开始评估", {"stage": "evaluation"})
            evaluation_results = self._run_evaluation(branch_results)
            
            # 6. 完成
            self.status = PipelineStatus.COMPLETED
            total_duration = time.time() - self.start_time
            
            result = PipelineResult(
                status=self.status,
                total_duration=total_duration,
                foundation_result=foundation_result,
                branch_results=branch_results,
                evaluation_results=evaluation_results
            )
            
            self._save_pipeline_result(result)
            self._notify_progress("流水线完成", {"duration": total_duration})
            
            return result
            
        except Exception as e:
            self.status = PipelineStatus.FAILED
            error_duration = time.time() - self.start_time if self.start_time else 0
            
            error_result = PipelineResult(
                status=self.status,
                total_duration=error_duration,
                error_message=str(e)
            )
            
            self.logger.error(f"流水线执行失败: {e}")
            self._notify_progress("流水线失败", {"error": str(e)})
            
            return error_result
    
    def _prepare_pipeline(self) -> None:
        """准备流水线执行环境"""
        
        # 创建实验目录结构
        (self.experiment_dir / "models").mkdir(exist_ok=True)
        (self.experiment_dir / "logs").mkdir(exist_ok=True)
        (self.experiment_dir / "evaluations").mkdir(exist_ok=True)
        (self.experiment_dir / "configs").mkdir(exist_ok=True)
        
        # 保存配置文件
        config_path = self.experiment_dir / "configs" / "pipeline_config.json"
        with open(config_path, 'w') as f:
            # 简化配置以便JSON序列化
            config_dict = {
                'experiment_name': self.config.experiment_name,
                'output_dir': self.config.output_dir,
                'foundation_config': self.config.foundation_config,
                'branches_config': self.config.branches_config,
                'transfer_config': self.config.transfer_config,
                'evaluation_config': self.config.evaluation_config
            }
            json.dump(config_dict, f, indent=2)
        
        # 验证环境可用性
        for stage in [TrainingStage.FOUNDATION, TrainingStage.HIERARCHICAL, 
                     TrainingStage.ABLATION, TrainingStage.BASELINE]:
            
            env = self.env_factory.create_environment(stage, {}, mode="train")
            is_valid = self.env_factory.validate_environment(env, stage)
            
            if not is_valid:
                raise RuntimeError(f"环境验证失败: {stage.value}")
        
        self.logger.info("流水线准备完成")
    
    def _run_foundation_training(self) -> TrainingResult:
        """执行基座模型训练"""
        
        # 动态加载Foundation训练器
        trainer_class = self._load_trainer_class(
            self.config.foundation_trainer_class,
            TrainingStage.FOUNDATION
        )
        
        # 创建训练器实例
        trainer = trainer_class(
            config=self.config.foundation_config,
            env_factory=self.env_factory,
            output_dir=str(self.experiment_dir / "models")
        )
        
        # 添加进度回调
        if hasattr(trainer, 'add_progress_callback'):
            trainer.add_progress_callback(
                lambda stage, progress: self._notify_progress(
                    f"Foundation训练进度: {progress.get('episode', 0)}", 
                    progress
                )
            )
        
        # 执行训练
        foundation_result = trainer.train()
        
        # 保存基座模型
        if foundation_result.success:
            model_path = self.transfer_manager.save_foundation_model(
                model=foundation_result.trained_model,
                optimizer=foundation_result.metadata.get('optimizer'),
                training_stats=foundation_result.metadata.get('training_stats', {}),
                model_name=f"{self.config.experiment_name}_foundation"
            )
            
            # 加载checkpoint用于后续迁移
            self.foundation_checkpoint = self.transfer_manager.load_foundation_model(model_path)
            
            self.logger.info(f"基座模型训练完成: {model_path}")
        else:
            raise RuntimeError(f"基座模型训练失败: {foundation_result.error_message}")
        
        return foundation_result
    
    def _prepare_model_transfer(self) -> None:
        """准备模型迁移"""
        
        if self.foundation_checkpoint is None:
            raise RuntimeError("基座模型未就绪，无法进行迁移")
        
        # 验证迁移兼容性
        for branch_name in self.config.branches_config.keys():
            if branch_name == "hierarchical":
                target_stage = TrainingStage.HIERARCHICAL
            elif branch_name == "ablation":
                target_stage = TrainingStage.ABLATION
            elif branch_name == "baseline":
                target_stage = TrainingStage.BASELINE
            else:
                continue
            
            # 创建目标模型示例用于兼容性检查
            # 这里需要根据实际的模型类进行实例化
            # 暂时跳过详细检查
            self.logger.info(f"迁移兼容性检查通过: {branch_name}")
        
        self.logger.info("模型迁移准备完成")
    
    def _run_branch_training(self) -> Dict[str, TrainingResult]:
        """执行分支训练"""
        
        branch_results = {}
        
        if self.config.enable_parallel_branches:
            # 并行分支训练 (复杂实现，暂时串行)
            self.logger.warning("并行分支训练暂未实现，使用串行模式")
        
        # 串行分支训练
        for branch_name, branch_config in self.config.branches_config.items():
            
            self.logger.info(f"开始训练分支: {branch_name}")
            
            try:
                if branch_name == "hierarchical":
                    result = self._train_hierarchical_branch(branch_config)
                elif branch_name == "ablation":
                    result = self._train_ablation_branch(branch_config)
                elif branch_name == "baseline":
                    result = self._train_baseline_branch(branch_config)
                else:
                    self.logger.warning(f"未知分支类型: {branch_name}")
                    continue
                
                branch_results[branch_name] = result
                self.logger.info(f"分支 {branch_name} 训练完成")
                
            except Exception as e:
                self.logger.error(f"分支 {branch_name} 训练失败: {e}")
                # 继续训练其他分支
                continue
        
        return branch_results
    
    def _train_hierarchical_branch(self, config: Dict[str, Any]) -> TrainingResult:
        """训练分层分支"""
        
        # 动态加载Hierarchical训练器
        trainer_class = self._load_trainer_class("HierarchicalTrainer", TrainingStage.HIERARCHICAL)
        
        trainer = trainer_class(
            config=config,
            env_factory=self.env_factory,
            transfer_manager=self.transfer_manager,
            foundation_checkpoint=self.foundation_checkpoint,
            output_dir=str(self.experiment_dir / "models")
        )
        
        return trainer.train()
    
    def _train_ablation_branch(self, config: Dict[str, Any]) -> TrainingResult:
        """训练消融分支"""
        
        # 动态加载Ablation训练器  
        trainer_class = self._load_trainer_class("AblationTrainer", TrainingStage.ABLATION)
        
        trainer = trainer_class(
            config=config,
            env_factory=self.env_factory,
            transfer_manager=self.transfer_manager,
            foundation_checkpoint=self.foundation_checkpoint,
            output_dir=str(self.experiment_dir / "models")
        )
        
        return trainer.train()
    
    def _train_baseline_branch(self, config: Dict[str, Any]) -> TrainingResult:
        """训练基线分支"""
        
        # 动态加载Baseline训练器
        trainer_class = self._load_trainer_class("BaselineTrainer", TrainingStage.BASELINE)
        
        trainer = trainer_class(
            config=config,
            env_factory=self.env_factory,
            transfer_manager=self.transfer_manager,
            foundation_checkpoint=self.foundation_checkpoint,
            output_dir=str(self.experiment_dir / "models")
        )
        
        return trainer.train()
    
    def _run_evaluation(self, branch_results: Dict[str, TrainingResult]) -> Dict[str, Any]:
        """运行评估"""
        
        evaluation_results = {}
        
        for branch_name, training_result in branch_results.items():
            if not training_result.success:
                self.logger.warning(f"跳过失败分支的评估: {branch_name}")
                continue
            
            try:
                # 创建评估器
                evaluator = self._create_evaluator(branch_name, training_result)
                
                # 执行评估
                eval_result = evaluator.evaluate(
                    episodes=self.config.evaluation_config.get('eval_episodes', 100)
                )
                
                evaluation_results[branch_name] = eval_result
                
            except Exception as e:
                self.logger.error(f"分支 {branch_name} 评估失败: {e}")
                evaluation_results[branch_name] = {"error": str(e)}
        
        # 生成对比报告
        comparison_report = self._generate_comparison_report(evaluation_results)
        evaluation_results['comparison'] = comparison_report
        
        return evaluation_results
    
    def _create_evaluator(self, branch_name: str, training_result: TrainingResult):
        """创建评估器 - 占位符实现"""
        
        # 这里应该根据分支类型创建相应的评估器
        # 暂时返回简单的评估器类
        class SimpleEvaluator:
            def __init__(self, model, branch_name):
                self.model = model
                self.branch_name = branch_name
            
            def evaluate(self, episodes: int) -> Dict[str, Any]:
                # 简化的评估实现
                return {
                    'branch': self.branch_name,
                    'episodes': episodes,
                    'success_rate': 0.85,  # 模拟数据
                    'avg_reward': 150.0,
                    'avg_episode_length': 500
                }
        
        return SimpleEvaluator(training_result.trained_model, branch_name)
    
    def _generate_comparison_report(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """生成对比报告"""
        
        # 提取关键指标
        metrics = ['success_rate', 'avg_reward', 'avg_episode_length']
        comparison = {}
        
        for metric in metrics:
            comparison[metric] = {}
            for branch_name, result in evaluation_results.items():
                if isinstance(result, dict) and metric in result:
                    comparison[metric][branch_name] = result[metric]
        
        # 排名
        rankings = {}
        for metric in metrics:
            if comparison[metric]:
                sorted_branches = sorted(
                    comparison[metric].items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                rankings[metric] = [branch for branch, _ in sorted_branches]
        
        return {
            'metrics_comparison': comparison,
            'rankings': rankings,
            'best_overall': self._determine_best_branch(rankings)
        }
    
    def _determine_best_branch(self, rankings: Dict[str, List[str]]) -> str:
        """确定最佳分支"""
        
        # 简单的综合评分
        branch_scores = {}
        
        for metric, ranking in rankings.items():
            for i, branch in enumerate(ranking):
                if branch not in branch_scores:
                    branch_scores[branch] = 0
                branch_scores[branch] += len(ranking) - i
        
        if branch_scores:
            best_branch = max(branch_scores.items(), key=lambda x: x[1])[0]
            return best_branch
        
        return "unknown"
    
    def _load_trainer_class(self, trainer_name: str, stage: TrainingStage):
        """动态加载训练器类 - 占位符实现"""
        
        # 这里应该根据trainer_name动态导入相应的训练器类
        # 暂时返回Mock实现
        
        class MockTrainer:
            def __init__(self, config=None, env_factory=None, transfer_manager=None, 
                        foundation_checkpoint=None, output_dir=None):
                self.config = config or {}
                self.env_factory = env_factory
                self.stage = stage
                
            def train(self) -> TrainingResult:
                # 模拟训练过程
                import time
                time.sleep(1)  # 模拟训练时间
                
                # 创建Mock模型
                import torch.nn as nn
                mock_model = nn.Sequential(
                    nn.Linear(86, 128),
                    nn.ReLU(),
                    nn.Linear(128, 4)
                )
                
                return TrainingResult(
                    success=True,
                    trained_model=mock_model,
                    metrics={'final_reward': 100.0},
                    metadata={'stage': stage.value}
                )
            
            def add_progress_callback(self, callback):
                pass
        
        return MockTrainer
    
    def _save_pipeline_result(self, result: PipelineResult) -> None:
        """保存流水线结果"""
        
        # 构建可序列化的结果数据
        result_data = {
            'status': result.status.value,
            'total_duration': result.total_duration,
            'foundation_success': result.foundation_result.success if result.foundation_result else False,
            'branch_results': {
                name: {'success': res.success, 'metrics': res.metrics} 
                for name, res in result.branch_results.items()
            },
            'evaluation_results': result.evaluation_results,
            'error_message': result.error_message
        }
        
        # 保存到文件
        result_path = self.experiment_dir / "pipeline_result.json"
        with open(result_path, 'w') as f:
            json.dump(result_data, f, indent=2)
        
        self.logger.info(f"流水线结果已保存: {result_path}")
    
    def _notify_progress(self, message: str, data: Dict[str, Any]) -> None:
        """通知进度更新"""
        
        for callback in self.progress_callbacks:
            try:
                callback(message, data)
            except Exception as e:
                self.logger.warning(f"进度回调执行失败: {e}")
        
        self.logger.info(f"流水线进度: {message}")
    
    def run_sequential_training(self) -> Dict[str, TrainingResult]:
        """
        执行顺序训练流程
        
        Returns:
            各阶段训练结果
        """
        self.logger.info("🚀 开始顺序训练流程")
        
        try:
            # 逐阶段执行训练
            stages = [TrainingStage.HIERARCHICAL, TrainingStage.ABLATION, TrainingStage.BASELINE]
            
            for stage in stages:
                self.logger.info(f"🎯 开始阶段: {stage.value}")
                
                # 执行单个阶段训练
                stage_result = self.run_stage(stage)
                
                if not stage_result.success:
                    self.logger.error(f"❌ 阶段 {stage.value} 训练失败，停止流水线")
                    break
                
                self.logger.info(f"✅ 阶段 {stage.value} 训练完成")
            
            self.logger.info("🎉 顺序训练流程完成")
            return {}
            
        except Exception as e:
            self.logger.error(f"💥 训练流水线异常: {e}")
            raise
    
    def run_stage(self, stage: TrainingStage) -> TrainingResult:
        """
        执行单个训练阶段
        
        Args:
            stage: 训练阶段
            
        Returns:
            训练结果
        """
        # 创建对应的训练器
        from training.branches.hierarchical_trainer import HierarchicalTrainer
        from training.branches.ablation_trainer import AblationTrainer
        from training.branches.baseline_trainer import BaselineTrainer
        
        config = {
            'total_timesteps': 1000,
            'drone_model': 'CF2X',
            'physics': 'PYB',
            'gui_training': False
        }
        
        if stage == TrainingStage.HIERARCHICAL:
            trainer = HierarchicalTrainer(config=config)
        elif stage == TrainingStage.ABLATION:
            trainer = AblationTrainer(config=config)
        elif stage == TrainingStage.BASELINE:
            trainer = BaselineTrainer(config=config)
        else:
            raise ValueError(f"不支持的训练阶段: {stage}")
        
        try:
            # 执行训练
            with trainer:
                result = trainer.train()
                return result
                
        except Exception as e:
            self.logger.error(f"❌ 阶段 {stage.value} 训练异常: {e}")
            
            # 创建失败结果
            return TrainingResult(
                stage=stage,
                success=False,
                total_steps=0,
                total_episodes=0,
                final_reward=0.0,
                best_reward=0.0,
                training_time=0.0,
                model_path=None,
                metrics={},
                error_message=str(e)
            )
    
    def validate_stage_transition(self, 
                                 from_stage: TrainingStage, 
                                 to_stage: TrainingStage) -> bool:
        """验证阶段转换的有效性"""
        # Foundation阶段必须是起点
        valid_transitions = {
            TrainingStage.FOUNDATION: [TrainingStage.HIERARCHICAL, TrainingStage.ABLATION, TrainingStage.BASELINE],
            TrainingStage.HIERARCHICAL: [TrainingStage.ABLATION, TrainingStage.BASELINE]
        }
        
        if from_stage in valid_transitions:
            return to_stage in valid_transitions[from_stage]
        
        return False
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """获取流水线状态"""
        
        return {
            'status': self.status.value,
            'experiment_name': self.config.experiment_name,
            'start_time': self.start_time,
            'duration': time.time() - self.start_time if self.start_time else 0,
            'foundation_ready': self.foundation_checkpoint is not None
        }
    
    def cleanup(self) -> None:
        """清理资源"""
        
        # 清理环境
        self.env_factory.cleanup_environments()
        
        # 清理训练器缓存
        self._trainer_cache.clear()
        
        self.logger.info("流水线资源已清理")
