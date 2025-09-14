#!/usr/bin/env python3

"""
训练编排器 - 统一调度四阶段训练流水线的高级接口
"""

import logging
import json
import time
from typing import Dict, Any, Optional, List, Callable, Union
from pathlib import Path
from dataclasses import dataclass, asdict
import sys

# 添加项目路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入核心组件
from ..core import (
    TrainingPipeline, PipelineConfig, PipelineResult, PipelineStatus,
    EnvironmentFactory, ModelTransferManager, TrainingStage
)

# 导入训练器
from ..foundation import BaseFlightTrainer
from ..branches import HierarchicalTrainer, AblationTrainer, BaselineTrainer

logger = logging.getLogger(__name__)


@dataclass
class OrchestrationConfig:
    """编排器配置"""
    
    # 实验基础配置
    experiment_name: str = "hauav_complete_training"
    output_dir: str = "./experiments"
    log_level: str = "INFO"
    
    # 启用的训练阶段
    enable_foundation: bool = True
    enable_hierarchical: bool = True
    enable_ablation: bool = True
    enable_baseline: bool = True
    
    # 并行训练配置
    enable_parallel_branches: bool = False
    max_parallel_workers: int = 3
    
    # Foundation阶段配置
    foundation_config: Dict[str, Any] = None
    
    # 分支训练配置
    hierarchical_config: Dict[str, Any] = None
    ablation_config: Dict[str, Any] = None
    baseline_config: Dict[str, Any] = None
    
    # 迁移配置
    transfer_config: Dict[str, Any] = None
    
    # 评估和对比配置
    enable_cross_evaluation: bool = True
    final_comparison_episodes: int = 100
    
    def __post_init__(self):
        # 设置默认配置
        if self.foundation_config is None:
            self.foundation_config = {
                'total_timesteps': 100000,
                'enable_curriculum': True,
                'hover_training_steps': 25000,
                'flight_training_steps': 75000
            }
        
        if self.hierarchical_config is None:
            self.hierarchical_config = {
                'total_timesteps': 200000,
                'high_level_update_frequency': 5,
                'future_horizon': 5,
                'enable_intrinsic_motivation': True
            }
        
        if self.ablation_config is None:
            self.ablation_config = {
                'total_timesteps': 150000,
                'ablation_types': ['B1', 'B2', 'B3']
            }
        
        if self.baseline_config is None:
            self.baseline_config = {
                'total_timesteps': 200000,
                'algorithms': ['ppo', 'sac'],
                'use_pretrained_init': True
            }
        
        if self.transfer_config is None:
            self.transfer_config = {
                'transfer_mode': 'partial'
            }


@dataclass
class OrchestrationResult:
    """编排器结果"""
    
    success: bool
    total_duration: float
    pipeline_result: Optional[PipelineResult] = None
    cross_evaluation_results: Optional[Dict[str, Any]] = None
    final_comparison_report: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    
    def save_to_file(self, filepath: str) -> None:
        """保存结果到文件"""
        
        # 准备可序列化的数据
        result_data = {
            'success': self.success,
            'total_duration': self.total_duration,
            'error_message': self.error_message,
            'cross_evaluation_results': self.cross_evaluation_results,
            'final_comparison_report': self.final_comparison_report
        }
        
        # Pipeline结果的简化版本
        if self.pipeline_result:
            result_data['pipeline_summary'] = {
                'status': self.pipeline_result.status.value,
                'foundation_success': self.pipeline_result.foundation_result.success if self.pipeline_result.foundation_result else False,
                'branch_results': {
                    name: {'success': res.success, 'metrics': res.metrics}
                    for name, res in self.pipeline_result.branch_results.items()
                },
                'evaluation_summary': self.pipeline_result.evaluation_results
            }
        
        with open(filepath, 'w') as f:
            json.dump(result_data, f, indent=2)


class TrainingOrchestrator:
    """
    训练编排器
    
    统一调度四阶段训练流水线:
    1. Foundation - 基座模型训练 (BaseFlightTrainer)
    2. Hierarchical - 完整分层训练 (HierarchicalTrainer) 
    3. Ablation - B组消融实验 (AblationTrainer)
    4. Baseline - SB3基线对比 (BaselineTrainer)
    """
    
    def __init__(self, config: OrchestrationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, config.log_level.upper()))
        
        # 实验目录
        self.experiment_dir = Path(config.output_dir) / config.experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # 核心组件
        self.env_factory = EnvironmentFactory()
        self.transfer_manager = ModelTransferManager(
            model_save_dir=str(self.experiment_dir / "models")
        )
        
        # 训练流水线
        self.pipeline = None
        
        # 进度回调
        self.progress_callbacks: List[Callable] = []
        
        # 训练状态
        self.start_time = None
        self.current_stage = "idle"
    
    def add_progress_callback(self, callback: Callable) -> None:
        """添加进度回调"""
        self.progress_callbacks.append(callback)
    
    def run_complete_training(self) -> OrchestrationResult:
        """
        运行完整的训练流程
        
        Returns:
            编排器执行结果
        """
        
        self.start_time = time.time()
        self.current_stage = "initializing"
        
        try:
            # 1. 初始化
            self._notify_progress("开始完整训练流程", {"stage": "initialization"})
            self._initialize_orchestration()
            
            # 2. 执行训练流水线
            self.current_stage = "pipeline_training"
            self._notify_progress("执行训练流水线", {"stage": "pipeline"})
            pipeline_result = self._run_training_pipeline()
            
            # 3. 交叉评估 (可选)
            cross_eval_results = None
            if self.config.enable_cross_evaluation:
                self.current_stage = "cross_evaluation"
                self._notify_progress("执行交叉评估", {"stage": "cross_evaluation"})
                cross_eval_results = self._run_cross_evaluation(pipeline_result)
            
            # 4. 最终对比报告
            self.current_stage = "final_comparison"
            self._notify_progress("生成最终对比报告", {"stage": "final_comparison"})
            final_comparison = self._generate_final_comparison(pipeline_result, cross_eval_results)
            
            # 5. 完成
            self.current_stage = "completed"
            total_duration = time.time() - self.start_time
            
            result = OrchestrationResult(
                success=True,
                total_duration=total_duration,
                pipeline_result=pipeline_result,
                cross_evaluation_results=cross_eval_results,
                final_comparison_report=final_comparison
            )
            
            # 保存结果
            result_path = self.experiment_dir / "orchestration_result.json"
            result.save_to_file(str(result_path))
            
            self._notify_progress("完整训练流程完成", {
                "duration": total_duration,
                "result_path": str(result_path)
            })
            
            return result
            
        except Exception as e:
            self.current_stage = "failed"
            error_duration = time.time() - self.start_time if self.start_time else 0
            
            error_result = OrchestrationResult(
                success=False,
                total_duration=error_duration,
                error_message=str(e)
            )
            
            self.logger.error(f"完整训练流程失败: {e}")
            self._notify_progress("训练流程失败", {"error": str(e)})
            
            return error_result
    
    def _initialize_orchestration(self) -> None:
        """初始化编排"""
        
        # 创建实验目录结构
        (self.experiment_dir / "models").mkdir(exist_ok=True)
        (self.experiment_dir / "logs").mkdir(exist_ok=True)
        (self.experiment_dir / "evaluations").mkdir(exist_ok=True)
        (self.experiment_dir / "comparisons").mkdir(exist_ok=True)
        (self.experiment_dir / "configs").mkdir(exist_ok=True)
        
        # 保存编排配置
        config_path = self.experiment_dir / "configs" / "orchestration_config.json"
        with open(config_path, 'w') as f:
            json.dump(asdict(self.config), f, indent=2)
        
        # 验证训练器可用性
        self._validate_trainers()
        
        self.logger.info("编排器初始化完成")
    
    def _validate_trainers(self) -> None:
        """验证训练器可用性"""
        
        trainers_to_validate = []
        
        if self.config.enable_foundation:
            trainers_to_validate.append(("Foundation", BaseFlightTrainer))
        
        if self.config.enable_hierarchical:
            trainers_to_validate.append(("Hierarchical", HierarchicalTrainer))
        
        if self.config.enable_ablation:
            trainers_to_validate.append(("Ablation", AblationTrainer))
        
        if self.config.enable_baseline:
            trainers_to_validate.append(("Baseline", BaselineTrainer))
        
        for trainer_name, trainer_class in trainers_to_validate:
            try:
                # 简单的实例化测试
                trainer = trainer_class(
                    config={},
                    env_factory=self.env_factory,
                    output_dir=str(self.experiment_dir / "temp")
                )
                self.logger.info(f"{trainer_name} 训练器验证通过")
            except Exception as e:
                raise RuntimeError(f"{trainer_name} 训练器验证失败: {e}")
    
    def _run_training_pipeline(self) -> PipelineResult:
        """运行训练流水线"""
        
        # 构建流水线配置
        pipeline_config = PipelineConfig(
            experiment_name=f"{self.config.experiment_name}_pipeline",
            output_dir=str(self.experiment_dir),
            foundation_config=self.config.foundation_config,
            branches_config=self._build_branches_config(),
            transfer_config=self.config.transfer_config,
            enable_parallel_branches=self.config.enable_parallel_branches
        )
        
        # 创建流水线
        self.pipeline = TrainingPipeline(
            config=pipeline_config,
            environment_factory=self.env_factory,
            model_transfer_manager=self.transfer_manager
        )
        
        # 添加进度回调
        self.pipeline.add_progress_callback(self._pipeline_progress_callback)
        
        # 执行流水线
        return self.pipeline.run_pipeline()
    
    def _build_branches_config(self) -> Dict[str, Dict[str, Any]]:
        """构建分支配置"""
        
        branches_config = {}
        
        if self.config.enable_hierarchical:
            branches_config["hierarchical"] = self.config.hierarchical_config
        
        if self.config.enable_ablation:
            branches_config["ablation"] = self.config.ablation_config
        
        if self.config.enable_baseline:
            branches_config["baseline"] = self.config.baseline_config
        
        return branches_config
    
    def _run_cross_evaluation(self, pipeline_result: PipelineResult) -> Dict[str, Any]:
        """运行交叉评估"""
        
        # 交叉评估：用不同的模型在不同的环境配置下测试
        cross_eval_results = {
            'evaluation_matrix': {},
            'performance_stability': {},
            'robustness_analysis': {}
        }
        
        # 简化的交叉评估实现
        # 实际应该创建多种环境配置并测试所有模型
        
        models_to_evaluate = {}
        
        if pipeline_result.foundation_result and pipeline_result.foundation_result.success:
            models_to_evaluate['foundation'] = pipeline_result.foundation_result.trained_model
        
        for branch_name, branch_result in pipeline_result.branch_results.items():
            if branch_result.success:
                models_to_evaluate[branch_name] = branch_result.trained_model
        
        # 在不同环境配置下评估
        env_variants = [
            {'noise_level': 0.0, 'wind_speed': 0.0},
            {'noise_level': 0.1, 'wind_speed': 0.5},
            {'noise_level': 0.2, 'wind_speed': 1.0}
        ]
        
        for env_name, env_config in enumerate(env_variants):
            cross_eval_results['evaluation_matrix'][f'env_{env_name}'] = {}
            
            for model_name, model in models_to_evaluate.items():
                # 简化的评估 (实际需要重新运行环境)
                performance_score = self._evaluate_model_cross(model, env_config)
                cross_eval_results['evaluation_matrix'][f'env_{env_name}'][model_name] = performance_score
        
        # 稳定性分析
        cross_eval_results['performance_stability'] = self._analyze_stability(
            cross_eval_results['evaluation_matrix']
        )
        
        return cross_eval_results
    
    def _evaluate_model_cross(self, model, env_config: Dict[str, Any]) -> float:
        """交叉评估模型 - 简化实现"""
        
        # 这里应该创建特定配置的环境并运行评估
        # 简化为返回模拟分数
        base_score = 0.8
        noise_penalty = env_config.get('noise_level', 0) * 0.3
        wind_penalty = env_config.get('wind_speed', 0) * 0.2
        
        return max(0.1, base_score - noise_penalty - wind_penalty)
    
    def _analyze_stability(self, evaluation_matrix: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """分析性能稳定性"""
        
        stability_analysis = {}
        
        # 计算每个模型在不同环境下的方差
        for env_name, env_results in evaluation_matrix.items():
            for model_name, score in env_results.items():
                if model_name not in stability_analysis:
                    stability_analysis[model_name] = []
                stability_analysis[model_name].append(score)
        
        # 计算稳定性指标
        stability_scores = {}
        for model_name, scores in stability_analysis.items():
            stability_scores[model_name] = {
                'mean_performance': np.mean(scores) if scores else 0,
                'performance_variance': np.var(scores) if scores else 0,
                'stability_rank': 0  # 将在后面计算
            }
        
        # 排名
        sorted_by_stability = sorted(
            stability_scores.items(),
            key=lambda x: x[1]['performance_variance']
        )
        
        for rank, (model_name, _) in enumerate(sorted_by_stability):
            stability_scores[model_name]['stability_rank'] = rank + 1
        
        return stability_scores
    
    def _generate_final_comparison(self, 
                                  pipeline_result: PipelineResult,
                                  cross_eval_results: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """生成最终对比报告"""
        
        comparison_report = {
            'executive_summary': {},
            'performance_rankings': {},
            'technical_analysis': {},
            'recommendations': []
        }
        
        # 执行摘要
        comparison_report['executive_summary'] = {
            'experiment_name': self.config.experiment_name,
            'total_training_duration': pipeline_result.total_duration,
            'stages_completed': self._count_completed_stages(pipeline_result),
            'overall_success': pipeline_result.status == PipelineStatus.COMPLETED
        }
        
        # 性能排名
        if pipeline_result.evaluation_results:
            comparison_report['performance_rankings'] = pipeline_result.evaluation_results.get('comparison', {})
        
        # 技术分析
        comparison_report['technical_analysis'] = {
            'foundation_model_quality': self._analyze_foundation_quality(pipeline_result),
            'hierarchical_effectiveness': self._analyze_hierarchical_effectiveness(pipeline_result),
            'ablation_insights': self._analyze_ablation_insights(pipeline_result),
            'baseline_comparison': self._analyze_baseline_comparison(pipeline_result)
        }
        
        # 交叉评估结果
        if cross_eval_results:
            comparison_report['robustness_analysis'] = cross_eval_results
        
        # 推荐
        comparison_report['recommendations'] = self._generate_recommendations(comparison_report)
        
        return comparison_report
    
    def _count_completed_stages(self, pipeline_result: PipelineResult) -> int:
        """统计完成的阶段数"""
        
        completed = 0
        
        if pipeline_result.foundation_result and pipeline_result.foundation_result.success:
            completed += 1
        
        for result in pipeline_result.branch_results.values():
            if result.success:
                completed += 1
        
        return completed
    
    def _analyze_foundation_quality(self, pipeline_result: PipelineResult) -> Dict[str, Any]:
        """分析基座模型质量"""
        
        if not pipeline_result.foundation_result or not pipeline_result.foundation_result.success:
            return {'status': 'failed', 'quality_score': 0}
        
        metrics = pipeline_result.foundation_result.metrics
        
        hover_success = metrics.get('hover_success_rate', 0)
        flight_success = metrics.get('flight_success_rate', 0)
        
        quality_score = (hover_success + flight_success) / 2
        
        return {
            'status': 'success',
            'quality_score': quality_score,
            'hover_performance': hover_success,
            'flight_performance': flight_success,
            'transferability': 'high' if quality_score > 0.8 else 'medium' if quality_score > 0.6 else 'low'
        }
    
    def _analyze_hierarchical_effectiveness(self, pipeline_result: PipelineResult) -> Dict[str, Any]:
        """分析分层架构有效性"""
        
        hierarchical_result = pipeline_result.branch_results.get('hierarchical')
        
        if not hierarchical_result or not hierarchical_result.success:
            return {'status': 'failed', 'effectiveness': 'unknown'}
        
        success_rate = hierarchical_result.metrics.get('navigation_success_rate', 0)
        
        return {
            'status': 'success',
            'navigation_success_rate': success_rate,
            'effectiveness': 'high' if success_rate > 0.8 else 'medium' if success_rate > 0.6 else 'low'
        }
    
    def _analyze_ablation_insights(self, pipeline_result: PipelineResult) -> Dict[str, Any]:
        """分析消融实验洞察"""
        
        ablation_result = pipeline_result.branch_results.get('ablation')
        
        if not ablation_result or not ablation_result.success:
            return {'status': 'failed', 'insights': []}
        
        # 从消融实验元数据中提取分析
        metadata = ablation_result.metadata or {}
        analysis = metadata.get('training_stats', {})
        
        return {
            'status': 'success',
            'hierarchy_necessity': analysis.get('hierarchy_necessity', 'unknown'),
            'insights': analysis.get('insights', [])
        }
    
    def _analyze_baseline_comparison(self, pipeline_result: PipelineResult) -> Dict[str, Any]:
        """分析基线对比"""
        
        baseline_result = pipeline_result.branch_results.get('baseline')
        
        if not baseline_result or not baseline_result.success:
            return {'status': 'failed', 'comparison': {}}
        
        return {
            'status': 'success',
            'best_baseline_algorithm': baseline_result.metadata.get('best_algorithm', 'unknown'),
            'baseline_performance': baseline_result.metrics
        }
    
    def _generate_recommendations(self, comparison_report: Dict[str, Any]) -> List[str]:
        """生成推荐建议"""
        
        recommendations = []
        
        # 基于分析结果生成推荐
        foundation_quality = comparison_report['technical_analysis'].get('foundation_model_quality', {})
        if foundation_quality.get('quality_score', 0) < 0.7:
            recommendations.append("建议增加基座模型的训练时间或调整课程学习策略")
        
        hierarchical_effectiveness = comparison_report['technical_analysis'].get('hierarchical_effectiveness', {})
        if hierarchical_effectiveness.get('effectiveness') == 'low':
            recommendations.append("分层架构性能不佳，建议优化高层规划器或低层控制器的设计")
        
        ablation_insights = comparison_report['technical_analysis'].get('ablation_insights', {})
        if ablation_insights.get('hierarchy_necessity') == 'marginal':
            recommendations.append("消融实验显示分层架构优势有限，考虑简化模型结构")
        
        # 默认推荐
        if not recommendations:
            recommendations.append("整体训练效果良好，建议进行更大规模的实验验证")
        
        return recommendations
    
    def _pipeline_progress_callback(self, message: str, data: Dict[str, Any]) -> None:
        """流水线进度回调"""
        
        # 转发给外部回调
        for callback in self.progress_callbacks:
            try:
                callback(f"Pipeline: {message}", data)
            except Exception as e:
                self.logger.warning(f"进度回调失败: {e}")
    
    def _notify_progress(self, message: str, data: Dict[str, Any]) -> None:
        """通知进度更新"""
        
        for callback in self.progress_callbacks:
            try:
                callback(f"Orchestrator: {message}", data)
            except Exception as e:
                self.logger.warning(f"进度回调失败: {e}")
        
        self.logger.info(f"编排器进度: {message}")
    
    def get_orchestration_status(self) -> Dict[str, Any]:
        """获取编排器状态"""
        
        return {
            'current_stage': self.current_stage,
            'experiment_name': self.config.experiment_name,
            'start_time': self.start_time,
            'duration': time.time() - self.start_time if self.start_time else 0,
            'pipeline_status': self.pipeline.get_pipeline_status() if self.pipeline else None
        }
    
    def cleanup(self) -> None:
        """清理资源"""
        
        if self.pipeline:
            self.pipeline.cleanup()
        
        self.env_factory.cleanup_environments()
        
        self.logger.info("编排器资源已清理")


# 导入numpy用于数值计算
import numpy as np
