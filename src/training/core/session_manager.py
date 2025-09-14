#!/usr/bin/env python3

"""
智能会话管理器 - 基于训练阶段的目录管理和数据链路优化
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import asdict
import numpy as np

from src.training.core.base_trainer import TrainingStage


class SessionManager:
    """
    智能会话管理器 - 根据训练阶段创建特定目录结构
    
    目录命名规则:
    - Foundation: train_foundation_{时间戳}
    - Hierarchical: train_hierarchical_{时间戳}
    - Ablation: train_ablation_B{1/2/3}_{时间戳}
    - Baseline: train_baseline_{算法名}_{时间戳}
    """
    
    # 阶段到目录名的映射
    STAGE_DIR_MAPPING = {
        TrainingStage.FOUNDATION: "foundation",
        TrainingStage.HIERARCHICAL: "hierarchical", 
        TrainingStage.ABLATION: "ablation",
        TrainingStage.BASELINE: "baseline"
    }
    
    # 阶段标准化子目录结构
    STANDARD_SUBDIRS = [
        "Config",           # 配置文件
        "Model",            # 模型保存
        "Result",           # 结果数据
        "Checkpoints",      # 训练检查点
        "Logs",             # 文本日志
    ]
    
    # 可选子目录（根据配置启用）
    OPTIONAL_SUBDIRS = {
        'trajectory': ["Trajectory/train", "Trajectory/eval"],
        'tensorboard': ["Tensorboard/train", "Tensorboard/eval"],
        'visualization': ["Plot", "Visualization"],
        'pointcloud': ["PointCloud"],
        'analysis': ["Analysis", "Comparison"]
    }
    
    def __init__(self, 
                 experiment_name: str,
                 training_stage: TrainingStage,
                 config: Dict[str, Any],
                 base_log_dir: str = "logs",
                 stage_variant: Optional[str] = None,
                 enable_trajectory: bool = True,
                 enable_tensorboard: bool = True,
                 enable_visualization: bool = True,
                 enable_pointcloud: bool = False,
                 enable_analysis: bool = False):
        """
        初始化智能会话管理器
        
        Args:
            experiment_name: 实验名称
            training_stage: 训练阶段
            config: 训练配置字典
            base_log_dir: 基础日志目录
            stage_variant: 阶段变体（如B1/B2/B3、ppo/sac等）
            enable_*: 各种功能模块的启用标志
        """
        self.experiment_name = experiment_name
        self.training_stage = training_stage
        self.config = config
        self.base_log_dir = Path(base_log_dir)
        self.stage_variant = stage_variant
        
        # 功能开关
        self.feature_flags = {
            'trajectory': enable_trajectory,
            'tensorboard': enable_tensorboard,
            'visualization': enable_visualization,
            'pointcloud': enable_pointcloud,
            'analysis': enable_analysis
        }
        
        self.logger = logging.getLogger(__name__)
        
        # 创建智能目录结构
        self.session_dir = self._build_intelligent_directory()
        
        # 初始化数据管理器
        self._initialize_data_managers()
        
        # 保存会话配置
        self._save_session_metadata()
        
        self.logger.info(f"✅ 智能会话创建完成: {self.session_dir}")
        self._log_session_summary()
    
    def _build_intelligent_directory(self) -> Path:
        """
        构建智能的目录结构
        
        目录命名策略:
        - train_{stage}_{variant}_{timestamp}
        - 如: train_hierarchical_20250820_143022
        - 如: train_ablation_B1_20250820_143022
        - 如: train_baseline_ppo_20250820_143022
        """
        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 构建目录名 - 直接使用训练阶段值
        if self.stage_variant:
            session_name = f"train_{self.training_stage.value}_{self.stage_variant}_{timestamp}"
        else:
            session_name = f"train_{self.training_stage.value}_{timestamp}"
        
        session_dir = self.base_log_dir / session_name
        
        # 创建标准子目录
        self._create_subdirectories(session_dir)
        
        return session_dir
    
    def _create_subdirectories(self, session_dir: Path):
        """创建子目录结构"""
        # 创建标准目录
        for subdir in self.STANDARD_SUBDIRS:
            (session_dir / subdir).mkdir(parents=True, exist_ok=True)
        
        # 根据功能开关创建可选目录
        for feature, enabled in self.feature_flags.items():
            if enabled and feature in self.OPTIONAL_SUBDIRS:
                for subdir in self.OPTIONAL_SUBDIRS[feature]:
                    (session_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    def _initialize_data_managers(self):
        """初始化数据管理器"""
        self.data_managers = {}
        
        # 轨迹管理器
        if self.feature_flags['trajectory']:
            try:
                self._init_trajectory_managers()
            except ImportError:
                self.logger.warning("TrajectoryManager不可用，轨迹记录功能已禁用")
                self.feature_flags['trajectory'] = False
        
        # TensorBoard日志管理器
        if self.feature_flags['tensorboard']:
            self._init_tensorboard_paths()
        
        # 可视化数据管理器
        if self.feature_flags['visualization']:
            self._init_visualization_paths()
    
    def _init_trajectory_managers(self):
        """初始化轨迹管理器"""
        from src.utils.trajectory_manager import TrajectoryManager
        
        self.data_managers['trajectory'] = {
            'train': TrajectoryManager(
                session_dir=self.session_dir,
                mode="train"
            ),
            'eval': TrajectoryManager(
                session_dir=self.session_dir,
                mode="eval"
            )
        }
    
    def _init_tensorboard_paths(self):
        """初始化TensorBoard路径"""
        self.data_managers['tensorboard'] = {
            'train': self.session_dir / "Tensorboard" / "train",
            'eval': self.session_dir / "Tensorboard" / "eval"
        }
    
    def _init_visualization_paths(self):
        """初始化可视化路径"""
        self.data_managers['visualization'] = {
            'plot': self.session_dir / "Plot",
            'visualization': self.session_dir / "Visualization"
        }
    
    def _save_session_metadata(self):
        """保存会话元数据"""
        metadata = {
            'session_info': {
                'experiment_name': self.experiment_name,
                'training_stage': self.training_stage.value,
                'stage_variant': self.stage_variant,
                'session_dir': str(self.session_dir),
                'created_time': datetime.now().isoformat(),
                'python_env': os.environ.get('CONDA_DEFAULT_ENV', 'unknown'),
                'hostname': os.uname().nodename if hasattr(os, 'uname') else 'unknown'
            },
            'feature_flags': self.feature_flags,
            'training_config': self._deep_serialize(self.config),
            'directory_structure': self._get_directory_structure()
        }
        
        # 保存到配置目录
        metadata_file = self.session_dir / "Config" / "session_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # 同时保存简化的配置
        config_file = self.session_dir / "Config" / f"{self.training_stage.value}_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(self._deep_serialize(self.config), f, indent=2, ensure_ascii=False)
    
    def _deep_serialize(self, obj):
        """深度序列化对象"""
        if isinstance(obj, dict):
            return {k: self._deep_serialize(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._deep_serialize(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, TrainingStage):
            return obj.value
        elif hasattr(obj, '__dict__'):
            return {k: self._deep_serialize(v) for k, v in obj.__dict__.items()}
        else:
            return obj
    
    def _get_directory_structure(self) -> Dict[str, str]:
        """获取目录结构"""
        structure = {}
        
        # 标准目录
        for subdir in self.STANDARD_SUBDIRS:
            structure[subdir.lower()] = str(self.session_dir / subdir)
        
        # 可选目录
        for feature, enabled in self.feature_flags.items():
            if enabled and feature in self.OPTIONAL_SUBDIRS:
                for subdir in self.OPTIONAL_SUBDIRS[feature]:
                    key = f"{feature}_{subdir.split('/')[-1].lower()}"
                    structure[key] = str(self.session_dir / subdir)
        
        return structure
    
    def _log_session_summary(self):
        """记录会话摘要"""
        self.logger.info(f"📁 会话目录: {self.session_dir.name}")
        self.logger.info(f"🎯 训练阶段: {self.training_stage.value}")
        if self.stage_variant:
            self.logger.info(f"🔀 阶段变体: {self.stage_variant}")
        
        enabled_features = [k for k, v in self.feature_flags.items() if v]
        self.logger.info(f"🚀 启用功能: {', '.join(enabled_features)}")
    
    # === 数据管理接口 ===
    
    def get_session_paths(self) -> Dict[str, Path]:
        """获取所有会话路径"""
        paths = {
            'session_dir': self.session_dir,
            'config': self.session_dir / "Config",
            'model': self.session_dir / "Model",
            'result': self.session_dir / "Result",
            'checkpoints': self.session_dir / "Checkpoints",
            'logs': self.session_dir / "Logs"
        }
        
        # 添加可选路径
        for feature, enabled in self.feature_flags.items():
            if enabled and feature in self.data_managers:
                if isinstance(self.data_managers[feature], dict):
                    for sub_key, sub_path in self.data_managers[feature].items():
                        if isinstance(sub_path, Path):
                            paths[f"{feature}_{sub_key}"] = sub_path
                else:
                    paths[feature] = self.data_managers[feature]
        
        return paths
    
    def get_model_save_path(self, model_name: str = "model", suffix: str = ".zip") -> Path:
        """获取模型保存路径"""
        filename = f"{model_name}_{self.training_stage.value}"
        if self.stage_variant:
            filename += f"_{self.stage_variant}"
        filename += suffix
        return self.session_dir / "Model" / filename
    
    def get_checkpoint_path(self, step: int) -> Path:
        """获取检查点保存路径"""
        filename = f"checkpoint_{self.training_stage.value}"
        if self.stage_variant:
            filename += f"_{self.stage_variant}"
        filename += f"_step_{step}.zip"
        return self.session_dir / "Checkpoints" / filename
    
    def get_result_path(self, result_type: str = "training_results") -> Path:
        """获取结果保存路径"""
        filename = f"{result_type}_{self.training_stage.value}"
        if self.stage_variant:
            filename += f"_{self.stage_variant}"
        filename += ".json"
        return self.session_dir / "Result" / filename
    
    def get_log_path(self, log_type: str = "training") -> Path:
        """获取日志文件路径"""
        filename = f"{log_type}_{self.training_stage.value}"
        if self.stage_variant:
            filename += f"_{self.stage_variant}"
        filename += ".log"
        return self.session_dir / "Logs" / filename
    
    def get_tensorboard_log_dir(self, mode: str = "train") -> Optional[Path]:
        """获取TensorBoard日志目录"""
        if self.feature_flags['tensorboard']:
            return self.data_managers['tensorboard'].get(mode)
        return None
    
    def get_trajectory_manager(self, mode: str = "train"):
        """获取轨迹管理器"""
        if self.feature_flags['trajectory']:
            return self.data_managers['trajectory'].get(mode)
        return None
    
    def get_visualization_path(self, viz_type: str = "plot") -> Optional[Path]:
        """获取可视化路径"""
        if self.feature_flags['visualization']:
            return self.data_managers['visualization'].get(viz_type)
        return None
    
    # === 数据保存接口 ===
    
    def save_training_results(self, results: Dict[str, Any], result_type: str = "training_results"):
        """保存训练结果"""
        result_data = {
            'session_info': {
                'experiment_name': self.experiment_name,
                'training_stage': self.training_stage.value,
                'stage_variant': self.stage_variant,
                'session_dir': str(self.session_dir),
                'end_time': datetime.now().isoformat()
            },
            'results': self._deep_serialize(results)
        }
        
        result_path = self.get_result_path(result_type)
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"💾 训练结果已保存: {result_path}")
    
    def save_analysis_data(self, data: Dict[str, Any], filename: str):
        """保存分析数据"""
        if not self.feature_flags['analysis']:
            self.logger.warning("分析功能未启用，无法保存分析数据")
            return
        
        analysis_path = self.session_dir / "Analysis" / filename
        analysis_path.parent.mkdir(exist_ok=True)
        
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(self._deep_serialize(data), f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"📊 分析数据已保存: {analysis_path}")
    
    def save_visualization_data(self, data: Dict[str, Any], filename: str):
        """保存可视化数据"""
        if not self.feature_flags['visualization']:
            self.logger.warning("可视化功能未启用，无法保存可视化数据")
            return
        
        viz_path = self.get_visualization_path("plot") / filename
        
        with open(viz_path, 'w', encoding='utf-8') as f:
            json.dump(self._deep_serialize(data), f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"📈 可视化数据已保存: {viz_path}")
    
    def save_pointcloud_data(self, point_cloud: np.ndarray, filename: str):
        """保存点云数据"""
        if not self.feature_flags['pointcloud']:
            self.logger.warning("点云功能未启用，无法保存点云数据")
            return
        
        pc_path = self.session_dir / "PointCloud" / filename
        
        if filename.endswith('.npy'):
            np.save(pc_path, point_cloud)
        else:
            # 保存为JSON格式
            pc_data = {
                'timestamp': datetime.now().isoformat(),
                'shape': point_cloud.shape,
                'data': point_cloud.tolist()
            }
            
            with open(pc_path.with_suffix('.json'), 'w', encoding='utf-8') as f:
                json.dump(pc_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"☁️ 点云数据已保存: {pc_path}")
    
    # === 会话管理 ===
    
    def get_session_info(self) -> Dict[str, Any]:
        """获取完整会话信息"""
        return {
            'experiment_name': self.experiment_name,
            'training_stage': self.training_stage.value,
            'stage_variant': self.stage_variant,
            'session_dir': str(self.session_dir),
            'feature_flags': self.feature_flags,
            'paths': {k: str(v) for k, v in self.get_session_paths().items()},
            'created_time': datetime.now().isoformat()
        }
    
    def create_child_session(self, 
                           child_stage: TrainingStage, 
                           child_variant: Optional[str] = None,
                           inherit_features: bool = True) -> 'SessionManager':
        """
        创建子会话（用于流水线训练）
        
        Args:
            child_stage: 子阶段
            child_variant: 子阶段变体
            inherit_features: 是否继承功能设置
            
        Returns:
            新的SessionManager实例
        """
        if inherit_features:
            feature_kwargs = {f"enable_{k}": v for k, v in self.feature_flags.items()}
        else:
            feature_kwargs = {}
        
        return SessionManager(
            experiment_name=self.experiment_name,
            training_stage=child_stage,
            config=self.config,
            base_log_dir=str(self.base_log_dir),
            stage_variant=child_variant,
            **feature_kwargs
        )
    
    def link_parent_session(self, parent_session_dir: Path):
        """链接父会话（用于模型继承）"""
        link_info = {
            'parent_session': str(parent_session_dir),
            'inheritance_time': datetime.now().isoformat(),
            'inheritance_type': 'model_transfer'
        }
        
        link_file = self.session_dir / "Config" / "parent_link.json"
        with open(link_file, 'w', encoding='utf-8') as f:
            json.dump(link_info, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"🔗 已链接父会话: {parent_session_dir}")
    
    def close(self):
        """关闭会话管理器"""
        # 清理轨迹管理器
        if self.feature_flags['trajectory'] and 'trajectory' in self.data_managers:
            for manager in self.data_managers['trajectory'].values():
                if hasattr(manager, 'close'):
                    manager.close()
        
        # 保存最终会话状态
        final_state = {
            'close_time': datetime.now().isoformat(),
            'session_completed': True,
            'final_session_info': self.get_session_info()
        }
        
        final_state_file = self.session_dir / "Config" / "session_final_state.json"
        with open(final_state_file, 'w', encoding='utf-8') as f:
            json.dump(final_state, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"🔒 会话已关闭: {self.session_dir}")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()


# 便捷工厂函数
def create_session_manager(experiment_name: str,
                         training_stage: TrainingStage,
                         config: Dict[str, Any],
                         stage_variant: Optional[str] = None,
                         **kwargs) -> SessionManager:
    """创建会话管理器的便捷函数"""
    return SessionManager(
        experiment_name=experiment_name,
        training_stage=training_stage,
        config=config,
        stage_variant=stage_variant,
        **kwargs
    )
