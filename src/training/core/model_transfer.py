#!/usr/bin/env python3

"""
模型迁移管理器 - 处理不同训练阶段间的模型权重迁移
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union, List, Tuple
from pathlib import Path
import copy
import json
import numpy as np

from .base_trainer import TrainingStage

logger = logging.getLogger(__name__)


class ModelTransferManager:
    """
    模型迁移管理器
    
    负责在不同训练阶段间转移模型权重，实现增量学习
    """
    
    def __init__(self, model_save_dir: str = "./models"):
        self.model_save_dir = Path(model_save_dir)
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        
        # 模型权重映射策略
        self.transfer_strategies = {
            (TrainingStage.FOUNDATION, TrainingStage.HIERARCHICAL): self._foundation_to_hierarchical,
            (TrainingStage.FOUNDATION, TrainingStage.ABLATION): self._foundation_to_ablation,
            (TrainingStage.FOUNDATION, TrainingStage.BASELINE): self._foundation_to_baseline
        }
    
    def save_foundation_model(self, 
                             model: nn.Module, 
                             optimizer: torch.optim.Optimizer,
                             training_stats: Dict[str, Any],
                             model_name: str = "foundation_model") -> str:
        """
        保存基座模型
        
        Args:
            model: 训练好的基座模型
            optimizer: 优化器状态
            training_stats: 训练统计信息
            model_name: 模型名称
            
        Returns:
            保存路径
        """
        
        save_path = self.model_save_dir / f"{model_name}.zip"
        
        # 构建保存数据
        checkpoint = {
            'stage': TrainingStage.FOUNDATION.value,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'training_stats': training_stats,
            'model_architecture': self._extract_architecture_info(model),
            'transfer_metadata': self._create_transfer_metadata(model)
        }
        
        # 保存checkpoint
        torch.save(checkpoint, save_path)
        
        # 保存配套的元数据文件
        metadata_path = self.model_save_dir / f"{model_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump({
                'stage': TrainingStage.FOUNDATION.value,
                'model_name': model_name,
                'save_path': str(save_path),
                'training_stats': training_stats,
                'model_info': checkpoint['model_architecture']
            }, f, indent=2)
        
        self.logger.info(f"基座模型已保存: {save_path}")
        return str(save_path)
    
    def load_foundation_model(self, model_path: str) -> Dict[str, Any]:
        """
        加载基座模型
        
        Args:
            model_path: 模型路径
            
        Returns:
            模型数据字典
        """
        
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # 验证模型格式
        required_keys = ['stage', 'model_state_dict', 'training_stats', 'transfer_metadata']
        for key in required_keys:
            if key not in checkpoint:
                raise ValueError(f"模型文件缺少必要信息: {key}")
        
        # 验证是基座模型
        if checkpoint['stage'] != TrainingStage.FOUNDATION.value:
            raise ValueError(f"期望基座模型，但加载的是: {checkpoint['stage']}")
        
        self.logger.info(f"基座模型加载完成: {model_path}")
        return checkpoint
    
    def transfer_weights(self, 
                        foundation_checkpoint: Dict[str, Any],
                        target_model: nn.Module,
                        target_stage: TrainingStage,
                        transfer_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        权重迁移主函数
        
        Args:
            foundation_checkpoint: 基座模型checkpoint
            target_model: 目标模型
            target_stage: 目标训练阶段
            transfer_config: 迁移配置
            
        Returns:
            迁移结果报告
        """
        
        source_stage = TrainingStage.FOUNDATION
        transfer_key = (source_stage, target_stage)
        
        if transfer_key not in self.transfer_strategies:
            raise ValueError(f"不支持的迁移路径: {source_stage} -> {target_stage}")
        
        # 执行特定的迁移策略
        transfer_strategy = self.transfer_strategies[transfer_key]
        transfer_result = transfer_strategy(
            foundation_checkpoint, 
            target_model, 
            transfer_config or {}
        )
        
        self.logger.info(f"权重迁移完成: {source_stage.value} -> {target_stage.value}")
        return transfer_result
    
    def _foundation_to_hierarchical(self, 
                                   foundation_checkpoint: Dict[str, Any],
                                   hierarchical_model: nn.Module,
                                   config: Dict[str, Any]) -> Dict[str, Any]:
        """基座模型 -> 分层模型的权重迁移"""
        
        foundation_weights = foundation_checkpoint['model_state_dict']
        transfer_mode = config.get('transfer_mode', 'partial')  # 'full', 'partial', 'bias_only'
        
        transferred_layers = []
        skipped_layers = []
        
        hierarchical_state_dict = hierarchical_model.state_dict()
        
        for name, param in hierarchical_state_dict.items():
            if name in foundation_weights:
                # 直接匹配的层
                if foundation_weights[name].shape == param.shape:
                    param.data.copy_(foundation_weights[name])
                    transferred_layers.append(name)
                else:
                    # 形状不匹配的层
                    self.logger.warning(f"形状不匹配，跳过: {name}")
                    skipped_layers.append(name)
            
            elif 'low_level' in name and transfer_mode != 'bias_only':
                # 低层控制器权重迁移
                foundation_key = name.replace('low_level.', '')
                if foundation_key in foundation_weights:
                    if foundation_weights[foundation_key].shape == param.shape:
                        param.data.copy_(foundation_weights[foundation_key])
                        transferred_layers.append(f"{name} <- {foundation_key}")
                    else:
                        skipped_layers.append(f"{name} (shape mismatch)")
            
            elif transfer_mode == 'bias_only' and 'bias' in name:
                # 仅迁移偏置项
                foundation_key = name.replace('high_level.', '').replace('low_level.', '')
                if foundation_key in foundation_weights:
                    param.data.copy_(foundation_weights[foundation_key])
                    transferred_layers.append(f"{name} (bias only)")
        
        return {
            'transferred_layers': transferred_layers,
            'skipped_layers': skipped_layers,
            'transfer_mode': transfer_mode,
            'success_rate': len(transferred_layers) / len(hierarchical_state_dict)
        }
    
    def _foundation_to_ablation(self, 
                               foundation_checkpoint: Dict[str, Any],
                               ablation_model: nn.Module,
                               config: Dict[str, Any]) -> Dict[str, Any]:
        """基座模型 -> 消融实验模型的权重迁移"""
        
        ablation_type = config.get('ablation_type', 'B1')
        
        if ablation_type in ['B1', 'B2']:
            # B1（直接控制）和B2（扁平化）可以直接使用基座模型权重
            return self._full_weight_transfer(foundation_checkpoint, ablation_model)
        
        elif ablation_type == 'B3':
            # B3（单步分层）需要特殊处理
            return self._foundation_to_hierarchical(foundation_checkpoint, ablation_model, config)
        
        else:
            raise ValueError(f"未知的消融类型: {ablation_type}")
    
    def _foundation_to_baseline(self, 
                               foundation_checkpoint: Dict[str, Any],
                               baseline_model: nn.Module,
                               config: Dict[str, Any]) -> Dict[str, Any]:
        """基座模型 -> 基线模型的权重迁移"""
        
        baseline_algorithm = config.get('algorithm', 'ppo')
        use_pretrained_init = config.get('use_pretrained_init', True)
        
        if not use_pretrained_init:
            # 仅作为初始化偏置，不迁移权重
            return {
                'transferred_layers': [],
                'skipped_layers': list(baseline_model.state_dict().keys()),
                'transfer_mode': 'no_transfer',
                'success_rate': 0.0,
                'note': '基线模型使用随机初始化'
            }
        
        # SB3模型的权重迁移
        if baseline_algorithm in ['ppo', 'sac', 'td3']:
            return self._transfer_to_sb3_model(foundation_checkpoint, baseline_model, baseline_algorithm)
        else:
            raise ValueError(f"不支持的基线算法: {baseline_algorithm}")
    
    def _full_weight_transfer(self, 
                             foundation_checkpoint: Dict[str, Any],
                             target_model: nn.Module) -> Dict[str, Any]:
        """完整权重迁移"""
        
        foundation_weights = foundation_checkpoint['model_state_dict']
        target_state_dict = target_model.state_dict()
        
        transferred_layers = []
        skipped_layers = []
        
        for name, param in target_state_dict.items():
            if name in foundation_weights:
                if foundation_weights[name].shape == param.shape:
                    param.data.copy_(foundation_weights[name])
                    transferred_layers.append(name)
                else:
                    skipped_layers.append(f"{name} (shape mismatch)")
            else:
                skipped_layers.append(f"{name} (not found)")
        
        return {
            'transferred_layers': transferred_layers,
            'skipped_layers': skipped_layers,
            'transfer_mode': 'full',
            'success_rate': len(transferred_layers) / len(target_state_dict)
        }
    
    def _transfer_to_sb3_model(self, 
                              foundation_checkpoint: Dict[str, Any],
                              sb3_model: nn.Module,
                              algorithm: str) -> Dict[str, Any]:
        """迁移到SB3模型"""
        
        foundation_weights = foundation_checkpoint['model_state_dict']
        
        # SB3模型结构映射
        sb3_mapping = {
            'ppo': {
                'policy.features_extractor': 'encoder',
                'policy.mlp_extractor.policy_net': 'actor',
                'policy.mlp_extractor.value_net': 'critic',
                'policy.action_net': 'action_head'
            },
            'sac': {
                'policy.features_extractor': 'encoder',
                'policy.mu': 'actor',
                'policy.log_std': 'actor_log_std'
            },
            'td3': {
                'policy.features_extractor': 'encoder',
                'policy.mu': 'actor'
            }
        }
        
        if algorithm not in sb3_mapping:
            raise ValueError(f"不支持的SB3算法: {algorithm}")
        
        mapping = sb3_mapping[algorithm]
        transferred_layers = []
        
        sb3_state_dict = sb3_model.state_dict()
        
        for sb3_key, foundation_prefix in mapping.items():
            for name, param in sb3_state_dict.items():
                if name.startswith(sb3_key):
                    # 构建对应的foundation权重键
                    foundation_key = name.replace(sb3_key, foundation_prefix)
                    
                    if foundation_key in foundation_weights:
                        if foundation_weights[foundation_key].shape == param.shape:
                            param.data.copy_(foundation_weights[foundation_key])
                            transferred_layers.append(f"{name} <- {foundation_key}")
        
        return {
            'transferred_layers': transferred_layers,
            'skipped_layers': [],  # SB3的未迁移层将使用默认初始化
            'transfer_mode': f'sb3_{algorithm}',
            'success_rate': len(transferred_layers) / len(sb3_state_dict) if sb3_state_dict else 0.0
        }
    
    def _extract_architecture_info(self, model: nn.Module) -> Dict[str, Any]:
        """提取模型架构信息"""
        
        def count_parameters(m):
            return sum(p.numel() for p in m.parameters() if p.requires_grad)
        
        architecture_info = {
            'total_parameters': count_parameters(model),
            'model_class': model.__class__.__name__,
            'layer_info': {}
        }
        
        # 记录各层信息
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # 叶子节点
                architecture_info['layer_info'][name] = {
                    'type': module.__class__.__name__,
                    'parameters': count_parameters(module)
                }
        
        return architecture_info
    
    def _create_transfer_metadata(self, model: nn.Module) -> Dict[str, Any]:
        """创建迁移元数据"""
        
        metadata = {
            'transferable_layers': [],
            'frozen_layers': [],
            'architecture_constraints': {}
        }
        
        # 标记可迁移的层
        for name, param in model.named_parameters():
            if 'encoder' in name or 'features' in name:
                metadata['transferable_layers'].append(name)
            elif 'head' in name or 'output' in name:
                metadata['frozen_layers'].append(name)
        
        # 架构约束
        metadata['architecture_constraints'] = {
            'input_dim': 86,  # 雷达观测维度
            'action_dim': 4,  # 动作维度
            'expected_layers': ['encoder', 'actor', 'critic']
        }
        
        return metadata
    
    def validate_transfer_compatibility(self, 
                                       foundation_checkpoint: Dict[str, Any],
                                       target_model: nn.Module,
                                       target_stage: TrainingStage) -> Tuple[bool, List[str]]:
        """验证迁移兼容性"""
        
        issues = []
        
        # 检查架构兼容性
        if 'model_architecture' in foundation_checkpoint:
            foundation_arch = foundation_checkpoint['model_architecture']
            target_arch = self._extract_architecture_info(target_model)
            
            # 参数数量检查
            if target_arch['total_parameters'] < foundation_arch['total_parameters'] * 0.5:
                issues.append("目标模型参数数量过少，可能导致迁移效果不佳")
        
        # 检查迁移路径支持
        transfer_key = (TrainingStage.FOUNDATION, target_stage)
        if transfer_key not in self.transfer_strategies:
            issues.append(f"不支持的迁移路径: {transfer_key}")
        
        # 检查权重形状兼容性
        foundation_weights = foundation_checkpoint['model_state_dict']
        target_weights = target_model.state_dict()
        
        compatible_layers = 0
        for name, param in target_weights.items():
            if name in foundation_weights:
                if foundation_weights[name].shape == param.shape:
                    compatible_layers += 1
        
        compatibility_rate = compatible_layers / len(target_weights) if target_weights else 0
        if compatibility_rate < 0.3:
            issues.append(f"权重兼容性过低: {compatibility_rate:.2%}")
        
        is_compatible = len(issues) == 0
        return is_compatible, issues
    
    def create_transfer_report(self, 
                              transfer_result: Dict[str, Any],
                              save_path: Optional[str] = None) -> str:
        """创建迁移报告"""
        
        report = f"""
=== 模型权重迁移报告 ===

迁移模式: {transfer_result.get('transfer_mode', 'Unknown')}
成功率: {transfer_result.get('success_rate', 0):.2%}

成功迁移的层 ({len(transfer_result.get('transferred_layers', []))}):
{chr(10).join(f"  - {layer}" for layer in transfer_result.get('transferred_layers', []))}

跳过的层 ({len(transfer_result.get('skipped_layers', []))}):
{chr(10).join(f"  - {layer}" for layer in transfer_result.get('skipped_layers', []))}

备注: {transfer_result.get('note', '无')}
"""
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            self.logger.info(f"迁移报告已保存: {save_path}")
        
        return report
