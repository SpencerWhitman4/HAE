#!/usr/bin/env python3

"""
配置模块
"""

from .default_configs import (
    FOUNDATION_CONFIG,
    HIERARCHICAL_CONFIG, 
    ABLATION_CONFIG,
    BASELINE_CONFIG,
    TRANSFER_CONFIG,
    PRESET_CONFIGS,
    get_preset_config,
    merge_configs
)

__all__ = [
    'FOUNDATION_CONFIG',
    'HIERARCHICAL_CONFIG',
    'ABLATION_CONFIG', 
    'BASELINE_CONFIG',
    'TRANSFER_CONFIG',
    'PRESET_CONFIGS',
    'get_preset_config',
    'merge_configs'
]
