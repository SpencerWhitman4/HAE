#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PointCloudFilter.py - 点云过滤工具类

从CoordinateTransform模块中提取的核心点云过滤功能，简化为独立工具类。
提供噪声过滤、地面/天花板点过滤等基础功能。
"""

import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class PointCloudFilterConfig:
    """点云过滤配置"""
    noise_threshold: float = 2.0      # 噪声过滤阈值倍数
    filter_ground: bool = True        # 是否过滤地面点
    filter_ceiling: bool = True       # 是否过滤天花板点
    ground_height: float = -0.05      # 地面高度阈值(m)
    ceiling_height: float = 1.5       # 天花板高度阈值(m)
    min_points: int = 10              # 最少保留点数
    max_distance: float = 10.0        # 最大检测距离(m)


class PointCloudFilter:
    """
    点云过滤器 - 简化版本
    
    主要功能：
    1. 基于统计的噪声点过滤
    2. 地面点过滤
    3. 天花板点过滤
    4. 距离范围过滤
    5. 过滤统计记录
    """
    
    def __init__(self, config: Optional[PointCloudFilterConfig] = None):
        """
        初始化点云过滤器
        
        Args:
            config: 过滤配置，如果为None则使用默认配置
        """
        self.config = config or PointCloudFilterConfig()
        
        # 过滤统计
        self.stats = {
            'total_processed': 0,
            'noise_filtered': 0,
            'ground_filtered': 0,
            'ceiling_filtered': 0,
            'distance_filtered': 0,
            'empty_inputs': 0
        }
    
    def filter(self, point_cloud: np.ndarray) -> np.ndarray:
        """
        主要过滤接口
        
        Args:
            point_cloud: 输入点云 [N, 3]
            
        Returns:
            np.ndarray: 过滤后的点云
        """
        if point_cloud is None or len(point_cloud) == 0:
            self.stats['empty_inputs'] += 1
            return np.array([]).reshape(0, 3)
        
        self.stats['total_processed'] += 1
        original_count = len(point_cloud)
        processed_cloud = point_cloud.copy()
        
        # 1. 距离过滤 - 先执行以减少后续计算量
        if self.config.max_distance > 0:
            before_count = len(processed_cloud)
            processed_cloud = self._filter_by_distance(processed_cloud)
            self.stats['distance_filtered'] += before_count - len(processed_cloud)
        
        # 2. 噪声过滤
        if self.config.noise_threshold > 0:
            before_count = len(processed_cloud)
            processed_cloud = self._filter_noise_points(processed_cloud)
            self.stats['noise_filtered'] += before_count - len(processed_cloud)
        
        # 3. 地面点过滤
        if self.config.filter_ground:
            before_count = len(processed_cloud)
            processed_cloud = self._filter_ground_points(processed_cloud)
            self.stats['ground_filtered'] += before_count - len(processed_cloud)
        
        # 4. 天花板点过滤
        if self.config.filter_ceiling:
            before_count = len(processed_cloud)
            processed_cloud = self._filter_ceiling_points(processed_cloud)
            self.stats['ceiling_filtered'] += before_count - len(processed_cloud)
        
        # 5. 最少点数保护
        if len(processed_cloud) < self.config.min_points and original_count > 0:
            # 如果过滤后点数太少，保留距离最近的若干点
            distances = np.linalg.norm(point_cloud, axis=1)
            closest_indices = np.argsort(distances)[:self.config.min_points]
            processed_cloud = point_cloud[closest_indices]
        
        return processed_cloud
    
    def _filter_by_distance(self, point_cloud: np.ndarray) -> np.ndarray:
        """过滤距离过远的点"""
        if len(point_cloud) == 0:
            return point_cloud
        
        distances = np.linalg.norm(point_cloud, axis=1)
        valid_mask = distances <= self.config.max_distance
        return point_cloud[valid_mask]
    
    def _filter_noise_points(self, point_cloud: np.ndarray) -> np.ndarray:
        """
        基于距离统计的噪声点过滤
        使用中位数绝对偏差(MAD)检测异常点
        """
        if len(point_cloud) == 0:
            return point_cloud
        
        # 计算点到原点的距离
        distances = np.linalg.norm(point_cloud, axis=1)
        
        if len(distances) < 3:  # 点数太少，跳过过滤
            return point_cloud
        
        # 使用中位数绝对偏差(MAD)检测异常点
        median_dist = np.median(distances)
        mad = np.median(np.abs(distances - median_dist))
        
        # 避免除零错误
        if mad == 0:
            mad = np.std(distances)
            if mad == 0:
                return point_cloud
        
        # 定义噪声阈值
        threshold = median_dist + 3 * mad * self.config.noise_threshold
        
        # 过滤异常远距离点
        valid_mask = distances <= threshold
        return point_cloud[valid_mask]
    
    def _filter_ground_points(self, point_cloud: np.ndarray) -> np.ndarray:
        """过滤地面附近的点"""
        if len(point_cloud) == 0 or point_cloud.shape[1] < 3:
            return point_cloud
        
        # 过滤z坐标低于阈值的点（地面点）
        valid_mask = point_cloud[:, 2] > self.config.ground_height
        return point_cloud[valid_mask]
    
    def _filter_ceiling_points(self, point_cloud: np.ndarray) -> np.ndarray:
        """过滤天花板附近的点"""
        if len(point_cloud) == 0 or point_cloud.shape[1] < 3:
            return point_cloud
        
        # 过滤z坐标高于阈值的点（天花板点）
        valid_mask = point_cloud[:, 2] < self.config.ceiling_height
        return point_cloud[valid_mask]
    
    def get_filter_stats(self) -> Dict[str, Any]:
        """
        获取过滤统计信息
        
        Returns:
            Dict: 包含过滤统计的字典
        """
        total_filtered = (self.stats['noise_filtered'] + 
                         self.stats['ground_filtered'] + 
                         self.stats['ceiling_filtered'] +
                         self.stats['distance_filtered'])
        
        return {
            'total_processed': self.stats['total_processed'],
            'empty_inputs': self.stats['empty_inputs'],
            'noise_filtered': self.stats['noise_filtered'],
            'ground_filtered': self.stats['ground_filtered'],
            'ceiling_filtered': self.stats['ceiling_filtered'],
            'distance_filtered': self.stats['distance_filtered'],
            'total_filtered': total_filtered,
            'filter_rate': total_filtered / max(1, self.stats['total_processed']) * 100
        }
    
    def reset_stats(self):
        """重置过滤统计"""
        for key in self.stats:
            self.stats[key] = 0
    
    def update_config(self, **kwargs):
        """
        更新过滤配置
        
        Args:
            **kwargs: 配置参数
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                print(f"⚠️ Unknown config parameter: {key}")


# 便捷创建函数
def create_pointcloud_filter(
    noise_threshold: float = 2.0,
    filter_ground: bool = True,
    filter_ceiling: bool = True,
    ground_height: float = -0.05,
    ceiling_height: float = 1.5,
    max_distance: float = 10.0
) -> PointCloudFilter:
    """
    便捷创建点云过滤器
    
    Args:
        noise_threshold: 噪声过滤阈值倍数
        filter_ground: 是否过滤地面点
        filter_ceiling: 是否过滤天花板点
        ground_height: 地面高度阈值
        ceiling_height: 天花板高度阈值
        max_distance: 最大检测距离
        
    Returns:
        PointCloudFilter: 配置好的过滤器实例
    """
    config = PointCloudFilterConfig(
        noise_threshold=noise_threshold,
        filter_ground=filter_ground,
        filter_ceiling=filter_ceiling,
        ground_height=ground_height,
        ceiling_height=ceiling_height,
        max_distance=max_distance
    )
    
    return PointCloudFilter(config)


if __name__ == "__main__":
    # 简单测试
    print("🔧 Testing PointCloudFilter...")
    
    # 创建测试数据
    np.random.seed(42)
    test_cloud = np.random.randn(1000, 3) * 2
    test_cloud[:100, 2] = -0.1  # 添加一些地面点
    test_cloud[100:120, 2] = 2.0  # 添加一些天花板点
    test_cloud[120:130] *= 10  # 添加一些噪声点
    
    # 创建过滤器
    filter_obj = create_pointcloud_filter()
    
    # 执行过滤
    filtered_cloud = filter_obj.filter(test_cloud)
    
    # 显示结果
    stats = filter_obj.get_filter_stats()
    print(f"📊 Original points: {len(test_cloud)}")
    print(f"📊 Filtered points: {len(filtered_cloud)}")
    print(f"📊 Filter stats: {stats}")
    
    print("✅ PointCloudFilter test completed!")
