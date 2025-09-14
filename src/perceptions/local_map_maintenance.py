"""
local_map_maintenance.py - 以自我为中心的局部地图维护模块

严格遵循论文理论框架，维护单一三值栅格地图，无全局参考框架依赖。
仅实现JSON规范定义的函数，禁止添加任何额外功能、优化方法或复杂逻辑。
"""
import numpy as np
from typing import Tuple, List
from collections import deque


class LocalMapMaintenance:
    """以自我为中心的局部地图维护器
    
    严格实现论文中的egocentric局部坐标系框架：
    - 单一三值栅格：-1=未知，0=空闲，1=占据
    - 纯传感器驱动更新，无全局参考框架依赖
    - 基于可达自由空间|M_acc|的探索率计算
    """
    
    def __init__(self, resolution: float = 0.1, grid_size: Tuple[int, int] = (100, 100)):
        """
        初始化局部地图维护器
        
        Args:
            resolution: 栅格分辨率（米/格），默认0.1
            grid_size: 栅格尺寸（宽度, 高度），默认(100, 100)
        """
        self.resolution = resolution
        self.grid_size = grid_size
        
        # 栅格中心对应无人机位置
        self.center_x = grid_size[0] // 2
        self.center_y = grid_size[1] // 2
        
        # 创建三值栅格地图，全部初始化为-1（未知）
        self.grid = np.full(grid_size, -1, dtype=np.int8)
        
        # 计算并设置可达自由空间大小 |M_acc|
        self.accessible_area = self._compute_accessible_area()
        
        # 信息增益计算所需的前一帧状态
        self._previous_explored_count = 0
    
    def _world_to_local_coordinates(self, points: np.ndarray, 
                                   drone_position: np.ndarray, 
                                   drone_yaw: float) -> np.ndarray:
        """
        将世界坐标转换为局部坐标系
        
        严格实现论文中的坐标变换公式：
        1. 计算点相对于无人机的位置：(x_w - x_u, y_w - y_u)
        2. 应用逆时针旋转矩阵R(-θ_u)
        3. 转换为栅格坐标：grid_x = floor(x/δ) + center_x, grid_y = floor(y/δ) + center_y
        
        Args:
            points: 世界坐标点云，形状为(N, 3)
            drone_position: 无人机位置 [x, y, z]
            drone_yaw: 无人机偏航角（弧度）
            
        Returns:
            np.ndarray: 局部坐标系中的栅格坐标，形状为(N, 2)
        """
        if points.size == 0:
            return np.array([]).reshape(0, 2)
        
        # 仅使用x,y坐标，忽略z坐标
        points_2d = points[:, :2]
        
        # 计算点相对于无人机的位置
        relative_pos = points_2d - drone_position[:2]
        
        # 应用逆时针旋转矩阵R(-θ_u)
        cos_theta = np.cos(-drone_yaw)
        sin_theta = np.sin(-drone_yaw)
        rotation_matrix = np.array([[cos_theta, -sin_theta],
                                   [sin_theta, cos_theta]])
        
        # 旋转到局部坐标系
        local_coords = relative_pos @ rotation_matrix.T
        
        # 转换为栅格坐标
        grid_coords = local_coords / self.resolution
        grid_coords[:, 0] += self.center_x
        grid_coords[:, 1] += self.center_y
        
        # 转换为整数坐标
        grid_coords = np.floor(grid_coords).astype(int)
        
        # 过滤有效坐标（在栅格范围内）
        valid_mask = ((grid_coords[:, 0] >= 0) & (grid_coords[:, 0] < self.grid_size[0]) &
                     (grid_coords[:, 1] >= 0) & (grid_coords[:, 1] < self.grid_size[1]))
        
        return grid_coords[valid_mask]
    
    def _get_ray_path(self, start: np.ndarray, end: np.ndarray) -> List[Tuple[int, int]]:
        """
        计算激光射线路径上的栅格
        
        使用Bresenham算法计算从传感器位置到激光终点的直线路径
        
        Args:
            start: 传感器位置在栅格坐标系中的坐标 [x, y]
            end: 激光终点在栅格坐标系中的坐标 [x, y]
            
        Returns:
            List[Tuple[int, int]]: 射线路径上的栅格坐标列表
        """
        x0, y0 = int(start[0]), int(start[1])
        x1, y1 = int(end[0]), int(end[1])
        
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        x, y = x0, y0
        
        while True:
            points.append((x, y))
            if x == x1 and y == y1:
                break
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        
        return points
    
    def update(self, point_cloud: np.ndarray, 
              drone_position: np.ndarray, 
              drone_yaw: float) -> int:
        """
        更新局部地图
        
        严格遵循论文中的射线投射更新规则：
        1. 将点云转换到局部坐标系
        2. 对每条激光束：
           * 计算射线路径
           * 路径中未被阻挡区域标记为空闲（0）
           * 终点所在栅格标记为占据（1）
           * 仅当栅格被明确观测时才更新状态
        3. 更新探索历史C_t = C_{t-1} ∪ Grid(L_t)
        
        Args:
            point_cloud: 激光雷达点云数据，形状为(N, 3)
            drone_position: 无人机位置 [x, y, z]
            drone_yaw: 无人机偏航角（弧度）
            
        Returns:
            int: 本次更新的栅格数量
        """
        if point_cloud.size == 0:
            return 0
        
        # 将点云转换到局部坐标系
        local_points = self._world_to_local_coordinates(point_cloud, drone_position, drone_yaw)
        
        if local_points.size == 0:
            return 0
        
        updated_count = 0
        sensor_pos = np.array([self.center_x, self.center_y])
        
        # 对每条激光束进行处理
        for point in local_points:
            # 计算射线路径
            ray_path = self._get_ray_path(sensor_pos, point)
            
            # 更新射线路径上的栅格
            for i, (x, y) in enumerate(ray_path):
                # 检查栅格边界
                if not (0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]):
                    continue
                
                if i == len(ray_path) - 1:
                    # 终点所在栅格标记为占据（1）
                    if self.grid[x, y] != 1:
                        self.grid[x, y] = 1
                        updated_count += 1
                else:
                    # 路径中未被阻挡区域标记为空闲（0）
                    if self.grid[x, y] != 0:
                        self.grid[x, y] = 0
                        updated_count += 1
        
        # 更新可达自由空间
        self.accessible_area = self._compute_accessible_area()
        
        return updated_count
    
    def _compute_accessible_area(self) -> int:
        """
        计算可达自由空间大小 |M_acc|
        
        新逻辑：基于雷达探测范围计算理论可达区域
        而不是基于当前已探测的区域（避免循环依赖）
        
        Returns:
            int: 理论可达自由空间的栅格数量 |M_acc|
        """
        # 方案：基于雷达最大探测距离计算理论可达区域
        max_detection_range = 20.0  # 室内环境激光雷达最大探测距离50米
        max_detection_cells = int(max_detection_range / self.resolution)
        
        # 计算以无人机为中心的圆形区域内的栅格数
        # 面积 ≈ π * r²，转换为栅格数
        theoretical_accessible = int(np.pi * max_detection_cells ** 2)
        
        # 限制在实际地图大小内
        total_cells = self.grid_size[0] * self.grid_size[1]
        accessible_area = min(theoretical_accessible, total_cells)
        
        # 确保至少有1个可达栅格（避免除零错误）
        return max(1, accessible_area)
    
    def get_exploration_rate(self) -> float:
        """
        获取当前探索率
        
        严格遵循论文中的探索率定义：
        探索率 η_t = |C_t| / |M_acc|
        其中：
        - |C_t|: 已探索区域大小（非未知栅格数量）
        - |M_acc|: 可达自由空间大小
        
        Returns:
            float: 探索率，范围[0, 1]
        """
        # 计算已探索区域大小 |C_t|
        explored_count = int(np.sum(self.grid != -1))
        
        # 当|M_acc|=0时返回0.0
        if self.accessible_area == 0:
            return 0.0
        
        # 探索率 η_t = |C_t| / |M_acc|，确保结果在[0, 1]范围内
        exploration_rate = explored_count / self.accessible_area
        return float(min(1.0, max(0.0, exploration_rate)))
    
    def get_local_map_state(self, local_size: int = 20) -> np.ndarray:
        """
        获取局部地图状态用于策略输入
        
        提取以无人机为中心的局部区域：
        1. 通常为固定大小的正方形区域
        2. 对边界外区域进行适当填充
        
        Args:
            local_size: 局部区域大小，默认20
            
        Returns:
            np.ndarray: 局部地图状态，形状为(local_size, local_size)
                       -1=未知，0=空闲，1=占据
        """
        half_size = local_size // 2
        
        # 计算提取区域的边界
        x_start = max(0, self.center_x - half_size)
        x_end = min(self.grid_size[0], self.center_x + half_size)
        y_start = max(0, self.center_y - half_size)
        y_end = min(self.grid_size[1], self.center_y + half_size)
        
        # 提取局部区域
        local_region = self.grid[x_start:x_end, y_start:y_end].copy()
        
        # 如果提取的区域大小不足，用未知区域(-1)填充
        if local_region.shape != (local_size, local_size):
            # 创建填充后的区域
            padded_region = np.full((local_size, local_size), -1, dtype=np.int8)
            
            # 计算放置位置（居中）
            offset_x = (local_size - local_region.shape[0]) // 2
            offset_y = (local_size - local_region.shape[1]) // 2
            
            # 将提取的区域放置到填充区域中
            padded_region[offset_x:offset_x + local_region.shape[0],
                         offset_y:offset_y + local_region.shape[1]] = local_region
            
            local_region = padded_region
        
        return local_region
    
    def get_information_gain(self) -> int:
        """
        计算本次更新的信息增益
        
        严格实现论文中的r_info(t) = |C_t \ C_{t-1}|
        即新标记为已探索的栅格数量
        
        Returns:
            int: 信息增益值（非负整数）
        """
        # 计算本次更新新增的已探索区域：|C_t \ C_{t-1}|
        current_explored_count = int(np.sum(self.grid != -1))
        information_gain = max(0, current_explored_count - self._previous_explored_count)
        
        # 更新前一帧状态
        self._previous_explored_count = current_explored_count
        
        return information_gain
    
    def reset(self):
        """重置局部地图到初始状态"""
        # 重置栅格地图为全未知状态
        self.grid.fill(-1)
        
        # 重置可达自由空间大小
        self.accessible_area = 0
        
        # 重置前一帧探索计数
        self._previous_explored_count = 0
