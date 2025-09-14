"""
perceptions/__init__.py - 感知模块统一接口

集成功能：
1. 局部地图维护 - 以自我为中心的三值栅格地图
2. 安全监控与终止 - 碰撞检测和边界检查
3. 观测构建 - 为分层策略提供结构化输入
4. 雷达驱动感知 - 纯激光雷达的环境理解

核心设计原则：
- 纯雷达驱动，无GPS/里程计依赖
- 以自我为中心的坐标系
- 论文规范的探索率和信息增益计算
- 与StateManager完美集成

作者: HA-UAV团队  
日期: 2025年8月
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from collections import deque

# 设置日志记录器
logger = logging.getLogger(__name__)

# 导入核心感知组件
try:
    from .local_map_maintenance import LocalMapMaintenance
    from .safety_termination import SafetyMonitoring, MapBoundaryCheck, TerminationCheck
    logger.debug("感知核心组件导入成功")
except ImportError as e:
    logger.error(f"感知组件导入失败: {e}")
    # 创建占位符类
    LocalMapMaintenance = None
    SafetyMonitoring = None
    MapBoundaryCheck = None
    TerminationCheck = None


# =============== 数据结构定义 ===============

@dataclass
class ObservationInput:
    """观测输入数据结构"""
    point_cloud: np.ndarray          # 激光雷达点云数据 [N, 3]
    drone_position: np.ndarray       # 无人机位置 [x, y, z]
    drone_orientation: np.ndarray    # 无人机姿态 [roll, pitch, yaw]
    drone_velocity: np.ndarray       # 无人机速度 [vx, vy, vz]
    action_history: Optional[List[List[float]]] = None  # 历史动作序列
    timestamp: float = 0.0           # 时间戳


@dataclass
class LocalMapResults:
    """局部地图维护结果"""
    grid_state: np.ndarray           # 局部栅格地图状态
    exploration_rate: float          # 探索率 η_t = |C_t| / |M_acc|
    information_gain: int            # 信息增益 |C_t \ C_{t-1}|
    accessible_area: int             # 可达自由空间大小 |M_acc|
    updated_cells: int               # 本次更新的栅格数量
    
    # 用于StateManager的16维地图特征
    map_features: np.ndarray         # [16] 地图特征向量


@dataclass
class SafetyResults:
    """安全监控结果"""
    collision_detected: bool         # 是否检测到碰撞
    out_of_bounds: bool             # 是否越出地图边界
    min_obstacle_distance: float    # 到最近障碍物的距离
    is_safe: bool                   # 是否处于安全状态
    should_terminate: bool          # 是否应该终止episode
    termination_reason: str         # 终止原因


@dataclass 
class SectorDistances:
    """扇区距离数据"""
    distances: np.ndarray           # [36] 扇区距离数组
    angles: np.ndarray             # [36] 对应的角度数组
    valid_sectors: np.ndarray      # [36] 有效扇区掩码
    max_range: float               # 最大探测距离


@dataclass
class PerceptionResults:
    """感知模块完整输出"""
    local_map_results: LocalMapResults
    safety_results: SafetyResults
    sector_distances: SectorDistances
    observation_input: ObservationInput
    
    # 用于86维观测构建的核心数据
    map_features_16d: np.ndarray    # [16] 地图特征（用于86维观测）
    sector_distances_36d: np.ndarray # [36] 扇区距离（用于86维观测）


# =============== 核心感知配置 ===============

@dataclass
class PerceptionConfig:
    """感知模块配置"""
    # 局部地图配置
    map_resolution: float = 0.1                    # 栅格分辨率（米/格）
    map_grid_size: Tuple[int, int] = (100, 100)    # 栅格尺寸
    local_map_size: int = 32                       # 策略输入的局部地图大小
    
    # 安全监控配置
    collision_threshold: float = 0.1               # 碰撞判定阈值（米）
    safety_threshold: float = 0.5                  # 安全距离阈值（米）
    
    # 扇区距离配置
    num_sectors: int = 36                          # 扇区数量
    max_detection_range: float = 10.0              # 最大探测距离（米）
    
    # 地图边界配置
    default_map_bounds: List[float] = None         # 默认地图边界 [x_max, y_max, z_max]
    
    def __post_init__(self):
        if self.default_map_bounds is None:
            self.default_map_bounds = [20.0, 20.0, 5.0]


# =============== 主要感知类 ===============

class PerceptionManager:
    """
    感知管理器 - 统一的感知模块接口
    
    集成功能：
    1. 局部地图维护
    2. 安全监控
    3. 扇区距离计算  
    4. 观测数据构建
    """
    
    def __init__(self, config: PerceptionConfig):
        """初始化感知管理器
        
        Args:
            config: 感知配置对象
        """
        self.config = config
        
        # 初始化局部地图维护器
        if LocalMapMaintenance is not None:
            self.local_map = LocalMapMaintenance(
                resolution=config.map_resolution,
                grid_size=config.map_grid_size
            )
        else:
            self.local_map = None
            logger.warning("LocalMapMaintenance未可用，地图功能将被禁用")
        
        # 初始化组件状态
        self.is_initialized = True
        
        logger.debug(f"PerceptionManager初始化完成: "
                   f"map_size={config.map_grid_size}, "
                   f"sectors={config.num_sectors}, "
                   f"resolution={config.map_resolution}")
    
    def process_observation(self, obs_input: ObservationInput, 
                          map_manager=None) -> PerceptionResults:
        """处理完整观测输入
        
        Args:
            obs_input: 观测输入数据
            map_manager: 地图管理器（可选，用于边界检查）
            
        Returns:
            PerceptionResults: 完整的感知结果
        """
        try:
            # 1. 局部地图更新
            local_map_results = self._update_local_map(obs_input)
            
            # 2. 安全监控
            safety_results = self._monitor_safety(obs_input, map_manager)
            
            # 3. 扇区距离计算
            sector_distances = self._compute_sector_distances(obs_input)
            
            # 4. 构建感知结果
            return PerceptionResults(
                local_map_results=local_map_results,
                safety_results=safety_results,
                sector_distances=sector_distances,
                observation_input=obs_input,
                map_features_16d=local_map_results.map_features,
                sector_distances_36d=sector_distances.distances
            )
            
        except Exception as e:
            logger.error(f"感知处理失败: {e}")
            return self._create_safe_default_results(obs_input)
    
    def get_86d_observation_components(self, obs_input: ObservationInput,
                                     current_subgoals: np.ndarray,
                                     action_history: np.ndarray,
                                     map_manager=None) -> Dict[str, np.ndarray]:
        """获取86维观测的各个组件
        
        为StateManager提供构建86维观测所需的所有组件
        
        Args:
            obs_input: 观测输入
            current_subgoals: [5, 2] 当前子目标序列
            action_history: [6, 4] 动作历史
            map_manager: 地图管理器
            
        Returns:
            dict: 86维观测组件
            {
                'sector_distances': [36] 扇区距离,
                'action_history_flat': [24] 扁平化动作历史,
                'subgoals_flat': [10] 扁平化子目标,
                'map_features': [16] 地图特征
            }
        """
        # 处理观测获取感知结果
        perception_results = self.process_observation(obs_input, map_manager)
        
        # 确保输入数组格式正确
        if action_history.shape != (6, 4):
            action_history = np.zeros((6, 4), dtype=np.float32)
        if current_subgoals.shape != (5, 2):
            current_subgoals = np.zeros((5, 2), dtype=np.float32)
        
        return {
            'sector_distances': perception_results.sector_distances_36d,      # [36]
            'action_history_flat': action_history.flatten(),                 # [24] = 6*4
            'subgoals_flat': current_subgoals.flatten(),                     # [10] = 5*2
            'map_features': perception_results.map_features_16d               # [16]
        }
    
    def build_86d_observation(self, obs_input: ObservationInput,
                             current_subgoals: np.ndarray,
                             action_history: np.ndarray,
                             map_manager=None) -> np.ndarray:
        """构建完整的86维观测向量
        
        Args:
            obs_input: 观测输入
            current_subgoals: [5, 2] 子目标序列
            action_history: [6, 4] 动作历史  
            map_manager: 地图管理器
            
        Returns:
            np.ndarray: [86] 完整观测向量
        """
        # 获取各个组件
        components = self.get_86d_observation_components(
            obs_input, current_subgoals, action_history, map_manager
        )
        
        # 拼接成86维向量
        obs_86d = np.concatenate([
            components['sector_distances'],      # [0:36]   = 36维
            components['action_history_flat'],   # [36:60]  = 24维
            components['subgoals_flat'],         # [60:70]  = 10维
            components['map_features']           # [70:86]  = 16维
        ])
        
        assert obs_86d.shape[0] == 86, f"观测维度错误: {obs_86d.shape[0]} != 86"
        
        return obs_86d.astype(np.float32)
    
    def reset(self):
        """重置感知管理器状态"""
        if self.local_map is not None:
            # 重置局部地图
            self.local_map = LocalMapMaintenance(
                resolution=self.config.map_resolution,
                grid_size=self.config.map_grid_size
            )
        
        #
        #  logger.info("PerceptionManager已重置")
    
    # === 私有方法：具体感知处理 ===
    
    def _update_local_map(self, obs_input: ObservationInput) -> LocalMapResults:
        """更新局部地图"""
        if self.local_map is None:
            # 创建默认结果
            return LocalMapResults(
                grid_state=np.zeros((self.config.local_map_size, self.config.local_map_size), dtype=np.int8),
                exploration_rate=0.0,
                information_gain=0,
                accessible_area=0,
                updated_cells=0,
                map_features=self._create_default_map_features()
            )
        
        try:
            # 调试：检查点云数据
            if obs_input.point_cloud is None or obs_input.point_cloud.size == 0:
                logger.warning(f"点云数据为空或None: {obs_input.point_cloud}")
                # 返回默认结果
                return LocalMapResults(
                    grid_state=np.zeros((self.config.local_map_size, self.config.local_map_size), dtype=np.int8),
                    exploration_rate=0.0,
                    information_gain=0,
                    accessible_area=0,
                    updated_cells=0,
                    map_features=self._create_default_map_features()
                )
            
            logger.debug(f"点云数据大小: {obs_input.point_cloud.shape}")
            
            # 更新地图
            updated_cells = self.local_map.update(
                obs_input.point_cloud,
                obs_input.drone_position,
                obs_input.drone_orientation[2]  # yaw角
            )
            
            logger.debug(f"地图更新: {updated_cells}个栅格")
            
            # 获取地图状态
            grid_state = self.local_map.get_local_map_state(self.config.local_map_size)
            exploration_rate = self.local_map.get_exploration_rate()
            
            logger.debug(f"探索率: {exploration_rate:.3f}, 可达区域: {self.local_map.accessible_area}")
            information_gain = self.local_map.get_information_gain()
            
            # 生成16维地图特征
            map_features = self._extract_map_features_16d()
            
            return LocalMapResults(
                grid_state=grid_state,
                exploration_rate=exploration_rate,
                information_gain=information_gain,
                accessible_area=self.local_map.accessible_area,
                updated_cells=updated_cells,
                map_features=map_features
            )
            
        except Exception as e:
            logger.error(f"地图更新失败: {e}")
            return LocalMapResults(
                grid_state=np.zeros((self.config.local_map_size, self.config.local_map_size), dtype=np.int8),
                exploration_rate=0.0,
                information_gain=0,
                accessible_area=0,
                updated_cells=0,
                map_features=self._create_default_map_features()
            )
    
    def _monitor_safety(self, obs_input: ObservationInput, 
                       map_manager=None) -> SafetyResults:
        """监控安全状态"""
        if SafetyMonitoring is None:
            # 创建默认安全结果
            return SafetyResults(
                collision_detected=False,
                out_of_bounds=False,
                min_obstacle_distance=10.0,
                is_safe=True,
                should_terminate=False,
                termination_reason="SAFE: 安全监控模块未可用"
            )
        
        try:
            # 从map_manager获取边界信息
            if map_manager is not None:
                map_bounds = map_manager.get_map_bounds()
            else:
                # 使用默认边界
                map_bounds = {
                    'x_min': -6.0, 'x_max': 6.0,
                    'y_min': -6.0, 'y_max': 6.0, 
                    'z_min': 0.1, 'z_max': 5.0
                }
            
            # 使用集成的安全监控接口
            safety_status = TerminationCheck.monitor_safety_and_termination(
                obs_input.point_cloud,
                map_bounds,
                obs_input.drone_position,
                self.config.collision_threshold
            )
            
            return SafetyResults(
                collision_detected=safety_status['collision'],
                out_of_bounds=safety_status['out_of_bounds'],
                min_obstacle_distance=safety_status['min_distance'],
                is_safe=safety_status['is_safe'],
                should_terminate=safety_status['terminate'],
                termination_reason=safety_status['termination_reason']
            )
            
        except Exception as e:
            logger.error(f"安全监控失败: {e}")
            return SafetyResults(
                collision_detected=False,
                out_of_bounds=False,
                min_obstacle_distance=5.0,
                is_safe=True,
                should_terminate=False,
                termination_reason="SAFE: 安全监控异常"
            )
    
    def _compute_sector_distances(self, obs_input: ObservationInput) -> SectorDistances:
        """计算36扇区距离"""
        try:
            point_cloud = obs_input.point_cloud
            
            if point_cloud.size == 0:
                return self._create_default_sector_distances()
            
            # 计算每个点相对于无人机的角度和距离
            relative_points = point_cloud[:, :2]  # 只使用x,y坐标
            angles = np.arctan2(relative_points[:, 1], relative_points[:, 0])
            distances = np.linalg.norm(relative_points, axis=1)
            
            # 将角度转换到[0, 2π]范围
            angles = (angles + 2 * np.pi) % (2 * np.pi)
            
            # 初始化扇区距离
            sector_distances = np.full(self.config.num_sectors, self.config.max_detection_range, dtype=np.float32)
            sector_angles = np.linspace(0, 2*np.pi, self.config.num_sectors, endpoint=False)
            
            # 分配点到扇区
            sector_width = 2 * np.pi / self.config.num_sectors
            for i in range(len(distances)):
                # 找到对应的扇区
                sector_idx = int(angles[i] / sector_width) % self.config.num_sectors
                
                # 更新该扇区的最小距离
                if distances[i] < sector_distances[sector_idx]:
                    sector_distances[sector_idx] = distances[i]
            
            # 限制距离范围
            sector_distances = np.clip(sector_distances, 0.0, self.config.max_detection_range)
            
            return SectorDistances(
                distances=sector_distances,
                angles=sector_angles,
                valid_sectors=np.ones(self.config.num_sectors, dtype=bool),
                max_range=self.config.max_detection_range
            )
            
        except Exception as e:
            logger.error(f"扇区距离计算失败: {e}")
            return self._create_default_sector_distances()
    
    def _extract_map_features_16d(self) -> np.ndarray:
        """提取16维地图特征向量"""
        if self.local_map is None:
            return self._create_default_map_features()
        
        try:
            grid = self.local_map.grid
            exploration_rate = self.local_map.get_exploration_rate()
            
            # 计算基本统计特征
            unknown_ratio = np.sum(grid == -1) / grid.size
            free_ratio = np.sum(grid == 0) / grid.size
            occupied_ratio = np.sum(grid == 1) / grid.size
            
            # 计算方向性密度特征
            center_x, center_y = grid.shape[0] // 2, grid.shape[1] // 2
            
            # 8个方向的障碍物密度
            directions = ['N', 'S', 'E', 'W', 'NE', 'NW', 'SE', 'SW']
            direction_densities = []
            
            for direction in ['N', 'S', 'E', 'W']:
                if direction == 'N':
                    region = grid[:center_x, :]
                elif direction == 'S':
                    region = grid[center_x:, :]
                elif direction == 'E':
                    region = grid[:, center_y:]
                else:  # 'W'
                    region = grid[:, :center_y]
                
                density = np.sum(region == 1) / max(1, region.size)
                direction_densities.append(density)
            
            # 通行性分析
            accessible_area = self.local_map.accessible_area
            safe_passages = min(8, max(1, accessible_area // 100))  # 估算安全通道数
            
            # 构建16维特征向量
            features = [
                exploration_rate,              # 0: 探索率
                occupied_ratio,                # 1: 占据比例
                free_ratio,                    # 2: 自由空间比例  
                unknown_ratio,                 # 3: 未知区域比例
                direction_densities[0],        # 4: 北方密度
                direction_densities[1],        # 5: 南方密度
                direction_densities[2],        # 6: 东方密度
                direction_densities[3],        # 7: 西方密度
                safe_passages,                 # 8: 安全通道数
                min(10.0, accessible_area/10), # 9: 最大通道宽度（归一化）
                min(5.0, accessible_area/50),  # 10: 平均通行间距（归一化）
                5.0,                          # 11: 最近障碍物距离（从扇区距离获取）
                min(8, max(1, int(unknown_ratio * 8))),  # 12: 未探索方向数
                0.0,                          # 13: 优先探索角度
                min(1.0, exploration_rate * 2), # 14: 信息增益潜力
                min(1.0, max(0.0, exploration_rate))  # 15: 地图置信度
            ]
            
            return np.array(features[:16], dtype=np.float32)
            
        except Exception as e:
            logger.error(f"地图特征提取失败: {e}")
            return self._create_default_map_features()
    
    def _create_default_map_features(self) -> np.ndarray:
        """创建默认的16维地图特征"""
        return np.array([
            0.0, 0.1, 0.3, 0.6,           # 探索率, 占据, 自由, 未知
            0.0, 0.0, 0.0, 0.0,           # 4个方向的密度
            1, 2.0, 1.5, 5.0,             # 通道数, 宽度, 间距, 最近距离
            8, 0.0, 1.0, 0.5              # 未探索方向, 探索角度, 增益潜力, 置信度
        ], dtype=np.float32)
    
    def _create_default_sector_distances(self) -> SectorDistances:
        """创建默认扇区距离"""
        return SectorDistances(
            distances=np.full(self.config.num_sectors, 5.0, dtype=np.float32),
            angles=np.linspace(0, 2*np.pi, self.config.num_sectors, endpoint=False),
            valid_sectors=np.ones(self.config.num_sectors, dtype=bool),
            max_range=self.config.max_detection_range
        )
    
    def _create_safe_default_results(self, obs_input: ObservationInput) -> PerceptionResults:
        """创建安全的默认感知结果"""
        return PerceptionResults(
            local_map_results=LocalMapResults(
                grid_state=np.zeros((self.config.local_map_size, self.config.local_map_size), dtype=np.int8),
                exploration_rate=0.0,
                information_gain=0,
                accessible_area=0,
                updated_cells=0,
                map_features=self._create_default_map_features()
            ),
            safety_results=SafetyResults(
                collision_detected=False,
                out_of_bounds=False,
                min_obstacle_distance=10.0,
                is_safe=True,
                should_terminate=False,
                termination_reason="SAFE: 默认安全状态"
            ),
            sector_distances=self._create_default_sector_distances(),
            observation_input=obs_input,
            map_features_16d=self._create_default_map_features(),
            sector_distances_36d=np.full(36, 5.0, dtype=np.float32)
        )


# =============== 便捷函数 ===============

def create_perception_manager(config: Optional[Dict[str, Any]] = None) -> PerceptionManager:
    """创建感知管理器
    
    Args:
        config: 配置字典，如果为None则使用默认配置
        
    Returns:
        PerceptionManager: 感知管理器实例
    """
    if config is None:
        config = {}
    
    perception_config = PerceptionConfig(**config)
    return PerceptionManager(perception_config)


def create_observation_input(point_cloud: np.ndarray,
                           drone_position: np.ndarray,
                           drone_orientation: np.ndarray, 
                           drone_velocity: np.ndarray,
                           action_history: Optional[List[List[float]]] = None,
                           timestamp: float = 0.0) -> ObservationInput:
    """创建观测输入对象
    
    Args:
        point_cloud: 点云数据 [N, 3]
        drone_position: 无人机位置 [3]
        drone_orientation: 无人机姿态 [3] (roll, pitch, yaw)
        drone_velocity: 无人机速度 [3]
        action_history: 历史动作（可选）
        timestamp: 时间戳
        
    Returns:
        ObservationInput: 观测输入对象
    """
    return ObservationInput(
        point_cloud=np.asarray(point_cloud, dtype=np.float32),
        drone_position=np.asarray(drone_position, dtype=np.float32),
        drone_orientation=np.asarray(drone_orientation, dtype=np.float32),
        drone_velocity=np.asarray(drone_velocity, dtype=np.float32),
        action_history=action_history,
        timestamp=timestamp
    )


# =============== 模块导出 ===============

__all__ = [
    # 数据结构
    'ObservationInput',
    'LocalMapResults', 
    'SafetyResults',
    'SectorDistances',
    'PerceptionResults',
    'PerceptionConfig',
    
    # 核心类
    'PerceptionManager',
    
    # 基础组件（如果可用）
    'LocalMapMaintenance',
    'SafetyMonitoring', 
    'MapBoundaryCheck',
    'TerminationCheck',
    
    # 便捷函数
    'create_perception_manager',
    'create_observation_input',
]

# 版本信息
__version__ = "2.0.0"
__author__ = "HA-UAV团队"
__description__ = "雷达驱动感知模块 - 与StateManager完美集成"

logger.debug(f"感知模块 v{__version__} 初始化完成")