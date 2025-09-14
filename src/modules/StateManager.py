"""
StateManager.py - 统一状态管理器 - 纯雷达驱动版本

集成功能：
1. 86维观测解析 (DataAdapter功能)
2. 状态历史管理 (StateHistoryManager功能)  
3. 高层决策周期控制 (HighLevelDecisionCycle功能)
4. 局部地图状态集成

实现论文中的完整状态管理流程
"""

import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Union
import logging


logger = logging.getLogger(__name__)


@dataclass
class StructuredState:
    """结构化状态表示 - 优化版纯雷达驱动"""
    # 基础状态（保持兼容性）
    drone_position: List[float]
    drone_velocity: List[float]
    drone_orientation: List[float]
    target_position: List[float]
    obstacle_info: List[float]
    control_history: List[float]
    timestamp: float
    
    # 优化版雷达驱动字段
    sector_distances: Optional[List[float]] = None     # [36] 激光雷达扇区距离
    action_history: Optional[List[List[float]]] = None # [6, 4] 历史动作序列 ✓ 优化步数
    subgoal_sequence: Optional[List[List[float]]] = None # [5, 2] 子目标序列(ψ,d)
    motion_context: Optional[Dict[str, Any]] = None    # 运动上下文信息
    
    # 新增：局部地图状态字段
    local_map_features: Optional[List[float]] = None   # [16] 局部地图特征向量
    parsed_map_state: Optional[Dict[str, float]] = None # 解析后的地图状态
    
    def to_vector(self) -> np.ndarray:
        """转换为86维向量表示（与环境观测维度一致）"""
        vector_components = []
        
        # 1. 激光雷达特征 (36维) [0:36]
        if self.sector_distances is not None:
            vector_components.extend(self.sector_distances)
        else:
            vector_components.extend([5.0] * 36)
        
        # 2. 动作历史 (24维) [36:60] - 6步×4维动作
        # 为了与86维标准一致，需要包含动作历史
        if hasattr(self, 'action_history') and self.action_history is not None:
            if isinstance(self.action_history, list):
                # 如果是列表，转换为numpy数组
                action_array = np.array(self.action_history)
            else:
                action_array = self.action_history
            
            action_flat = action_array.flatten()[:24]  # 取前24维
            if len(action_flat) < 24:
                action_flat = np.pad(action_flat, (0, 24-len(action_flat)), 'constant')
            vector_components.extend(action_flat.tolist())
        else:
            vector_components.extend([0.0] * 24)
        
        # 3. 运动特征 (7维) [60:67]
        if self.motion_context is not None:
            vector_components.extend(self.motion_context['velocity'].tolist())  # 3维
            vector_components.extend([
                self.motion_context['yaw_rate'],                    # 1维
                self.motion_context['motion_consistency'],          # 1维
                self.motion_context.get('trajectory_curvature', 0.0), # 1维
                self.motion_context.get('speed_stability', 1.0),   # 1维
            ])
        else:
            vector_components.extend([0.0] * 7)
        
        # 4. 环境特征 (11维) [67:78]
        vector_components.extend(self.obstacle_info[:11])
        
        # 5. 局部地图核心特征 (8维) [78:86]
        if self.parsed_map_state is not None:
            map_features = [
                self.parsed_map_state.get('exploration_rate', 0.0),
                self.parsed_map_state.get('occupancy_ratio', 0.1),
                self.parsed_map_state.get('free_space_ratio', 0.3),
                self.parsed_map_state.get('safe_passages', 1) / 10.0,  # 归一化
                self.parsed_map_state.get('nearest_obstacle_dist', 5.0) / 10.0,  # 归一化
                self.parsed_map_state.get('information_gain_potential', 1.0),
                self.parsed_map_state.get('map_confidence', 0.5),
                self.parsed_map_state.get('exploration_progress', 0.0),  # 新增1维
            ]
            vector_components.extend(map_features)  # 8维
        else:
            vector_components.extend([0.0, 0.1, 0.3, 0.1, 0.5, 1.0, 0.5, 0.0])  # 8维默认值
        
        # 总维度: 36 + 24 + 7 + 11 + 8 = 86维
        result = np.array(vector_components, dtype=np.float32)
        
        # 确保精确86维
        if len(result) < 86:
            result = np.pad(result, (0, 86-len(result)), 'constant')
        elif len(result) > 86:
            result = result[:86]
            
        return result
    
    def get_occupancy_grid_representation(self) -> np.ndarray:
        """获取占据栅格表示（增强版）- 集成地图维护"""
        if self.sector_distances is None:
            return np.zeros((32, 32), dtype=np.float32)
        
        grid = np.zeros((32, 32), dtype=np.float32)
        center = np.array([16, 16])
        
        # 基于扇区距离构建栅格
        for i, distance in enumerate(self.sector_distances):
            angle = 2 * np.pi * i / len(self.sector_distances)
            
            # 标记自由空间和障碍物
            if distance < 15.0:  # 在探测范围内
                for r in np.arange(0.5, min(distance, 15.0), 0.5):
                    pos = center + r * np.array([np.cos(angle), np.sin(angle)])
                    pos = pos.astype(int)
                    if 0 <= pos[0] < 32 and 0 <= pos[1] < 32:
                        grid[pos[0], pos[1]] = 0.5  # 自由空间
                
                # 标记障碍物
                if distance < 15.0:
                    obstacle_pos = center + distance * np.array([np.cos(angle), np.sin(angle)])
                    obstacle_pos = obstacle_pos.astype(int)
                    if 0 <= obstacle_pos[0] < 32 and 0 <= obstacle_pos[1] < 32:
                        grid[obstacle_pos[0], obstacle_pos[1]] = 1.0  # 障碍物
        
        # 融合地图维护模块的信息
        if self.parsed_map_state is not None:
            confidence = self.parsed_map_state.get('map_confidence', 0.5)
            grid = grid * confidence  # 根据置信度调整栅格值
        
        return grid
    
    def get_yaw_history(self) -> np.ndarray:
        """获取yaw历史（6步版本）"""
        if self.action_history is not None:
            action_array = np.array(self.action_history)
            yaw_rates = action_array[:, 3] if action_array.shape[1] > 3 else np.zeros(len(action_array))
            
            # 积分获取yaw历史
            dt = 0.02
            yaw_history = np.cumsum(yaw_rates) * dt
            
            return yaw_history  # 6步历史
        else:
            return np.array([self.drone_orientation[2]] * 6)  # 使用当前yaw，6步


class StateManager:
    """
    统一状态管理器 - 集成所有状态相关功能
    
    集成功能：
    1. 86维观测解析 (原DataAdapter功能)
    2. 状态历史管理 (原StateHistoryManager功能)  
    3. 高层决策周期控制 (原HighLevelDecisionCycle功能)
    4. 局部地图状态集成
    """
    
    def __init__(self, 
                 history_length: int = 20,
                 high_level_update_frequency: int = 5,
                 future_horizon: int = 5):
        """初始化统一状态管理器
        
        Args:
            history_length: K步状态历史长度
            high_level_update_frequency: τ步高层更新频率
            future_horizon: T步未来子目标数量
        """
        # === 86维观测解析配置 ===
        self.obs_dim = 86
        self.offsets = {
            'sector_distances': 0,     # [0:36] 激光雷达扇区距离 - 36维
            'action_history': 36,      # [36:60] 历史动作序列 - 24维 (6步 × 4维)
            'subgoal_sequence': 60,    # [60:70] 子目标序列 - 10维 (5步 × 2维(ψ,d))
            'local_map_state': 70,     # [70:86] 局部地图状态 - 16维
        }
        
        # 维度定义
        self.sector_dim = 36
        self.history_steps = 6        # ✓ 优化：6步历史动作
        self.action_dim = 4
        self.subgoal_steps = 5
        self.subgoal_dim = 2
        self.map_state_dim = 16       # ✓ 地图特征维度
        
        # 验证维度一致性
        expected_dim = (self.sector_dim + 
                       (self.history_steps * self.action_dim) + 
                       (self.subgoal_steps * self.subgoal_dim) + 
                       self.map_state_dim)
        assert expected_dim == self.obs_dim, f"维度不匹配: {expected_dim} != {self.obs_dim}"
        
        # === 状态历史管理 ===
        self.history_length = history_length
        self.state_history = deque(maxlen=history_length)
        
        # === 高层决策周期控制 ===
        self.high_level_update_frequency = high_level_update_frequency  # τ=5步
        self.future_horizon = future_horizon                            # T=5步
        self.step_counter = 0
        self.current_subgoal_sequence = np.zeros((future_horizon, 2))   # (ψ, d)格式
        self.sequence_step_counter = 0
        self.last_high_level_update_step = 0
        
        # === 局部地图维护 ===
        try:
            from src.perceptions.local_map_maintenance import LocalMapMaintenance
            # 调整栅格大小以获得更合理的探索率增长
            # 200x200栅格 @ 0.1米分辨率 = 20x20米区域，适合室内房间探索
            self.local_map = LocalMapMaintenance(resolution=0.1, grid_size=(200, 200))
            logger.debug("LocalMapMaintenance集成成功")
        except ImportError:
            self.local_map = None
            logger.warning("LocalMapMaintenance未找到，使用模拟地图特征")
        
        logger.debug(f"StateManager初始化完成: K={history_length}, τ={high_level_update_frequency}, T={future_horizon}")
    
    def parse_and_update(self, obs_86d: Union[np.ndarray, torch.Tensor]) -> StructuredState:
        """解析86维观测并更新状态历史
        
        Args:
            obs_86d: 86维观测向量
            
        Returns:
            structured_state: 当前结构化状态
        """
        # 1. 解析86维观测
        structured_state = self._parse_observation(obs_86d)
        
        # 2. 添加到历史缓冲区
        structured_state.timestamp = self.step_counter
        self.state_history.append(structured_state)
        
        # 3. 步进计数器
        self.step_counter += 1
        self.sequence_step_counter += 1
        
        return structured_state
    
    def should_update_high_level(self) -> bool:
        """判断是否应该更新高层策略
        
        Returns:
            should_update: 是否需要高层更新
        """
        return (self.step_counter - self.last_high_level_update_step) >= self.high_level_update_frequency
    
    def update_subgoal_sequence(self, new_sequence: np.ndarray):
        """更新子目标序列
        
        Args:
            new_sequence: [T, 2] 新的子目标序列 (ψ, d)格式
        """
        self.current_subgoal_sequence = new_sequence.copy()
        self.sequence_step_counter = 0
        self.last_high_level_update_step = self.step_counter
        
    def get_current_subgoal(self) -> np.ndarray:
        """获取当前子目标
        
        Returns:
            current_subgoal: [2] 当前子目标 (ψ, d)格式
        """
        if self.sequence_step_counter < self.future_horizon:
            return self.current_subgoal_sequence[self.sequence_step_counter].copy()
        else:
            # 使用最后一个子目标
            return self.current_subgoal_sequence[-1].copy()
    
    def get_history_for_high_level_encoding(self) -> Optional[np.ndarray]:
        """获取用于高层编码的历史状态序列
        
        Returns:
            history_array: [K, state_dim] 历史状态数组，如果历史不足则返回None
        """
        if len(self.state_history) < self.history_length:
            return None
        
        # 将历史状态转换为向量形式
        history_vectors = []
        for state in list(self.state_history):
            history_vectors.append(state.to_vector())
        
        return np.stack(history_vectors, axis=0)  # [K, state_dim]
    
    def get_occupancy_grids_sequence(self) -> Optional[np.ndarray]:
        """获取占据栅格历史序列（用于StateEncoder）
        
        Returns:
            grids_sequence: [K, 32, 32] 占据栅格序列
        """
        if len(self.state_history) < self.history_length:
            return None
        
        grids = []
        for state in list(self.state_history):
            grids.append(state.get_occupancy_grid_representation())
        
        return np.stack(grids, axis=0)  # [K, 32, 32]
    
    def get_yaw_history_sequence(self) -> Optional[np.ndarray]:
        """获取yaw历史序列（用于StateEncoder）
        
        Returns:
            yaw_sequence: [K, 6] yaw历史序列
        """
        if len(self.state_history) < self.history_length:
            return None
        
        yaw_histories = []
        for state in list(self.state_history):
            yaw_histories.append(state.get_yaw_history())
        
        return np.stack(yaw_histories, axis=0)  # [K, 6]
    
    def convert_86d_to_hierarchical(self, observations: torch.Tensor) -> Dict[str, torch.Tensor]:
        """将86维观测批量转换为分层数据结构
        
        Args:
            observations: [batch_size, 86] 观测张量
            
        Returns:
            hierarchical_data: 分层数据字典
        """
        batch_size = observations.size(0)
        
        # 提取各组件
        sector_distances = observations[:, self.offsets['sector_distances']:
                                       self.offsets['sector_distances']+self.sector_dim]
        
        action_start = self.offsets['action_history']
        action_history = observations[:, action_start:action_start+(self.history_steps * self.action_dim)]
        action_history = action_history.view(batch_size, self.history_steps, self.action_dim)
        
        subgoal_start = self.offsets['subgoal_sequence']
        subgoal_sequence = observations[:, subgoal_start:subgoal_start+(self.subgoal_steps * self.subgoal_dim)]
        subgoal_sequence = subgoal_sequence.view(batch_size, self.subgoal_steps, self.subgoal_dim)
        
        map_start = self.offsets['local_map_state']
        local_map_features = observations[:, map_start:map_start+self.map_state_dim]
        
        return {
            'sector_distances': sector_distances,      # [batch_size, 36]
            'action_history': action_history,          # [batch_size, 6, 4] ✓ 优化步数
            'subgoal_sequence': subgoal_sequence,      # [batch_size, 5, 2]
            'current_subgoal': subgoal_sequence[:, 0], # [batch_size, 2] 当前子目标
            'local_map_features': local_map_features,  # [batch_size, 16] ✓ 地图特征
        }
    
    def is_ready_for_high_level(self) -> bool:
        """检查是否准备好进行高层决策
        
        Returns:
            is_ready: 是否有足够的历史状态
        """
        return len(self.state_history) >= self.history_length
    
    def reset(self):
        """重置状态管理器"""
        self.state_history.clear()
        self.step_counter = 0
        self.sequence_step_counter = 0
        self.last_high_level_update_step = 0
        self.current_subgoal_sequence.fill(0.0)
        
        if self.local_map:
            self.local_map.reset()
        
        # logger.info("StateManager已重置")
    
    # === 私有方法：86维观测解析 ===
    def _parse_observation(self, obs: Union[np.ndarray, torch.Tensor]) -> StructuredState:
        """解析86维观测为结构化状态
        
        Args:
            obs: 86维观测向量
            
        Returns:
            StructuredState: 结构化状态对象
        """
        if isinstance(obs, torch.Tensor):
            obs = obs.detach().cpu().numpy()
        if obs.ndim > 1:
            obs = obs.flatten()
            
        try:
            # 1. 提取激光雷达扇区距离 [0:36]
            sector_distances = obs[self.offsets['sector_distances']:
                                  self.offsets['sector_distances']+self.sector_dim]
            
            # 2. 提取历史动作序列 [36:60] (6步 × 4维)
            action_start = self.offsets['action_history']
            action_history = obs[action_start:action_start+(self.history_steps * self.action_dim)]
            action_history = action_history.reshape(self.history_steps, self.action_dim)
            
            # 3. 提取子目标序列 [60:70] (5步 × 2维)
            subgoal_start = self.offsets['subgoal_sequence']
            subgoal_sequence = obs[subgoal_start:subgoal_start+(self.subgoal_steps * self.subgoal_dim)]
            subgoal_sequence = subgoal_sequence.reshape(self.subgoal_steps, self.subgoal_dim)
            
            # 4. 提取局部地图状态 [70:86] (16维)
            map_state_start = self.offsets['local_map_state']
            local_map_features = obs[map_state_start:map_state_start+self.map_state_dim]
            
            # 5. 从激光雷达数据推导局部环境特征
            local_occupancy_features = self._extract_occupancy_features_from_sectors(sector_distances)
            
            # 6. 从动作历史提取运动特征（6步）
            motion_features = self._extract_motion_features(action_history)
            
            # 7. 当前朝向估计（基于动作历史）
            current_yaw = self._estimate_current_yaw(action_history)
            
            # 8. 解析局部地图特征
            parsed_map_state = self._parse_local_map_features(local_map_features)
            
            # 构建StructuredState
            return StructuredState(
                # 基础状态信息
                drone_position=[0.0, 0.0, 0.0],  # 占位符
                drone_velocity=motion_features['velocity'].tolist(),
                drone_orientation=[0.0, 0.0, current_yaw],
                target_position=[0.0, 0.0, 0.0],  # 不使用
                
                # 核心雷达驱动信息
                obstacle_info=local_occupancy_features.tolist(),
                control_history=action_history[-1].tolist(),
                
                # 优化版字段
                sector_distances=sector_distances.tolist(),
                action_history=action_history.tolist(),  # 6步历史
                subgoal_sequence=subgoal_sequence.tolist(),
                motion_context=motion_features,
                
                # 地图状态
                local_map_features=local_map_features.tolist(),
                parsed_map_state=parsed_map_state,
                
                timestamp=0.0
            )
            
        except Exception as e:
            logger.error(f"观测解析失败: {e}")
            return self._create_default_state(obs)
    
    def _extract_occupancy_features_from_sectors(self, sector_distances: np.ndarray) -> np.ndarray:
        """从扇区距离提取占据栅格特征"""
        features = []
        
        # 1. 基本统计特征
        features.extend([
            np.min(sector_distances),        # 最近障碍物距离
            np.mean(sector_distances),       # 平均距离
            np.max(sector_distances),        # 最远距离
            np.std(sector_distances),        # 距离标准差
        ])
        
        # 2. 方向性特征：8个主要方向的代表距离
        sectors_per_direction = len(sector_distances) // 8
        for i in range(8):
            start_idx = i * sectors_per_direction
            end_idx = (i + 1) * sectors_per_direction
            direction_distances = sector_distances[start_idx:end_idx]
            features.append(np.min(direction_distances))  # 该方向最近障碍物
        
        # 3. 通行性分析
        safe_threshold = 2.0  # 安全距离阈值
        safe_sectors = (sector_distances > safe_threshold).astype(float)
        
        # 连续安全扇区分析
        safe_segments = self._find_continuous_segments(safe_sectors)
        features.extend([
            len(safe_segments),  # 安全通道数量
            max([len(seg) for seg in safe_segments] + [0]),  # 最长安全通道长度
        ])
        
        # 4. 局部密度特征
        dense_threshold = 1.0
        dense_sectors = (sector_distances < dense_threshold).sum()
        features.append(dense_sectors / len(sector_distances))  # 密集障碍物比例
        
        return np.array(features, dtype=np.float32)
    
    def _extract_motion_features(self, action_history: np.ndarray) -> Dict[str, Any]:
        """从6步动作历史提取运动特征"""
        if len(action_history) == 0:
            return {
                'velocity': np.array([0.0, 0.0, 0.0]),
                'acceleration': np.array([0.0, 0.0, 0.0]),
                'yaw_rate': 0.0,
                'motion_consistency': 0.0,
                'trajectory_curvature': 0.0,
                'speed_stability': 1.0,
            }
        
        # 动作格式为 [vx, vy, vz, yaw_rate]
        velocities = action_history[:, :3]  # [6, 3]
        yaw_rates = action_history[:, 3]    # [6]
        
        # 当前速度估计（最近3步平均）
        recent_steps = min(3, len(velocities))
        current_velocity = np.mean(velocities[-recent_steps:], axis=0)
        
        # 加速度估计
        if len(velocities) >= 2:
            acceleration = velocities[-1] - velocities[-2]
        else:
            acceleration = np.zeros(3)
        
        # Yaw角速度
        current_yaw_rate = np.mean(yaw_rates[-recent_steps:]) if len(yaw_rates) > 0 else 0.0
        
        # 运动一致性（6步内的动作变化平滑度）
        if len(action_history) >= 2:
            action_diffs = np.diff(action_history, axis=0)
            motion_consistency = 1.0 / (1.0 + np.mean(np.linalg.norm(action_diffs, axis=1)))
        else:
            motion_consistency = 1.0
        
        # 轨迹曲率（基于yaw_rate变化）
        if len(yaw_rates) >= 3:
            yaw_rate_changes = np.abs(np.diff(yaw_rates))
            trajectory_curvature = np.mean(yaw_rate_changes)
        else:
            trajectory_curvature = 0.0
        
        # 速度稳定性
        if len(velocities) >= 3:
            velocity_magnitudes = np.linalg.norm(velocities, axis=1)
            speed_stability = 1.0 / (1.0 + np.std(velocity_magnitudes))
        else:
            speed_stability = 1.0
        
        return {
            'velocity': current_velocity,
            'acceleration': acceleration,
            'yaw_rate': current_yaw_rate,
            'motion_consistency': motion_consistency,
            'trajectory_curvature': trajectory_curvature,
            'speed_stability': speed_stability,
        }
    
    def _estimate_current_yaw(self, action_history: np.ndarray) -> float:
        """基于6步动作历史估计当前朝向"""
        if len(action_history) == 0:
            return 0.0
        
        # 使用yaw_rate积分估计
        yaw_rates = action_history[:, 3]
        
        # 时间步长
        dt = 0.02  # 50Hz控制频率
        
        # 积分估计yaw变化
        estimated_yaw = np.sum(yaw_rates) * dt
        
        # 限制在 [-π, π] 范围内
        estimated_yaw = ((estimated_yaw + np.pi) % (2 * np.pi)) - np.pi
        
        return estimated_yaw
    
    def _parse_local_map_features(self, map_features: np.ndarray) -> Dict[str, float]:
        """解析局部地图特征"""
        try:
            return {
                'exploration_rate': float(map_features[0]),
                'occupancy_ratio': float(map_features[1]),
                'free_space_ratio': float(map_features[2]),
                'unknown_ratio': float(map_features[3]),
                'north_density': float(map_features[4]),
                'south_density': float(map_features[5]),
                'east_density': float(map_features[6]),
                'west_density': float(map_features[7]),
                'safe_passages': int(map_features[8]),
                'max_passage_width': float(map_features[9]),
                'average_clearance': float(map_features[10]),
                'nearest_obstacle_dist': float(map_features[11]),
                'unexplored_directions': int(map_features[12]),
                'exploration_priority_angle': float(map_features[13]),
                'information_gain_potential': float(map_features[14]),
                'map_confidence': float(map_features[15]),
            }
        except:
            # 返回默认值
            return {
                'exploration_rate': 0.0, 'occupancy_ratio': 0.1, 'free_space_ratio': 0.3,
                'unknown_ratio': 0.6, 'north_density': 0.0, 'south_density': 0.0,
                'east_density': 0.0, 'west_density': 0.0, 'safe_passages': 1,
                'max_passage_width': 2.0, 'average_clearance': 1.5, 'nearest_obstacle_dist': 5.0,
                'unexplored_directions': 8, 'exploration_priority_angle': 0.0,
                'information_gain_potential': 1.0, 'map_confidence': 0.5
            }
    
    def _find_continuous_segments(self, binary_array: np.ndarray) -> List[List[int]]:
        """找到二进制数组中的连续1段"""
        segments = []
        current_segment = []
        
        for i, val in enumerate(binary_array):
            if val == 1:
                current_segment.append(i)
            else:
                if current_segment:
                    segments.append(current_segment)
                    current_segment = []
        
        if current_segment:
            segments.append(current_segment)
        
        return segments
    
    def _create_default_state(self, obs: np.ndarray) -> StructuredState:
        """创建默认安全状态"""
        return StructuredState(
            drone_position=[0.0, 0.0, 0.0],
            drone_velocity=[0.0, 0.0, 0.0],
            drone_orientation=[0.0, 0.0, 0.0],
            target_position=[0.0, 0.0, 0.0],
            obstacle_info=[0.0] * 15,
            control_history=[0.0, 0.0, 0.0, 0.0],
            sector_distances=obs[:36].tolist() if len(obs) >= 36 else [5.0] * 36,
            action_history=[[0.0, 0.0, 0.0, 0.0]] * 6,  # 6步历史
            subgoal_sequence=[[0.0, 1.0]] * 5,           # 5步子目标
            motion_context={
                'velocity': np.array([0.0, 0.0, 0.0]),
                'acceleration': np.array([0.0, 0.0, 0.0]),
                'yaw_rate': 0.0, 'motion_consistency': 1.0,
                'trajectory_curvature': 0.0, 'speed_stability': 1.0,
            },
            local_map_features=[0.0, 0.1, 0.3, 0.6, 0.0, 0.0, 0.0, 0.0, 
                              1, 2.0, 1.5, 5.0, 8, 0.0, 1.0, 0.5],
            parsed_map_state=self._parse_local_map_features(np.array([0.0] * 16)),
            timestamp=0.0
        )
    
    def extract_low_level_observation(self, obs_86d: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """直接从86维观测提取64维低层观测
        
        低层控制专用的局部感知数据：
        - 扇区距离：36维局部障碍物感知 [0:36]
        - 运动状态：12维速度/运动信息 [36:48]
        - 动作历史：8维最近2步动作 [48:56] 
        - 局部地图：8维局部占用信息 [78:86]
        
        Args:
            obs_86d: 86维观测，支持批处理 [batch_size, 86] 或 [86]
            
        Returns:
            torch.Tensor: [batch_size, 64] 或 [64] 低层观测张量
        """
        if isinstance(obs_86d, np.ndarray):
            obs_86d = torch.from_numpy(obs_86d).float()
        
        # 处理单个观测情况
        if obs_86d.dim() == 1:
            obs_86d = obs_86d.unsqueeze(0)
            single_obs = True
        else:
            single_obs = False
        
        batch_size = obs_86d.size(0)
        low_level_obs = []
        
        # 1. 扇区距离 - 36维 [0:36]
        sector_distances = obs_86d[:, 0:36]
        low_level_obs.append(sector_distances)
        
        # 2. 运动状态 - 12维：从动作历史推导
        action_start = 36
        action_history = obs_86d[:, action_start:action_start+24].view(batch_size, 6, 4)  # [batch_size, 6, 4]
        
        # 提取运动特征：当前速度(3维) + 角度信息(3维) + 运动统计(6维)
        current_velocity = torch.mean(action_history[:, -3:, :3], dim=1)  # 最近3步平均速度
        current_yaw_rate = torch.mean(action_history[:, -3:, 3], dim=1, keepdim=True)  # 当前角速度
        speed_magnitude = torch.norm(current_velocity, dim=1, keepdim=True)  # 速度大小
        
        # 加速度估计
        if action_history.size(1) >= 2:
            acceleration = action_history[:, -1, :3] - action_history[:, -2, :3]
        else:
            acceleration = torch.zeros_like(current_velocity)
        
        # 运动一致性
        action_diffs = torch.diff(action_history, dim=1)  # [batch_size, 5, 4]
        motion_consistency = 1.0 / (1.0 + torch.mean(torch.norm(action_diffs, dim=2), dim=1, keepdim=True))
        
        # 轨迹曲率
        yaw_rate_changes = torch.abs(torch.diff(action_history[:, :, 3], dim=1))  # [batch_size, 5]
        trajectory_curvature = torch.mean(yaw_rate_changes, dim=1, keepdim=True)
        
        motion_state = torch.cat([
            current_velocity,      # 3维：vx, vy, vz
            acceleration,          # 3维：ax, ay, az
            speed_magnitude,       # 1维：速度大小
            current_yaw_rate,      # 1维：角速度
            motion_consistency,    # 1维：运动一致性
            trajectory_curvature,  # 1维：轨迹曲率
            torch.zeros(batch_size, 2, device=obs_86d.device)  # 2维：padding
        ], dim=1)  # [batch_size, 12]
        low_level_obs.append(motion_state)
        
        # 3. 最近动作历史 - 8维（最近2步）
        recent_actions = action_history[:, -2:, :].reshape(batch_size, 8)  # [batch_size, 2, 4] -> [batch_size, 8]
        low_level_obs.append(recent_actions)
        
        # 4. 局部地图特征 - 8维 [78:86]
        local_map_features = obs_86d[:, 78:86]
        low_level_obs.append(local_map_features)
        
        # 拼接所有组件: 36 + 12 + 8 + 8 = 64
        result = torch.cat(low_level_obs, dim=1)
        
        return result.squeeze(0) if single_obs else result
    
    def extract_control_relevant_state(self, obs_86d: Union[torch.Tensor, np.ndarray]) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
        """提取用于增量控制的相关状态信息
        
        从86维观测中提取控制器所需的关键状态信息，支持增量控制决策
        
        Args:
            obs_86d: 86维观测向量
            
        Returns:
            control_state: 控制相关状态字典，包含：
                - current_velocity: [3] 当前速度估计
                - velocity_history: [6, 3] 最近6步速度历史
                - yaw_rate_history: [6] 最近6步角速度历史
                - obstacle_proximity: [8] 8个方向的最近障碍物距离
                - motion_stability: float 运动稳定性指标
                - control_confidence: float 控制置信度
        """
        if isinstance(obs_86d, np.ndarray):
            obs_86d = torch.from_numpy(obs_86d).float()
        
        # 处理单个观测情况
        if obs_86d.dim() == 1:
            obs_86d = obs_86d.unsqueeze(0)
            single_obs = True
        else:
            single_obs = False
        
        batch_size = obs_86d.size(0)
        
        # 1. 提取动作历史 [36:60] -> [6, 4]
        action_start = 36
        action_history = obs_86d[:, action_start:action_start+24].view(batch_size, 6, 4)
        
        # 2. 当前速度估计（最近3步平均）
        current_velocity = torch.mean(action_history[:, -3:, :3], dim=1)  # [batch_size, 3]
        
        # 3. 速度历史
        velocity_history = action_history[:, :, :3]  # [batch_size, 6, 3]
        
        # 4. Yaw率历史
        yaw_rate_history = action_history[:, :, 3]   # [batch_size, 6]
        
        # 5. 障碍物邻近度分析（8个主要方向）
        sector_distances = obs_86d[:, 0:36]  # [batch_size, 36]
        sectors_per_direction = 36 // 8
        obstacle_proximity = []
        
        for i in range(8):
            start_idx = i * sectors_per_direction
            end_idx = (i + 1) * sectors_per_direction
            direction_min_dist = torch.min(sector_distances[:, start_idx:end_idx], dim=1)[0]
            obstacle_proximity.append(direction_min_dist)
        
        obstacle_proximity = torch.stack(obstacle_proximity, dim=1)  # [batch_size, 8]
        
        # 6. 运动稳定性指标
        if action_history.size(1) >= 2:
            action_diffs = torch.diff(action_history, dim=1)  # [batch_size, 5, 4]
            motion_variations = torch.norm(action_diffs, dim=2)  # [batch_size, 5]
            motion_stability = 1.0 / (1.0 + torch.mean(motion_variations, dim=1))  # [batch_size]
        else:
            motion_stability = torch.ones(batch_size)
        
        # 7. 控制置信度（基于环境可预测性）
        # 基于障碍物分布的复杂性
        obstacle_variance = torch.var(sector_distances, dim=1)  # [batch_size]
        min_clearance = torch.min(sector_distances, dim=1)[0]   # [batch_size]
        
        # 综合置信度：运动稳定性 + 环境可预测性
        control_confidence = motion_stability * torch.exp(-obstacle_variance / 10.0) * torch.clamp(min_clearance / 2.0, 0, 1)
        
        # 构建结果字典
        control_state = {
            'current_velocity': current_velocity,           # [batch_size, 3]
            'velocity_history': velocity_history,           # [batch_size, 6, 3]
            'yaw_rate_history': yaw_rate_history,          # [batch_size, 6]
            'obstacle_proximity': obstacle_proximity,       # [batch_size, 8]
            'motion_stability': motion_stability,           # [batch_size]
            'control_confidence': control_confidence,       # [batch_size]
            'min_clearance': min_clearance,                # [batch_size]
            'sector_distances': sector_distances,           # [batch_size, 36] 完整扇区距离
        }
        
        # 如果是单个观测，移除batch维度
        if single_obs:
            control_state = {k: v.squeeze(0) if isinstance(v, torch.Tensor) else v 
                           for k, v in control_state.items()}
        
        return control_state
    
    def compute_increment_control_feedback(self, 
                                         current_state: Dict[str, torch.Tensor],
                                         target_increment: Union[torch.Tensor, np.ndarray],
                                         safety_threshold: float = 1.0) -> Dict[str, float]:
        """为增量控制计算反馈信息
        
        Args:
            current_state: 从extract_control_relevant_state获得的状态
            target_increment: [4] 目标增量 [dvx, dvy, dvz, dyaw]
            safety_threshold: 安全距离阈值
            
        Returns:
            feedback: 控制反馈字典
        """
        if isinstance(target_increment, np.ndarray):
            target_increment = torch.from_numpy(target_increment).float()
        
        # 1. 安全性检查
        min_clearance = current_state['min_clearance']
        safety_factor = torch.clamp(min_clearance / safety_threshold, 0, 1).item()
        
        # 2. 运动连续性检查
        motion_stability = current_state['motion_stability'].item()
        
        # 3. 增量合理性检查
        velocity_magnitude = torch.norm(target_increment[:3]).item()
        yaw_rate_magnitude = abs(target_increment[3].item())
        
        # 速度增量限制（基于安全距离调整）
        max_safe_velocity_increment = min(2.0 * safety_factor, 1.5)
        max_safe_yaw_rate = min(1.0 * safety_factor, 0.8)
        
        velocity_feasibility = 1.0 if velocity_magnitude <= max_safe_velocity_increment else max_safe_velocity_increment / velocity_magnitude
        yaw_feasibility = 1.0 if yaw_rate_magnitude <= max_safe_yaw_rate else max_safe_yaw_rate / yaw_rate_magnitude
        
        # 4. 综合控制质量
        control_quality = safety_factor * motion_stability * min(velocity_feasibility, yaw_feasibility)
        
        return {
            'safety_factor': safety_factor,
            'motion_stability': motion_stability,
            'velocity_feasibility': velocity_feasibility,
            'yaw_feasibility': yaw_feasibility,
            'control_quality': control_quality,
            'recommended_velocity_scale': velocity_feasibility,
            'recommended_yaw_scale': yaw_feasibility,
        }

    def extract_high_level_observation_sequence(self, obs_86d: Union[torch.Tensor, np.ndarray], K: int = 5) -> torch.Tensor:
        """直接从86维观测构建28维×K步高层观测序列
        
        高层规划相关信息：
        - 全局地图状态：16维探索率/区域统计 [70:86]
        - 长期运动模式：8维运动统计信息（从动作历史计算）
        - 子目标执行状态：4维当前子目标信息 [60:64]
        
        Args:
            obs_86d: 86维观测，支持批处理 [batch_size, 86] 或 [86]
            K: 序列长度，默认5步
            
        Returns:
            torch.Tensor: [batch_size, K, 28] 或 [K, 28] 高层观测序列
        """
        if isinstance(obs_86d, np.ndarray):
            obs_86d = torch.from_numpy(obs_86d).float()
        
        # 处理单个观测情况
        if obs_86d.dim() == 1:
            obs_86d = obs_86d.unsqueeze(0)
            single_obs = True
        else:
            single_obs = False
        
        batch_size = obs_86d.size(0)
        
        # 为每个batch构建高层观测
        high_level_obs = []
        
        # 1. 全局地图状态 - 16维 [70:86]
        global_map_features = obs_86d[:, 70:86]  # [batch_size, 16]
        
        # 2. 长期运动模式 - 8维：从动作历史计算
        action_start = 36
        action_history = obs_86d[:, action_start:action_start+24].view(batch_size, 6, 4)  # [batch_size, 6, 4]
        
        # 计算长期运动统计
        avg_velocity = torch.mean(action_history[:, :, :3], dim=1)  # [batch_size, 3] 平均速度
        avg_speed = torch.norm(avg_velocity, dim=1, keepdim=True)   # [batch_size, 1] 平均速度大小
        
        # 运动方向一致性
        velocity_directions = F.normalize(action_history[:, :, :3], p=2, dim=2)  # [batch_size, 6, 3]
        direction_consistency = torch.mean(torch.sum(velocity_directions[:, :-1] * velocity_directions[:, 1:], dim=2), dim=1, keepdim=True)
        
        # Yaw变化率
        yaw_rates = action_history[:, :, 3]  # [batch_size, 6]
        avg_yaw_rate = torch.mean(yaw_rates, dim=1, keepdim=True)  # [batch_size, 1]
        yaw_stability = 1.0 / (1.0 + torch.std(yaw_rates, dim=1, keepdim=True))  # [batch_size, 1]
        
        # 轨迹复杂度
        position_changes = torch.cumsum(action_history[:, :, :3], dim=1)  # 估计位置变化
        trajectory_length = torch.sum(torch.norm(torch.diff(position_changes, dim=1), dim=2), dim=1, keepdim=True)  # [batch_size, 1]
        
        long_term_motion = torch.cat([
            avg_speed,              # 1维：平均速度大小
            direction_consistency,  # 1维：方向一致性
            avg_yaw_rate,          # 1维：平均角速度
            yaw_stability,         # 1维：角度稳定性
            trajectory_length,      # 1维：轨迹复杂度
            torch.zeros(batch_size, 3, device=obs_86d.device)  # 3维：padding
        ], dim=1)  # [batch_size, 8]
        
        # 3. 子目标执行状态 - 4维：从子目标序列 [60:70]
        subgoal_start = 60
        subgoal_sequence = obs_86d[:, subgoal_start:subgoal_start+10].view(batch_size, 5, 2)  # [batch_size, 5, 2]
        current_subgoal = subgoal_sequence[:, 0, :]  # [batch_size, 2] 当前子目标 (ψ, d)
        
        # 子目标进度和执行状态（简化版）
        subgoal_execution = torch.cat([
            current_subgoal,  # 2维：当前子目标 (ψ, d)
            torch.zeros(batch_size, 2, device=obs_86d.device)  # 2维：进度和偏差（简化为0）
        ], dim=1)  # [batch_size, 4]
        
        # 拼接所有高层特征：16 + 8 + 4 = 28
        single_high_level_obs = torch.cat([
            global_map_features,  # [batch_size, 16]
            long_term_motion,     # [batch_size, 8]
            subgoal_execution     # [batch_size, 4]
        ], dim=1)  # [batch_size, 28]
        
        # 构建K步序列（这里简化为重复当前观测K次，实际应该是历史序列）
        high_level_sequence = single_high_level_obs.unsqueeze(1).expand(-1, K, -1)  # [batch_size, K, 28]
        
        return high_level_sequence.squeeze(0) if single_obs else high_level_sequence
    
    def convert_to_training_format(self, obs_batch: Union[torch.Tensor, np.ndarray]) -> Dict[str, torch.Tensor]:
        """为SB3训练转换数据格式
        
        直接从86维观测批次转换为训练所需格式
        
        Args:
            obs_batch: [batch_size, 86] 观测批次
            
        Returns:
            dict: 训练数据字典
        """
        # 转换为torch tensor
        if isinstance(obs_batch, np.ndarray):
            obs_batch = torch.from_numpy(obs_batch).float()
        
        batch_size = obs_batch.size(0)
        
        # 提取低层观测
        low_level_obs = self.extract_low_level_observation(obs_batch)  # [batch_size, 64]
        
        # 提取高层观测序列
        high_level_obs = self.extract_high_level_observation_sequence(obs_batch, K=5)  # [batch_size, 5, 28]
        
        return {
            'low_level_obs': low_level_obs,      # [batch_size, 64]
            'high_level_obs': high_level_obs,    # [batch_size, 5, 28]
            'original_obs': obs_batch            # [batch_size, 86]
        }