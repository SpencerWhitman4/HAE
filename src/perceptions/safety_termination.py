"""
simplified_safety_termination.py - 简化版安全监控与终止检查模块

严格按照JSON规范实现的极简版本，仅关注：
1. 障碍物距离检测和碰撞判断
2. 地图边界检测
3. 基于上述条件的终止决策

禁止添加任何额外功能、优化方法或复杂逻辑。
"""
import numpy as np
from typing import Dict, Any


class SafetyMonitoring:
    """极简安全监控模块，仅关注障碍物距离和碰撞状态"""
    
    @staticmethod
    def check_collision(point_cloud: np.ndarray, collision_threshold: float = 0.1) -> bool:
        """
        检测无人机是否与障碍物发生碰撞
        
        Args:
            point_cloud: 雷达点云数据，形状为(N, 3)，N为点云数量，每个点包含(x, y, z)坐标
            collision_threshold: 碰撞判定阈值（米），默认0.1
            
        Returns:
            bool: 是否发生碰撞，True=碰撞发生，False=无碰撞
        """
        if len(point_cloud) == 0:
            return False
        
        # 计算所有点云数据点到无人机中心的距离（假设无人机在原点）
        distances = np.linalg.norm(point_cloud, axis=1)
        
        # 检查是否存在距离小于碰撞阈值的点
        return bool(np.any(distances < collision_threshold))
    
    @staticmethod
    def get_min_obstacle_distance(point_cloud: np.ndarray) -> float:
        """
        计算到最近障碍物的距离
        
        Args:
            point_cloud: 雷达点云数据，形状为(N, 3)
            
        Returns:
            float: 到最近障碍物的距离（米），无距离时返回10.0米
        """
        if len(point_cloud) == 0:
            return 10.0
        
        # 计算所有点云数据点到无人机中心的距离
        distances = np.linalg.norm(point_cloud, axis=1)
        
        # 返回最小距离值
        return float(np.min(distances))
    
    @staticmethod
    def is_safe(point_cloud: np.ndarray, safety_threshold: float = 0.5) -> bool:
        """
        检查当前是否处于安全状态
        
        Args:
            point_cloud: 雷达点云数据，形状为(N, 3)
            safety_threshold: 安全距离阈值（米），默认0.5
            
        Returns:
            bool: 是否安全，True=安全，False=不安全
        """
        # 调用get_min_obstacle_distance获取最小距离
        min_distance = SafetyMonitoring.get_min_obstacle_distance(point_cloud)
        
        # 与安全阈值比较
        return bool(min_distance >= safety_threshold)
    
    @staticmethod
    def monitor_safety(point_cloud: np.ndarray, 
                      collision_threshold: float = 0.1, 
                      safety_threshold: float = 0.5) -> Dict[str, Any]:
        """
        安全监控主接口
        
        Args:
            point_cloud: 雷达点云数据，形状为(N, 3)
            collision_threshold: 碰撞判定阈值（米），默认0.1
            safety_threshold: 安全距离阈值（米），默认0.5
            
        Returns:
            dict: 包含关键安全指标的字典
            {
                "collision": bool - 是否发生碰撞,
                "min_distance": float - 到最近障碍物的距离（米）,
                "is_safe": bool - 是否处于安全状态
            }
        """
        # 调用基础函数进行碰撞检测和距离计算
        collision = SafetyMonitoring.check_collision(point_cloud, collision_threshold)
        min_distance = SafetyMonitoring.get_min_obstacle_distance(point_cloud)
        is_safe = SafetyMonitoring.is_safe(point_cloud, safety_threshold)
        
        # 返回包含关键安全指标的字典
        return {
            "collision": bool(collision),
            "min_distance": float(min_distance),
            "is_safe": bool(is_safe)
        }


class MapBoundaryCheck:
    """地图边界检测模块，与安全监控完全整合"""
    
    @staticmethod
    def check_map_boundary(position: np.ndarray, map_bounds: Dict[str, float]) -> bool:
        """
        检查无人机是否越出地图边界
        
        Args:
            position: 无人机当前位置，形状为(3,)，包含[x, y, z]
            map_bounds: 地图边界坐标字典，格式为:
                {
                    "x_min": float, "x_max": float,
                    "y_min": float, "y_max": float,
                    "z_min": float, "z_max": float
                }
            
        Returns:
            bool: 是否越出地图边界，True=越界，False=在边界内
        """
        # 检查边界字典是否有效
        if map_bounds is None:
            return False  # 没有边界信息时认为不越界
        
        # 获取当前位置坐标
        x, y, z = position[0], position[1], position[2]
        
        # 检查是否越出边界
        out_of_bounds = (x < map_bounds.get('x_min', -10.0) or x > map_bounds.get('x_max', 10.0) or 
                        y < map_bounds.get('y_min', -10.0) or y > map_bounds.get('y_max', 10.0) or
                        z < map_bounds.get('z_min', 0.1) or z > map_bounds.get('z_max', 5.0))
        
        return bool(out_of_bounds)



class TerminationCheck:
    """简化终止检查模块，与安全监控和地图边界检测完全整合"""
    
    @staticmethod
    def check_termination_condition(min_distance: float, 
                                   is_out_of_bounds: bool, 
                                   collision_threshold: float = 0.1) -> bool:
        """
        检查是否需要终止episode（基于碰撞和地图边界条件）
        
        Args:
            min_distance: 到最近障碍物的距离（米）
            is_out_of_bounds: 是否越出地图边界
            collision_threshold: 碰撞判定阈值（米），默认0.1
            
        Returns:
            bool: 是否需要终止episode，True=需要终止，False=继续
        """
        # 直接比较最小距离与碰撞阈值
        collision_detected = min_distance < collision_threshold
        
        # 如果任一条件满足，返回True表示需要终止
        return bool(collision_detected or is_out_of_bounds)
    
    @staticmethod
    def get_termination_reason(min_distance: float, 
                              is_out_of_bounds: bool, 
                              collision_threshold: float = 0.1) -> str:
        """
        获取终止原因
        
        Args:
            min_distance: 到最近障碍物的距离（米）
            is_out_of_bounds: 是否越出地图边界
            collision_threshold: 碰撞判定阈值（米），默认0.1
            
        Returns:
            str: 终止原因
            - "COLLISION: 与障碍物发生碰撞"
            - "MAP_BOUNDARY: 越出地图边界"  
            - "SAFE: 无需终止，继续飞行"
        """
        # 检查是否满足碰撞条件
        if min_distance < collision_threshold:
            return "COLLISION: 与障碍物发生碰撞"
        
        # 检查是否越出地图边界
        if is_out_of_bounds:
            return "MAP_BOUNDARY: 越出地图边界"
        
        # 返回相应的终止原因字符串
        return "SAFE: 无需终止，继续飞行"
    
    @staticmethod
    def is_episode_active(min_distance: float, 
                         is_out_of_bounds: bool, 
                         collision_threshold: float = 0.1) -> bool:
        """
        检查episode是否处于活动状态
        
        Args:
            min_distance: 到最近障碍物的距离（米）
            is_out_of_bounds: 是否越出地图边界
            collision_threshold: 碰撞判定阈值（米），默认0.1
            
        Returns:
            bool: episode是否活跃，True=活跃，False=已终止
        """
        # 调用check_termination_condition判断是否需要终止
        should_terminate = TerminationCheck.check_termination_condition(
            min_distance, is_out_of_bounds, collision_threshold
        )
        
        # 返回相反的结果表示episode是否活跃
        return bool(not should_terminate)
    
    @staticmethod
    def monitor_safety_and_termination(point_cloud: np.ndarray,
                                      map_bounds: Dict[str, float],
                                      position: np.ndarray,
                                      collision_threshold: float = 0.1) -> Dict[str, Any]:
        """
        安全监控与终止检查的整合接口（包含地图边界检查）
        
        Args:
            point_cloud: 雷达点云数据，形状为(N, 3)
            map_bounds: 地图边界坐标字典，格式为:
                {
                    "x_min": float, "x_max": float,
                    "y_min": float, "y_max": float,
                    "z_min": float, "z_max": float
                }
            position: 无人机当前位置，形状为(3,)，包含[x, y, z]
            collision_threshold: 碰撞判定阈值（米），默认0.1
            
        Returns:
            dict: 包含安全状态和终止决策的字典
            {
                "collision": bool - 是否发生碰撞,
                "out_of_bounds": bool - 是否越出地图边界,
                "min_distance": float - 到最近障碍物的距离（米）,
                "is_safe": bool - 是否处于安全状态,
                "terminate": bool - 是否需要终止episode,
                "termination_reason": str - 终止原因（COLLISION或MAP_BOUNDARY）
            }
        """
        # 计算到最近障碍物的距离
        min_distance = SafetyMonitoring.get_min_obstacle_distance(point_cloud)
        
        # 判断是否发生碰撞
        collision = SafetyMonitoring.check_collision(point_cloud, collision_threshold)
        
        # 检查是否处于安全状态
        is_safe = SafetyMonitoring.is_safe(point_cloud, 0.5)  # 使用默认安全阈值0.5
        
        # 检查是否越出地图边界
        out_of_bounds = MapBoundaryCheck.check_map_boundary(position, map_bounds)
        
        # 检查是否需要终止
        terminate = TerminationCheck.check_termination_condition(
            min_distance, out_of_bounds, collision_threshold
        )
        
        # 获取终止原因
        termination_reason = TerminationCheck.get_termination_reason(
            min_distance, out_of_bounds, collision_threshold
        )
        
        # 返回包含安全状态和终止决策的字典
        return {
            "collision": bool(collision),
            "out_of_bounds": bool(out_of_bounds),
            "min_distance": float(min_distance),
            "is_safe": bool(is_safe),
            "terminate": bool(terminate),
            "termination_reason": str(termination_reason)
        }
