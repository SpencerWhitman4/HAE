"""
障碍物地图配置文件
统一管理所有地图场景的障碍物配置
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pybullet as p


class MapManager:
    
    def __init__(self, map_json_path: str = "room_map.json"):
        """
        初始化地图管理器
        
        参数:
        ----
        map_json_path : str
            地图配置文件路径，默认为当前目录下的 room_map.json
        """
        self.map_json_path = Path(map_json_path)
        self.obstacle_maps = self._load_obstacle_maps()
        
        # 当前加载的地图状态
        self.current_map_name = None
        self.current_map_config = None
        self.pybullet_client = None
        self.loaded_obstacles = []  # 存储已加载的障碍物ID
    
    def _load_obstacle_maps(self):
        """加载地图配置文件"""
        with open(self.map_json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def set_pybullet_client(self, client):
        """设置PyBullet客户端"""
        self.pybullet_client = client
    
    def load_map(self, map_name: str):
        """
        加载指定的地图到PyBullet环境中
        
        参数:
        ----
        map_name : str
            地图名称
        """
        if self.pybullet_client is None:
            raise ValueError("请先设置PyBullet客户端")
        
        if map_name not in self.obstacle_maps:
            print(f"⚠️ 未找到地图: {map_name}")
            return  # 早期返回，不加载任何障碍物
        
        # 清理之前加载的障碍物
        self.clear_current_map()
        
        # 加载新地图
        self.current_map_name = map_name
        self.current_map_config = self.obstacle_maps[map_name].copy()
        
        # 加载障碍物
        for i, obstacle in enumerate(self.current_map_config['obstacles']):
            obstacle_id = self._load_single_obstacle(obstacle)
            self.loaded_obstacles.append(obstacle_id)
        
    
    def _load_single_obstacle(self, obstacle_config: dict) -> int:
        """
        加载单个障碍物到PyBullet环境
        
        参数:
        ----
        obstacle_config : dict
            障碍物配置
            
        返回:
        ----
        int
            障碍物在PyBullet中的ID
        """
        urdf_path = obstacle_config['urdf_path']
        position = obstacle_config['position']
        
        # 处理URDF路径
        if not urdf_path.startswith('/') and not urdf_path.startswith('assets/'):
            project_root = Path(__file__).parent.parent.parent
            assets_path = project_root / "assets" / urdf_path
            urdf_path = str(assets_path)
        
        # 加载障碍物
        obstacle_id = p.loadURDF(
            urdf_path, 
            position, 
            p.getQuaternionFromEuler([0, 0, 0]),
            physicsClientId=self.pybullet_client
        )
        
        # 设置为静态物体
        p.changeDynamics(obstacle_id, -1, mass=0, physicsClientId=self.pybullet_client)
        
        return obstacle_id
    
    
    def clear_current_map(self):
        """清理当前加载的地图"""
        if self.pybullet_client is None:
            return
        
        # 移除所有已加载的障碍物
        for obstacle_id in self.loaded_obstacles:
            try:
                # 先检查body是否存在
                info = p.getBodyInfo(obstacle_id, physicsClientId=self.pybullet_client)
                if info:  # 如果body存在，才尝试删除
                    p.removeBody(obstacle_id, physicsClientId=self.pybullet_client)
            except Exception as e:
                # 静默处理错误，避免警告输出
                pass
        
        self.loaded_obstacles.clear()
        self.current_map_name = None
        self.current_map_config = None
    
    def get_current_map_info(self) -> dict:
        """获取当前加载地图的信息"""
        if self.current_map_config is None:
            return None
        
        return {
            'name': self.current_map_name,
            'config': self.current_map_config,
            'loaded_obstacles_count': len(self.loaded_obstacles)
        }
    
    def get_map_info(self) -> Optional[Dict[str, Any]]:
        """获取当前地图的详细信息（用于显示）"""
        if self.current_map_config is None:
            return None
        
        return self.current_map_config.copy()
    
    def get_start_position(self) -> List[float]:
        """获取当前地图的起始位置"""
        if self.current_map_config is None:
            raise ValueError("未加载任何地图")
        return self.current_map_config['start_pos']
    
    def get_map_bounds(self) -> Dict[str, float]:
        """
        获取当前地图的边界信息
        
        Returns:
        --------
        dict: 包含边界信息的字典
            x_min, x_max, y_min, y_max, z_min, z_max
        """
        if self.current_map_config is None:
            # 返回默认边界
            return {
                'x_min': -10.0, 'x_max': 10.0,
                'y_min': -10.0, 'y_max': 10.0, 
                'z_min': 0.1, 'z_max': 5.0
            }
        
        # 从地图配置中获取边界，如果没有则使用默认值
        bounds = self.current_map_config.get('bounds', {})
        return {
            'x_min': bounds.get('x_min', -10.0),
            'x_max': bounds.get('x_max', 10.0),
            'y_min': bounds.get('y_min', -10.0),
            'y_max': bounds.get('y_max', 10.0),
            'z_min': bounds.get('z_min', 0.1),
            'z_max': bounds.get('z_max', 5.0)
        }
    
    def get_min_distance_to_obstacles(self, drone_pos: Tuple[float, float, float]) -> float:
        """
        获取无人机到最近障碍物的距离
        
        参数:
        ----
        drone_pos : Tuple[float, float, float]
            无人机位置
            
        返回:
        ----
        float
            到最近障碍物的距离
        """
        if not self.current_map_config or not self.current_map_config['obstacles']:
            return float('inf')
        
        drone_pos = np.array(drone_pos)
        min_distance = float('inf')
        
        for obstacle in self.current_map_config['obstacles']:
            obs_pos = np.array(obstacle['position'])
            center_distance = np.linalg.norm(drone_pos - obs_pos)
            obstacle_radius = obstacle.get('radius', 0.3)
            distance_to_surface = center_distance - obstacle_radius
            min_distance = min(min_distance, distance_to_surface)
        
        return max(0.0, min_distance)
    
    def list_available_maps(self) -> List[str]:
        """列出所有可用的地图"""
        return list(self.obstacle_maps.keys())
    
    def print_map_info(self, map_name: str = None):
        """打印地图信息"""
        if map_name is None:
            print("🗺️  可用地图列表:")
            print("=" * 50)
            for name, config in self.obstacle_maps.items():
                status = "✅ 已加载" if name == self.current_map_name else ""
                print(f"📍 {name} {status}")
                print(f"   名称: {config['name']}")
                print(f"   描述: {config['description']}")
                print(f"   障碍物数量: {len(config['obstacles'])}")
                print()
        else:
            if map_name not in self.obstacle_maps:
                print(f"❌ 未找到地图: {map_name}")
                return
            
            config = self.obstacle_maps[map_name]
            status = "✅ 已加载" if map_name == self.current_map_name else ""
            print(f"🗺️  地图信息: {map_name} {status}")
            print("=" * 50)
            print(f"名称: {config['name']}")
            print(f"描述: {config['description']}")
            print(f"目标位置: {config['target_pos']}")
            print(f"起始位置: {config['start_pos']}")
            print(f"障碍物数量: {len(config['obstacles'])}")
    
    def get_max_obstacles_count(self) -> int:
        """获取所有地图中的最大障碍物数量"""
        return max(len(config['obstacles']) for config in self.obstacle_maps.values())
    
    def get_map_config(self, map_name: str) -> Optional[dict]:
        """
        获取指定地图的配置
        
        参数:
        ----
        map_name : str
            地图名称
            
        返回:
        ----
        dict or None
            地图配置字典，如果未找到地图则返回None
        """
        return self.obstacle_maps.get(map_name, None)
    
    def get_obstacle_distance(self, drone_pos: Tuple[float, float, float], obstacle_id: int = None) -> float:
        """
        获取无人机到指定障碍物或最近障碍物的距离
        
        参数:
        ----
        drone_pos : Tuple[float, float, float]
            无人机位置
        obstacle_id : int, optional
            障碍物ID，如果不指定则返回到最近障碍物的距离
            
        返回:
        ----
        float
            到障碍物的距离
        """
        if obstacle_id is not None:
            # 返回到指定障碍物的距离
            if self.pybullet_client is not None and obstacle_id in self.loaded_obstacles:
                try:
                    # 获取障碍物位置
                    obstacle_pos, _ = p.getBasePositionAndOrientation(
                        obstacle_id, physicsClientId=self.pybullet_client
                    )
                    drone_pos = np.array(drone_pos)
                    obs_pos = np.array(obstacle_pos)
                    return np.linalg.norm(drone_pos - obs_pos)
                except:
                    pass
            return float('inf')
        else:
            # 返回到最近障碍物的距离
            return self.get_min_distance_to_obstacles(drone_pos)
    
    def check_collision(self, drone_pos: Tuple[float, float, float], safety_radius: float = 0.5) -> bool:
        """
        检查无人机是否与障碍物发生碰撞
        
        参数:
        ----
        drone_pos : Tuple[float, float, float]
            无人机位置
        safety_radius : float
            安全半径
            
        返回:
        ----
        bool
            是否发生碰撞
        """
        min_distance = self.get_min_distance_to_obstacles(drone_pos)
        return min_distance < safety_radius
    
    def get_all_obstacles_distances(self, drone_pos: Tuple[float, float, float]) -> List[float]:
        """
        获取无人机到所有障碍物的距离列表
        
        参数:
        ----
        drone_pos : Tuple[float, float, float]
            无人机位置
            
        返回:
        ----
        List[float]
            到所有障碍物的距离列表
        """
        if not self.current_map_config or not self.current_map_config['obstacles']:
            return []
        
        drone_pos = np.array(drone_pos)
        distances = []
        
        for obstacle in self.current_map_config['obstacles']:
            obs_pos = np.array(obstacle['position'])
            center_distance = np.linalg.norm(drone_pos - obs_pos)
            obstacle_radius = obstacle.get('radius', 0.3)
            distance_to_surface = center_distance - obstacle_radius
            distances.append(max(0.0, distance_to_surface))
        
        return distances
    
    def is_position_safe(self, position: Tuple[float, float, float], safety_radius: float = 0.5) -> bool:
        """
        检查指定位置是否安全（不与障碍物碰撞）
        
        参数:
        ----
        position : Tuple[float, float, float]
            要检查的位置
        safety_radius : float
            安全半径
            
        返回:
        ----
        bool
            位置是否安全
        """
        return not self.check_collision(position, safety_radius)