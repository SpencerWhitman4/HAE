import os
import sys
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, List
from dataclasses import dataclass
import json
import time
from collections import deque
from gymnasium import spaces

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入环境基类
from src.envs.BaseRLAviary import BaseRLAviary
from src.utils.enums import DroneModel, Physics, ActionType, ObservationType

# 导入重构后的核心模块
from src.modules import StateManager, StructuredState
from src.perceptions import (
    create_perception_manager, 
    create_observation_input
)
from src.utils.MapManager import MapManager

# 设置日志记录器
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class HAUAVConfig:
    """HAUAV环境配置"""
    MAX_EPISODE_STEPS: int = 1000
    EXPLORATION_THRESHOLD: float = 0.95
    COLLISION_THRESHOLD: float = 0.2
    SAFETY_THRESHOLD: float = 0.5
    
    OBSERVATION_DIM: int = 86
    ACTION_DIM: int = 4
    
    REWARD_WEIGHTS: Dict[str, float] = None
    
    MAP_FILE: str = "/home/lxy/Code/gym-pybullet-drones-RL/src/maps/room_map.json"
    MAP_NAME: str = "room_basic"  # 使用室内房间基础地图
    
    ENABLE_TRAJECTORY_RECORDING: bool = True
    TRAJECTORY_LOG_DIR: str = "./logs/trajectories"
    
    STATE_MANAGER_CONFIG: Dict[str, Any] = None
    PERCEPTION_CONFIG: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.REWARD_WEIGHTS is None:
            self.REWARD_WEIGHTS = {
                'exploration': 0.4,
                'safety': 0.3,
                'execution': 0.2,
                'completion': 0.1
            }
        
        if self.STATE_MANAGER_CONFIG is None:
            self.STATE_MANAGER_CONFIG = {
                'history_length': 20,
                'high_level_update_frequency': 5,
                'future_horizon': 5
            }
        
        if self.PERCEPTION_CONFIG is None:
            self.PERCEPTION_CONFIG = {
                'map_resolution': 0.1,
                'map_grid_size': (100, 100),
                'num_sectors': 36,
                'max_detection_range': 10.0
            }


class HAUAVAviary(BaseRLAviary):
    """分层强化学习无人机环境"""
    
    def __init__(
        self,
        drone_model: DroneModel = DroneModel.CF2X,
        num_drones: int = 1,
        neighbourhood_radius: float = np.inf,
        initial_xyzs=None,
        initial_rpys=None,
        physics: Physics = Physics.PYB,
        pyb_freq: int = 240,
        ctrl_freq: int = 30,
        gui: bool = False,
        record: bool = False,
        obstacles: bool = True,
        user_debug_gui: bool = True,
        vision_attributes: bool = True,
        config: Optional[HAUAVConfig] = None,
        map_file: Optional[str] = None,
        enable_logging: bool = False,
        trajectory_manager=None,
        # === TensorBoard集成参数 ===
        enable_tensorboard: bool = False,
        tensorboard_mode: str = "train",  # "train" or "eval"
        session_dir: Optional[Path] = None,  # 外部传入的会话目录
        **kwargs
    ):
        """初始化HAUAV环境"""
        
        # 配置管理
        self.config = config if config is not None else HAUAVConfig()
        if map_file is not None:
            self.config.MAP_FILE = map_file
        
        # 初始化日志
        self.enable_logging = enable_logging
        self.logger = self._setup_logger()
        
        # === TensorBoard初始化 ===
        self.enable_tensorboard = enable_tensorboard
        self.tensorboard_mode = tensorboard_mode
        self.tensorboard_logger = None
        self.global_step = 0  # 全局步数计数器
        
        if self.enable_tensorboard and session_dir:
            self._initialize_tensorboard(session_dir)
        elif self.enable_tensorboard and not session_dir:
            self.logger.warning("启用了TensorBoard但未提供session_dir，TensorBoard记录将被禁用")
            self.enable_tensorboard = False
        
        # 🔧 修复：在调用父类初始化前先设置正确的INIT_XYZS
        # 预设置地图位置
        self._pre_initialize_map_position()
        map_initial_xyzs = getattr(self, 'INIT_XYZS', None)
        
        # 如果用户没有指定initial_xyzs，使用地图配置的位置
        if initial_xyzs is None and map_initial_xyzs is not None:
            initial_xyzs = map_initial_xyzs
            self.logger.info(f"使用地图配置的起始位置: {initial_xyzs}")
        
        # 调用父类初始化
        super().__init__(
            drone_model=drone_model,
            num_drones=num_drones,
            neighbourhood_radius=neighbourhood_radius,
            initial_xyzs=initial_xyzs,  # 传递地图位置
            initial_rpys=initial_rpys,
            physics=physics,
            pyb_freq=pyb_freq,
            ctrl_freq=ctrl_freq,
            gui=gui,
            record=record,
            act=ActionType.VEL,
            obs=ObservationType.KIN,
            **kwargs
        )
        
        # 环境状态
        self.episode_step = 0
        self.episode_count = 0
        self.total_steps = 0
        
        # 历史数据缓存
        self.action_history = deque(maxlen=6)
        self.subgoal_history = deque(maxlen=5)
        self.reward_history = deque(maxlen=100)
        self.exploration_rate_history = deque(maxlen=100)
        self.episode_history = []
        
        # 增量控制标志
        self.use_increment_control = True
        
        # 初始化核心模块
        self._initialize_core_modules()
        
        # 轨迹记录器（由外部传递）
        self.trajectory_logger = trajectory_manager
        if self.trajectory_logger:
            self.logger.info("外部轨迹记录器已设置")
        else:
            self.logger.info("未设置轨迹记录器")
        
        # 统计信息
        self.episode_stats = {
            'exploration_rate': 0.0,
            'collision_count': 0,
            'safety_violations': 0,
            'completion_status': False,
            'total_reward': 0.0,
            'step_count': 0
        }
        
        # 缓存变量
        self.latest_perception_results = None
        self.current_structured_state = None
        self.current_observation = None
        
        self.logger.debug("HAUAV环境初始化完成")
    
    def _setup_logger(self):
        """设置日志记录器"""
        logger = logging.getLogger(f'HAUAV_{id(self)}')
        logger.setLevel(logging.INFO if self.enable_logging else logging.WARNING)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _initialize_tensorboard(self, session_dir: Path):
        """
        初始化TensorBoard记录器
        
        Args:
            session_dir: 外部传入的会话目录（由LogDirectoryBuilder创建）
        """
        from src.utils.tensorboard_logger import TensorboardLogger
        
        # 使用会话目录下的Tensorboard/train 或 Tensorboard/eval
        tensorboard_dir = session_dir / "Tensorboard" / self.tensorboard_mode
        
        # 创建TensorBoard记录器
        self.tensorboard_logger = TensorboardLogger(tensorboard_dir, self.tensorboard_mode)
        
        self.logger.debug(f"TensorBoard记录器初始化完成: {tensorboard_dir}")
    
    def _initialize_core_modules(self):
        """初始化核心模块"""
        self.state_manager = StateManager(**self.config.STATE_MANAGER_CONFIG)
        self.perception_manager = create_perception_manager(self.config.PERCEPTION_CONFIG)
        
        # MapManager初始化 - 如果没有指定地图文件，使用默认值
        map_file = self.config.MAP_FILE if self.config.MAP_FILE else "assets/default_map.json"
        self.map_manager = MapManager(map_json_path=map_file)
        
        # 初始化点云过滤器
        self._initialize_pointcloud_filter()
        
        self.logger.debug("核心模块初始化成功")
    
    def _pre_initialize_map_position(self):
        """预初始化地图位置 - 在父类初始化前设置正确的INIT_XYZS"""
        # 从配置中获取地图文件路径和地图名称
        map_file = self.config.MAP_FILE
        map_name = self.config.MAP_NAME
        
        if map_file:
            try:
                # 直接解析地图文件获取起始位置
                import json
                with open(map_file, 'r', encoding='utf-8') as f:
                    map_data = json.load(f)
                
                # 查找指定地图的起始位置
                if map_name in map_data:
                    map_info = map_data[map_name]
                    start_position = map_info.get('start_pos', [0, 0, 1])  # 使用正确的键名
                    self.INIT_XYZS = np.array(start_position).reshape(1, 3)
                    self.logger.info(f"从地图文件加载起始位置: {start_position}")
                else:
                    self.logger.warning(f"地图 {map_name} 未找到，使用默认起始位置")
                    self.INIT_XYZS = np.array([[0, 0, 1]])
                
            except Exception as e:
                self.logger.warning(f"预加载地图失败: {e}, 使用默认起始位置")
                self.INIT_XYZS = np.array([[0, 0, 1]])
        else:
            self.logger.warning("未指定地图文件，使用默认起始位置")
            self.INIT_XYZS = np.array([[0, 0, 1]])

    def _initialize_pointcloud_filter(self):
        """初始化点云过滤器"""
        from src.utils.PointCloudFilter import PointCloudFilter, PointCloudFilterConfig
        
        # 创建点云过滤配置
        filter_config = PointCloudFilterConfig(
            max_distance=self.config.PERCEPTION_CONFIG.get('max_detection_range', 10.0),
            filter_ground=False,  # 临时禁用地面过滤
            noise_threshold=2.0,
            min_points=10
        )
        
        self.pointcloud_filter = PointCloudFilter(config=filter_config)
        self.logger.debug("点云过滤器初始化成功")
    
    def _load_map(self):
        """加载地图"""
        pybullet_client = self.getPyBulletClient()
        self.map_manager.set_pybullet_client(pybullet_client)
        
        # 从配置中获取地图文件路径和地图名称
        map_file = self.config.MAP_FILE
        map_name = self.config.MAP_NAME
        
        # 添加地图加载状态跟踪，避免重复打印
        if not hasattr(self, '_map_loaded') or not self._map_loaded:
            self._map_loaded = True
            map_load_verbose = True
        else:
            map_load_verbose = False
        
        if map_file:
            try:
                self.map_manager.load_map(map_name)
                start_position = np.array(self.map_manager.get_start_position())
                
                self.INIT_XYZS = start_position.reshape(1, 3)
                
                if map_load_verbose:  # 只在第一次加载时显示详细信息
                    print(f"✅ 地图加载完成: {map_name} (来自文件: {map_file})")
                    print(f"   起始位置: {start_position}")
                    
                    # 打印地图信息
                    map_info = self.map_manager.get_map_info()
                    if map_info:
                        print(f"   地图描述: {map_info.get('description', '无描述')}")
                        bounds = map_info.get('bounds', {})
                        if bounds:
                            print(f"   地图边界: X[{bounds.get('x_min', 0):.1f}, {bounds.get('x_max', 0):.1f}]"
                                  f" Y[{bounds.get('y_min', 0):.1f}, {bounds.get('y_max', 0):.1f}]"
                                  f" Z[{bounds.get('z_min', 0):.1f}, {bounds.get('z_max', 0):.1f}]")
                        
                        # 显示障碍物和墙体数量
                        obstacles = map_info.get('obstacles', [])
                        cave_walls = map_info.get('cave_walls', [])
                        print(f"   障碍物数量: {len(obstacles)}, 墙体数量: {len(cave_walls)}")
                else:
                    # 后续加载只显示简单信息
                    pass  # 静默重置
                    
            except Exception as e:
                print(f"❌ 地图加载失败: {e}")
                print(f"   地图名称: {map_name}")
                print(f"   地图文件: {map_file}")
                print("   将使用默认起始位置")
                # 使用默认起始位置
                if not hasattr(self, 'INIT_XYZS') or self.INIT_XYZS is None:
                    self.INIT_XYZS = np.array([[0, 0, 1]])
        else:
            print("⚠️  未指定地图文件，使用默认起始位置")
            if not hasattr(self, 'INIT_XYZS') or self.INIT_XYZS is None:
                self.INIT_XYZS = np.array([[0, 0, 1]])
    
    
    # ============ BaseRLAviary标准接口 ============

    def set_increment_control_mode(self, enable: bool = True, drone_id: int = 0):
        """启用/禁用增量控制模式"""
        if hasattr(self, 'ctrl') and self.ctrl:
            if hasattr(self.ctrl[drone_id], 'increment_mode'):
                self.ctrl[drone_id].increment_mode = enable
                self.logger.info(f"无人机 {drone_id} 增量控制模式: {'启用' if enable else '禁用'}")
            else:
                self.logger.warning(f"控制器不支持增量控制模式")
        else:
            self.logger.error("控制器未初始化")
    
    def apply_increment_action(self, increment: np.ndarray, drone_id: int = 0):
        """应用增量动作到指定无人机"""
        if hasattr(self, 'ctrl') and self.ctrl:
            if hasattr(self.ctrl[drone_id], 'set_increment_action'):
                self.ctrl[drone_id].set_increment_action(increment)
                self.logger.debug(f"无人机 {drone_id} 增量动作: {increment}")
            else:
                self.logger.warning(f"控制器不支持增量动作设置")
        else:
            self.logger.error("控制器未初始化")

    def get_control_relevant_state(self, drone_id: int = 0) -> Dict:
        """获取用于控制的相关状态信息"""
        if hasattr(self, 'current_structured_state') and self.current_structured_state:
            state = self.current_structured_state
            control_state = {
                'position': state.position.copy(),
                'velocity': state.velocity.copy(),
                'attitude_quaternion': state.attitude_quaternion.copy(),
                'angular_velocity': state.angular_velocity.copy(),
                'target_velocity': getattr(self.ctrl[drone_id], 'target_velocity', np.zeros(4)) if hasattr(self, 'ctrl') else np.zeros(4)
            }
            return control_state
        return {}
    
    def _is_increment_action(self, action):
        """判断是否为增量动作格式"""
        return (isinstance(action, np.ndarray) and 
                ((action.ndim == 1 and action.shape[0] == 4) or 
                 (action.ndim == 2 and action.shape[1] == 4)))

    def _extract_increment_action(self, action):
        """提取增量动作"""
        return action.reshape(1, -1) if action.ndim == 1 else action

    def _get_increment_control_info(self):
        """获取增量控制信息"""
        info = {}
        for k in range(self.NUM_DRONES):
            if hasattr(self.ctrl[k], 'get_target_action'):
                target_action = self.ctrl[k].get_target_action()
                if target_action is not None:
                    info[f'drone_{k}_target'] = target_action.tolist()
        return info
    
    def _actionSpace(self):
        """动作空间定义"""
        return spaces.Box(
            low=-2.0,
            high=2.0,
            shape=(self.config.ACTION_DIM,),
            dtype=np.float32
        )
    
    def _observationSpace(self):
        """观测空间定义"""
        return spaces.Box(
            low=-np.inf,
            high=np.inf, 
            shape=(self.config.OBSERVATION_DIM,),
            dtype=np.float32
        )
    
    def _computeObs(self):
        """计算86维观测向量 - 统一构建所有感知数据"""
        # 获取原始传感器数据
        sensor_data = self._get_raw_sensor_data()
        
        # 使用PerceptionManager构建86维观测并获取完整感知结果
        obs_86d, perception_results = self._build_observation_with_perception_manager(sensor_data)
        
        # 保存完整感知结果供其他方法使用
        self.latest_perception_results = perception_results
        
        # 更新StateManager状态
        structured_state = self.state_manager.parse_and_update(obs_86d)
        self.current_structured_state = structured_state
        
        # 记录统计信息 - 直接从PerceptionResults获取最新的探索率
        if self.latest_perception_results and self.latest_perception_results.local_map_results:
            # 直接从LocalMapResults获取探索率，确保数据同步
            current_exploration_rate = getattr(self.latest_perception_results.local_map_results, 'exploration_rate', 0.0)
            self.episode_stats['exploration_rate'] = float(current_exploration_rate)
        elif structured_state.parsed_map_state:
            # 备用方案：从StateManager获取
            self.episode_stats['exploration_rate'] = structured_state.parsed_map_state.get('exploration_rate', 0.0)
        else:
            # 默认值
            self.episode_stats['exploration_rate'] = 0.0
        
        # 保存当前观测
        self.current_observation = obs_86d.astype(np.float32)
        
        return self.current_observation.reshape(1, -1)  # BaseRLAviary期望(NUM_DRONES, OBS_DIM)
    
    
    def _computeTerminated(self):
        """计算终止条件 - 使用已构建的感知结果"""
        terminated = False
        termination_reason = "CONTINUE"
        
        # 1. 安全终止检查（使用PerceptionResults）
        safety_results = self.latest_perception_results.safety_results
        
        if safety_results.should_terminate:
            terminated = True
            termination_reason = safety_results.termination_reason
        elif safety_results.collision_detected:
            terminated = True
            termination_reason = "COLLISION_DETECTED"
            self.episode_stats['collision_count'] += 1
        elif safety_results.out_of_bounds:
            terminated = True
            termination_reason = "OUT_OF_BOUNDS"
        
        # 2. 探索完成检查
        if not terminated:
            exploration_rate = self.episode_stats.get('exploration_rate', 0.0)
            if exploration_rate >= self.config.EXPLORATION_THRESHOLD:
                terminated = True
                termination_reason = "EXPLORATION_COMPLETE"
                self.episode_stats['completion_status'] = True
        
        # 记录终止信息
        if terminated:
            self.logger.debug(f"Episode终止: {termination_reason}, 步数: {self.episode_step}, "
                           f"探索率: {self.episode_stats.get('exploration_rate', 0.0):.3f}")
        
        return np.array([terminated])
    
    def _computeTruncated(self):
        """计算截断条件"""
        truncated = self.episode_step >= self.config.MAX_EPISODE_STEPS
        return np.array([truncated])

    def _computeInfo(self):
        """计算信息字典 - 使用已构建的所有数据"""
        info = {
            'episode_step': self.episode_step,
            'episode_count': self.episode_count,
            'exploration_rate': self.episode_stats.get('exploration_rate', 0.0),
            'collision_count': self.episode_stats.get('collision_count', 0),
            'safety_violations': self.episode_stats.get('safety_violations', 0),
            'total_reward': self.episode_stats.get('total_reward', 0.0),
            'completion_status': self.episode_stats.get('completion_status', False),
        }
        
        # StateManager状态信息
        info['state_manager_ready'] = self.state_manager.is_ready_for_high_level()
        info['should_update_high_level'] = self.state_manager.should_update_high_level()
        info['state_history_length'] = len(self.state_manager.state_history)
        
        # PerceptionResults状态信息
        safety_results = self.latest_perception_results.safety_results
        info['min_obstacle_distance'] = safety_results.min_obstacle_distance
        info['is_safe'] = safety_results.is_safe
        
        # 动作历史信息
        info['action_history_length'] = len(self.action_history)
        info['reward_history_length'] = len(self.reward_history)
        
        return info
    
    # ============ 环境生命周期方法 ============
    
    def reset(self, seed=None, options=None):
        """重置环境 - 完整的重置流程"""
        # 调用父类reset
        obs, info = super().reset(seed=seed, options=options)
        
        # 重置环境状态计数器
        self.episode_step = 0
        self.episode_count += 1
        
        # 重置所有历史缓存
        self.action_history.clear()
        self.subgoal_history.clear()
        self.reward_history.clear()
        self.exploration_rate_history.clear()
        
        # 重置统计信息
        self.episode_stats = {
            'exploration_rate': 0.0,
            'collision_count': 0,
            'safety_violations': 0,
            'completion_status': False,
            'total_reward': 0.0,
            'step_count': 0
        }
        
        # 重置核心模块状态
        self.state_manager.reset()
        self.perception_manager.reset()
        
        # 清空缓存的感知数据
        self.latest_perception_results = None
        self.current_structured_state = None
        self.current_observation = None
        
        # 先加载地图，再计算观测
        self._load_map()
        
        # 重新计算初始观测（这会触发完整的感知流程）
        obs = self._computeObs()

        # 更新info信息
        info.update(self._computeInfo())
        
        # ===== 重要：启动新episode的轨迹记录 =====
        if self.trajectory_logger:
            self.trajectory_logger.start_new_episode(self.episode_count)
            self.logger.debug(f"轨迹记录已启动 - Episode {self.episode_count}")
        
        self.logger.debug(f"环境重置完成 - Episode {self.episode_count}, 观测维度: {obs.shape}")
        
        return obs[0], info  # 返回单智能体观测
    

    def step(self, action):
        """重写step方法，集成增量控制和TensorBoard记录"""

        # ============ 步骤1: 动作预处理和状态更新 ============
        self.episode_step += 1
        self.total_steps += 1
        self.global_step += 1
        
        # 1. 处理增量动作
        if self.use_increment_control and self._is_increment_action(action):
            increment_action = self._extract_increment_action(action)
            # 设置增量到控制器
            for k in range(self.NUM_DRONES):
                drone_increment = increment_action[k] if increment_action.ndim > 1 else increment_action
                if hasattr(self.ctrl[k], 'set_increment_action'):
                    self.ctrl[k].set_increment_action(drone_increment)
        
        # 处理动作格式 - 支持多种输入格式
        if isinstance(action, dict):
            # 多智能体格式：{0: action_array}
            action_array = action[0] if 0 in action else list(action.values())[0]
        else:
            # 单智能体格式
            action_array = action
        
        # 标准化动作格式
        if isinstance(action_array, (list, tuple)):
            action_array = np.array(action_array, dtype=np.float32)
        elif not isinstance(action_array, np.ndarray):
            action_array = np.array([action_array], dtype=np.float32)
        
        # 确保动作维度正确
        if action_array.ndim == 1:
            action_array = action_array.reshape(1, -1)
        
        # 记录动作历史 - 支持连续性分析
        action_4d = action_array.flatten()[:4]  # 确保4维
        if len(action_4d) < 4:
            action_4d = np.pad(action_4d, (0, 4 - len(action_4d)))
        
        # 保存历史动作供连续性奖励计算
        if hasattr(self, 'last_action') and self.last_action is not None:
            self.prev_action = self.last_action.copy()
        else:
            self.prev_action = None
        
        self.last_action = action_4d.copy()
        self.action_history.append(action_4d)
        
        # ============ 步骤2: 执行物理仿真 ============
        # 先预处理动作以获取RPM输出（在BaseRLAviary.step之前）
        rpm_output = self._preprocessAction(action_array)
        self.last_rpm = rpm_output  # 保存RPM输出供轨迹记录使用
        
        # 调用BaseRLAviary.step执行物理仿真
        # 这会自动调用 _computeObs, _computeReward, _computeTerminated, _computeTruncated, _computeInfo
        base_obs, base_reward, base_terminated, base_truncated, base_info = super().step(action_array)
        
        # ============ 步骤3: 获取统一构建的感知数据 ============
        # 在_computeObs中已经构建并保存了以下数据:
        # - self.latest_perception_results (完整感知结果)
        # - self.current_structured_state (StateManager解析状态)
        # - self.current_observation (86维观测)
        
        # 验证感知数据完整性
        
        # ============ 步骤4: 使用统一感知数据的返回值 ============
        # BaseRLAviary.step已经调用了我们的方法，直接使用返回值
        observation_86d = base_obs[0]  # 单智能体观测
        hierarchical_reward = base_reward[0]  # 分层奖励
        terminated = base_terminated[0]  # 终止状态
        truncated = base_truncated[0]  # 截断状态
        
        # ============ 步骤5: 扩展信息字典 ============
        # 基础info已经包含了_computeInfo的内容，现在添加HAUAVAviary特有信息
        info = base_info.copy()
        
        # 添加感知模块详细信息
        if self.latest_perception_results:
            safety_results = self.latest_perception_results.safety_results
            local_map_results = self.latest_perception_results.local_map_results
            
            info.update({
                'perception_safety': {
                    'min_obstacle_distance': safety_results.min_obstacle_distance,
                    'collision_detected': safety_results.collision_detected,
                    'out_of_bounds': safety_results.out_of_bounds,
                    'should_terminate': safety_results.should_terminate,
                    'is_safe': safety_results.is_safe
                },
                'perception_map': {
                    'exploration_rate': getattr(local_map_results, 'exploration_rate', 0.0),
                    'occupied_cells': getattr(local_map_results, 'occupied_cells', 0),
                    'free_cells': getattr(local_map_results, 'free_cells', 0),
                    'unknown_cells': getattr(local_map_results, 'unknown_cells', 0)
                }
            })
        
        # 添加增量控制信息
        info['incremental_control'] = self._get_increment_control_info()
        
        # 添加控制相关状态
        if hasattr(self, 'state_manager'):
            direct_state = self.state_manager.extract_control_relevant_state(observation_86d)
            info['direct_state'] = direct_state
        
        # ============ 步骤6: TensorBoard数据记录 ============
        if self.tensorboard_logger:
            self._log_step_data_to_tensorboard(action_4d, hierarchical_reward, info)
        
        # ============ 重要：轨迹记录 ============
        if self.trajectory_logger:
            trajectory_data = self.get_trajectory_step_data()
            self.trajectory_logger.log_step(trajectory_data)
        
        # ============ 步骤7: Episode结束处理 ============
        if terminated or truncated:
            if self.tensorboard_logger:
                self._log_episode_data_to_tensorboard()
            
            # ===== 重要：完成轨迹记录 =====
            if self.trajectory_logger:
                termination_reason = self._determine_termination_reason(terminated, info)
                final_exploration_rate = self.episode_stats.get('exploration_rate', 0.0)
                total_reward = self.episode_stats.get('total_reward', 0.0)
                
                self.trajectory_logger.finalize_episode(
                    termination_reason=termination_reason,
                    final_exploration_rate=final_exploration_rate,
                    total_reward=total_reward
                )
                self.logger.debug(f"轨迹记录已完成 - Episode {self.episode_count}")
            
            # 记录Episode统计
            termination_reason = self._determine_termination_reason(terminated, info)
            self.logger.info(f"Episode {self.episode_count} 结束: {termination_reason}")
        
        return observation_86d, hierarchical_reward, terminated, truncated, info

    
    
    # ============ 感知处理方法 ============
    
    def _get_raw_sensor_data(self) -> Dict[str, Any]:
        """获取原始传感器数据"""
        # 获取原始点云数据
        raw_point_cloud = self._getDronePointCloud(0)
        
        # 预处理点云数据
        point_cloud = self._preprocess_pointcloud(raw_point_cloud)
        
        # 获取无人机状态
        drone_state = self._getDroneStateVector(0)
        drone_position = self.pos[0]
        drone_velocity = self.vel[0]
        drone_orientation = self.rpy[0]
        
        return {
            'point_cloud': point_cloud,
            'drone_position': drone_position,
            'drone_velocity': drone_velocity,
            'drone_orientation': drone_orientation,
            'drone_state': drone_state,
            'timestamp': time.time()
        }
    def _preprocess_pointcloud(self, point_cloud: np.ndarray) -> np.ndarray:
        """
        预处理点云数据 - 使用独立的PointCloudFilter
        
        Args:
            point_cloud: 原始点云 [N, 3]
            
        Returns:
            np.ndarray: 预处理后的点云
        """
        # return self.pointcloud_filter.filter(point_cloud)
        return point_cloud

    def _build_observation_with_perception_manager(self, sensor_data: Dict) -> Tuple[np.ndarray, Any]:
        """使用PerceptionManager构建86维观测并返回完整感知结果"""
        # 创建ObservationInput
        obs_input = create_observation_input(
            point_cloud=sensor_data['point_cloud'],
            drone_position=sensor_data['drone_position'],
            drone_orientation=sensor_data['drone_orientation'],
            drone_velocity=sensor_data['drone_velocity'],
            action_history=list(self.action_history),
            timestamp=sensor_data['timestamp']
        )
        
        # 获取当前子目标序列
        if self.state_manager.is_ready_for_high_level():
            current_subgoals = self.state_manager.current_subgoal_sequence
        else:
            current_subgoals = np.zeros((5, 2), dtype=np.float32)
        
        # 获取动作历史
        if len(self.action_history) > 0:
            action_history = np.array(list(self.action_history), dtype=np.float32)
            if action_history.shape[0] < 6:
                # 填充到6步
                padding = np.zeros((6 - action_history.shape[0], 4), dtype=np.float32)
                action_history = np.vstack([padding, action_history])
        else:
            action_history = np.zeros((6, 4), dtype=np.float32)
        
        # 重要：先处理观测获取完整的PerceptionResults
        perception_results = self.perception_manager.process_observation(
            obs_input=obs_input,
            map_manager=self.map_manager
        )
        
        # 使用PerceptionManager构建86维观测
        obs_86d = self.perception_manager.build_86d_observation(
            obs_input=obs_input,
            current_subgoals=current_subgoals,
            action_history=action_history,
            map_manager=self.map_manager
        )
        
        return obs_86d, perception_results
    
    # ============ 奖励计算方法 ============
    
    def _compute_exploration_reward(self, structured_state: StructuredState) -> float:
        """计算探索奖励"""
        if not structured_state.parsed_map_state:
            return 0.0
            
        exploration_rate = structured_state.parsed_map_state.get('exploration_rate', 0.0)
        information_gain = structured_state.parsed_map_state.get('information_gain_potential', 0.0)
        
        # 基础探索奖励
        base_reward = exploration_rate * 0.1
        
        # 信息增益奖励
        gain_reward = information_gain * 0.05
        
        # 探索完成大奖励
        completion_bonus = 1.0 if exploration_rate >= self.config.EXPLORATION_THRESHOLD else 0.0
        
        return base_reward + gain_reward + completion_bonus
    
    

    def _compute_completion_reward(self, structured_state: StructuredState) -> float:
        """计算完成奖励"""
        if not structured_state.parsed_map_state:
            return 0.0
            
        exploration_rate = structured_state.parsed_map_state.get('exploration_rate', 0.0)
        
        if exploration_rate >= self.config.EXPLORATION_THRESHOLD:
            return 2.0  # 完成大奖励
        else:
            # 渐进奖励
            return exploration_rate * 0.1
        
        
    def _compute_action_consistency(self) -> float:
        """计算动作连续性 - 用于执行奖励"""
        if self.prev_action is None or self.last_action is None:
            return 0.0
        
        # 计算动作差异
        action_diff = np.linalg.norm(self.last_action - self.prev_action)
        
        # 转换为一致性分数 (0-1)，差异越小一致性越高
        consistency = np.exp(-action_diff)  # 指数衰减
        
        return float(consistency)

    def _get_last_reward_breakdown(self) -> Dict[str, float]:
        """获取最后一次奖励计算的详细分解"""
        if not hasattr(self, 'last_reward_breakdown'):
            return {}
        
        return getattr(self, 'last_reward_breakdown', {})

    def _is_success(self) -> bool:
        """判断成功条件 - 增强版"""
        # 基于探索率的成功判断
        exploration_rate = self.episode_stats.get('exploration_rate', 0.0)
        exploration_success = exploration_rate >= self.config.EXPLORATION_THRESHOLD
        
        # 基于安全性的成功判断
        safety_success = True
        if self.latest_perception_results and self.latest_perception_results.safety_results:
            safety_results = self.latest_perception_results.safety_results
            safety_success = (not safety_results.collision_detected and 
                            not safety_results.out_of_bounds and
                            safety_results.is_safe)
        
        return exploration_success and safety_success

    # ============ 增强奖励计算方法 ============

    def _compute_safety_reward(self, safety_results) -> float:
        """基于PerceptionResults.safety_results计算安全奖励"""
        # 基于PerceptionResults的安全状态
        if safety_results.collision_detected:
            self.episode_stats['collision_count'] += 1
            return -2.0  # 严重碰撞惩罚
        
        if safety_results.should_terminate:
            return -1.0  # 终止状态惩罚
        
        # 基于最小障碍物距离的奖励
        min_distance = safety_results.min_obstacle_distance
        
        if min_distance < self.config.COLLISION_THRESHOLD:
            # 危险接近
            self.episode_stats['safety_violations'] += 1
            return -0.5
        elif min_distance < self.config.SAFETY_THRESHOLD:
            # 安全距离不足
            return -0.1
        elif min_distance > self.config.SAFETY_THRESHOLD * 2:
            # 良好的安全距离
            return 0.2
        else:
            # 基本安全状态
            return 0.1

    def _computeReward(self):
        """计算分层奖励 - 增强版，记录奖励分解"""
        structured_state = self.current_structured_state
        perception_results = self.latest_perception_results
        
        # 1. 探索奖励
        exploration_reward = self._compute_exploration_reward(structured_state)
        
        # 2. 安全奖励（使用PerceptionResults）
        safety_reward = self._compute_safety_reward(perception_results.safety_results)
        
        # 3. 执行奖励 (包含动作连续性)
        execution_reward = self._compute_execution_reward(structured_state)
        
        # 4. 完成奖励
        completion_reward = self._compute_completion_reward(structured_state)
        
        # 加权总奖励
        total_reward = (
            self.config.REWARD_WEIGHTS['exploration'] * exploration_reward +
            self.config.REWARD_WEIGHTS['safety'] * safety_reward +
            self.config.REWARD_WEIGHTS['execution'] * execution_reward +
            self.config.REWARD_WEIGHTS['completion'] * completion_reward
        )
        
        # 保存奖励分解用于轨迹记录
        self.last_reward_breakdown = {
            'exploration': exploration_reward,
            'safety': safety_reward,
            'execution': execution_reward,
            'completion': completion_reward,
            'total': total_reward
        }
        
        # 更新统计信息
        self.episode_stats['total_reward'] += total_reward
        self.reward_history.append(total_reward)
        
        # 详细奖励记录
        if self.episode_step % 50 == 0:
            self.logger.debug(f"步骤 {self.episode_step} 奖励分解: "
                            f"探索={exploration_reward:.3f}, 安全={safety_reward:.3f}, "
                            f"执行={execution_reward:.3f}, 完成={completion_reward:.3f}, "
                            f"总计={total_reward:.3f}")
        
        return np.array([total_reward])

    def _compute_execution_reward(self, structured_state: StructuredState) -> float:
        """计算执行奖励"""
        base_reward = 0.0
        
        # 基础执行奖励
        if structured_state.motion_context:
            motion_consistency = structured_state.motion_context.get('motion_consistency', 0.0)
            speed_stability = structured_state.motion_context.get('speed_stability', 0.0)
            base_reward = 0.05 * (motion_consistency + speed_stability)
        
        # 动作连续性奖励
        action_consistency = self._compute_action_consistency()
        consistency_reward = 0.02 * action_consistency
        
        # 速度适应性奖励
        velocity_reward = self._compute_velocity_reward()
        
        return base_reward + consistency_reward + velocity_reward
    
    def _compute_velocity_reward(self) -> float:
        """计算速度奖励"""
        current_velocity = self.vel[0]
        speed = np.linalg.norm(current_velocity)
        
        # 适当速度区间奖励
        if 1 <= speed <= 4:
            return 0.1
        elif speed < 0.2:
            return 0.02
        else:
            return -0.05  # 过快惩罚
    
    def get_trajectory_step_data(self, drone_idx: int = 0) -> Dict[str, Any]:
        """获取轨迹记录所需的5元组数据"""
        # 1. 当前位置 (current_position)
        current_position = self.pos[drone_idx].tolist()  # [x, y, z]
        
        # 2. 当前速度 (current_velocity)  
        current_velocity = self.vel[drone_idx].tolist()  # [vx, vy, vz]
        
        # 3. 目标速度 (target_velocity) - 从控制器获取
        if hasattr(self, 'ctrl') and self.ctrl and hasattr(self.ctrl[drone_idx], 'target_velocity'):
            target_velocity = self.ctrl[drone_idx].target_velocity.tolist()  # [vx, vy, vz, yaw_rate]
        else:
            target_velocity = [0.0, 0.0, 0.0, 0.0]  # 默认值
            
        # 4. 模型输出动作 (model_action)
        model_action = getattr(self, 'last_action', np.zeros(4)).tolist()  # [vx, vy, vz, yaw_rate]
        
        # 5. 电机输出 (rpm_action) - 从最近的计算中获取
        if hasattr(self, 'last_rpm') and self.last_rpm is not None:
            rpm_action = self.last_rpm[drone_idx].tolist()  # [rpm1, rpm2, rpm3, rpm4]
        else:
            rpm_action = [0.0, 0.0, 0.0, 0.0]  # 默认值
            
        # 构建精简的轨迹数据 - 仅保存5元组和基本信息
        trajectory_data = {
            # 基本信息
            'step': self.episode_step,
            'exploration_rate': self.episode_stats.get('exploration_rate', 0.0),
            
            # 核心5元组数据
            'current_position': current_position,
            'current_velocity': current_velocity, 
            'target_velocity': target_velocity,
            'model_action': model_action,
            'rpm_action': rpm_action
        }
        
        return trajectory_data
    
    def getPyBulletClient(self):
        """获取PyBullet客户端 - 兼容BaseRLAviary"""
        return getattr(self, '_p', None) or getattr(self, 'CLIENT', None)
    
    def _getDronePointCloud(self, drone_idx: int) -> np.ndarray:
        """获取无人机点云数据 - 兼容方法"""
        # 使用BaseRLAviary的激光雷达方法
        return self._getDroneRays(drone_idx)
    
    # ============ 诊断和调试方法 ============
    
    def get_system_diagnostics(self) -> Dict[str, Any]:
        """获取系统诊断信息"""
        return {
            'episode_step': self.episode_step,
            'episode_count': self.episode_count,
            'episode_stats': self.episode_stats.copy(),
            'action_history_length': len(self.action_history),
            'reward_history_length': len(self.reward_history),
            'state_manager_status': {
                'history_length': len(self.state_manager.state_history),
                'is_ready_for_high_level': self.state_manager.is_ready_for_high_level(),
                'should_update_high_level': self.state_manager.should_update_high_level(),
                'step_counter': self.state_manager.step_counter
            }
        }
    
    def validate_observation_structure(self, obs: np.ndarray) -> Dict[str, bool]:
        """验证观测结构"""
        return {
            'correct_dimension': obs.shape[-1] == self.config.OBSERVATION_DIM,
            'no_nan_values': not np.any(np.isnan(obs)),
            'no_inf_values': not np.any(np.isinf(obs)),
            'finite_values': np.all(np.isfinite(obs))
        }

    # ============ TensorBoard记录方法 ============
    
    def _log_step_data_to_tensorboard(self, action: np.ndarray, reward: float, info: Dict):
        """记录步骤级数据到TensorBoard"""
        
        # 1. 奖励组件记录
        reward_breakdown = getattr(self, 'last_reward_breakdown', {})
        self.tensorboard_logger.log_reward_components(self.global_step, reward_breakdown)
        
        # 2. 位置和动作数据
        position_data = (self.pos[0][0], self.pos[0][1], self.pos[0][2], self.rpy[0][2])  # x,y,z,yaw
        action_data = (action[0], action[1], action[2], action[3])  # vx,vy,vz,yaw_rate
        
        self.tensorboard_logger.log_position_data(self.global_step, position_data)
        self.tensorboard_logger.log_action_data(self.global_step, action_data)
        
        # 3. 安全数据
        safety_info = info.get('perception_safety', {})
        safety_data = {
            'min_distance': safety_info.get('min_obstacle_distance', 10.0),
            'is_safe': safety_info.get('is_safe', True),
            'collision': safety_info.get('collision_detected', False)
        }
        self.tensorboard_logger.log_safety_data(self.global_step, safety_data)
        
        # 4. 感知数据
        perception_info = {
            'local_map_coverage': info.get('perception_map', {}).get('exploration_rate', 0.0),
            'information_gain': reward_breakdown.get('exploration', 0.0),
            'point_cloud_size': getattr(self.latest_perception_results, 'point_cloud_size', 0) if self.latest_perception_results else 0
        }
        self.tensorboard_logger.log_perception_data(self.global_step, perception_info)
        
        # 5. 状态管理数据
        state_metrics = {
            'history_length': len(self.state_manager.state_history) if hasattr(self, 'state_manager') else 0,
            'high_level_ready': float(self.state_manager.is_ready_for_high_level()) if hasattr(self, 'state_manager') else 0.0,
            'should_update_high_level': float(self.state_manager.should_update_high_level()) if hasattr(self, 'state_manager') else 0.0,
            'step_counter': getattr(self.state_manager, 'step_counter', 0) if hasattr(self, 'state_manager') else 0
        }
        self.tensorboard_logger.log_training_metrics(self.global_step, state_metrics)
        
        # 6. 动作连续性和速度数据
        execution_metrics = {
            'action_consistency': self._compute_action_consistency(),
            'velocity_magnitude': np.linalg.norm(self.vel[0]),
            'angular_velocity_magnitude': np.linalg.norm(self.rpy[0])
        }
        self.tensorboard_logger.log_training_metrics(self.global_step, execution_metrics)

    def _log_episode_data_to_tensorboard(self):
        """记录回合级数据到TensorBoard"""
        episode_data = {
            'total_reward': self.episode_stats.get('total_reward', 0.0),
            'length': self.episode_step,
            'exploration_rate': self.episode_stats.get('exploration_rate', 0.0),
            'success': self._is_success(),
            'collision_count': self.episode_stats.get('collision_count', 0),
            'safety_violations': self.episode_stats.get('safety_violations', 0),
            'completion_status': self.episode_stats.get('completion_status', False)
        }
        
        self.tensorboard_logger.log_episode_data(self.episode_count, episode_data)

    def _determine_termination_reason(self, terminated: bool, info: Dict) -> str:
        """确定终止原因"""
        if not terminated:
            return "max_steps"
        
        safety_info = info.get('perception_safety', {})
        if safety_info.get('collision_detected', False):
            return "collision"
        elif safety_info.get('out_of_bounds', False):
            return "out_of_bounds"
        elif info.get('perception_map', {}).get('exploration_rate', 0.0) >= self.config.EXPLORATION_THRESHOLD:
            return "exploration_complete"
        else:
            return "safety_violation"

    def _compute_action_consistency(self) -> float:
        """计算动作连续性指标"""
        if hasattr(self, 'prev_action') and self.prev_action is not None and hasattr(self, 'last_action'):
            diff = np.linalg.norm(self.last_action - self.prev_action)
            return 1.0 / (1.0 + diff)
        return 1.0

    def start_tensorboard_logging(self):
        """启动TensorBoard记录"""
        if self.tensorboard_logger:
            self.tensorboard_logger.start()
            self.logger.info("TensorBoard记录已启动")

    def close_tensorboard_logging(self):
        """关闭TensorBoard记录"""
        if self.tensorboard_logger:
            self.tensorboard_logger.close()
            self.logger.info("TensorBoard记录已关闭")

    def close(self):
        """关闭环境，包括TensorBoard"""
        if self.tensorboard_logger:
            self.close_tensorboard_logging()
        super().close()
