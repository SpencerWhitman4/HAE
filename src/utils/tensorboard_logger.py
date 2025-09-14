"""
纯Tensorboard管理器 - 仅负责Tensorboard日志记录
"""
import math
from pathlib import Path
from typing import Dict, Any, Tuple

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None


class TensorboardLogger:
    """纯Tensorboard管理器，仅负责Tensorboard日志记录"""
    
    def __init__(self, session_dir: Path, mode: str = "train"):
        """
        初始化Tensorboard记录器
        
        Args:
            session_dir: 会话目录路径 
            mode: 模式（'train'或'eval'）
        """
        if mode not in ["train", "eval"]:
            raise ValueError(f"mode必须为'train'或'eval'，得到: {mode}")
        
        self.mode = mode
        self.tensorboard_dir = session_dir / "Tensorboard" / mode
        self.tensorboard_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化SummaryWriter
        if TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(log_dir=str(self.tensorboard_dir))
        else:
            self.writer = None
            print("⚠️ Tensorboard不可用，跳过日志记录")
    
    def start(self) -> None:
        """启动Tensorboard记录"""
        # SummaryWriter在初始化时已准备就绪，无需额外操作
        pass
    
    def close(self) -> None:
        """关闭Tensorboard记录器"""
        if self.writer is not None:
            self.writer.close()
    
    def _filter_invalid_value(self, value: float) -> float:
        """过滤NaN和inf值"""
        if isinstance(value, (int, float)):
            if math.isnan(value) or math.isinf(value):
                return 0.0
        return float(value)
    
    def log_episode_data(self, episode: int, episode_data: Dict[str, Any]) -> None:
        """
        记录回合级数据
        
        Args:
            episode: 回合编号
            episode_data: 回合数据字典
        """
        if self.writer is None:
            return
        
        # 记录回合总奖励
        total_reward = episode_data.get('total_reward', 0)
        self.writer.add_scalar('Episodes/TotalReward', 
                              self._filter_invalid_value(total_reward), episode)
        
        # 记录回合长度
        length = episode_data.get('length', 0)
        self.writer.add_scalar('Episodes/Length', 
                              self._filter_invalid_value(length), episode)
        
        # 记录探索率
        exploration_rate = episode_data.get('exploration_rate', 0)
        self.writer.add_scalar('Episodes/ExplorationRate', 
                              self._filter_invalid_value(exploration_rate), episode)
        
        # 记录成功标志
        success = float(episode_data.get('success', False))
        self.writer.add_scalar('Episodes/Success', success, episode)
    
    def log_reward_components(self, step: int, reward_info: Dict[str, float]) -> None:
        """
        记录奖励组件数据
        
        Args:
            step: 当前训练步数
            reward_info: 奖励组件字典
        """
        if self.writer is None:
            return
        
        for component_name, value in reward_info.items():
            if isinstance(value, (int, float)):
                filtered_value = self._filter_invalid_value(value)
                self.writer.add_scalar(f'Reward/{component_name}', filtered_value, step)
    
    def log_position_data(self, step: int, position: Tuple[float, float, float, float]) -> None:
        """
        记录位置数据
        
        Args:
            step: 当前训练步数
            position: 位置和朝向 (x,y,z,yaw)
        """
        if self.writer is None:
            return
        
        x, y, z, yaw = position
        self.writer.add_scalar('Position/X', self._filter_invalid_value(x), step)
        self.writer.add_scalar('Position/Y', self._filter_invalid_value(y), step)
        self.writer.add_scalar('Position/Z', self._filter_invalid_value(z), step)
        self.writer.add_scalar('Position/Yaw', self._filter_invalid_value(yaw), step)
    
    def log_action_data(self, step: int, action: Tuple[float, float, float, float]) -> None:
        """
        记录动作数据
        
        Args:
            step: 当前训练步数
            action: 动作四元组 (vx,vy,vz,yaw_rate)
        """
        if self.writer is None:
            return
        
        vx, vy, vz, yaw_rate = action
        self.writer.add_scalar('Action/VX', self._filter_invalid_value(vx), step)
        self.writer.add_scalar('Action/VY', self._filter_invalid_value(vy), step)
        self.writer.add_scalar('Action/VZ', self._filter_invalid_value(vz), step)
    def log_safety_data(self, step: int, safety_info: Dict[str, Any]) -> None:
        """
        记录安全相关数据
        
        Args:
            step: 当前训练步数
            safety_info: 安全信息字典
        """
        if self.writer is None:
            return
        
        # 记录最小障碍物距离
        min_distance = safety_info.get('min_distance', 10.0)
        self.writer.add_scalar('Safety/MinObstacleDistance', self._filter_invalid_value(min_distance), step)
        
        # 记录安全状态
        is_safe = safety_info.get('is_safe', True)
        self.writer.add_scalar('Safety/IsSafe', 1.0 if is_safe else 0.0, step)
        
        # 记录碰撞状态
        collision = safety_info.get('collision', False)
        self.writer.add_scalar('Safety/Collision', 1.0 if collision else 0.0, step)
    
    def log_training_metrics(self, step: int, metrics: Dict[str, float]) -> None:
        """
        记录训练指标数据
        
        Args:
            step: 当前训练步数
            metrics: 训练指标字典
        """
        if self.writer is None:
            return
        
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                filtered_value = self._filter_invalid_value(value)
                self.writer.add_scalar(f'Training/{metric_name}', filtered_value, step)
    
    def log_perception_data(self, step: int, perception_info: Dict[str, Any]) -> None:
        """
        记录感知相关数据
        
        Args:
            step: 当前训练步数
            perception_info: 感知信息字典
        """
        if self.writer is None:
            return
        
        # 记录点云信息
        if 'point_cloud_size' in perception_info:
            self.writer.add_scalar('Perception/PointCloudSize', 
                                 int(perception_info['point_cloud_size']), step)
        
        # 记录局部地图信息
        if 'local_map_coverage' in perception_info:
            coverage = self._filter_invalid_value(perception_info['local_map_coverage'])
            self.writer.add_scalar('Perception/LocalMapCoverage', coverage, step)
        
        # 记录信息增益
        if 'information_gain' in perception_info:
            info_gain = self._filter_invalid_value(perception_info['information_gain'])
            self.writer.add_scalar('Perception/InformationGain', info_gain, step)
