import os
import sys
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass
from collections import deque

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入HAUAVAviary作为基类
from src.envs.HAUAV_Aviary import HAUAVAviary, HAUAVConfig
from src.utils.enums import DroneModel, Physics, ActionType, ObservationType

logger = logging.getLogger(__name__)


@dataclass
class BaseFlightConfig(HAUAVConfig):
    """基础飞行训练配置"""
    
    # 基础飞行任务配置
    task_type: str = "basic_flight"         # 任务类型：基础飞行
    hover_threshold: float = 0.8            # 悬停阈值（米）
    flight_speed_target: float = 1.0        # 目标飞行速度
    
    # 奖励权重配置
    stability_reward_weight: float = 1.0    # 稳定性奖励权重
    task_reward_weight: float = 2.0         # 任务奖励权重
    height_reward_weight: float = 0.5       # 高度保持奖励权重
    
    # 任务切换配置
    curriculum_learning: bool = True        # 是否启用课程学习
    hover_success_threshold: int = 20       # 悬停成功阈值（连续步数）
    flight_unlock_episodes: int = 10        # 解锁飞行任务的回合数
    
    # 飞行任务配置
    target_change_interval: int = 200       # 目标更换间隔（步数）
    max_target_distance: float = 5.0       # 最大目标距离
    min_target_distance: float = 2.0       # 最小目标距离


class BaseFlightAviary(HAUAVAviary):
    """
    基础飞行训练环境 - 复用HAUAVAviary的86维观测系统
    
    设计目标：
    1. 复用父类的完整观测和感知系统
    2. 专门针对悬停和稳定飞行任务提供奖励函数
    3. 支持课程学习：悬停 → 飞行 → 混合训练
    """
    
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
        config: Optional[BaseFlightConfig] = None,
        map_file: Optional[str] = None,
        **kwargs
    ):
        """初始化基础飞行环境"""
        
        # 使用默认配置或传入的配置
        if config is None:
            config = BaseFlightConfig()
        
        # 调用父类初始化 - 这会自动处理86维观测系统
        super().__init__(
            drone_model=drone_model,
            num_drones=num_drones,
            neighbourhood_radius=neighbourhood_radius,
            initial_xyzs=initial_xyzs,
            initial_rpys=initial_rpys,
            physics=physics,
            pyb_freq=pyb_freq,
            ctrl_freq=ctrl_freq,
            gui=gui,
            record=record,
            obstacles=obstacles,
            user_debug_gui=user_debug_gui,
            vision_attributes=vision_attributes,
            config=config,
            map_file=map_file,
            **kwargs
        )
        
        # 基础飞行特定状态
        self.flight_config = config
        self.current_task = "hover"          # 当前任务：hover 或 flight
        self.training_stage = "hover"        # 训练阶段
        self.target_position = None          # 目标位置（用于奖励计算）
        self.hover_success_count = 0         # 连续悬停成功计数
        self.last_target_change = 0          # 上次目标变换时间
        
        # 基础飞行历史数据
        self.success_history = deque(maxlen=100)
        self.task_reward_history = deque(maxlen=50)
        
        logger.info(f"BaseFlightAviary初始化完成，训练阶段: {self.training_stage}")
    
    def reset(self, seed=None, options=None):
        """重置环境 - 复用父类reset并添加基础飞行任务生成"""
        # 调用父类reset（处理86维观测系统）
        obs, info = super().reset(seed=seed, options=options)
        
        # 重置基础飞行状态
        self.hover_success_count = 0
        self.last_target_change = 0
        self.success_history.clear()
        self.task_reward_history.clear()
        
        # 生成初始任务
        self._generate_task()
        
        # 添加任务信息到info
        info.update({
            'base_flight_task': self.current_task,
            'training_stage': self.training_stage,
            'target_position': self.target_position.tolist() if self.target_position is not None else None
        })
        
        return obs, info
    
    def step(self, action):
        """重写step方法，添加悬停稳定性信息"""
        # 调用父类step方法
        obs, reward, terminated, truncated, info = super().step(action)
        
        # 添加BaseFlightAviary特有的信息
        info.update({
            'base_flight_task': self.current_task,
            'training_stage': self.training_stage,
            'target_position': self.target_position.tolist() if self.target_position is not None else None,
            'hover_success_count': self.hover_success_count,
            'stable_time': self.hover_success_count,  # 提供稳定悬停时间（以步数为单位）
            'hover_success': self.hover_success_count >= self.flight_config.hover_success_threshold,
            'flight_success': False  # 暂时简化，后续可以根据需要改进
        })
        
        # 如果是flight任务，检查flight成功条件
        if self.current_task == "flight" and self.target_position is not None:
            distance_to_target = np.linalg.norm(self.pos[0] - self.target_position)
            info['flight_success'] = distance_to_target < 2.0  # 2米内认为成功
            info['distance_to_target'] = distance_to_target
        
        return obs, reward, terminated, truncated, info
    
    def _computeReward(self):
        """重写奖励函数 - 专注于悬停和稳定飞行，包含失败惩罚"""
        # 调用父类获取基础奖励（探索、安全等）
        base_reward_array = super()._computeReward()
        base_reward = base_reward_array[0] if hasattr(base_reward_array, '__len__') else base_reward_array
        
        # 计算基础飞行任务奖励
        task_reward = self._compute_task_reward()
        
        # 计算稳定性奖励
        stability_reward = self._compute_stability_reward()
        
        # 计算高度保持奖励
        height_reward = self._compute_height_reward()
        
        # 全局安全惩罚检查
        safety_penalty = self._compute_safety_penalty()
        
        # 组合奖励
        total_reward = (
            0.3 * base_reward +  # 保留一些基础奖励（安全、探索）
            self.flight_config.task_reward_weight * task_reward +
            self.flight_config.stability_reward_weight * stability_reward +
            self.flight_config.height_reward_weight * height_reward +
            safety_penalty  # 直接加入安全惩罚（通常为负值）
        )
        
        # 记录奖励历史
        reward_breakdown = {
            'base_reward': float(base_reward),
            'task_reward': float(task_reward),
            'stability_reward': float(stability_reward),
            'height_reward': float(height_reward),
            'safety_penalty': float(safety_penalty),
            'total_reward': float(total_reward)
        }
        self.task_reward_history.append(reward_breakdown)
        
        # 检查任务完成和阶段切换
        self._check_task_completion()
        self._maybe_update_training_stage()
        
        return np.array([total_reward])
    
    def _compute_task_reward(self) -> float:
        """计算任务特定奖励"""
        if self.target_position is None:
            return 0.0
        
        current_pos = self.pos[0]  # 使用父类的位置信息
        current_vel = self.vel[0]
        
        if self.current_task == "hover":
            return self._compute_hover_reward(current_pos, current_vel)
        elif self.current_task == "flight":
            return self._compute_flight_reward(current_pos, current_vel)
        else:
            return 0.0
    
    def _compute_hover_reward(self, position: np.ndarray, velocity: np.ndarray) -> float:
        """计算悬停奖励 - 包含失败惩罚（精细化阈值）"""
        # 距离目标的位置误差
        position_error = np.linalg.norm(position - self.target_position)
        position_reward = max(0, 2.0 - position_error)
        
        # 速度应该接近0
        velocity_magnitude = np.linalg.norm(velocity)
        velocity_reward = max(0, 1.0 - velocity_magnitude)
        
        # 完成检查
        is_hovering = (position_error < self.flight_config.hover_threshold and 
                      velocity_magnitude < 0.5)
        
        # 悬停失败的精细惩罚机制（适合几米范围的任务）
        hover_failure_penalty = 0.0
        
        # 位置偏离惩罚（精细化）
        if position_error > 2.0:  # 距离目标超过2米（中等严重）
            hover_failure_penalty += -3.0  # 严重位置偏离惩罚
        elif position_error > 1.5:  # 距离目标超过1.5米
            hover_failure_penalty += -1.5  # 中等位置偏离惩罚
        elif position_error > 1.0:  # 距离目标超过1米
            hover_failure_penalty += -0.5  # 轻微位置偏离惩罚
        
        # 速度失控惩罚（精细化）
        if velocity_magnitude > 1.5:  # 速度过大（失控）
            hover_failure_penalty += -2.0  # 速度失控惩罚
        elif velocity_magnitude > 1.0:  # 速度较大
            hover_failure_penalty += -0.8  # 速度过大惩罚
        elif velocity_magnitude > 0.7:  # 速度偏大
            hover_failure_penalty += -0.3  # 轻微速度惩罚
        
        # 高度失败惩罚（精细化）
        current_height = position[2]
        if current_height < 0.5:  # 太低（可能撞地）
            hover_failure_penalty += -5.0  # 严重高度惩罚
        elif current_height > 3.5:  # 太高
            hover_failure_penalty += -1.0  # 高度过高惩罚
        
        # 连续失败惩罚：如果连续多步无法悬停，增加惩罚
        if not is_hovering:
            consecutive_failures = max(0, 30 - self.hover_success_count)  # 降低到30步
            if consecutive_failures > 20:  # 连续20步以上失败
                hover_failure_penalty += -1.0  # 连续失败惩罚（减少）
        
        completion_bonus = 1.0 if is_hovering else 0.0
        
        # 连续成功奖励：鼓励持续稳定悬停
        continuous_success_bonus = 0.0
        if is_hovering:
            self.hover_success_count += 1
            # 连续成功越多，额外奖励越大（但有上限）
            continuous_success_bonus = min(self.hover_success_count * 0.1, 2.0)
        else:
            self.hover_success_count = 0
        
        total_reward = (position_reward + velocity_reward + 
                       completion_bonus + continuous_success_bonus + 
                       hover_failure_penalty)
        
        return total_reward
    
    def _compute_flight_reward(self, position: np.ndarray, velocity: np.ndarray) -> float:
        """计算飞行奖励"""
        # 接近目标
        position_error = np.linalg.norm(position - self.target_position)
        position_reward = max(0, 3.0 - position_error * 0.5)
        
        # 合理速度
        velocity_magnitude = np.linalg.norm(velocity)
        target_speed = self.flight_config.flight_speed_target
        speed_error = abs(velocity_magnitude - target_speed)
        velocity_reward = max(0, 1.0 - speed_error)
        
        # 方向正确性
        if position_error > 0.1:
            to_target = (self.target_position - position) / position_error
            velocity_direction = velocity / (velocity_magnitude + 1e-6)
            direction_alignment = np.dot(to_target, velocity_direction)
            direction_reward = max(0, direction_alignment) * 0.5
        else:
            direction_reward = 0.5
        
        return position_reward + velocity_reward + direction_reward
    
    def _compute_stability_reward(self) -> float:
        """计算稳定性奖励（基于姿态和角速度）"""
        rpy = self.rpy[0]  # 使用父类的姿态信息
        ang_vel = self.ang_v[0]
        
        # 姿态稳定性（roll和pitch应接近0）
        roll, pitch, yaw = rpy
        attitude_error = abs(roll) + abs(pitch)
        attitude_reward = max(0, 1.0 - attitude_error * 2.0)
        
        # 角速度稳定性
        ang_vel_magnitude = np.linalg.norm(ang_vel)
        ang_vel_reward = max(0, 1.0 - ang_vel_magnitude * 0.5)
        
        return attitude_reward + ang_vel_reward
    
    def _compute_height_reward(self) -> float:
        """计算高度保持奖励"""
        current_height = self.pos[0][2]
        
        # 期望高度范围
        target_height = self.target_position[2] if self.target_position is not None else 2.0
        height_error = abs(current_height - target_height)
        
        return max(0, 1.0 - height_error)
    
    def _compute_safety_penalty(self) -> float:
        """计算全局安全惩罚（精细化阈值）"""
        penalty = 0.0
        
        current_pos = self.pos[0]
        current_vel = self.vel[0] 
        current_rpy = self.rpy[0]
        current_ang_vel = self.ang_v[0]
        
        # 1. 精细姿态惩罚（适合悬停任务）
        roll, pitch, yaw = current_rpy
        if abs(roll) > np.radians(30) or abs(pitch) > np.radians(30):
            penalty += -5.0   # 严重姿态失控惩罚（从45度降到30度）
        elif abs(roll) > np.radians(20) or abs(pitch) > np.radians(20):
            penalty += -2.0   # 姿态过度倾斜惩罚（从30度降到20度）
        elif abs(roll) > np.radians(15) or abs(pitch) > np.radians(15):
            penalty += -0.5   # 轻微姿态惩罚
        
        # 2. 精细角速度惩罚
        ang_vel_magnitude = np.linalg.norm(current_ang_vel)
        if ang_vel_magnitude > 3.0:  # 角速度过大（从5.0降到3.0）
            penalty += -3.0   # 旋转失控惩罚
        elif ang_vel_magnitude > 2.0:
            penalty += -1.0   # 旋转过快惩罚
        elif ang_vel_magnitude > 1.5:
            penalty += -0.3   # 轻微旋转惩罚
        
        # 3. 精细线速度惩罚
        vel_magnitude = np.linalg.norm(current_vel)
        if vel_magnitude > 3.0:  # 速度严重过大（从8.0降到3.0）
            penalty += -4.0   # 速度失控惩罚
        elif vel_magnitude > 2.0:  # 从5.0降到2.0
            penalty += -1.5   # 速度过大惩罚
        elif vel_magnitude > 1.5:
            penalty += -0.5   # 轻微速度惩罚
        
        # 4. 边界惩罚（适合小范围悬停）
        x, y, z = current_pos
        
        # 缩小安全区域到合理范围（假设悬停区域在±10米内）
        if abs(x) > 8 or abs(y) > 8:
            penalty += -2.0   # 接近边界惩罚
        if abs(x) > 10 or abs(y) > 10:
            penalty += -5.0   # 超出边界惩罚（不再是-20）
        
        if z < 0.3:  # 太低（接近地面）
            penalty += -8.0   # 撞地风险惩罚（从-20降到-8）
        elif z < 0.5:
            penalty += -2.0   # 高度过低惩罚
            
        if z > 4.0:  # 太高（从5.0降到4.0）
            penalty += -1.5   # 高度过高惩罚
        elif z > 3.5:
            penalty += -0.5   # 轻微高度惩罚
        
        # 5. 任务相关的精细偏离惩罚
        if self.current_task == "hover" and self.target_position is not None:
            distance_to_target = np.linalg.norm(current_pos - self.target_position)
            if distance_to_target > 3.0:  # 严重偏离悬停目标（从5.0降到3.0）
                penalty += -3.0   # 降低惩罚强度
            elif distance_to_target > 2.5:
                penalty += -1.0   # 中等偏离惩罚
        
        return penalty
    
    def _generate_task(self):
        """生成任务目标"""
        if self.training_stage == "hover":
            self._generate_hover_task()
        elif self.training_stage == "flight":
            self._generate_flight_task()
        else:  # mixed
            self.current_task = np.random.choice(["hover", "flight"], p=[0.4, 0.6])
            if self.current_task == "hover":
                self._generate_hover_task()
            else:
                self._generate_flight_task()
    
    def _generate_hover_task(self):
        """生成悬停任务"""
        self.current_task = "hover"
        
        # 在当前位置附近生成悬停目标
        current_pos = self.pos[0] if hasattr(self, 'pos') else np.array([0, 0, 2])
        
        x = current_pos[0] + np.random.uniform(-1.5, 1.5)
        y = current_pos[1] + np.random.uniform(-1.5, 1.5)
        z = np.random.uniform(1.5, 2.5)
        
        self.target_position = np.array([x, y, z])
        logger.debug(f"生成悬停任务，目标: {self.target_position}")
    
    def _generate_flight_task(self):
        """生成飞行任务"""
        self.current_task = "flight"
        
        # 生成稍远的飞行目标
        current_pos = self.pos[0] if hasattr(self, 'pos') else np.array([0, 0, 2])
        
        distance = np.random.uniform(
            self.flight_config.min_target_distance,
            self.flight_config.max_target_distance
        )
        angle = np.random.uniform(0, 2 * np.pi)
        
        x = current_pos[0] + distance * np.cos(angle)
        y = current_pos[1] + distance * np.sin(angle)
        z = np.random.uniform(1.5, 3.0)
        
        self.target_position = np.array([x, y, z])
        logger.debug(f"生成飞行任务，目标: {self.target_position}")
    
    def _check_task_completion(self):
        """检查任务完成并可能更新目标"""
        if self.target_position is None:
            return
        
        current_pos = self.pos[0]
        distance_to_target = np.linalg.norm(current_pos - self.target_position)
        
        # 检查是否需要更新目标
        if (self.current_task == "flight" and distance_to_target < 1.0) or \
           (self.episode_step - self.last_target_change > self.flight_config.target_change_interval):
            self._generate_task()
            self.last_target_change = self.episode_step
    
    def _maybe_update_training_stage(self):
        """根据表现更新训练阶段"""
        if not self.flight_config.curriculum_learning:
            return
        
        # 悬停阶段 → 飞行阶段
        if (self.training_stage == "hover" and 
            self.hover_success_count >= self.flight_config.hover_success_threshold and
            self.episode_count >= self.flight_config.flight_unlock_episodes):
            
            self.training_stage = "flight"
            logger.info(f"训练阶段升级到：flight")
        
        # 飞行阶段 → 混合阶段（基于成功率）
        elif (self.training_stage == "flight" and 
              len(self.success_history) >= 20 and
              sum(self.success_history[-20:]) >= 15):  # 最近20次中成功15次
            
            self.training_stage = "mixed"
            logger.info(f"训练阶段升级到：mixed")
    
    def get_training_progress(self) -> Dict[str, Any]:
        """获取训练进度信息"""
        progress = {
            'training_stage': self.training_stage,
            'current_task': self.current_task,
            'episode_count': self.episode_count,
            'episode_step': self.episode_step,
            'hover_success_count': self.hover_success_count,
        }
        
        # 最近奖励统计
        if self.task_reward_history:
            recent_rewards = list(self.task_reward_history)[-20:]
            progress['avg_task_reward'] = np.mean([r['task_reward'] for r in recent_rewards])
            progress['avg_stability_reward'] = np.mean([r['stability_reward'] for r in recent_rewards])
            progress['avg_total_reward'] = np.mean([r['total_reward'] for r in recent_rewards])
        
        # 成功率统计
        if self.success_history:
            progress['recent_success_rate'] = np.mean(self.success_history[-50:]) if len(self.success_history) >= 50 else np.mean(self.success_history)
        
        return progress
    
    def set_training_stage(self, stage: str):
        """手动设置训练阶段"""
        if stage in ["hover", "flight", "mixed"]:
            old_stage = self.training_stage
            self.training_stage = stage
            logger.info(f"训练阶段手动切换: {old_stage} -> {stage}")
            # 重新生成任务
            self._generate_task()
        else:
            logger.warning(f"无效的训练阶段: {stage}")
    
    def get_target_info(self):
        """获取当前目标信息（用于可视化和调试）"""
        return {
            'target_position': self.target_position.tolist() if self.target_position is not None else None,
            'current_task': self.current_task,
            'training_stage': self.training_stage,
            'distance_to_target': np.linalg.norm(self.pos[0] - self.target_position) if self.target_position is not None else None
        }


# =============== 便捷函数 ===============

def create_base_flight_config(**kwargs) -> BaseFlightConfig:
    """创建基础飞行配置"""
    return BaseFlightConfig(**kwargs)

def create_base_flight_aviary(
    gui: bool = False,
    training_stage: str = "hover",
    obstacles: bool = False,  # 基础飞行训练通常不需要障碍物
    **kwargs
) -> BaseFlightAviary:
    """
    创建基础飞行训练环境
    
    Args:
        gui: 是否显示GUI
        training_stage: 训练阶段 ("hover", "flight", "mixed")
        obstacles: 是否启用障碍物
        **kwargs: 其他配置参数
    
    Returns:
        BaseFlightAviary: 配置好的训练环境
    """
    
    config = create_base_flight_config(**kwargs)
    
    env = BaseFlightAviary(
        gui=gui,
        obstacles=obstacles,
        config=config,
        **kwargs
    )
    
    env.set_training_stage(training_stage)
    
    logger.info(f"创建BaseFlightAviary环境：")
    logger.info(f"  - 训练阶段: {training_stage}")
    logger.info(f"  - 观测系统: 复用HAUAVAviary 86维观测")
    logger.info(f"  - 任务类型: 悬停和稳定飞行训练")
    
    return env


# =============== 使用示例 ===============

def example_usage():
    """使用示例"""
    
    # 创建基础悬停训练环境
    hover_env = create_base_flight_aviary(
        gui=False,
        training_stage="hover",
        obstacles=False
    )
    
    # 获取环境信息
    target_info = hover_env.get_target_info()
    progress = hover_env.get_training_progress()
    
    print("=== BaseFlightAviary 环境信息 ===")
    print(f"观测空间维度: 86 (复用HAUAVAviary)")
    print(f"当前任务: {target_info['current_task']}")
    print(f"训练阶段: {target_info['training_stage']}")
    print(f"目标位置: {target_info['target_position']}")
    
    # 运行一个回合
    obs, info = hover_env.reset()
    for step in range(100):
        action = hover_env.action_space.sample()  # 随机动作
        obs, reward, terminated, truncated, info = hover_env.step(action)
        
        if terminated or truncated:
            break
    
    hover_env.close()
    print("示例运行完成！")


if __name__ == "__main__":
    example_usage()
