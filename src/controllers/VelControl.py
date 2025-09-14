#!/usr/bin/env python3
"""
VelControl - 增量速度控制器（基于DSLPIDControl参数扩展）
"""

import math
import numpy as np
import pybullet as p
import sys
from pathlib import Path
from typing import Optional, Tuple
from scipy.spatial.transform import Rotation
from collections import deque

# 添加项目根路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.controllers.BaseControl import BaseControl
from src.utils.enums import DroneModel

class VelControl(BaseControl):
    """
    增量速度控制器 - 基于BaseControl扩展增量控制能力
    """

    def __init__(self, drone_model: DroneModel, g: float = 9.8):
        # 验证支持的无人机型号
        if drone_model not in [DroneModel.CF2X, DroneModel.CF2P]:
            raise ValueError(f"VelControl只支持CF2X和CF2P, 当前: {drone_model}")
        
        # 滑动窗口属性
        self.window_size = 5
        self.velocity_window = np.zeros((self.window_size, 4))
        self.window_index = 0
        
        # 轨迹记录接口
        self.last_target_action = None
        
        # 增量控制状态
        self.increment_mode = False
        self.pending_increment = None
        
        # 调用父类初始化
        super().__init__(drone_model, g)
        
        # 初始化物理参数
        self._init_physical_parameters()
        
        # self.logger.debug("[VelControl] 增量控制器初始化完成")

    def _init_physical_parameters(self):
        """初始化物理控制参数（调整后适合速度控制）"""
        # 速度控制的PID参数（比位置控制更温和）
        self.P_COEFF_FOR = np.array([0.8, 0.8, 1.0])     # 位置P增益（降低）
        self.I_COEFF_FOR = np.array([0.02, 0.02, 0.03])  # 位置I增益（降低）
        self.D_COEFF_FOR = np.array([0.4, 0.4, 0.3])     # 位置D增益（降低）
        
        # 姿态控制参数（大幅降低避免饱和）
        self.P_COEFF_TOR = np.array([12000., 12000., 20000.])  # 姿态P增益（降低）
        self.I_COEFF_TOR = np.array([0.0, 0.0, 200.])         # 姿态I增益（降低）
        self.D_COEFF_TOR = np.array([3000., 3000., 4000.])     # 姿态D增益（降低）
        
        # PWM-RPM转换参数（与DSLPIDControl一致）
        self.PWM2RPM_SCALE = 0.2685
        self.PWM2RPM_CONST = 4070.3
        self.MIN_PWM = 20000
        self.MAX_PWM = 65535
        
        # 混合矩阵（基于无人机模型）
        if self.DRONE_MODEL == DroneModel.CF2X:
            self.MIXER_MATRIX = np.array([ 
                                    [-.5, -.5, -1],
                                    [-.5,  .5,  1],
                                    [.5, .5, -1],
                                    [.5, -.5,  1]
                                    ])
        elif self.DRONE_MODEL == DroneModel.CF2P:
            self.MIXER_MATRIX = np.array([
                                    [0, -1,  -1],
                                    [+1, 0, 1],
                                    [0,  1,  -1],
                                    [-1, 0, 1]
                                    ])
        
        # 控制增益和限制
        self.MAX_TILT_ANGLE = math.radians(15)
        self.MAX_INTEGRAL_ERROR = 2.0
        self.MAX_INTEGRAL_ERROR_Z = 0.15
        
        # PID积分误差存储
        self.integral_pos_e = np.zeros(3)
        self.integral_rpy_e = np.zeros(3)
        self.last_rpy = np.zeros(3)
        
        # 目标速度状态
        self.target_velocity = np.zeros(4)
        self.last_target_rpy = np.zeros(3)

    def reset(self):
        """重置控制器状态"""
        super().reset()
        
        # 重置滑动窗口
        self.velocity_window.fill(0.0)
        self.window_index = 0
        
        # 重置目标状态
        self.target_velocity = np.zeros(4)
        self.last_target_rpy = np.zeros(3)
        
        # 重置PID积分误差（与DSLPIDControl一致）
        self.integral_pos_e = np.zeros(3)
        self.integral_rpy_e = np.zeros(3)
        self.last_rpy = np.zeros(3)
        
        # 重置增量控制状态
        self.increment_mode = False
        self.pending_increment = None

    def set_increment_action(self, increment: np.ndarray):
        """设置增量动作供下次computeControl使用"""
        self.pending_increment = increment.copy()
        self.increment_mode = True

    def update_velocity_window(self, velocity_increment: np.ndarray, confidence: Optional[np.ndarray] = None) -> np.ndarray:
        """滑动窗口平滑速度增量"""
        # 验证输入
        velocity_increment = np.array(velocity_increment, dtype=np.float32)
        if velocity_increment.shape != (4,):
            raise ValueError(f"velocity_increment必须是4维向量，当前: {velocity_increment.shape}")
        
        # 物理范围约束
        max_linear_vel = 8.33  # m/s
        max_angular_vel = 4.0  # rad/s
        
        velocity_increment = np.clip(velocity_increment, 
                                   [-max_linear_vel, -max_linear_vel, -max_linear_vel, -max_angular_vel],
                                   [max_linear_vel, max_linear_vel, max_linear_vel, max_angular_vel])
        
        # 更新滑动窗口
        self.velocity_window[self.window_index] = velocity_increment
        self.window_index = (self.window_index + 1) % self.window_size
        
        # 计算加权移动平均
        if confidence is not None:
            confidence = np.array(confidence, dtype=np.float32)
            confidence = np.clip(confidence, 0.0, 1.0)
            weights = np.array([confidence * (0.8 ** i) for i in range(self.window_size)])
        else:
            weights = np.array([0.8 ** i for i in range(self.window_size)])
        
        # 归一化权重
        weights = weights / np.sum(weights)
        
        # 应用权重计算平滑增量
        smoothed_increment = np.average(self.velocity_window, axis=0, weights=weights)
        
        # 存储当前目标
        self.target_velocity = smoothed_increment
        
        return smoothed_increment

    def compute_target_state(self, cur_quat: np.ndarray, cur_vel: np.ndarray, 
                           smoothed_increment: np.ndarray, control_timestep: float) -> Tuple[np.ndarray, np.ndarray]:
        """计算目标速度和姿态"""
        # 计算目标速度（积分增量）
        target_vel = cur_vel + smoothed_increment[:3] * control_timestep
        target_vel = np.clip(target_vel, -8.33, 8.33)  # 物理速度限制
        
        # 当前姿态转换
        current_rpy = Rotation.from_quat(cur_quat).as_euler('xyz')
        
        # 计算目标偏航角
        target_yaw = current_rpy[2] + smoothed_increment[3] * control_timestep
        
        # 基于目标速度计算所需的俯仰和横滚角
        # 速度到角度的转换增益
        vel_to_angle_gain = 0.3
        
        # 目标俯仰角（前向速度）
        target_pitch = -np.clip(target_vel[0] * vel_to_angle_gain, -self.MAX_TILT_ANGLE, self.MAX_TILT_ANGLE)
        
        # 目标横滚角（侧向速度）
        target_roll = np.clip(target_vel[1] * vel_to_angle_gain, -self.MAX_TILT_ANGLE, self.MAX_TILT_ANGLE)
        
        target_rpy = np.array([target_roll, target_pitch, target_yaw])
        
        # 保存用于连续性
        self.last_target_rpy = target_rpy
        
        return target_vel, target_rpy

    def compute_attitude_torques(self, cur_quat: np.ndarray, target_rpy: np.ndarray) -> np.ndarray:
        """计算姿态控制力矩（使用DSLPIDControl的姿态控制逻辑）"""
        from scipy.spatial.transform import Rotation
        import pybullet as p
        
        # 当前姿态矩阵和欧拉角
        cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        cur_rpy = np.array(p.getEulerFromQuaternion(cur_quat))
        
        # 目标姿态矩阵
        target_quat = (Rotation.from_euler('XYZ', target_rpy, degrees=False)).as_quat()
        w, x, y, z = target_quat
        target_rotation = (Rotation.from_quat([w, x, y, z])).as_matrix()
        
        # 旋转误差矩阵（DSLPIDControl方式）
        rot_matrix_e = (np.dot((target_rotation.transpose()), cur_rotation) - 
                       np.dot(cur_rotation.transpose(), target_rotation))
        rot_e = np.array([rot_matrix_e[2, 1], rot_matrix_e[0, 2], rot_matrix_e[1, 0]])
        
        # 角速度误差（简化版，目标角速度为0）
        rpy_rates_e = np.zeros(3) - (cur_rpy - self.last_rpy) / 0.033  # 假设30Hz控制频率
        self.last_rpy = cur_rpy
        
        # 积分误差更新
        self.integral_rpy_e = self.integral_rpy_e - rot_e * 0.033
        self.integral_rpy_e = np.clip(self.integral_rpy_e, -1500., 1500.)
        self.integral_rpy_e[0:2] = np.clip(self.integral_rpy_e[0:2], -1., 1.)
        
        # PID力矩计算（DSLPIDControl方式）
        target_torques = (-np.multiply(self.P_COEFF_TOR, rot_e) +
                         np.multiply(self.D_COEFF_TOR, rpy_rates_e) +
                         np.multiply(self.I_COEFF_TOR, self.integral_rpy_e))
        
        return target_torques

    def computeControl(self,
                      control_timestep: float,
                      cur_pos: np.ndarray,
                      cur_quat: np.ndarray,
                      cur_vel: np.ndarray,
                      cur_ang_vel: np.ndarray,
                      target_pos: np.ndarray,
                      target_rpy: np.ndarray,
                      target_vel: np.ndarray,
                      target_rpy_rates: np.ndarray) -> np.ndarray:
        """
        增量控制增强版computeControl
        """
        
        if self.increment_mode and self.pending_increment is not None:
            # === 增量控制路径 ===
            
            # 1. 滑动窗口平滑
            smoothed_increment = self.update_velocity_window(self.pending_increment)
            
            # 2. 计算增量目标状态
            target_vel_adjusted, target_rpy_adjusted = self.compute_target_state(
                cur_quat, cur_vel, smoothed_increment, control_timestep
            )
            
            # 3. 计算姿态控制力矩
            attitude_torques = self.compute_attitude_torques(cur_quat, target_rpy_adjusted)
            
            # 4. 构建完整控制指令
            target_rpy_rates_adjusted = np.array([0, 0, smoothed_increment[3]])
            
            # 5. 调用RPM计算逻辑
            rpm = self._compute_rpm_from_controls(
                cur_vel, target_vel_adjusted, attitude_torques, control_timestep
            )
            
            # 6. 存储轨迹记录
            self.last_target_action = np.concatenate([
                target_vel_adjusted, [target_rpy_rates_adjusted[2]]
            ])
            
            # 7. 重置增量状态
            self.increment_mode = False
            self.pending_increment = None
            
            return rpm
            
        else:
            # === 标准控制路径 ===
            # 输入验证
            cur_quat = np.array(cur_quat, dtype=np.float32)
            cur_vel = np.array(cur_vel, dtype=np.float32)
            target_vel = np.array(target_vel, dtype=np.float32)
            target_rpy_rates = np.array(target_rpy_rates, dtype=np.float32)
            
            # 存储target_action用于轨迹记录
            self.last_target_action = np.concatenate([
                target_vel, [target_rpy_rates[2]]
            ])
            
            # 构建速度增量指令
            velocity_increment = np.concatenate([target_vel, [target_rpy_rates[2]]])
            
            # 使用完整控制逻辑
            smoothed_increment = self.update_velocity_window(velocity_increment)
            target_vel_computed, target_rpy_computed = self.compute_target_state(
                cur_quat, cur_vel, smoothed_increment, control_timestep
            )
            attitude_torques = self.compute_attitude_torques(cur_quat, target_rpy_computed)
            
            # RPM计算
            rpm = self._compute_rpm_from_controls(
                cur_vel, target_vel_computed, attitude_torques, control_timestep
            )
            
            return rpm

    def _compute_rpm_from_controls(self, cur_vel, target_vel, attitude_torques, control_timestep):
        """将控制指令转换为RPM（增强电机差异版本）"""
        # 速度误差计算
        vel_e = target_vel - cur_vel
        
        # 积分误差更新（仅用于I项）
        self.integral_pos_e += vel_e * control_timestep
        self.integral_pos_e = np.clip(self.integral_pos_e, -self.MAX_INTEGRAL_ERROR, self.MAX_INTEGRAL_ERROR)
        self.integral_pos_e[2] = np.clip(self.integral_pos_e[2], -self.MAX_INTEGRAL_ERROR_Z, self.MAX_INTEGRAL_ERROR_Z)
        
        # 基础推力计算（用于保持悬停）
        base_thrust = self.GRAVITY + np.multiply(self.P_COEFF_FOR[2], vel_e[2])
        base_thrust = max(0.5, base_thrust)  # 避免负推力
        
        # 转换基础推力为基础RPM
        base_rpm = math.sqrt(base_thrust / (4 * self.KF))
        base_rpm = np.clip(base_rpm, 8000.0, 12000.0)  # 基础RPM在中等范围
        
        # 关键修改：放大力矩差异的影响
        # 增加混合矩阵的缩放系数，让电机差异更大
        torque_scale = 15.0  # 大幅增加缩放因子
        scaled_torques = attitude_torques * torque_scale
        
        # 更宽松的力矩限制，但避免过度饱和
        max_torque_effect = 4000.0  # 允许±4000 RPM的差异
        limited_torques = np.clip(scaled_torques, -max_torque_effect, max_torque_effect)
        
        # 使用混合矩阵计算每个电机的RPM差异
        # 放大混合矩阵的效果
        enhanced_mixer = self.MIXER_MATRIX * 2.0  # 再次放大混合矩阵
        rpm_diff = np.dot(enhanced_mixer, limited_torques)
        
        # 构建最终RPM：基础RPM + 差异RPM
        rpm = base_rpm + rpm_diff
        
        # 保持在5000-16000范围，但允许更大的电机间差异
        return np.clip(rpm, 5000.0, 16000.0).astype(np.float32)

    def reset_increment_state(self):
        """重置增量控制状态"""
        self.increment_mode = False
        self.pending_increment = None
        self.velocity_window.fill(0.0)
        self.window_index = 0

    def get_target_action(self):
        """轨迹记录接口"""
        if self.last_target_action is not None:
            return np.round(self.last_target_action.copy(), 4)
        return None


if __name__ == "__main__":
    print("🧪 VelControl 增量控制测试")
    
    try:
        controller = VelControl(DroneModel.CF2X)
        # print("✅ 控制器创建成功")
        
        # 测试增量控制
        increment = np.array([0.5, 0.0, 0.1, 0.1])  # [dvx, dvy, dvz, dyaw]
        controller.set_increment_action(increment)
        
        test_rpm = controller.computeControl(
            control_timestep=1.0/30.0,
            cur_pos=np.array([0, 0, 1]),
            cur_quat=np.array([0, 0, 0, 1]),
            cur_vel=np.array([0, 0, 0]),
            cur_ang_vel=np.array([0, 0, 0]),
            target_pos=np.array([0, 0, 1]),
            target_rpy=np.array([0, 0, 0]),
            target_vel=np.array([0, 0, 0]),
            target_rpy_rates=np.array([0, 0, 0])
        )
        
        print(f"✅ 增量控制测试成功 - RPM: {test_rpm}")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
