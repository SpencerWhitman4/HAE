import math
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from src.controllers.BaseControl import BaseControl
from src.utils.enums import DroneModel

class DSLPIDControl(BaseControl):
    """Crazyflie 无人机的 PID 控制类。

    基于 UTIAS DSL 团队的相关工作。贡献者：SiQi Zhou, James Xu, Tracy Du, Mario Vukosavljev, Calvin Ngan, Jingyuan Hou。

    """

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel,
                 g: float=9.8
                 ):
        """通用控制类的初始化方法。

        参数说明
        ----------
        drone_model : DroneModel
            要控制的无人机类型（详见 `assets` 文件夹下的 .urdf 文件）。
        g : float, optional
            重力加速度（单位 m/s^2）。

        """
        super().__init__(drone_model=drone_model, g=g)
        # if self.DRONE_MODEL != DroneModel.CF2X and self.DRONE_MODEL != DroneModel.CF2P:
        #     print("[ERROR] in DSLPIDControl.__init__(), DSLPIDControl requires DroneModel.CF2X or DroneModel.CF2P")
        #     exit()
        # 调整PID参数以改善速度控制模式的性能
        self.P_COEFF_FOR = np.array([1.2, 1.2, 1.25])    # 增加位置P增益
        self.I_COEFF_FOR = np.array([.05, .05, .05])
        self.D_COEFF_FOR = np.array([.6, .6, .5])        # 增加位置D增益
        self.P_COEFF_TOR = np.array([35000., 35000., 60000.])  # 减少姿态P增益
        self.I_COEFF_TOR = np.array([.0, .0, 500.])
        self.D_COEFF_TOR = np.array([10000., 10000., 12000.])  # 减少姿态D增益
        self.PWM2RPM_SCALE = 0.2685
        self.PWM2RPM_CONST = 4070.3
        self.MIN_PWM = 20000
        self.MAX_PWM = 65535
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
        self.reset()

    ################################################################################

    def reset(self):
        """重置控制类。

        上一步的误差和积分误差（位置和姿态）全部归零。

        """
        super().reset()
        #### Store the last roll, pitch, and yaw ###################
        self.last_rpy = np.zeros(3)
        #### Initialized PID control variables #####################
        self.last_pos_e = np.zeros(3)
        self.integral_pos_e = np.zeros(3)
        self.last_rpy_e = np.zeros(3)
        self.integral_rpy_e = np.zeros(3)

    ################################################################################
    
    def computeControl(self,
                       control_timestep,
                       cur_pos,
                       cur_quat,
                       cur_vel,
                       cur_ang_vel,
                       target_pos,
                       target_rpy=np.zeros(3),
                       target_vel=np.zeros(3),
                       target_rpy_rates=np.zeros(3)
                       ):
        """计算单架无人机的 PID 控制量（以 RPM 输出）。

        该方法依次调用 `_dslPIDPositionControl()` 和 `_dslPIDAttitudeControl()`。
        参数 `cur_ang_vel` 未被使用。

        参数说明
        ----------
        control_timestep : float
            控制步长。
        cur_pos : ndarray
            (3,1) 当前无人机位置。
        cur_quat : ndarray
            (4,1) 当前无人机四元数姿态。
        cur_vel : ndarray
            (3,1) 当前无人机速度。
        cur_ang_vel : ndarray
            (3,1) 当前无人机角速度。
        target_pos : ndarray
            (3,1) 期望位置。
        target_rpy : ndarray, optional
            (3,1) 期望欧拉角（roll, pitch, yaw）。
        target_vel : ndarray, optional
            (3,1) 期望速度。
        target_rpy_rates : ndarray, optional
            (3,1) 期望角速度（roll, pitch, yaw）。

        返回值
        -------
        ndarray
            (4,1) 各电机 RPM。
        ndarray
            (3,1) 当前 XYZ 位置误差。
        float
            当前偏航误差。

        """
        self.control_counter += 1
        thrust, computed_target_rpy, pos_e = self._dslPIDPositionControl(control_timestep,
                                                                         cur_pos,
                                                                         cur_quat,
                                                                         cur_vel,
                                                                         target_pos,
                                                                         target_rpy,
                                                                         target_vel
                                                                         )
        rpm = self._dslPIDAttitudeControl(control_timestep,
                                          thrust,
                                          cur_quat,
                                          computed_target_rpy,
                                          target_rpy_rates
                                          )
        cur_rpy = p.getEulerFromQuaternion(cur_quat)
        return rpm, pos_e, computed_target_rpy[2] - cur_rpy[2]
    
    ################################################################################

    def _dslPIDPositionControl(self,
                               control_timestep,
                               cur_pos,
                               cur_quat,
                               cur_vel,
                               target_pos,
                               target_rpy,
                               target_vel
                               ):
        """DSL 的 CF2.x PID 位置控制。

        参数说明
        ----------
        control_timestep : float
            控制步长。
        cur_pos : ndarray
            (3,1) 当前无人机位置。
        cur_quat : ndarray
            (4,1) 当前无人机四元数姿态。
        cur_vel : ndarray
            (3,1) 当前无人机速度。
        target_pos : ndarray
            (3,1) 期望位置。
        target_rpy : ndarray
            (3,1) 期望欧拉角（roll, pitch, yaw）。
        target_vel : ndarray
            (3,1) 期望速度。

        返回值
        -------
        float
            沿无人机 z 轴的目标推力。
        ndarray
            (3,1) 目标欧拉角（roll, pitch, yaw）。
        float
            当前位置误差。

        """
        cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        pos_e = target_pos - cur_pos
        vel_e = target_vel - cur_vel
        self.integral_pos_e = self.integral_pos_e + pos_e*control_timestep
        self.integral_pos_e = np.clip(self.integral_pos_e, -2., 2.)
        self.integral_pos_e[2] = np.clip(self.integral_pos_e[2], -0.15, .15)
        #### PID target thrust #####################################
        target_thrust = np.multiply(self.P_COEFF_FOR, pos_e) \
                        + np.multiply(self.I_COEFF_FOR, self.integral_pos_e) \
                        + np.multiply(self.D_COEFF_FOR, vel_e) + np.array([0, 0, self.GRAVITY])
        scalar_thrust = max(0., np.dot(target_thrust, cur_rotation[:,2]))
        thrust = (math.sqrt(scalar_thrust / (4*self.KF)) - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE
        target_z_ax = target_thrust / np.linalg.norm(target_thrust)
        target_x_c = np.array([math.cos(target_rpy[2]), math.sin(target_rpy[2]), 0])
        target_y_ax = np.cross(target_z_ax, target_x_c) / np.linalg.norm(np.cross(target_z_ax, target_x_c))
        target_x_ax = np.cross(target_y_ax, target_z_ax)
        target_rotation = (np.vstack([target_x_ax, target_y_ax, target_z_ax])).transpose()
        #### Target rotation #######################################
        target_euler = (Rotation.from_matrix(target_rotation)).as_euler('XYZ', degrees=False)
        if np.any(np.abs(target_euler) > math.pi):
            print("\n[ERROR] ctrl it", self.control_counter, "in Control._dslPIDPositionControl(), values outside range [-pi,pi]")
        return thrust, target_euler, pos_e
    
    ################################################################################

    def _dslPIDAttitudeControl(self,
                               control_timestep,
                               thrust,
                               cur_quat,
                               target_euler,
                               target_rpy_rates
                               ):
        """DSL 的 CF2.x PID 姿态控制。

        参数说明
        ----------
        control_timestep : float
            控制步长。
        thrust : float
            沿无人机 z 轴的目标推力。
        cur_quat : ndarray
            (4,1) 当前无人机四元数姿态。
        target_euler : ndarray
            (3,1) 计算得到的目标欧拉角。
        target_rpy_rates : ndarray
            (3,1) 期望欧拉角速度。

        返回值
        -------
        ndarray
            (4,1) 各电机 RPM。

        """
        cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        cur_rpy = np.array(p.getEulerFromQuaternion(cur_quat))
        target_quat = (Rotation.from_euler('XYZ', target_euler, degrees=False)).as_quat()
        w,x,y,z = target_quat
        target_rotation = (Rotation.from_quat([w, x, y, z])).as_matrix()
        rot_matrix_e = np.dot((target_rotation.transpose()),cur_rotation) - np.dot(cur_rotation.transpose(),target_rotation)
        rot_e = np.array([rot_matrix_e[2, 1], rot_matrix_e[0, 2], rot_matrix_e[1, 0]]) 
        rpy_rates_e = target_rpy_rates - (cur_rpy - self.last_rpy)/control_timestep
        self.last_rpy = cur_rpy
        self.integral_rpy_e = self.integral_rpy_e - rot_e*control_timestep
        self.integral_rpy_e = np.clip(self.integral_rpy_e, -1500., 1500.)
        self.integral_rpy_e[0:2] = np.clip(self.integral_rpy_e[0:2], -1., 1.)
        #### PID target torques ####################################
        target_torques = - np.multiply(self.P_COEFF_TOR, rot_e) \
                         + np.multiply(self.D_COEFF_TOR, rpy_rates_e) \
                         + np.multiply(self.I_COEFF_TOR, self.integral_rpy_e)
        target_torques = np.clip(target_torques, -3200, 3200)
        pwm = thrust + np.dot(self.MIXER_MATRIX, target_torques)
        pwm = np.clip(pwm, self.MIN_PWM, self.MAX_PWM)
        return self.PWM2RPM_SCALE * pwm + self.PWM2RPM_CONST
    
    ################################################################################

    def _one23DInterface(self,
                         thrust
                         ):
        """用于 1、2 或 3 维推力输入场景的接口工具函数。

        参数说明
        ----------
        thrust : ndarray
            长度为 1、2 或 4 的数组，表示期望推力输入。

        返回值
        -------
        ndarray
            (4,1) 各电机 PWM（非 RPM）。

        """
        DIM = len(np.array(thrust))
        pwm = np.clip((np.sqrt(np.array(thrust)/(self.KF*(4/DIM)))-self.PWM2RPM_CONST)/self.PWM2RPM_SCALE, self.MIN_PWM, self.MAX_PWM)
        if DIM in [1, 4]:
            return np.repeat(pwm, 4/DIM)
        elif DIM==2:
            return np.hstack([pwm, np.flip(pwm)])
        else:
            print("[ERROR] in DSLPIDControl._one23DInterface()")
            exit()
