import os
import numpy as np
import xml.etree.ElementTree as etxml
import pkg_resources
import socket 
import struct

from transforms3d.quaternions import rotate_vector, qconjugate, mat2quat, qmult
from transforms3d.utils import normalized_vector

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from src.utils.enums import DroneModel

class CTBRControl(object):
    """控制基类。

    实现了 `__init__()`、`reset()` 和接口 `computeControlFromState()`，
    主要控制方法 `computeControl()` 需由子类实现。

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
        #### 设置通用常量 #############################
        self.DRONE_MODEL = drone_model
        """DroneModel: The type of drone to control."""
        self.GRAVITY = g*self._getURDFParameter('m')
        """float: The gravitational force (M*g) acting on each drone."""
        self.KF = self._getURDFParameter('kf')
        """float: The coefficient converting RPMs into thrust."""
        self.KM = self._getURDFParameter('km')
        """float: The coefficient converting RPMs into torque."""
        
        self.reset()

    ################################################################################

    def reset(self):
        """重置控制类。

        通用计数器归零。

        """
        self.control_counter = 0

    ################################################################################

    def computeControlFromState(self,
                                control_timestep,
                                state,
                                target_pos,
                                target_rpy=np.zeros(3),
                                target_vel=np.zeros(3),
                                target_rpy_rates=np.zeros(3)
                                ):
        """接口方法，基于 `computeControl`。

        可直接根据 BaseAviary.step() 返回的 `obs` 字典中的 "state" 键值计算控制量。

        参数说明
        ----------
        control_timestep : float
            控制步长。
        state : ndarray
            (20,) 形状的数组，当前无人机状态。
        target_pos : ndarray
            (3,1) 期望位置。
        target_rpy : ndarray, optional
            (3,1) 期望欧拉角（roll, pitch, yaw）。
        target_vel : ndarray, optional
            (3,1) 期望速度。
        target_rpy_rates : ndarray, optional
            (3,1) 期望角速度（roll, pitch, yaw）。
        """

        return self.computeControl(control_timestep=control_timestep,
                                   cur_pos=state[0:3],
                                   cur_quat=np.array([state[6], state[3], state[4], state[5]]),
                                   cur_vel=state[10:13],
                                   cur_ang_vel=state[13:16],
                                   target_pos=target_pos,
                                   target_rpy=target_rpy,
                                   target_vel=target_vel,
                                   target_rpy_rates=target_rpy_rates
                                   )

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
        """抽象方法：计算单架无人机的控制量。

        必须由 BaseControl 的子类实现。

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
        """
        assert(cur_pos.shape == (3,)), f"cur_pos {cur_pos.shape}"
        assert(cur_quat.shape == (4,)), f"cur_quat {cur_quat.shape}"
        assert(cur_vel.shape == (3,)), f"cur_vel {cur_vel.shape}"
        assert(cur_ang_vel.shape == (3,)), f"cur_ang_vel {cur_ang_vel.shape}"
        assert(target_pos.shape == (3,)), f"target_pos {target_pos.shape}"
        assert(target_rpy.shape == (3,)), f"target_rpy {target_rpy.shape}"
        assert(target_vel.shape == (3,)), f"target_vel {target_vel.shape}"
        assert(target_rpy_rates.shape == (3,)), f"target_rpy_rates {target_rpy_rates.shape}"

        G = np.array([.0, .0, -9.8])
        K_P = np.array([3., 3., 8.])
        K_D = np.array([2.5, 2.5, 5.])
        K_RATES = np.array([5., 5., 1.])
        P = target_pos - cur_pos
        D = target_vel - cur_vel
        tar_acc = K_P * P + K_D * D - G
        norm_thrust = np.dot(tar_acc, rotate_vector([.0, .0, 1.], cur_quat))
        # Calculate target attitude
        z_body = normalized_vector(tar_acc)
        x_body = normalized_vector(np.cross(np.array([.0, 1., .0]), z_body))
        y_body = normalized_vector(np.cross(z_body, x_body))
        tar_att = mat2quat(np.vstack([x_body, y_body, z_body]).T)
        # Calculate body rates
        q_error = qmult(qconjugate(cur_quat), tar_att)
        body_rates = 2 * K_RATES * q_error[1:]
        if q_error[0] < 0:
            body_rates = -body_rates

        return norm_thrust, *body_rates

################################################################################

    def setPIDCoefficients(self,
                           p_coeff_pos=None,
                           i_coeff_pos=None,
                           d_coeff_pos=None,
                           p_coeff_att=None,
                           i_coeff_att=None,
                           d_coeff_att=None
                           ):
        """设置 PID 控制器的系数。

        如果未初始化 PID 系数（如当前控制器不是 PID），则报错并退出。

        参数说明
        ----------
        p_coeff_pos : ndarray, optional
            (3,1) 位置控制比例系数。
        i_coeff_pos : ndarray, optional
            (3,1) 位置控制积分系数。
        d_coeff_pos : ndarray, optional
            (3,1) 位置控制微分系数。
        p_coeff_att : ndarray, optional
            (3,1) 姿态控制比例系数。
        i_coeff_att : ndarray, optional
            (3,1) 姿态控制积分系数。
        d_coeff_att : ndarray, optional
            (3,1) 姿态控制微分系数。

        """
        ATTR_LIST = ['P_COEFF_FOR', 'I_COEFF_FOR', 'D_COEFF_FOR', 'P_COEFF_TOR', 'I_COEFF_TOR', 'D_COEFF_TOR']
        if not all(hasattr(self, attr) for attr in ATTR_LIST):
            print("[错误] BaseControl.setPIDCoefficients()：实例化的控制类中并非所有 PID 系数都已定义。")
            exit()
        else:
            self.P_COEFF_FOR = self.P_COEFF_FOR if p_coeff_pos is None else p_coeff_pos
            self.I_COEFF_FOR = self.I_COEFF_FOR if i_coeff_pos is None else i_coeff_pos
            self.D_COEFF_FOR = self.D_COEFF_FOR if d_coeff_pos is None else d_coeff_pos
            self.P_COEFF_TOR = self.P_COEFF_TOR if p_coeff_att is None else p_coeff_att
            self.I_COEFF_TOR = self.I_COEFF_TOR if i_coeff_att is None else i_coeff_att
            self.D_COEFF_TOR = self.D_COEFF_TOR if d_coeff_att is None else d_coeff_att

    ################################################################################
    
    def _getURDFParameter(self,
                          parameter_name: str
                          ):
        """从无人机的 URDF 文件中读取参数。

        该方法本质上是对 `assets/` 文件夹下 .urdf 文件的自定义 XML 解析。

        参数说明
        ----------
        parameter_name : str
            要读取的参数名。

        返回值
        -------
        float
            参数的数值。

        """
        #### 获取当前无人机模型的 XML 树 ########
        URDF = self.DRONE_MODEL.value + ".urdf"
        path = pkg_resources.resource_filename('gym_pybullet_drones', 'assets/'+URDF)
        URDF_TREE = etxml.parse(path).getroot()
        #### 查找并返回目标参数 #################
        if parameter_name == 'm':
            return float(URDF_TREE[1][0][1].attrib['value'])
        elif parameter_name in ['ixx', 'iyy', 'izz']:
            return float(URDF_TREE[1][0][2].attrib[parameter_name])
        elif parameter_name in ['arm', 'thrust2weight', 'kf', 'km', 'max_speed_kmh', 'gnd_eff_coeff' 'prop_radius', \
                                'drag_coeff_xy', 'drag_coeff_z', 'dw_coeff_1', 'dw_coeff_2', 'dw_coeff_3']:
            return float(URDF_TREE[0].attrib[parameter_name])
        elif parameter_name in ['length', 'radius']:
            return float(URDF_TREE[1][2][1][0].attrib[parameter_name])
        elif parameter_name == 'collision_z_offset':
            COLLISION_SHAPE_OFFSETS = [float(s) for s in URDF_TREE[1][2][0].attrib['xyz'].split(' ')]
            return COLLISION_SHAPE_OFFSETS[2]
