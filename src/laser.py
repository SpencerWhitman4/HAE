"""脚本演示了仿真与控制的联合使用。

仿真由 `Baimport sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
from src.utils.enums import DroneModel, Physics
from src.envs.BaseRLAviary import BaseRLAviary
from src.controllers.DSLPIDControl import DSLPIDControl
from src.utils.Logger import Logger
from src.utils.utils import sync, str2booliary` 环境运行。
控制由 `DSLPIDControl` 中的 PID 实现提供。

示例
-------
在终端中运行：

    $ python pid.py

注意
-----
无人机在 X-Y 平面内以不同的高度沿圆形轨迹移动，
围绕点 (0, -.3)。

"""
import os
import time
import argparse
from datetime import datetime
import pdb
import math
import random
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from src.utils.enums import DroneModel, Physics
from src.envs.BaseRLAviary import BaseRLAviary
from src.controllers.DSLPIDControl import DSLPIDControl
from src.utils.Logger import Logger
from src.utils.utils import sync, str2bool

DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 1
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_RECORD_LASER = True
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_OBSTACLES = True
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 12
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

def run(
        drone=DEFAULT_DRONES,
        num_drones=DEFAULT_NUM_DRONES,
        physics=DEFAULT_PHYSICS,
        gui=DEFAULT_GUI,
        laser_record=DEFAULT_RECORD_LASER,
        plot=DEFAULT_PLOT,
        user_debug_gui=DEFAULT_USER_DEBUG_GUI,
        obstacles=DEFAULT_OBSTACLES,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        duration_sec=DEFAULT_DURATION_SEC,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        colab=DEFAULT_COLAB
        ):
    #### 初始化仿真环境 #############################
    H = 3.5  # 初始高度
    H_STEP = .5  # 高度步长
    R = .3  # 圆形轨迹半径
    INIT_XYZS = np.array([[R*np.cos((i/6)*2*np.pi+np.pi/2), R*np.sin((i/6)*2*np.pi+np.pi/2)-R, H+i*H_STEP] for i in range(num_drones)])
    INIT_RPYS = np.array([[0, 0,  i * (np.pi/2)/num_drones] for i in range(num_drones)])

    #### 初始化圆形轨迹 ##############################
    PERIOD = 10  # 轨迹周期（秒）
    NUM_WP = control_freq_hz*PERIOD  # 轨迹点数量
    TARGET_POS = np.zeros((NUM_WP,3))
    for i in range(NUM_WP):
        TARGET_POS[i, :] = R*np.cos((i/NUM_WP)*(2*np.pi)+np.pi/2)+INIT_XYZS[0, 0], R*np.sin((i/NUM_WP)*(2*np.pi)+np.pi/2)-R+INIT_XYZS[0, 1], 0
    wp_counters = np.array([int((i*NUM_WP/6)%NUM_WP) for i in range(num_drones)])


    #### 创建仿真环境 ################################
    env = BaseRLAviary(drone_model=drone,
                        num_drones=num_drones,
                        initial_xyzs=INIT_XYZS,
                        initial_rpys=INIT_RPYS,
                        physics=physics,
                        neighbourhood_radius=10,
                        pyb_freq=simulation_freq_hz,
                        ctrl_freq=control_freq_hz,
                        gui=gui,
                        laser_record=laser_record,
                        output_folder="/home/lxy/Code/gym-pybullet-drones-RL/log"
                        # obstacles=obstacles,
                        # user_debug_gui=user_debug_gui
                        )

    #### 获取 PyBullet 客户端 ID ####################
    PYB_CLIENT = env.getPyBulletClient()

    #### 初始化控制器 ###############################
    if drone in [DroneModel.CF2X, DroneModel.CF2P]:
        ctrl = [DSLPIDControl(drone_model=drone) for i in range(num_drones)]

    #### 运行仿真 ###################################
    action = np.zeros((num_drones,4))  # VEL模式需要4个元素
    START = time.time()
    for i in range(0, int(duration_sec*env.CTRL_FREQ)):

        #### 执行仿真步 #################################
        obs, reward, terminated, truncated, info = env.step(action)

        #### 计算当前轨迹点的控制 #########################
        for j in range(num_drones):
            # 获取当前位置
            current_pos = obs[j][0:3]
            # 计算目标位置（圆形轨迹点）
            target_pos = np.array([TARGET_POS[wp_counters[j], 0], TARGET_POS[wp_counters[j], 1], INIT_XYZS[j, 2]])
            # 计算方向向量
            direction = target_pos - current_pos
            # 如果距离太小，就保持当前位置
            if np.linalg.norm(direction) < 0.01:
                direction = np.array([0.0, 0.0, 0.0])
            # 设置动作：方向向量(0:3) + 速度因子(3)
            action[j, 0:3] = direction
            action[j, 3] = 1.0  # 速度因子，1.0表示使用最大允许速度

        #### 更新到下一个轨迹点并循环 #####################
        for j in range(num_drones):
            wp_counters[j] = wp_counters[j] + 1 if wp_counters[j] < (NUM_WP-1) else 0

        #### 渲染仿真 ###################################
        env.render()

        #### 同步仿真 ###################################
        if gui:
            sync(i, START, env.CTRL_TIMESTEP)

    #### 关闭仿真环境 #################################
    env.close()

if __name__ == "__main__":
    #### 定义并解析脚本的可选参数 #####################
    parser = argparse.ArgumentParser(description='使用 BaseLaserAviary 和 DSLPIDControl 的螺旋飞行脚本')
    parser.add_argument('--drone',              default=DEFAULT_DRONES,     type=DroneModel,    help='无人机模型（默认: CF2X）', metavar='', choices=DroneModel)
    parser.add_argument('--num_drones',         default=DEFAULT_NUM_DRONES,          type=int,           help='无人机数量（默认: 1）', metavar='')
    parser.add_argument('--physics',            default=DEFAULT_PHYSICS,      type=Physics,       help='物理更新方式（默认: PYB）', metavar='', choices=Physics)
    parser.add_argument('--gui',                default=DEFAULT_GUI,       type=str2bool,      help='是否使用 PyBullet GUI（默认: True）', metavar='')
    parser.add_argument('--laser_record',       default=DEFAULT_RECORD_LASER,      type=str2bool,      help='是否记录点云（默认: False）', metavar='')
    parser.add_argument('--plot',               default=DEFAULT_PLOT,       type=str2bool,      help='是否绘制仿真结果（默认: True）', metavar='')
    parser.add_argument('--user_debug_gui',     default=DEFAULT_USER_DEBUG_GUI,      type=str2bool,      help='是否在 GUI 中添加调试线和参数（默认: False）', metavar='')
    parser.add_argument('--obstacles',          default=DEFAULT_OBSTACLES,       type=str2bool,      help='是否在环境中添加障碍物（默认: True）', metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ,        type=int,           help='仿真频率（Hz）（默认: 240）', metavar='')
    parser.add_argument('--control_freq_hz',    default=DEFAULT_CONTROL_FREQ_HZ,         type=int,           help='控制频率（Hz）（默认: 48）', metavar='')
    parser.add_argument('--duration_sec',       default=DEFAULT_DURATION_SEC,         type=int,           help='仿真持续时间（秒）（默认: 12）', metavar='')
    parser.add_argument('--output_folder',     default=DEFAULT_OUTPUT_FOLDER, type=str,           help='保存日志的文件夹（默认: "results"）', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB, type=bool,           help='是否在 Jupyter Notebook 中运行（默认: False）', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))