from enum import Enum

class DroneModel(Enum):
    """无人机模型枚举类。"""

    CF2X = "cf2x"   # Bitcraze Craziflie 2.0，X 架构
    CF2P = "cf2p"   # Bitcraze Craziflie 2.0，+ 架构
    RACE = "racer"  # 竞速型无人机，X 架构


################################################################################

class Physics(Enum):
    """物理实现枚举类。"""

    PYB = "pyb"                         # 基础 PyBullet 物理更新
    DYN = "dyn"                         # 显式动力学模型
    PYB_GND = "pyb_gnd"                 # 带地面效应的 PyBullet 物理更新
    PYB_DRAG = "pyb_drag"               # 带空气阻力的 PyBullet 物理更新
    PYB_DW = "pyb_dw"                   # 带下洗气流的 PyBullet 物理更新
    PYB_GND_DRAG_DW = "pyb_gnd_drag_dw" # 同时包含地面效应、空气阻力和下洗气流的 PyBullet 物理更新

################################################################################

class ImageType(Enum):
    """相机捕获图像类型枚举类。"""

    RGB = 0     # 彩色图像（含 alpha 通道）
    DEP = 1     # 深度图像
    SEG = 2     # 按对象 ID 分割的图像
    BW = 3      # 黑白图像

################################################################################

class ActionType(Enum):
    """动作类型枚举类。"""
    RPM = "rpm"                 # 电机转速输入
    PID = "pid"                 # PID 控制输入
    VEL = "vel"                 # 速度输入（使用 PID 控制）
    VEL_CTRL = "vel_ctrl"       # 速度输入（使用专门的VelControl控制器）
    ONE_D_RPM = "one_d_rpm"     # 1D（所有电机输入相同转速）
    ONE_D_PID = "one_d_pid"     # 1D（所有电机输入相同 PID 控制）

################################################################################

class ObservationType(Enum):
    """观测类型枚举类。"""
    KIN = "kin"     # 运动学信息（位姿、线速度、角速度）
    RGB = "rgb"     # 每架无人机视角下的 RGB 图像
    LASER = "laser" # laser capture in each drone's FOV in world frame

