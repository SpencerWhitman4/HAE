import os
import numpy as np
import pybullet as p
from gymnasium import spaces
from collections import deque

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from src.envs.BaseAviary import BaseAviary
from src.utils.enums import DroneModel, Physics, ActionType, ObservationType, ImageType
from src.controllers.DSLPIDControl import DSLPIDControl

class BaseRLAviary(BaseAviary):
    """强化学习用的单智能体和多智能体环境基类。"""
    
    ################################################################################

    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=1,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 240,
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.KIN,
                 laser_record=False,
                 act: ActionType=ActionType.VEL,
                 output_folder='results',
                 enable_trajectory_recording: bool = False,
                 trajectory_manager=None
                 ):
        """通用单智能体和多智能体强化学习环境的初始化。

        属性 `vision_attributes` 和 `dynamics_attributes` 会根据 `obs` 和 `act` 的选择自动设置；
        `obstacles` 默认为 True，视觉任务下会用地标覆盖障碍物；
        `user_debug_gui` 默认为 False 以提升性能。

        参数说明
        ----------
        drone_model : DroneModel, optional
            期望的无人机类型（详见 `assets` 文件夹下的 .urdf 文件）。
        num_drones : int, optional
            群体中无人机数量。
        neighbourhood_radius : float, optional
            用于计算邻接矩阵的半径（米）。
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3) 形状的数组，初始位置。
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3) 形状的数组，初始欧拉角（弧度）。
        physics : Physics, optional
            物理仿真类型。
        pyb_freq : int, optional
            PyBullet 步进频率。
        ctrl_freq : int, optional
            环境步进频率。
        gui : bool, optional
            是否启用 PyBullet GUI。
        record : bool, optional
            是否保存仿真视频。
        obs : ObservationType, optional
            观测空间类型（运动学信息或视觉）。
        act : ActionType, optional
            动作空间类型（1维/3维，RPM、推力与力矩、航点或带PID的速度等）。

        """
        #### 创建最近0.5秒动作的缓冲区 ########
        self.ACTION_BUFFER_SIZE = int(ctrl_freq//2)
        self.action_buffer = deque(maxlen=self.ACTION_BUFFER_SIZE)
        ####
        vision_attributes = True if obs == ObservationType.RGB else False
        self.OBS_TYPE = obs
        self.ACT_TYPE = act
        #### 创建集成控制器 #########################
        if act in [ActionType.PID, ActionType.VEL, ActionType.ONE_D_PID]:
            os.environ['KMP_DUPLICATE_LIB_OK']='True'
            if drone_model in [DroneModel.CF2X, DroneModel.CF2P]:
                self.ctrl = [DSLPIDControl(drone_model=DroneModel.CF2X) for i in range(num_drones)]
            else:
                print("[ERROR] in BaseRLAviary.__init()__, no controller is available for the specified drone_model")
        elif act == ActionType.VEL_CTRL:
            # 🔥 新增：为VEL_CTRL创建VelControl控制器
            os.environ['KMP_DUPLICATE_LIB_OK']='True'
            if drone_model in [DroneModel.CF2X, DroneModel.CF2P]:
                from src.controllers.VelControl import VelControl
                self.ctrl = [VelControl(drone_model=drone_model) for i in range(num_drones)]
                # 基于URDF设置速度限制 (30 km/h = 8.33 m/s)
                self.SPEED_LIMIT = 8.33
                print(f"✅ [BaseRLAviary] VelControl控制器已创建，数量: {num_drones}")
            else:
                print("[ERROR] in BaseRLAviary.__init()__, VelControl only supports CF2X and CF2P")
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record, 
                         obstacles=True, # Add obstacles for RGB observations and/or FlyThruGate
                         user_debug_gui=False, # Remove of RPM sliders from all single agent learning aviaries
                         vision_attributes=False, # 确保不启用视觉，除非明确需要
                         laser_attributes=True, # 启用激光雷达属性
                         laser_record=laser_record,
                         output_folder=output_folder
                         )
        
        # 轨迹记录系统初始化
        self.enable_trajectory_recording = enable_trajectory_recording
        self.trajectory_manager = trajectory_manager
        self.last_model_action = None  # 存储最后的model_action
        
        #### 设置最大目标速度限制 ###############
        if act == ActionType.VEL:
            self.SPEED_LIMIT = 3.0  # 增加到3 m/s，原来是0.25 m/s


    ################################################################################

    def _actionSpace(self):
        """返回环境的动作空间。

        返回值
        -------
        spaces.Box
            大小为 NUM_DRONES x 4、3 或 1 的 Box，取决于动作类型。

        """
        if self.ACT_TYPE in [ActionType.RPM, ActionType.VEL]:
            size = 4
            # 传统动作范围 [-1, 1]
            act_lower_bound = np.array([-1*np.ones(size) for i in range(self.NUM_DRONES)])
            act_upper_bound = np.array([+1*np.ones(size) for i in range(self.NUM_DRONES)])
        elif self.ACT_TYPE == ActionType.VEL_CTRL:
            size = 4
            # 🔥 JSON规范：基于物理限制的动作范围 [vx, vy, vz, yaw_rate]
            # 基于URDF max_speed_kmh=30km/h=8.33m/s, 偏航率限制为4.0rad/s
            act_lower_bound = np.array([[-8.33, -8.33, -8.33, -4.0] for i in range(self.NUM_DRONES)])
            act_upper_bound = np.array([[8.33, 8.33, 8.33, 4.0] for i in range(self.NUM_DRONES)])
        elif self.ACT_TYPE==ActionType.PID:
            size = 3
            act_lower_bound = np.array([-1*np.ones(size) for i in range(self.NUM_DRONES)])
            act_upper_bound = np.array([+1*np.ones(size) for i in range(self.NUM_DRONES)])
        elif self.ACT_TYPE in [ActionType.ONE_D_RPM, ActionType.ONE_D_PID]:
            size = 1
            act_lower_bound = np.array([-1*np.ones(size) for i in range(self.NUM_DRONES)])
            act_upper_bound = np.array([+1*np.ones(size) for i in range(self.NUM_DRONES)])
        else:
            print("[ERROR] in BaseRLAviary._actionSpace()")
            print(f"[ERROR] Unknown ACT_TYPE: {self.ACT_TYPE}")
            exit()
        #
        for i in range(self.ACTION_BUFFER_SIZE):
            self.action_buffer.append(np.zeros((self.NUM_DRONES,size)))
        #
        return spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)

    ################################################################################
    
    def _addObstacles(self):
        """向环境中添加障碍物。

        仅当观测类型为 RGB 时，添加4个地标。
        重写 BaseAviary 的方法。

        """
        if self.OBS_TYPE == ObservationType.RGB:
            p.loadURDF("block.urdf",
                       [1, 0, .1],
                       p.getQuaternionFromEuler([0, 0, 0]),
                       physicsClientId=self.CLIENT
                       )
            p.loadURDF("cube_small.urdf",
                       [0, 1, .1],
                       p.getQuaternionFromEuler([0, 0, 0]),
                       physicsClientId=self.CLIENT
                       )
            p.loadURDF("duck_vhacd.urdf",
                       [-1, 0, .1],
                       p.getQuaternionFromEuler([0, 0, 0]),
                       physicsClientId=self.CLIENT
                       )
            p.loadURDF("teddy_vhacd.urdf",
                       [0, -1, .1],
                       p.getQuaternionFromEuler([0, 0, 0]),
                       physicsClientId=self.CLIENT
                       )
        else:
            pass
        
    
    def _preprocessAction(self,
                          action
                          ):
        """将 step() 输入的动作预处理为电机 RPM。

        参数 `action` 会根据不同动作类型进行不同处理：
        对于第 n 架无人机，`action[n]` 可以为长度 1、3 或 4，分别表示 RPM、期望推力与力矩，或用 PID 控制的下一个目标位置。

        参数 `action` 的处理方式随动作类型不同：可以为长度 1、3 或 4，分别代表 RPM、期望推力与力矩、PID 控制的目标位置、期望速度向量等。

        参数说明
        ----------
        action : ndarray
            每架无人机的输入动作，将被转换为 RPM。

        返回值
        -------
        ndarray
            (NUM_DRONES, 4) 形状的数组，包含每架无人机4个电机的裁剪后 RPM。

        """
        self.action_buffer.append(action)
        rpm = np.zeros((self.NUM_DRONES,4))
        
        # 存储model_action用于轨迹记录
        if self.enable_trajectory_recording and self.NUM_DRONES == 1:
            self.last_model_action = action[0].copy() if len(action) > 0 else None
        
        for k in range(action.shape[0]):
            target = action[k, :]
            if self.ACT_TYPE == ActionType.RPM:
                rpm[k,:] = np.array(self.HOVER_RPM * (1+0.05*target))
            elif self.ACT_TYPE == ActionType.PID:
                state = self._getDroneStateVector(k)
                next_pos = self._calculateNextStep(
                    current_position=state[0:3],
                    destination=target,
                    step_size=1,
                    )
                rpm_k, _, _ = self.ctrl[k].computeControl(control_timestep=self.CTRL_TIMESTEP,
                                                        cur_pos=state[0:3],
                                                        cur_quat=state[3:7],
                                                        cur_vel=state[10:13],
                                                        cur_ang_vel=state[13:16],
                                                        target_pos=next_pos
                                                        )
                rpm[k,:] = rpm_k
            elif self.ACT_TYPE == ActionType.VEL:
                state = self._getDroneStateVector(k)
                if np.linalg.norm(target[0:3]) != 0:
                    v_unit_vector = target[0:3] / np.linalg.norm(target[0:3])
                else:
                    v_unit_vector = np.zeros(3)
                temp, _, _ = self.ctrl[k].computeControl(control_timestep=self.CTRL_TIMESTEP,
                                                        cur_pos=state[0:3],
                                                        cur_quat=state[3:7],
                                                        cur_vel=state[10:13],
                                                        cur_ang_vel=state[13:16],
                                                        target_pos=state[0:3], # same as the current position
                                                        target_rpy=np.array([0,0,state[9]]), # keep current yaw
                                                        target_vel=self.SPEED_LIMIT * np.abs(target[3]) * v_unit_vector # target the desired velocity vector
                                                        )
                rpm[k,:] = temp
            elif self.ACT_TYPE == ActionType.VEL_CTRL:
                # 🔥 JSON规范：使用VelControl控制器处理速度输入
                state = self._getDroneStateVector(k)
                
                # 处理3维输入，自动补全yaw_rate=0
                if len(target) == 3:
                    target = np.append(target, 0.0)
                elif len(target) > 4:
                    target = target[:4]
                
                # 检查是否为增量控制模式
                controller = self.ctrl[k]
                if hasattr(controller, 'increment_mode') and controller.increment_mode:
                    # 增量控制路径：target被解释为增量 [dvx, dvy, dvz, dyaw]
                    controller.set_increment_action(target)
                    
                # 调用VelControl的computeControl方法（支持增量和标准模式）
                rpm_k = controller.computeControl(
                    control_timestep=self.CTRL_TIMESTEP,
                    cur_pos=state[0:3],         # 当前位置
                    cur_quat=state[3:7],        # 当前四元数
                    cur_vel=state[10:13],       # 当前速度
                    cur_ang_vel=state[13:16],   # 当前角速度
                    target_pos=state[0:3],      # 位置保持当前（速度控制不直接使用）
                    target_rpy=np.array([0, 0, state[9]]),  # 保持当前yaw
                    target_vel=target[:3] if not (hasattr(controller, 'increment_mode') and controller.increment_mode) else np.zeros(3),      # 标准模式使用目标速度
                    target_rpy_rates=np.array([0, 0, target[3]]) if not (hasattr(controller, 'increment_mode') and controller.increment_mode) else np.zeros(3)  # 标准模式使用yaw_rate
                )
                rpm[k,:] = rpm_k
            elif self.ACT_TYPE == ActionType.ONE_D_RPM:
                rpm[k,:] = np.repeat(self.HOVER_RPM * (1+0.05*target), 4)
            elif self.ACT_TYPE == ActionType.ONE_D_PID:
                state = self._getDroneStateVector(k)
                res, _, _ = self.ctrl[k].computeControl(control_timestep=self.CTRL_TIMESTEP,
                                                        cur_pos=state[0:3],
                                                        cur_quat=state[3:7],
                                                        cur_vel=state[10:13],
                                                        cur_ang_vel=state[13:16],
                                                        target_pos=state[0:3]+0.1*np.array([0,0,target[0]])
                                                        )
                rpm[k,:] = res
            else:
                print("[ERROR] in BaseRLAviary._preprocessAction()")
                exit()
        
        # 存储最终的RPM值用于轨迹记录
        self.last_clipped_action = rpm.copy()
        
        return rpm

    ################################################################################

    def _observationSpace(self):
        """返回环境的观测空间。

        返回值
        -------
        ndarray
            形状为 (NUM_DRONES,H,W,4)、(NUM_DRONES,numRays,3) 或 (NUM_DRONES,12) 的 Box，取决于观测类型。

        """
        if self.OBS_TYPE == ObservationType.RGB:
            return spaces.Box(low=0,
                              high=255,
                              shape=(self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0], 4), dtype=np.uint8)
        elif self.OBS_TYPE == ObservationType.LASER:
            ############################################################
            #### 🔥 新增：激光雷达观测空间
            return spaces.Box(low=self.LASER_MASK,
                              high=self.LASER_RANGE,
                              shape=(self.NUM_DRONES, self.numRays, 3), 
                              dtype=np.float32)
            ############################################################
        elif self.OBS_TYPE == ObservationType.KIN:
            ############################################################
            #### 12维观测空间
            #### 观测向量 ### X        Y        Z       Q1   Q2   Q3   Q4   R       P       Y       VX       VY       VZ       WX       WY       WZ
            lo = -np.inf
            hi = np.inf
            obs_lower_bound = np.array([[lo,lo,0, lo,lo,lo,lo,lo,lo,lo,lo,lo] for i in range(self.NUM_DRONES)])
            obs_upper_bound = np.array([[hi,hi,hi,hi,hi,hi,hi,hi,hi,hi,hi,hi] for i in range(self.NUM_DRONES)])
            #### 将动作缓冲区添加到观测空间 ################
            act_lo = -1
            act_hi = +1
            for i in range(self.ACTION_BUFFER_SIZE):
                if self.ACT_TYPE in [ActionType.RPM, ActionType.VEL, ActionType.VEL_CTRL]:
                    obs_lower_bound = np.hstack([obs_lower_bound, np.array([[act_lo,act_lo,act_lo,act_lo] for i in range(self.NUM_DRONES)])])
                    obs_upper_bound = np.hstack([obs_upper_bound, np.array([[act_hi,act_hi,act_hi,act_hi] for i in range(self.NUM_DRONES)])])
                elif self.ACT_TYPE==ActionType.PID:
                    obs_lower_bound = np.hstack([obs_lower_bound, np.array([[act_lo,act_lo,act_lo] for i in range(self.NUM_DRONES)])])
                    obs_upper_bound = np.hstack([obs_upper_bound, np.array([[act_hi,act_hi,act_hi] for i in range(self.NUM_DRONES)])])
                elif self.ACT_TYPE in [ActionType.ONE_D_RPM, ActionType.ONE_D_PID]:
                    obs_lower_bound = np.hstack([obs_lower_bound, np.array([[act_lo] for i in range(self.NUM_DRONES)])])
                    obs_upper_bound = np.hstack([obs_upper_bound, np.array([[act_hi] for i in range(self.NUM_DRONES)])])
            return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)
            ############################################################
        else:
            print("[ERROR] in BaseRLAviary._observationSpace()")
            print(f"[ERROR] Unknown OBS_TYPE: {self.OBS_TYPE}")
            exit()
    
    ################################################################################

    def _computeObs(self):
        """返回环境的当前观测。

        返回值
        -------
        ndarray
            形状为 (NUM_DRONES,H,W,4)、(NUM_DRONES,numRays,3) 或 (NUM_DRONES,12) 的 Box，取决于观测类型。

        """
        if self.OBS_TYPE == ObservationType.RGB:
            if self.step_counter%self.IMG_CAPTURE_FREQ == 0:
                for i in range(self.NUM_DRONES):
                    self.rgb[i], self.dep[i], self.seg[i] = self._getDroneImages(i,
                                                                                 segmentation=False
                                                                                 )
                    #### Printing observation to PNG frames example ############
                    if self.RECORD:
                        self._exportImage(img_type=ImageType.RGB,
                                          img_input=self.rgb[i],
                                          path=self.ONBOARD_IMG_PATH+"drone_"+str(i),
                                          frame_num=int(self.step_counter/self.IMG_CAPTURE_FREQ)
                                          )
            return np.array([self.rgb[i] for i in range(self.NUM_DRONES)]).astype('float32')
        elif self.OBS_TYPE == ObservationType.LASER:
            ############################################################
            #### 🔥 新增：激光雷达观测处理
            for i in range(self.NUM_DRONES):
                if self.step_counter % self.LASER_CAPTURE_FREQ == 0:
                    self.pointcloud[i] = self._getDroneRays(i)
                    if self.LASER_RECORD:
                        self._exportPointCloud(pointcloud_input=self.pointcloud[i],
                                             path=self.ONBOARD_LASER_PATH+"drone_"+str(i),
                                             frame_num=int(self.step_counter/self.LASER_CAPTURE_FREQ))
            return np.array([self.pointcloud[i] for i in range(self.NUM_DRONES)]).astype('float32')
            ############################################################
        elif self.OBS_TYPE == ObservationType.KIN:
            ############################################################
            #### 12维观测空间
            obs_12 = np.zeros((self.NUM_DRONES,12))
            for i in range(self.NUM_DRONES):
                #obs = self._clipAndNormalizeState(self._getDroneStateVector(i)) # 可选：归一化状态
                obs = self._getDroneStateVector(i)
                obs_12[i, :] = np.hstack([obs[0:3], obs[7:10], obs[10:13], obs[13:16]]).reshape(12,)
            ret = np.array([obs_12[i, :] for i in range(self.NUM_DRONES)]).astype('float32')
            #### 将动作缓冲区添加到观测 #######################
            for i in range(self.ACTION_BUFFER_SIZE):
                ret = np.hstack([ret, np.array([self.action_buffer[i][j, :] for j in range(self.NUM_DRONES)])])
            return ret
            ############################################################
        else:
            print("[ERROR] in BaseRLAviary._computeObs()")
            print(f"[ERROR] Unknown OBS_TYPE: {self.OBS_TYPE}")
            exit()

    ################################################################################
    
    def get_trajectory_step_data(self, drone_id: int = 0):
        """
        获取轨迹记录所需的五元组数据
        
        Args:
            drone_id: 无人机ID（默认0）
            
        Returns:
            dict: 包含轨迹数据的字典 或 None（如果未启用记录）
        """
        if not self.enable_trajectory_recording or drone_id >= self.NUM_DRONES:
            return None
            
        # 获取当前状态
        state = self._getDroneStateVector(drone_id)
        current_position = state[0:3]  # [x, y, z]
        current_v = state[10:13]       # [vx, vy, vz] - 当前速度
        
        # 获取target_action（从控制器）
        target_action = None
        if hasattr(self.ctrl[drone_id], 'get_target_action'):
            target_action = self.ctrl[drone_id].get_target_action()
        
        # model_action已在_preprocessAction中存储
        model_action = self.last_model_action
        
        # rpm_action是电机输出（最后一次处理的RPM）
        rpm_action = None
        if hasattr(self, 'last_clipped_action') and self.last_clipped_action is not None:
            if len(self.last_clipped_action) > drone_id:
                rpm_action = self.last_clipped_action[drone_id]
        
        return {
            "current_position": current_position,
            "current_v": current_v,
            "target_action": target_action,
            "model_action": model_action,
            "rpm_action": rpm_action
        }

    ################################################################################
            
    
