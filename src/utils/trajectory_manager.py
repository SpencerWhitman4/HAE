"""
轨迹管理器 - 专为分层强化学习控制器设计
严格按照JSON规范实现，支持训练和评估模式分离
记录五个关键数据元组：current_position, current_velocity, target_velocity, model_action, rpm_action
附加信息：step数、exploration_rate
数据精度：保存到小数点后4位 (0.0001)
"""
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple

class TrajectoryManager:
    """轨迹管理器，记录精简的控制链路数据（5元组+基本信息）"""
    
    def __init__(self, session_dir: Path, mode: str):
        """
        初始化轨迹管理器
        
        Args:
            session_dir: 会话目录路径
            mode: 'train' 或 'eval'
        """
        if mode not in ['train', 'eval']:
            raise ValueError("mode必须是'train'或'eval'")
        
        # 确保session_dir是Path对象
        if isinstance(session_dir, str):
            session_dir = Path(session_dir)
            
        self.session_dir = session_dir
        self.mode = mode
        self.trajectory_dir = session_dir / "Trajectory" / mode
        self.trajectory_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化logger
        self.logger = logging.getLogger(f"{__name__}.{mode}")
        
        # 当前回合状态
        self.current_episode_num = None
        self.current_trajectory_buffer = []
        self.current_file_path = None
    
    def start_new_episode(self, episode_num: int) -> None:
        """开始新回合的轨迹记录"""
        self.current_episode_num = episode_num
        self.current_trajectory_buffer = []
        
        # 根据mode创建不同命名的文件
        if self.mode == 'train':
            filename = f"episode_{episode_num:06d}.json"
        else:  # eval
            filename = f"episode_eval_{episode_num:06d}.json"
        
        self.current_file_path = self.trajectory_dir / filename
        
        # 初始化JSON文件结构
        initial_data = {
            "episode_info": {
                "episode_num": episode_num,
                "mode": self.mode,
                "status": "in_progress"
            },
            "trajectory": []
        }
        
        try:
            with open(self.current_file_path, 'w', encoding='utf-8') as f:
                json.dump(initial_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.warning(f"创建轨迹文件失败: {e}")
    
    def log_step(self, step_data: Dict[str, Any]) -> None:
        """记录单步轨迹数据 - 五个数据元组（含current_v）"""
        # 处理NumPy数组序列化并设置精度
        serialized_data = self._serialize_step_data(step_data)
        
        # 添加到缓冲区
        self.current_trajectory_buffer.append(serialized_data)
        
        # 追加到JSON文件
        self._append_to_json_file(serialized_data)
    
    def finalize_episode(self, termination_reason: str, 
                        final_exploration_rate: float, 
                        total_reward: float) -> None:
        """完成当前回合的轨迹记录"""
        if self.current_file_path is None:
            return
        
        try:
            # 读取现有文件
            with open(self.current_file_path, 'r', encoding='utf-8') as f:
                episode_data = json.load(f)
            
            # 添加终止元数据
            episode_data["episode_info"].update({
                "status": "completed",
                "termination_reason": termination_reason,
                "final_exploration_rate": float(final_exploration_rate),
                "total_reward": float(total_reward),
                "total_steps": len(self.current_trajectory_buffer)
            })
            
            # 原子性写入完整数据
            temp_path = self.current_file_path.with_suffix('.tmp')
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(episode_data, f, indent=2, ensure_ascii=False)
            
            # 原子性重命名
            temp_path.replace(self.current_file_path)
            
        except (json.JSONDecodeError, FileNotFoundError) as e:
            self.logger.warning(f"完成轨迹记录时JSON错误: {e}")
        except Exception as e:
            self.logger.warning(f"完成轨迹记录失败: {e}")
        finally:
            # 清理状态
            self.current_trajectory_buffer = []
            self.current_file_path = None
    
    def _serialize_step_data(self, step_data: Dict[str, Any]) -> Dict[str, Any]:
        """序列化步骤数据，处理NumPy数组并设置精度为6位小数"""
        def round_value(val):
            """递归处理任何类型的数值，确保精度为6位小数"""
            if isinstance(val, np.ndarray):
                return np.round(val, 6).tolist()
            elif isinstance(val, (np.integer, np.floating, int, float)):
                # 特别处理exploration_rate，使用更高精度
                return round(float(val), 6)
            elif isinstance(val, (bool, np.bool_)):
                # 处理布尔值 - 确保是Python布尔值而不是numpy布尔值
                return bool(val)
            elif isinstance(val, list):
                return [round_value(item) for item in val]
            elif isinstance(val, dict):
                return {k: round_value(v) for k, v in val.items()}
            elif val is None:
                return None
            else:
                # 其他类型转换为字符串
                return str(val)
        
        # 对整个字典递归应用精度控制
        return {key: round_value(value) for key, value in step_data.items()}
    
    def _append_to_json_file(self, step_data: Dict[str, Any]) -> None:
        """追加数据到JSON文件"""
        if self.current_file_path is None:
            return
        
        try:
            # 读取现有数据
            if self.current_file_path.exists():
                with open(self.current_file_path, 'r', encoding='utf-8') as f:
                    episode_data = json.load(f)
            else:
                # 如果文件不存在，创建初始结构
                episode_data = {
                    "episode_info": {
                        "episode_num": self.current_episode_num,
                        "mode": self.mode,
                        "status": "in_progress"
                    },
                    "trajectory": []
                }
            
            # 添加新步骤
            episode_data["trajectory"].append(step_data)
            
            # 原子性写入：先写临时文件，再重命名
            temp_path = self.current_file_path.with_suffix('.tmp')
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(episode_data, f, indent=2, ensure_ascii=False)
            
            # 原子性重命名
            temp_path.replace(self.current_file_path)
            
        except (json.JSONDecodeError, FileNotFoundError) as e:
            # 如果JSON损坏或文件不存在，重新创建
            self.logger.warning(f"JSON文件损坏，重新创建: {e}")
            episode_data = {
                "episode_info": {
                    "episode_num": self.current_episode_num,
                    "mode": self.mode,
                    "status": "in_progress"
                },
                "trajectory": [step_data]
            }
            with open(self.current_file_path, 'w', encoding='utf-8') as f:
                json.dump(episode_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            # 其他错误，记录但不中断训练
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"轨迹写入失败: {e}")
