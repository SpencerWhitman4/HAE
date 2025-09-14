"""通用工具函数。
"""
import time
import argparse
import numpy as np
from pathlib import Path
from scipy.optimize import nnls

################################################################################

def sync(i, start_time, timestep):
    """使仿真步进与真实时间同步。

    函数 `sync` 会调用 time.sleep()，用于让 for 循环的执行速度不超过期望的步长。

    参数说明
    ----------
    i : int
        当前仿真迭代次数。
    start_time : timestamp
        仿真开始的时间戳。
    timestep : float
        希望渲染的真实时间步长。

    """
    if timestep > .04 or i%(int(1/(24*timestep))) == 0:
        elapsed = time.time() - start_time
        if elapsed < (i*timestep):
            time.sleep(timestep*i - elapsed)

################################################################################

def str2bool(val):
    """将字符串转换为布尔值。

    参数说明
    ----------
    val : str | bool
        输入值（可能为字符串），用于解释为布尔类型。

    返回值
    -------
    bool
        将 `val` 解释为 True 或 False。

    """
    if isinstance(val, bool):
        return val
    elif val.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif val.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError("[错误] 在 str2bool() 中，期望一个布尔值")


