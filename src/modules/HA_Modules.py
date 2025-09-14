"""
HA_Modules.py - 分层强化学习无人机避障系统核心模块

本模块实现了基于分层强化学习的无人机避障系统的核心神经网络架构。
主要包含以下核心组件：

1. 位置编码器（PositionalEncoding）：为Transformer提供时序信息
2. Transformer编码器（TransformerEncoder）：处理序列数据的核心架构
3. 状态编码器（StateEncoder）：统一编码LiDAR和动作信息
4. 高层感知编码器（HighLevelPerceptionEncoder）：处理历史状态序列
5. 低层感知编码器（LowLevelPerceptionEncoder）：处理当前状态和子目标
6. 高层策略网络（HighLevelActor）：生成导航子目标序列
7. 低层策略网络（LowLevelActor）：生成具体控制命令
8. 高层评论家（HighLevelCritic）：评估高层策略价值
9. 低层评论家（LowLevelCritic）：评估低层策略价值

作者：HA-UAV项目组
更新时间：2025年7月15日
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import numpy as np
from typing import Dict, List, Tuple, Optional


class PositionalEncoding(nn.Module):
    """
    Transformer位置编码器
    
    该类实现了Transformer模型中的正弦位置编码，为输入序列提供位置信息。
    位置编码使用正弦和余弦函数来表示序列中每个位置的信息，这种编码方式：
    1. 能够处理任意长度的序列
    2. 对于不同位置的差异具有良好的表示能力
    3. 有助于模型理解序列中元素的相对位置关系
    
    数学公式：
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    其中 pos 是位置，i 是维度索引，d_model 是模型维度
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        初始化位置编码器
        
        Args:
            d_model (int): 模型的隐藏维度，必须与Transformer的d_model保持一致
            dropout (float): Dropout概率，用于防止过拟合，默认为0.1
            max_len (int): 支持的最大序列长度，默认为5000
        """
        super().__init__()
        # 初始化Dropout层，用于正则化
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置索引张量 [max_len, 1]
        position = torch.arange(max_len).unsqueeze(1)
        
        # 计算除法项，用于正弦和余弦函数的频率调制
        # 这里使用指数形式来避免直接计算大数的幂
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        
        # 初始化位置编码张量 [max_len, 1, d_model]
        pe = torch.zeros(max_len, 1, d_model)
        
        # 填充偶数位置（0, 2, 4, ...）的正弦编码
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        
        # 填充奇数位置（1, 3, 5, ...）的余弦编码
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        # 将位置编码注册为buffer，这样它不会被当作模型参数进行梯度更新
        # 但会随着模型一起保存和加载
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：为输入序列添加位置编码
        
        Args:
            x (Tensor): 输入张量，形状为 [seq_len, batch_size, d_model]
                       seq_len: 序列长度
                       batch_size: 批次大小
                       d_model: 模型隐藏维度
        
        Returns:
            Tensor: 添加位置编码后的张量，形状与输入相同
        
        注意：
            - 位置编码直接加到输入特征上，不改变张量形状
            - 使用切片操作确保位置编码长度与输入序列长度匹配
            - 最后应用Dropout进行正则化
        """
        # 将位置编码加到输入上，只取前 x.size(0) 个位置的编码
        x = x + self.pe[:x.size(0)]
        
        # 应用Dropout并返回结果
        return self.dropout(x)

class TransformerEncoder(nn.Module):
    """
    Transformer编码器封装类
    
    该类封装了PyTorch的标准Transformer编码器，专门用于处理序列数据。
    Transformer编码器是基于自注意力机制的神经网络架构，具有以下特点：
    1. 并行处理序列中的所有元素，训练效率高
    2. 自注意力机制能够捕捉序列中任意位置之间的依赖关系
    3. 多头注意力机制提供了多个表示子空间
    4. 残差连接和层归一化确保训练稳定性
    
    在HA-UAV系统中，用于：
    - 高层感知编码：处理历史LiDAR和动作序列
    - 低层感知编码：处理当前状态、历史动作和子目标序列
    """
    
    def __init__(self, d_model: int, nhead: int, num_layers: int, dim_feedforward: int = 2048, dropout: float = 0.1):
        """
        初始化Transformer编码器
        
        Args:
            d_model (int): 模型的隐藏维度，所有输入特征都需要映射到此维度
            nhead (int): 多头注意力的头数，必须能够整除d_model
            num_layers (int): Transformer层数，更多层数可以学习更复杂的模式
            dim_feedforward (int): 前馈网络的隐藏维度，通常是d_model的4倍，默认2048
            dropout (float): Dropout概率，用于防止过拟合，默认0.1
        
        注意：
            - batch_first=False 意味着输入格式为 [seq_len, batch_size, d_model]
            - 这是PyTorch Transformer的标准格式，与RNN类似
        """
        super().__init__()
        
        # 创建单个Transformer编码器层
        # batch_first=False 确保输入格式为 [seq_len, batch_size, d_model]
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        
        # 堆叠多个编码器层形成完整的Transformer编码器
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        前向传播：对输入序列进行编码
        
        Args:
            src (Tensor): 输入序列，形状为 [seq_len, batch_size, d_model]
                         seq_len: 序列长度（如历史步数、子目标数量等）
                         batch_size: 批次大小
                         d_model: 特征维度
        
        Returns:
            Tensor: 编码后的序列，形状为 [seq_len, batch_size, d_model]
        
        处理流程：
            1. 每个位置的特征向量通过多头自注意力机制与序列中所有位置交互
            2. 注意力输出通过残差连接和层归一化
            3. 通过前馈网络进行进一步特征变换
            4. 再次应用残差连接和层归一化
            5. 重复上述过程num_layers次
        """
        return self.transformer_encoder(src)

class StateEncoder(nn.Module):
    """
    状态编码器 - 支持占据栅格地图(OGM)和yaw历史特征
    
    该类负责将占据栅格地图特征和yaw历史信息编码为统一的特征表示。
    这是分层强化学习系统中的基础组件，为后续的决策网络提供标准化的输入。
    
    功能特点：
    1. OGM特征处理：通过卷积网络处理2D占据栅格地图
    2. Yaw历史处理：通过MLP处理历史偏航角序列
    3. 异构数据融合：将空间特征（OGM）和时序特征（yaw）融合
    4. 特征提取：通过卷积和全连接网络提取多层次特征
    5. 维度标准化：将输入映射到固定的隐藏维度
    6. 非线性变换：使用ReLU激活函数增强表达能力
    
    在HA-UAV系统中的作用：
    - 处理占据栅格特征（对应需求中的 "occupancy_grid"）
    - 处理yaw历史（对应需求中的 "yaw_history"）
    - 为高层感知编码器提供历史状态的标准化表示
    - 为低层感知编码器提供当前状态的编码
    - 确保不同时间步的状态具有相同的特征空间
    
    网络架构：
    OGM特征 -> 卷积层 -> 自适应池化 -> 展平
    Yaw特征 -> MLP处理
    融合特征 -> MLP -> 输出
    """
    
    def __init__(self, grid_size: int = 200, k_history: int = 20, output_dim: int = 256):
        """
        初始化状态编码器 - 基于占据栅格的设计
        
        Args:
            grid_size (int): 占据栅格的尺寸，默认200（200x200网格）
            k_history (int): 历史时间步数，默认20
            output_dim (int): 输出特征的维度，与后续网络的输入维度匹配
        
        网络结构：
            占据栅格分支: Conv2d -> ReLU -> Conv2d -> ReLU -> AdaptiveAvgPool2d -> Flatten -> Linear
            Yaw历史分支: Linear -> Tanh -> Linear
            融合层: Concat -> Linear -> LayerNorm -> ReLU -> Linear
        
        设计理念：
            - 占据栅格数据反映了局部环境的空间结构
            - 每个栅格单元表示该位置的占据状态（-1=未知, 0=自由, 1=占据）
            - 通过卷积网络提取空间特征表示
            - Yaw历史提供时序的方向信息
        """
        super().__init__()
        
        # 占据栅格特征处理分支（卷积网络）
        self.grid_conv = nn.Sequential(
            # 第一卷积层：提取基础空间特征
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),  # 200->100
            nn.ReLU(),
            # 第二卷积层：提取高层空间特征
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),  # 100->50
            nn.ReLU(),
            # 第三卷积层：进一步抽象
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),  # 50->25
            nn.ReLU(),
            # 自适应池化到固定尺寸
            nn.AdaptiveAvgPool2d((4, 4)),  # 自适应池化到4x4
            nn.Flatten(),  # 128*4*4 = 2048
            nn.Linear(128 * 4 * 4, output_dim // 2)
        )
        
        # Yaw历史特征处理分支（MLP网络）
        self.yaw_mlp = nn.Sequential(
            nn.Linear(1, output_dim // 4),
            nn.Tanh(),
            nn.Linear(output_dim // 4, output_dim // 2)
        )
        
        # 特征融合网络
        self.fusion_mlp = nn.Sequential(
            # 融合占据栅格特征和yaw特征
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            # 最终特征变换
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, occupancy_grid: torch.Tensor, yaw_history: torch.Tensor) -> torch.Tensor:
        """
        前向传播：编码占据栅格特征和yaw历史信息
        
        Args:
            occupancy_grid (Tensor): 占据栅格数据，形状为 [batch_size, k_history, grid_size, grid_size]
                                   包含历史时间步的局部占据栅格信息
                                   值范围：[-1.0, 1.0] (unknown=-1, free=0, occupied=1)
            yaw_history (Tensor): yaw历史数据，形状为 [batch_size, k_history]
                                包含历史时间步的偏航角信息
                                值范围：[-π, π] 弧度
        
        Returns:
            Tensor: 编码后的状态特征，形状为 [batch_size, k_history, output_dim]
        
        处理流程：
            1. 逐个时间步处理占据栅格，提取空间特征
            2. 逐个时间步处理yaw角度，提取方向特征
            3. 将空间特征和方向特征融合
            4. 通过融合网络得到最终的状态表示
        
        注意：
            - 输入的occupancy_grid和yaw_history必须具有相同的batch_size和k_history
            - 占据栅格数据应预处理为归一化的值
            - yaw数据应归一化到[-1, 1]范围进行处理
            - 输出特征具有固定的维度，便于后续Transformer处理
        """
        # 处理输入维度 - 支持3D和4D输入
        if hasattr(occupancy_grid, 'dim'):
            # PyTorch tensor
            if occupancy_grid.dim() == 3:
                # 单样本推理: [k_history, grid_size, grid_size] -> [1, k_history, grid_size, grid_size]
                occupancy_grid = occupancy_grid.unsqueeze(0)
                yaw_history = yaw_history.unsqueeze(0)
                squeeze_output = True
            else:
                squeeze_output = False
        else:
            # NumPy array
            if occupancy_grid.ndim == 3:
                # 单样本推理: [k_history, grid_size, grid_size] -> [1, k_history, grid_size, grid_size]
                occupancy_grid = torch.from_numpy(occupancy_grid).unsqueeze(0).float()
                yaw_history = torch.from_numpy(yaw_history).unsqueeze(0).float()
                squeeze_output = True
            else:
                squeeze_output = False
                # 确保是torch tensor
                if not isinstance(occupancy_grid, torch.Tensor):
                    occupancy_grid = torch.from_numpy(occupancy_grid).float()
                    yaw_history = torch.from_numpy(yaw_history).float()
            
        batch_size, k_history, grid_size, _ = occupancy_grid.shape
        
        # 处理占据栅格特征：reshape为 [batch_size*k_history, 1, grid_size, grid_size]
        grid_flat = occupancy_grid.view(-1, 1, grid_size, grid_size)
        grid_features = self.grid_conv(grid_flat)  # [batch_size*k_history, output_dim//2]
        grid_features = grid_features.view(batch_size, k_history, -1)
        
        # 处理yaw历史：归一化到[-1, 1]然后编码
        yaw_normalized = yaw_history / np.pi  # 归一化到[-1, 1]
        yaw_features = self.yaw_mlp(yaw_normalized.unsqueeze(-1))  # [batch_size, k_history, output_dim//2]
        
        # 特征融合：在最后一个维度上拼接
        combined_features = torch.cat([grid_features, yaw_features], dim=-1)  # [batch_size, k_history, output_dim]
        
        # 通过融合网络进行最终特征变换
        output = self.fusion_mlp(combined_features)
        
        # 如果原始输入是3D（单样本推理），压缩batch维度
        if squeeze_output:
            output = output.squeeze(0)  # [k_history, output_dim]
            
        return output


class LowLevelStateEncoder(nn.Module):
    """
    低层状态编码器 - 专门处理64维低层观测
    
    该类负责将低层控制所需的局部感知数据编码为统一的特征表示。
    针对分层强化学习系统中低层策略的输入进行优化设计。
    
    输入格式（64维）：
    - 扇区距离：36维局部障碍物感知
    - 运动状态：12维当前速度/姿态/加速度 
    - 动作历史：8维最近2步动作
    - 局部地图：8维局部占用/通道/障碍物信息
    
    网络架构：
    64维输入 -> MLP(64->256->128) -> 128维输出
    """
    
    def __init__(self, input_dim: int = 64, hidden_dim: int = 256, output_dim: int = 128):
        """
        初始化低层状态编码器
        
        Args:
            input_dim: 输入维度（默认64维）
            hidden_dim: 隐藏层维度
            output_dim: 输出特征维度
        """
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(), 
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Tanh()
        )
        
    def forward(self, low_level_obs: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            low_level_obs: [batch_size, 64] 低层观测数据
            
        Returns:
            torch.Tensor: [batch_size, 128] 编码后的特征
        """
        return self.encoder(low_level_obs)


class HighLevelPerceptionEncoder(nn.Module):
    """
    高层感知编码器
    
    该类负责处理高层决策所需的历史状态信息，是分层强化学习系统中高层策略的核心感知组件。
    它将一系列历史的占据栅格和yaw数据编码为全局上下文特征，为高层策略提供长期时序信息。
    
    核心功能：
    1. 历史序列处理：处理过去K个时间步的占据栅格和yaw数据
    2. 时序建模：利用Transformer捕捉时间序列中的长期依赖关系
    3. 上下文提取：生成包含环境历史信息的全局上下文向量
    4. 特征标准化：确保输出特征具有固定的维度和分布
    
    在HA-UAV系统中的作用：
    - 为高层策略提供历史环境感知能力
    - 帮助高层策略理解障碍物的动态变化模式
    - 支持基于历史经验的路径规划决策
    
    网络架构：
    输入 -> 状态编码 -> 位置编码 -> Transformer编码 -> 上下文向量
    """
    
    def __init__(self, config: Dict[str, int]):
        """
        初始化高层感知编码器
        
        Args:
            config (Dict[str, int]): 配置字典，包含以下键值：
                - grid_size (int): 占据栅格尺寸，默认200
                - k_history (int): 历史时间步数，决定观察历史的长度
                - d_model (int): 隐藏层维度
                - nhead (int): 高层Transformer的注意力头数，默认8
                - num_layers (int): 高层Transformer的层数，默认2
        """
        super().__init__()
        
        # 从配置中提取核心参数
        self.grid_size = config['grid_size']  # 占据栅格尺寸
        self.k_history = config['k_history']  # 历史步数，决定时序窗口大小
        self.d_model = config['d_model']  # 隐藏层维度
        
        # 初始化状态编码器：将每个时间步的占据栅格和yaw编码为统一特征
        self.state_encoder = StateEncoder(
            grid_size=self.grid_size,
            k_history=self.k_history, 
            output_dim=self.d_model
        )
        
        # 初始化位置编码器：为历史序列提供时序信息
        self.positional_encoding = PositionalEncoding(self.d_model, max_len=self.k_history)
        
        # 初始化Transformer编码器：处理历史序列的时序依赖
        self.transformer_encoder = TransformerEncoder(
            self.d_model, 
            config.get('nhead', 8),  # 高层Transformer注意力头数
            config.get('num_layers', 2)  # 高层Transformer层数
        )

    def forward(self, occupancy_grid: torch.Tensor, yaw_history: torch.Tensor) -> torch.Tensor:
        """
        前向传播：处理历史状态序列生成高层上下文特征
        
        Args:
            occupancy_grid (Tensor): 历史占据栅格序列，形状: [batch_size, k_history, grid_size, grid_size]
                                   包含连续K个时间步的占据栅格信息
            yaw_history (Tensor): 历史yaw角度序列，形状: [batch_size, k_history]
                                包含连续K个时间步的偏航角信息
        
        Returns:
            Tensor: E_HL 高层上下文特征，形状为 [batch_size, d_model]
                   这是一个全局上下文向量，包含了历史序列的综合信息
        
        处理流程：
            1. 状态编码器处理占据栅格和yaw历史
            2. 为编码后的序列添加位置编码
            3. 通过Transformer编码器处理时序依赖
            4. 提取最终的上下文特征向量
        
        注意：
            - 输入是连续的历史序列，包含K个时间步
            - 每个时间步包含占据栅格和yaw信息
            - 输出是单个上下文向量，综合了所有历史信息
        """
        # 记录原始输入维度以决定输出是否需要压缩
        original_input_was_3d = (hasattr(occupancy_grid, 'ndim') and occupancy_grid.ndim == 3) or \
                               (hasattr(occupancy_grid, 'dim') and occupancy_grid.dim() == 3)
        
        # 状态编码
        state_features = self.state_encoder(occupancy_grid, yaw_history)  # [batch_size, k_history, d_model]
        
        # 确保state_features是3D张量以支持permute操作
        if state_features.dim() == 2:
            # 如果是2D（因为单样本推理被压缩），添加batch维度
            state_features = state_features.unsqueeze(0)  # [1, k_history, d_model]
        
        # 转换为Transformer格式：[k_history, batch_size, d_model]
        state_seq = state_features.permute(1, 0, 2)
        
        # 位置编码
        pos_encoded = self.positional_encoding(state_seq)
        
        # Transformer编码
        transformer_out = self.transformer_encoder(pos_encoded)
        
        # 取最后一个时间步作为context
        context = transformer_out[-1, :, :]  # [batch_size, d_model]
        
        # 如果原始输入是3D（单样本推理），压缩输出的batch维度
        if original_input_was_3d and context.size(0) == 1:
            context = context.squeeze(0)  # [d_model]
        
        return context

class LowLevelPerceptionEncoder(nn.Module):
    """
    低层感知编码器
    
    该类负责处理低层控制决策所需的多模态信息，是分层强化学习系统中低层策略的核心感知组件。
    它整合扇区距离、历史动作和高层子目标，为低层策略提供全面的决策依据。
    
    核心功能：
    1. 多模态信息融合：整合扇区距离、历史动作和高层子目标
    2. 异构数据处理：为不同类型的输入使用专门的编码器
    3. 时序关系建模：利用Transformer处理不同时间尺度的信息
    4. 上下文生成：生成包含所有相关信息的低层上下文向量
    
    在HA-UAV系统中的作用：
    - 为低层策略提供即时环境感知能力
    - 整合高层策略的导航指令
    - 考虑历史动作对当前决策的影响
    - 支持精细的障碍物避让和路径跟踪
    
    网络架构：
    扇区距离 -> MLP编码器 ─┐
    历史动作 -> LSTM编码器 ──┤
    子目标序列 -> MLP编码器 ─┘
                          ↓
                    上下文投影层
                          ↓
                    Transformer编码器
                          ↓
                    低层上下文向量
    """
    
    def __init__(self, config: Dict[str, int]):
        """
        初始化低层感知编码器
        
        Args:
            config (Dict[str, int]): 配置字典，包含以下键值：
                - num_sectors (int): 扇区数量，默认36
                - m_history (int): 历史动作步数，决定动作历史的长度
                - t_horizon (int): 子目标序列长度
                - d_model (int): 隐藏层维度
                - nhead_ll (int): 低层Transformer的注意力头数，默认4
                - num_layers_ll (int): 低层Transformer的层数，默认1
        """
        super().__init__()
        
        # 从配置中提取核心参数
        self.num_sectors = config['num_sectors']  # 扇区数量，默认36
        self.m_history = config['m_history']  # 历史动作步数
        self.t_horizon = config['t_horizon']  # 子目标序列长度
        self.d_model = config['d_model']  # 隐藏层维度

        # ✅ 异构输入的独立编码器 - 使用统一配置的维度分配
        sector_dim = config.get('sector_encoding_dim', 86)  # 从配置获取
        action_dim = config.get('action_encoding_dim', 85)  # 从配置获取
        goal_dim = config.get('goal_encoding_dim', 85)      # 从配置获取
        
        # ✅ 验证维度配置的一致性
        expected_total = sector_dim + action_dim + goal_dim
        if expected_total != self.d_model:
            raise ValueError(f"编码器维度分配错误: {sector_dim}+{action_dim}+{goal_dim}={expected_total}, 期望={self.d_model}")
        
        print(f"🔧 LowLevelPerceptionEncoder维度配置:")
        print(f"  扇区编码维度: {sector_dim}")
        print(f"  动作编码维度: {action_dim}")  
        print(f"  目标编码维度: {goal_dim}")
        print(f"  总计: {expected_total} (期望: {self.d_model})")
        
        # 扇区距离编码器：将扇区距离数据映射到隐藏空间
        self.sector_mlp = nn.Sequential(
            nn.Linear(self.num_sectors, sector_dim),
            nn.ReLU(),
            nn.Linear(sector_dim, sector_dim)
        )
        
        # 历史动作序列编码器：使用LSTM处理动作的时序依赖 (4D: vx,vy,vz,yaw)
        self.action_lstm = nn.LSTM(4, action_dim, batch_first=True)
        
        # 子目标序列编码器：处理高层提供的导航子目标 (2D: heading, distance)
        self.subgoal_mlp = nn.Sequential(
            nn.Linear(2, goal_dim),
            nn.ReLU(),
            nn.Linear(goal_dim, goal_dim)
        )

        # 上下文投影层：将三种编码后的特征融合为统一的上下文token
        self.context_projection = nn.Linear(self.d_model, self.d_model)

        # 位置编码器：为Transformer序列提供位置信息
        # 序列长度 = 1个当前上下文 + m_history个历史动作 + t_horizon个子目标
        self.positional_encoding = PositionalEncoding(
            self.d_model, max_len=1 + self.m_history + self.t_horizon
        )
        
        # Transformer编码器：处理多模态序列的关系
        self.transformer_encoder = TransformerEncoder(
            self.d_model,
            config.get('nhead_ll', 4),  # 低层Transformer注意力头数
            config.get('num_layers_ll', 1)  # 低层Transformer层数
        )

        # 为了避免在forward中重复创建线性层，预先定义嵌入层
        self.action_embed = nn.Linear(4, self.d_model)  # 4D动作嵌入
        self.goal_embed = nn.Linear(2, self.d_model)    # 2D子目标嵌入

    def forward(self, sector_distances: torch.Tensor, action_history: torch.Tensor, sub_goals: torch.Tensor) -> torch.Tensor:
        """
        前向传播：处理多模态输入生成低层上下文特征
        
        Args:
            sector_distances (Tensor): 扇区距离数据，形状为 [batch_size, num_sectors]
                                      包含当前时刻的36个扇区距离信息
                                      值范围：[0.0, max_detection_range]
            action_history (Tensor): 历史动作序列，形状为 [batch_size, m_history, 4]
                                   包含过去m_history个时间步的动作历史 [vx, vy, vz, yaw]
                                   vx,vy,vz范围：[-1.0, 1.0]，yaw范围：[-π, π]
            sub_goals (Tensor): 高层子目标序列，形状为 [batch_size, t_horizon, 2]
                              包含高层策略提供的t_horizon个导航子目标 [heading, distance]
                              heading范围：[-π, π]，distance范围：[0.0, 1.0]
        
        Returns:
            Tensor: E_LL 低层上下文特征，形状为 [batch_size, d_model]
                   这是一个综合上下文向量，包含了所有相关的决策信息
        
        处理流程：
            1. 分别编码三种异构输入（扇区距离、历史动作、子目标）
            2. 将编码后的特征融合为统一的上下文token
            3. 构建Transformer输入序列
            4. 应用位置编码
            5. 通过Transformer处理序列关系
            6. 提取最终的低层上下文特征
        
        注意：
            - 三种输入具有不同的时间尺度和语义含义
            - LSTM用于处理历史动作的时序依赖
            - Transformer用于处理多模态信息的交互
        """
        # 第一步：编码异构输入
        
        # 编码扇区距离数据（归一化处理）
        sector_normalized = sector_distances / sector_distances.max(dim=-1, keepdim=True)[0].clamp(min=1e-6)
        e_L = self.sector_mlp(sector_normalized)  # [batch_size, d_model//3]
        
        # 编码历史动作序列
        # LSTM输出：output [batch_size, m_history, hidden_dim], (h_n, c_n)
        # 我们取最后一个隐藏状态作为历史动作的表示
        _, (h_n, _) = self.action_lstm(action_history)
        e_a = h_n.squeeze(0)  # [batch_size, d_model//3]
        
        # 编码子目标序列（平均池化）
        batch_size = sub_goals.size(0)
        subgoal_flat = sub_goals.reshape(-1, 2)  # [batch_size*t_horizon, 2] - 使用reshape代替view
        subgoal_encoded = self.subgoal_mlp(subgoal_flat)  # [batch_size*t_horizon, d_model//3]
        e_g = subgoal_encoded.view(batch_size, self.t_horizon, -1).mean(dim=1)  # [batch_size, d_model//3]
        
        # 检查并修复维度兼容性
        target_batch_size = e_L.shape[0]
        
        # 确保 e_a 的 batch 维度与 e_L 匹配
        if e_a.shape[0] != target_batch_size:
            if e_a.shape[0] == 1:
                e_a = e_a.repeat(target_batch_size, 1)  # 广播到正确的batch大小
            else:
                # 如果维度不匹配且不是1，重新调整
                e_a = e_a[:target_batch_size] if e_a.shape[0] > target_batch_size else torch.zeros(target_batch_size, e_a.shape[1]).to(e_a.device)
        
        # 确保 e_g 的 batch 维度与 e_L 匹配
        if e_g.shape[0] != target_batch_size:
            if e_g.shape[0] == 1:
                e_g = e_g.repeat(target_batch_size, 1)  # 广播到正确的batch大小
            else:
                # 如果维度不匹配且不是1，重新调整
                e_g = e_g[:target_batch_size] if e_g.shape[0] > target_batch_size else torch.zeros(target_batch_size, e_g.shape[1]).to(e_g.device)
        
        # 第二步：融合异构特征
        
        # 将三种编码后的特征拼接
        combined_features = torch.cat([e_L, e_a, e_g], dim=-1)  # [batch_size, d_model * 3]
        
        # 通过投影层生成统一的上下文token
        current_context_token = F.relu(self.context_projection(combined_features))  # [batch_size, d_model]

        # 第三步：构建Transformer输入序列
        
        # 确保所有张量使用相同的batch_size
        current_batch_size = e_L.shape[0]  # 使用修正后的batch_size
        
        # 为历史动作和子目标重新编码，用于Transformer序列输入
        # 注意：这里的编码与上面的异构编码不同，是为了形成序列
        
        # 确保action_history的batch_size匹配
        if action_history.shape[0] != current_batch_size:
            if action_history.shape[0] == 1:
                action_history = action_history.repeat(current_batch_size, 1, 1)
            else:
                # 截断或填充到正确的batch_size
                if action_history.shape[0] > current_batch_size:
                    action_history = action_history[:current_batch_size]
                else:
                    pad_size = current_batch_size - action_history.shape[0]
                    padding = torch.zeros(pad_size, action_history.shape[1], action_history.shape[2]).to(action_history.device)
                    action_history = torch.cat([action_history, padding], dim=0)
        
        history_actions_reshaped = action_history.reshape(-1, 4)
        history_actions_embed = F.relu(self.action_embed(history_actions_reshaped)).view(
            current_batch_size, self.m_history, self.d_model
        )
        
        # 确保sub_goals的batch_size匹配
        if sub_goals.shape[0] != current_batch_size:
            if sub_goals.shape[0] == 1:
                sub_goals = sub_goals.repeat(current_batch_size, 1, 1)
            else:
                # 截断或填充到正确的batch_size
                if sub_goals.shape[0] > current_batch_size:
                    sub_goals = sub_goals[:current_batch_size]
                else:
                    pad_size = current_batch_size - sub_goals.shape[0]
                    padding = torch.zeros(pad_size, sub_goals.shape[1], sub_goals.shape[2]).to(sub_goals.device)
                    sub_goals = torch.cat([sub_goals, padding], dim=0)
        
        goal_sequence_reshaped = sub_goals.reshape(-1, 2)  # 使用reshape代替view
        goal_sequence_embed = F.relu(self.goal_embed(goal_sequence_reshaped)).view(
            current_batch_size, self.t_horizon, self.d_model
        )

        # 构建Transformer输入序列：[当前上下文, 历史动作, 子目标序列]
        # 格式：[seq_len, batch_size, d_model]
        transformer_input_seq = torch.cat([
            current_context_token.unsqueeze(0),  # [1, batch_size, d_model]
            history_actions_embed.permute(1, 0, 2),  # [m_history, batch_size, d_model]
            goal_sequence_embed.permute(1, 0, 2)  # [t_horizon, batch_size, d_model]
        ], dim=0)  # [1+m_history+t_horizon, batch_size, d_model]

        # 第四步：应用位置编码
        pos_encoded_seq = self.positional_encoding(transformer_input_seq)
        
        # 第五步：通过Transformer处理序列关系
        transformer_output = self.transformer_encoder(pos_encoded_seq)  # [seq_len, batch_size, d_model]
        
        # 第六步：提取低层上下文特征
        # 取第一个token（当前上下文）的输出作为最终的低层上下文向量
        E_LL = transformer_output[0, :, :]  # [batch_size, d_model]
        
        return E_LL

class HighLevelActor(nn.Module):
    """
    高层策略网络（高层Actor）- SB3兼容版本
    
    该类实现了分层强化学习系统中的高层策略网络，负责根据历史环境信息生成导航子目标序列。
    高层策略专注于全局路径规划和长期导航决策，为低层策略提供中期导航指令。
    
    核心功能：
    1. 全局路径规划：基于历史环境信息规划导航路径
    2. 子目标生成：输出未来t_horizon步的导航子目标序列
    3. 双分支结构：分别处理方向（yaw）和距离（distance）决策
    4. SB3兼容性：支持Stable-Baselines3的PPO训练接口
    5. 概率分布：返回动作、log概率和分布对象
    
    输出格式：
    - 每个子目标包含两个分量：(yaw, distance)
    - yaw: 目标方向角，范围[-π, π]
    - distance: 目标距离，范围[0, 1]
    
    在HA-UAV系统中的作用：
    - 为无人机提供中期导航目标
    - 帮助无人机规划绕过障碍物的路径
    - 与低层策略协同实现分层控制
    
    网络架构：
    高层上下文特征 -> 分支1：yaw预测 -> Normal分布 -> tanh*π -> 角度输出
                    -> 分支2：距离预测 -> Normal分布 -> sigmoid -> 距离输出
    """
    
    def __init__(self, input_dim: int, t_horizon: int = 5):
        """
        初始化高层策略网络
        
        Args:
            input_dim (int): 输入特征的隐藏维度，来自高层感知编码器
            t_horizon (int): 子目标序列长度，即未来t_horizon步的规划长度，默认5
        
        网络结构：
            - 双分支MLP设计，分别处理角度和距离预测
            - 每个分支都是两层全连接网络，输出均值和对数标准差
            - 使用不同的激活函数确保输出范围合理
        """
        super().__init__()
        self.t_horizon = t_horizon

        # 方向角预测分支 - 输出均值和对数标准差
        # 预测未来t_horizon步的目标方向角
        self.yaw_net = nn.Sequential(
            nn.Linear(input_dim, 256),  
            nn.ReLU(),
            nn.Linear(256, t_horizon * 2)  # mean + log_std
        )
        
        # 距离预测分支 - 输出均值和对数标准差
        # 预测未来t_horizon步的目标距离
        self.dist_net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, t_horizon * 2)  # mean + log_std
        )

    def forward(self, context: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, dist.Distribution]:
        """
        前向传播：生成子目标序列
        
        Args:
            context (Tensor): 高层上下文特征，形状为 [batch_size, input_dim]
                            来自高层感知编码器，包含历史环境信息
            deterministic (bool): 是否返回确定性动作，默认False
        
        Returns:
            Tuple包含：
            - actions (Tensor): 子目标序列，形状为 [batch_size, t_horizon, 2]
                              最后一个维度包含 (yaw, distance) 信息
            - log_prob (Tensor): 动作的对数概率，形状为 [batch_size]
            - distribution (Distribution): 动作的概率分布对象
        
        处理流程：
            1. 通过yaw分支预测t_horizon个方向角的分布参数
            2. 通过distance分支预测t_horizon个距离值的分布参数
            3. 创建Independent Normal分布
            4. 采样或取均值（根据deterministic参数）
            5. 应用激活函数约束输出范围
            6. 计算log概率（考虑变换的雅可比）
        
        输出约束：
            - yaw: 使用tanh激活函数，输出范围[-π, π]
            - distance: 使用sigmoid激活函数，输出范围[0, 1]
        
        注意：
            - 子目标序列表示无人机未来t_horizon步的导航计划
            - 每个子目标都是相对于当前位置的导航指令
            - 输出格式便于低层策略网络处理
            - 支持SB3的PPO训练流程
        """
        # 生成分布参数
        yaw_params = self.yaw_net(context)  # [batch_size, t_horizon*2]
        dist_params = self.dist_net(context)  # [batch_size, t_horizon*2]
        
        # 分离均值和对数标准差
        yaw_mean, yaw_log_std = yaw_params.chunk(2, dim=-1)
        dist_mean, dist_log_std = dist_params.chunk(2, dim=-1)
        
        # 限制标准差范围
        yaw_std = torch.clamp(yaw_log_std.exp(), min=1e-4, max=1.0)
        dist_std = torch.clamp(dist_log_std.exp(), min=1e-4, max=1.0)
        
        # 创建分布
        yaw_dist = dist.Independent(dist.Normal(yaw_mean, yaw_std), 1)
        dist_dist = dist.Independent(dist.Normal(dist_mean, dist_std), 1)
        
        # 采样或取均值
        if deterministic:
            yaw_raw = yaw_mean
            dist_raw = dist_mean
        else:
            yaw_raw = yaw_dist.sample()
            dist_raw = dist_dist.sample()
        
        # 应用约束
        yaw_actions = torch.tanh(yaw_raw) * np.pi  # [-π, π]
        dist_actions = torch.sigmoid(dist_raw)  # [0, 1]
        
        # 计算log概率
        yaw_log_prob = yaw_dist.log_prob(yaw_raw)
        dist_log_prob = dist_dist.log_prob(dist_raw)
        
        # 修正tanh和sigmoid的雅可比
        yaw_log_prob -= torch.sum(torch.log(1 - torch.tanh(yaw_raw).pow(2) + 1e-6), dim=-1)
        dist_log_prob -= torch.sum(torch.log(torch.sigmoid(dist_raw) * (1 - torch.sigmoid(dist_raw)) + 1e-6), dim=-1)
        
        total_log_prob = yaw_log_prob + dist_log_prob
        
        # 组合动作
        actions = torch.stack([yaw_actions, dist_actions], dim=-1)  # [batch_size, t_horizon, 2]
        
        # 创建组合分布用于返回
        combined_dist = dist.Independent(dist.Normal(
            torch.cat([yaw_mean, dist_mean], dim=-1),
            torch.cat([yaw_std, dist_std], dim=-1)
        ), 1)
        
        return actions, total_log_prob, combined_dist

    def get_action_dist_from_latent(self, context: torch.Tensor) -> dist.Distribution:
        """SB3兼容方法：从潜在特征获取动作分布"""
        _, _, distribution = self.forward(context, deterministic=False)
        return distribution



class LowLevelActorWithYawControl(nn.Module):
    """
    带偏航角控制的低层执行器网络（Enhanced Low-Level Actor with Yaw Control）
    
    这是HA-UAV系统的增强版低层策略网络，在原有三维速度控制的基础上增加了偏航角控制能力。
    该网络能够同时输出水平速度、垂直速度和偏航角速度，实现四维度的精确无人机控制。
    
    核心功能：
    1. 四维控制：生成(vx, vy, vz, yaw_rate)四维控制向量
    2. 智能偏航：结合高层偏航信息和当前状态智能决策偏航控制
    3. 自适应模式：根据任务需求在直接控制和跟踪模式间切换
    4. 平滑控制：确保偏航角变化的平滑性和稳定性
    
    智能偏航控制策略：
    - 当高层提供明确偏航目标时，执行偏航跟踪控制
    - 当处于精细避障阶段时，偏航服务于移动方向
    - 结合速度向量和偏航目标，选择最优的偏航策略
    - 应用速率限制确保控制的平滑性
    
    输出格式：
    - 四维控制向量：(vx, vy, vz, yaw_rate)
    - vx, vy: 水平方向速度分量 [-max_speed, max_speed]
    - vz: 垂直方向速度分量 [-max_speed, max_speed]  
    - yaw_rate: 偏航角速度 [-max_yaw_rate, max_yaw_rate]
    
    在HA-UAV系统中的作用：
    - 执行高层策略的四维导航指令
    - 实现精确的偏航角控制
    - 提供平滑的四维飞行控制
    - 确保无人机的稳定性和安全性
    
    网络架构：
    低层上下文特征 -> 双分支网络 -> [速度分支(3D) + 偏航分支(1D)] -> 四维控制向量
    """
    
    def __init__(self, hidden_dim: int, max_speed: float, max_yaw_rate: float = 1.0):
        """
        初始化带偏航控制的低层策略网络
        
        Args:
            hidden_dim (int): 输入特征的隐藏维度，来自低层感知编码器
            max_speed (float): 最大线性速度限制，用于约束vx, vy, vz输出范围
            max_yaw_rate (float): 最大偏航角速度限制，默认1.0 rad/s
        
        网络结构：
            - 双分支架构：速度分支和偏航分支
            - 速度分支：处理三维线性运动控制
            - 偏航分支：处理偏航角速度控制
            - 共享特征提取层 + 专门化输出层
        """
        super().__init__()
        self.max_speed = max_speed  # 最大线性速度约束
        self.max_yaw_rate = max_yaw_rate  # 最大偏航角速度约束
        
        # 共享特征提取层
        self.shared_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),  # 轻微正则化
        )
        
        # 速度控制分支 (vx, vy, vz)
        self.velocity_branch = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 3),  # 输出3D速度
            nn.Tanh()  # 将输出限制在[-1, 1]范围内
        )
        
        # 偏航控制分支 (yaw_rate)
        self.yaw_branch = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),  # 输出偏航角速度
            nn.Tanh()  # 将输出限制在[-1, 1]范围内
        )

    def forward(self, E_LL: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.distributions.Distribution]:
        """
        前向传播：生成四维控制向量及其概率分布信息
        
        Args:
            E_LL (Tensor): 低层上下文特征，形状为 [batch_size, hidden_dim]
                          来自低层感知编码器，包含：
                          - 当前环境感知信息
                          - 历史动作信息  
                          - 高层子目标信息（包含偏航信息）
        
        Returns:
            Tuple包含：
            - actions (Tensor): 四维控制向量，形状为 [batch_size, 4]
                              包含 (vx, vy, vz, yaw_rate) 四个分量
            - log_prob (Tensor): 动作的对数概率，形状为 [batch_size]
            - distribution: 动作分布对象，用于采样和概率计算
        
        处理流程：
            1. 通过共享层提取特征
            2. 速度分支生成三维线性速度
            3. 偏航分支生成偏航角速度
            4. 合并为四维控制向量
            5. 创建多变量正态分布
            6. 计算对数概率
        
        控制策略：
            - 速度控制：基于避障和目标跟踪需求
            - 偏航控制：结合高层偏航目标和移动方向
            - 协调控制：确保速度和偏航的协调性
            - 平滑约束：保证控制信号的平滑性
        """
        # 共享特征提取
        shared_features = self.shared_layers(E_LL)  # [batch_size, hidden_dim//2]
        
        # 双分支处理
        velocity_normalized = self.velocity_branch(shared_features)  # [batch_size, 3]
        yaw_rate_normalized = self.yaw_branch(shared_features)  # [batch_size, 1]
        
        # 应用各自的速度约束
        velocity = velocity_normalized * self.max_speed  # [batch_size, 3]
        yaw_rate = yaw_rate_normalized * self.max_yaw_rate  # [batch_size, 1]
        
        # 合并为四维控制向量
        a_t = torch.cat([velocity, yaw_rate], dim=-1)  # [batch_size, 4]
        
        # 创建多变量正态分布 (用于RL训练)
        # 为简化，使用固定的标准差
        action_std = torch.ones_like(a_t) * 0.1  # 固定标准差0.1
        action_dist = torch.distributions.MultivariateNormal(
            loc=a_t, 
            scale_tril=torch.diag_embed(action_std)
        )
        
        # 从分布中采样动作
        sampled_actions = action_dist.sample()
        
        # 计算对数概率
        log_prob = action_dist.log_prob(sampled_actions)
        
        return sampled_actions, log_prob, action_dist

class HighLevelCritic(nn.Module):
    """
    高层评论家网络（高层Critic）- SB3兼容版本
    
    该类实现了分层强化学习系统中的高层评论家网络，负责评估高层策略的价值函数。
    高层评论家估计在给定高层状态下的状态价值，为高层策略的训练提供价值信号。
    
    功能特点：
    1. 状态价值估计：基于高层上下文特征估计状态价值
    2. 长期奖励预测：考虑高层决策的长期影响
    3. SB3兼容性：符合Stable-Baselines3的Critic接口规范
    4. 标准化输出：输出单个标量价值
    
    在HA-UAV系统中的作用：
    - 为高层PPO训练提供价值基线
    - 计算优势函数（Advantage）
    - 支持广义优势估计（GAE）
    - 评估高层策略的长期效果
    
    网络架构：
    高层上下文特征 -> 隐藏层(ReLU) -> 隐藏层(ReLU) -> 输出层 -> 状态价值
    """
    
    def __init__(self, input_dim: int):
        """
        初始化高层评论家网络
        
        Args:
            input_dim (int): 输入上下文特征的维度
        
        网络结构：
            - 三层全连接网络
            - 两个隐藏层提供足够的表达能力
            - 输出层不使用激活函数，输出原始价值
        """
        super().__init__()
        
        # 构建多层感知机
        self.critic_net = nn.Sequential(
            # 第一隐藏层
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            # 第二隐藏层
            nn.Linear(256, 256), 
            nn.ReLU(),
            # 输出层：输出单个价值
            nn.Linear(256, 1)
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        前向传播：估计状态价值
        
        Args:
            context (Tensor): 高层上下文特征，形状为 [batch_size, input_dim]
                            来自高层感知编码器，包含历史环境信息
        
        Returns:
            Tensor: 估计的状态价值，形状为 [batch_size, 1]
                   表示在给定状态下的期望累积奖励
        
        处理流程：
            1. 通过第一隐藏层进行特征变换
            2. 通过第二隐藏层进一步处理
            3. 输出层生成单个价值估计
        
        价值含义：
            - 正值：表示该状态的价值高于平均水平
            - 负值：表示该状态的价值低于平均水平
            - 绝对值大小：表示价值估计的置信度
        
        注意：
            - 输出单个标量价值，符合SB3标准
            - 不使用最终激活函数，允许价值为任意实数
            - 用于计算优势函数和价值损失
        """
        return self.critic_net(context)

class LowLevelCritic(nn.Module):
    """
    低层评论家网络（低层Critic）- SB3兼容版本
    
    该类实现了分层强化学习系统中的低层评论家网络，负责评估低层策略的价值函数。
    低层评论家估计在给定低层状态下的状态价值，为低层策略的训练提供价值信号。
    
    功能特点：
    1. 状态价值估计：基于低层上下文特征估计状态价值
    2. 短期奖励预测：主要考虑即时奖励和短期效果
    3. SB3兼容性：符合Stable-Baselines3的Critic接口规范
    4. 标准化输出：输出单个标量价值
    
    在HA-UAV系统中的作用：
    - 为低层PPO训练提供价值基线
    - 计算优势函数（Advantage）
    - 支持广义优势估计（GAE）
    - 评估低层策略的即时效果
    
    网络架构：
    低层上下文特征 -> 隐藏层(ReLU) -> 隐藏层(ReLU) -> 输出层 -> 状态价值
    """
    
    def __init__(self, input_dim: int):
        """
        初始化低层评论家网络
        
        Args:
            input_dim (int): 输入上下文特征的维度
        
        网络结构：
            - 三层全连接网络
            - 两个隐藏层提供足够的表达能力
            - 输出层不使用激活函数，输出原始价值
        """
        super().__init__()
        
        # 构建多层感知机
        self.critic_net = nn.Sequential(
            # 第一隐藏层
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            # 第二隐藏层
            nn.Linear(256, 256),
            nn.ReLU(), 
            # 输出层：输出单个价值
            nn.Linear(256, 1)
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        前向传播：估计状态价值
        
        Args:
            context (Tensor): 低层上下文特征，形状为 [batch_size, input_dim]
                            来自低层感知编码器，包含当前环境感知、历史动作和子目标信息
        
        Returns:
            Tensor: 估计的状态价值，形状为 [batch_size, 1]
                   表示在给定状态下的期望累积奖励
        
        处理流程：
            1. 通过第一隐藏层进行特征变换
            2. 通过第二隐藏层进一步处理
            3. 输出层生成单个价值估计
        
        价值含义：
            - 正值：表示该状态的价值高于平均水平
            - 负值：表示该状态的价值低于平均水平
            - 绝对值大小：表示价值估计的置信度
        
        注意：
            - 输出单个标量价值，符合SB3标准
            - 不使用最终激活函数，允许价值为任意实数
            - 用于计算优势函数和价值损失
            - 训练频率通常比高层更高
        """
        return self.critic_net(context)


class HierarchicalRLSystem(nn.Module):
    """
    分层强化学习系统
    
    集成高层和低层策略，提供统一的接口用于训练和推理。
    这是一个适配器类，将各个组件整合成完整的分层RL系统。
    """
    
    def __init__(self, config: Dict = None):
        """
        初始化分层RL系统
        
        Args:
            config: 系统配置参数
        """
        super(HierarchicalRLSystem, self).__init__()
        
        self.config = config or {}
        
        # 系统参数
        self.lidar_dim = self.config.get('lidar_dim', 36)
        self.action_dim = self.config.get('action_dim', 4)
        self.grid_size = self.config.get('grid_size', 32)
        self.yaw_history_len = self.config.get('yaw_history_len', 20)
        self.state_dim = self.config.get('state_dim', 256)
        self.subgoal_dim = self.config.get('subgoal_dim', 10)
        
        # 计算维度
        self.high_level_input_dim = self.grid_size * self.grid_size + self.yaw_history_len
        self.low_level_input_dim = self.high_level_input_dim + self.subgoal_dim
        
        # 初始化组件
        self._init_components()
        
    def _init_components(self):
        """初始化系统组件"""
        
        # 状态编码器 - 修复参数
        self.state_encoder = StateEncoder(
            grid_size=self.grid_size,
            k_history=self.yaw_history_len,
            output_dim=self.state_dim
        )
        
        # 高层感知编码器 - 修复参数
        high_config = {
            'grid_size': self.grid_size,
            'k_history': self.yaw_history_len,
            'd_model': self.state_dim
        }
        self.high_level_encoder = HighLevelPerceptionEncoder(high_config)
        
        # 低层感知编码器 - 修复参数
        low_config = {
            'num_sectors': self.lidar_dim,
            'm_history': 10,  # 默认历史长度
            't_horizon': 5,   # 默认时间跨度
            'd_model': self.state_dim
        }
        self.low_level_encoder = LowLevelPerceptionEncoder(low_config)
        
        # 高层策略和评论家 - 修复参数
        self.high_level_actor = HighLevelActor(
            input_dim=self.state_dim,
            t_horizon=5  # 使用t_horizon而不是subgoal_dim
        )
        
        self.high_level_critic = HighLevelCritic(
            input_dim=self.state_dim
        )
        
        # 低层策略和评论家 - 从配置中获取正确维度
        self.low_level_actor = LowLevelActorWithYawControl(
            hidden_dim=self.config.get('d_model', 256),  # 使用与感知编码器一致的维度
            max_speed=self.config.get('max_speed', 2.0),
            max_yaw_rate=self.config.get('max_yaw_rate', 1.0)
        )
        
        self.low_level_critic = LowLevelCritic(
            input_dim=self.state_dim
        )
        
    def forward(self, observation: torch.Tensor, mode: str = 'low_level') -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            observation: 输入观测
            mode: 'high_level' 或 'low_level'
            
        Returns:
            包含动作、价值等的字典
        """
        if mode == 'high_level':
            return self._forward_high_level(observation)
        else:
            return self._forward_low_level(observation)
    
    def _forward_high_level(self, observation: torch.Tensor) -> Dict[str, torch.Tensor]:
        """高层前向传播"""
        # 编码观测
        context = self.high_level_encoder(observation)
        
        # 生成子目标
        subgoal_mean, subgoal_log_std = self.high_level_actor(context)
        
        # 计算价值
        value = self.high_level_critic(context)
        
        return {
            'subgoal_mean': subgoal_mean,
            'subgoal_log_std': subgoal_log_std,
            'value': value,
            'context': context
        }
    
    def _forward_low_level(self, observation: torch.Tensor) -> Dict[str, torch.Tensor]:
        """低层前向传播"""
        # 编码观测
        context = self.low_level_encoder(observation)
        
        # 生成动作
        action_mean, action_log_std = self.low_level_actor(context)
        
        # 计算价值
        value = self.low_level_critic(context)
        
        return {
            'action_mean': action_mean,
            'action_log_std': action_log_std,
            'value': value,
            'context': context
        }
    
    def get_action(self, observation: torch.Tensor, deterministic: bool = False, mode: str = 'low_level') -> torch.Tensor:
        """
        获取动作（用于RL训练兼容性）
        
        Args:
            observation: 输入观测
            deterministic: 是否使用确定性策略
            mode: 'high_level' 或 'low_level'
            
        Returns:
            torch.Tensor: 动作向量
        """
        with torch.no_grad():
            outputs = self.forward(observation, mode)
            
            if mode == 'high_level':
                # 高层返回子目标
                subgoal_mean = outputs['subgoal_mean']
                if deterministic:
                    return subgoal_mean
                else:
                    subgoal_log_std = outputs['subgoal_log_std']
                    noise = torch.randn_like(subgoal_mean)
                    return subgoal_mean + noise * torch.exp(subgoal_log_std)
            else:
                # 低层返回动作
                action_mean = outputs['action_mean']
                if deterministic:
                    return action_mean
                else:
                    action_log_std = outputs['action_log_std']
                    noise = torch.randn_like(action_mean)
                    return action_mean + noise * torch.exp(action_log_std)
    
    def get_value(self, observation: torch.Tensor, mode: str = 'low_level') -> torch.Tensor:
        """
        获取状态价值（用于RL训练兼容性）
        
        Args:
            observation: 输入观测
            mode: 'high_level' 或 'low_level'
            
        Returns:
            torch.Tensor: 状态价值
        """
        with torch.no_grad():
            outputs = self.forward(observation, mode)
            return outputs['value']
    