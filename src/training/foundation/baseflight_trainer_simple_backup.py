#!/usr/bin/env python3

"""
简化版基座模型训练器 - 专注于BaseFlightAviary训练
删除不必要的复杂功能，专注核心训练逻辑
"""

import logging
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import sys
import traceback

# 添加项目路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入环境
from src.envs.BaseFlightAviary import BaseFlightAviary, BaseFlightConfig, create_base_flight_aviary

logger = logging.getLogger(__name__)


class SimpleFlightModel(nn.Module):
    """简化飞行模型 - 去除过度复杂的设计"""
    
    def __init__(self, obs_dim: int = 86, action_dim: int = 4):
        super().__init__()
        
        # 简单的网络架构
        self.policy_net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()
        )
        
        self.value_net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # 简单的参数初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, 0.01)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        action = self.policy_net(obs)
        value = self.value_net(obs)
        return action, value
    
    def act(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """获取动作"""
        with torch.no_grad():
            action, _ = self.forward(obs)
            if not deterministic:
                # 简单的探索噪声
                noise = torch.randn_like(action) * 0.1
                action = torch.clamp(action + noise, -1.0, 1.0)
            return action


class SimpleBaseFlightTrainer:
    """
    简化的基座飞行训练器
    
    删除复杂的会话管理、轨迹记录、TensorBoard等功能
    专注于核心的强化学习训练逻辑
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 基本训练参数
        self.total_timesteps = config.get('total_timesteps', 50000)
        self.learning_rate = config.get('learning_rate', 3e-4)
        self.batch_size = config.get('batch_size', 64)
        self.n_steps = config.get('n_steps', 2048)
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        
        # 环境和模型
        self.env = None
        self.model = None
        self.optimizer = None
        
        # 训练状态
        self.current_step = 0
        self.episode_count = 0
        self.best_reward = -float('inf')
        
        # 简化的统计
        self.episode_rewards = []
        self.episode_lengths = []
        
        logger.info("SimpleBaseFlightTrainer初始化完成")
    
    def setup(self) -> bool:
        """简化的设置"""
        try:
            # 创建环境
            self.env = create_base_flight_aviary(
                gui=self.config.get('gui', False),
                training_stage='hover',
                obstacles=self.config.get('obstacles', False),
                record=False
            )
            
            # 创建模型
            self.model = SimpleFlightModel()
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            
            logger.info("✅ 训练器设置完成")
            return True
            
        except Exception as e:
            logger.error(f"❌ 训练器设置失败: {e}")
            traceback.print_exc()
            return False
    
    def train(self) -> Dict[str, Any]:
        """核心训练循环"""
        logger.info(f"开始训练: {self.total_timesteps} 步")
        
        start_time = time.time()
        
        # 重置环境
        obs, info = self.env.reset()
        episode_reward = 0
        episode_length = 0
        rollout_buffer = []
        
        while self.current_step < self.total_timesteps:
            try:
                # 收集经验
                for _ in range(self.n_steps):
                    if self.current_step >= self.total_timesteps:
                        break
                    
                    # 获取动作
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                    action = self.model.act(obs_tensor)
                    action_np = action.squeeze(0).numpy()
                    
                    # 执行动作
                    next_obs, reward, terminated, truncated, info = self.env.step(action_np)
                    done = terminated or truncated
                    
                    # 获取价值估计
                    _, value = self.model.forward(obs_tensor)
                    
                    # 存储经验
                    rollout_buffer.append({
                        'obs': obs,
                        'action': action_np,
                        'reward': reward,
                        'value': value.item(),
                        'done': done
                    })
                    
                    # 更新状态
                    obs = next_obs
                    episode_reward += reward
                    episode_length += 1
                    self.current_step += 1
                    
                    # 处理episode结束
                    if done:
                        self.episode_rewards.append(episode_reward)
                        self.episode_lengths.append(episode_length)
                        self.episode_count += 1
                        
                        if episode_reward > self.best_reward:
                            self.best_reward = episode_reward
                        
                        # 重置环境
                        obs, info = self.env.reset()
                        episode_reward = 0
                        episode_length = 0
                        
                        # 打印进度
                        if self.episode_count % 10 == 0:
                            recent_rewards = self.episode_rewards[-10:]
                            avg_reward = np.mean(recent_rewards)
                            logger.info(f"Episode {self.episode_count}: 平均奖励={avg_reward:.2f}, "
                                      f"步数={self.current_step}/{self.total_timesteps}")
                
                # 执行PPO训练
                if len(rollout_buffer) > 0:
                    self._train_step(rollout_buffer)
                    rollout_buffer.clear()
                
            except Exception as e:
                logger.error(f"训练步骤失败: {e}")
                traceback.print_exc()
                break
        
        # 训练完成
        elapsed_time = time.time() - start_time
        
        results = {
            'total_episodes': self.episode_count,
            'total_steps': self.current_step,
            'best_reward': self.best_reward,
            'avg_reward': np.mean(self.episode_rewards[-50:]) if self.episode_rewards else 0.0,
            'training_time': elapsed_time
        }
        
        logger.info(f"训练完成: {results}")
        return results
    
    def _train_step(self, rollout_buffer: List[Dict]):
        """简化的PPO训练步骤"""
        if len(rollout_buffer) < self.batch_size:
            return
        
        # 转换数据
        batch_obs = torch.FloatTensor([exp['obs'] for exp in rollout_buffer])
        batch_actions = torch.FloatTensor([exp['action'] for exp in rollout_buffer])
        batch_rewards = torch.FloatTensor([exp['reward'] for exp in rollout_buffer])
        batch_values = torch.FloatTensor([exp['value'] for exp in rollout_buffer])
        batch_dones = torch.FloatTensor([float(exp['done']) for exp in rollout_buffer])
        
        # 计算GAE
        returns, advantages = self._compute_gae(batch_rewards, batch_values, batch_dones)
        
        # 归一化advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO更新
        for _ in range(4):  # 4次更新
            # 前向传播
            pred_actions, pred_values = self.model.forward(batch_obs)
            
            # 损失计算
            value_loss = nn.MSELoss()(pred_values.squeeze(), returns)
            
            # 简化的策略损失（行为克隆 + advantage加权）
            action_diff = ((pred_actions - batch_actions) ** 2).mean()
            policy_loss = action_diff - 0.01 * advantages.mean()
            
            total_loss = value_loss + policy_loss
            
            # 反向传播
            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
    
    def _compute_gae(self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor):
        """计算GAE"""
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        
        gae = 0
        next_value = 0
        
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[i + 1] * (1 - dones[i])
            
            delta = rewards[i] + self.gamma * next_value - values[i]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[i]) * gae
            advantages[i] = gae
            returns[i] = advantages[i] + values[i]
        
        return returns, advantages
    
    def evaluate(self, num_episodes: int = 5) -> Dict[str, float]:
        """简化的评估"""
        logger.info(f"开始评估: {num_episodes} episodes")
        
        eval_rewards = []
        eval_lengths = []
        
        self.model.eval()
        
        for episode in range(num_episodes):
            obs, info = self.env.reset()
            episode_reward = 0
            episode_length = 0
            
            while True:
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                action = self.model.act(obs_tensor, deterministic=True)
                action_np = action.squeeze(0).numpy()
                
                obs, reward, terminated, truncated, info = self.env.step(action_np)
                episode_reward += reward
                episode_length += 1
                
                if terminated or truncated or episode_length >= 500:
                    break
            
            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)
            
            logger.info(f"评估 episode {episode + 1}: 奖励={episode_reward:.2f}")
        
        self.model.train()
        
        return {
            'mean_reward': float(np.mean(eval_rewards)),
            'std_reward': float(np.std(eval_rewards)),
            'mean_length': float(np.mean(eval_lengths))
        }
    
    def save_model(self, filepath: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'training_stats': {
                'episode_rewards': self.episode_rewards,
                'episode_lengths': self.episode_lengths,
                'best_reward': self.best_reward,
                'current_step': self.current_step
            }
        }, filepath)
        logger.info(f"模型已保存: {filepath}")
    
    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        stats = checkpoint.get('training_stats', {})
        self.episode_rewards = stats.get('episode_rewards', [])
        self.episode_lengths = stats.get('episode_lengths', [])
        self.best_reward = stats.get('best_reward', -float('inf'))
        self.current_step = stats.get('current_step', 0)
        
        logger.info(f"模型已加载: {filepath}")
    
    def cleanup(self):
        """清理资源"""
        if self.env:
            self.env.close()
        logger.info("训练器资源已清理")


def create_simple_trainer(config: Dict[str, Any]) -> SimpleBaseFlightTrainer:
    """创建简化trainer的便捷函数"""
    return SimpleBaseFlightTrainer(config)


if __name__ == "__main__":
    # 测试简化版trainer
    test_config = {
        'total_timesteps': 10000,
        'learning_rate': 3e-4,
        'batch_size': 64,
        'n_steps': 512,
        'gui': False,
        'obstacles': False
    }
    
    trainer = create_simple_trainer(test_config)
    
    if trainer.setup():
        logger.info("✅ 开始简化版训练测试")
        try:
            results = trainer.train()
            logger.info(f"训练结果: {results}")
            
            # 简单评估
            eval_results = trainer.evaluate(3)
            logger.info(f"评估结果: {eval_results}")
            
        except KeyboardInterrupt:
            logger.info("训练被用户中断")
        finally:
            trainer.cleanup()
    else:
        logger.error("❌ 训练器设置失败")
