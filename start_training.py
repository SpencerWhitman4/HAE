#!/usr/bin/env python3

"""
HA-UAV训练启动脚本 - 统一的训练入口点
支持所有训练阶段：Foundation、Hierarchical、Ablation、Baseline
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.training.core.base_trainer import TrainingStage


# 内置默认配置
DEFAULT_CONFIGS = {
        'foundation': {
        'total_timesteps': 1_00_000,
        'batch_size': 256,
        'learning_rate': 3e-4,
        'evaluation_frequency': 5_000,  # 每5000步评估一次
        'save_frequency': 1_000,       #
        'checkpoint_frequency': 10_000, 
        'max_episode_steps': 300,
        'n_steps': 2048,
        'n_epochs': 10,
        'clip_range': 0.2,
        'vf_coef': 0.5,
        'ent_coef': 0.01,
        'gae_lambda': 0.95,
        'gamma': 0.99,
        # 显示配置
        'buffer_size': 2048,  # PPO n_steps
        'network_arch': 'PPO-MLP',
        # 续训设置
        'resume_training': True,  # 默认启用续训
        'auto_load_best': True,  # 自动加载最佳模型
    },
    'hierarchical': {
        'total_timesteps': 200000,
        'batch_size': 128,
        'learning_rate': 1e-4,
        'environment_type': 'hauav',
        'evaluation_frequency': 20000,
        'save_frequency': 20000,
        'enable_tensorboard': True,
        'enable_trajectory_recording': True,
        'enable_visualization': True,
        # HA-UAV特定参数
        'high_level_update_frequency': 10,
        'future_horizon': 5,
        'buffer_size': 100_000,
        # 显示配置
        'network_arch': 'HA-UAV-Hierarchical',
    },
    'ablation': {
        'total_timesteps': 100000,
        'batch_size': 128,
        'learning_rate': 1e-4,
        'environment_type': 'hauav',
        'evaluation_frequency': 10000,
        'save_frequency': 10000,
        'enable_tensorboard': True,
        'enable_trajectory_recording': True,
        'enable_visualization': True,
        # 消融研究特定参数
        'high_level_update_frequency': 10,
        'future_horizon': 5,
        'buffer_size': 100_000,
        # 显示配置
        'network_arch': 'HA-UAV-Ablation',
    },
    'baseline': {
        'total_timesteps': 100000,
        'batch_size': 256,
        'learning_rate': 3e-4,
        'environment_type': 'hauav',
        'evaluation_frequency': 10000,
        'save_frequency': 10000,
        'enable_tensorboard': True,
        'enable_trajectory_recording': True,
        'enable_visualization': True,
        # 基线算法参数
        'buffer_size': 1_000_000,  # 典型RL算法缓冲区大小
        # 显示配置
        'network_arch': 'SAC/TD3-MLP',
    }
}


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="HA-UAV训练系统启动器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # Foundation训练
  python start_training.py --stage foundation --timesteps 100000
  
  # Hierarchical训练
  python start_training.py --stage hierarchical --foundation-model foundation_model.zip
  
  # Ablation实验
  python start_training.py --stage ablation --variant B1 --foundation-model foundation_model.zip
  
  # Baseline对比
  python start_training.py --stage baseline --variant ppo --timesteps 50000
  
  # 带GUI的调试训练
  python start_training.py --stage foundation --gui --timesteps 10000 --debug
        """
    )
    
    # 必需参数
    parser.add_argument(
        '--stage', 
        type=str, 
        required=True,
        choices=['foundation', 'hierarchical', 'ablation', 'baseline'],
        help='训练阶段'
    )
    
    # 可选参数
    parser.add_argument(
        '--variant',
        type=str,
        help='阶段变体 (如: B1/B2/B3 for ablation, ppo/sac for baseline)'
    )
    
    parser.add_argument(
        '--timesteps',
        type=int,
        help='训练总步数'
    )
    
    parser.add_argument(
        '--foundation-model',
        type=str,
        help='基座模型路径 (用于hierarchical和ablation阶段)'
    )
    
    parser.add_argument(
        '--experiment-name',
        type=str,
        default='HA-UAV',
        help='实验名称'
    )
    
    # 环境配置
    parser.add_argument(
        '--gui',
        action='store_true',
        help='启用GUI显示'
    )
    
    parser.add_argument(
        '--drone-model',
        type=str,
        choices=['CF2X', 'CF2P'],
        default='CF2X',
        help='无人机模型'
    )
    
    parser.add_argument(
        '--physics',
        type=str,
        choices=['PYB', 'DYN'],
        default='PYB',
        help='物理引擎'
    )
    
    # 训练配置
    parser.add_argument(
        '--learning-rate',
        type=float,
        help='学习率'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        help='批次大小'
    )
    
    parser.add_argument(
        '--eval-freq',
        type=int,
        help='评估频率'
    )
    
    parser.add_argument(
        '--save-freq',
        type=int,
        help='保存频率'
    )
    
    # 功能开关
    parser.add_argument(
        '--no-tensorboard',
        action='store_true',
        help='禁用TensorBoard日志'
    )
    
    parser.add_argument(
        '--no-trajectory',
        action='store_true',
        help='禁用轨迹记录'
    )
    
    parser.add_argument(
        '--no-visualization',
        action='store_true',
        help='禁用实时可视化'
    )
    
    parser.add_argument(
        '--enable-pointcloud',
        action='store_true',
        help='启用点云记录'
    )
    
    parser.add_argument(
        '--enable-analysis',
        action='store_true',
        help='启用详细分析'
    )
    
    # 调试选项
    parser.add_argument(
        '--debug',
        action='store_true',
        help='启用调试模式'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='日志级别'
    )
    
    parser.add_argument(
        '--config-file',
        type=str,
        help='自定义配置文件路径'
    )
    
    return parser.parse_args()


def setup_logging(level: str, debug: bool = False):
    """设置日志系统"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    if debug:
        level = 'DEBUG'
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    
    logging.basicConfig(
        level=getattr(logging, level),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('training.log')
        ]
    )


def load_default_config(stage: str) -> Dict[str, Any]:
    """加载默认配置"""
    return DEFAULT_CONFIGS.get(stage, {}).copy()


def load_custom_config(config_file: str) -> Dict[str, Any]:
    """加载自定义配置文件"""
    import json
    import yaml
    
    config_path = Path(config_file)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_file}")
    
    if config_path.suffix == '.json':
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    elif config_path.suffix in ['.yaml', '.yml']:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    else:
        raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")


def build_config(args) -> Dict[str, Any]:
    """构建训练配置"""
    # 加载基础配置
    if args.config_file:
        config = load_custom_config(args.config_file)
    else:
        config = load_default_config(args.stage)
    
    # 命令行参数覆盖
    if args.timesteps:
        config['total_timesteps'] = args.timesteps
    
    if args.learning_rate:
        config['learning_rate'] = args.learning_rate
    
    if args.batch_size:
        config['batch_size'] = args.batch_size
    
    if args.eval_freq:
        config['evaluation_frequency'] = args.eval_freq
    
    if args.save_freq:
        config['save_frequency'] = args.save_freq
    
    # 环境配置
    config.update({
        'drone_model': args.drone_model,
        'physics': args.physics,
        'gui_training': args.gui,
        'gui_evaluation': args.gui,
        'experiment_name': args.experiment_name
    })
    
    # 功能开关
    config.update({
        'enable_tensorboard': not args.no_tensorboard,
        'enable_trajectory_recording': not args.no_trajectory,
        'enable_visualization': not args.no_visualization,
        'enable_pointcloud': args.enable_pointcloud,
        'enable_analysis': args.enable_analysis
    })
    
    # 阶段特定配置
    if args.stage in ['hierarchical', 'ablation'] and args.foundation_model:
        foundation_path = Path(args.foundation_model)
        if not foundation_path.exists():
            raise FileNotFoundError(f"基座模型文件不存在: {args.foundation_model}")
        config['foundation_model_path'] = str(foundation_path.absolute())
    
    if args.variant:
        config['stage_variant'] = args.variant
    
    # 调试模式配置
    if args.debug:
        config.update({
            'total_timesteps': min(config.get('total_timesteps', 1000), 10000),
            'evaluation_frequency': 1000,
            'save_frequency': 1000,
            'max_episode_steps': 100
        })
    
    return config


def validate_config(stage: str, config: Dict[str, Any]) -> bool:
    """验证配置有效性"""
    logger = logging.getLogger(__name__)
    
    # 基本验证
    required_keys = ['total_timesteps', 'drone_model', 'physics']
    for key in required_keys:
        if key not in config:
            logger.error(f"配置缺少必需参数: {key}")
            return False
    
    # 阶段特定验证
    if stage in ['hierarchical', 'ablation']:
        if 'foundation_model_path' not in config:
            logger.warning(f"{stage} 阶段建议使用预训练的基座模型")
    
    # 数值范围验证
    if config['total_timesteps'] <= 0:
        logger.error("总训练步数必须大于0")
        return False
    
    if 'learning_rate' in config and (config['learning_rate'] <= 0 or config['learning_rate'] > 1):
        logger.error("学习率必须在(0, 1]范围内")
        return False
    
    return True


def create_trainer(stage: str, config: Dict[str, Any], args):
    """创建训练器"""
    logger = logging.getLogger(__name__)
    
    # 转换stage字符串为枚举
    stage_enum = TrainingStage(stage)
    
    try:
        # 根据阶段创建相应的训练器
        if stage == 'foundation':
            # 使用BaseFlightTrainer的基座模型训练器
            from src.training.foundation.baseflight_trainer import BaseFlightTrainer
            trainer = BaseFlightTrainer(config)
            
        elif stage == 'hierarchical':
            from training.branches.hierarchical_trainer import HierarchicalTrainer
            trainer = HierarchicalTrainer(config)
            
        elif stage == 'ablation':
            # 使用修改版的分层训练器进行消融实验
            from training.branches.hierarchical_trainer import HierarchicalTrainer
            trainer = HierarchicalTrainer(config)
            
        elif stage == 'baseline':
            from training.branches.baseline_trainer import BaselineTrainer
            trainer = BaselineTrainer(config)
            
        else:
            raise ValueError(f"不支持的训练阶段: {stage}")
        
        # 初始化会话
        session_info = trainer.initialize_session(
            enable_trajectory=config['enable_trajectory_recording'],
            enable_tensorboard=config['enable_tensorboard'],
            enable_visualization=config['enable_visualization'],
            enable_pointcloud=config['enable_pointcloud'],
            enable_analysis=config['enable_analysis'],
            enable_rich_display=not args.debug  # 调试模式禁用富文本显示
        )
        
        logger.info(f"✅ {stage} 训练器创建成功")
        logger.info(f"📁 会话目录: {session_info['session_dir']}")
        
        return trainer
        
    except Exception as e:
        logger.error(f"❌ 训练器创建失败: {e}")
        raise


def print_training_summary(stage: str, config: Dict[str, Any], session_dir: str):
    """打印训练摘要"""
    print("\n" + "="*60)
    print(f"🚁 HA-UAV {stage.upper()} 训练启动")
    print("="*60)
    print(f"📁 会话目录: {session_dir}")
    print(f"🎯 训练步数: {config['total_timesteps']:,}")
    print(f"🤖 无人机模型: {config['drone_model']}")
    print(f"⚙️  物理引擎: {config['physics']}")
    print(f"👁️  GUI显示: {'启用' if config.get('gui_training', False) else '禁用'}")
    
    if 'learning_rate' in config:
        print(f"📈 学习率: {config['learning_rate']}")
    if 'batch_size' in config:
        print(f"📦 批次大小: {config['batch_size']}")
    
    print(f"📊 TensorBoard: {'启用' if config['enable_tensorboard'] else '禁用'}")
    print(f"🎬 轨迹记录: {'启用' if config['enable_trajectory_recording'] else '禁用'}")
    
    if stage in ['hierarchical', 'ablation'] and 'foundation_model_path' in config:
        print(f"🏗️  基座模型: {config['foundation_model_path']}")
    
    if 'stage_variant' in config:
        print(f"🔀 变体: {config['stage_variant']}")
    
    print("="*60)


def main():
    """主函数"""
    # 解析参数
    args = parse_arguments()
    
    # 设置日志
    setup_logging(args.log_level, args.debug)
    logger = logging.getLogger(__name__)
    
    try:
        # 构建配置
        logger.info("📝 构建训练配置...")
        config = build_config(args)
        
        # 验证配置
        logger.info("✅ 验证配置有效性...")
        if not validate_config(args.stage, config):
            logger.error("❌ 配置验证失败")
            sys.exit(1)
        
        # 创建训练器
        logger.info("🔧 创建训练器...")
        trainer = create_trainer(args.stage, config, args)
        
        # 打印训练摘要
        session_info = trainer.get_session_paths()
        print_training_summary(args.stage, config, str(session_info.get('session', '')))
        
        # 开始训练
        logger.info("🚀 开始训练...")
        result = trainer.train()
        
        # 训练结果
        if result.success:
            print(f"\n🎉 {args.stage.upper()} 训练成功完成！")
            print(f"📈 最佳奖励: {result.best_reward:.3f}")
            print(f"⏱️  训练时间: {result.training_time:.1f}s")
            print(f"💾 模型保存: {result.model_path}")
            logger.info("✅ 训练成功完成")
        else:
            print(f"\n❌ {args.stage.upper()} 训练失败")
            print(f"🔥 错误信息: {result.error_message}")
            logger.error(f"训练失败: {result.error_message}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.warning("⚠️ 用户中断训练")
        print("\n⚠️ 训练被用户中断")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"❌ 训练过程中发生错误: {e}")
        print(f"\n❌ 训练失败: {e}")
        import traceback
        if args.debug:
            traceback.print_exc()
        sys.exit(1)
        
    finally:
        # 清理资源
        try:
            if 'trainer' in locals():
                trainer.cleanup()
        except:
            pass


if __name__ == "__main__":
    main()
