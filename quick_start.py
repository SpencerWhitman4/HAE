#!/usr/bin/env python3

"""
HA-UAV训练快速启动示例脚本
提供常用训练配置的快捷命令
使用start_training.py中的默认配置
"""

import subprocess
import sys
import importlib.util
from pathlib import Path


def get_default_configs():
    """动态获取start_training.py中的默认配置"""
    try:
        # 导入start_training模块来获取DEFAULT_CONFIGS
        spec = importlib.util.spec_from_file_location("start_training", "start_training.py")
        start_training = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(start_training)
        return start_training.DEFAULT_CONFIGS
    except Exception as e:
        print(f"⚠️ 无法获取默认配置: {e}")
        return {}


def show_config_info(stage):
    """显示指定阶段的配置信息"""
    configs = get_default_configs()
    if stage in configs:
        config = configs[stage]
        print(f"\n📋 {stage.upper()} 阶段默认配置:")
        print("-" * 40)
        
        # 修复格式化问题：分别处理数字和字符串值
        total_timesteps = config.get('total_timesteps', 'N/A')
        if isinstance(total_timesteps, int):
            print(f"  总训练步数: {total_timesteps:,}")
        else:
            print(f"  总训练步数: {total_timesteps}")
        
        print(f"  批处理大小: {config.get('batch_size', 'N/A')}")
        print(f"  学习率: {config.get('learning_rate', 'N/A')}")
        
        evaluation_frequency = config.get('evaluation_frequency', 'N/A')
        if isinstance(evaluation_frequency, int):
            print(f"  评估频率: {evaluation_frequency:,} 步")
        else:
            print(f"  评估频率: {evaluation_frequency} 步")
        
        save_frequency = config.get('save_frequency', 'N/A')
        if isinstance(save_frequency, int):
            print(f"  保存频率: {save_frequency:,} 步")
        else:
            print(f"  保存频率: {save_frequency} 步")
        
        checkpoint_frequency = config.get('checkpoint_frequency', 'N/A')
        if isinstance(checkpoint_frequency, int):
            print(f"  检查点频率: {checkpoint_frequency:,} 步")
        else:
            print(f"  检查点频率: {checkpoint_frequency} 步")
        
        if 'network_arch' in config:
            print(f"  网络架构: {config['network_arch']}")
        print("-" * 40)
    else:
        print(f"⚠️ 未找到 {stage} 阶段的配置")


def run_command(cmd, description, show_config=None):
    """运行命令并显示描述"""
    print(f"\n🚀 {description}")
    
    if show_config:
        show_config_info(show_config)
    
    print(f"📝 命令: {' '.join(cmd)}")
    print("-" * 60)
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"✅ {description} 完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} 失败: {e}")
        return False
    except KeyboardInterrupt:
        print(f"⚠️ {description} 被用户中断")
        return False


def quick_debug_test():
    """快速调试测试 - 使用默认配置"""
    cmd = [
        "python", "start_training.py",
        "--stage", "hierarchical",
        "--timesteps", "50",  # 仅覆盖步数用于快速测试
        "--debug"
    ]
    
    return run_command(cmd, "快速调试测试 (50步)", show_config="hierarchical")


def foundation_training():
    """Foundation训练 - 使用默认配置"""
    cmd = [
        "python", "start_training.py", 
        "--stage", "foundation"
        # 使用所有默认配置，包括默认的100万步
    ]
    
    return run_command(cmd, "Foundation基座模型训练 (使用默认配置)", show_config="foundation")


def foundation_short_training():
    """Foundation短期训练 - 仅修改训练步数"""
    cmd = [
        "python", "start_training.py", 
        "--stage", "foundation",
        "--timesteps", "50000"  # 5万步用于快速测试
    ]
    
    return run_command(cmd, "Foundation基座模型训练 (5万步)", show_config="foundation")


def hierarchical_training():
    """Hierarchical训练 - 使用默认配置"""
    # 查找最新的foundation模型
    logs_dir = Path("logs")
    foundation_models = list(logs_dir.glob("train_foundation_*/Model/best_model_foundation.zip"))
    
    cmd = [
        "python", "start_training.py",
        "--stage", "hierarchical"
        # 使用默认的20万步配置
    ]
    
    if foundation_models:
        latest_model = max(foundation_models, key=lambda p: p.parent.parent.stat().st_mtime)
        cmd.extend(["--foundation-model", str(latest_model)])
        print(f"📦 将使用Foundation模型: {latest_model}")
    else:
        print("⚠️ 未找到Foundation模型，将从头开始训练")
    
    return run_command(cmd, "Hierarchical分层训练 (使用默认配置)", show_config="hierarchical")


def baseline_comparison():
    """Baseline对比训练 - 使用默认配置"""
    algorithms = ["ppo", "sac"]
    
    for alg in algorithms:
        cmd = [
            "python", "start_training.py",
            "--stage", "baseline",
            "--variant", alg
            # 使用默认的10万步配置
        ]
        
        if not run_command(cmd, f"Baseline {alg.upper()} 训练 (使用默认配置)", show_config="baseline"):
            return False
    
    return True


def ablation_study():
    """消融研究 - 使用默认配置"""
    cmd = [
        "python", "start_training.py",
        "--stage", "ablation"
        # 使用默认配置
    ]
    
    return run_command(cmd, "消融研究训练 (使用默认配置)", show_config="ablation")


def gui_demo():
    """GUI演示 - 最小配置"""
    cmd = [
        "python", "start_training.py",
        "--stage", "foundation",
        "--timesteps", "1000",  # 仅1000步用于演示
        "--gui",
        "--debug"
    ]
    
    return run_command(cmd, "GUI可视化演示 (1000步)", show_config="foundation")


def main():
    """主菜单"""
    print("🚁 HA-UAV训练系统快速启动")
    print("=" * 60)
    
    # 动态显示可用配置
    configs = get_default_configs()
    if configs:
        print("📋 可用训练阶段及其默认配置:")
        for stage, config in configs.items():
            timesteps = config.get('total_timesteps', 'N/A')
            arch = config.get('network_arch', 'N/A')
            if isinstance(timesteps, int):
                print(f"  {stage.upper()}: {timesteps:,} 步, {arch}")
            else:
                print(f"  {stage.upper()}: {timesteps} 步, {arch}")
        print("-" * 60)
    
    print("1. 快速调试测试 (50步)")
    print("2. Foundation训练 (使用默认配置)")
    print("3. Foundation短期训练 (5万步)")
    print("4. Hierarchical训练 (使用默认配置)")
    print("5. Baseline对比训练 (使用默认配置)")
    print("6. 消融研究 (使用默认配置)")
    print("7. GUI可视化演示 (1000步)")
    print("8. 完整训练流程")
    print("9. 显示所有默认配置")
    print("0. 退出")
    print("=" * 60)
    print("💡 所有训练都使用start_training.py中的默认配置")
    print("   只有训练步数等必要参数会被覆盖")
    
    while True:
        try:
            choice = input("\n请选择训练模式 (0-9): ").strip()
            
            if choice == "0":
                print("👋 退出程序")
                break
                
            elif choice == "1":
                quick_debug_test()
                
            elif choice == "2":
                foundation_training()
                
            elif choice == "3":
                foundation_short_training()
                
            elif choice == "4":
                hierarchical_training()
                
            elif choice == "5":
                baseline_comparison()
                
            elif choice == "6":
                ablation_study()
                
            elif choice == "7":
                print("⚠️ 确保您的系统支持GUI显示")
                if input("继续？ (y/N): ").lower() == 'y':
                    gui_demo()
                
            elif choice == "8":
                print("🔄 开始完整训练流程...")
                print("阶段1: Foundation训练")
                if foundation_short_training():  # 使用短期训练节省时间
                    print("阶段2: Hierarchical训练")
                    if hierarchical_training():
                        print("阶段3: Baseline对比")
                        baseline_comparison()
                        
            elif choice == "9":
                print("\n📋 所有默认配置详情:")
                print("=" * 50)
                configs = get_default_configs()
                for stage, config in configs.items():
                    show_config_info(stage)
                    print()
                
            else:
                print("❌ 无效选择，请输入0-9")
                
        except KeyboardInterrupt:
            print("\n👋 退出程序")
            break
        except Exception as e:
            print(f"❌ 发生错误: {e}")


if __name__ == "__main__":
    # 检查start_training.py是否存在
    if not Path("start_training.py").exists():
        print("❌ 未找到start_training.py，请确保在正确的目录下运行")
        sys.exit(1)
    
    main()
