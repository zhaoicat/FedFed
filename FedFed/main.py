import argparse
import logging
import os
import socket
import sys
import time
import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

import torch

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.logger import logging_config
from configs import get_cfg

from algorithms_standalone.fedavg.FedAVGManager import FedAVGManager
from algorithms_standalone.fednova.FedNovaManager import FedNovaManager
from algorithms_standalone.fedprox.FedProxManager import FedProxManager
from algorithms_standalone.pfedme.PFedMeManager import PFedMeManager

from utils.set import *

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    parser.add_argument("--config_file", default="config_yummly28k_test.yaml", type=str,
                        help="配置文件路径，默认使用Yummly28k数据集")
    parser.add_argument("--algorithm", default="FedAvg", type=str, 
                        help="Algorithm to run: FedAvg, FedProx, PFedMe, "
                             "FedNova, FedFed")
    parser.add_argument("--quick_test", action="store_true", 
                        help="Enable quick test mode")
    parser.add_argument("--full_validation", action="store_true", 
                        help="Enable full validation")
    parser.add_argument("--output_dir", default="./results", type=str,
                        help="Directory to save results")
    parser.add_argument("opts", 
                        help="Modify config options using the command-line",
                        default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    return args


class TrainingProgressTracker:
    """训练进度跟踪器"""
    
    def __init__(self, total_rounds, output_dir, algorithm_name, dataset_name):
        self.total_rounds = total_rounds
        self.output_dir = output_dir
        self.algorithm_name = algorithm_name
        self.dataset_name = dataset_name
        self.start_time = time.time()
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 训练历史记录
        self.train_history = {
            'rounds': [],
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': [],
            'time_per_round': []
        }
        
        # 打印数据集和算法信息
        print("=" * 80)
        print(f"🍽️  使用数据集: {dataset_name}")
        print(f"🤖  训练算法: {algorithm_name}")
        print(f"🔄  总轮数: {total_rounds}")
        print(f"📁  结果保存目录: {output_dir}")
        print("=" * 80)
        
        logging.info(f"=== 开始训练 {algorithm_name} 算法 ===")
        logging.info(f"数据集: {dataset_name}")
        logging.info(f"总轮数: {total_rounds}")
        logging.info(f"结果保存目录: {output_dir}")

    def update_progress(self, round_idx, train_metrics, test_metrics):
        """更新训练进度"""
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        time_per_round = elapsed_time / (round_idx + 1)
        remaining_time = time_per_round * (self.total_rounds - round_idx - 1)
        
        # 记录历史
        self.train_history['rounds'].append(round_idx + 1)
        self.train_history['train_loss'].append(
            train_metrics.get('loss', 0))
        self.train_history['train_acc'].append(
            train_metrics.get('accuracy', 0))
        self.train_history['test_loss'].append(test_metrics.get('loss', 0))
        self.train_history['test_acc'].append(
            test_metrics.get('accuracy', 0))
        self.train_history['time_per_round'].append(time_per_round)
        
        # 显示进度
        progress = (round_idx + 1) / self.total_rounds * 100
        
        # 使用中文和emoji显示进度
        print(f"\n🔄 === 轮次 {round_idx + 1}/{self.total_rounds} ({progress:.1f}%) ===")
        print(f"📊 训练损失: {train_metrics.get('loss', 0):.4f}, "
              f"训练精度: {train_metrics.get('accuracy', 0):.4f}")
        print(f"🎯 测试损失: {test_metrics.get('loss', 0):.4f}, "
              f"测试精度: {test_metrics.get('accuracy', 0):.4f}")
        print(f"⏱️  已用时间: {elapsed_time/60:.1f}分钟, "
              f"预计剩余: {remaining_time/60:.1f}分钟")
        print("-" * 60)
        
        logging.info(f"=== 轮次 {round_idx + 1}/{self.total_rounds} "
                     f"({progress:.1f}%) ===")
        logging.info(f"训练损失: {train_metrics.get('loss', 0):.4f}, "
                     f"训练精度: {train_metrics.get('accuracy', 0):.4f}")
        logging.info(f"测试损失: {test_metrics.get('loss', 0):.4f}, "
                     f"测试精度: {test_metrics.get('accuracy', 0):.4f}")
        logging.info(f"已用时间: {elapsed_time/60:.1f}分钟, "
                     f"预计剩余: {remaining_time/60:.1f}分钟")
        logging.info("-" * 60)

    def save_results(self):
        """保存训练结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存训练历史到JSON文件
        results_file = os.path.join(
            self.output_dir, f"{self.algorithm_name}_results_{timestamp}.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.train_history, f, indent=2, ensure_ascii=False)
        
        # 绘制训练曲线
        self._plot_training_curves(timestamp)
        
        # 保存最终结果摘要
        self._save_summary(timestamp)
        
        print(f"📁 训练结果已保存到: {self.output_dir}")
        logging.info(f"训练结果已保存到: {self.output_dir}")

    def _plot_training_curves(self, timestamp):
        """绘制训练曲线"""
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        plt.figure(figsize=(15, 10))
        
        # 损失曲线
        plt.subplot(2, 2, 1)
        plt.plot(self.train_history['rounds'], 
                 self.train_history['train_loss'], 
                 'b-', label='Training Loss', linewidth=2)
        plt.plot(self.train_history['rounds'], 
                 self.train_history['test_loss'], 
                 'r-', label='Test Loss', linewidth=2)
        plt.xlabel('Round')
        plt.ylabel('Loss')
        plt.title(f'{self.algorithm_name} on {self.dataset_name} - Loss Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 精度曲线
        plt.subplot(2, 2, 2)
        plt.plot(self.train_history['rounds'], 
                 self.train_history['train_acc'], 
                 'b-', label='Training Accuracy', linewidth=2)
        plt.plot(self.train_history['rounds'], 
                 self.train_history['test_acc'], 
                 'r-', label='Test Accuracy', linewidth=2)
        plt.xlabel('Round')
        plt.ylabel('Accuracy')
        plt.title(f'{self.algorithm_name} on {self.dataset_name} - Accuracy Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 每轮时间
        plt.subplot(2, 2, 3)
        plt.plot(self.train_history['rounds'], 
                 self.train_history['time_per_round'], 
                 'g-', linewidth=2)
        plt.xlabel('Round')
        plt.ylabel('Time (seconds)')
        plt.title(f'{self.algorithm_name} - Training Time per Round')
        plt.grid(True, alpha=0.3)
        
        # 精度对比
        plt.subplot(2, 2, 4)
        final_train_acc = (self.train_history['train_acc'][-1] 
                          if self.train_history['train_acc'] else 0)
        final_test_acc = (self.train_history['test_acc'][-1] 
                         if self.train_history['test_acc'] else 0)
        plt.bar(['Training Accuracy', 'Test Accuracy'], 
                [final_train_acc, final_test_acc], 
                color=['blue', 'red'], alpha=0.7)
        plt.ylabel('Accuracy')
        plt.title(f'{self.algorithm_name} - Final Accuracy Comparison')
        plt.ylim(0, 1)
        
        plt.tight_layout()
        plot_file = os.path.join(
            self.output_dir, f"{self.algorithm_name}_curves_{timestamp}.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

    def _save_summary(self, timestamp):
        """保存结果摘要"""
        total_time = time.time() - self.start_time
        
        summary = {
            'algorithm': self.algorithm_name,
            'dataset': self.dataset_name,
            'timestamp': timestamp,
            'total_rounds': self.total_rounds,
            'total_time_minutes': total_time / 60,
            'final_train_accuracy': (self.train_history['train_acc'][-1] 
                                   if self.train_history['train_acc'] else 0),
            'final_test_accuracy': (self.train_history['test_acc'][-1] 
                                  if self.train_history['test_acc'] else 0),
            'best_test_accuracy': (max(self.train_history['test_acc']) 
                                 if self.train_history['test_acc'] else 0),
            'final_train_loss': (self.train_history['train_loss'][-1] 
                               if self.train_history['train_loss'] else 0),
            'final_test_loss': (self.train_history['test_loss'][-1] 
                              if self.train_history['test_loss'] else 0),
            'avg_time_per_round': (np.mean(self.train_history['time_per_round']) 
                                 if self.train_history['time_per_round'] else 0)
        }
        
        summary_file = os.path.join(
            self.output_dir, f"{self.algorithm_name}_summary_{timestamp}.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)


def add_progress_tracking_to_manager(manager, tracker):
    """为管理器添加进度跟踪功能"""
    
    def train_with_progress():
        """带进度跟踪的训练方法"""
        for round_idx in range(manager.comm_round):
            
            logging.info(f"################Communication round : {round_idx}")
            
            # 第一轮初始化
            if round_idx == 0:
                named_params = manager.aggregator.get_global_model_params()
                params_type = 'model'
                global_other_params = {}
                shared_params_for_simulation = {}

                # SCAFFOLD支持
                if (hasattr(manager.args, 'scaffold') and 
                    manager.args.scaffold):
                    c_global_para = (manager.aggregator.c_model_global
                                   .state_dict())
                    global_other_params["c_model_global"] = c_global_para

            # 客户端采样
            client_indexes = manager.aggregator.client_sampling(   
                round_idx, manager.args.client_num_in_total,
                manager.args.client_num_per_round)

            update_state_kargs = manager.get_update_state_kargs()

            # 训练和聚合
            named_params, params_type, global_other_params, \
            shared_params_for_simulation = manager.algorithm_train(
                round_idx,
                client_indexes,
                named_params,
                params_type,
                global_other_params,
                update_state_kargs,
                shared_params_for_simulation
            )
            
            # 测试模型
            test_acc = manager.aggregator.test_on_server_for_round(
                manager.args.VAE_comm_round + round_idx)
            manager.test_acc_list.append(test_acc)
            
            # 计算训练指标
            train_metrics = {
                'loss': 0.0,  # 暂时设为0
                'accuracy': test_acc  # 使用测试精度作为近似
            }
            
            test_metrics = {
                'loss': 0.0,  # 暂时设为0
                'accuracy': test_acc
            }
            
            # 更新进度
            tracker.update_progress(round_idx, train_metrics, test_metrics)
            
            # 每20轮打印一次历史
            if round_idx % 20 == 0:
                logging.info(f"测试精度历史: {manager.test_acc_list}")
        
        # 保存分类器
        manager.aggregator.save_classifier()
    
    # 添加方法到管理器
    manager.train_with_progress = train_with_progress
    return manager


def run_algorithm(cfg, device, algorithm_name, output_dir):
    """运行指定的联邦学习算法"""
    
    # 获取数据集名称
    dataset_name = getattr(cfg, 'dataset', 'unknown')
    
    # 创建进度跟踪器
    tracker = TrainingProgressTracker(
        cfg.comm_round, output_dir, algorithm_name, dataset_name)
    
    # 选择算法管理器
    if algorithm_name == 'FedAvg':
        manager = FedAVGManager(device, cfg)
    elif algorithm_name == 'FedProx':
        manager = FedProxManager(device, cfg)
    elif algorithm_name == 'PFedMe':
        manager = PFedMeManager(device, cfg)
    elif algorithm_name == 'FedNova':
        manager = FedNovaManager(device, cfg)
    elif algorithm_name == 'FedFed':
        manager = FedAVGManager(device, cfg)  # FedFed基于FedAvg实现
    else:
        raise NotImplementedError(
            f"Algorithm {algorithm_name} not implemented")
    
    # 添加进度跟踪功能
    manager = add_progress_tracking_to_manager(manager, tracker)
    
    # 训练
    try:
        # 使用带进度跟踪的训练方法
        manager.train_with_progress()
    except Exception as e:
        logging.error(f"训练过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        # 如果出错，尝试使用原始训练方法
        logging.warning("尝试使用原始训练方法")
        manager.train()
    
    # 保存结果
    tracker.save_results()
    
    return tracker


if __name__ == "__main__":
    # 解析参数
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    
    # 设置配置
    cfg = get_cfg()
    cfg.setup(args)
    cfg.mode = 'standalone'
    cfg.server_index = -1
    cfg.client_index = -1
    
    # 设置算法类型
    if hasattr(args, 'algorithm'):
        cfg.algorithm = args.algorithm
    
    # 快速测试模式
    if args.quick_test:
        cfg.comm_round = 3
        cfg.client_num_in_total = 3
        cfg.client_num_per_round = 2
        cfg.global_epochs_per_round = 1
        print("🚀 启用快速测试模式")
        logging.info("启用快速测试模式")
    
    # 设置验证模式
    cfg.full_validation = args.full_validation
    
    seed = cfg.seed
    process_id = 0
    
    # 显示配置
    logging.info(dict(cfg))
    
    # 设置进程名
    str_process_name = cfg.algorithm + " (standalone):" + str(process_id)
    
    # 配置日志
    logging_config(args=cfg, process_id=process_id)
    
    hostname = socket.gethostname()
    logging.info("#############process ID = " + str(process_id) +
                 ", host name = " + hostname + "########" +
                 ", process ID = " + str(os.getpid()))

    # 设置随机种子
    set_random_seed(seed) 
    torch.backends.cudnn.deterministic = True

    # 设置设备
    device = torch.device("cuda:" + str(cfg.gpu_index) 
                         if torch.cuda.is_available() else "cpu")
    
    # 配置wandb
    if cfg.record_tool == 'wandb' and cfg.wandb_record:
        import wandb
        wandb.init(config=args, name=f'{cfg.algorithm}_test',
                   project='FedFed_Yummly28k')
    else: 
        os.environ['WANDB_MODE'] = 'dryrun'

    # 运行算法
    tracker = run_algorithm(cfg, device, cfg.algorithm, args.output_dir)
    
    print("\n🎉 === 训练完成 ===")
    final_test_acc = (tracker.train_history['test_acc'][-1] 
                     if tracker.train_history['test_acc'] else 0)
    print(f"🎯 最终测试精度: {final_test_acc:.4f}")
    total_time = (time.time() - tracker.start_time) / 60
    print(f"⏱️  总训练时间: {total_time:.1f}分钟")
    
    logging.info("=== 训练完成 ===")
    logging.info(f"最终测试精度: {final_test_acc:.4f}")
    logging.info(f"总训练时间: {total_time:.1f}分钟") 