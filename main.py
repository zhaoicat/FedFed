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
# add the FedML root directory to the python path

from utils.logger import logging_config
from configs import get_cfg

from algorithms_standalone.fedavg.FedAVGManager import FedAVGManager
from algorithms_standalone.fedprox.FedProxManager import FedProxManager
from algorithms_standalone.pfedme.PFedMeManager import PFedMeManager
from algorithms_standalone.basePS.basePSmanager import BasePSManager

from utils.set import *

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    parser.add_argument("--config_file", default=None, type=str)
    parser.add_argument("--algorithm", default="FedAvg", type=str, 
                       help="Algorithm to run: FedAvg, FedProx, PFedMe, FedNova, FedFed")
    parser.add_argument("--quick_test", action="store_true", 
                       help="Enable quick test mode")
    parser.add_argument("--full_validation", action="store_true", 
                       help="Enable full validation")
    parser.add_argument("--output_dir", default="./results", type=str,
                       help="Directory to save results")
    parser.add_argument("opts", help="Modify config options using the command-line",
                        default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    return args


class TrainingProgressTracker:
    """训练进度跟踪器"""
    
    def __init__(self, total_rounds, output_dir, algorithm_name):
        self.total_rounds = total_rounds
        self.output_dir = output_dir
        self.algorithm_name = algorithm_name
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
        
        logging.info(f"=== 开始训练 {algorithm_name} 算法 ===")
        logging.info(f"总轮数: {total_rounds}")
        logging.info(f"结果保存目录: {output_dir}")

    def update_round(self, round_idx):
        """更新当前轮次"""
        self.current_round = round_idx
        progress = (round_idx + 1) / self.total_rounds * 100
        logging.info(f"🔄 开始第 {round_idx + 1}/{self.total_rounds} 轮训练 ({progress:.1f}%)")

    def record_round_results(self, round_idx, train_loss, train_acc, test_loss, test_acc):
        """记录轮次结果"""
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        time_per_round = elapsed_time / (round_idx + 1)
        remaining_time = time_per_round * (self.total_rounds - round_idx - 1)
        
        # 记录历史
        self.train_history['rounds'].append(round_idx + 1)
        self.train_history['train_loss'].append(train_loss)
        self.train_history['train_acc'].append(train_acc)
        self.train_history['test_loss'].append(test_loss)
        self.train_history['test_acc'].append(test_acc)
        self.train_history['time_per_round'].append(time_per_round)
        
        # 显示进度
        logging.info(f"📊 第 {round_idx + 1} 轮结果:")
        logging.info(f"   🎯 训练 - 损失: {train_loss:.4f}, 精度: {train_acc:.4f}")
        logging.info(f"   🎯 测试 - 损失: {test_loss:.4f}, 精度: {test_acc:.4f}")
        logging.info(f"   ⏱️  已用时间: {elapsed_time/60:.1f}分钟, 预计剩余: {remaining_time/60:.1f}分钟")
        logging.info("-" * 60)

    def finish_training(self):
        """完成训练"""
        total_time = time.time() - self.start_time
        logging.info(f"🎉 训练完成！总用时: {total_time/60:.1f}分钟")

    def save_results(self):
        """保存训练结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存训练历史到JSON文件
        results_file = os.path.join(
            self.output_dir, f"{self.algorithm_name}_results_{timestamp}.json")
        with open(results_file, 'w') as f:
            json.dump(self.train_history, f, indent=2)
        
        # 绘制训练曲线
        self._plot_training_curves(timestamp)
        
        # 保存最终结果摘要
        self._save_summary(timestamp)
        
        logging.info(f"训练结果已保存到: {self.output_dir}")

    def _plot_training_curves(self, timestamp):
        """绘制训练曲线"""
        plt.figure(figsize=(15, 10))
        
        # 损失曲线
        plt.subplot(2, 2, 1)
        plt.plot(self.train_history['rounds'], self.train_history['train_loss'], 
                'b-', label='训练损失', linewidth=2)
        plt.plot(self.train_history['rounds'], self.train_history['test_loss'], 
                'r-', label='测试损失', linewidth=2)
        plt.xlabel('轮次')
        plt.ylabel('损失')
        plt.title(f'{self.algorithm_name} - 损失曲线')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 精度曲线
        plt.subplot(2, 2, 2)
        plt.plot(self.train_history['rounds'], self.train_history['train_acc'], 
                'b-', label='训练精度', linewidth=2)
        plt.plot(self.train_history['rounds'], self.train_history['test_acc'], 
                'r-', label='测试精度', linewidth=2)
        plt.xlabel('轮次')
        plt.ylabel('精度')
        plt.title(f'{self.algorithm_name} - 精度曲线')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 每轮时间
        plt.subplot(2, 2, 3)
        plt.plot(self.train_history['rounds'], self.train_history['time_per_round'], 
                'g-', linewidth=2)
        plt.xlabel('轮次')
        plt.ylabel('时间 (秒)')
        plt.title(f'{self.algorithm_name} - 每轮训练时间')
        plt.grid(True, alpha=0.3)
        
        # 精度对比
        plt.subplot(2, 2, 4)
        final_train_acc = self.train_history['train_acc'][-1] if self.train_history['train_acc'] else 0
        final_test_acc = self.train_history['test_acc'][-1] if self.train_history['test_acc'] else 0
        plt.bar(['训练精度', '测试精度'], [final_train_acc, final_test_acc], 
               color=['blue', 'red'], alpha=0.7)
        plt.ylabel('精度')
        plt.title(f'{self.algorithm_name} - 最终精度对比')
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
            'timestamp': timestamp,
            'total_rounds': self.total_rounds,
            'total_time_minutes': total_time / 60,
            'final_train_accuracy': self.train_history['train_acc'][-1] if self.train_history['train_acc'] else 0,
            'final_test_accuracy': self.train_history['test_acc'][-1] if self.train_history['test_acc'] else 0,
            'best_test_accuracy': max(self.train_history['test_acc']) if self.train_history['test_acc'] else 0,
            'final_train_loss': self.train_history['train_loss'][-1] if self.train_history['train_loss'] else 0,
            'final_test_loss': self.train_history['test_loss'][-1] if self.train_history['test_loss'] else 0,
            'avg_time_per_round': np.mean(self.train_history['time_per_round']) if self.train_history['time_per_round'] else 0
        }
        
        summary_file = os.path.join(
            self.output_dir, f"{self.algorithm_name}_summary_{timestamp}.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)


def run_algorithm(cfg, device, algorithm_name, output_dir):
    """运行指定的联邦学习算法"""
    
    # 创建进度跟踪器
    tracker = TrainingProgressTracker(
        cfg.comm_round, output_dir, algorithm_name)
    
    # 标准化算法名称（支持小写输入）
    algorithm_name_lower = algorithm_name.lower()
    
    # 选择算法管理器
    if algorithm_name_lower in ['fedavg', 'FedAvg']:
        manager = FedAVGManager(device, cfg)
    elif algorithm_name_lower in ['fedprox', 'FedProx']:
        manager = FedProxManager(device, cfg)
    elif algorithm_name_lower in ['pfedme', 'PFedMe']:
        manager = PFedMeManager(device, cfg)
    elif algorithm_name_lower in ['fedfed', 'FedFed']:
        manager = FedAVGManager(device, cfg)  # FedFed基于FedAvg实现
    else:
        raise NotImplementedError(f"Algorithm {algorithm_name} not supported. "
                                f"Supported algorithms: FedAvg, FedProx, PFedMe, FedFed")
    
    # 尝试使用带进度跟踪的训练方法
    try:
        if hasattr(manager, 'train_with_progress'):
            manager.train_with_progress(tracker)
        else:
            logging.warning("使用原始训练方法，无法显示详细进度")
            manager.train()
    except Exception as e:
        logging.error(f"训练过程中出现错误: {e}")
        raise
    
    # 完成训练并生成结果
    tracker.finish_training()
    
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
    device = torch.device("cuda:" + str(cfg.gpu_index) if torch.cuda.is_available() else "cpu")
    
    # 配置wandb
    if cfg.record_tool == 'wandb' and cfg.wandb_record:
        import wandb
        wandb.init(config=args, name=f'{cfg.algorithm}_test',
                   project='FedFed_Yummly28k')
    else: 
        os.environ['WANDB_MODE'] = 'dryrun'

    # 运行算法
    tracker = run_algorithm(cfg, device, cfg.algorithm, args.output_dir)
    
    logging.info("=== 训练完成 ===")
    logging.info(f"最终测试精度: {tracker.train_history['test_acc'][-1] if tracker.train_history['test_acc'] else 0:.4f}")
    logging.info(f"总训练时间: {(time.time() - tracker.start_time)/60:.1f}分钟")








