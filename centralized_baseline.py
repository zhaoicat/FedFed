#!/usr/bin/env python3
"""
单机集中式学习基线测试脚本
用于比较联邦学习与集中式学习的性能差异
"""

import argparse
import logging
import os
import sys
import time
import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# 添加项目路径
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from configs import get_cfg
from data_preprocessing.loader import Data_Loader
from model.build import create_model
from utils.set import set_random_seed


class CentralizedTrainer:
    """集中式训练器"""
    
    def __init__(self, cfg, device, output_dir):
        self.cfg = cfg
        self.device = device
        self.output_dir = output_dir
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 训练历史
        self.train_history = {
            'epochs': [],
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': [],
            'time_per_epoch': []
        }
        
        self.start_time = time.time()
        
        logging.info(f"集中式训练器初始化完成")
        logging.info(f"设备: {device}")
        logging.info(f"输出目录: {output_dir}")

    def setup_data(self):
        """设置数据加载器"""
        logging.info("设置数据加载器...")
        
        # 创建数据加载器
        data_loader = Data_Loader(
            dataset=self.cfg.dataset,
            datadir=self.cfg.data_dir,
            partition_method="homo",  # 集中式学习使用所有数据
            client_number=1,
            batch_size=self.cfg.batch_size,
            num_workers=0,
            resize=self.cfg.dataset_load_image_size,
            augmentation=self.cfg.dataset_aug
        )
        
        # 加载集中式数据
        data_loader.load_centralized_data()
        
        self.train_dataloader = data_loader.train_dl
        self.test_dataloader = data_loader.test_dl
        self.train_data_num = data_loader.train_data_num
        self.test_data_num = data_loader.test_data_num
        self.class_num = data_loader.class_num
        
        logging.info(f"训练样本数: {self.train_data_num}")
        logging.info(f"测试样本数: {self.test_data_num}")
        logging.info(f"类别数: {self.class_num}")

    def setup_model(self):
        """设置模型"""
        logging.info("设置模型...")
        
        # 创建模型
        self.model = create_model(
            self.cfg, 
            model_name=self.cfg.model,
            output_dim=self.cfg.model_output_dim,
            device=self.device
        )
        
        # 设置损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.cfg.lr,
            momentum=self.cfg.momentum,
            weight_decay=self.cfg.wd
        )
        
        # 学习率调度器
        if hasattr(self.cfg, 'sched') and self.cfg.sched != 'no':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.cfg.step_size,
                gamma=self.cfg.lr_decay_rate
            )
        else:
            self.scheduler = None
        
        logging.info(f"模型: {self.cfg.model}")
        logging.info(f"学习率: {self.cfg.lr}")
        logging.info(f"优化器: SGD")

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        epoch_start_time = time.time()
        
        for batch_idx, (data, target) in enumerate(self.train_dataloader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # 显示进度
            if batch_idx % 100 == 0:
                logging.info(f'Epoch {epoch}, Batch {batch_idx}/{len(self.train_dataloader)}, '
                           f'Loss: {loss.item():.6f}')
        
        epoch_time = time.time() - epoch_start_time
        avg_loss = total_loss / len(self.train_dataloader)
        accuracy = correct / total
        
        return avg_loss, accuracy, epoch_time

    def test(self):
        """测试模型"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.test_dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(self.test_dataloader)
        accuracy = correct / total
        
        return avg_loss, accuracy

    def train(self, epochs):
        """训练模型"""
        logging.info(f"开始训练，总epochs: {epochs}")
        
        best_test_acc = 0.0
        
        for epoch in range(1, epochs + 1):
            # 训练
            train_loss, train_acc, epoch_time = self.train_epoch(epoch)
            
            # 测试
            test_loss, test_acc = self.test()
            
            # 更新学习率
            if self.scheduler:
                self.scheduler.step()
            
            # 记录历史
            self.train_history['epochs'].append(epoch)
            self.train_history['train_loss'].append(train_loss)
            self.train_history['train_acc'].append(train_acc)
            self.train_history['test_loss'].append(test_loss)
            self.train_history['test_acc'].append(test_acc)
            self.train_history['time_per_epoch'].append(epoch_time)
            
            # 更新最佳精度
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                # 保存最佳模型
                torch.save(self.model.state_dict(), 
                          os.path.join(self.output_dir, 'best_model.pth'))
            
            # 显示进度
            elapsed_time = time.time() - self.start_time
            remaining_time = (elapsed_time / epoch) * (epochs - epoch)
            
            logging.info(f"Epoch {epoch}/{epochs} - "
                        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                        f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, "
                        f"Best Test Acc: {best_test_acc:.4f}")
            logging.info(f"Epoch Time: {epoch_time:.1f}s, "
                        f"Elapsed: {elapsed_time/60:.1f}min, "
                        f"Remaining: {remaining_time/60:.1f}min")
            logging.info("-" * 80)
        
        total_time = time.time() - self.start_time
        logging.info(f"训练完成！总时间: {total_time/60:.1f}分钟")
        logging.info(f"最佳测试精度: {best_test_acc:.4f}")
        
        return best_test_acc

    def save_results(self):
        """保存训练结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存训练历史
        results_file = os.path.join(self.output_dir, f"centralized_results_{timestamp}.json")
        with open(results_file, 'w') as f:
            json.dump(self.train_history, f, indent=2)
        
        # 绘制训练曲线
        self._plot_training_curves(timestamp)
        
        # 保存摘要
        self._save_summary(timestamp)
        
        logging.info(f"结果已保存到: {self.output_dir}")

    def _plot_training_curves(self, timestamp):
        """绘制训练曲线"""
        plt.figure(figsize=(15, 10))
        
        # 损失曲线
        plt.subplot(2, 2, 1)
        plt.plot(self.train_history['epochs'], self.train_history['train_loss'], 
                'b-', label='训练损失', linewidth=2)
        plt.plot(self.train_history['epochs'], self.train_history['test_loss'], 
                'r-', label='测试损失', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('损失')
        plt.title('集中式学习 - 损失曲线')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 精度曲线
        plt.subplot(2, 2, 2)
        plt.plot(self.train_history['epochs'], self.train_history['train_acc'], 
                'b-', label='训练精度', linewidth=2)
        plt.plot(self.train_history['epochs'], self.train_history['test_acc'], 
                'r-', label='测试精度', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('精度')
        plt.title('集中式学习 - 精度曲线')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 每epoch时间
        plt.subplot(2, 2, 3)
        plt.plot(self.train_history['epochs'], self.train_history['time_per_epoch'], 
                'g-', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('时间 (秒)')
        plt.title('集中式学习 - 每Epoch训练时间')
        plt.grid(True, alpha=0.3)
        
        # 最终精度对比
        plt.subplot(2, 2, 4)
        final_train_acc = self.train_history['train_acc'][-1]
        final_test_acc = self.train_history['test_acc'][-1]
        best_test_acc = max(self.train_history['test_acc'])
        
        plt.bar(['训练精度', '最终测试精度', '最佳测试精度'], 
               [final_train_acc, final_test_acc, best_test_acc], 
               color=['blue', 'red', 'green'], alpha=0.7)
        plt.ylabel('精度')
        plt.title('集中式学习 - 精度对比')
        plt.ylim(0, 1)
        
        plt.tight_layout()
        plot_file = os.path.join(self.output_dir, f"centralized_curves_{timestamp}.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

    def _save_summary(self, timestamp):
        """保存结果摘要"""
        total_time = time.time() - self.start_time
        
        summary = {
            'algorithm': 'Centralized',
            'timestamp': timestamp,
            'total_epochs': len(self.train_history['epochs']),
            'total_time_minutes': total_time / 60,
            'final_train_accuracy': self.train_history['train_acc'][-1],
            'final_test_accuracy': self.train_history['test_acc'][-1],
            'best_test_accuracy': max(self.train_history['test_acc']),
            'final_train_loss': self.train_history['train_loss'][-1],
            'final_test_loss': self.train_history['test_loss'][-1],
            'avg_time_per_epoch': np.mean(self.train_history['time_per_epoch']),
            'train_data_num': self.train_data_num,
            'test_data_num': self.test_data_num,
            'class_num': self.class_num
        }
        
        summary_file = os.path.join(self.output_dir, f"centralized_summary_{timestamp}.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="集中式学习基线测试")
    parser.add_argument("--config_file", default="config_quick_test.yaml", type=str,
                       help="配置文件路径")
    parser.add_argument("--output_dir", default="./centralized_results", type=str,
                       help="结果输出目录")
    parser.add_argument("--epochs", default=10, type=int,
                       help="训练epochs数")
    parser.add_argument("--quick_test", action="store_true",
                       help="启用快速测试模式")
    
    args = parser.parse_args()
    
    # 配置日志
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'centralized.log')),
            logging.StreamHandler()
        ]
    )
    
    # 设置配置
    cfg = get_cfg()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    
    # 快速测试模式
    if args.quick_test:
        args.epochs = 3
        cfg.batch_size = 64
        logging.info("启用快速测试模式")
    
    # 设置随机种子
    set_random_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True
    
    # 设置设备
    device = torch.device("cuda:" + str(cfg.gpu_index) if torch.cuda.is_available() else "cpu")
    
    logging.info("=== 开始集中式学习基线测试 ===")
    logging.info(f"配置文件: {args.config_file}")
    logging.info(f"训练epochs: {args.epochs}")
    logging.info(f"设备: {device}")
    
    # 创建训练器
    trainer = CentralizedTrainer(cfg, device, args.output_dir)
    
    # 设置数据和模型
    trainer.setup_data()
    trainer.setup_model()
    
    # 训练
    best_acc = trainer.train(args.epochs)
    
    # 保存结果
    trainer.save_results()
    
    logging.info("=== 集中式学习基线测试完成 ===")
    logging.info(f"最佳测试精度: {best_acc:.4f}")


if __name__ == "__main__":
    main() 