#!/usr/bin/env python3
"""
联邦学习算法比较脚本
比较FedAvg、FedProx、PFedMe和FedFed算法在Yummly28k数据集上的性能
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
import pandas as pd

# 添加项目路径
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from main import run_algorithm, TrainingProgressTracker
from configs import get_cfg
from utils.logger import logging_config
from utils.set import set_random_seed
import torch


class AlgorithmComparator:
    """算法比较器"""
    
    def __init__(self, output_dir, config_file=None):
        self.output_dir = output_dir
        self.config_file = config_file
        self.results = {}
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 要比较的算法列表
        self.algorithms = ['FedAvg', 'FedProx', 'PFedMe', 'FedFed']
        
        logging.info(f"算法比较器初始化完成，输出目录: {output_dir}")
        logging.info(f"将比较的算法: {', '.join(self.algorithms)}")

    def run_comparison(self, quick_test=False, full_validation=False):
        """运行算法比较"""
        
        logging.info("=== 开始算法比较 ===")
        start_time = time.time()
        
        for algorithm in self.algorithms:
            logging.info(f"\n{'='*60}")
            logging.info(f"开始运行算法: {algorithm}")
            logging.info(f"{'='*60}")
            
            try:
                # 设置配置
                cfg = self._setup_config(algorithm, quick_test, full_validation)
                
                # 设置设备
                device = torch.device(
                    "cuda:" + str(cfg.gpu_index) 
                    if torch.cuda.is_available() else "cpu")
                
                # 运行算法
                algorithm_output_dir = os.path.join(self.output_dir, algorithm)
                tracker = run_algorithm(cfg, device, algorithm, algorithm_output_dir)
                
                # 保存结果
                self.results[algorithm] = {
                    'tracker': tracker,
                    'final_test_acc': tracker.train_history['test_acc'][-1] 
                                     if tracker.train_history['test_acc'] else 0,
                    'best_test_acc': max(tracker.train_history['test_acc']) 
                                    if tracker.train_history['test_acc'] else 0,
                    'final_train_acc': tracker.train_history['train_acc'][-1] 
                                      if tracker.train_history['train_acc'] else 0,
                    'total_time': time.time() - tracker.start_time,
                    'convergence_round': self._find_convergence_round(tracker)
                }
                
                logging.info(f"算法 {algorithm} 完成")
                logging.info(f"最终测试精度: {self.results[algorithm]['final_test_acc']:.4f}")
                logging.info(f"最佳测试精度: {self.results[algorithm]['best_test_acc']:.4f}")
                logging.info(f"训练时间: {self.results[algorithm]['total_time']/60:.1f}分钟")
                
            except Exception as e:
                logging.error(f"算法 {algorithm} 运行失败: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        total_time = time.time() - start_time
        logging.info(f"\n=== 算法比较完成 ===")
        logging.info(f"总耗时: {total_time/60:.1f}分钟")
        
        # 生成比较报告
        self._generate_comparison_report()
        self._plot_comparison_charts()

    def _setup_config(self, algorithm, quick_test, full_validation):
        """设置算法配置"""
        cfg = get_cfg()
        
        # 基本配置
        if self.config_file:
            cfg.merge_from_file(self.config_file)
        
        cfg.algorithm = algorithm
        cfg.mode = 'standalone'
        cfg.server_index = -1
        cfg.client_index = -1
        
        # 快速测试模式
        if quick_test:
            cfg.comm_round = 5
            cfg.client_num_in_total = 3
            cfg.client_num_per_round = 2
            cfg.global_epochs_per_round = 1
        
        # 验证模式
        cfg.full_validation = full_validation
        
        # 算法特定参数
        if algorithm == 'FedProx':
            cfg.fedprox_mu = 0.1
        elif algorithm == 'PFedMe':
            cfg.pfedme_beta = 1.0
            cfg.pfedme_lamda = 15.0
            cfg.pfedme_K = 5
            cfg.pfedme_personal_lr = 0.01
        
        # 设置随机种子确保公平比较
        cfg.seed = 42
        set_random_seed(cfg.seed)
        torch.backends.cudnn.deterministic = True
        
        return cfg

    def _find_convergence_round(self, tracker):
        """找到收敛轮次（测试精度不再显著提升的轮次）"""
        if not tracker.train_history['test_acc']:
            return -1
        
        test_accs = tracker.train_history['test_acc']
        if len(test_accs) < 3:
            return len(test_accs)
        
        # 寻找连续3轮精度提升小于0.01的轮次
        for i in range(2, len(test_accs)):
            if (test_accs[i] - test_accs[i-2]) < 0.01:
                return i + 1
        
        return len(test_accs)

    def _generate_comparison_report(self):
        """生成比较报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建比较数据
        comparison_data = []
        for algorithm, result in self.results.items():
            comparison_data.append({
                'Algorithm': algorithm,
                'Final_Test_Accuracy': result['final_test_acc'],
                'Best_Test_Accuracy': result['best_test_acc'],
                'Final_Train_Accuracy': result['final_train_acc'],
                'Training_Time_Minutes': result['total_time'] / 60,
                'Convergence_Round': result['convergence_round']
            })
        
        # 保存为CSV
        df = pd.DataFrame(comparison_data)
        csv_file = os.path.join(self.output_dir, f"algorithm_comparison_{timestamp}.csv")
        df.to_csv(csv_file, index=False)
        
        # 保存为JSON
        json_file = os.path.join(self.output_dir, f"algorithm_comparison_{timestamp}.json")
        with open(json_file, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        
        # 生成文本报告
        report_file = os.path.join(self.output_dir, f"comparison_report_{timestamp}.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("联邦学习算法性能比较报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"比较算法: {', '.join(self.algorithms)}\n\n")
            
            # 排序显示结果
            sorted_results = sorted(comparison_data, 
                                  key=lambda x: x['Best_Test_Accuracy'], reverse=True)
            
            f.write("按最佳测试精度排序:\n")
            f.write("-" * 30 + "\n")
            for i, result in enumerate(sorted_results, 1):
                f.write(f"{i}. {result['Algorithm']}\n")
                f.write(f"   最佳测试精度: {result['Best_Test_Accuracy']:.4f}\n")
                f.write(f"   最终测试精度: {result['Final_Test_Accuracy']:.4f}\n")
                f.write(f"   训练时间: {result['Training_Time_Minutes']:.1f}分钟\n")
                f.write(f"   收敛轮次: {result['Convergence_Round']}\n\n")
        
        logging.info(f"比较报告已保存: {report_file}")

    def _plot_comparison_charts(self):
        """绘制比较图表"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建综合比较图
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('联邦学习算法性能比较', fontsize=16, fontweight='bold')
        
        # 1. 最终测试精度比较
        algorithms = list(self.results.keys())
        final_accs = [self.results[alg]['final_test_acc'] for alg in algorithms]
        
        axes[0, 0].bar(algorithms, final_accs, color=['blue', 'red', 'green', 'orange'])
        axes[0, 0].set_title('最终测试精度比较')
        axes[0, 0].set_ylabel('精度')
        axes[0, 0].set_ylim(0, 1)
        for i, v in enumerate(final_accs):
            axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        # 2. 最佳测试精度比较
        best_accs = [self.results[alg]['best_test_acc'] for alg in algorithms]
        axes[0, 1].bar(algorithms, best_accs, color=['blue', 'red', 'green', 'orange'])
        axes[0, 1].set_title('最佳测试精度比较')
        axes[0, 1].set_ylabel('精度')
        axes[0, 1].set_ylim(0, 1)
        for i, v in enumerate(best_accs):
            axes[0, 1].text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        # 3. 训练时间比较
        train_times = [self.results[alg]['total_time']/60 for alg in algorithms]
        axes[0, 2].bar(algorithms, train_times, color=['blue', 'red', 'green', 'orange'])
        axes[0, 2].set_title('训练时间比较')
        axes[0, 2].set_ylabel('时间 (分钟)')
        for i, v in enumerate(train_times):
            axes[0, 2].text(i, v + max(train_times)*0.01, f'{v:.1f}', ha='center')
        
        # 4. 收敛轮次比较
        convergence_rounds = [self.results[alg]['convergence_round'] for alg in algorithms]
        axes[1, 0].bar(algorithms, convergence_rounds, color=['blue', 'red', 'green', 'orange'])
        axes[1, 0].set_title('收敛轮次比较')
        axes[1, 0].set_ylabel('轮次')
        for i, v in enumerate(convergence_rounds):
            axes[1, 0].text(i, v + max(convergence_rounds)*0.01, f'{v}', ha='center')
        
        # 5. 测试精度曲线比较
        for alg in algorithms:
            if alg in self.results and self.results[alg]['tracker'].train_history['test_acc']:
                rounds = self.results[alg]['tracker'].train_history['rounds']
                test_accs = self.results[alg]['tracker'].train_history['test_acc']
                axes[1, 1].plot(rounds, test_accs, label=alg, linewidth=2, marker='o')
        
        axes[1, 1].set_title('测试精度收敛曲线')
        axes[1, 1].set_xlabel('轮次')
        axes[1, 1].set_ylabel('测试精度')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. 综合性能雷达图
        categories = ['最终精度', '最佳精度', '训练效率', '收敛速度']
        
        # 标准化数据到0-1范围
        normalized_data = {}
        for alg in algorithms:
            final_acc = self.results[alg]['final_test_acc']
            best_acc = self.results[alg]['best_test_acc']
            # 训练效率 = 1 / (训练时间/最短训练时间)
            time_efficiency = min(train_times) / (self.results[alg]['total_time']/60)
            # 收敛速度 = 1 / (收敛轮次/最少收敛轮次)
            conv_speed = min([r for r in convergence_rounds if r > 0]) / max(self.results[alg]['convergence_round'], 1)
            
            normalized_data[alg] = [final_acc, best_acc, time_efficiency, conv_speed]
        
        # 绘制雷达图
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形
        
        ax_radar = plt.subplot(2, 3, 6, projection='polar')
        colors = ['blue', 'red', 'green', 'orange']
        
        for i, alg in enumerate(algorithms):
            values = normalized_data[alg] + [normalized_data[alg][0]]  # 闭合数据
            ax_radar.plot(angles, values, 'o-', linewidth=2, label=alg, color=colors[i])
            ax_radar.fill(angles, values, alpha=0.25, color=colors[i])
        
        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(categories)
        ax_radar.set_ylim(0, 1)
        ax_radar.set_title('综合性能雷达图')
        ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        
        # 保存图表
        chart_file = os.path.join(self.output_dir, f"algorithm_comparison_{timestamp}.png")
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"比较图表已保存: {chart_file}")


def main():
    parser = argparse.ArgumentParser(description="联邦学习算法比较")
    parser.add_argument("--config_file", default="config_quick_test.yaml", type=str,
                       help="配置文件路径")
    parser.add_argument("--output_dir", default="./comparison_results", type=str,
                       help="结果输出目录")
    parser.add_argument("--quick_test", action="store_true",
                       help="启用快速测试模式")
    parser.add_argument("--full_validation", action="store_true",
                       help="启用完整验证")
    
    args = parser.parse_args()
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'comparison.log')),
            logging.StreamHandler()
        ]
    )
    
    # 创建比较器并运行
    comparator = AlgorithmComparator(args.output_dir, args.config_file)
    comparator.run_comparison(args.quick_test, args.full_validation)
    
    logging.info("算法比较完成！")


if __name__ == "__main__":
    main() 