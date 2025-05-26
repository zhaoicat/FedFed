# 联邦学习实验报告

**生成时间**: 2025-05-25 19:03:10

**总耗时**: 33.0分钟

**成功率**: 1/1 (100.0%)

## 实验内容

本次实验包含以下内容：

1. **联邦学习算法训练**
   - FedAvg: 经典联邦平均算法
   - FedProx: 带近端项的联邦学习算法
   - PFedMe: 个性化联邦学习算法
   - FedFed: 基于特征蒸馏的联邦学习算法

2. **算法性能比较**
   - 测试精度对比
   - 训练时间对比
   - 收敛速度对比
   - 综合性能评估

3. **集中式学习基线**
   - 单机集中式训练
   - 与联邦学习性能对比

## 数据集信息

- **数据集**: Yummly28k食品图片数据集
- **类别数**: 16个菜系
- **图片尺寸**: 32×32 (适配VAE架构)
- **数据分布**: 使用Dirichlet分布模拟非独立同分布(Non-IID)场景

## 结果文件说明

实验结果保存在以下目录结构中：

```
experiment_results/
├── single_FedAvg/          # FedAvg算法单独运行结果
├── single_FedProx/         # FedProx算法单独运行结果
├── single_PFedMe/          # PFedMe算法单独运行结果
├── single_FedFed/          # FedFed算法单独运行结果
├── comparison/             # 算法比较结果
├── centralized/            # 集中式基线结果
└── experiment_report.md    # 本报告
```

每个算法目录包含：
- `*_results_*.json`: 训练历史数据
- `*_curves_*.png`: 训练曲线图
- `*_summary_*.json`: 结果摘要

比较结果目录包含：
- `algorithm_comparison_*.csv`: 算法性能对比表
- `algorithm_comparison_*.png`: 综合比较图表
- `comparison_report_*.txt`: 详细比较报告

## 使用说明

### 快速测试
```bash
python run_experiments.py --quick_test
```

### 运行单个算法
```bash
python run_experiments.py --mode single --algorithm FedAvg
```

### 只进行算法比较
```bash
python run_experiments.py --mode comparison
```

### 只运行集中式基线
```bash
python run_experiments.py --mode centralized
```

## 实验状态

✅ **所有实验成功完成**

所有算法都已成功训练并生成了相应的结果文件。您可以查看各个目录中的图表和数据文件来分析算法性能。

