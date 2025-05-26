# 联邦学习算法实现与比较

本项目实现了多种联邦学习算法，并提供了完整的训练、比较和可视化功能。

## 🚀 快速开始


### 快速测试


这将在几分钟内完成所有算法的测试，并生成比较报告。

## 📊 支持的算法

### 1. FedAvg (联邦平均)
- **描述**: 经典的联邦学习算法，通过加权平均聚合客户端模型
- **特点**: 简单有效，适合IID数据分布
- **论文**: Communication-Efficient Learning of Deep Networks from Decentralized Data

### 2. FedProx (联邦近端)
- **描述**: 在FedAvg基础上添加近端项，提高Non-IID场景下的稳定性
- **特点**: 通过近端项约束客户端模型不偏离全局模型太远
- **参数**: `fedprox_mu` (近端项系数，默认0.1)
- **论文**: Federated Optimization in Heterogeneous Networks

### 3. PFedMe (个性化联邦学习)
- **描述**: 支持个性化的联邦学习算法，每个客户端维护个性化模型
- **特点**: 适合客户端数据分布差异较大的场景
- **参数**: 
  - `pfedme_beta`: 个性化参数 (默认1.0)
  - `pfedme_lamda`: 正则化参数 (默认15.0)
  - `pfedme_K`: 本地更新步数 (默认5)
  - `pfedme_personal_lr`: 个性化学习率 (默认0.01)
- **论文**: Personalized Federated Learning with Moreau Envelopes

### 4. FedFed (特征蒸馏联邦学习)
- **描述**: 基于特征蒸馏的联邦学习框架，通过VAE生成共享数据
- **特点**: 通过特征蒸馏提高模型性能，适合数据隐私要求高的场景
- **论文**: FedFed: Feature Distillation against Data Heterogeneity in Federated Learning

## 🛠️ 使用方法

### 单独运行算法

```bash
# 运行FedAvg算法
python main.py --config_file config_yummly28k_test.yaml --algorithm fedavg --output_dir enhanced_results/fedavg


# 运行FedProx算法
python main.py --config_file config_yummly28k_test.yaml --algorithm FedProx --output_dir enhanced_results/fedavg


# 运行PFedMe算法
python main.py --config_file config_yummly28k_test.yaml --algorithm PFedMe --output_dir enhanced_results/fedavg


# 运行FedFed算法
python main.py --config_file config_yummly28k_test.yaml --algorithm FedFed --output_dir enhanced_results/fedavg

```

## ⚙️ 配置参数

### 基本配置 (config_quick_test.yaml)


### 算法特定参数

#### FedProx参数
```yaml
fedprox_mu: 0.1              # 近端项系数
```

#### PFedMe参数
```yaml
pfedme_beta: 1.0             # 个性化参数
pfedme_lamda: 15.0           # 正则化参数
pfedme_K: 5                  # 本地更新步数
pfedme_personal_lr: 0.01     # 个性化学习率
```

## 📈 结果输出

### 训练结果

每个算法训练完成后会生成：

1. **训练历史数据** (`*_results_*.json`)
   - 每轮的训练/测试损失和精度
   - 训练时间统计

2. **训练曲线图** (`*_curves_*.png`)
   - 损失曲线
   - 精度曲线
   - 训练时间曲线
   - 最终精度对比

3. **结果摘要** (`*_summary_*.json`)
   - 最终性能指标
   - 训练统计信息

### 比较结果

算法比较会生成：

1. **性能对比表** (`algorithm_comparison_*.csv`)
   - 各算法的关键指标对比

2. **综合比较图** (`algorithm_comparison_*.png`)
   - 精度对比柱状图
   - 收敛曲线对比
   - 综合性能雷达图

3. **详细报告** (`comparison_report_*.txt`)
   - 按性能排序的算法排名
   - 详细性能分析

## 🎯 性能指标

### 主要评估指标

1. **测试精度** (Test Accuracy)
   - 最终测试精度
   - 最佳测试精度

2. **训练效率** (Training Efficiency)
   - 总训练时间
   - 每轮平均时间

3. **收敛性能** (Convergence Performance)
   - 收敛轮次
   - 收敛速度

4. **稳定性** (Stability)
   - 训练过程稳定性
   - 精度波动情况

### 比较维度

- **精度对比**: 各算法在相同条件下的测试精度
- **效率对比**: 达到相同精度所需的时间和轮次
- **鲁棒性对比**: 在Non-IID数据分布下的性能表现
- **可扩展性对比**: 不同客户端数量下的性能变化

## 🔧 高级功能

### 1. 快速验证模式

```bash
# 启用快速测试，减少训练轮次和客户端数量
python main.py --quick_test --algorithm FedAvg
```

### 2. 完整验证模式

```bash
# 启用完整验证，包含详细的性能评估
python main.py --full_validation --algorithm FedAvg
```

### 3. 自定义输出目录

```bash
# 指定结果输出目录
python main.py --output_dir ./my_results --algorithm FedAvg
```

### 4. 进度跟踪

所有训练过程都包含详细的进度显示：
- 实时训练进度
- 剩余时间估计
- 性能指标更新
- 可视化图表生成

## 📝 实验建议

### 快速验证
```bash
# 3-5分钟完成所有算法测试
python run_experiments.py --quick_test
```

### 完整实验
```bash
# 完整的算法比较实验（可能需要1-2小时）
python run_experiments.py --mode comparison
```

### 单算法深入分析
```bash
# 详细分析特定算法
python main.py --algorithm PFedMe --full_validation
```

## 🐛 故障排除

### 常见问题

1. **内存不足**
   - 减少batch_size
   - 减少客户端数量
   - 启用快速测试模式

2. **训练时间过长**
   - 使用快速测试模式
   - 减少通信轮次
   - 减少本地训练epochs

3. **精度不收敛**
   - 调整学习率
   - 增加训练轮次
   - 检查数据分布设置

### 日志查看

所有实验都会生成详细日志：
- `experiments.log`: 实验运行日志
- `comparison.log`: 算法比较日志
- `centralized.log`: 集中式训练日志

## 📚 参考文献

1. McMahan, B., et al. "Communication-efficient learning of deep networks from decentralized data." AISTATS 2017.

2. Li, T., et al. "Federated optimization in heterogeneous networks." MLSys 2020.

3. Dinh, C. T., et al. "Personalized federated learning with moreau envelopes." NeurIPS 2020.

4. Lin, T., et al. "FedFed: Feature distillation against data heterogeneity in federated learning." NeurIPS 2023.

## 🤝 贡献

欢迎提交Issue和Pull Request来改进项目！

## 📄 许可证

本项目采用MIT许可证。 