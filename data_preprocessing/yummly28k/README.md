# Yummly28k 数据集

## 概述

Yummly28k是一个包含27,638个食谱的食品图像数据集，每个食谱包含一张图片、配料、菜系和课程信息。该数据集支持两种分类任务：

- **菜系分类**: 16种菜系（美国、意大利、墨西哥、中国、印度、法国、泰国、日本、希腊、西班牙、韩国、越南、地中海、中东、英国、德国）
- **食谱类型分类**: 13种类型（主菜、甜点、开胃菜、配菜、午餐、小吃、早餐、晚餐、汤、沙拉、饮料、面包、酱料）

## 数据集信息

- **总样本数**: 27,638个食谱
- **图像格式**: RGB图像
- **下载地址**: http://123.57.42.89/Dataset_ict/Yummly-66K-28K/Yummly28K.zip
- **参考论文**: "Being a Super Cook: Joint Food Attributes and Multi-Modal Content Modeling for Recipe Retrieval and Exploration" (2017)

## 使用方法

### 基本使用

```python
from data_preprocessing.yummly28k.datasets import Yummly28k

# 菜系分类
dataset_cuisine = Yummly28k(
    root="./data",
    train=True,
    download=True,
    classification_type="cuisine"
)

# 食谱类型分类
dataset_course = Yummly28k(
    root="./data",
    train=True,
    download=True,
    classification_type="course"
)

print(f"菜系分类类别数: {dataset_cuisine.get_num_classes()}")
print(f"食谱类型分类类别数: {dataset_course.get_num_classes()}")
```

### 联邦学习数据分区

```python
from data_preprocessing.yummly28k.data_loader import load_partition_data_yummly28k

# 创建参数对象
class Args:
    def __init__(self):
        self.data_efficient_load = False
        self.dirichlet_balance = True
        self.dirichlet_min_p = 0.01
        self.batch_size = 32

args = Args()

# 加载分区数据
result = load_partition_data_yummly28k(
    dataset="yummly28k",
    data_dir="./data",
    partition_method="hetero",  # 异构分布
    partition_alpha=0.1,       # 迪利克雷参数
    client_number=10,          # 客户端数量
    batch_size=32,
    classification_type="cuisine",
    args=args
)

(train_data_num, test_data_num, train_data_global, test_data_global,
 data_local_num_dict, train_data_local_dict, test_data_local_dict,
 class_num) = result
```

### 支持的分区方法

1. **同构分布 (homo)**: 数据在客户端间均匀分布
2. **异构分布 (hetero)**: 使用迪利克雷分布创建非IID数据分布
3. **标签限制 (noniid-#label[0-9])**: 每个客户端只包含指定数量的标签
4. **长尾分布 (long-tail)**: 部分客户端拥有大部分数据，其他客户端数据较少
5. **固定分布 (hetero-fix)**: 从预定义文件加载分布

## 数据变换

数据集使用ImageNet预训练模型的标准化参数：
- Mean: [0.485, 0.456, 0.406]
- Std: [0.229, 0.224, 0.225]

训练时的数据增强包括：
- 随机裁剪
- 随机水平翻转
- 颜色抖动

## 文件结构

```
yummly28k/
├── __init__.py          # 模块初始化
├── datasets.py          # 数据集类定义
├── data_loader.py       # 数据加载和分区功能
├── transform.py         # 数据变换函数
└── README.md           # 说明文档
```

## 测试

运行测试脚本验证功能：

```bash
# 运行所有测试
python test_yummly28k.py --test all

# 只测试下载功能
python test_yummly28k.py --test download

# 只测试数据分区
python test_yummly28k.py --test partition

# 只测试联邦学习数据加载
python test_yummly28k.py --test federated
```

## 注意事项

1. 首次使用时会自动下载数据集（约几GB大小）
2. 数据集会自动解压到指定目录
3. 支持断点续传，如果下载中断可以重新运行
4. 图像加载失败时会使用默认的灰色图像替代
5. 数据集按80:20的比例划分训练集和测试集

## 依赖项

- torch
- torchvision
- PIL (Pillow)
- numpy
- requests
- tqdm
- json

## 许可证

请遵循原始数据集的许可证要求。 