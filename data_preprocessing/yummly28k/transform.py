import torchvision.transforms as transforms


def data_transforms_yummly28k(resize=32, augmentation="default", dataset_type="full_dataset", image_resolution=32):
    """
    创建Yummly28k数据集的数据变换
    Args:
        resize: 图像大小，默认32
        augmentation: 数据增强类型
        dataset_type: 数据集类型
        image_resolution: 图像分辨率
    Returns:
        mean, std, train_transform, test_transform
    """
    # ImageNet预训练模型的标准化参数
    YUMMLY_MEAN = [0.485, 0.456, 0.406]
    YUMMLY_STD = [0.229, 0.224, 0.225]

    # 训练变换：确保输出固定大小
    train_transform = transforms.Compose([
        transforms.Resize((resize, resize)),  # 直接resize到目标大小
    ])
    
    # 测试变换：确保输出固定大小
    test_transform = transforms.Compose([
        transforms.Resize((resize, resize)),  # 直接resize到目标大小
    ])

    # 添加数据增强（仅训练时）
    if augmentation == "default" or augmentation is True:
        if resize >= 32:  # 对于足够大的图像添加数据增强
            train_transform.transforms.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
            ])
            if resize >= 64:  # 对于更大的图像添加更多增强
                train_transform.transforms.append(
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, 
                                         saturation=0.2, hue=0.1)
                )

    # 添加tensor转换和标准化
    for transform in [train_transform, test_transform]:
        transform.transforms.extend([
            transforms.ToTensor(),
            transforms.Normalize(YUMMLY_MEAN, YUMMLY_STD)
        ])

    return YUMMLY_MEAN, YUMMLY_STD, train_transform, test_transform 