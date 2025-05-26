import logging
import os
import zipfile
import json
import requests
from PIL import Image
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from typing import Any, Callable, Optional, Tuple, List
from tqdm import tqdm

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def data_transforms_yummly28k(resize=224, augmentation="default", dataset_type="full_dataset",
                              image_resolution=224):
    """
    创建Yummly28k数据集的数据变换
    Args:
        resize: 图像大小
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


class Yummly28k(data.Dataset):
    """
    Yummly28k数据集类
    包含27,638个食谱，16种菜系，13种食谱类型
    """
    
    url = "http://123.57.42.89/Dataset_ict/Yummly-66K-28K/Yummly28K.zip"
    filename = "Yummly28K.zip"
    
    # 16种菜系
    cuisines = [
        'american', 'italian', 'mexican', 'chinese', 'indian', 'french',
        'thai', 'japanese', 'greek', 'spanish', 'korean', 'vietnamese',
        'mediterranean', 'middle_eastern', 'british', 'german'
    ]
    
    # 13种食谱类型
    courses = [
        'main_dish', 'dessert', 'appetizer', 'side_dish', 'lunch',
        'snack', 'breakfast', 'dinner', 'soup', 'salad',
        'beverage', 'bread', 'sauce'
    ]

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        classification_type: str = "cuisine"  # "cuisine" or "course"
    ) -> None:
        """
        Args:
            root: 数据集根目录
            train: 是否为训练集
            transform: 图像变换
            target_transform: 标签变换
            download: 是否下载数据集
            classification_type: 分类类型，"cuisine"(菜系)或"course"(食谱类型)
        """
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.classification_type = classification_type
        
        if download:
            self.download()
        
        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted. '
                             'You can use download=True to download it')
        
        self.data, self.targets = self._load_data()

    def _check_integrity(self) -> bool:
        """检查数据集完整性"""
        data_dir = os.path.join(self.root, "Yummly28K")
        return os.path.exists(data_dir) and os.path.exists(os.path.join(data_dir, "recipe.json"))

    def download(self) -> None:
        """下载并解压数据集"""
        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        os.makedirs(self.root, exist_ok=True)
        
        # 下载文件
        filepath = os.path.join(self.root, self.filename)
        
        # 如果文件存在但损坏，删除它
        if os.path.exists(filepath):
            try:
                with zipfile.ZipFile(filepath, 'r') as zip_ref:
                    pass  # 只是测试文件是否有效
            except zipfile.BadZipFile:
                logger.warning(f"Corrupted zip file detected, removing: {filepath}")
                os.remove(filepath)
        
        if not os.path.exists(filepath):
            print(f"Downloading {self.url}")
            try:
                self._download_file(self.url, filepath)
            except Exception as e:
                logger.error(f"Download failed: {e}")
                if os.path.exists(filepath):
                    os.remove(filepath)
                raise
        
        # 验证下载的文件
        try:
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                pass  # 测试文件是否有效
        except zipfile.BadZipFile:
            logger.error(f"Downloaded file is corrupted: {filepath}")
            os.remove(filepath)
            raise RuntimeError("Downloaded file is corrupted")
        
        # 解压文件
        print("Extracting files...")
        try:
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(self.root)
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            raise
        
        print("Download and extraction completed!")

    def _download_file(self, url: str, filepath: str) -> None:
        """下载文件并显示进度条，支持断点续传和多线程下载"""
        import threading
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        # 检查是否已有部分下载的文件
        initial_pos = 0
        if os.path.exists(filepath):
            initial_pos = os.path.getsize(filepath)
        
        # 获取文件总大小
        try:
            head_response = requests.head(url)
            head_response.raise_for_status()
            total_size = int(head_response.headers.get('content-length', 0))
        except Exception as e:
            logger.warning(f"无法获取文件大小，使用单线程下载: {e}")
            return self._download_file_single_thread(url, filepath)
        
        # 如果文件已经完整下载，直接返回
        if initial_pos >= total_size and total_size > 0:
            print(f"File {os.path.basename(filepath)} already downloaded completely")
            return
        
        # 检查服务器是否支持范围请求
        if head_response.headers.get('accept-ranges') != 'bytes':
            logger.info("服务器不支持范围请求，使用单线程下载")
            return self._download_file_single_thread(url, filepath)
        
        # 多线程下载配置
        num_threads = 4  # 线程数
        chunk_size = (total_size - initial_pos) // num_threads
        
        if chunk_size < 1024 * 1024:  # 如果每个块小于1MB，使用单线程
            logger.info("文件较小，使用单线程下载")
            return self._download_file_single_thread(url, filepath)
        
        print(f"使用 {num_threads} 个线程下载文件...")
        
        # 创建临时文件存储各个线程的数据
        temp_files = []
        download_ranges = []
        
        for i in range(num_threads):
            start = initial_pos + i * chunk_size
            if i == num_threads - 1:
                end = total_size - 1
            else:
                end = initial_pos + (i + 1) * chunk_size - 1
            
            if start <= end:
                temp_file = f"{filepath}.part{i}"
                temp_files.append(temp_file)
                download_ranges.append((start, end, temp_file))
        
        # 创建进度条
        progress_lock = threading.Lock()
        downloaded_bytes = [initial_pos]  # 使用列表以便在线程间共享
        
        pbar = tqdm(
            desc=os.path.basename(filepath),
            total=total_size,
            initial=initial_pos,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        )
        
        def download_chunk(start, end, temp_file):
            """下载文件的一个块"""
            try:
                headers = {'Range': f'bytes={start}-{end}'}
                response = requests.get(url, headers=headers, stream=True)
                response.raise_for_status()
                
                with open(temp_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            with progress_lock:
                                downloaded_bytes[0] += len(chunk)
                                pbar.update(len(chunk))
                return True
            except Exception as e:
                logger.error(f"下载块 {start}-{end} 失败: {e}")
                return False
        
        try:
            # 使用线程池下载
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = []
                for start, end, temp_file in download_ranges:
                    future = executor.submit(download_chunk, start, end, temp_file)
                    futures.append(future)
                
                # 等待所有下载完成
                success = True
                for future in as_completed(futures):
                    if not future.result():
                        success = False
            
            pbar.close()
            
            if not success:
                raise Exception("部分下载失败")
            
            # 合并文件
            print("合并文件...")
            mode = 'ab' if initial_pos > 0 else 'wb'
            with open(filepath, mode) as output_file:
                for temp_file in temp_files:
                    if os.path.exists(temp_file):
                        with open(temp_file, 'rb') as input_file:
                            output_file.write(input_file.read())
                        os.remove(temp_file)
            
        except Exception as e:
            pbar.close()
            # 清理临时文件
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            raise e

    def _download_file_single_thread(self, url: str, filepath: str) -> None:
        """单线程下载文件，支持断点续传"""
        # 检查是否已有部分下载的文件
        resume_header = {}
        initial_pos = 0
        if os.path.exists(filepath):
            initial_pos = os.path.getsize(filepath)
            resume_header['Range'] = f'bytes={initial_pos}-'
        
        try:
            response = requests.get(url, headers=resume_header, stream=True)
            response.raise_for_status()
            
            # 获取文件总大小
            if 'content-range' in response.headers:
                # 断点续传情况
                total_size = int(response.headers['content-range'].split('/')[-1])
            else:
                # 全新下载情况
                total_size = int(response.headers.get('content-length', 0))
                initial_pos = 0  # 重新开始下载
            
            # 如果文件已经完整下载，直接返回
            if initial_pos >= total_size and total_size > 0:
                print(f"File {os.path.basename(filepath)} already downloaded completely")
                return
            
            # 打开文件进行写入（追加模式如果是续传）
            mode = 'ab' if initial_pos > 0 else 'wb'
            
            with open(filepath, mode) as file, tqdm(
                desc=os.path.basename(filepath),
                total=total_size,
                initial=initial_pos,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # 过滤掉保持连接的空块
                        size = file.write(chunk)
                        pbar.update(size)
                        
        except requests.exceptions.RequestException as e:
            logger.error(f"Download error: {e}")
            # 不删除部分下载的文件，以便下次续传
            raise
        except Exception as e:
            logger.error(f"Unexpected error during download: {e}")
            # 对于其他错误，删除可能损坏的文件
            if os.path.exists(filepath):
                os.remove(filepath)
            raise

    def _load_data(self) -> Tuple[List, List]:
        """加载数据和标签"""
        data_dir = os.path.join(self.root, "Yummly28K")
        recipe_file = os.path.join(data_dir, "recipe.json")
        
        with open(recipe_file, 'r', encoding='utf-8') as f:
            recipes = json.load(f)
        
        images = []
        labels = []
        
        # 创建标签映射
        if self.classification_type == "cuisine":
            label_map = {cuisine: idx for idx, cuisine in enumerate(self.cuisines)}
        else:  # course
            label_map = {course: idx for idx, course in enumerate(self.courses)}
        
        for recipe in recipes:
            img_path = os.path.join(data_dir, "images", recipe.get("image", ""))
            
            # 检查图像文件是否存在
            if not os.path.exists(img_path):
                continue
            
            # 获取标签
            if self.classification_type == "cuisine":
                label_name = recipe.get("cuisine", "").lower()
            else:  # course
                label_name = recipe.get("course", "").lower()
            
            if label_name in label_map:
                images.append(img_path)
                labels.append(label_map[label_name])
        
        # 划分训练集和测试集 (80:20)
        total_samples = len(images)
        train_size = int(0.8 * total_samples)
        
        # 设置随机种子确保可重复性
        np.random.seed(42)
        indices = np.random.permutation(total_samples)
        
        if self.train:
            selected_indices = indices[:train_size]
        else:
            selected_indices = indices[train_size:]
        
        selected_images = [images[i] for i in selected_indices]
        selected_labels = [labels[i] for i in selected_indices]
        
        return selected_images, selected_labels

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index: 索引
        Returns:
            tuple: (image, target)
        """
        img_path, target = self.data[index], self.targets[index]
        
        # 加载图像
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.warning(f"Error loading image {img_path}: {e}")
            # 创建固定大小的默认图像（使用常见的默认大小）
            image = Image.new('RGB', (224, 224), color=(128, 128, 128))
        
        if self.transform is not None:
            image = self.transform(image)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return image, target

    def __len__(self) -> int:
        return len(self.data)

    def get_num_classes(self) -> int:
        """获取类别数量"""
        if self.classification_type == "cuisine":
            return len(self.cuisines)
        else:  # course
            return len(self.courses)


class Yummly28k_truncated(data.Dataset):
    """
    Yummly28k截断数据集，用于联邦学习中的客户端数据分割
    """

    def __init__(self, root, dataidxs=None, train=True, transform=None, 
                 target_transform=None, download=False, classification_type="cuisine"):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.classification_type = classification_type

        self.data, self.targets = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        yummly_dataobj = Yummly28k(
            self.root, self.train, self.transform, self.target_transform, 
            self.download, self.classification_type
        )

        data = yummly_dataobj.data
        targets = np.array(yummly_dataobj.targets)

        if self.dataidxs is not None:
            data = [data[i] for i in self.dataidxs]
            targets = targets[self.dataidxs]

        return data, targets

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target)
        """
        img_path, target = self.data[index], self.targets[index]
        
        # 加载图像
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.warning(f"Error loading image {img_path}: {e}")
            # 创建固定大小的默认图像
            img = Image.new('RGB', (224, 224), color=(128, 128, 128))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


class Yummly28k_truncated_WO_reload(data.Dataset):
    """
    Yummly28k截断数据集，不重新加载完整数据集
    """

    def __init__(self, root, dataidxs=None, train=True, transform=None, 
                 target_transform=None, full_dataset=None):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.full_dataset = full_dataset

        self.data, self.targets = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        if self.train:
            data = [self.full_dataset.data[i] for i in self.dataidxs]
            targets = np.array(self.full_dataset.targets)[self.dataidxs]
        else:
            data = self.full_dataset.data
            targets = np.array(self.full_dataset.targets)

        return data, targets

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target)
        """
        img_path, target = self.data[index], self.targets[index]
        
        # 加载图像
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.warning(f"Error loading image {img_path}: {e}")
            # 创建固定大小的默认图像
            img = Image.new('RGB', (224, 224), color=(128, 128, 128))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data) 