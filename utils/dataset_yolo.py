"""
YOLO格式数据集类
支持数据加载、数据增强、Mosaic混合等
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Tuple, Optional, Callable, Dict, Any
import random


class YOLODroneDataset(Dataset):
    """YOLO格式无人机数据集"""

    def __init__(
        self,
        image_dir: Path,
        label_dir: Path,
        transforms: Optional[Callable] = None,
        use_mosaic: bool = False,
        mosaic_prob: float = 0.5,
        img_size: int = 640,
        num_classes: int = 1,
        class_names: List[str] = None
    ):
        """
        Args:
            image_dir: 图像目录
            label_dir: 标签目录
            transforms: 数据增强函数
            use_mosaic: 是否使用Mosaic增强
            mosaic_prob: Mosaic增强概率
            img_size: 目标图像尺寸
            num_classes: 类别数量
            class_names: 类别名称列表
        """
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.transforms = transforms
        self.use_mosaic = use_mosaic
        self.mosaic_prob = mosaic_prob
        self.img_size = img_size
        self.num_classes = num_classes

        if class_names is None:
            self.class_names = ['drone']
        else:
            self.class_names = class_names

        # 获取所有图像文件
        self.image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            self.image_files.extend(list(self.image_dir.glob(ext)))

        self.image_files.sort()

        print(f"加载YOLO数据集: {self.image_dir}")
        print(f"图像数量: {len(self.image_files)}")
        print(f"类别数量: {self.num_classes}")

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """获取单条数据"""
        # 如果使用Mosaic增强且满足概率
        if self.use_mosaic and random.random() < self.mosaic_prob:
            return self._mosaic_augment(idx)
        else:
            return self._standard_augment(idx)

    def _standard_augment(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """标准数据增强"""
        img_file = self.image_files[idx]

        # 读取图像
        image = cv2.imread(str(img_file))
        if image is None:
            raise ValueError(f"无法读取图像: {img_file}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 读取标签
        label_file = self.label_dir / img_file.with_suffix('.txt').name

        boxes = []
        labels = []

        if label_file.exists():
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])

                        # 转换为xyxy格式（像素坐标）
                        img_h, img_w = image.shape[:2]
                        x1 = (x_center - width / 2) * img_w
                        y1 = (y_center - height / 2) * img_h
                        x2 = (x_center + width / 2) * img_w
                        y2 = (y_center + height / 2) * img_h

                        boxes.append([x1, y1, x2, y2])

                        # 标签转换：YOLO(从0或1开始) -> Faster R-CNN(0背景, 1起)
                        # 如果YOLO标签从0开始: 0 -> 1
                        # 如果YOLO标签从1开始: 1 -> 1
                        # 统一处理: 确保Faster R-CNN标签 >= 1
                        adjusted_label = class_id if class_id > 0 else 1
                        labels.append(adjusted_label)

        # 转换为numpy数组
        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)

        # 数据增强
        if self.transforms is not None:
            try:
                augmented = self.transforms(image=image, bboxes=boxes, labels=labels)
                image = augmented['image']
                boxes = np.array(augmented['bboxes'], dtype=np.float32)
                labels = np.array(augmented['labels'], dtype=np.int64)
            except Exception as e:
                # 如果增强失败，使用原图
                pass

        # 转换为Tensor
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.from_numpy(boxes)
            labels = torch.from_numpy(labels)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx])
        }

        return image, target

    def _mosaic_augment(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Mosaic数据增强（4张图像拼接）"""
        # 随机选择4张图像
        indices = random.choices(range(len(self)), k=4)

        # 创建画布
        canvas = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

        # 计算分割点
        xc = random.randint(int(self.img_size * 0.25), int(self.img_size * 0.75))
        yc = random.randint(int(self.img_size * 0.25), int(self.img_size * 0.75))

        # 四个区域
        positions = [
            (0, 0, xc, yc),  # 左上
            (xc, 0, self.img_size, yc),  # 右上
            (0, yc, xc, self.img_size),  # 左下
            (xc, yc, self.img_size, self.img_size)  # 右下
        ]

        all_boxes = []
        all_labels = []

        for i, img_idx in enumerate(indices):
            # 读取图像
            img_file = self.image_files[img_idx]
            image = cv2.imread(str(img_file))
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 读取标签
            label_file = self.label_dir / img_file.with_suffix('.txt').name
            boxes = []
            labels = []

            if label_file.exists():
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])

                            img_h, img_w = image.shape[:2]
                            x1 = (x_center - width / 2) * img_w
                            y1 = (y_center - height / 2) * img_h
                            x2 = (x_center + width / 2) * img_w
                            y2 = (y_center + height / 2) * img_h

                            boxes.append([x1, y1, x2, y2])

                            # 标签转换：确保Faster R-CNN标签 >= 1
                            adjusted_label = class_id if class_id > 0 else 1
                            labels.append(adjusted_label)

            # 调整图像大小
            x1, y1, x2, y2 = positions[i]
            w = x2 - x1
            h = y2 - y1

            # 缩放图像
            img_resized = cv2.resize(image, (w, h))

            # 放到画布上
            canvas[y1:y2, x1:x2] = img_resized

            # 调整框坐标
            for box in boxes:
                # 按比例缩放
                box_scaled = [
                    box[0] * w / image.shape[1] + x1,
                    box[1] * h / image.shape[0] + y1,
                    box[2] * w / image.shape[1] + x1,
                    box[3] * h / image.shape[0] + y1
                ]
                all_boxes.append(box_scaled)
                all_labels.append(labels[0] if labels else 1)

        # 转换为Tensor
        image = torch.from_numpy(canvas).permute(2, 0, 1).float() / 255.0

        if len(all_boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.from_numpy(np.array(all_boxes, dtype=np.float32))
            labels = torch.from_numpy(np.array(all_labels, dtype=np.int64))

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx])
        }

        return image, target

    def get_classes(self) -> List[str]:
        """获取类别列表"""
        return self.class_names

    def get_class_names(self) -> List[str]:
        """获取类别名称列表"""
        return self.class_names

    def get_num_classes(self) -> int:
        """获取类别数量"""
        return self.num_classes


def create_yolo_datasets(
    base_dir: str = 'data',
    transforms_train: Optional[Callable] = None,
    transforms_val: Optional[Callable] = None,
    use_mosaic: bool = True,
    mosaic_prob: float = 0.5,
    img_size: int = 640,
    num_classes: int = 1,
    class_names: List[str] = None
) -> Tuple[YOLODroneDataset, Optional[YOLODroneDataset], Optional[YOLODroneDataset]]:
    """
    创建YOLO格式的训练集、验证集、测试集

    Args:
        base_dir: 数据集根目录
        transforms_train: 训练集数据增强
        transforms_val: 验证集数据增强
        use_mosaic: 是否使用Mosaic增强
        mosaic_prob: Mosaic增强概率
        img_size: 目标图像尺寸
        num_classes: 类别数量
        class_names: 类别名称列表

    Returns:
        train_dataset, val_dataset, test_dataset
    """
    base_path = Path(base_dir)

    # 检查目录是否存在
    train_img_dir = base_path / 'images' / 'train'
    val_img_dir = base_path / 'images' / 'val'
    test_img_dir = base_path / 'images' / 'test'

    train_label_dir = base_path / 'labels' / 'train'
    val_label_dir = base_path / 'labels' / 'val'
    test_label_dir = base_path / 'labels' / 'test'

    # 创建训练集
    train_dataset = YOLODroneDataset(
        image_dir=train_img_dir,
        label_dir=train_label_dir,
        transforms=transforms_train,
        use_mosaic=use_mosaic,
        mosaic_prob=mosaic_prob,
        img_size=img_size,
        num_classes=num_classes,
        class_names=class_names
    )

    # 创建验证集
    val_dataset = YOLODroneDataset(
        image_dir=val_img_dir,
        label_dir=val_label_dir,
        transforms=transforms_val,
        use_mosaic=False,  # 验证集不使用Mosaic
        img_size=img_size,
        num_classes=num_classes,
        class_names=class_names
    )

    # 创建测试集（可选）
    test_dataset = None
    if test_img_dir.exists() and len(list(test_img_dir.glob('*'))) > 0:
        test_dataset = YOLODroneDataset(
            image_dir=test_img_dir,
            label_dir=test_label_dir,
            transforms=transforms_val,
            use_mosaic=False,
            img_size=img_size,
            num_classes=num_classes,
            class_names=class_names
        )

    return train_dataset, val_dataset, test_dataset


def collate_fn(batch: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]) -> Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]]:
    """
    自定义collate函数，处理不同大小的目标检测批次

    Args:
        batch: 批次数据列表

    Returns:
        images: 图像列表
        targets: 目标字典列表
    """
    images = []
    targets = []

    for image, target in batch:
        images.append(image)
        targets.append(target)

    return images, targets