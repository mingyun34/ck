"""
数据增强工具
支持多种数据增强策略
"""

import albumentations as A
import cv2
import numpy as np
from typing import Optional, Dict, Any


class AugmentationConfig:
    """数据增强配置"""

    # 图像增强参数
    BRIGHTNESS_LIMIT: float = 0.2
    CONTRAST_LIMIT: float = 0.2
    HUE_SHIFT_LIMIT: int = 20
    SAT_SHIFT_LIMIT: int = 30
    VAL_SHIFT_LIMIT: int = 20

    # 几何变换参数
    ROTATE_LIMIT: int = 15
    SCALE_LIMIT: float = 0.1
    SHIFT_LIMIT: float = 0.1

    # 概率
    HORIZONTAL_FLIP_PROB: float = 0.5
    ROTATE_PROB: float = 0.5
    BRIGHTNESS_PROB: float = 0.5
    NOISE_PROB: float = 0.3
    BLUR_PROB: float = 0.2

    # Mosaic参数
    MOSAIC_PROB: float = 0.5
    MIXUP_PROB: float = 0.1


def get_train_transforms_v1() -> A.Compose:
    """
    基础训练集增强 v1
    仅包含最基本的增强
    """
    return A.Compose([
        A.HorizontalFlip(p=AugmentationConfig.HORIZONTAL_FLIP_PROB),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels'],
        min_visibility=0.3
    ))


def get_train_transforms_v2() -> A.Compose:
    """
    中等强度训练集增强 v2
    包含颜色抖动和旋转
    """
    return A.Compose([
        A.HorizontalFlip(p=AugmentationConfig.HORIZONTAL_FLIP_PROB),
        A.RandomBrightnessContrast(
            brightness_limit=AugmentationConfig.BRIGHTNESS_LIMIT,
            contrast_limit=AugmentationConfig.CONTRAST_LIMIT,
            p=AugmentationConfig.BRIGHTNESS_PROB
        ),
        A.Rotate(
            limit=AugmentationConfig.ROTATE_LIMIT,
            p=AugmentationConfig.ROTATE_PROB,
            border_mode=cv2.BORDER_CONSTANT,
            value=0
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels'],
        min_visibility=0.3
    ))


def get_train_transforms_v3() -> A.Compose:
    """
    强力训练集增强 v3
    包含多种增强，适合小目标检测
    """
    return A.Compose([
        # 几何变换
        A.HorizontalFlip(p=AugmentationConfig.HORIZONTAL_FLIP_PROB),
        A.ShiftScaleRotate(
            shift_limit=AugmentationConfig.SHIFT_LIMIT,
            scale_limit=AugmentationConfig.SCALE_LIMIT,
            rotate_limit=AugmentationConfig.ROTATE_LIMIT,
            p=0.5,
            border_mode=cv2.BORDER_CONSTANT,
            value=0
        ),

        # 颜色增强
        A.RandomBrightnessContrast(
            brightness_limit=AugmentationConfig.BRIGHTNESS_LIMIT,
            contrast_limit=AugmentationConfig.CONTRAST_LIMIT,
            p=AugmentationConfig.BRIGHTNESS_PROB
        ),
        A.HueSaturationValue(
            hue_shift_limit=AugmentationConfig.HUE_SHIFT_LIMIT,
            sat_shift_limit=AugmentationConfig.SAT_SHIFT_LIMIT,
            val_shift_limit=AugmentationConfig.VAL_SHIFT_LIMIT,
            p=AugmentationConfig.BRIGHTNESS_PROB
        ),

        # 噪声和模糊
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
        ], p=AugmentationConfig.NOISE_PROB),

        A.OneOf([
            A.MotionBlur(blur_limit=3, p=1.0),
            A.MedianBlur(blur_limit=3, p=1.0),
            A.GaussianBlur(blur_limit=3, p=1.0),
        ], p=AugmentationConfig.BLUR_PROB),

        # 归一化
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels'],
        min_visibility=0.3,
        min_area=1024
    ))


def get_train_transforms_v4() -> A.Compose:
    """
    极强训练集增强 v4
    包含更多激进的增强策略
    """
    return A.Compose([
        # 几何变换
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.ShiftScaleRotate(
            shift_limit=0.2,
            scale_limit=0.2,
            rotate_limit=30,
            p=0.7,
            border_mode=cv2.BORDER_CONSTANT,
            value=0
        ),

        # 裁剪和缩放
        A.RandomResizedCrop(
            height=640,
            width=640,
            scale=(0.7, 1.0),
            ratio=(0.8, 1.2),
            p=0.5
        ),

        # 颜色增强
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=0.7
        ),
        A.HueSaturationValue(
            hue_shift_limit=30,
            sat_shift_limit=50,
            val_shift_limit=30,
            p=0.7
        ),
        A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.3),

        # 噪声和模糊
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 80.0), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.1), intensity=(0.1, 0.8), p=1.0),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
        ], p=0.5),

        A.OneOf([
            A.MotionBlur(blur_limit=5, p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
            A.GaussianBlur(blur_limit=5, p=1.0),
        ], p=0.3),

        # 归一化
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels'],
        min_visibility=0.2,
        min_area=512
    ))


def get_val_transforms() -> A.Compose:
    """
    验证集/测试集增强
    仅包含归一化
    """
    return A.Compose([
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels']
    ))


def get_test_transforms() -> A.Compose:
    """
    测试集增强
    仅包含归一化，与验证集相同
    """
    return get_val_transforms()


def get_inference_transforms() -> A.Compose:
    """
    推理时增强
    用于单张图像预测
    """
    return A.Compose([
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_tta_transforms() -> list:
    """
    测试时增强（Test Time Augmentation）
    返回多个变换用于预测融合
    """
    return [
        # 原始
        A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'])),

        # 水平翻转
        A.Compose([
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'])),

        # 垂直翻转
        A.Compose([
            A.VerticalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'])),

        # 旋转90度
        A.Compose([
            A.Rotate(limit=90, p=1.0, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'])),
    ]


def mixup_augmentation(
    image1: np.ndarray,
    boxes1: np.ndarray,
    labels1: np.ndarray,
    image2: np.ndarray,
    boxes2: np.ndarray,
    labels2: np.ndarray,
    alpha: float = 0.2
) -> tuple:
    """
    Mixup数据增强

    Args:
        image1: 第一张图像
        boxes1: 第一张图像的边界框
        labels1: 第一张图像的标签
        image2: 第二张图像
        boxes2: 第二张图像的边界框
        labels2: 第二张图像的标签
        alpha: Beta分布参数

    Returns:
        mixed_image, mixed_boxes, mixed_labels, lambda_value
    """
    # 生成lambda值
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    # 混合图像
    mixed_image = lam * image1 + (1 - lam) * image2

    # 混合边界框和标签
    mixed_boxes = np.vstack([boxes1, boxes2])
    mixed_labels = np.hstack([labels1, labels2])

    return mixed_image, mixed_boxes, mixed_labels, lam


def cutmix_augmentation(
    image1: np.ndarray,
    boxes1: np.ndarray,
    labels1: np.ndarray,
    image2: np.ndarray,
    boxes2: np.ndarray,
    labels2: np.ndarray,
    alpha: float = 1.0
) -> tuple:
    """
    CutMix数据增强

    Args:
        image1: 第一张图像
        boxes1: 第一张图像的边界框
        labels1: 第一张图像的标签
        image2: 第二张图像
        boxes2: 第二张图像的边界框
        labels2: 第二张图像的标签
        alpha: Beta分布参数

    Returns:
        cutmix_image, cutmix_boxes, cutmix_labels
    """
    h, w = image1.shape[:2]

    # 生成裁剪区域
    lam = np.random.beta(alpha, alpha)
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)

    # 随机选择裁剪位置
    cx = np.random.randint(w)
    cy = np.random.randint(h)

    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)

    # 创建CutMix图像
    cutmix_image = image1.copy()
    cutmix_image[bby1:bby2, bbx1:bbx2] = image2[bby1:bby2, bbx1:bbx2]

    # 计算lambda
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))

    # 合并边界框和标签
    cutmix_boxes = np.vstack([boxes1, boxes2])
    cutmix_labels = np.hstack([labels1, labels2])

    return cutmix_image, cutmix_boxes, cutmix_labels
