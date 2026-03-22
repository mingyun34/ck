"""
Faster R-CNN模型定义
支持基线模型和改进模型
"""

import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def create_baseline_model(num_classes: int = 2, pretrained: bool = True):
    """
    创建基线Faster R-CNN模型

    Args:
        num_classes: 类别数（背景 + 目标）
        pretrained: 是否使用预训练权重

    Returns:
        model: Faster R-CNN模型
    """
    # 加载预训练的Faster R-CNN模型
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=pretrained,
        pretrained_backbone=pretrained
    )

    # 替换分类头
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        num_classes
    )

    return model


def create_improved_model(num_classes: int = 2, pretrained: bool = True, use_cbam: bool = False):
    """
    创建改进的Faster R-CNN模型（预留接口）

    Args:
        num_classes: 类别数
        pretrained: 是否使用预训练权重
        use_cbam: 是否使用CBAM注意力模块

    Returns:
        model: 改进的Faster R-CNN模型
    """
    # 加载基础模型
    model = create_baseline_model(num_classes, pretrained)

    # TODO: 添加改进模块
    # if use_cbam:
    #     model = add_cbam(model)

    return model


def get_model_summary(model: nn.Module, input_size: tuple = (3, 640, 640)):
    """
    获取模型摘要信息

    Args:
        model: 模型
        input_size: 输入尺寸 (C, H, W)
    """
    # 统计参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n" + "=" * 70)
    print("  模型摘要")
    print("=" * 70)
    print(f"模型类型: {model.__class__.__name__}")
    print(f"输入尺寸: {input_size}")
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    print(f"固定参数数量: {total_params - trainable_params:,}")
    print("=" * 70 + "\n")
