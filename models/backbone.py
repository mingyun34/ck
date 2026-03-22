"""
骨干网络定义（预留接口）
支持ResNet, MobileNet等多种骨干网络
"""

import torch
import torch.nn as nn
import torchvision.models as models


class BackboneType:
    """骨干网络类型枚举"""
    RESNET50 = 'resnet50'
    RESNET101 = 'resnet101'
    MOBILENET_V2 = 'mobilenet_v2'
    MOBILENET_V3_SMALL = 'mobilenet_v3_small'
    MOBILENET_V3_LARGE = 'mobilenet_v3_large'


def get_backbone(backbone_type: str = BackboneType.RESNET50, pretrained: bool = True):
    """
    获取骨干网络

    Args:
        backbone_type: 骨干网络类型
        pretrained: 是否使用预训练权重

    Returns:
        backbone: 骨干网络
        out_channels: 输出通道数
    """
    if backbone_type == BackboneType.RESNET50:
        model = models.resnet50(pretrained=pretrained)
        # 返回特征提取部分（去除最后的全连接层）
        backbone = nn.Sequential(*list(model.children())[:-2])
        out_channels = 2048  # ResNet50最后一层输出通道数

    elif backbone_type == BackboneType.RESNET101:
        model = models.resnet101(pretrained=pretrained)
        backbone = nn.Sequential(*list(model.children())[:-2])
        out_channels = 2048

    elif backbone_type == BackboneType.MOBILENET_V2:
        model = models.mobilenet_v2(pretrained=pretrained)
        backbone = model.features
        out_channels = 1280

    elif backbone_type == BackboneType.MOBILENET_V3_SMALL:
        model = models.mobilenet_v3_small(pretrained=pretrained)
        backbone = model.features
        out_channels = 576

    elif backbone_type == BackboneType.MOBILENET_V3_LARGE:
        model = models.mobilenet_v3_large(pretrained=pretrained)
        backbone = model.features
        out_channels = 960

    else:
        raise ValueError(f"不支持的骨干网络类型: {backbone_type}")

    return backbone, out_channels


def freeze_backbone(backbone: nn.Module, freeze_at: int = 2):
    """
    冻结骨干网络部分层

    Args:
        backbone: 骨干网络
        freeze_at: 冻结前几层（从0开始计数）
    """
    if freeze_at < 0:
        return

    # 遍历骨干网络的子模块
    for i, child in enumerate(backbone.children()):
        if i < freeze_at:
            for param in child.parameters():
                param.requires_grad = False
        else:
            for param in child.parameters():
                param.requires_grad = True


class BackboneWithFPN(nn.Module):
    """带FPN的骨干网络（预留）"""
    def __init__(self, backbone_type: str = BackboneType.RESNET50, pretrained: bool = True):
        super().__init__()
        self.backbone, self.out_channels = get_backbone(backbone_type, pretrained)
        # TODO: 添加FPN结构

    def forward(self, x):
        return self.backbone(x)