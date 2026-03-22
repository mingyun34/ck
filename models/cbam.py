"""
CBAM (Convolutional Block Attention Module) 注意力模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """通道注意力模块"""

    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        """
        Args:
            in_channels: 输入通道数
            reduction_ratio: 降维比例
        """
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征图，形状为 (B, C, H, W)

        Returns:
            经过通道注意力加权的特征图
        """
        b, c, _, _ = x.size()

        # 平均池化和最大池化
        avg_out = self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)
        max_out = self.fc(self.max_pool(x).view(b, c)).view(b, c, 1, 1)

        # 通道注意力
        attention = self.sigmoid(avg_out + max_out)

        # 应用注意力
        return x * attention


class SpatialAttention(nn.Module):
    """空间注意力模块"""

    def __init__(self, kernel_size: int = 7):
        """
        Args:
            kernel_size: 卷积核大小
        """
        super().__init__()

        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征图，形状为 (B, C, H, W)

        Returns:
            经过空间注意力加权的特征图
        """
        # 在通道维度上进行平均和最大池化
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # 拼接并卷积
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(attention)
        attention = self.sigmoid(attention)

        # 应用注意力
        return x * attention


class CBAM(nn.Module):
    """卷积块注意力模块 (Channel and Spatial Attention Module)"""

    def __init__(self, in_channels: int, reduction_ratio: int = 16, kernel_size: int = 7):
        """
        Args:
            in_channels: 输入通道数
            reduction_ratio: 通道注意力降维比例
            kernel_size: 空间注意力卷积核大小
        """
        super().__init__()

        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征图，形状为 (B, C, H, W)

        Returns:
            经过CBAM注意力加权的特征图
        """
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class ResidualCBAMBlock(nn.Module):
    """带CBAM注意力的残差块"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        """
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            stride: 步长
        """
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # CBAM注意力模块
        self.cbam = CBAM(out_channels)

        # 残差连接
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 应用CBAM注意力
        out = self.cbam(out)

        # 残差连接
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out