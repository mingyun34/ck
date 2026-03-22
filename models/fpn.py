"""
特征金字塔网络 (Feature Pyramid Network, FPN)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class FPN(nn.Module):
    """
    特征金字塔网络

    从不同层次的特征图构建金字塔特征
    """

    def __init__(
            self,
            in_channels_list: List[int],
            out_channels: int,
            extra_blocks=None
    ):
        """
        Args:
            in_channels_list: 输入特征图的通道数列表
            out_channels: 输出特征图的通道数
            extra_blocks: 额外的块（如用于RPN的侧向连接）
        """
        super().__init__()

        # 侧向连接卷积层
        self.lateral_convs = nn.ModuleList()
        # 顶部卷积层
        self.top_down_convs = nn.ModuleList()

        for in_channels in in_channels_list:
            lateral_conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0
            )
            top_down_conv = nn.Conv2d(
                out_channels, out_channels, kernel_size=3, stride=1, padding=1
            )

            self.lateral_convs.append(lateral_conv)
            self.top_down_convs.append(top_down_conv)

        # 初始化权重
        self._init_weights()

        self.extra_blocks = extra_blocks

    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            x: 不同层次的特征图列表，从低层到高层

        Returns:
            金字塔特征图列表，从高层到低层
        """
        # 侧向连接
        laterals = [
            lateral_conv(x[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # 自顶向下路径
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # 上采样
            laterals[i - 1] += F.interpolate(
                laterals[i], size=laterals[i - 1].shape[2:], mode='nearest'
            )

        # 应用顶部卷积
        results = [
            self.top_down_convs[i](laterals[i])
            for i in range(used_backbone_levels)
        ]

        # 如果有额外块（如用于RPN）
        if self.extra_blocks is not None:
            results = self.extra_blocks(results, x)

        return results


class LastLevelMaxPool(nn.Module):
    """
    最后一层最大池化，用于生成额外的金字塔层
    """

    def forward(self, x: List[torch.Tensor], y: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            x: FPN输出的特征图列表
            y: 骨干网络输出的特征图列表

        Returns:
            添加了最大池化层的特征图列表
        """
        x.append(F.max_pool2d(x[-1], 1, 2, 0))
        return x


class FPNWithCBAM(FPN):
    """带CBAM注意力的FPN（预留）"""

    def __init__(
            self,
            in_channels_list: List[int],
            out_channels: int,
            extra_blocks=None
    ):
        super().__init__(in_channels_list, out_channels, extra_blocks)

        # 添加CBAM注意力模块
        from .cbam import CBAM
        self.cbam_blocks = nn.ModuleList([
            CBAM(out_channels) for _ in range(len(in_channels_list))
        ])

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        """前向传播，添加CBAM注意力"""
        results = super().forward(x)

        # 应用CBAM注意力
        results = [
            self.cbam_blocks[i](results[i])
            for i in range(len(results))
        ]

        return results