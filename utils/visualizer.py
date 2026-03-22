# 可视化工具（完整版）
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class VisualizationConfig:
    """可视化配置类"""
    # 边界框配置
    bbox_thickness: int = 2
    bbox_alpha: float = 0.8

    # 文本配置
    font_scale: float = 0.5
    font_thickness: int = 2
    text_padding: int = 5

    # 颜色配置
    colors: List[Tuple[int, int, int]] = None
    text_color: Tuple[int, int, int] = (255, 255, 255)

    # 显示配置
    show_scores: bool = True
    score_threshold: float = 0.0
    class_names: List[str] = None

    def __post_init__(self):
        if self.colors is None:
            # 默认颜色方案
            self.colors = [
                (0, 255, 0),  # 绿色
                (0, 0, 255),  # 红色
                (255, 0, 0),  # 蓝色
                (255, 255, 0),  # 青色
                (255, 0, 255),  # 洋红
                (0, 255, 255),  # 黄色
                (128, 0, 128),  # 紫色
                (255, 165, 0),  # 橙色
                (255, 192, 203),  # 粉色
                (0, 128, 128),  # 深青色
            ]


def draw_bboxes(
        image: np.ndarray,
        bboxes: List[List[float]],
        labels: List[int],
        scores: Optional[List[float]] = None,
        class_names: Optional[List[str]] = None,
        config: Optional[VisualizationConfig] = None
) -> np.ndarray:
    """
    在图像上绘制边界框

    Args:
        image: 输入图像 (BGR格式)
        bboxes: 边界框列表, 每个框为 [x1, y1, x2, y2]
        labels: 标签列表
        scores: 置信度分数列表 (可选)
        class_names: 类别名称列表 (可选)
        config: 可视化配置 (可选)

    Returns:
        绘制后的图像
    """
    if config is None:
        config = VisualizationConfig()

    # 复制图像以避免修改原图
    image_vis = image.copy()

    # 处理类别名称
    if class_names is None:
        class_names = [f"Class {i}" for i in range(100)]

    # 绘制每个边界框
    for i, (bbox, label) in enumerate(zip(bboxes, labels)):
        # 过滤低置信度
        if scores is not None and scores[i] < config.score_threshold:
            continue

        x1, y1, x2, y2 = map(int, bbox)

        # 选择颜色
        color = config.colors[label % len(config.colors)]

        # 绘制边界框
        cv2.rectangle(
            image_vis,
            (x1, y1),
            (x2, y2),
            color,
            config.bbox_thickness
        )

        # 准备标签文本
        class_name = class_names[label] if label < len(class_names) else f"Class {label}"
        label_text = f"{class_name}"

        if scores is not None and config.show_scores:
            label_text += f" {scores[i]:.2f}"

        # 计算文本区域大小
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            config.font_scale,
            config.font_thickness
        )

        # 绘制文本背景
        text_bg_x1 = x1
        text_bg_y1 = y1 - text_height - 2 * config.text_padding
        text_bg_x2 = x1 + text_width + 2 * config.text_padding
        text_bg_y2 = y1

        # 确保文本背景在图像范围内
        text_bg_y1 = max(0, text_bg_y1)

        cv2.rectangle(
            image_vis,
            (text_bg_x1, text_bg_y1),
            (text_bg_x2, text_bg_y2),
            color,
            -1  # 填充
        )

        # 绘制文本
        cv2.putText(
            image_vis,
            label_text,
            (x1 + config.text_padding, y1 - config.text_padding),
            cv2.FONT_HERSHEY_SIMPLEX,
            config.font_scale,
            config.text_color,
            config.font_thickness,
            cv2.LINE_AA
        )

    return image_vis


def visualize_dataset_sample(
        image: np.ndarray,
        bboxes: List[List[float]],
        labels: List[int],
        scores: Optional[List[float]] = None,
        class_names: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        show: bool = False,
        config: Optional[VisualizationConfig] = None
) -> np.ndarray:
    """
    可视化数据集样本（通用函数）

    Args:
        image: 输入图像 (可以是Tensor或numpy数组)
        bboxes: 边界框列表, 格式为 [x1, y1, x2, y2]
        labels: 标签列表
        scores: 置信度分数列表 (可选)
        class_names: 类别名称列表 (可选)
        save_path: 保存路径 (可选)
        show: 是否显示图像 (可选)
        config: 可视化配置 (可选)

    Returns:
        可视化后的图像
    """
    # 转换Tensor为numpy数组
    if torch.is_tensor(image):
        # 如果是CHW格式，转换为HWC
        if len(image.shape) == 3 and image.shape[0] == 3:
            image = image.permute(1, 2, 0)

        image = image.cpu().numpy()

        # 反归一化（假设是ImageNet归一化）
        if image.max() <= 1.0:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = image * std + mean
            image = np.clip(image, 0, 1)

        # 转换为0-255范围
        image = (image * 255).astype(np.uint8)

    # 转换为BGR格式
    if len(image.shape) == 3 and image.shape[2] == 3:
        # 假设输入是RGB，转换为BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 绘制边界框
    image_vis = draw_bboxes(image, bboxes, labels, scores, class_names, config)

    # 保存图像
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(save_path, image_vis)

    # 显示图像
    if show:
        # 转换为RGB用于matplotlib显示
        image_rgb = cv2.cvtColor(image_vis, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(12, 8))
        plt.imshow(image_rgb)
        plt.axis('off')
        plt.title(f"Detected {len(bboxes)} objects")
        plt.tight_layout()
        plt.show()

    return image_vis


def visualize_detections(image, boxes, labels, scores=None, class_names=None, threshold=0.5):
    """
    可视化检测结果（向后兼容函数）

    Args:
        image: 图像（Tensor或numpy数组）
        boxes: 边界框列表
        labels: 标签列表
        scores: 置信度分数列表
        class_names: 类别名称列表
        threshold: 置信度阈值
    """
    config = VisualizationConfig(score_threshold=threshold, show_scores=True)

    # 转换为列表格式
    if torch.is_tensor(boxes):
        boxes_list = boxes.tolist()
    else:
        boxes_list = boxes.tolist() if hasattr(boxes, 'tolist') else list(boxes)

    if torch.is_tensor(labels):
        labels_list = labels.tolist()
    else:
        labels_list = labels.tolist() if hasattr(labels, 'tolist') else list(labels)

    scores_list = None
    if scores is not None:
        if torch.is_tensor(scores):
            scores_list = scores.tolist()
        else:
            scores_list = scores.tolist() if hasattr(scores, 'tolist') else list(scores)

    return draw_bboxes(image, boxes_list, labels_list, scores_list, class_names, config)


def save_visualization(image, save_path):
    """保存可视化结果"""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(save_path, image)