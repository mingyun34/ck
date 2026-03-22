# 包初始化文件
"""
无人机检测与跟踪系统
包含Faster R-CNN检测和DeepSORT跟踪功能
"""

# 版本信息
__version__ = '1.0.0'
__author__ = 'Your Name'
__email__ = 'your.email@example.com'

# 导出主要类和函数
from .models.faster_rcnn import create_baseline_model, create_improved_model
from .models.tracker import DroneTracker, Track, Detection, draw_tracks
from .utils.dataset_yolo import YOLODroneDataset, create_yolo_datasets
from .utils.config_yolo import YOLOConfig
from .utils.visualizer import draw_bboxes, VisualizationConfig
from .utils.trainer import FasterRCNNTrainer