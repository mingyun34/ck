"""
YOLO格式数据集配置类
管理数据集路径、类别信息、训练参数等
"""

import os
import yaml
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np

class YOLOConfig:
    """YOLO数据集配置类"""

    # 项目根目录
    ROOT_DIR = Path(__file__).parent.parent

    # 数据目录
    DATA_DIR = ROOT_DIR / 'data'
    IMAGE_DIR = DATA_DIR / 'images'
    LABEL_DIR = DATA_DIR / 'labels'

    # 各个子集目录
    TRAIN_IMAGE_DIR = IMAGE_DIR / 'train'
    VAL_IMAGE_DIR = IMAGE_DIR / 'val'
    TEST_IMAGE_DIR = IMAGE_DIR / 'test'
    TRAIN_LABEL_DIR = LABEL_DIR / 'train'
    VAL_LABEL_DIR = LABEL_DIR / 'val'
    TEST_LABEL_DIR = LABEL_DIR / 'test'

    # 模型相关目录
    MODEL_DIR = ROOT_DIR / 'models'
    WEIGHTS_DIR = ROOT_DIR / 'weights'
    LOGS_DIR = ROOT_DIR / 'logs'
    OUTPUT_DIR = ROOT_DIR / 'outputs'
    CONFIGS_DIR = ROOT_DIR / 'configs'
    VISUALIZATION_DIR = OUTPUT_DIR / 'visualizations'

    # YOLO类别配置
    NUM_CLASSES = 1
    CLASS_NAMES = ['drone']

    # 训练配置
    SEED = 42  # 随机种子，保证实验可复现
    BATCH_SIZE = 4  # 降低batch size以适应CPU训练
    NUM_WORKERS = 0  # Windows下设为0
    EPOCHS = 50  # epoch数量
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 0.0005
    LR_MIN = 1e-6
    OPTIMIZER = 'AdamW'
    DEVICE = 'cuda'
    EVAL_INTERVAL = 5  # 每隔5轮评估一次
    SAVE_INTERVAL = 10  # 每隔10轮保存一次检查点

    # 数据增强配置
    USE_MOSAIC = True
    MOSAIC_PROB = 0.5
    IMG_SIZE = 640

    # 输入配置
    INPUT_SIZE = (640, 640)

    @classmethod
    def create_directories(cls):
        """创建所有必要的目录"""
        dirs_to_create = [
            cls.DATA_DIR,
            cls.IMAGE_DIR,
            cls.LABEL_DIR,
            cls.TRAIN_IMAGE_DIR,
            cls.VAL_IMAGE_DIR,
            cls.TEST_IMAGE_DIR,
            cls.TRAIN_LABEL_DIR,
            cls.VAL_LABEL_DIR,
            cls.TEST_LABEL_DIR,
            cls.MODEL_DIR,
            cls.WEIGHTS_DIR,
            cls.LOGS_DIR,
            cls.OUTPUT_DIR,
            cls.CONFIGS_DIR,
            cls.VISUALIZATION_DIR,
        ]

        for dir_path in dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def check_dataset(cls) -> bool:
        """检查数据集是否完整"""
        print(f"检查数据集...")

        # 检查训练集
        train_images = list(cls.TRAIN_IMAGE_DIR.glob('*.jpg')) + \
                       list(cls.TRAIN_IMAGE_DIR.glob('*.png')) + \
                       list(cls.TRAIN_IMAGE_DIR.glob('*.jpeg'))
        train_labels = list(cls.TRAIN_LABEL_DIR.glob('*.txt'))

        print(f"训练集: {len(train_images)} 张图像, {len(train_labels)} 个标签")

        # 检查验证集
        val_images = list(cls.VAL_IMAGE_DIR.glob('*.jpg')) + \
                     list(cls.VAL_IMAGE_DIR.glob('*.png')) + \
                     list(cls.VAL_IMAGE_DIR.glob('*.jpeg'))
        val_labels = list(cls.VAL_LABEL_DIR.glob('*.txt'))

        print(f"验证集: {len(val_images)} 张图像, {len(val_labels)} 个标签")

        # 检查测试集（可选）
        test_images = list(cls.TEST_IMAGE_DIR.glob('*.jpg')) + \
                      list(cls.TEST_IMAGE_DIR.glob('*.png')) + \
                      list(cls.TEST_IMAGE_DIR.glob('*.jpeg'))
        test_labels = list(cls.TEST_LABEL_DIR.glob('*.txt'))

        print(f"测试集: {len(test_images)} 张图像, {len(test_labels)} 个标签")

        # 至少要有训练集和验证集
        if len(train_images) == 0 or len(val_images) == 0:
            return False

        return True

    @classmethod
    def analyze_yolo_labels(cls, split: str = 'train'):
        """
        分析YOLO格式标签统计信息

        Args:
            split: 数据集分割 ('train', 'val', 'test')
        """
        if split == 'train':
            label_dir = cls.TRAIN_LABEL_DIR
        elif split == 'val':
            label_dir = cls.VAL_LABEL_DIR
        elif split == 'test':
            label_dir = cls.TEST_LABEL_DIR
        else:
            raise ValueError(f"Invalid split: {split}")

        label_files = list(label_dir.glob('*.txt'))
        if len(label_files) == 0:
            print(f"  没有找到 {split} 标签文件")
            return

        print(f"分析 {split} 标签...")

        all_boxes = []
        total_boxes = 0

        for label_file in label_files:
            with open(label_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])

                        all_boxes.append([x_center, y_center, width, height])
                        total_boxes += 1

        if total_boxes == 0:
            print(f"  没有找到有效的标注")
            return

        all_boxes = np.array(all_boxes)

        # 统计信息
        widths = all_boxes[:, 2]
        heights = all_boxes[:, 3]
        areas = widths * heights

        print(f"标注文件数: {len(label_files)}")
        print(f"总标注框数: {total_boxes}")
        print(f"平均每张图像: {total_boxes / len(label_files):.2f} 个框")

        print(f"标注框宽度: 最小: {widths.min():.4f} 最大: {widths.max():.4f} 平均: {widths.mean():.4f}")
        print(f"标注框高度: 最小: {heights.min():.4f} 最大: {heights.max():.4f} 平均: {heights.mean():.4f}")
        print(f"标注框面积: 最小: {areas.min():.6f} 最大: {areas.max():.6f} 平均: {areas.mean():.6f}")

        # 小目标比例（面积 < 0.01）
        small_objects = (areas < 0.01).sum()
        small_ratio = small_objects / total_boxes * 100
        print(f"小目标比例: {small_ratio:.1f}%")

    @classmethod
    def save_to_yaml(cls, config_path: Optional[str] = None):
        """
        保存配置到YAML文件

        Args:
            config_path: 配置文件路径，默认为 configs/yolo_config.yaml
        """
        if config_path is None:
            config_path = cls.CONFIGS_DIR / 'yolo_config.yaml'

        config_dict = {
            'data': {
                'root_dir': str(cls.ROOT_DIR),
                'data_dir': str(cls.DATA_DIR),
                'image_dir': str(cls.IMAGE_DIR),
                'label_dir': str(cls.LABEL_DIR),
            },
            'dataset': {
                'num_classes': cls.NUM_CLASSES,
                'class_names': cls.CLASS_NAMES,
                'train_images': str(cls.TRAIN_IMAGE_DIR),
                'train_labels': str(cls.TRAIN_LABEL_DIR),
                'val_images': str(cls.VAL_IMAGE_DIR),
                'val_labels': str(cls.VAL_LABEL_DIR),
                'test_images': str(cls.TEST_IMAGE_DIR),
                'test_labels': str(cls.TEST_LABEL_DIR),
            },
            'training': {
                'batch_size': cls.BATCH_SIZE,
                'num_workers': cls.NUM_WORKERS,
                'epochs': cls.EPOCHS,
                'learning_rate': cls.LEARNING_RATE,
                'weight_decay': cls.WEIGHT_DECAY,
                'lr_min': cls.LR_MIN,
                'optimizer': cls.OPTIMIZER,
                'device': cls.DEVICE,
            },
            'augmentation': {
                'use_mosaic': cls.USE_MOSAIC,
                'mosaic_prob': cls.MOSAIC_PROB,
                'img_size': cls.IMG_SIZE,
            },
            'paths': {
                'model_dir': str(cls.MODEL_DIR),
                'weights_dir': str(cls.WEIGHTS_DIR),
                'logs_dir': str(cls.LOGS_DIR),
                'output_dir': str(cls.OUTPUT_DIR),
                'visualization_dir': str(cls.VISUALIZATION_DIR),
            }
        }

        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)

        print(f"配置已保存到: {config_path}")

    @classmethod
    def load_from_yaml(cls, config_path: str):
        """
        从YAML文件加载配置

        Args:
            config_path: 配置文件路径
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)

        # 更新配置
        if 'data' in config_dict:
            cls.ROOT_DIR = Path(config_dict['data'].get('root_dir', cls.ROOT_DIR))
            cls.DATA_DIR = Path(config_dict['data'].get('data_dir', cls.DATA_DIR))
            cls.IMAGE_DIR = Path(config_dict['data'].get('image_dir', cls.IMAGE_DIR))
            cls.LABEL_DIR = Path(config_dict['data'].get('label_dir', cls.LABEL_DIR))

        if 'dataset' in config_dict:
            cls.NUM_CLASSES = config_dict['dataset'].get('num_classes', cls.NUM_CLASSES)
            cls.CLASS_NAMES = config_dict['dataset'].get('class_names', cls.CLASS_NAMES)
            cls.TRAIN_IMAGE_DIR = Path(config_dict['dataset'].get('train_images', cls.TRAIN_IMAGE_DIR))
            cls.TRAIN_LABEL_DIR = Path(config_dict['dataset'].get('train_labels', cls.TRAIN_LABEL_DIR))
            cls.VAL_IMAGE_DIR = Path(config_dict['dataset'].get('val_images', cls.VAL_IMAGE_DIR))
            cls.VAL_LABEL_DIR = Path(config_dict['dataset'].get('val_labels', cls.VAL_LABEL_DIR))
            cls.TEST_IMAGE_DIR = Path(config_dict['dataset'].get('test_images', cls.TEST_IMAGE_DIR))
            cls.TEST_LABEL_DIR = Path(config_dict['dataset'].get('test_labels', cls.TEST_LABEL_DIR))

        if 'training' in config_dict:
            cls.BATCH_SIZE = config_dict['training'].get('batch_size', cls.BATCH_SIZE)
            cls.NUM_WORKERS = config_dict['training'].get('num_workers', cls.NUM_WORKERS)
            cls.EPOCHS = config_dict['training'].get('epochs', cls.EPOCHS)
            cls.LEARNING_RATE = config_dict['training'].get('learning_rate', cls.LEARNING_RATE)
            cls.WEIGHT_DECAY = config_dict['training'].get('weight_decay', cls.WEIGHT_DECAY)
            cls.LR_MIN = config_dict['training'].get('lr_min', cls.LR_MIN)
            cls.OPTIMIZER = config_dict['training'].get('optimizer', cls.OPTIMIZER)
            cls.DEVICE = config_dict['training'].get('device', cls.DEVICE)

        if 'augmentation' in config_dict:
            cls.USE_MOSAIC = config_dict['augmentation'].get('use_mosaic', cls.USE_MOSAIC)
            cls.MOSAIC_PROB = config_dict['augmentation'].get('mosaic_prob', cls.MOSAIC_PROB)
            cls.IMG_SIZE = config_dict['augmentation'].get('img_size', cls.IMG_SIZE)

        if 'paths' in config_dict:
            cls.MODEL_DIR = Path(config_dict['paths'].get('model_dir', cls.MODEL_DIR))
            cls.WEIGHTS_DIR = Path(config_dict['paths'].get('weights_dir', cls.WEIGHTS_DIR))
            cls.LOGS_DIR = Path(config_dict['paths'].get('logs_dir', cls.LOGS_DIR))
            cls.OUTPUT_DIR = Path(config_dict['paths'].get('output_dir', cls.OUTPUT_DIR))
            cls.VISUALIZATION_DIR = Path(config_dict['paths'].get('visualization_dir', cls.VISUALIZATION_DIR))