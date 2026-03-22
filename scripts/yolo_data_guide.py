"""
YOLO格式数据集完整使用指南
适配已按7:2:1划分的8000+图像数据集
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from drone.utils.config_yolo import YOLOConfig
from drone.utils.dataset_yolo import create_yolo_datasets, YOLODroneDataset, collate_fn
from drone.utils.augment import get_train_transforms_v3, get_val_transforms, AugmentationConfig
from drone.utils.visualizer import visualize_dataset_sample, VisualizationConfig, draw_bboxes


def print_section(title):
    """打印章节标题"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def step1_initialize():
    """步骤1: 初始化环境"""
    print_section("步骤1: 初始化环境")

    print("正在创建项目目录结构...")
    YOLOConfig.create_directories()

    print("\n项目目录结构:")
    print(f"  根目录: {YOLOConfig.ROOT_DIR}")
    print(f"  数据目录: {YOLOConfig.DATA_DIR}")
    print(f"    - images/train/")
    print(f"    - images/val/")
    print(f"    - images/test/")
    print(f"    - labels/train/")
    print(f"    - labels/val/")
    print(f"    - labels/test/")

    print(f"\n  模型目录: {YOLOConfig.MODEL_DIR}")
    print(f"  权重目录: {YOLOConfig.WEIGHTS_DIR}")
    print(f"  日志目录: {YOLOConfig.LOGS_DIR}")
    print(f"  输出目录: {YOLOConfig.OUTPUT_DIR}")

    print("\n✓ 环境初始化完成！")
    return True


def step2_check_dataset():
    """步骤2: 检查数据集"""
    print_section("步骤2: 检查数据集")

    print("正在检查数据集...")
    dataset_ok = YOLOConfig.check_dataset()

    if not dataset_ok:
        print("\n⚠ 数据集检查失败！")
        print("\n请确保你的数据集目录结构如下：")
        print("""
data/
├── images/
│   ├── train/      # 训练图像 (~5600张)
│   ├── val/        # 验证图像 (~1600张)
│   └── test/       # 测试图像 (~800张，可选)
└── labels/
    ├── train/      # 训练标签 (.txt文件)
    ├── val/        # 验证标签 (.txt文件)
    └── test/       # 测试标签 (.txt文件，可选）
        """)
        return False

    print("\n✓ 数据集检查通过！")

    # 分析标签统计
    print("\n正在分析训练集标签...")
    YOLOConfig.analyze_yolo_labels('train')

    print("\n正在分析验证集标签...")
    YOLOConfig.analyze_yolo_labels('val')

    return True


def step3_create_datasets():
    """步骤3: 创建数据集对象"""
    print_section("步骤3: 创建数据集对象")

    print("正在创建数据集对象...")

    # 创建数据增强
    print("\n加载训练集数据增强...")
    train_transforms = get_train_transforms_v3()
    print(f"  使用增强策略: v3")
    print(f"  包含: 水平翻转、旋转、颜色抖动、Mosaic等")

    print("\n加载验证集数据增强...")
    val_transforms = get_val_transforms()
    print(f"  验证集仅使用归一化")

    # 创建数据集
    print("\n创建数据集...")
    train_dataset, val_dataset, test_dataset = create_yolo_datasets(
        base_dir='data',
        transforms_train=train_transforms,
        transforms_val=val_transforms,
        use_mosaic=YOLOConfig.USE_MOSAIC,
        mosaic_prob=YOLOConfig.MOSAIC_PROB
    )

    print(f"\n数据集创建成功！")
    print(f"  训练集: {len(train_dataset)} 张图像")
    print(f"  验证集: {len(val_dataset)} 张图像")
    if test_dataset:
        print(f"  测试集: {len(test_dataset)} 张图像")
    else:
        print(f"  测试集: 未找到（可选）")

    print(f"\n类别信息:")
    print(f"  类别数量: {train_dataset.num_classes}")
    print(f"  类别名称: {train_dataset.get_classes()}")

    return train_dataset, val_dataset, test_dataset


def step4_test_data_loading(train_dataset, val_dataset):
    """步骤4: 测试数据加载"""
    print_section("步骤4: 测试数据加载")

    if train_dataset is None or val_dataset is None:
        print("⚠ 数据集对象不存在，请先完成步骤3！")
        return False

    print("测试训练集数据加载...")
    try:
        image, target = train_dataset[0]
        print(f"  ✓ 成功加载训练样本")
        print(f"    图像形状: {image.shape}")
        print(f"    图像数据类型: {image.dtype}")
        print(f"    图像数值范围: [{image.min():.3f}, {image.max():.3f}]")
        print(f"    目标框数量: {len(target['boxes'])}")
        if len(target['boxes']) > 0:
            print(f"    标签: {target['labels'].tolist()}")
            print(f"    边界框示例: {target['boxes'][0].tolist()}")
    except Exception as e:
        print(f"  ✗ 训练集加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n测试验证集数据加载...")
    try:
        image, target = val_dataset[0]
        print(f"  ✓ 成功加载验证样本")
        print(f"    图像形状: {image.shape}")
        print(f"    目标框数量: {len(target['boxes'])}")
    except Exception as e:
        print(f"  ✗ 验证集加载失败: {e}")
        return False

    print("\n✓ 数据加载测试完成！")
    return True


def step5_visualize_samples(train_dataset, val_dataset):
    """步骤5: 可视化样本"""
    print_section("步骤5: 可视化样本")

    if train_dataset is None or val_dataset is None:
        print("⚠ 数据集对象不存在，请先完成步骤3！")
        return False

    print("可视化训练集样本（带数据增强）...")

    # 可视化几个训练样本
    num_samples = min(5, len(train_dataset))
    for i in range(num_samples):
        try:
            save_path = os.path.join(
                YOLOConfig.VISUALIZATION_DIR,
                f'train_sample_{i}.jpg'
            )
            print(f"\n  样本 {i}:")

            # 获取样本
            image, target = train_dataset[i]

            # 反归一化用于显示
            if isinstance(image, torch.Tensor):
                import torch
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                image_np = image.permute(1, 2, 0).numpy()
                image_np = image_np * std + mean
                image_np = np.clip(image_np, 0, 1)
                image_np = (image_np * 255).astype(np.uint8)
            else:
                image_np = image

            # 转换为BGR
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            # 绘制边界框
            bboxes = target['boxes'].numpy().tolist()
            labels = target['labels'].numpy().tolist()

            image_vis = draw_bboxes(
                image_np,
                bboxes,
                labels,
                class_names=train_dataset.get_class_names(),
                config=VisualizationConfig()
            )

            # 保存
            cv2.imwrite(save_path, image_vis)
            print(f"    ✓ 已保存: {save_path}")
            print(f"    目标数: {len(bboxes)}")

        except Exception as e:
            print(f"    ✗ 可视化失败: {e}")

    print("\n可视化验证集样本（无增强）...")
    try:
        save_path = os.path.join(
            YOLOConfig.VISUALIZATION_DIR,
            'val_sample_0.jpg'
        )

        # 创建不带增强的数据集
        val_dataset_no_aug = YOLODroneDataset(
            image_dir=YOLOConfig.VAL_IMAGE_DIR,
            label_dir=YOLOConfig.VAL_LABEL_DIR,
            transforms=None,
            use_mosaic=False
        )

        image, target = val_dataset_no_aug[0]

        # 转换为numpy
        if isinstance(image, torch.Tensor):
            import torch
            image_np = image.permute(1, 2, 0).numpy()
            image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image

        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # 绘制
        bboxes = target['boxes'].numpy().tolist()
        labels = target['labels'].numpy().tolist()

        image_vis = draw_bboxes(
            image_np,
            bboxes,
            labels,
            class_names=val_dataset_no_aug.get_class_names(),
            config=VisualizationConfig()
        )

        cv2.imwrite(save_path, image_vis)
        print(f"  ✓ 已保存: {save_path}")
        print(f"  目标数: {len(bboxes)}")

    except Exception as e:
        print(f"  ✗ 可视化失败: {e}")

    print("\n✓ 样本可视化完成！")
    return True


def step6_create_dataloader(train_dataset, val_dataset):
    """步骤6: 创建DataLoader"""
    print_section("步骤6: 创建DataLoader")

    if train_dataset is None or val_dataset is None:
        print("⚠ 数据集对象不存在，请先完成步骤3！")
        return False

    import torch
    from torch.utils.data import DataLoader

    print("创建训练集DataLoader...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=YOLOConfig.BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Windows下建议设为0
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=True
    )

    print(f"  ✓ 训练集批次数: {len(train_loader)}")
    print(f"    每批大小: {YOLOConfig.BATCH_SIZE}")
    print(f"    总样本数: {len(train_dataset)}")

    print("\n创建验证集DataLoader...")
    val_loader = DataLoader(
        val_dataset,
        batch_size=YOLOConfig.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        drop_last=False,
        pin_memory=True
    )

    print(f"  ✓ 验证集批次数: {len(val_loader)}")

    # 测试加载一个批次
    print("\n测试加载一个训练批次...")
    try:
        images, targets = next(iter(train_loader))
        print(f"  ✓ 成功加载训练批次")
        print(f"    批次图像形状: {images.shape}")
        print(f"    批次数据类型: {images.dtype}")
        print(f"    批次大小: {len(targets)}")

        # 统计每个样本的目标数
        for i, target in enumerate(targets):
            print(f"    样本 {i}: {len(target['boxes'])} 个目标")
    except Exception as e:
        print(f"  ✗ 批次加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n✓ DataLoader创建完成！")
    return True


def step7_save_config():
    """步骤7: 保存配置"""
    print_section("步骤7: 保存配置")

    print("保存配置文件...")
    YOLOConfig.save_to_yaml()

    print("\n✓ 配置保存完成！")


def step8_summary():
    """步骤8: 总结"""
    print_section("步骤8: 总结")

    print("\n数据准备流程已完成！")

    print("\n已创建的文件:")
    print("  ✓ utils/dataset_yolo.py - YOLO格式数据集类")
    print("  ✓ utils/config_yolo.py - YOLO数据集配置")
    print("  ✓ utils/augment.py - 数据增强工具（已存在）")
    print("  ✓ utils/visualizer.py - 可视化工具（已存在）")
    print("  ✓ scripts/yolo_data_guide.py - 本指南脚本")

    print("\n目录结构:")
    print(f"  {YOLOConfig.DATA_DIR}/")
    print(f"    ├── images/")
    print(f"    │   ├── train/     # ~5600张图像")
    print(f"    │   ├── val/       # ~1600张图像")
    print(f"    │   └── test/      # ~800张图像（可选）")
    print(f"    └── labels/")
    print(f"        ├── train/")
    print(f"        ├── val/")
    print(f"        └── test/")
    print(f"  {YOLOConfig.VISUALIZATION_DIR}/  # 可视化结果")
    print(f"  {YOLOConfig.LOGS_DIR}/          # 训练日志")
    print(f"  {YOLOConfig.WEIGHTS_DIR}/       # 模型权重")

    print("\n下一步操作:")
    print("\n1. 开始训练模型:")
    print("   python scripts/train_faster_rcnn_yolo.py")

    print("\n2. 或使用DataLoader进行自定义训练:")
    print("""
    from utils.dataset_yolo import create_yolo_datasets, collate_fn
    from utils.config_yolo import YOLOConfig
    from torch.utils.data import DataLoader

    # 创建数据集
    train_dataset, val_dataset, _ = create_yolo_datasets()

    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=YOLOConfig.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )

    # 开始训练...
    """)

    print("\n3. 查看可视化结果:")
    print(f"   训练样本: {YOLOConfig.VISUALIZATION_DIR}/train_sample_*.jpg")
    print(f"   验证样本: {YOLOConfig.VISUALIZATION_DIR}/val_sample_0.jpg")

    print("\n" + "=" * 70)
    print("  YOLO数据集准备完成！")
    print("=" * 70)


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("  YOLO格式数据集 - 完整使用指南")
    print("  适配7:2:1划分的8000+图像数据集")
    print("=" * 70)

    # 执行各个步骤
    success = True

    # 步骤1: 初始化环境
    if not step1_initialize():
        success = False

    # 步骤2: 检查数据集
    dataset_ok = step2_check_dataset()

    # 步骤3-6: 处理数据集（如果数据集存在）
    train_dataset, val_dataset, test_dataset = None, None, None
    if dataset_ok:
        train_dataset, val_dataset, test_dataset = step3_create_datasets()
        if train_dataset is not None:
            if step4_test_data_loading(train_dataset, val_dataset):
                step5_visualize_samples(train_dataset, val_dataset)
                step6_create_dataloader(train_dataset, val_dataset)

    # 步骤7: 保存配置
    step7_save_config()

    # 步骤8: 总结
    step8_summary()

    return success


if __name__ == "__main__":
    # 导入torch
    import torch

    try:
        success = main()
        if success:
            print("\n✓ 所有步骤执行成功！")
            print("\n你的数据集已经准备好，可以开始训练模型了！")
        else:
            print("\n⚠ 部分步骤需要手动检查，请参考上述说明。")
    except Exception as e:
        print(f"\n✗ 执行过程中发生错误: {e}")
        import traceback

        traceback.print_exc()