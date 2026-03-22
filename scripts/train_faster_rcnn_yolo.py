"""
Faster R-CNN训练脚本 - YOLO格式数据集版本
支持基线模型和改进模型训练
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json

# 添加项目根目录到路径
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from drone.utils.config_yolo import YOLOConfig
from drone.utils.dataset_yolo import create_yolo_datasets, collate_fn
from drone.utils.augment import get_train_transforms_v3, get_val_transforms
from drone.models.faster_rcnn import create_baseline_model


def create_baseline_model_wrapper(num_classes=2):
    """
    创建基线Faster R-CNN模型（包装函数）

    Args:
        num_classes: 类别数（背景 + 无人机 = 2）

    Returns:
        model: Faster R-CNN模型
    """
    return create_baseline_model(num_classes=num_classes, pretrained=True)


def train_one_epoch(model, data_loader, optimizer, device, epoch):
    """训练一个epoch"""
    model.train()

    total_loss = 0
    loss_dict_accum = {}

    pbar = tqdm(data_loader, desc=f"Epoch {epoch} [Train]")

    for images, targets in pbar:
        # 移动到设备
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # 前向传播
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # 反向传播
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # 累积损失
        total_loss += losses.item()
        for k, v in loss_dict.items():
            if k not in loss_dict_accum:
                loss_dict_accum[k] = []
            loss_dict_accum[k].append(v.item())

        # 更新进度条
        pbar.set_postfix({
            'loss': f'{losses.item():.4f}',
            'cls_loss': f'{loss_dict["loss_classifier"]:.4f}',
            'box_loss': f'{loss_dict["loss_box_reg"]:.4f}'
        })

    # 计算平均损失
    avg_loss = total_loss / len(data_loader)
    avg_loss_dict = {k: np.mean(v) for k, v in loss_dict_accum.items()}

    return avg_loss, avg_loss_dict


def evaluate(model, data_loader, device):
    """评估模型"""
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        pbar = tqdm(data_loader, desc="[Eval]")

        for images, targets in pbar:
            images = [img.to(device) for img in images]

            # 推理
            outputs = model(images)

            # 收集预测和目标
            for output, target in zip(outputs, targets):
                all_preds.append(output)
                all_targets.append(target)

    # 计算mAP（简化版本）
    # 实际应用中应该使用pycocotools进行完整评估
    # 这里只是示例
    return {}


def save_checkpoint(model, optimizer, epoch, loss, save_path):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }

    torch.save(checkpoint, save_path)
    print(f"  检查点已保存: {save_path}")


def train_baseline():
    """训练基线模型"""
    print("=" * 70)
    print("  Faster R-CNN 基线模型训练")
    print("=" * 70)

    # 配置
    config = YOLOConfig
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")

    # 新增：检查数据集路径
    print("\n检查数据集路径...")
    print(f"  DATA_DIR: {config.DATA_DIR}")
    print(f"  IMAGE_DIR: {config.IMAGE_DIR}")
    print(f"  LABEL_DIR: {config.LABEL_DIR}")

    # 新增：创建必要的目录
    print("\n创建必要的目录...")
    os.makedirs(config.IMAGE_DIR, exist_ok=True)
    os.makedirs(config.LABEL_DIR, exist_ok=True)
    for subset in ['train', 'val', 'test']:
        os.makedirs(os.path.join(config.IMAGE_DIR, subset), exist_ok=True)
        os.makedirs(os.path.join(config.LABEL_DIR, subset), exist_ok=True)
    print("  ✓ 目录创建完成")

    train_dataset, val_dataset, test_dataset = create_yolo_datasets(
        base_dir=config.DATA_DIR,  # 使用配置文件中的绝对路径
        transforms_train=get_train_transforms_v3(),
        transforms_val=get_val_transforms(),
        use_mosaic=config.USE_MOSAIC
    )

    print(f"训练集: {len(train_dataset)} 张图像")
    print(f"验证集: {len(val_dataset)} 张图像")

    # 计算数据集比例
    total = len(train_dataset) + len(val_dataset)
    if test_dataset is not None:
        total += len(test_dataset)
        print(f"测试集: {len(test_dataset)} 张图像")

    train_ratio = len(train_dataset) / total
    val_ratio = len(val_dataset) / total
    test_ratio = len(test_dataset) / total if test_dataset is not None else 0

    print("\n数据集比例:")
    print(f"  训练集: {train_ratio:.2%}")
    print(f"  验证集: {val_ratio:.2%}")
    print(f"  测试集: {test_ratio:.2%}")
    print(f"  目标比例: 训练集 70%, 验证集 20%, 测试集 10%")

    # 创建DataLoader
    print("\n创建DataLoader...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn,
        drop_last=False
    )

    print(f"训练批次: {len(train_loader)}")
    print(f"验证批次: {len(val_loader)}")

    # 创建模型
    print("\n创建Faster R-CNN模型...")
    # Faster R-CNN需要: 背景类 + 目标类
    num_classes = config.NUM_CLASSES + 1
    model = create_baseline_model(num_classes=num_classes)
    model.to(device)

    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.EPOCHS,
        eta_min=config.LR_MIN
    )

    print(f"优化器: {config.OPTIMIZER}")
    print(f"学习率: {config.LEARNING_RATE}")
    print(f"训练轮数: {config.EPOCHS}")

    # 创建保存目录
    os.makedirs(config.WEIGHTS_DIR, exist_ok=True)
    os.makedirs(config.LOGS_DIR, exist_ok=True)

    # 训练记录
    train_history = {
        'loss': [],
        'epoch': []
    }

    # 开始训练
    print("\n" + "=" * 70)
    print("开始训练")
    print("=" * 70)

    best_loss = float('inf')

    for epoch in range(1, config.EPOCHS + 1):
        print(f"\nEpoch {epoch}/{config.EPOCHS}")
        print("-" * 70)

        # 训练
        avg_loss, loss_dict = train_one_epoch(
            model, train_loader, optimizer, device, epoch
        )

        print(f"\n训练损失: {avg_loss:.4f}")
        print("  详细损失:")
        for k, v in loss_dict.items():
            print(f"    {k}: {v:.4f}")

        # 保存训练记录
        train_history['loss'].append(avg_loss)
        train_history['epoch'].append(epoch)

        # 学习率调度
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"当前学习率: {current_lr:.6f}")

        # 评估（每隔一定轮数）
        if epoch % config.EVAL_INTERVAL == 0:
            print("\n开始评估...")
            evaluate(model, val_loader, device)

        # 保存检查点
        if epoch % config.SAVE_INTERVAL == 0:
            save_path = os.path.join(
                config.WEIGHTS_DIR,
                f'baseline_epoch_{epoch}.pth'
            )
            save_checkpoint(model, optimizer, epoch, avg_loss, save_path)

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(
                config.WEIGHTS_DIR,
                'baseline_best.pth'
            )
            save_checkpoint(model, optimizer, epoch, avg_loss, save_path)
            print(f"  ✓ 保存最佳模型 (loss: {best_loss:.4f})")

        # 保存训练历史
        history_path = os.path.join(config.LOGS_DIR, 'baseline_train_history.json')
        with open(history_path, 'w') as f:
            json.dump(train_history, f, indent=2)

    print("\n" + "=" * 70)
    print("训练完成！")
    print(f"最佳损失: {best_loss:.4f}")
    print(f"最佳模型: {config.WEIGHTS_DIR}/baseline_best.pth")
    print("=" * 70)


if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(YOLOConfig.SEED)
    np.random.seed(YOLOConfig.SEED)

    try:
        train_baseline()
    except KeyboardInterrupt:
        print("\n训练被中断")
    except Exception as e:
        print(f"\n训练出错: {e}")
        import traceback
        traceback.print_exc()
