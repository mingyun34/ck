"""
验证路径修复效果
检查配置文件是否正确设置
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到路径
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from drone.utils.config_yolo import YOLOConfig

def main():
    print("=" * 70)
    print("验证路径修复效果")
    print("=" * 70)

    print(f"\n项目根目录: {ROOT_DIR}\n")

    # 显示配置的路径
    print("配置的路径:")
    print("-" * 70)
    print(f"  数据目录: {YOLOConfig.DATA_DIR}")
    print(f"  图像目录: {YOLOConfig.IMAGE_DIR}")
    print(f"  标签目录: {YOLOConfig.LABEL_DIR}\n")

    print("训练集路径:")
    print(f"  图像: {YOLOConfig.TRAIN_IMAGE_DIR}")
    print(f"  标签: {YOLOConfig.TRAIN_LABEL_DIR}\n")

    print("验证集路径:")
    print(f"  图像: {YOLOConfig.VAL_IMAGE_DIR}")
    print(f"  标签: {YOLOConfig.VAL_LABEL_DIR}\n")

    print("测试集路径:")
    print(f"  图像: {YOLOConfig.TEST_IMAGE_DIR}")
    print(f"  标签: {YOLOConfig.TEST_LABEL_DIR}\n")

    # 检查目录是否存在
    print("=" * 70)
    print("检查目录存在性")
    print("=" * 70)

    directories = {
        '图像目录': YOLOConfig.IMAGE_DIR,
        '标签目录': YOLOConfig.LABEL_DIR,
        '训练集图像': YOLOConfig.TRAIN_IMAGE_DIR,
        '训练集标签': YOLOConfig.TRAIN_LABEL_DIR,
        '验证集图像': YOLOConfig.VAL_IMAGE_DIR,
        '验证集标签': YOLOConfig.VAL_LABEL_DIR,
        '测试集图像': YOLOConfig.TEST_IMAGE_DIR,
        '测试集标签': YOLOConfig.TEST_LABEL_DIR,
    }

    all_exist = True
    for name, path in directories.items():
        exists = Path(path).exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {name}: {'存在' if exists else '不存在'}")
        if not exists:
            all_exist = False

    # 统计文件数量
    print("\n" + "=" * 70)
    print("统计文件数量")
    print("=" * 70)

    if Path(YOLOConfig.IMAGE_DIR).exists():
        train_images = len(list(Path(YOLOConfig.TRAIN_IMAGE_DIR).glob('*.jpg'))) + \
                      len(list(Path(YOLOConfig.TRAIN_IMAGE_DIR).glob('*.png')))
        val_images = len(list(Path(YOLOConfig.VAL_IMAGE_DIR).glob('*.jpg'))) + \
                    len(list(Path(YOLOConfig.VAL_IMAGE_DIR).glob('*.png')))
        test_images = len(list(Path(YOLOConfig.TEST_IMAGE_DIR).glob('*.jpg'))) + \
                     len(list(Path(YOLOConfig.TEST_IMAGE_DIR).glob('*.png')))

        print(f"\n图像统计:")
        print(f"  训练集: {train_images} 张")
        print(f"  验证集: {val_images} 张")
        print(f"  测试集: {test_images} 张")
        print(f"  总计: {train_images + val_images + test_images} 张")
    else:
        print("\n✗ 图像目录不存在")
        all_exist = False

    if Path(YOLOConfig.LABEL_DIR).exists():
        train_labels = len(list(Path(YOLOConfig.TRAIN_LABEL_DIR).glob('*.txt')))
        val_labels = len(list(Path(YOLOConfig.VAL_LABEL_DIR).glob('*.txt')))
        test_labels = len(list(Path(YOLOConfig.TEST_LABEL_DIR).glob('*.txt')))

        print(f"\n标签统计:")
        print(f"  训练集: {train_labels} 个")
        print(f"  验证集: {val_labels} 个")
        print(f"  测试集: {test_labels} 个")
        print(f"  总计: {train_labels + val_labels + test_labels} 个")
    else:
        print("\n✗ 标签目录不存在")
        all_exist = False

    # 检查匹配情况
    print("\n" + "=" * 70)
    print("检查图像和标签匹配情况")
    print("=" * 70)

    if Path(YOLOConfig.IMAGE_DIR).exists() and Path(YOLOConfig.LABEL_DIR).exists():
        if train_images == train_labels:
            print(f"  ✓ 训练集匹配: {train_images} 张图像 = {train_labels} 个标签")
        else:
            print(f"  ⚠ 训练集不匹配: {train_images} 张图像 ≠ {train_labels} 个标签")
            all_exist = False

        if val_images == val_labels:
            print(f"  ✓ 验证集匹配: {val_images} 张图像 = {val_labels} 个标签")
        else:
            print(f"  ⚠ 验证集不匹配: {val_images} 张图像 ≠ {val_labels} 个标签")
            all_exist = False

        if test_images == test_labels:
            print(f"  ✓ 测试集匹配: {test_images} 张图像 = {test_labels} 个标签")
        else:
            print(f"  ⚠ 测试集不匹配: {test_images} 张图像 ≠ {test_labels} 个标签")
            all_exist = False

    # 计算数据集比例
    if train_images + val_images + test_images > 0:
        print("\n" + "=" * 70)
        print("数据集比例")
        print("=" * 70)

        total = train_images + val_images + test_images
        train_ratio = train_images / total * 100
        val_ratio = val_images / total * 100
        test_ratio = test_images / total * 100

        print(f"\n  训练集: {train_images} 张 ({train_ratio:.2f}%)")
        print(f"  验证集: {val_images} 张 ({val_ratio:.2f}%)")
        print(f"  测试集: {test_images} 张 ({test_ratio:.2f}%)")
        print(f"\n  目标比例: 7:2:1 (70% : 20% : 10%)")

        # 检查是否符合目标比例
        error_train = abs(train_ratio - 70)
        error_val = abs(val_ratio - 20)
        error_test = abs(test_ratio - 10)

        if error_train < 5 and error_val < 5 and error_test < 5:
            print(f"  ✓✓✓ 数据集比例符合要求！")
        else:
            print(f"  ⚠ 数据集比例偏差较大")

    # 最终结论
    print("\n" + "=" * 70)
    print("验证结论")
    print("=" * 70)

    if all_exist and train_labels > 0:
        print("\n  ✓✓✓ 路径配置正确！数据集已准备就绪！")
        print("\n  下一步：")
        print("    1. 运行数据集初始化: python scripts/init_yolo_dataset.py")
        print("    2. 开始训练: python scripts/train_faster_rcnn_yolo.py")
    else:
        print("\n  ✗ 路径配置仍有问题，请检查：")
        print("    1. 确保 data/images/ 和 data/labels/ 目录存在")
        print("    2. 确保每个子目录（train/val/test）都有对应的文件")
        print("    3. 检查 utils/config_yolo.py 中的路径配置")

    print("\n" + "=" * 70)

    return all_exist and train_labels > 0


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ 验证失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
