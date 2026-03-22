"""
验证数据集路径配置
检查图像和标签文件的实际位置
"""

import os
from pathlib import Path

def verify_paths():
    """验证路径配置"""

    # 项目根目录
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    print("=" * 70)
    print("数据集路径验证")
    print("=" * 70)
    print(f"\n项目根目录: {ROOT_DIR}\n")

    # 定义可能的路径组合
    data_root = Path(ROOT_DIR) / 'data'

    # 方案1: 直接在 data/ 下
    paths_v1 = {
        'images': data_root / 'images' / 'images',
        'labels': data_root / 'images' / 'labels'
    }

    # 方案2: 在 data/images/ 下
    paths_v2 = {
        'images': data_root / 'images',
        'labels': data_root / 'labels'
    }

    # 方案3: 在 data/images/images/ 下
    paths_v3 = {
        'images': data_root / 'images' / 'images',
        'labels': data_root / 'images' / 'labels'
    }

    # 方案4: 在 data/ 下
    paths_v4 = {
        'images': data_root / 'images',
        'labels': data_root / 'labels'
    }

    # 测试所有方案
    for version, paths in [('方案1', paths_v1), ('方案2', paths_v2), ('方案3', paths_v3), ('方案4', paths_v4)]:
        print(f"\n{version}:")
        print("-" * 70)

        image_dir = paths['images']
        label_dir = paths['labels']

        print(f"  图像目录: {image_dir}")
        print(f"  标签目录: {label_dir}")

        # 检查图像目录
        if image_dir.exists():
            train_images = len(list(image_dir.glob('train/*.jpg'))) + len(list(image_dir.glob('train/*.png')))
            val_images = len(list(image_dir.glob('val/*.jpg'))) + len(list(image_dir.glob('val/*.png')))
            test_images = len(list(image_dir.glob('test/*.jpg'))) + len(list(image_dir.glob('test/*.png')))

            print(f"  ✓ 图像目录存在")
            print(f"    训练集: {train_images} 张")
            print(f"    验证集: {val_images} 张")
            print(f"    测试集: {test_images} 张")
        else:
            print(f"  ✗ 图像目录不存在")

        # 检查标签目录
        if label_dir.exists():
            train_labels = len(list(label_dir.glob('train/*.txt')))
            val_labels = len(list(label_dir.glob('val/*.txt')))
            test_labels = len(list(label_dir.glob('test/*.txt')))

            print(f"  ✓ 标签目录存在")
            print(f"    训练集: {train_labels} 个")
            print(f"    验证集: {val_labels} 个")
            print(f"    测试集: {test_labels} 个")
        else:
            print(f"  ✗ 标签目录不存在")

        # 检查是否匹配
        if image_dir.exists() and label_dir.exists():
            train_total = train_images + train_labels
            val_total = val_images + val_labels
            test_total = test_images + test_labels

            if train_total > 0 and val_total > 0:
                print(f"\n  ✓✓✓ 此方案可行！")
                print(f"    总计: {train_total + val_total + test_total} 个文件")

                # 显示匹配情况
                if train_images == train_labels:
                    print(f"    ✓ 训练集图像和标签数量匹配")
                else:
                    print(f"    ⚠ 训练集不匹配: 图像{train_images} vs 标签{train_labels}")

                if val_images == val_labels:
                    print(f"    ✓ 验证集图像和标签数量匹配")
                else:
                    print(f"    ⚠ 验证集不匹配: 图像{val_images} vs 标签{val_labels}")

    # 扫描实际的目录结构
    print("\n" + "=" * 70)
    print("扫描实际的目录结构")
    print("=" * 70)

    if data_root.exists():
        print(f"\n扫描目录: {data_root}\n")

        # 递归列出所有目录
        all_dirs = []
        for item in data_root.rglob('*'):
            if item.is_dir():
                rel_path = item.relative_to(data_root)
                all_dirs.append(rel_path)

        # 只显示包含文件的目录
        print("包含文件的目录:")
        print("-" * 70)
        for dir_path in sorted(all_dirs):
            full_path = data_root / dir_path

            # 统计文件
            jpg_count = len(list(full_path.glob('*.jpg')))
            png_count = len(list(full_path.glob('*.png')))
            txt_count = len(list(full_path.glob('*.txt')))

            if jpg_count + png_count + txt_count > 0:
                files_str = []
                if jpg_count > 0:
                    files_str.append(f"{jpg_count}个jpg")
                if png_count > 0:
                    files_str.append(f"{png_count}个png")
                if txt_count > 0:
                    files_str.append(f"{txt_count}个txt")

                print(f"  {dir_path}: {', '.join(files_str)}")

    # 推荐方案
    print("\n" + "=" * 70)
    print("推荐配置")
    print("=" * 70)

    # 根据实际结构推荐
    labels_under_images = (data_root / 'images' / 'labels').exists()
    images_under_images = (data_root / 'images' / 'images').exists()

    if labels_under_images and images_under_images:
        print("\n✓ 检测到标准结构: data/images/{images, labels}")
        print("\n推荐的配置:")
        print("  IMAGE_DIR = data/images/images")
        print("  LABEL_DIR = data/images/labels")

        print("\n在 utils/config_yolo.py 中设置:")
        print("  IMAGE_DIR = os.path.join(DATA_DIR, 'images', 'images')")
        print("  LABEL_DIR = os.path.join(DATA_DIR, 'images', 'labels')")
    elif labels_under_images:
        print("\n✓ 检测到标签在 images/ 下")
        print("\n推荐的配置:")
        print("  IMAGE_DIR = data/images")
        print("  LABEL_DIR = data/images/labels")

        print("\n在 utils/config_yolo.py 中设置:")
        print("  IMAGE_DIR = os.path.join(DATA_DIR, 'images')")
        print("  LABEL_DIR = os.path.join(DATA_DIR, 'images', 'labels')")
    else:
        print("\n⚠ 未能自动识别标准结构")
        print("\n请检查上面的目录扫描结果，手动配置路径。")


if __name__ == "__main__":
    verify_paths()
