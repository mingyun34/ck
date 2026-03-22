"""
YOLO数据集快速初始化脚本
一键检查和准备数据集
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到路径
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from drone.utils.config_yolo import YOLOConfig


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("  YOLO数据集快速初始化")
    print("=" * 70)

    # 1. 创建目录结构
    print("\n[1/4] 创建目录结构...")
    YOLOConfig.create_directories()
    print("  ✓ 目录创建完成")

    # 2. 检查数据集
    print("\n[2/4] 检查数据集...")
    dataset_ok = YOLOConfig.check_dataset()

    if not dataset_ok:
        print("\n  ⚠ 数据集检查失败！")
        print("\n  请确保你的数据集目录结构如下：")
        print("""
  data/
  ├── images/
  │   ├── train/      # 训练图像
  │   ├── val/        # 验证图像
  │   └── test/       # 测试图像（可选）
  └── labels/
      ├── train/      # 训练标签 (.txt文件)
      ├── val/        # 验证标签 (.txt文件)
      └── test/       # 测试标签 (.txt文件，可选）
        """)
        print("\n  标注文件格式示例（YOLO格式）：")
        print("  class_id x_center y_center width height")
        print("  0 0.5 0.5 0.1 0.2  # (归一化坐标)")
        return False

    # 3. 分析数据集统计
    print("\n[3/4] 分析数据集统计...")
    YOLOConfig.analyze_yolo_labels('train')
    YOLOConfig.analyze_yolo_labels('val')

    # 4. 保存配置
    print("\n[4/4] 保存配置文件...")
    YOLOConfig.save_to_yaml()

    # 5. 显示下一步操作
    print("\n" + "=" * 70)
    print("  初始化完成！")
    print("=" * 70)

    print("\n下一步操作：")

    print("\n1. 运行完整的数据集测试和可视化：")
    print("   python scripts/yolo_data_guide.py")

    print("\n2. 或直接创建数据集并开始训练：")
    print("""
   from utils.dataset_yolo import create_yolo_datasets, collate_fn
   from utils.augment import get_train_transforms_v3, get_val_transforms
   from torch.utils.data import DataLoader

   # 创建数据集
   train_dataset, val_dataset, _ = create_yolo_datasets(
       base_dir='data',
       transforms_train=get_train_transforms_v3(),
       transforms_val=get_val_transforms()
   )

   # 创建DataLoader
   train_loader = DataLoader(
       train_dataset,
       batch_size=16,
       shuffle=True,
       collate_fn=collate_fn
   )
    """)

    print("\n3. 查看配置参数：")
    print(f"   配置文件: {ROOT_DIR}/configs/yolo_config.yaml")

    print("\n✓ 数据集已准备就绪，可以开始训练了！\n")

    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ 初始化失败: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)