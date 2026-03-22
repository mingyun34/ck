```plaintext
项目根目录/
│
├── data/                           # 数据目录
│   ├── images/                     # 原始图像
│   ├── annotations/                # 标注文件
│   └── splits/                     # 划分后的数据集
│
├── models/                         # 模型定义
│   ├── __init__.py
│   ├── faster_rcnn.py
│   ├── backbone.py
│   ├── fpn.py
│   ├── cbam.py
│   └── tracker.py
│
├── utils/                          # 工具模块 ✓ 已创建
│   ├── dataset.py
│   ├── augment.py
│   ├── config.py
│   ├── visualizer.py
│   └── metrics.py                  # 后续添加
│
├── scripts/                        # 脚本 ✓ 已创建
│   ├── split_dataset.py
│   ├── train_base.py               # 后续添加
│   ├── train_improved.py           # 后续添加
│   └── evaluate.py                 # 后续添加
│
├── configs/                        # 配置文件
│   └── config.yaml                 # 由config.py自动生成
│
├── weights/                        # 模型权重
├── logs/                           # 训练日志
├── outputs/                        # 输出结果
│   └── visualizations/             # 可视化结果
│
├── requirements.txt
├── README.md                       # 项目说明
└── main.py                         # 主程序入口
```
