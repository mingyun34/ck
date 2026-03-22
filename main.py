"""
无人机检测与跟踪主程序
集成Faster R-CNN检测和DeepSORT跟踪
"""

import os
import sys
import cv2
import torch
import numpy as np
import argparse
from pathlib import Path
from typing import Optional, List, Dict

# 添加项目根目录到路径
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

from drone.models.faster_rcnn import create_baseline_model
from drone.models.tracker import DroneTracker, Track, draw_tracks, draw_no_drone_message
from drone.utils.config_yolo import YOLOConfig


class DroneDetectorAndTracker:
    """
    无人机检测与跟踪系统

    集成Faster R-CNN检测和DeepSORT跟踪功能
    """

    def __init__(self,
                 checkpoint_path: str,
                 device: str = 'cuda',
                 confidence_threshold: float = 0.5,
                 nms_threshold: float = 0.5,
                 tracker_config: Optional[Dict] = None):
        """
        初始化检测与跟踪系统

        Args:
            checkpoint_path: 训练好的模型权重路径
            device: 设备 ('cuda' 或 'cpu')
            confidence_threshold: 检测置信度阈值
            nms_threshold: NMS阈值
            tracker_config: 跟踪器配置
        """
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold

        print("=" * 70)
        print("初始化无人机检测与跟踪系统")
        print("=" * 70)

        # 加载检测模型
        self.model = self._load_detection_model(checkpoint_path)

        # 初始化跟踪器
        tracker_config = tracker_config or {
            'max_age': 30,
            'min_confidence': confidence_threshold * 0.8,  # 跟踪阈值略低于检测阈值
            'n_init': 3,
            'max_iou_distance': 0.7
        }
        self.tracker = DroneTracker(**tracker_config)

        # 类别名称
        self.class_names = YOLOConfig.CLASS_NAMES

        print(f"✓ 系统初始化完成")
        print(f"  设备: {self.device}")
        print(f"  检测阈值: {self.confidence_threshold}")
        print(f"  跟踪器: DeepSORT" if self.tracker.has_deepsort else "  跟踪器: Simple")
        print("=" * 70)

    def _load_detection_model(self, checkpoint_path: str):
        """加载Faster R-CNN检测模型"""
        print(f"\n加载检测模型: {checkpoint_path}")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"模型文件不存在: {checkpoint_path}")

        # 创建模型
        num_classes = YOLOConfig.NUM_CLASSES + 1  # 背景类 + 目标类
        model = create_baseline_model(num_classes=num_classes, pretrained=False)

        # 加载权重
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # 处理不同的checkpoint格式
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"  训练轮数: {checkpoint.get('epoch', 'N/A')}")
            print(f"  训练损失: {checkpoint.get('loss', 'N/A'):.4f}")
        else:
            model.load_state_dict(checkpoint)
            print("  加载原始权重格式")

        model.to(self.device)
        model.eval()

        print("✓ 检测模型加载完成")
        return model

    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        预处理图像

        Args:
            image: 输入图像 (BGR格式)

        Returns:
            处理后的张量
        """
        # 转换为RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 转换为张量
        image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.to(self.device)

        return image_tensor

    def detect(self, image: np.ndarray) -> Dict:
        """
        检测图像中的无人机

        Args:
            image: 输入图像

        Returns:
            检测结果字典
        """
        # 预处理
        image_tensor = self.preprocess_image(image)

        # 推理
        with torch.no_grad():
            outputs = self.model([image_tensor])[0]

        # 后处理
        # 过滤低置信度检测
        keep = outputs['scores'] >= self.confidence_threshold

        boxes = outputs['boxes'][keep].cpu().numpy()
        labels = outputs['labels'][keep].cpu().numpy()
        scores = outputs['scores'][keep].cpu().numpy()

        # 应用NMS
        if len(boxes) > 0:
            keep_indices = self._apply_nms(boxes, scores, self.nms_threshold)
            boxes = boxes[keep_indices]
            labels = labels[keep_indices]
            scores = scores[keep_indices]

        return {
            'boxes': boxes,
            'labels': labels,
            'scores': scores
        }

    def _apply_nms(self, boxes: np.ndarray, scores: np.ndarray, threshold: float) -> List[int]:
        """应用非极大值抑制"""
        if len(boxes) == 0:
            return []

        # 转换为torch张量
        boxes_tensor = torch.from_numpy(boxes).float()
        scores_tensor = torch.from_numpy(scores).float()

        # 应用NMS
        keep = torch.ops.torchvision.nms(boxes_tensor, scores_tensor, threshold)

        return keep.cpu().numpy().tolist()

    def track(self, image: np.ndarray, detection_result: Optional[Dict] = None) -> List[Track]:
        """
        跟踪图像中的无人机

        Args:
            image: 输入图像
            detection_result: 可选的检测结果，如果为None则自动检测

        Returns:
            跟踪结果列表
        """
        # 如果没有提供检测结果，则自动检测
        if detection_result is None:
            detection_result = self.detect(image)

        # 更新跟踪器
        tracks = self.tracker.update_from_faster_rcnn(detection_result, image)

        return tracks

    def detect_and_track(self, image: np.ndarray) -> tuple:
        """
        同时进行检测和跟踪

        Args:
            image: 输入图像

        Returns:
            (detection_result, tracks)
        """
        detection_result = self.detect(image)
        tracks = self.track(image, detection_result)

        return detection_result, tracks

    def visualize_tracking(self, image: np.ndarray, tracks: List[Track]) -> np.ndarray:
        """
        可视化跟踪结果

        Args:
            image: 输入图像
            tracks: 跟踪结果

        Returns:
            可视化后的图像
        """
        if len(tracks) == 0:
            # 没有检测到无人机，显示提示信息
            return draw_no_drone_message(image)
        else:
            # 有无人机，显示跟踪结果
            return draw_tracks(image, tracks, self.class_names)

    def process_video(self, video_path: str, output_path: Optional[str] = None, show: bool = True):
        """
        处理视频文件

        Args:
            video_path: 输入视频路径
            output_path: 输出视频路径
            show: 是否显示结果
        """
        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")

        # 获取视频信息
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"\n视频信息:")
        print(f"  分辨率: {width}x{height}")
        print(f"  帧率: {fps}")
        print(f"  总帧数: {total_frames}")

        # 创建视频写入器
        video_writer = None
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # 处理每一帧
        frame_count = 0
        track_history = {}  # 存储轨迹历史用于绘制轨迹线

        print("\n开始处理视频...")
        print("按 'q' 键退出")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # 检测与跟踪
            detection_result, tracks = self.detect_and_track(frame)

            # 绘制轨迹线（如果有跟踪目标）
            if len(tracks) > 0:
                for track in tracks:
                    track_id = track.track_id
                    if track_id not in track_history:
                        track_history[track_id] = []
                    track_history[track_id].append(tuple(track.bbox[:2]))

                    # 限制轨迹长度
                    if len(track_history[track_id]) > 30:
                        track_history[track_id].pop(0)

                # 绘制轨迹线
                for track_id, points in track_history.items():
                    if len(points) > 1:
                        pts = np.array(points, dtype=np.int32)
                        cv2.polylines(frame, [pts], False, (0, 255, 0), 2)

            # 可视化跟踪结果
            vis_frame = self.visualize_tracking(frame, tracks)

            # 添加统计信息
            info_text = f"Frame: {frame_count}/{total_frames} | Detections: {len(detection_result['boxes'])} | Tracks: {len(tracks)}"
            cv2.putText(vis_frame, info_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 写入视频
            if video_writer:
                video_writer.write(vis_frame)

            # 显示
            if show:
                cv2.imshow('Drone Detection & Tracking', vis_frame)

                # 每10帧打印一次进度
                if frame_count % 10 == 0:
                    print(f"  处理进度: {frame_count}/{total_frames} ({frame_count / total_frames * 100:.1f}%)")

                # 按q退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n用户中断处理")
                    break

        # 释放资源
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()

        print(f"\n✓ 视频处理完成")
        if output_path:
            print(f"  输出路径: {output_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='无人机检测与跟踪系统')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='训练好的模型权重路径')
    parser.add_argument('--video', type=str, required=True,
                        help='输入视频路径')
    parser.add_argument('--output', type=str, default=None,
                        help='输出视频路径')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'], help='设备')
    parser.add_argument('--confidence', type=float, default=0.5,
                        help='检测置信度阈值')
    parser.add_argument('--nms', type=float, default=0.5,
                        help='NMS阈值')
    parser.add_argument('--max-age', type=int, default=30,
                        help='跟踪器最大丢失帧数')
    parser.add_argument('--no-show', action='store_true',
                        help='不显示结果')

    args = parser.parse_args()

    # 检查CUDA可用性
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA不可用，自动切换到CPU")
        args.device = 'cpu'

    # 创建检测与跟踪系统
    tracker_config = {
        'max_age': args.max_age,
        'min_confidence': args.confidence * 0.8,
        'n_init': 3,
        'max_iou_distance': 0.7
    }

    system = DroneDetectorAndTracker(
        checkpoint_path=args.checkpoint,
        device=args.device,
        confidence_threshold=args.confidence,
        nms_threshold=args.nms,
        tracker_config=tracker_config
    )

    # 处理视频
    print(f"\n处理视频: {args.video}")
    system.process_video(args.video, args.output, show=not args.no_show)


if __name__ == "__main__":
    main()
