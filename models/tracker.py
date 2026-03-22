"""
DeepSORT 多目标跟踪模块
用于无人机目标跟踪
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import torch


@dataclass
class Detection:
    """检测结果"""
    bbox: np.ndarray  # [x, y, w, h]
    confidence: float
    class_id: int


@dataclass
class Track:
    """跟踪目标"""
    track_id: int
    bbox: np.ndarray  # [x, y, w, h]
    confidence: float
    age: int
    hits: int
    class_id: int = 1  # 默认为无人机类别


class DroneTracker:
    """
    无人机跟踪器

    基于DeepSORT实现多目标跟踪
    支持与Faster R-CNN检测模型无缝集成
    """

    def __init__(self,
                 max_age: int = 30,
                 min_confidence: float = 0.3,
                 n_init: int = 3,
                 max_iou_distance: float = 0.7,
                 max_cosine_distance: float = 0.3,
                 nn_budget: Optional[int] = None,
                 use_deepsort: bool = True):
        """
        初始化跟踪器

        Args:
            max_age: 最大丢失帧数，超过则删除轨迹
            min_confidence: 最小置信度阈值
            n_init: 初始化需要的连续检测数
            max_iou_distance: 最大IOU距离阈值
            max_cosine_distance: 最大外观特征距离
            nn_budget: 外观特征缓存大小
            use_deepsort: 是否使用DeepSORT，False则使用简单跟踪器
        """
        self.max_age = max_age
        self.min_confidence = min_confidence
        self.n_init = n_init
        self.max_iou_distance = max_iou_distance
        self.use_deepsort = use_deepsort

        # 尝试导入DeepSORT
        if use_deepsort:
            try:
                from deep_sort_realtime.deepsort_tracker import DeepSort
                self.tracker = DeepSort(
                    max_age=max_age,
                    min_confidence=min_confidence,
                    n_init=n_init,
                    max_iou_distance=max_iou_distance,
                    max_cosine_distance=max_cosine_distance,
                    nn_budget=nn_budget
                )
                self.has_deepsort = True
                print("✓ DeepSORT跟踪器初始化成功")
            except ImportError:
                print("⚠ 警告: deep-sort-realtime未安装，使用简单跟踪器")
                print("  安装命令: pip install deep-sort-realtime")
                self.has_deepsort = False
                self._init_simple_tracker()
        else:
            self.has_deepsort = False
            self._init_simple_tracker()

    def _init_simple_tracker(self):
        """初始化简单跟踪器（备用）"""
        self.next_id = 0
        self.tracks = {}  # track_id -> {'bbox': ..., 'age': ..., 'hits': ..., 'class_id': ...}
        self.iou_threshold = 0.3

    def update_from_faster_rcnn(self,
                                model_output: dict,
                                frame: Optional[np.ndarray] = None,
                                image_tensor: Optional[torch.Tensor] = None) -> List[Track]:
        """
        从Faster R-CNN输出更新跟踪状态

        Args:
            model_output: Faster R-CNN模型输出
                {
                    'boxes': Tensor[N, 4],  # [x1, y1, x2, y2]
                    'labels': Tensor[N],
                    'scores': Tensor[N]
                }
            frame: 当前帧图像（BGR格式，用于外观特征提取）
            image_tensor: 图像张量（可选）

        Returns:
            tracks: 当前活跃的跟踪目标
        """
        # 提取检测结果
        boxes = model_output.get('boxes', [])
        labels = model_output.get('labels', [])
        scores = model_output.get('scores', [])

        if len(boxes) == 0:
            return []

        # 转换为Detection对象
        detections = []
        for box, label, score in zip(boxes, labels, scores):
            # 转换框格式: [x1, y1, x2, y2] -> [x, y, w, h]
            if isinstance(box, torch.Tensor):
                box = box.cpu().numpy()
            if isinstance(label, torch.Tensor):
                label = label.cpu().item()
            if isinstance(score, torch.Tensor):
                score = score.cpu().item()

            x1, y1, x2, y2 = box
            x, y = x1, y1
            w, h = x2 - x1, y2 - y1

            # 只保留高置信度检测
            if score >= self.min_confidence:
                detections.append(Detection(
                    bbox=np.array([x, y, w, h], dtype=np.float32),
                    confidence=float(score),
                    class_id=int(label)
                ))

        # 更新跟踪
        return self.update(detections, frame)

    def update(self,
               detections: List[Detection],
               frame: Optional[np.ndarray] = None) -> List[Track]:
        """
        更新跟踪状态

        Args:
            detections: 当前帧的检测结果
            frame: 当前帧图像（用于外观特征提取）

        Returns:
            tracks: 当前活跃的跟踪目标
        """
        if self.has_deepsort:
            return self._update_deepsort(detections, frame)
        else:
            return self._update_simple(detections)

    def _update_deepsort(self,
                        detections: List[Detection],
                        frame: Optional[np.ndarray]) -> List[Track]:
        """使用DeepSORT更新"""
        # 转换检测格式为DeepSORT需要的格式
        # DeepSORT需要: [[left, top, w, h, confidence, class_id], ...]
        bboxes = []
        for det in detections:
            if det.confidence >= self.min_confidence:
                bboxes.append([
                    float(det.bbox[0]),  # left
                    float(det.bbox[1]),  # top
                    float(det.bbox[2]),  # w
                    float(det.bbox[3]),  # h
                    float(det.confidence),
                    int(det.class_id)
                ])

        if not bboxes:
            # 没有检测，直接返回空列表
            return []

        # 更新跟踪
        try:
            tracks = self.tracker.update_tracks(bboxes, frame=frame)
        except Exception as e:
            print(f"DeepSORT更新错误: {e}")
            return []

        # 转换输出格式
        result = []
        for track in tracks:
            if not track.is_confirmed():
                continue

            # 获取轨迹信息
            track_id = track.track_id

            try:
                # to_ltwh返回[left, top, w, h]
                ltwh = track.to_ltwh()
                bbox = np.array(ltwh, dtype=np.float32)

                # 获取置信度
                det_conf = track.get_det_conf()
                confidence = float(det_conf) if det_conf is not None else 0.0

                # 获取类别ID
                class_id = track.det_conf if hasattr(track, 'det_conf') else 1
                if isinstance(class_id, (list, tuple)) and len(class_id) > 5:
                    class_id = int(class_id[5])

                result.append(Track(
                    track_id=track_id,
                    bbox=bbox,
                    confidence=confidence,
                    age=track.age,
                    hits=track.hits,
                    class_id=class_id
                ))
            except Exception as e:
                print(f"处理轨迹{track_id}时出错: {e}")
                continue

        return result

    def _update_simple(self, detections: List[Detection]) -> List[Track]:
        """简单跟踪器更新（备用）"""
        # 过滤低置信度检测
        valid_dets = [d for d in detections if d.confidence >= self.min_confidence]

        if not valid_dets:
            # 更新所有轨迹年龄
            to_remove = []
            for track_id in self.tracks:
                self.tracks[track_id]['age'] += 1
                if self.tracks[track_id]['age'] > self.max_age:
                    to_remove.append(track_id)
            for track_id in to_remove:
                del self.tracks[track_id]
            return self._get_active_tracks()

        # 计算检测与轨迹的IoU矩阵
        track_ids = list(self.tracks.keys())
        det_boxes = np.array([d.bbox for d in valid_dets])

        if track_ids:
            track_boxes = np.array([self.tracks[tid]['bbox'] for tid in track_ids])
            iou_matrix = self._compute_iou_matrix(det_boxes, track_boxes)

            # 匹配（贪婪算法）
            matched_dets = set()
            matched_tracks = set()

            for det_idx in range(len(valid_dets)):
                best_iou = 0
                best_track_idx = -1

                for track_idx in range(len(track_ids)):
                    if track_idx in matched_tracks:
                        continue
                    if iou_matrix[det_idx, track_idx] > best_iou:
                        best_iou = iou_matrix[det_idx, track_idx]
                        best_track_idx = track_idx

                if best_iou > self.iou_threshold:
                    # 匹配成功
                    track_id = track_ids[best_track_idx]
                    self.tracks[track_id]['bbox'] = valid_dets[det_idx].bbox
                    self.tracks[track_id]['age'] = 0
                    self.tracks[track_id]['hits'] += 1
                    self.tracks[track_id]['class_id'] = valid_dets[det_idx].class_id
                    matched_dets.add(det_idx)
                    matched_tracks.add(best_track_idx)

        # 为未匹配的检测创建新轨迹
        for det_idx, det in enumerate(valid_dets):
            if det_idx not in matched_dets:
                self.tracks[self.next_id] = {
                    'bbox': det.bbox,
                    'confidence': det.confidence,
                    'age': 0,
                    'hits': 1,
                    'class_id': det.class_id
                }
                self.next_id += 1

        # 更新未匹配轨迹的年龄
        to_remove = []
        for track_id in track_ids:
            if track_ids.index(track_id) not in matched_tracks:
                self.tracks[track_id]['age'] += 1
                if self.tracks[track_id]['age'] > self.max_age:
                    to_remove.append(track_id)

        for track_id in to_remove:
            del self.tracks[track_id]

        return self._get_active_tracks()

    def _get_active_tracks(self) -> List[Track]:
        """获取活跃轨迹"""
        result = []
        for track_id, data in self.tracks.items():
            if data['hits'] >= self.n_init:
                result.append(Track(
                    track_id=track_id,
                    bbox=data['bbox'],
                    confidence=data['confidence'],
                    age=data['age'],
                    hits=data['hits'],
                    class_id=data['class_id']
                ))
        return result

    def _compute_iou_matrix(self, boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        """计算IoU矩阵"""
        n1 = len(boxes1)
        n2 = len(boxes2)
        iou_matrix = np.zeros((n1, n2))

        for i in range(n1):
            for j in range(n2):
                iou_matrix[i, j] = self._compute_iou(boxes1[i], boxes2[j])

        return iou_matrix

    def _compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """计算两个框的IoU"""
        # box格式: [x, y, w, h]
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[0] + box1[2], box2[0] + box2[2])
        y2 = min(box1[1] + box1[3], box2[1] + box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = box1[2] * box1[3]
        area2 = box2[2] * box2[3]
        union = area1 + area2 - inter

        return inter / (union + 1e-6)

    def reset(self):
        """重置跟踪器"""
        if self.has_deepsort:
            from deep_sort_realtime.deepsort_tracker import DeepSort
            self.tracker = DeepSort(
                max_age=self.max_age,
                min_confidence=self.min_confidence,
                n_init=self.n_init,
                max_iou_distance=self.max_iou_distance
            )
        else:
            self.next_id = 0
            self.tracks = {}
        print("跟踪器已重置")

    def get_track_count(self) -> int:
        """获取当前跟踪目标数量"""
        return len(self._get_active_tracks())


def draw_tracks(frame: np.ndarray,
                tracks: List[Track],
                class_names: Optional[List[str]] = None,
                show_confidence: bool = True,
                show_age: bool = False,
                colors: Optional[Dict[int, Tuple[int, int, int]]] = None) -> np.ndarray:
    """
    在图像上绘制跟踪结果

    Args:
        frame: 输入图像 (BGR格式)
        tracks: 跟踪结果
        class_names: 类别名称列表
        show_confidence: 是否显示置信度
        show_age: 是否显示轨迹年龄
        colors: ID到颜色的映射

    Returns:
        frame: 绘制后的图像
    """
    if colors is None:
        # 生成颜色映射
        np.random.seed(42)
        colors = {}

    frame_vis = frame.copy()

    for track in tracks:
        # 生成颜色
        if track.track_id not in colors:
            # 为每个ID生成固定颜色
            np.random.seed(track.track_id)
            colors[track.track_id] = tuple(
                np.random.randint(50, 255, 3).tolist()
            )

        color = colors[track.track_id]
        x, y, w, h = track.bbox

        # 绘制边界框
        cv2.rectangle(frame_vis,
                      (int(x), int(y)),
                      (int(x + w), int(y + h)),
                      color, 2)

        # 准备标签文本
        class_name = class_names[track.class_id] if class_names and track.class_id < len(class_names) else f"Class {track.class_id}"
        label_text = f'ID:{track.track_id} {class_name}'

        if show_confidence:
            label_text += f' {track.confidence:.2f}'

        if show_age:
            label_text += f' (age:{track.age})'

        # 计算文本区域大小
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            2
        )

        # 绘制文本背景
        text_bg_x1 = int(x)
        text_bg_y1 = int(y) - text_height - 8
        text_bg_x2 = int(x) + text_width + 8
        text_bg_y2 = int(y)

        # 确保文本背景在图像范围内
        text_bg_y1 = max(0, text_bg_y1)

        cv2.rectangle(
            frame_vis,
            (text_bg_x1, text_bg_y1),
            (text_bg_x2, text_bg_y2),
            color,
            -1  # 填充
        )

        # 绘制文本
        cv2.putText(frame_vis, label_text,
                    (int(x) + 4, int(y) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    return frame_vis


def draw_no_drone_message(frame: np.ndarray) -> np.ndarray:
    """
    在屏幕中心显示"未检测出无人机"消息

    Args:
        frame: 输入图像

    Returns:
        绘制后的图像
    """
    frame_vis = frame.copy()
    h, w = frame_vis.shape[:2]

    # 文本内容
    text = "未检测出无人机"

    # 计算文本大小
    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        3
    )

    # 计算文本位置（屏幕中心）
    text_x = (w - text_width) // 2
    text_y = (h + text_height) // 2

    # 绘制文本背景
    bg_x1 = text_x - 20
    bg_y1 = text_y - text_height - 20
    bg_x2 = text_x + text_width + 20
    bg_y2 = text_y + 20

    cv2.rectangle(frame_vis, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
    cv2.rectangle(frame_vis, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 255, 0), 2)

    # 绘制文本
    cv2.putText(frame_vis, text, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)

    return frame_vis


# ============== 使用示例 ==============

if __name__ == "__main__":
    # 创建跟踪器
    tracker = DroneTracker(
        max_age=30,
        min_confidence=0.3,
        n_init=3,
        use_deepsort=True
    )

    # 模拟Faster R-CNN输出
    model_output = {
        'boxes': torch.tensor([[100, 100, 150, 130], [200, 150, 260, 190]]),
        'labels': torch.tensor([1, 1]),
        'scores': torch.tensor([0.9, 0.85])
    }

    # 模拟图像
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # 更新跟踪
    tracks = tracker.update_from_faster_rcnn(model_output, frame)

    print(f"跟踪目标数: {len(tracks)}")
    for track in tracks:
        print(f"ID: {track.track_id}, bbox: {track.bbox}, confidence: {track.confidence}")

    # 绘制跟踪结果
    frame_with_tracks = draw_tracks(frame, tracks, class_names=['background', 'drone'])
    print(f"跟踪结果已绘制到图像上")