"""YOLO wrapper — object detection counts and coverage.

Uses YOLOv8 (via ultralytics) to detect objects and produce per-category
counts across the 80 COCO object categories, plus aggregate summary
features like total object count, category count, and object coverage.
"""

import numpy as np
from PIL import Image

from .base import BaseModel

_DEFAULT_MODEL = "yolov8n.pt"
_DEFAULT_CONF = 0.25
_DEFAULT_IOU = 0.45

# COCO 80 category names in index order.
_COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic_light", "fire_hydrant", "stop_sign",
    "parking_meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports_ball", "kite",
    "baseball_bat", "baseball_glove", "skateboard", "surfboard",
    "tennis_racket", "bottle", "wine_glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot_dog", "pizza", "donut", "cake", "chair", "couch", "potted_plant",
    "bed", "dining_table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell_phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy_bear",
    "hair_drier", "toothbrush",
]


class YOLOModel(BaseModel):
    """YOLO object detection with per-category counts and summary features.

    Outputs 80 per-category count columns (yolo_{category}), plus:
    - object_count: total detected objects
    - category_count: number of distinct categories present
    - object_coverage: fraction of image area covered by detections
    - largest_object_ratio: largest single detection / image area
    - mean_confidence: average detection confidence
    """

    name = "yolo"

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        conf: float = _DEFAULT_CONF,
        iou: float = _DEFAULT_IOU,
        device: str | None = None,
    ):
        super().__init__(device=device)
        self.model_name = model_name
        self.conf = conf
        self.iou = iou

    def load(self) -> None:
        from ultralytics import YOLO

        self.model = YOLO(self.model_name)
        self.model.to(self.device)

    def _results_to_dict(self, result, img_area: int) -> dict[str, float]:
        boxes = result.boxes
        classes = boxes.cls.cpu().numpy().astype(int) if len(boxes) else np.array([], dtype=int)
        confs = boxes.conf.cpu().numpy() if len(boxes) else np.array([])
        xyxy = boxes.xyxy.cpu().numpy() if len(boxes) else np.empty((0, 4))

        # Per-category counts.
        counts = np.zeros(len(_COCO_NAMES), dtype=int)
        for cls_id in classes:
            if 0 <= cls_id < len(_COCO_NAMES):
                counts[cls_id] += 1

        row: dict[str, float] = {}
        for i, name in enumerate(_COCO_NAMES):
            row[f"yolo_{name}"] = float(counts[i])

        # Aggregate features.
        row["object_count"] = float(len(classes))
        row["category_count"] = float(len(np.unique(classes))) if len(classes) else 0.0

        if len(xyxy) > 0:
            widths = xyxy[:, 2] - xyxy[:, 0]
            heights = xyxy[:, 3] - xyxy[:, 1]
            areas = widths * heights
            row["object_coverage"] = float(min(areas.sum() / img_area, 1.0))
            row["largest_object_ratio"] = float(areas.max() / img_area)
            row["mean_confidence"] = float(confs.mean())
        else:
            row["object_coverage"] = 0.0
            row["largest_object_ratio"] = 0.0
            row["mean_confidence"] = 0.0

        return row

    def predict(self, image: Image.Image) -> dict[str, float]:
        img = image.convert("RGB")
        img_area = img.size[0] * img.size[1]
        results = self.model(img, conf=self.conf, iou=self.iou, verbose=False)
        return self._results_to_dict(results[0], img_area)

    def predict_batch(self, images: list[Image.Image]) -> list[dict[str, float]]:
        imgs = [img.convert("RGB") for img in images]
        areas = [img.size[0] * img.size[1] for img in imgs]
        results = self.model(imgs, conf=self.conf, iou=self.iou, verbose=False)
        return [self._results_to_dict(r, a) for r, a in zip(results, areas)]
