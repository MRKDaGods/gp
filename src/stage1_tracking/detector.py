"""YOLO object detector wrapper using Ultralytics."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from loguru import logger

from src.core.constants import CLASS_NAMES
from src.core.data_models import Detection


class Detector:
    """Wraps Ultralytics YOLO for object detection.

    Filters detections by confidence, class, and NMS IoU threshold.
    """

    def __init__(
        self,
        model_path: str = "yolo26m.pt",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        classes: Optional[List[int]] = None,
        device: str = "cuda:0",
        half: bool = True,
        img_size: int = 640,
    ):
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.classes = classes  # COCO class IDs to detect
        self.device = device
        self.half = half
        self.img_size = img_size

        logger.info(
            f"Detector initialized: {model_path}, device={device}, "
            f"conf={confidence_threshold}, classes={classes}"
        )

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run detection on a single BGR frame.

        Args:
            frame: BGR uint8 numpy array (H, W, 3).

        Returns:
            List of Detection objects.
        """
        results = self.model.predict(
            frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            classes=self.classes,
            device=self.device,
            half=self.half,
            imgsz=self.img_size,
            verbose=False,
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())

                detections.append(
                    Detection(
                        bbox=(float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])),
                        confidence=conf,
                        class_id=cls_id,
                        class_name=CLASS_NAMES.get(cls_id, f"class_{cls_id}"),
                    )
                )

        return detections

    def detect_to_array(self, frame: np.ndarray) -> np.ndarray:
        """Run detection and return as numpy array for BoxMOT.

        Returns:
            np.ndarray of shape (N, 6): [x1, y1, x2, y2, confidence, class_id]
            Returns empty array with shape (0, 6) if no detections.
        """
        detections = self.detect(frame)
        if not detections:
            return np.empty((0, 6), dtype=np.float32)

        arr = np.array(
            [
                [d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3], d.confidence, d.class_id]
                for d in detections
            ],
            dtype=np.float32,
        )
        return arr
