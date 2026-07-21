"""YOLO object detector wrapper using Ultralytics."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import logging

logger = logging.getLogger(__name__)

from athar.core.constants import CLASS_NAMES
from athar.core.data_models import Detection


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
        agnostic_nms: bool = True,
    ):
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.classes = classes  # COCO class IDs to detect
        self.device = device
        self.half = half
        self.img_size = img_size
        self.agnostic_nms = agnostic_nms

        logger.info(
            f"Detector initialized: {model_path}, device={device}, "
            f"conf={confidence_threshold}, iou={iou_threshold}, agnostic_nms={agnostic_nms}"
        )

    def _predict_kwargs(self) -> dict:
        kwargs = dict(
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            classes=self.classes,
            device=self.device,
            imgsz=self.img_size,
            agnostic_nms=self.agnostic_nms,
            verbose=False,
        )
        # Only pass `half` when we actually want FP16. Passing it at all (even
        # False) triggers ultralytics' noisy "'half' is deprecated" warning, and
        # FP32 is the default - so omitting it keeps the log clean on the common path.
        if self.half:
            kwargs["half"] = True
        return kwargs

    def _parse_result(self, result) -> List[Detection]:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        detections: List[Detection] = []
        # Pull the whole batch off the GPU once, not per-box (far fewer syncs).
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        for i in range(len(boxes)):
            cls_id = int(cls_ids[i])
            detections.append(
                Detection(
                    bbox=(float(xyxy[i][0]), float(xyxy[i][1]), float(xyxy[i][2]), float(xyxy[i][3])),
                    confidence=float(confs[i]),
                    class_id=cls_id,
                    class_name=CLASS_NAMES.get(cls_id, f"class_{cls_id}"),
                )
            )
        return detections

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run detection on a single BGR frame."""
        results = self.model.predict(frame, **self._predict_kwargs())
        detections: List[Detection] = []
        for result in results:
            detections.extend(self._parse_result(result))
        return detections

    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """Run detection on a batch of BGR frames in ONE GPU call (keeps the GPU
        fed instead of idling between single-frame inferences). Returns one
        detection list per input frame, in order. Falls back to per-frame on OOM."""
        if not frames:
            return []
        try:
            results = self.model.predict(frames, **self._predict_kwargs())
            return [self._parse_result(r) for r in results]
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            import torch

            torch.cuda.empty_cache()
            logger.warning(
                f"Detection OOM at batch={len(frames)} - falling back to per-frame. "
                "Lower stage1.detector.img_size or MTMC_DET_BATCH to avoid this."
            )
            return [self.detect(f) for f in frames]

    def detect_to_array(self, frame: np.ndarray) -> np.ndarray:
        """Run detection and return as numpy array for BoxMOT."""
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
