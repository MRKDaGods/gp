"""v2 Detector protocol over the ported v1 Ultralytics YOLO wrapper."""

from __future__ import annotations

import logging
from typing import Optional, Sequence

from athar.components.protocols import FrameBatch
from athar.core.timebase import SceneClock
from athar.core.types import BBox, Detection, EntityClass

logger = logging.getLogger(__name__)

# COCO detector ids <-> ATHAR entity classes. TUKTUK has no COCO id — it
# arrives with a fine-tuned detector head later; the mapping is data, not code.
COCO_TO_ENTITY: dict[int, EntityClass] = {
    0: EntityClass.PERSON,
    2: EntityClass.CAR,
    3: EntityClass.MOTORCYCLE,
    5: EntityClass.BUS,
    7: EntityClass.TRUCK,
}
ENTITY_TO_COCO: dict[EntityClass, int] = {v: k for k, v in COCO_TO_ENTITY.items()}


class YoloDetectorAdapter:
    """One detection pass, all classes (D3) — branch filtering happens later.

    Wraps the ported ``athar.components.tracking.detector.Detector`` (the
    parity component, D18). ``scene_clock`` is injected by the stage at
    creation time; it is how pixel-space hits become scene-time Detections.
    """

    def __init__(
        self,
        scene_clock: SceneClock,
        model_path: str = "models/yolo26m.pt",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        coco_classes: Optional[Sequence[int]] = None,
        device: str = "cpu",
        half: bool = False,
        img_size: int = 640,
        agnostic_nms: bool = True,
    ) -> None:
        from athar.components.tracking.detector import Detector as V1Detector

        self._clock = scene_clock
        classes = list(coco_classes) if coco_classes is not None else sorted(COCO_TO_ENTITY)
        unknown = [c for c in classes if c not in COCO_TO_ENTITY]
        if unknown:
            raise ValueError(f"COCO classes with no EntityClass mapping: {unknown}")
        self._detector = V1Detector(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            classes=classes,
            device=device,
            half=half,
            img_size=img_size,
            agnostic_nms=agnostic_nms,
        )

    def detect(self, batch: FrameBatch) -> list[Detection]:
        timebase = self._clock.require(batch.camera_id)
        images = batch.images()
        per_frame = self._detector.detect_batch([images[i] for i in range(len(images))])
        detections: list[Detection] = []
        for frame_pos, v1_dets in enumerate(per_frame):
            frame_index = batch.frame_indices[frame_pos]
            ts = timebase.to_scene(frame_index)
            for d in v1_dets:
                entity = COCO_TO_ENTITY.get(d.class_id)
                if entity is None:  # detector was asked to filter; belt & braces
                    continue
                x1, y1, x2, y2 = d.bbox
                detections.append(
                    Detection(
                        camera_id=batch.camera_id,
                        frame_index=frame_index,
                        ts_scene_s=ts,
                        bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
                        entity_class=entity,
                        confidence=min(max(d.confidence, 0.0), 1.0),
                    )
                )
        return detections
