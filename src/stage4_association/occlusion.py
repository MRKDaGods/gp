"""Occlusion estimates for Stage 4 association."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Tuple

from src.core.data_models import Tracklet, TrackletFrame


BBox = Tuple[float, float, float, float]
TrackletKey = Tuple[str, int]


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    return cfg.get(key, default)


def _compute_iou(box_a: BBox, box_b: BBox) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / max(union, 1e-6)


def compute_tracklet_occlusion(
    tracklets_by_camera: Dict[str, List[Tracklet]],
    cfg: Any,
) -> Dict[TrackletKey, bool]:
    """Flag tracklets whose real detections are frequently occluded."""
    occ_box_thresh = float(_cfg_get(cfg, "occ_box_thresh", 0.6))
    occ_frac_thresh = float(_cfg_get(cfg, "occ_frac_thresh", 0.3))

    occluded: Dict[TrackletKey, bool] = {}
    total_real: DefaultDict[TrackletKey, int] = defaultdict(int)
    total_occluded: DefaultDict[TrackletKey, int] = defaultdict(int)
    by_frame: DefaultDict[Tuple[str, int], List[Tuple[TrackletKey, TrackletFrame]]] = defaultdict(list)

    for camera_id, tracklets in tracklets_by_camera.items():
        for tracklet in tracklets:
            tracklet_camera = tracklet.camera_id or camera_id
            key = (tracklet_camera, tracklet.track_id)
            occluded[key] = False
            for frame in tracklet.frames:
                if frame.confidence <= 0.0:
                    continue
                total_real[key] += 1
                by_frame[(tracklet_camera, frame.frame_id)].append((key, frame))

    for entries in by_frame.values():
        if len(entries) < 2:
            continue
        for index, (key, frame) in enumerate(entries):
            max_iou = 0.0
            for other_index, (_, other_frame) in enumerate(entries):
                if other_index == index:
                    continue
                max_iou = max(max_iou, _compute_iou(frame.bbox, other_frame.bbox))
            if max_iou >= occ_box_thresh:
                total_occluded[key] += 1

    for key, count in total_real.items():
        occluded[key] = (total_occluded[key] / count) >= occ_frac_thresh

    return occluded