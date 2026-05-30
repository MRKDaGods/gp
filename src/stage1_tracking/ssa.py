"""Stationary Sensitive Association post-processing for Stage 1 tracklets."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, List, Optional, Sequence, Tuple

from src.core.data_models import Tracklet, TrackletFrame


BBox = Tuple[float, float, float, float]


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    return cfg.get(key, default)


def _center(bbox: BBox) -> Tuple[float, float]:
    return ((bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5)


def _window_displacement(
    frames: Sequence[TrackletFrame],
    index: int,
    window: int,
) -> Optional[float]:
    if index + 1 < window or index <= 0:
        return None

    cx, cy = _center(frames[index].bbox)
    start = max(0, index - window + 1)
    return max(
        math.hypot(cx - px, cy - py)
        for px, py in (_center(frame.bbox) for frame in frames[start:index])
    )


def _best_recent_detection(
    frames: Sequence[TrackletFrame],
    index: int,
    lookback: int,
) -> Optional[TrackletFrame]:
    start = max(0, index - lookback + 1)
    candidates = [
        frame for frame in frames[start:index + 1]
        if frame.confidence > 0.0
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda frame: frame.confidence)


def apply_ssa(tracklets: List[Tracklet], cfg: Any) -> List[Tracklet]:
    """Freeze stationary tracklet boxes to recent high-confidence detections.

    This is a default-off Stage 1 post-processing pass. Interpolated boxes
    (confidence == 0) can be overwritten, but they are never used as anchors.
    """
    if not bool(_cfg_get(cfg, "enabled", True)):
        return tracklets

    window = max(int(_cfg_get(cfg, "window", 10)), 2)
    disp_thresh = float(_cfg_get(cfg, "disp_thresh", 8.0))
    freeze_lookback = max(int(_cfg_get(cfg, "freeze_lookback", 15)), 1)
    inherit_conf = bool(_cfg_get(cfg, "inherit_conf", False))

    stabilized: List[Tracklet] = []
    for tracklet in tracklets:
        frames = tracklet.frames
        if len(frames) < window:
            stabilized.append(tracklet)
            continue

        changed = False
        new_frames: List[TrackletFrame] = []
        for index, frame in enumerate(frames):
            disp = _window_displacement(frames, index, window)
            if disp is None or disp >= disp_thresh:
                new_frames.append(frame)
                continue

            anchor = _best_recent_detection(frames, index, freeze_lookback)
            if anchor is None:
                new_frames.append(frame)
                continue

            confidence = anchor.confidence if inherit_conf else frame.confidence
            replacement = replace(frame, bbox=anchor.bbox, confidence=confidence)
            changed = changed or replacement.bbox != frame.bbox or replacement.confidence != frame.confidence
            new_frames.append(replacement)

        stabilized.append(replace(tracklet, frames=new_frames) if changed else tracklet)

    return stabilized