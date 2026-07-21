"""Bidirectional tracking helpers for Stage 1."""

from __future__ import annotations

from typing import Any, Iterable, List, Optional

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

from athar.core.data_models import FrameInfo, Tracklet
from athar.components.tracking.detector import Detector
from athar.components.tracking.tracker import TrackerWrapper
from athar.components.tracking.tracklet_builder import TrackletBuilder


BWD_ID_OFFSET = 1_000_000


def run_backward_pass(
    cam_frames: Iterable[FrameInfo],
    detector: Detector,
    stage_cfg: Any,  # attribute-style config (OmegaConf-shaped); omegaconf not required
    camera_id: str,
    *,
    min_tracklet_length: int,
    min_tracklet_area: float,
    interpolate: bool,
    interpolation_max_gap: int,
    intra_merge: bool,
    merge_max_time_gap: float,
    merge_max_iou_distance: float,
    roi_mask: Optional[np.ndarray],
) -> List[Tracklet]:
    """Run a fresh tracker and builder over camera frames in reverse order."""
    frames = list(cam_frames)
    tracker = TrackerWrapper(
        algorithm=stage_cfg.tracker.algorithm,
        reid_weights=stage_cfg.tracker.get("reid_weights"),
        device=stage_cfg.tracker.device,
        half=stage_cfg.tracker.get("half", True),
        tracker_config=stage_cfg.tracker,
    )

    builder = TrackletBuilder(
        camera_id=camera_id,
        min_length=min_tracklet_length,
        min_area=min_tracklet_area,
        interpolate=interpolate,
        interpolation_max_gap=interpolation_max_gap,
        intra_merge=intra_merge,
        merge_max_time_gap=merge_max_time_gap,
        merge_max_iou_distance=merge_max_iou_distance,
    )

    failed_frames = 0
    for frame_info in reversed(frames):
        frame = cv2.imread(frame_info.frame_path)
        if frame is None:
            logger.warning(f"Cannot read frame during backward pass: {frame_info.frame_path}")
            failed_frames += 1
            continue

        if roi_mask is not None:
            frame_masked = cv2.bitwise_and(frame, frame, mask=roi_mask)
        else:
            frame_masked = frame

        detections = detector.detect(frame_masked)
        tracks = tracker.update(detections, frame)
        builder.add_frame(
            tracks=tracks,
            frame_id=frame_info.frame_id,
            timestamp=frame_info.timestamp,
        )

    tracklets = builder.finalize()
    for tracklet in tracklets:
        tracklet.track_id += BWD_ID_OFFSET

    if failed_frames:
        logger.warning(
            f"Backward pass frame read failures for {camera_id}: "
            f"{failed_frames}/{len(frames)}"
        )
    logger.info(f"  Camera {camera_id}: {len(tracklets)} backward tracklets")
    return tracklets