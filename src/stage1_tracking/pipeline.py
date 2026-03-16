"""Stage 1 — Per-Camera Detection & Tracking pipeline.

Runs YOLO detection and BoxMOT tracking on extracted frames for each camera,
producing per-camera Tracklet lists.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from loguru import logger
from omegaconf import DictConfig

from src.core.data_models import FrameInfo, Tracklet
from src.core.io_utils import save_tracklets_by_camera
from src.stage1_tracking.detector import Detector
from src.stage1_tracking.tracker import TrackerWrapper
from src.stage1_tracking.tracklet_builder import TrackletBuilder


def _load_roi_mask(cfg: DictConfig, camera_id: str) -> Optional[np.ndarray]:
    """Load ROI mask for a camera if available.

    Looks for roi.jpg in the camera's data directory.
    Returns a single-channel binary mask (uint8, 0 or 255), or None.
    """
    data_root = Path(cfg.stage0.input_dir)
    roi_path = data_root / camera_id / "roi.jpg"
    if not roi_path.exists():
        return None

    roi = cv2.imread(str(roi_path))
    if roi is None:
        return None

    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(roi_gray, 127, 255, cv2.THRESH_BINARY)
    coverage = mask.sum() / 255 / mask.size * 100
    if coverage < 10.0:
        logger.warning(
            f"ROI mask for {camera_id} has only {coverage:.1f}% coverage — "
            f"skipping (likely bad mask)"
        )
        return None
    logger.info(
        f"Loaded ROI mask for {camera_id}: {coverage:.1f}% coverage"
    )
    return mask


def run_stage1(
    cfg: DictConfig,
    frames: List[FrameInfo],
    output_dir: str | Path,
    smoke_test: bool = False,
) -> Dict[str, List[Tracklet]]:
    """Run detection and tracking on all cameras.

    Args:
        cfg: Full pipeline config (uses cfg.stage1).
        frames: List of FrameInfo from Stage 0.
        output_dir: Directory for this run's stage1 outputs.
        smoke_test: If True, process only first 10 frames per camera.

    Returns:
        Dict mapping camera_id to list of Tracklets.
    """
    stage_cfg = cfg.stage1
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group frames by camera
    frames_by_camera: Dict[str, List[FrameInfo]] = {}
    for f in frames:
        frames_by_camera.setdefault(f.camera_id, []).append(f)

    # Sort each camera's frames by frame_id
    for cam_id in frames_by_camera:
        frames_by_camera[cam_id].sort(key=lambda f: f.frame_id)

    # Initialize detector
    detector = Detector(
        model_path=stage_cfg.detector.model,
        confidence_threshold=stage_cfg.detector.confidence_threshold,
        iou_threshold=stage_cfg.detector.iou_threshold,
        classes=list(stage_cfg.detector.classes),
        device=stage_cfg.detector.device,
        half=stage_cfg.detector.get("half", True),
        img_size=stage_cfg.detector.get("img_size", 640),
        agnostic_nms=stage_cfg.detector.get("agnostic_nms", True),
    )

    min_tracklet_length = stage_cfg.get("min_tracklet_length", 5)
    min_tracklet_area = stage_cfg.get("min_tracklet_area", 500)

    # Interpolation & intra-camera merge settings
    interpolate = stage_cfg.get("interpolation", {}).get("enabled", True)
    interpolation_max_gap = stage_cfg.get("interpolation", {}).get("max_gap", 30)
    intra_merge = stage_cfg.get("intra_merge", {}).get("enabled", True)
    merge_max_time_gap = stage_cfg.get("intra_merge", {}).get("max_time_gap", 5.0)
    merge_max_iou_distance = stage_cfg.get("intra_merge", {}).get("max_iou_distance", 0.7)

    all_tracklets: Dict[str, List[Tracklet]] = {}
    failed_frames = 0
    total_frames = 0

    for camera_id, cam_frames in frames_by_camera.items():
        logger.info(f"Processing camera {camera_id}: {len(cam_frames)} frames")
        total_frames += len(cam_frames)

        if smoke_test:
            cam_frames = cam_frames[:10]

        # Load ROI mask (if available) to mask out non-road regions
        roi_cfg = stage_cfg.get("roi", {})
        roi_mask = None
        if roi_cfg.get("enabled", True):
            roi_mask = _load_roi_mask(cfg, camera_id)

        # Initialize tracker for this camera
        tracker = TrackerWrapper(
            algorithm=stage_cfg.tracker.algorithm,
            reid_weights=stage_cfg.tracker.get("reid_weights"),
            device=stage_cfg.tracker.device,
            half=stage_cfg.tracker.get("half", True),
            tracker_config=stage_cfg.tracker,
        )

        # Build tracklets
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

        for frame_info in cam_frames:
            import cv2

            frame = cv2.imread(frame_info.frame_path)
            if frame is None:
                logger.warning(f"Cannot read frame: {frame_info.frame_path}")
                failed_frames += 1
                continue

            # Apply ROI mask before detection (mask non-road regions to black).
            # The tracker still receives the original frame for ReID features.
            if roi_mask is not None:
                frame_masked = cv2.bitwise_and(frame, frame, mask=roi_mask)
            else:
                frame_masked = frame

            # Detect on masked frame (or original if no ROI)
            detections = detector.detect(frame_masked)

            # Track on original frame (unmasked for ReID appearance)
            tracks = tracker.update(detections, frame)

            # Accumulate
            builder.add_frame(
                tracks=tracks,
                frame_id=frame_info.frame_id,
                timestamp=frame_info.timestamp,
            )

        # Finalize tracklets
        tracklets = builder.finalize()
        all_tracklets[camera_id] = tracklets
        logger.info(f"  Camera {camera_id}: {len(tracklets)} tracklets")

    # Check frame failure rate — for forensics, >20% loss is unacceptable
    if failed_frames > 0:
        failure_rate = failed_frames / max(total_frames, 1) * 100
        logger.warning(
            f"Frame read failures: {failed_frames}/{total_frames} ({failure_rate:.1f}%)"
        )
        if failure_rate > 20.0:
            raise RuntimeError(
                f"Too many frame read failures: {failed_frames}/{total_frames} "
                f"({failure_rate:.1f}%) — aborting stage 1 to prevent incomplete evidence"
            )

    # Save
    save_tracklets_by_camera(all_tracklets, output_dir)
    logger.info(f"Saved tracklets for {len(all_tracklets)} cameras to {output_dir}")

    return all_tracklets
