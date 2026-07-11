"""Stage 1 - Per-Camera Detection & Tracking pipeline."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from loguru import logger
from omegaconf import DictConfig

from src.core.data_models import FrameInfo, Tracklet
from src.core.gpu_utils import effective_half
from src.core.io_utils import save_tracklets_by_camera
from src.stage1_tracking.detector import Detector
from src.stage1_tracking.ssa import apply_ssa
from src.stage1_tracking.tracker import TrackerWrapper
from src.stage1_tracking.tracklet_builder import TrackletBuilder

# Ultralytics logs a noisy "'half' is deprecated" warning on every predict() call.
# We already choose precision deliberately (see effective_half); silence the spam.
logging.getLogger("ultralytics").setLevel(logging.ERROR)


def _load_roi_mask(cfg: DictConfig, camera_id: str) -> Optional[np.ndarray]:
    """Load ROI mask for a camera if available."""
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
            f"ROI mask for {camera_id} has only {coverage:.1f}% coverage - "
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
    """Run detection and tracking on all cameras."""
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

    # Initialize detector. FP16 is honored only on GPUs where it's actually fast
    # (Volta+); on Pascal/Maxwell (e.g. GTX 1050 Ti) it silently uses FP32.
    detector = Detector(
        model_path=stage_cfg.detector.model,
        confidence_threshold=stage_cfg.detector.confidence_threshold,
        iou_threshold=stage_cfg.detector.iou_threshold,
        classes=list(stage_cfg.detector.classes),
        device=stage_cfg.detector.device,
        half=effective_half(stage_cfg.detector.get("half", True), str(stage_cfg.detector.device)),
        img_size=stage_cfg.detector.get("img_size", 640),
        agnostic_nms=stage_cfg.detector.get("agnostic_nms", True),
    )

    # Detection batch size - process this many frames per GPU call so the GPU
    # stays fed instead of idling between single-frame inferences. Batching only
    # helps once a single frame no longer saturates the GPU, so scale the default
    # by resolution: a 1280px frame already maxes a modest GPU (batch=1, no
    # regression), while 640px leaves headroom (batch=4, ~1.4x). Override via
    # MTMC_DET_BATCH.
    _imgsz = int(stage_cfg.detector.get("img_size", 640))
    _default_batch = 4 if _imgsz <= 768 else (2 if _imgsz <= 1024 else 1)
    try:
        det_batch = max(1, int(os.environ.get("MTMC_DET_BATCH", str(_default_batch))))
    except ValueError:
        det_batch = _default_batch

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

    # Emit total camera count so consumers (e.g. the serving backend) can render
    # accurate cross-camera progress before any camera has finished.
    logger.info(f"[PROGRESS] cameras_total={len(frames_by_camera)}")

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

        # Initialize tracker for this camera (FP16 only where it's fast).
        tracker = TrackerWrapper(
            algorithm=stage_cfg.tracker.algorithm,
            reid_weights=stage_cfg.tracker.get("reid_weights"),
            device=stage_cfg.tracker.device,
            half=effective_half(stage_cfg.tracker.get("half", True), str(stage_cfg.tracker.device)),
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

        cam_total = len(cam_frames)
        # Throttle progress logs to ~every 1% (min 1 frame) to keep stdout light
        # while still giving the backend a smooth per-frame signal to interpolate.
        progress_every = max(1, cam_total // 100)

        # Process frames in batches: detection runs on the whole batch in one GPU
        # call (high utilization), then tracking is applied sequentially in frame
        # order (the tracker is stateful and must stay ordered).
        for batch_start in range(0, cam_total, det_batch):
            batch_infos = cam_frames[batch_start:batch_start + det_batch]

            loaded: List[tuple] = []  # (frame_info, original_frame, masked_frame)
            for frame_info in batch_infos:
                frame = cv2.imread(frame_info.frame_path)
                if frame is None:
                    logger.warning(f"Cannot read frame: {frame_info.frame_path}")
                    failed_frames += 1
                    continue
                # ROI mask (non-road -> black) before detection; the tracker still
                # gets the original frame so ReID appearance features are clean.
                masked = cv2.bitwise_and(frame, frame, mask=roi_mask) if roi_mask is not None else frame
                loaded.append((frame_info, frame, masked))

            if not loaded:
                continue

            batch_dets = detector.detect_batch([m for (_, _, m) in loaded])

            for (frame_info, frame, _), detections in zip(loaded, batch_dets):
                tracks = tracker.update(detections, frame)
                builder.add_frame(
                    tracks=tracks,
                    frame_id=frame_info.frame_id,
                    timestamp=frame_info.timestamp,
                )

            done = min(batch_start + det_batch, cam_total)
            if batch_start % max(progress_every, det_batch) < det_batch or done == cam_total:
                logger.info(f"[PROGRESS] camera={camera_id} frame={done} total={cam_total}")

        # Finalize tracklets
        tracklets = builder.finalize()
        bt_cfg = stage_cfg.get("bidirectional", {})
        if bt_cfg.get("enabled", False):
            from src.stage1_tracking.bidirectional import run_backward_pass

            backward_tracklets = run_backward_pass(
                cam_frames,
                detector,
                stage_cfg,
                camera_id,
                min_tracklet_length=min_tracklet_length,
                min_tracklet_area=min_tracklet_area,
                interpolate=interpolate,
                interpolation_max_gap=interpolation_max_gap,
                intra_merge=intra_merge,
                merge_max_time_gap=merge_max_time_gap,
                merge_max_iou_distance=merge_max_iou_distance,
                roi_mask=roi_mask,
            )
            tracklets = tracklets + backward_tracklets
        ssa_cfg = stage_cfg.get("ssa", {})
        if ssa_cfg.get("enabled", False):
            tracklets = apply_ssa(tracklets, ssa_cfg)
        all_tracklets[camera_id] = tracklets
        logger.info(f"  Camera {camera_id}: {len(tracklets)} tracklets")

    # Check frame failure rate - for forensics, >20% loss is unacceptable
    if failed_frames > 0:
        failure_rate = failed_frames / max(total_frames, 1) * 100
        logger.warning(
            f"Frame read failures: {failed_frames}/{total_frames} ({failure_rate:.1f}%)"
        )
        if failure_rate > 20.0:
            raise RuntimeError(
                f"Too many frame read failures: {failed_frames}/{total_frames} "
                f"({failure_rate:.1f}%) - aborting stage 1 to prevent incomplete evidence"
            )

    # Save
    save_tracklets_by_camera(all_tracklets, output_dir)
    logger.info(f"Saved tracklets for {len(all_tracklets)} cameras to {output_dir}")

    return all_tracklets
