"""Generate ROI masks from CityFlowV2 video files.

Since the official roi.jpg files are not included in our dataset copy,
this script generates approximate ROI masks by:
1. Sampling N frames uniformly from each video
2. Computing the median frame (static background)
3. Using edge detection + morphological operations to identify road regions
4. Saving as roi.jpg in each camera directory

The masks don't need to be perfect — even coarse masks that remove 
obvious non-road regions (sky, buildings, sidewalks) significantly 
reduce false positive detections.

Usage:
    python scripts/generate_roi_masks.py [--data-dir data/raw/cityflowv2]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from loguru import logger


def generate_roi_mask(
    video_path: str | Path,
    n_samples: int = 100,
    dilate_kernel: int = 25,
) -> np.ndarray:
    """Generate a binary ROI mask from a traffic camera video.

    Strategy: Use background subtraction on sampled frames to
    identify regions where motion occurs (= road area).

    Args:
        video_path: Path to the video file.
        n_samples: Number of frames to sample for background estimation.
        dilate_kernel: Kernel size for morphological dilation (fills gaps).

    Returns:
        Binary mask (uint8, 0 or 255) with same dimensions as video.
    """
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if total_frames < 10:
        logger.warning(f"Video has only {total_frames} frames, generating full mask")
        cap.release()
        return np.full((height, width), 255, dtype=np.uint8)

    # Sample frame indices uniformly
    indices = np.linspace(0, total_frames - 1, n_samples, dtype=int)

    # Accumulate motion across frame pairs using background subtractor
    motion_accum = np.zeros((height, width), dtype=np.float64)

    # Read reference frame (first)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        return np.full((height, width), 255, dtype=np.uint8)

    # Use MOG2 background subtractor for motion detection
    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=n_samples, varThreshold=50, detectShadows=False,
    )
    # Feed the first frame multiple times to establish background
    gray_first = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    for _ in range(5):
        bg_sub.apply(first_frame, learningRate=0.5)

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        # Apply background subtractor
        fg_mask = bg_sub.apply(frame, learningRate=0.01)
        motion_accum += (fg_mask > 0).astype(np.float64)

    cap.release()

    # Normalize motion accumulator
    if motion_accum.max() > 0:
        motion_norm = (motion_accum / motion_accum.max() * 255).astype(np.uint8)
    else:
        return np.full((height, width), 255, dtype=np.uint8)

    # Threshold: areas where motion occurred in >5% of frames = road
    threshold = max(1, int(n_samples * 0.05))
    road_mask = (motion_accum >= threshold).astype(np.uint8) * 255

    # Morphological operations to clean up
    # Close gaps in the road surface
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilate_kernel, dilate_kernel),
    )
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    # Dilate to capture road edges
    small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    road_mask = cv2.dilate(road_mask, small_kernel, iterations=2)

    # Fill holes: find the largest contour and fill it
    contours, _ = cv2.findContours(
        road_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    if contours:
        # Keep only large contours (>1% of image area)
        min_area = height * width * 0.01
        filled = np.zeros_like(road_mask)
        for c in contours:
            if cv2.contourArea(c) >= min_area:
                cv2.drawContours(filled, [c], -1, 255, cv2.FILLED)
        road_mask = filled

    logger.info(
        f"ROI mask: {road_mask.sum() / 255 / (height * width) * 100:.1f}% "
        f"coverage ({width}x{height})"
    )
    return road_mask


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ROI masks")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw/cityflowv2",
        help="Root directory containing camera folders",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=200,
        help="Number of frames to sample per video",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing roi.jpg files",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    cameras = sorted(
        d for d in data_dir.iterdir()
        if d.is_dir() and (d / "vdo.avi").exists()
    )

    if not cameras:
        logger.error(f"No camera directories with vdo.avi found in {data_dir}")
        return

    for cam_dir in cameras:
        roi_path = cam_dir / "roi.jpg"
        if roi_path.exists() and not args.force:
            logger.info(f"Skipping {cam_dir.name} (roi.jpg exists)")
            continue

        video_path = cam_dir / "vdo.avi"
        logger.info(f"Generating ROI mask for {cam_dir.name}...")

        mask = generate_roi_mask(video_path, n_samples=args.n_samples)
        cv2.imwrite(str(roi_path), mask)
        logger.info(f"Saved {roi_path}")


if __name__ == "__main__":
    main()
