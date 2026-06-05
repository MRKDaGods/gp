"""Frame extraction from video files."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
from loguru import logger

from src.core.data_models import FrameInfo
from src.core.video_utils import get_video_info, read_video_frames
from src.stage0_ingestion.preprocessor import preprocess_frame


def extract_frames_from_video(
    video_path: str | Path,
    output_dir: str | Path,
    camera_id: str,
    target_fps: float = 10.0,
    target_size: Optional[Tuple[int, int]] = None,
    normalize: bool = False,
    denoise: bool = False,
    denoise_strength: int = 3,
    max_frames: Optional[int] = None,
    clahe: bool = False,
    clahe_clip_limit: float = 2.0,
    time_offset: float = 0.0,
    lossless: bool = False,
) -> List[FrameInfo]:
    """Extract frames from a single video and save as images.

    Args:
        video_path: Path to the source video.
        output_dir: Directory to write extracted frames.
        camera_id: Camera identifier for this video.
        target_fps: Target frame rate for extraction.
        target_size: (width, height) to resize to, or None for original.
        normalize: Whether to apply pixel normalization.
        denoise: Whether to apply denoising.
        denoise_strength: Bilateral filter d parameter.
        max_frames: Maximum frames to extract (for smoke tests).
        clahe: Whether to apply CLAHE enhancement.
        clahe_clip_limit: CLAHE contrast clip limit.
        time_offset: Camera-specific time offset in seconds for synchronization.
        lossless: If True, save as PNG (lossless) instead of JPEG.

    Returns:
        List of FrameInfo for each extracted frame.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    info = get_video_info(video_path)
    logger.debug(
        f"Video {video_path.name}: {info.width}x{info.height} @ {info.fps:.1f} FPS, "
        f"{info.total_frames} frames, {info.duration:.1f}s"
    )

    # Estimate how many frames we'll extract so we can emit per-frame progress.
    # The backend parses "[PROGRESS] camera=X frame=i total=N" (same protocol as
    # stage 1) to drive a smooth ingestion progress bar instead of a frozen one.
    native_fps = info.fps if info.fps and info.fps > 0 else 30.0
    if target_fps and 0 < target_fps < native_fps:
        sample_interval = max(1, round(native_fps / target_fps))
    else:
        sample_interval = 1
    expected_total = info.total_frames // sample_interval if info.total_frames > 0 else 0
    if max_frames is not None:
        expected_total = min(expected_total, max_frames) if expected_total > 0 else max_frames
    expected_total = max(expected_total, 1)
    # Throttle to ~1% steps so stdout stays light while the bar still moves.
    progress_every = max(1, expected_total // 100)

    frames: List[FrameInfo] = []

    for extracted_count, (frame_idx, timestamp, frame) in enumerate(
        read_video_frames(video_path, target_fps=target_fps, max_frames=max_frames),
        start=1,
    ):
        # Apply preprocessing
        frame = preprocess_frame(
            frame,
            target_size=target_size,
            normalize=normalize,
            denoise=denoise,
            denoise_strength=denoise_strength,
            clahe=clahe,
            clahe_clip_limit=clahe_clip_limit,
        )

        # Apply camera time synchronization offset
        synced_timestamp = timestamp + time_offset

        # Save frame (PNG for lossless / JPEG for speed)
        if lossless:
            frame_filename = f"frame_{frame_idx:06d}.png"
            frame_path = output_dir / frame_filename
            cv2.imwrite(str(frame_path), frame)
        else:
            frame_filename = f"frame_{frame_idx:06d}.jpg"
            frame_path = output_dir / frame_filename
            cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        h, w = frame.shape[:2]
        frames.append(
            FrameInfo(
                frame_id=frame_idx,
                camera_id=camera_id,
                timestamp=synced_timestamp,
                frame_path=str(frame_path),
                width=w,
                height=h,
            )
        )

        # Emit throttled per-frame progress (consumed by the serving backend).
        if extracted_count % progress_every == 0 or extracted_count >= expected_total:
            logger.info(
                f"[PROGRESS] camera={camera_id} frame={extracted_count} "
                f"total={max(expected_total, extracted_count)}"
            )

    # Final marker so the camera reads as fully complete even if the estimate was off.
    actual = len(frames)
    if actual:
        logger.info(f"[PROGRESS] camera={camera_id} frame={actual} total={actual}")

    return frames
