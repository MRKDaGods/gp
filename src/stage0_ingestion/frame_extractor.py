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
) -> List[FrameInfo]:
    """Extract frames from a single video and save as JPEG images.

    Args:
        video_path: Path to the source video.
        output_dir: Directory to write extracted JPEG frames.
        camera_id: Camera identifier for this video.
        target_fps: Target frame rate for extraction.
        target_size: (width, height) to resize to, or None for original.
        normalize: Whether to apply pixel normalization.
        denoise: Whether to apply denoising.
        denoise_strength: Bilateral filter d parameter.
        max_frames: Maximum frames to extract (for smoke tests).

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

    frames: List[FrameInfo] = []

    for frame_idx, timestamp, frame in read_video_frames(
        video_path, target_fps=target_fps, max_frames=max_frames
    ):
        # Apply preprocessing
        frame = preprocess_frame(
            frame,
            target_size=target_size,
            normalize=normalize,
            denoise=denoise,
            denoise_strength=denoise_strength,
        )

        # Save frame
        frame_filename = f"frame_{frame_idx:06d}.jpg"
        frame_path = output_dir / frame_filename
        cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        h, w = frame.shape[:2]
        frames.append(
            FrameInfo(
                frame_id=frame_idx,
                camera_id=camera_id,
                timestamp=timestamp,
                frame_path=str(frame_path),
                width=w,
                height=h,
            )
        )

    return frames
