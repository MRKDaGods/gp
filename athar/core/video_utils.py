"""Video reading and frame extraction utilities."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Stage 0 frame decode: use the GPU's hardware video decoder (NVDEC / DXVA /
# VAAPI) when the OpenCV+FFmpeg build exposes it, so ingestion isn't pinned to
# CPU decode. VIDEO_ACCELERATION_ANY negotiates HW if available and transparently
# falls back to software otherwise (same BGR frames either way), so it's a safe
# default. Set MTMC_DISABLE_HW_DECODE=1 to force plain CPU decode.
_HW_DECODE_AVAILABLE = hasattr(cv2, "CAP_PROP_HW_ACCELERATION") and hasattr(
    cv2, "VIDEO_ACCELERATION_ANY"
)
_HW_DECODE_ENABLED = _HW_DECODE_AVAILABLE and os.environ.get(
    "MTMC_DISABLE_HW_DECODE", ""
).lower() not in ("1", "true", "yes")
_hw_decode_logged = False


def _open_capture(video_path: str) -> cv2.VideoCapture:
    """Open a video, preferring GPU-accelerated decode with a CPU fallback."""
    global _hw_decode_logged
    if _HW_DECODE_ENABLED:
        try:
            cap = cv2.VideoCapture(
                video_path,
                cv2.CAP_FFMPEG,
                [int(cv2.CAP_PROP_HW_ACCELERATION), int(cv2.VIDEO_ACCELERATION_ANY)],
            )
            if cap.isOpened():
                if not _hw_decode_logged:
                    _hw_decode_logged = True
                    negotiated = cap.get(cv2.CAP_PROP_HW_ACCELERATION)
                    logger.info(
                        "Stage 0 decode: hardware acceleration "
                        f"{'ACTIVE' if negotiated and negotiated > 0 else 'requested (FFmpeg negotiated software on this build)'}"
                    )
                return cap
            cap.release()
        except cv2.error:
            pass  # HW path unsupported by this build - fall back to plain decode
    return cv2.VideoCapture(video_path)


@dataclass
class VideoInfo:
    """Metadata about a video file."""

    path: str
    width: int
    height: int
    fps: float
    total_frames: int
    duration: float  # seconds
    codec: str


def get_video_info(video_path: str | Path) -> VideoInfo:
    """Read video metadata without decoding frames."""
    video_path = str(video_path)
    cap = _open_capture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0.0
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = (
            chr(fourcc & 0xFF)
            + chr((fourcc >> 8) & 0xFF)
            + chr((fourcc >> 16) & 0xFF)
            + chr((fourcc >> 24) & 0xFF)
        )
        return VideoInfo(
            path=video_path,
            width=width,
            height=height,
            fps=fps,
            total_frames=total_frames,
            duration=duration,
            codec=codec.strip(),
        )
    finally:
        cap.release()


def read_video_frames(
    video_path: str | Path,
    target_fps: Optional[float] = None,
    max_frames: Optional[int] = None,
) -> Iterator[Tuple[int, float, np.ndarray]]:
    """Yield (frame_index, timestamp_seconds, frame_bgr) from a video."""
    video_path = str(video_path)
    cap = _open_capture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if native_fps <= 0:
        native_fps = 30.0
    frame_interval = 1
    if target_fps is not None and 0 < target_fps < native_fps:
        frame_interval = max(1, round(native_fps / target_fps))

    frame_idx = 0
    yielded = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                timestamp = frame_idx / native_fps
                yield frame_idx, timestamp, frame
                yielded += 1

                if max_frames is not None and yielded >= max_frames:
                    break

            frame_idx += 1
    finally:
        cap.release()


def read_single_frame(video_path: str | Path, frame_index: int) -> np.ndarray:
    """Read a specific frame from a video by index."""
    cap = _open_capture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            raise IOError(f"Cannot read frame {frame_index} from {video_path}")
        return frame
    finally:
        cap.release()


def write_video(
    frames: Iterator[np.ndarray],
    output_path: str | Path,
    fps: float = 10.0,
    codec: str = "mp4v",
    size: Optional[Tuple[int, int]] = None,
) -> None:
    """Write frames to a video file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    try:
        for frame in frames:
            if writer is None:
                h, w = frame.shape[:2]
                frame_size = size or (w, h)
                fourcc = cv2.VideoWriter_fourcc(*codec)
                writer = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)
            writer.write(frame)
    finally:
        if writer is not None:
            writer.release()
