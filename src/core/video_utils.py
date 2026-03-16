"""Video reading and frame extraction utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np


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
    cap = cv2.VideoCapture(video_path)
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
    """Yield (frame_index, timestamp_seconds, frame_bgr) from a video.

    Args:
        video_path: Path to the video file.
        target_fps: If set, subsample to this frame rate. None = use native FPS.
        max_frames: Stop after this many yielded frames. None = read all.

    Yields:
        (frame_index, timestamp_sec, frame) where frame is BGR uint8 numpy array.
    """
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
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
    cap = cv2.VideoCapture(str(video_path))
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
    """Write frames to a video file.

    Args:
        frames: Iterator of BGR uint8 numpy arrays.
        output_path: Output video file path.
        fps: Output frame rate.
        codec: FourCC codec string.
        size: (width, height). If None, inferred from first frame.
    """
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
