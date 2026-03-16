"""Stage 0 — Ingestion & Pre-Processing pipeline.

Reads raw videos, extracts frames at a target FPS, applies preprocessing,
and writes frames + manifest to the stage output directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from loguru import logger
from omegaconf import DictConfig

from src.core.constants import FRAME_MANIFEST_FILE
from src.core.data_models import FrameInfo
from src.core.io_utils import save_frame_manifest
from src.stage0_ingestion.frame_extractor import extract_frames_from_video
from src.stage0_ingestion.preprocessor import preprocess_frame


def run_stage0(
    cfg: DictConfig,
    output_dir: str | Path,
    smoke_test: bool = False,
) -> List[FrameInfo]:
    """Run the full ingestion pipeline.

    Args:
        cfg: Full pipeline config (uses cfg.stage0).
        output_dir: Directory for this run's stage0 outputs.
        smoke_test: If True, process only the first 10 frames per video.

    Returns:
        List of FrameInfo for all extracted frames across all cameras.
    """
    stage_cfg = cfg.stage0
    input_dir = Path(stage_cfg.input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate input directory is accessible
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    # Discover videos
    video_extensions = stage_cfg.get("video_extensions", [".mp4", ".avi", ".mkv", ".mov"])
    video_paths = _discover_videos(input_dir, video_extensions)

    if not video_paths:
        logger.warning(f"No videos found in {input_dir}")
        return []

    # Apply camera filter if specified (cameras: [S01_c001, S01_c002, ...])
    camera_filter = stage_cfg.get("cameras", None)
    if camera_filter:
        allowed = set(camera_filter)
        video_paths = [
            v for v in video_paths
            if _camera_id_from_path(v, input_dir) in allowed
        ]
        logger.info(f"Camera filter active — keeping {len(video_paths)} cameras: {sorted(allowed)}")

    logger.info(f"Processing {len(video_paths)} videos from {input_dir}")

    # Parse target size
    target_size = stage_cfg.get("target_size")
    if target_size is not None:
        target_size = tuple(target_size)  # (width, height)

    all_frames: List[FrameInfo] = []
    max_frames = 10 if smoke_test else None
    failed_videos: List[str] = []

    # Load per-camera time offsets for synchronization
    time_offsets = dict(stage_cfg.get("time_offsets", {}))

    for video_path in video_paths:
        camera_id = _camera_id_from_path(video_path, input_dir)
        frames_dir = output_dir / camera_id
        frames_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Processing camera {camera_id}: {video_path}")

        cam_time_offset = float(time_offsets.get(camera_id, 0.0))
        if cam_time_offset != 0.0:
            logger.info(f"  Applying time offset: {cam_time_offset:.3f}s")

        try:
            frames = extract_frames_from_video(
                video_path=video_path,
                output_dir=frames_dir,
                camera_id=camera_id,
                target_fps=stage_cfg.get("output_fps", 10),
                target_size=target_size,
                normalize=stage_cfg.get("normalize", False),
                denoise=stage_cfg.get("denoise", False),
                denoise_strength=stage_cfg.get("denoise_strength", 3),
                max_frames=max_frames,
                clahe=stage_cfg.get("clahe", False),
                clahe_clip_limit=stage_cfg.get("clahe_clip_limit", 2.0),
                time_offset=cam_time_offset,
                lossless=stage_cfg.get("lossless", False),
            )
        except Exception as exc:
            logger.error(f"FAILED to process video {video_path}: {exc}")
            failed_videos.append(str(video_path))
            continue

        logger.info(f"  Extracted {len(frames)} frames from camera {camera_id}")
        all_frames.extend(frames)

    if failed_videos:
        logger.error(
            f"{len(failed_videos)}/{len(video_paths)} videos failed: "
            f"{failed_videos}"
        )
        if len(failed_videos) == len(video_paths):
            raise RuntimeError("All videos failed to process — aborting stage 0")

    # Save manifest
    manifest_path = output_dir / FRAME_MANIFEST_FILE
    save_frame_manifest(all_frames, manifest_path)
    logger.info(f"Saved frame manifest ({len(all_frames)} frames) to {manifest_path}")

    return all_frames


def _discover_videos(
    input_dir: Path, extensions: List[str]
) -> List[Path]:
    """Find all video files in the input directory (recursive)."""
    videos = []
    for ext in extensions:
        videos.extend(input_dir.rglob(f"*{ext}"))
    return sorted(videos)


def _camera_id_from_path(video_path: Path, root_dir: Path) -> str:
    """Derive camera ID from the video's relative path.

    Uses the parent directory name if nested (e.g., cam001/video.mp4 -> cam001),
    otherwise uses the video filename stem.
    """
    relative = video_path.relative_to(root_dir)
    if len(relative.parts) > 1:
        return relative.parts[-2]
    return video_path.stem
