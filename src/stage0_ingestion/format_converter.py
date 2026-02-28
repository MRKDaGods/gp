"""Format converters for various dataset annotation schemes.

Converts annotations from dataset-specific formats (AI City Challenge,
MOTChallenge, etc.) into the unified schema used by subsequent stages.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from loguru import logger

from src.core.constants import CLASS_NAMES
from src.core.data_models import Tracklet, TrackletFrame


def load_mot_annotations(
    gt_path: str | Path,
    camera_id: str,
    fps: float = 10.0,
    class_filter: Optional[set] = None,
) -> List[Tracklet]:
    """Load MOTChallenge-format ground truth and convert to Tracklets.

    MOT format (per line): frame, id, bb_left, bb_top, bb_width, bb_height, conf, class, visibility

    Args:
        gt_path: Path to gt.txt or similar MOT-format file.
        camera_id: Camera identifier.
        fps: Frame rate (used to compute timestamps).
        class_filter: Set of class IDs to keep. None = keep all.

    Returns:
        List of Tracklets.
    """
    gt_path = Path(gt_path)
    if not gt_path.exists():
        logger.warning(f"GT file not found: {gt_path}")
        return []

    # Collect frames per track
    tracks: Dict[int, List[TrackletFrame]] = {}
    track_classes: Dict[int, int] = {}

    with open(gt_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 7:
                continue

            frame_id = int(row[0])
            track_id = int(row[1])
            x, y, w, h = float(row[2]), float(row[3]), float(row[4]), float(row[5])
            conf = float(row[6])
            class_id = int(row[7]) if len(row) > 7 else 0

            if class_filter and class_id not in class_filter:
                continue

            bbox = (x, y, x + w, y + h)  # convert xywh to xyxy
            timestamp = frame_id / fps

            if track_id not in tracks:
                tracks[track_id] = []
                track_classes[track_id] = class_id

            tracks[track_id].append(
                TrackletFrame(
                    frame_id=frame_id,
                    timestamp=timestamp,
                    bbox=bbox,
                    confidence=conf,
                )
            )

    # Build Tracklet objects
    tracklets = []
    for track_id, frames in tracks.items():
        frames.sort(key=lambda f: f.frame_id)
        class_id = track_classes[track_id]
        tracklets.append(
            Tracklet(
                track_id=track_id,
                camera_id=camera_id,
                class_id=class_id,
                class_name=CLASS_NAMES.get(class_id, f"class_{class_id}"),
                frames=frames,
            )
        )

    logger.info(f"Loaded {len(tracklets)} tracklets from {gt_path}")
    return tracklets


def load_aic_annotations(
    gt_path: str | Path,
    camera_id: str,
    fps: float = 10.0,
) -> List[Tracklet]:
    """Load AI City Challenge format annotations.

    AIC format (per line): camera_id frame_id track_id x y w h
    Or varies by track — this handles the common format.

    Args:
        gt_path: Path to annotation file.
        camera_id: Camera identifier.
        fps: Frame rate for timestamp computation.

    Returns:
        List of Tracklets.
    """
    gt_path = Path(gt_path)
    if not gt_path.exists():
        logger.warning(f"AIC GT file not found: {gt_path}")
        return []

    tracks: Dict[int, List[TrackletFrame]] = {}

    with open(gt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue

            cam_id_raw = parts[0]
            frame_id = int(parts[1])
            track_id = int(parts[2])
            x, y, w, h = float(parts[3]), float(parts[4]), float(parts[5]), float(parts[6])

            bbox = (x, y, x + w, y + h)
            timestamp = frame_id / fps

            if track_id not in tracks:
                tracks[track_id] = []

            tracks[track_id].append(
                TrackletFrame(
                    frame_id=frame_id,
                    timestamp=timestamp,
                    bbox=bbox,
                    confidence=1.0,
                )
            )

    tracklets = []
    for track_id, frames in tracks.items():
        frames.sort(key=lambda f: f.frame_id)
        tracklets.append(
            Tracklet(
                track_id=track_id,
                camera_id=camera_id,
                class_id=2,  # AIC track 2 = vehicles, default to car
                class_name="car",
                frames=frames,
            )
        )

    logger.info(f"Loaded {len(tracklets)} AIC tracklets from {gt_path}")
    return tracklets


def tracklets_to_mot_format(
    tracklets: List[Tracklet],
) -> List[Tuple[int, int, float, float, float, float, float, int]]:
    """Convert Tracklets to MOTChallenge submission format.

    Returns list of tuples: (frame, id, x, y, w, h, conf, class)
    where x/y/w/h are in top-left xywh format.
    """
    rows = []
    for t in tracklets:
        for f in t.frames:
            x1, y1, x2, y2 = f.bbox
            w = x2 - x1
            h = y2 - y1
            rows.append((f.frame_id, t.track_id, x1, y1, w, h, f.confidence, t.class_id))

    rows.sort(key=lambda r: (r[0], r[1]))
    return rows


def save_mot_format(
    rows: List[Tuple], output_path: str | Path
) -> None:
    """Write MOTChallenge-format rows to a text file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for row in rows:
            f.write(",".join(str(v) for v in row) + "\n")
