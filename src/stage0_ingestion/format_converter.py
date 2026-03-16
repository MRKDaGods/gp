"""Format converters for various dataset annotation schemes.

Converts annotations from dataset-specific formats (AI City Challenge,
MOTChallenge, WILDTRACK, etc.) into the unified schema used by subsequent stages.
"""

from __future__ import annotations

import csv
import json
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

    AIC format (per line): camera_id obj_id frame_id x y w h [-1 -1]

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
            track_id = int(parts[1])
            frame_id = int(parts[2])
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


def load_wildtrack_annotations(
    annotations_dir: str | Path,
    fps: float = 2.0,
) -> Dict[str, List[Tracklet]]:
    """Load WILDTRACK JSON annotations and convert to per-camera Tracklets.

    WILDTRACK uses per-frame JSON files in ``annotations_positions/``.
    Each JSON file is a list of entries::

        [
          {
            "personID": 0,
            "positionID": 12345,
            "views": [
              {"viewNum": 0, "xmin": 100, "ymin": 50, "xmax": 200, "ymax": 300},
              ...
            ]
          },
          ...
        ]

    ``viewNum`` maps to camera index (0 -> "C1", 1 -> "C2", ... 6 -> "C7").
    Frame numbers are encoded in filenames (e.g. ``00000005.json`` = frame 5).

    Args:
        annotations_dir: Path to ``annotations_positions/`` folder.
        fps: Annotation frame rate (WILDTRACK annotates at 2 fps).

    Returns:
        Dict mapping camera_id ("C1"..."C7") to list of Tracklets.
    """
    annotations_dir = Path(annotations_dir)
    if not annotations_dir.exists():
        logger.warning(f"WILDTRACK annotations dir not found: {annotations_dir}")
        return {}

    json_files = sorted(annotations_dir.glob("*.json"))
    if not json_files:
        logger.warning(f"No JSON annotation files in {annotations_dir}")
        return {}

    logger.info(f"Loading {len(json_files)} WILDTRACK annotation frames")

    # Accumulate: tracks[camera_id][person_id] -> list of TrackletFrame
    tracks: Dict[str, Dict[int, List[TrackletFrame]]] = {}

    for json_path in json_files:
        # Frame number from filename (e.g. "00000005.json" -> 5)
        frame_id = int(json_path.stem)
        timestamp = frame_id / (60.0 / fps)  # original 60fps, annotated every 1/fps seconds

        with open(json_path, "r") as f:
            annotations = json.load(f)

        for entry in annotations:
            person_id = entry["personID"]

            for view in entry.get("views", []):
                view_num = view["viewNum"]
                xmin = view.get("xmin", -1)
                ymin = view.get("ymin", -1)
                xmax = view.get("xmax", -1)
                ymax = view.get("ymax", -1)

                # Skip entries with no valid bounding box
                if xmin < 0 or ymin < 0 or xmax <= xmin or ymax <= ymin:
                    continue

                camera_id = f"C{view_num + 1}"

                if camera_id not in tracks:
                    tracks[camera_id] = {}
                if person_id not in tracks[camera_id]:
                    tracks[camera_id][person_id] = []

                tracks[camera_id][person_id].append(
                    TrackletFrame(
                        frame_id=frame_id,
                        timestamp=timestamp,
                        bbox=(float(xmin), float(ymin), float(xmax), float(ymax)),
                        confidence=1.0,
                    )
                )

    # Build Tracklet objects per camera
    result: Dict[str, List[Tracklet]] = {}
    for camera_id, person_tracks in sorted(tracks.items()):
        tracklets = []
        for person_id, frames in sorted(person_tracks.items()):
            frames.sort(key=lambda f: f.frame_id)
            tracklets.append(
                Tracklet(
                    track_id=person_id,
                    camera_id=camera_id,
                    class_id=0,
                    class_name="person",
                    frames=frames,
                )
            )
        result[camera_id] = tracklets

    total_tracklets = sum(len(tl) for tl in result.values())
    logger.info(
        f"Loaded WILDTRACK annotations: {len(result)} cameras, "
        f"{total_tracklets} tracklets"
    )
    return result
