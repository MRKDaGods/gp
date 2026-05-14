"""Serialization utilities for inter-stage data exchange.

Each stage writes its outputs to disk so stages can be run independently.
Tracklets and trajectories use JSON; embeddings use numpy .npy files.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.core.data_models import (
    EvaluationResult,
    FrameInfo,
    GlobalTrajectory,
    Tracklet,
    TrackletFeatures,
    TrackletFrame,
)


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# FrameInfo I/O
# ---------------------------------------------------------------------------

def save_frame_manifest(frames: List[FrameInfo], path: str | Path) -> None:
    path = Path(path)
    _ensure_dir(path)
    data = [asdict(f) for f in frames]
    path.write_text(json.dumps(data, indent=2, cls=_NumpyEncoder))


def load_frame_manifest(path: str | Path) -> List[FrameInfo]:
    data = json.loads(Path(path).read_text())
    return [FrameInfo(**d) for d in data]


# ---------------------------------------------------------------------------
# Tracklet I/O
# ---------------------------------------------------------------------------

def _tracklet_to_dict(t: Tracklet) -> dict:
    return {
        "track_id": t.track_id,
        "camera_id": t.camera_id,
        "class_id": t.class_id,
        "class_name": t.class_name,
        "frames": [
            {
                "frame_id": f.frame_id,
                "timestamp": f.timestamp,
                "bbox": list(f.bbox),
                "confidence": f.confidence,
            }
            for f in t.frames
        ],
    }


def _dict_to_tracklet(d: dict) -> Tracklet:
    frames = [
        TrackletFrame(
            frame_id=f["frame_id"],
            timestamp=f["timestamp"],
            bbox=tuple(f["bbox"]),
            confidence=f["confidence"],
        )
        for f in d["frames"]
    ]
    return Tracklet(
        track_id=d["track_id"],
        camera_id=d["camera_id"],
        class_id=d["class_id"],
        class_name=d["class_name"],
        frames=frames,
    )


def save_tracklets(tracklets: List[Tracklet], path: str | Path) -> None:
    """Save tracklets to a JSON file."""
    path = Path(path)
    _ensure_dir(path)
    data = [_tracklet_to_dict(t) for t in tracklets]
    path.write_text(json.dumps(data, indent=2, cls=_NumpyEncoder))


def load_tracklets(path: str | Path) -> List[Tracklet]:
    """Load tracklets from a JSON file."""
    data = json.loads(Path(path).read_text())
    return [_dict_to_tracklet(d) for d in data]


def save_tracklets_by_camera(
    tracklets_by_camera: Dict[str, List[Tracklet]], output_dir: str | Path
) -> None:
    """Save tracklets grouped by camera to separate JSON files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for camera_id, tracklets in tracklets_by_camera.items():
        save_tracklets(tracklets, output_dir / f"tracklets_{camera_id}.json")


def load_tracklets_by_camera(input_dir: str | Path) -> Dict[str, List[Tracklet]]:
    """Load all tracklet files from a directory."""
    input_dir = Path(input_dir)
    result = {}
    for path in sorted(input_dir.glob("tracklets_*.json")):
        camera_id = path.stem.replace("tracklets_", "")
        result[camera_id] = load_tracklets(path)
    return result


# ---------------------------------------------------------------------------
# Embeddings I/O
# ---------------------------------------------------------------------------

def save_embeddings(
    embeddings: np.ndarray,
    index_map: List[Dict[str, Any]],
    output_dir: str | Path,
) -> None:
    """Save embedding matrix and index mapping.

    Args:
        embeddings: (N, D) float32 array of embeddings.
        index_map: List of dicts with {track_id, camera_id, class_id} for each row.
        output_dir: Directory to write files to.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "embeddings.npy", embeddings)
    (output_dir / "embedding_index.json").write_text(
        json.dumps(index_map, indent=2, cls=_NumpyEncoder)
    )


def load_embeddings(input_dir: str | Path) -> tuple[np.ndarray, List[Dict[str, Any]]]:
    """Load embedding matrix and index mapping."""
    input_dir = Path(input_dir)
    embeddings = np.load(input_dir / "embeddings.npy")
    index_map = json.loads((input_dir / "embedding_index.json").read_text())
    return embeddings, index_map


def save_hsv_features(hsv: np.ndarray, output_dir: str | Path) -> None:
    """Save HSV histogram matrix."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "hsv_features.npy", hsv)


def load_hsv_features(input_dir: str | Path) -> np.ndarray:
    """Load HSV histogram matrix."""
    return np.load(Path(input_dir) / "hsv_features.npy")


def save_multi_query_embeddings(
    mq_embeddings: List[np.ndarray],
    output_dir: str | Path,
) -> None:
    """Save dense multi-query embeddings as a compressed NPZ artifact.

    Args:
        mq_embeddings: List of (K, D) arrays, one per tracklet.
        output_dir: Directory to write the artifact to.
    """
    if not mq_embeddings:
        return

    first_shape = mq_embeddings[0].shape
    if len(first_shape) != 2:
        raise ValueError("Multi-query embeddings must have shape (K, D)")

    for idx, mq in enumerate(mq_embeddings):
        if mq.shape != first_shape:
            raise ValueError(
                "Inconsistent multi-query embedding shape at index "
                f"{idx}: expected {first_shape}, got {mq.shape}"
            )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stacked = np.stack(mq_embeddings, axis=0).astype(np.float32)
    np.savez_compressed(output_dir / "multi_query_embeddings.npz", embeddings=stacked)


def load_multi_query_embeddings(
    input_dir: str | Path,
    n: int,
) -> List[Optional[np.ndarray]]:
    """Load dense multi-query embeddings.

    Returns a list of length ``n``. Missing files yield ``[None] * n``.
    """
    path = Path(input_dir) / "multi_query_embeddings.npz"
    if not path.exists():
        return [None] * n

    with np.load(path) as data:
        if "embeddings" not in data:
            return [None] * n
        embeddings = data["embeddings"].astype(np.float32)

    if embeddings.ndim != 3:
        raise ValueError(
            f"Expected multi-query embeddings with shape (N, K, D), got {embeddings.shape}"
        )

    result: List[Optional[np.ndarray]] = [None] * n
    count = min(n, embeddings.shape[0])
    for idx in range(count):
        result[idx] = embeddings[idx]
    return result


# ---------------------------------------------------------------------------
# GlobalTrajectory I/O
# ---------------------------------------------------------------------------

def save_global_trajectories(
    trajectories: List[GlobalTrajectory], path: str | Path
) -> None:
    """Save global trajectories to JSON, including forensic metadata."""
    path = Path(path)
    _ensure_dir(path)
    data = []
    for gt in trajectories:
        entry = {
            "global_id": gt.global_id,
            "tracklets": [_tracklet_to_dict(t) for t in gt.tracklets],
            # Forensic fields (new — zero/empty for legacy trajectories)
            "confidence": gt.confidence,
            "evidence": gt.evidence,
            "timeline": gt.timeline,
        }
        data.append(entry)
    path.write_text(json.dumps(data, indent=2, cls=_NumpyEncoder))


def load_global_trajectories(path: str | Path) -> List[GlobalTrajectory]:
    """Load global trajectories from JSON (backwards-compatible)."""
    data = json.loads(Path(path).read_text())
    trajectories = []
    for d in data:
        tracklets = [_dict_to_tracklet(t) for t in d["tracklets"]]
        trajectories.append(
            GlobalTrajectory(
                global_id=d["global_id"],
                tracklets=tracklets,
                # Forensic fields — default to 0/empty for legacy files
                confidence=float(d.get("confidence", 0.0)),
                evidence=d.get("evidence", []),
                timeline=d.get("timeline", []),
            )
        )
    return trajectories


# ---------------------------------------------------------------------------
# EvaluationResult I/O
# ---------------------------------------------------------------------------

def save_evaluation_result(result: EvaluationResult, path: str | Path) -> None:
    path = Path(path)
    _ensure_dir(path)
    path.write_text(json.dumps(asdict(result), indent=2, cls=_NumpyEncoder))


def load_evaluation_result(path: str | Path) -> EvaluationResult:
    data = json.loads(Path(path).read_text())
    return EvaluationResult(**data)
