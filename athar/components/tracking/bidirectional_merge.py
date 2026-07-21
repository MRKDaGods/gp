"""ReID-gated merge for bidirectional Stage-1 tracklets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import logging

logger = logging.getLogger(__name__)

from athar.core.data_models import Tracklet, TrackletFeatures, TrackletFrame
from athar.components.tracking.bidirectional import BWD_ID_OFFSET
from athar.components.tracking.tracklet_builder import _compute_iou


TrackletKey = Tuple[str, int]


@dataclass(frozen=True)
class _Candidate:
    fwd_key: TrackletKey
    bwd_key: TrackletKey
    iou: float
    cosine: float

    @property
    def score(self) -> float:
        return self.iou + self.cosine


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return cfg.get(key, default)


def _is_backward(track_id: int) -> bool:
    return track_id >= BWD_ID_OFFSET


def _l2_normalize_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    return (vector / max(norm, 1e-8)).astype(np.float32)


def _l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        return _l2_normalize_vector(matrix)
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return (matrix / np.maximum(norms, 1e-8)).astype(np.float32)


def _frame_map(tracklet: Tracklet) -> Dict[int, TrackletFrame]:
    return {frame.frame_id: frame for frame in tracklet.frames}


def _mean_shared_iou(
    fwd: Tracklet,
    bwd: Tracklet,
    min_shared_frames: int,
) -> Optional[float]:
    fwd_frames = _frame_map(fwd)
    bwd_frames = _frame_map(bwd)
    shared_frame_ids = sorted(set(fwd_frames) & set(bwd_frames))
    if len(shared_frame_ids) < min_shared_frames:
        return None

    ious = [
        _compute_iou(fwd_frames[frame_id].bbox, bwd_frames[frame_id].bbox)
        for frame_id in shared_frame_ids
    ]
    return float(np.mean(ious))


def _merge_tracklet_frames(fwd: Tracklet, bwd: Tracklet) -> None:
    merged = _frame_map(fwd)
    for frame in bwd.frames:
        existing = merged.get(frame.frame_id)
        if existing is None or frame.confidence > existing.confidence:
            merged[frame.frame_id] = frame
    fwd.frames = [merged[frame_id] for frame_id in sorted(merged)]


def _pool_feature_embeddings(fwd: TrackletFeatures, bwd: TrackletFeatures) -> None:
    fwd.embedding = _l2_normalize_vector(0.5 * (fwd.embedding + bwd.embedding))
    fwd.hsv_histogram = (0.5 * (fwd.hsv_histogram + bwd.hsv_histogram)).astype(np.float32)

    if fwd.multi_query_embeddings is None:
        fwd.multi_query_embeddings = bwd.multi_query_embeddings
        return
    if bwd.multi_query_embeddings is None:
        return
    if fwd.multi_query_embeddings.shape != bwd.multi_query_embeddings.shape:
        return
    fwd.multi_query_embeddings = _l2_normalize_rows(
        0.5 * (fwd.multi_query_embeddings + bwd.multi_query_embeddings)
    )


def _pool_aligned_rows(
    aligned_matrices: Dict[str, Optional[np.ndarray]],
    fwd_idx: int,
    bwd_idx: int,
) -> None:
    for name, matrix in aligned_matrices.items():
        if matrix is None:
            continue
        if name == "hsv":
            matrix[fwd_idx] = (0.5 * (matrix[fwd_idx] + matrix[bwd_idx])).astype(np.float32)
        else:
            matrix[fwd_idx] = _l2_normalize_rows(0.5 * (matrix[fwd_idx] + matrix[bwd_idx]))


def _build_candidates(
    camera_id: str,
    fwd_tracklets: List[Tracklet],
    bwd_tracklets: List[Tracklet],
    features: List[TrackletFeatures],
    feature_index: Dict[TrackletKey, int],
    min_shared_frames: int,
    iou_thresh: float,
    cos_thresh: float,
) -> List[_Candidate]:
    candidates: List[_Candidate] = []
    for fwd in fwd_tracklets:
        fwd_key = (camera_id, fwd.track_id)
        fwd_idx = feature_index.get(fwd_key)
        if fwd_idx is None:
            continue

        for bwd in bwd_tracklets:
            if fwd.class_id != bwd.class_id:
                continue
            bwd_key = (camera_id, bwd.track_id)
            bwd_idx = feature_index.get(bwd_key)
            if bwd_idx is None:
                continue

            mean_iou = _mean_shared_iou(fwd, bwd, min_shared_frames)
            if mean_iou is None or mean_iou < iou_thresh:
                continue

            cosine = float(np.dot(features[fwd_idx].embedding, features[bwd_idx].embedding))
            if cosine < cos_thresh:
                continue

            candidates.append(_Candidate(fwd_key=fwd_key, bwd_key=bwd_key, iou=mean_iou, cosine=cosine))

    candidates.sort(key=lambda candidate: (candidate.score, candidate.cosine, candidate.iou), reverse=True)
    return candidates


def _select_one_to_one(candidates: List[_Candidate]) -> List[_Candidate]:
    selected: List[_Candidate] = []
    used_fwd: set[TrackletKey] = set()
    used_bwd: set[TrackletKey] = set()

    for candidate in candidates:
        if candidate.fwd_key in used_fwd or candidate.bwd_key in used_bwd:
            continue
        selected.append(candidate)
        used_fwd.add(candidate.fwd_key)
        used_bwd.add(candidate.bwd_key)

    return selected


def _validate_aligned_matrices(
    features: List[TrackletFeatures],
    aligned_matrices: Dict[str, Optional[np.ndarray]],
) -> None:
    expected_rows = len(features)
    for name, matrix in aligned_matrices.items():
        if matrix is None:
            continue
        if matrix.shape[0] != expected_rows:
            raise ValueError(
                f"Aligned matrix '{name}' has {matrix.shape[0]} rows; "
                f"expected {expected_rows} rows to match features"
            )


def merge_bidirectional(
    features: List[TrackletFeatures],
    tracklets_by_camera: Dict[str, List[Tracklet]],
    aligned_matrices: Dict[str, Optional[np.ndarray]],
    index_map: List[dict],
    cfg: dict,
) -> tuple[List[TrackletFeatures], Dict[str, List[Tracklet]], Dict[str, Optional[np.ndarray]], List[dict]]:
    """Collapse forward/backward tracklets while preserving row alignment."""
    if not bool(_cfg_get(cfg, "enabled", False)):
        return features, tracklets_by_camera, aligned_matrices, index_map

    _validate_aligned_matrices(features, aligned_matrices)

    min_shared_frames = int(_cfg_get(cfg, "min_shared_frames", 3))
    iou_thresh = float(_cfg_get(cfg, "iou_thresh", 0.5))
    cos_thresh = float(_cfg_get(cfg, "cos_thresh", 0.5))
    keep_unmatched_backward = bool(_cfg_get(cfg, "keep_unmatched_backward", True))

    feature_index: Dict[TrackletKey, int] = {
        (feature.camera_id, feature.track_id): idx
        for idx, feature in enumerate(features)
    }
    tracklet_lookup: Dict[TrackletKey, Tracklet] = {
        (camera_id, tracklet.track_id): tracklet
        for camera_id, tracklets in tracklets_by_camera.items()
        for tracklet in tracklets
    }

    drop_indices: set[int] = set()
    selected_by_camera: Dict[str, List[_Candidate]] = {}

    for camera_id, camera_tracklets in tracklets_by_camera.items():
        fwd_tracklets = [tracklet for tracklet in camera_tracklets if not _is_backward(tracklet.track_id)]
        bwd_tracklets = [tracklet for tracklet in camera_tracklets if _is_backward(tracklet.track_id)]
        if not bwd_tracklets:
            continue

        candidates = _build_candidates(
            camera_id=camera_id,
            fwd_tracklets=fwd_tracklets,
            bwd_tracklets=bwd_tracklets,
            features=features,
            feature_index=feature_index,
            min_shared_frames=min_shared_frames,
            iou_thresh=iou_thresh,
            cos_thresh=cos_thresh,
        )
        selected = _select_one_to_one(candidates)
        selected_by_camera[camera_id] = selected

        for match in selected:
            fwd_tracklet = tracklet_lookup[match.fwd_key]
            bwd_tracklet = tracklet_lookup[match.bwd_key]
            fwd_idx = feature_index[match.fwd_key]
            bwd_idx = feature_index[match.bwd_key]

            _merge_tracklet_frames(fwd_tracklet, bwd_tracklet)
            _pool_feature_embeddings(features[fwd_idx], features[bwd_idx])
            _pool_aligned_rows(aligned_matrices, fwd_idx, bwd_idx)
            drop_indices.add(bwd_idx)

    for camera_id, camera_tracklets in tracklets_by_camera.items():
        selected = selected_by_camera.get(camera_id, [])
        matched_bwd_keys = {match.bwd_key for match in selected}
        used_ids = {tracklet.track_id for tracklet in camera_tracklets if not _is_backward(tracklet.track_id)}
        next_id = max(used_ids, default=0) + 1
        rewritten_tracklets: List[Tracklet] = []

        for tracklet in camera_tracklets:
            key = (camera_id, tracklet.track_id)
            if key in matched_bwd_keys:
                continue

            if _is_backward(tracklet.track_id):
                feature_idx = feature_index.get(key)
                if not keep_unmatched_backward:
                    if feature_idx is not None:
                        drop_indices.add(feature_idx)
                    continue

                while next_id in used_ids:
                    next_id += 1
                new_track_id = next_id
                next_id += 1
                used_ids.add(new_track_id)

                old_track_id = tracklet.track_id
                tracklet.track_id = new_track_id
                if feature_idx is not None:
                    features[feature_idx].track_id = new_track_id
                    index_map[feature_idx]["track_id"] = new_track_id
                logger.debug(
                    f"BT kept unmatched backward tracklet {camera_id}:{old_track_id} "
                    f"as {new_track_id}"
                )

            rewritten_tracklets.append(tracklet)

        camera_tracklets[:] = rewritten_tracklets

    if not drop_indices:
        return features, tracklets_by_camera, aligned_matrices, index_map

    keep_indices = [idx for idx in range(len(features)) if idx not in drop_indices]
    merged_features = [features[idx] for idx in keep_indices]
    merged_index_map = [index_map[idx] for idx in keep_indices]
    merged_aligned = {
        name: None if matrix is None else matrix[keep_indices].astype(np.float32, copy=False)
        for name, matrix in aligned_matrices.items()
    }

    logger.info(
        f"BT merged {len(drop_indices)} backward feature rows; "
        f"features {len(features)} -> {len(merged_features)}"
    )
    return merged_features, tracklets_by_camera, merged_aligned, merged_index_map