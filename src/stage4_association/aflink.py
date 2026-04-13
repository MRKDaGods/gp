"""AFLink motion-based post-association linking for Stage 4 trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from src.core.data_models import GlobalTrajectory, Tracklet
from src.stage4_association.global_trajectories import merge_tracklets_to_trajectories


TrackletKey = Tuple[str, int]


@dataclass(frozen=True)
class _TrajectorySummary:
    """Cached trajectory endpoint data used by AFLink candidate evaluation."""

    index: int
    class_id: int
    camera_ids: frozenset[str]
    start_key: TrackletKey
    end_key: TrackletKey
    start_frame: int
    end_frame: int
    start_center: np.ndarray
    end_center: np.ndarray
    start_velocity: Optional[np.ndarray]
    end_velocity: Optional[np.ndarray]


@dataclass(frozen=True)
class _AFLinkCandidate:
    """A directional trajectory merge proposal that passed all AFLink checks."""

    source_index: int
    target_index: int
    tracklet_a: TrackletKey
    tracklet_b: TrackletKey
    time_gap_frames: int
    spatial_gap_px: float
    direction_cos: float
    velocity_ratio: float
    score: float


def aflink_post_association(
    trajectories: List[GlobalTrajectory],
    feature_to_tracklet_key: List[TrackletKey],
    tracklet_lookup: Dict[TrackletKey, Tracklet],
    embeddings: Optional[np.ndarray] = None,
    combined_sim: Optional[Dict[Tuple[int, int], float]] = None,
    *,
    max_time_gap_frames: int = 150,
    max_spatial_gap_px: float = 200.0,
    min_direction_cos: float = 0.7,
    min_velocity_ratio: float = 0.5,
    velocity_window: int = 5,
) -> List[GlobalTrajectory]:
    """Merge additional cross-camera trajectories using motion consistency.

    Args:
        trajectories: Stage 4 trajectories after graph-based association.
        feature_to_tracklet_key: Feature index to (camera_id, track_id) mapping.
        tracklet_lookup: Lookup for full Tracklet objects.
        embeddings: Optional embedding matrix used to recompute trajectory confidence.
        combined_sim: Optional appearance similarity matrix used for audit evidence.
        max_time_gap_frames: Maximum allowed gap between trajectory endpoints.
        max_spatial_gap_px: Maximum allowed pixel distance between endpoints.
        min_direction_cos: Minimum cosine similarity between endpoint velocity vectors.
        min_velocity_ratio: Minimum ratio between endpoint speeds.
        velocity_window: Number of detections used to estimate endpoint velocity.

    Returns:
        Updated list of GlobalTrajectory objects with AFLink merges applied.
    """
    if len(trajectories) < 2:
        return trajectories

    key_to_feature_idx = {
        key: idx for idx, key in enumerate(feature_to_tracklet_key)
    }
    current = list(trajectories)
    accepted_links: List[Dict[str, object]] = []
    total_merges = 0

    max_passes = max(len(current) - 1, 1)
    for pass_idx in range(max_passes):
        summaries = [
            _summarise_trajectory(idx, trajectory, velocity_window)
            for idx, trajectory in enumerate(current)
        ]
        candidates = _build_candidates(
            summaries=summaries,
            max_time_gap_frames=max_time_gap_frames,
            max_spatial_gap_px=max_spatial_gap_px,
            min_direction_cos=min_direction_cos,
            min_velocity_ratio=min_velocity_ratio,
        )
        if not candidates:
            break

        parent = list(range(len(current)))
        component_cameras = [set(summary.camera_ids) for summary in summaries]
        merged_this_pass = 0

        for candidate in candidates:
            src_root = _find(parent, candidate.source_index)
            dst_root = _find(parent, candidate.target_index)
            if src_root == dst_root:
                continue
            if component_cameras[src_root] & component_cameras[dst_root]:
                continue

            parent[dst_root] = src_root
            component_cameras[src_root].update(component_cameras[dst_root])
            component_cameras[dst_root].clear()
            merged_this_pass += 1
            total_merges += 1
            accepted_links.append(
                {
                    "tracklet_a": f"{candidate.tracklet_a[0]}:{candidate.tracklet_a[1]}",
                    "tracklet_b": f"{candidate.tracklet_b[0]}:{candidate.tracklet_b[1]}",
                    "merge_stage": "aflink",
                    "time_gap_frames": candidate.time_gap_frames,
                    "spatial_gap_px": round(candidate.spatial_gap_px, 3),
                    "direction_cos": round(candidate.direction_cos, 4),
                    "velocity_ratio": round(candidate.velocity_ratio, 4),
                    "motion_score": round(candidate.score, 4),
                }
            )

        if merged_this_pass == 0:
            break

        current = _rebuild_trajectories(
            trajectories=current,
            parent=parent,
            key_to_feature_idx=key_to_feature_idx,
            feature_to_tracklet_key=feature_to_tracklet_key,
            tracklet_lookup=tracklet_lookup,
            embeddings=embeddings,
            combined_sim=combined_sim,
            aflink_records=accepted_links,
        )
        logger.info(
            "AFLink pass {}: {} trajectory merges accepted ({} trajectories remain)",
            pass_idx + 1,
            merged_this_pass,
            len(current),
        )

    if total_merges == 0:
        logger.info("AFLink: no additional motion-consistent merges found")
        return trajectories

    logger.info(
        "AFLink complete: {} additional merges applied across {} trajectories",
        total_merges,
        len(current),
    )
    return current


def _build_candidates(
    summaries: List[_TrajectorySummary],
    *,
    max_time_gap_frames: int,
    max_spatial_gap_px: float,
    min_direction_cos: float,
    min_velocity_ratio: float,
) -> List[_AFLinkCandidate]:
    """Evaluate all valid trajectory pairs and keep only passing AFLink proposals."""
    candidates: List[_AFLinkCandidate] = []
    for i, summary_a in enumerate(summaries):
        for j in range(i + 1, len(summaries)):
            summary_b = summaries[j]
            if summary_a.class_id != summary_b.class_id:
                continue
            if summary_a.camera_ids & summary_b.camera_ids:
                continue

            candidate_ab = _evaluate_directional_pair(
                source=summary_a,
                target=summary_b,
                max_time_gap_frames=max_time_gap_frames,
                max_spatial_gap_px=max_spatial_gap_px,
                min_direction_cos=min_direction_cos,
                min_velocity_ratio=min_velocity_ratio,
            )
            candidate_ba = _evaluate_directional_pair(
                source=summary_b,
                target=summary_a,
                max_time_gap_frames=max_time_gap_frames,
                max_spatial_gap_px=max_spatial_gap_px,
                min_direction_cos=min_direction_cos,
                min_velocity_ratio=min_velocity_ratio,
            )
            chosen = _pick_better_candidate(candidate_ab, candidate_ba)
            if chosen is not None:
                candidates.append(chosen)

    candidates.sort(
        key=lambda candidate: (
            candidate.score,
            candidate.direction_cos,
            candidate.velocity_ratio,
            -candidate.time_gap_frames,
            -candidate.spatial_gap_px,
        ),
        reverse=True,
    )
    return candidates


def _pick_better_candidate(
    first: Optional[_AFLinkCandidate],
    second: Optional[_AFLinkCandidate],
) -> Optional[_AFLinkCandidate]:
    """Choose the stronger of two directional AFLink proposals."""
    if first is None:
        return second
    if second is None:
        return first
    return first if first.score >= second.score else second


def _evaluate_directional_pair(
    source: _TrajectorySummary,
    target: _TrajectorySummary,
    *,
    max_time_gap_frames: int,
    max_spatial_gap_px: float,
    min_direction_cos: float,
    min_velocity_ratio: float,
) -> Optional[_AFLinkCandidate]:
    """Check whether source -> target is motion-consistent enough to merge."""
    if source.end_velocity is None or target.start_velocity is None:
        return None

    time_gap_frames = max(0, target.start_frame - source.end_frame)
    if time_gap_frames > max_time_gap_frames:
        return None

    spatial_gap_px = float(np.linalg.norm(target.start_center - source.end_center))
    if spatial_gap_px > max_spatial_gap_px:
        return None

    source_speed = float(np.linalg.norm(source.end_velocity))
    target_speed = float(np.linalg.norm(target.start_velocity))
    if source_speed <= 1e-6 or target_speed <= 1e-6:
        return None

    direction_cos = float(
        np.dot(source.end_velocity, target.start_velocity) / (source_speed * target_speed)
    )
    if direction_cos < min_direction_cos:
        return None

    velocity_ratio = min(source_speed, target_speed) / max(source_speed, target_speed)
    if velocity_ratio < min_velocity_ratio:
        return None

    score = (
        direction_cos
        + velocity_ratio
        - (time_gap_frames / max(max_time_gap_frames, 1))
        - (spatial_gap_px / max(max_spatial_gap_px, 1e-6))
    )
    return _AFLinkCandidate(
        source_index=source.index,
        target_index=target.index,
        tracklet_a=source.end_key,
        tracklet_b=target.start_key,
        time_gap_frames=time_gap_frames,
        spatial_gap_px=spatial_gap_px,
        direction_cos=direction_cos,
        velocity_ratio=velocity_ratio,
        score=score,
    )


def _rebuild_trajectories(
    trajectories: List[GlobalTrajectory],
    parent: List[int],
    key_to_feature_idx: Dict[TrackletKey, int],
    feature_to_tracklet_key: List[TrackletKey],
    tracklet_lookup: Dict[TrackletKey, Tracklet],
    embeddings: Optional[np.ndarray],
    combined_sim: Optional[Dict[Tuple[int, int], float]],
    aflink_records: List[Dict[str, object]],
) -> List[GlobalTrajectory]:
    """Convert AFLink unions back into GlobalTrajectory objects."""
    clusters: Dict[int, set[int]] = {}
    for traj_idx, trajectory in enumerate(trajectories):
        root = _find(parent, traj_idx)
        cluster = clusters.setdefault(root, set())
        for tracklet in trajectory.tracklets:
            feature_idx = key_to_feature_idx.get((tracklet.camera_id, tracklet.track_id))
            if feature_idx is not None:
                cluster.add(feature_idx)

    ordered_clusters = [clusters[root] for root in sorted(clusters, key=lambda item: min(clusters[item]))]
    rebuilt = merge_tracklets_to_trajectories(
        clusters=ordered_clusters,
        feature_to_tracklet_key=feature_to_tracklet_key,
        tracklet_lookup=tracklet_lookup,
        embeddings=embeddings,
        combined_sim=combined_sim,
    )
    _append_aflink_evidence(rebuilt, aflink_records)
    return rebuilt


def _append_aflink_evidence(
    trajectories: List[GlobalTrajectory],
    aflink_records: List[Dict[str, object]],
) -> None:
    """Attach AFLink audit records to the trajectories they helped merge."""
    if not aflink_records:
        return

    for trajectory in trajectories:
        tracklet_keys = {
            f"{tracklet.camera_id}:{tracklet.track_id}"
            for tracklet in trajectory.tracklets
        }
        matching = [
            record
            for record in aflink_records
            if record["tracklet_a"] in tracklet_keys and record["tracklet_b"] in tracklet_keys
        ]
        if matching:
            trajectory.evidence.extend(matching)


def _summarise_trajectory(
    index: int,
    trajectory: GlobalTrajectory,
    velocity_window: int,
) -> _TrajectorySummary:
    """Extract the endpoint motion summary AFLink uses for one trajectory."""
    ordered_tracklets = sorted(
        trajectory.tracklets,
        key=lambda tracklet: (
            tracklet.frames[0].frame_id if tracklet.frames else 0,
            tracklet.start_time,
        ),
    )
    start_tracklet = ordered_tracklets[0]
    end_tracklet = ordered_tracklets[-1]
    start_frame = start_tracklet.frames[0].frame_id if start_tracklet.frames else 0
    end_frame = end_tracklet.frames[-1].frame_id if end_tracklet.frames else 0
    return _TrajectorySummary(
        index=index,
        class_id=start_tracklet.class_id,
        camera_ids=frozenset(tracklet.camera_id for tracklet in trajectory.tracklets),
        start_key=(start_tracklet.camera_id, start_tracklet.track_id),
        end_key=(end_tracklet.camera_id, end_tracklet.track_id),
        start_frame=start_frame,
        end_frame=end_frame,
        start_center=_bbox_center(start_tracklet.frames[0].bbox),
        end_center=_bbox_center(end_tracklet.frames[-1].bbox),
        start_velocity=_compute_velocity(start_tracklet, velocity_window, from_start=True),
        end_velocity=_compute_velocity(end_tracklet, velocity_window, from_start=False),
    )


def _compute_velocity(
    tracklet: Tracklet,
    velocity_window: int,
    *,
    from_start: bool,
) -> Optional[np.ndarray]:
    """Estimate average endpoint velocity from a short frame window."""
    if len(tracklet.frames) < 2:
        return None

    if from_start:
        window_frames = tracklet.frames[: max(2, velocity_window)]
    else:
        window_frames = tracklet.frames[-max(2, velocity_window):]

    first = window_frames[0]
    last = window_frames[-1]
    frame_delta = last.frame_id - first.frame_id
    if frame_delta <= 0:
        return None

    start_center = _bbox_center(first.bbox)
    end_center = _bbox_center(last.bbox)
    return (end_center - start_center) / float(frame_delta)


def _bbox_center(bbox: Tuple[float, float, float, float]) -> np.ndarray:
    """Compute the centre point of a bounding box."""
    x1, y1, x2, y2 = bbox
    return np.array(((x1 + x2) * 0.5, (y1 + y2) * 0.5), dtype=np.float32)


def _find(parent: List[int], index: int) -> int:
    """Union-find parent lookup with path compression."""
    while parent[index] != index:
        parent[index] = parent[parent[index]]
        index = parent[index]
    return index