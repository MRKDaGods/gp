"""Merge tracklet clusters into GlobalTrajectory objects."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from src.core.data_models import GlobalTrajectory, Tracklet


def merge_tracklets_to_trajectories(
    clusters: List[Set[int]],
    feature_to_tracklet_key: List[Tuple[str, int]],
    tracklet_lookup: Dict[Tuple[str, int], Tracklet],
    embeddings: Optional[np.ndarray] = None,
    combined_sim: Optional[Dict[Tuple[int, int], float]] = None,
) -> List[GlobalTrajectory]:
    """Convert identity clusters to GlobalTrajectory objects with forensic metadata.

    Args:
        clusters: List of sets, each containing feature indices for one identity.
        feature_to_tracklet_key: Maps feature index -> (camera_id, track_id).
        tracklet_lookup: Maps (camera_id, track_id) -> Tracklet object.
        embeddings: Optional (N, D) L2-normed embedding matrix for confidence scoring.
        combined_sim: Optional pre-computed pairwise similarity dict for evidence records.

    Returns:
        List of GlobalTrajectory sorted by global_id, each with confidence +
        evidence audit trail populated.
    """
    trajectories = []

    for global_id, cluster in enumerate(clusters):
        member_list = list(cluster)
        tracklets = []
        for feat_idx in member_list:
            if feat_idx >= len(feature_to_tracklet_key):
                continue
            key = feature_to_tracklet_key[feat_idx]
            tracklet = tracklet_lookup.get(key)
            if tracklet is not None:
                tracklets.append((feat_idx, tracklet))

        if not tracklets:
            continue

        # Sort tracklets within trajectory by start time
        tracklets.sort(key=lambda x: x[1].start_time)
        feat_indices = [x[0] for x in tracklets]
        ordered_tracklets = [x[1] for x in tracklets]

        # ── Confidence: mean pairwise cosine similarity ──────────────────
        confidence = _compute_cluster_confidence(feat_indices, embeddings, combined_sim)

        # ── Evidence: pairwise similarity records for audit trail ─────────
        evidence = _build_evidence_records(
            feat_indices, feature_to_tracklet_key, combined_sim,
        )

        # ── Timeline: ordered camera appearances ─────────────────────────
        timeline = [
            {
                "camera_id": t.camera_id,
                "track_id": t.track_id,
                "start": round(t.start_time, 3),
                "end": round(t.end_time, 3),
                "duration_s": round(t.duration, 3),
                "num_frames": t.num_frames,
                "mean_confidence": round(t.mean_confidence, 3),
            }
            for t in ordered_tracklets
        ]

        trajectories.append(
            GlobalTrajectory(
                global_id=global_id,
                tracklets=ordered_tracklets,
                confidence=confidence,
                evidence=evidence,
                timeline=timeline,
            )
        )

    trajectories.sort(key=lambda t: t.global_id)
    return trajectories


def _compute_cluster_confidence(
    feat_indices: List[int],
    embeddings: Optional[np.ndarray],
    combined_sim: Optional[Dict[Tuple[int, int], float]],
) -> float:
    """Mean pairwise cosine similarity within a cluster, in [0, 1]."""
    n = len(feat_indices)
    if n <= 1:
        return 1.0 if n == 1 else 0.0

    sims = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = feat_indices[i], feat_indices[j]
            if combined_sim is not None:
                sim = combined_sim.get((a, b), combined_sim.get((b, a), None))
                if sim is not None:
                    sims.append(float(sim))
                    continue
            # Fallback: compute from raw embeddings
            if embeddings is not None and a < len(embeddings) and b < len(embeddings):
                sims.append(float(np.dot(embeddings[a], embeddings[b])))

    return float(max(0.0, np.mean(sims))) if sims else 0.0


def _build_evidence_records(
    feat_indices: List[int],
    feature_to_tracklet_key: List[Tuple[str, int]],
    combined_sim: Optional[Dict[Tuple[int, int], float]],
) -> List[Dict[str, Any]]:
    """Build pairwise evidence records for the forensic audit trail."""
    records = []
    n = len(feat_indices)
    if n <= 1 or combined_sim is None:
        return records

    for i in range(n):
        for j in range(i + 1, n):
            a, b = feat_indices[i], feat_indices[j]
            sim = combined_sim.get((a, b), combined_sim.get((b, a), None))
            if sim is None:
                continue
            key_a = feature_to_tracklet_key[a] if a < len(feature_to_tracklet_key) else ("?", -1)
            key_b = feature_to_tracklet_key[b] if b < len(feature_to_tracklet_key) else ("?", -1)
            records.append({
                "tracklet_a": f"{key_a[0]}:{key_a[1]}",
                "tracklet_b": f"{key_b[0]}:{key_b[1]}",
                "similarity": round(float(sim), 4),
            })

    # Sort strongest evidence first
    records.sort(key=lambda r: r["similarity"], reverse=True)
    return records
