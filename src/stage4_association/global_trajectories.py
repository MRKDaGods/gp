"""Merge tracklet clusters into GlobalTrajectory objects."""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

from src.core.data_models import GlobalTrajectory, Tracklet


def merge_tracklets_to_trajectories(
    clusters: List[Set[int]],
    feature_to_tracklet_key: List[Tuple[str, int]],
    tracklet_lookup: Dict[Tuple[str, int], Tracklet],
) -> List[GlobalTrajectory]:
    """Convert identity clusters to GlobalTrajectory objects.

    Args:
        clusters: List of sets, each containing feature indices for one identity.
        feature_to_tracklet_key: Maps feature index -> (camera_id, track_id).
        tracklet_lookup: Maps (camera_id, track_id) -> Tracklet object.

    Returns:
        List of GlobalTrajectory, sorted by global_id.
    """
    trajectories = []

    for global_id, cluster in enumerate(clusters):
        tracklets = []
        for feat_idx in cluster:
            if feat_idx >= len(feature_to_tracklet_key):
                continue
            key = feature_to_tracklet_key[feat_idx]
            tracklet = tracklet_lookup.get(key)
            if tracklet is not None:
                tracklets.append(tracklet)

        if not tracklets:
            continue

        # Sort tracklets within trajectory by start time
        tracklets.sort(key=lambda t: t.start_time)

        trajectories.append(
            GlobalTrajectory(global_id=global_id, tracklets=tracklets)
        )

    # Sort trajectories by global_id
    trajectories.sort(key=lambda t: t.global_id)

    return trajectories
