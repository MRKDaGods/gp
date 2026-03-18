"""Multi-modal similarity computation for cross-camera association.

Supports class-adaptive weighting (person vs vehicle), mutual
nearest-neighbor filtering, and temporal overlap bonus for
overlapping-FOV camera pairs.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from loguru import logger

from src.core.constants import PERSON_CLASSES
from src.stage4_association.spatial_temporal import SpatioTemporalValidator


def compute_hsv_similarity(hist_a: np.ndarray, hist_b: np.ndarray) -> float:
    """Compute similarity between two L2-normalized HSV histograms.

    Uses dot product (equivalent to cosine similarity for L2-normed vectors).
    """
    return float(np.dot(hist_a, hist_b))


def compute_temporal_overlap_ratio(
    start_i: float,
    end_i: float,
    start_j: float,
    end_j: float,
) -> float:
    """Compute temporal IoU between two tracklets.

    Returns the fraction of temporal overlap relative to the shorter tracklet's
    duration.  This is more informative than IoU for asymmetric durations —
    a short tracklet fully contained in a long one should get score ≈ 1.0.

    Returns:
        Ratio in [0, 1].  0 means no temporal overlap.
    """
    overlap_start = max(start_i, start_j)
    overlap_end = min(end_i, end_j)
    overlap = max(0.0, overlap_end - overlap_start)
    if overlap <= 0:
        return 0.0
    min_duration = min(end_i - start_i, end_j - start_j)
    if min_duration <= 0:
        return 0.0
    return min(overlap / min_duration, 1.0)


def mutual_nearest_neighbor_filter(
    candidate_pairs: List[Tuple[int, int, float]],
    top_k_per_query: int = 10,
) -> List[Tuple[int, int, float]]:
    """Filter candidate pairs to mutual nearest neighbors.

    A pair (i, j) is kept only if j is in i's top-K *and* i is in j's top-K.
    This dramatically reduces false-positive edges in the similarity graph.

    Args:
        candidate_pairs: List of (i, j, similarity) tuples.
        top_k_per_query: How many top matches per query to consider.

    Returns:
        Filtered list of mutual-NN pairs.
    """
    # Build top-K sets for each node
    from collections import defaultdict
    neighbors: Dict[int, List[Tuple[int, float]]] = defaultdict(list)

    for i, j, sim in candidate_pairs:
        neighbors[i].append((j, sim))
        neighbors[j].append((i, sim))

    # Keep only top-K per node
    top_k_sets: Dict[int, Set[int]] = {}
    for node, nbrs in neighbors.items():
        nbrs.sort(key=lambda x: x[1], reverse=True)
        top_k_sets[node] = {n for n, _ in nbrs[:top_k_per_query]}

    # Filter: keep only mutual pairs
    result = []
    for i, j, sim in candidate_pairs:
        if j in top_k_sets.get(i, set()) and i in top_k_sets.get(j, set()):
            result.append((i, j, sim))

    logger.debug(
        f"Mutual NN filter: {len(candidate_pairs)} -> {len(result)} pairs "
        f"(top_k={top_k_per_query})"
    )
    return result


def compute_combined_similarity(
    appearance_sim: Dict[Tuple[int, int], float],
    hsv_features: np.ndarray,
    start_times: List[float],
    end_times: List[float],
    camera_ids: List[str],
    class_ids: List[int],
    st_validator: SpatioTemporalValidator,
    weights: dict,
    num_frames: Optional[List[int]] = None,
    temporal_overlap_cfg: Optional[dict] = None,
) -> Dict[Tuple[int, int], float]:
    """Compute weighted combined similarity with class-adaptive weights.

    Uses different weight profiles for persons vs vehicles:
    - **Persons**: Higher appearance weight, lower HSV (clothing vs. lighting).
    - **Vehicles**: Higher HSV weight (more stable colour across cameras).

    Length weighting (when *num_frames* is provided): edges between longer
    tracklets receive a confidence boost.  The geometric mean of the two
    tracklet lengths, normalised by the max length, produces a factor in
    (0, 1] that is mixed 50/50 with 1.0 so short-tracklet pairs are only
    mildly penalised (factor range [0.5, 1.0]).  This is the same approach
    used in SOTA MTMC systems (cf. reference Stage 4 pipeline).

    Args:
        appearance_sim: Dict[(i, j)] -> appearance similarity score.
        hsv_features: (N, bins) HSV histogram matrix.
        start_times: Start timestamp for each tracklet.
        end_times: End timestamp for each tracklet.
        camera_ids: Camera ID for each tracklet.
        class_ids: Class ID for each tracklet.
        st_validator: Spatio-temporal validator instance.
        weights: Dict with keys 'appearance', 'hsv', 'spatiotemporal',
                 and optionally 'person' / 'vehicle' sub-dicts.
        num_frames: Optional frame count per tracklet for length weighting.

    Returns:
        Dict[(i, j)] -> combined similarity score.
    """
    # Default weights
    default_w = {
        "appearance": weights.get("appearance", 0.6),
        "hsv": weights.get("hsv", 0.15),
        "spatiotemporal": weights.get("spatiotemporal", 0.25),
    }
    # Class-specific overrides
    person_w = weights.get("person", {
        "appearance": 0.65, "hsv": 0.10, "spatiotemporal": 0.25,
    })
    vehicle_w = weights.get("vehicle", {
        "appearance": 0.50, "hsv": 0.25, "spatiotemporal": 0.25,
    })

    # Validate weight sums (off-by-one from rounding ok, >5% drift is a bug)
    for label, w_dict in [("default", default_w), ("person", person_w), ("vehicle", vehicle_w)]:
        w_sum = sum(w_dict.get(k, 0) for k in ("appearance", "hsv", "spatiotemporal"))
        if abs(w_sum - 1.0) > 0.05:
            logger.warning(
                f"{label} weights sum to {w_sum:.3f} (expected ~1.0): {w_dict}"
            )

    # Length weighting config
    length_power = float(weights.get("length_weight_power", 0.0))
    use_length_weight = length_power > 0 and num_frames is not None

    # Temporal overlap bonus for overlapping-FOV camera pairs.
    # When two tracklets co-exist in time across cameras with overlapping FOV
    # (mean transition time < threshold), this is a strong positive signal.
    to_cfg = temporal_overlap_cfg or {}
    to_enabled = to_cfg.get("enabled", False)
    to_bonus = float(to_cfg.get("bonus", 0.05))
    to_max_mean_time = float(to_cfg.get("max_mean_time", 5.0))
    to_count = 0

    combined: Dict[Tuple[int, int], float] = {}

    for (i, j), app_sim in appearance_sim.items():
        # Spatio-temporal validation.
        # Use minimum temporal gap: max(0, later_start - earlier_end).
        # This correctly handles both overlapping cameras (gap=0) and
        # sequential cameras, regardless of FAISS-index ordering of (i, j).
        later_start = max(start_times[i], start_times[j])
        earlier_end = min(end_times[i], end_times[j])
        min_gap = max(0.0, later_start - earlier_end)
        # Select camera order so the "earlier" camera is cam_a
        if start_times[i] <= start_times[j]:
            cam_a, cam_b = camera_ids[i], camera_ids[j]
        else:
            cam_a, cam_b = camera_ids[j], camera_ids[i]
        st_score = st_validator.transition_score(
            cam_a=cam_a,
            cam_b=cam_b,
            time_a=0.0,
            time_b=min_gap,
        )

        if st_score <= 0:
            continue  # Invalid transition (e.g. cross-scene), skip

        # HSV similarity
        hsv_sim = compute_hsv_similarity(hsv_features[i], hsv_features[j])

        # Select class-adaptive weights
        if class_ids[i] in PERSON_CLASSES:
            w = person_w
        else:
            w = vehicle_w

        w_app = w.get("appearance", default_w["appearance"])
        w_hsv = w.get("hsv", default_w["hsv"])
        w_st = w.get("spatiotemporal", default_w["spatiotemporal"])

        # Combined score
        score = w_app * app_sim + w_hsv * hsv_sim + w_st * st_score

        # Temporal overlap bonus: for overlapping-FOV cameras, temporal
        # co-existence is a strong positive signal — the same vehicle is
        # visible in both cameras simultaneously.
        if to_enabled:
            t_overlap = compute_temporal_overlap_ratio(
                start_times[i], end_times[i],
                start_times[j], end_times[j],
            )
            if t_overlap > 0:
                # Check if this camera pair has overlapping FOV
                pair_prior = st_validator._get_pair_prior(cam_a, cam_b)
                if pair_prior is not None:
                    mean_t = pair_prior.get("mean_time", float("inf"))
                    if mean_t <= to_max_mean_time:
                        score += to_bonus * t_overlap
                        to_count += 1

        # Length weighting: shorter tracklets have less reliable embeddings
        # Uses the minimum length (weakest link) with hyperbolic saturation.
        # Unlike the ratio min/max, this doesn't over-penalize asymmetric pairs
        # (e.g. S02 c008 short tracklet matching a longer c006 tracklet).
        if use_length_weight:
            li = max(float(num_frames[i]), 1.0)
            lj = max(float(num_frames[j]), 1.0)
            min_len = min(li, lj)
            confidence = min_len / (min_len + 10.0)  # saturates: 5f→0.33, 10f→0.5, 20f→0.67, 50f→0.83
            length_w = math.pow(confidence, length_power)
            score *= 0.5 + 0.5 * length_w  # penalty range [0.5, 1.0]

        combined[(i, j)] = score

    if to_enabled and to_count > 0:
        logger.info(
            f"Temporal overlap bonus applied to {to_count} pairs "
            f"(bonus={to_bonus}, max_mean_time={to_max_mean_time}s)"
        )

    return combined
