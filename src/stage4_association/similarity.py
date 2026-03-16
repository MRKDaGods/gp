"""Multi-modal similarity computation for cross-camera association.

Supports class-adaptive weighting (person vs vehicle) and mutual
nearest-neighbor filtering.
"""

from __future__ import annotations

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
    import math

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

    # Length weighting config
    length_power = float(weights.get("length_weight_power", 0.0))
    use_length_weight = length_power > 0 and num_frames is not None

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

        # Length weighting: longer tracklets = more reliable embeddings
        if use_length_weight:
            li = max(float(num_frames[i]), 1.0)
            lj = max(float(num_frames[j]), 1.0)
            geom_mean = math.pow(li * lj, length_power)
            max_len = math.pow(max(li, lj), length_power) + 1e-8
            length_w = geom_mean / max_len  # in (0, 1]
            score *= 0.5 + 0.5 * length_w  # mild penalty range [0.5, 1.0]

        combined[(i, j)] = score

    return combined
