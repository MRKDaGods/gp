"""Multi-modal similarity computation for cross-camera association."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from src.stage4_association.spatial_temporal import SpatioTemporalValidator


def compute_hsv_similarity(hist_a: np.ndarray, hist_b: np.ndarray) -> float:
    """Compute similarity between two L2-normalized HSV histograms.

    Uses dot product (equivalent to cosine similarity for L2-normed vectors).
    """
    return float(np.dot(hist_a, hist_b))


def compute_combined_similarity(
    appearance_sim: Dict[Tuple[int, int], float],
    hsv_features: np.ndarray,
    start_times: List[float],
    end_times: List[float],
    camera_ids: List[str],
    st_validator: SpatioTemporalValidator,
    weights: dict,
) -> Dict[Tuple[int, int], float]:
    """Compute weighted combined similarity for all candidate pairs.

    Combines appearance similarity, HSV color similarity, and
    spatio-temporal compatibility into a single score.

    Args:
        appearance_sim: Dict[(i, j)] -> appearance similarity score.
        hsv_features: (N, bins) HSV histogram matrix.
        start_times: Start timestamp for each tracklet.
        end_times: End timestamp for each tracklet.
        camera_ids: Camera ID for each tracklet.
        st_validator: Spatio-temporal validator instance.
        weights: Dict with keys 'appearance', 'hsv', 'spatiotemporal'.

    Returns:
        Dict[(i, j)] -> combined similarity score.
    """
    w_app = weights.get("appearance", 0.7)
    w_hsv = weights.get("hsv", 0.1)
    w_st = weights.get("spatiotemporal", 0.2)

    combined = {}

    for (i, j), app_sim in appearance_sim.items():
        # Spatio-temporal validation
        st_score = st_validator.transition_score(
            cam_a=camera_ids[i],
            cam_b=camera_ids[j],
            time_a=end_times[i],
            time_b=start_times[j],
        )

        if st_score <= 0:
            continue  # Invalid transition, skip

        # HSV similarity
        hsv_sim = compute_hsv_similarity(hsv_features[i], hsv_features[j])

        # Combined score
        score = w_app * app_sim + w_hsv * hsv_sim + w_st * st_score

        combined[(i, j)] = score

    return combined
