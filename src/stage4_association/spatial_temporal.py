"""Spatio-temporal validation for cross-camera tracklet matching.

Uses transition time priors between camera pairs to gate impossible matches
and score plausible ones.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

from loguru import logger


class SpatioTemporalValidator:
    """Validates and scores temporal plausibility of cross-camera transitions.

    If camera transition priors are provided, uses them for strict gating.
    Otherwise, uses configurable min/max time gaps as a global prior.
    """

    def __init__(
        self,
        min_time_gap: float = 2.0,
        max_time_gap: float = 300.0,
        camera_transitions: Optional[Dict] = None,
    ):
        """
        Args:
            min_time_gap: Minimum seconds between tracklet end and next start.
            max_time_gap: Maximum seconds between tracklet end and next start.
            camera_transitions: Optional per-pair priors, e.g.:
                {"cam01": {"cam02": {"min_time": 5, "max_time": 60, "mean_time": 30}}}
        """
        self.min_time_gap = min_time_gap
        self.max_time_gap = max_time_gap
        self.camera_transitions = camera_transitions or {}

    def is_valid_transition(
        self,
        cam_a: str,
        cam_b: str,
        time_a: float,
        time_b: float,
    ) -> bool:
        """Check if a transition from cam_a to cam_b is temporally plausible.

        Args:
            cam_a: Source camera ID.
            cam_b: Destination camera ID.
            time_a: End time of tracklet in cam_a.
            time_b: Start time of tracklet in cam_b.

        Returns:
            True if the transition is plausible.
        """
        time_diff = time_b - time_a

        # Check per-pair priors first
        pair_prior = self._get_pair_prior(cam_a, cam_b)
        if pair_prior is not None:
            return pair_prior["min_time"] <= time_diff <= pair_prior["max_time"]

        # Fall back to global priors (allow both directions)
        abs_diff = abs(time_diff)
        return self.min_time_gap <= abs_diff <= self.max_time_gap

    def transition_score(
        self,
        cam_a: str,
        cam_b: str,
        time_a: float,
        time_b: float,
    ) -> float:
        """Compute a transition plausibility score in [0, 1].

        Uses a Gaussian centered on the expected transition time.
        Returns 0 for invalid transitions.

        Args:
            cam_a: Source camera.
            cam_b: Destination camera.
            time_a: End time in source camera.
            time_b: Start time in destination camera.

        Returns:
            Score in [0, 1], where 1 = perfect temporal match.
        """
        time_diff = time_b - time_a
        abs_diff = abs(time_diff)

        pair_prior = self._get_pair_prior(cam_a, cam_b)

        if pair_prior is not None:
            min_t = pair_prior["min_time"]
            max_t = pair_prior["max_time"]
            mean_t = pair_prior.get("mean_time", (min_t + max_t) / 2)

            if abs_diff < min_t or abs_diff > max_t:
                return 0.0

            sigma = (max_t - min_t) / 4  # ~95% within range
            sigma = max(sigma, 1.0)  # prevent zero sigma
            return math.exp(-0.5 * ((abs_diff - mean_t) / sigma) ** 2)

        # Global prior: validity check + simple score
        if abs_diff < self.min_time_gap or abs_diff > self.max_time_gap:
            return 0.0

        # Gaussian centered at half the max gap
        mean_t = self.max_time_gap / 2
        sigma = self.max_time_gap / 4
        return math.exp(-0.5 * ((abs_diff - mean_t) / sigma) ** 2)

    def _get_pair_prior(
        self, cam_a: str, cam_b: str
    ) -> Optional[Dict]:
        """Look up per-camera-pair transition prior."""
        if cam_a in self.camera_transitions:
            if cam_b in self.camera_transitions[cam_a]:
                return self.camera_transitions[cam_a][cam_b]
        # Try reverse direction
        if cam_b in self.camera_transitions:
            if cam_a in self.camera_transitions[cam_b]:
                return self.camera_transitions[cam_b][cam_a]
        return None

    def learn_transitions(
        self,
        camera_pairs: list[Tuple[str, str, float]],
    ) -> Dict:
        """Learn transition priors from labeled data.

        Args:
            camera_pairs: List of (cam_a, cam_b, time_diff) from ground truth.

        Returns:
            Camera transition dictionary.
        """
        from collections import defaultdict
        pair_times: Dict[Tuple[str, str], list] = defaultdict(list)

        for cam_a, cam_b, time_diff in camera_pairs:
            key = tuple(sorted([cam_a, cam_b]))
            pair_times[key].append(abs(time_diff))

        transitions = {}
        for (cam_a, cam_b), times in pair_times.items():
            if not times:
                continue
            transitions.setdefault(cam_a, {})[cam_b] = {
                "min_time": min(times) * 0.8,  # 20% margin
                "max_time": max(times) * 1.2,
                "mean_time": sum(times) / len(times),
            }

        self.camera_transitions = transitions
        logger.info(f"Learned transition priors for {len(transitions)} camera pairs")
        return transitions
