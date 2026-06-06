"""Spatio-temporal validation for cross-camera tracklet matching."""

from __future__ import annotations

import math
import statistics
from typing import Dict, List, Optional, Tuple

from loguru import logger


class SpatioTemporalValidator:
    """Validates and scores temporal plausibility of cross-camera transitions."""

    def __init__(
        self,
        min_time_gap: float = 2.0,
        max_time_gap: float = 300.0,
        camera_transitions: Optional[Dict] = None,
    ):
        """Args:"""
        self.min_time_gap = min_time_gap
        self.max_time_gap = max_time_gap
        self.camera_transitions = camera_transitions or {}

    # Public API

    def is_valid_transition(
        self,
        cam_a: str,
        cam_b: str,
        time_a: float,
        time_b: float,
    ) -> bool:
        """Check if a transition from cam_a to cam_b is temporally plausible."""
        time_diff = time_b - time_a
        abs_diff = abs(time_diff)  # direction-agnostic: pairs are ordered by FAISS index, not time

        pair_prior = self._get_pair_prior(cam_a, cam_b)
        if pair_prior is not None:
            return pair_prior["min_time"] <= abs_diff <= pair_prior["max_time"]

        # If camera_transitions is non-empty and cam_a is a listed source,
        # block any target that is NOT explicitly in the transitions map.
        if self.camera_transitions:
            cam_a_listed = cam_a in self.camera_transitions
            cam_b_listed = cam_b in self.camera_transitions
            if cam_a_listed and cam_b not in self.camera_transitions.get(cam_a, {}):
                return False
            if cam_b_listed and cam_a not in self.camera_transitions.get(cam_b, {}):
                return False

        abs_diff = abs(time_diff)
        return self.min_time_gap <= abs_diff <= self.max_time_gap

    def transition_score(
        self,
        cam_a: str,
        cam_b: str,
        time_a: float,
        time_b: float,
    ) -> float:
        """Compute a transition plausibility score in [0, 1]."""
        if not self.is_valid_transition(cam_a, cam_b, time_a, time_b):
            return 0.0

        time_diff = time_b - time_a
        abs_diff = abs(time_diff)

        # per-pair prior (learned from GT)
        pair_prior = self._get_pair_prior(cam_a, cam_b)
        if pair_prior is not None:
            return self._score_with_prior(abs_diff, pair_prior)

        # global fallback
        if abs_diff < self.min_time_gap or abs_diff > self.max_time_gap:
            return 0.0

        return self._global_score(abs_diff)

    # Internal scoring helpers

    @staticmethod
    def _score_with_prior(abs_diff: float, prior: Dict) -> float:
        """Score using a per-pair learned prior."""
        min_t = prior["min_time"]
        max_t = prior["max_time"]
        mean_t = prior.get("mean_time", (min_t + max_t) / 2)
        std_t = prior.get("std_time", None)

        if abs_diff < min_t or abs_diff > max_t:
            return 0.0

        # Use the wider of learned std vs. range/3 to avoid over-narrow Gaussian
        if std_t is not None and std_t > 0:
            sigma = max(std_t, (max_t - min_t) / 3.0)
        else:
            sigma = max((max_t - min_t) / 3.0, 1.0)

        return math.exp(-0.5 * ((abs_diff - mean_t) / sigma) ** 2)

    def _global_score(self, abs_diff: float) -> float:
        """Score using the global min/max time gap as a prior."""
        if self.min_time_gap == 0:
            # Half-Gaussian peaking at 0, dropping to ~0.01 at max
            sigma = self.max_time_gap / math.sqrt(2 * math.log(100))
            sigma = max(sigma, 1.0)
            return math.exp(-0.5 * (abs_diff / sigma) ** 2)

        # Centre at min_time_gap - short transitions are most likely.
        # Choose sigma so score at max_time_gap ~ 0.01:
        #   exp(-0.5 * ((max-min)/sigma)^2) = 0.01
        #   sigma = (max-min) / sqrt(2*ln(100)) ~ (max-min) / 3.035
        range_t = self.max_time_gap - self.min_time_gap
        sigma = range_t / math.sqrt(2 * math.log(100))
        sigma = max(sigma, 1.0)
        return math.exp(-0.5 * ((abs_diff - self.min_time_gap) / sigma) ** 2)

    # Lookup

    def _get_pair_prior(
        self, cam_a: str, cam_b: str
    ) -> Optional[Dict]:
        """Look up per-camera-pair transition prior."""
        if cam_a in self.camera_transitions:
            if cam_b in self.camera_transitions[cam_a]:
                return self.camera_transitions[cam_a][cam_b]
        if cam_b in self.camera_transitions:
            if cam_a in self.camera_transitions[cam_b]:
                return self.camera_transitions[cam_b][cam_a]
        return None

    # Learning from ground truth

    def learn_transitions(
        self,
        camera_pairs: List[Tuple[str, str, float]],
    ) -> Dict:
        """Learn transition priors from labeled data."""
        from collections import defaultdict
        pair_times: Dict[Tuple[str, str], list] = defaultdict(list)

        for cam_a, cam_b, time_diff in camera_pairs:
            key = tuple(sorted([cam_a, cam_b]))
            pair_times[key].append(abs(time_diff))

        transitions: Dict[str, Dict] = {}
        for (cam_a, cam_b), times in pair_times.items():
            if not times:
                continue
            mean_t = statistics.mean(times)
            std_t = statistics.stdev(times) if len(times) >= 2 else (max(times) - min(times)) / 4.0
            transitions.setdefault(cam_a, {})[cam_b] = {
                "min_time": min(times) * 0.8,   # 20 % margin
                "max_time": max(times) * 1.2,
                "mean_time": mean_t,
                "std_time": max(std_t, 1.0),
            }

        self.camera_transitions = transitions
        logger.info(f"Learned transition priors for {len(transitions)} camera pairs")
        return transitions
