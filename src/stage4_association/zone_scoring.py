"""Zone-based transition scoring for cross-camera association.

Assigns entry/exit zones to tracklets based on their bounding box positions,
then scores candidate pairs based on whether their zone transition matches
learned ground-truth transition patterns. This is a key technique from
AIC21/22 top solutions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger


class ZoneScorer:
    """Score tracklet pairs based on zone transition validity."""

    def __init__(self, zone_data_path: str | Path, min_count: int = 2):
        with open(zone_data_path) as f:
            data = json.load(f)

        self._cameras = {}
        for cam_id, cam_info in data["cameras"].items():
            self._cameras[cam_id] = {
                "entry_centers": np.array(cam_info["entry_zones"]["centers"]),
                "exit_centers": np.array(cam_info["exit_zones"]["centers"]),
            }

        self._transitions: Dict[str, int] = data.get("transitions", {})
        self._min_count = min_count

        # Pre-compute valid transitions set for fast lookup
        self._valid = set()
        for key, count in self._transitions.items():
            if count >= self._min_count:
                self._valid.add(key)

        logger.info(
            f"Zone scorer loaded: {len(self._cameras)} cameras, "
            f"{len(self._valid)} valid transitions (min_count={min_count})"
        )

    def assign_zones(
        self,
        entry_positions: List[Optional[Tuple[float, float]]],
        exit_positions: List[Optional[Tuple[float, float]]],
        camera_ids: List[str],
    ) -> Tuple[List[int], List[int]]:
        """Assign entry and exit zone IDs to each tracklet.

        Returns:
            (entry_zones, exit_zones) — zone IDs per tracklet (-1 if unknown).
        """
        n = len(camera_ids)
        entry_zones = [-1] * n
        exit_zones = [-1] * n

        for i in range(n):
            cam = camera_ids[i]
            if cam not in self._cameras:
                continue

            cam_data = self._cameras[cam]

            if entry_positions[i] is not None:
                pos = np.array([[entry_positions[i][0], entry_positions[i][1]]])
                dists = np.linalg.norm(cam_data["entry_centers"] - pos, axis=1)
                entry_zones[i] = int(np.argmin(dists))

            if exit_positions[i] is not None:
                pos = np.array([[exit_positions[i][0], exit_positions[i][1]]])
                dists = np.linalg.norm(cam_data["exit_centers"] - pos, axis=1)
                exit_zones[i] = int(np.argmin(dists))

        assigned = sum(1 for z in entry_zones if z >= 0)
        logger.info(f"Zone assignment: {assigned}/{n} tracklets assigned zones")
        return entry_zones, exit_zones

    def is_valid_transition(
        self,
        cam_a: str, exit_zone_a: int,
        cam_b: str, entry_zone_b: int,
    ) -> bool:
        """Check if a specific zone transition is valid."""
        key = f"{cam_a}|exit|{exit_zone_a}|{cam_b}|entry|{entry_zone_b}"
        return key in self._valid

    def transition_score(
        self,
        cam_i: str, entry_zone_i: int, exit_zone_i: int,
        cam_j: str, entry_zone_j: int, exit_zone_j: int,
    ) -> float:
        """Compute zone transition score for a pair of tracklets.

        Checks both directions (i exits → j enters, j exits → i enters)
        and returns the max score. Returns:
          1.0 if a valid transition exists
          0.0 if no zone info available (either zone == -1)
         -1.0 if zones are assigned but transition is invalid
        """
        if exit_zone_i < 0 or entry_zone_j < 0 or exit_zone_j < 0 or entry_zone_i < 0:
            return 0.0  # Unknown — don't penalise

        # Check both directions
        fwd = self.is_valid_transition(cam_i, exit_zone_i, cam_j, entry_zone_j)
        rev = self.is_valid_transition(cam_j, exit_zone_j, cam_i, entry_zone_i)

        if fwd or rev:
            return 1.0
        return -1.0  # Both assigned but neither direction valid

    def apply_to_similarities(
        self,
        combined_sim: Dict[Tuple[int, int], float],
        camera_ids: List[str],
        entry_zones: List[int],
        exit_zones: List[int],
        bonus: float = 0.03,
        penalty: float = 0.03,
    ) -> Dict[Tuple[int, int], float]:
        """Apply zone scoring to combined similarity dict.

        Adds bonus for valid transitions, subtracts penalty for invalid ones.
        """
        adjusted = {}
        n_bonus = 0
        n_penalty = 0
        n_neutral = 0

        for (i, j), sim in combined_sim.items():
            score = self.transition_score(
                camera_ids[i], entry_zones[i], exit_zones[i],
                camera_ids[j], entry_zones[j], exit_zones[j],
            )
            if score > 0:
                adjusted[(i, j)] = sim + bonus
                n_bonus += 1
            elif score < 0:
                adjusted[(i, j)] = sim - penalty
                n_penalty += 1
            else:
                adjusted[(i, j)] = sim
                n_neutral += 1

        logger.info(
            f"Zone scoring: {n_bonus} bonus (+{bonus}), "
            f"{n_penalty} penalty (-{penalty}), {n_neutral} neutral"
        )
        return adjusted
