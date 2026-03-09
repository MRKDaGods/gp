"""Camera-aware distance bias for cross-camera association.

Learns per-camera-pair distance offsets from training data or initial
matching results. This is a key technique from AI City Challenge winners
(Liu et al., CVPRW 2021) that calibrates cross-camera distances to
account for systematic appearance shifts between cameras.

Also provides zone-based spatio-temporal transition model for CityFlow-style
datasets where cameras observe non-overlapping areas.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import json
import numpy as np
from loguru import logger


class CameraDistanceBias:
    """Learn and apply per-camera-pair distance bias.

    For each camera pair (ci, cj), computes a bias term that represents
    the typical distance between same-identity tracklets across those cameras.
    This allows the association to compensate for systematic appearance
    differences (lighting, viewpoint, resolution) between cameras.
    """

    def __init__(self):
        self._bias: Dict[Tuple[str, str], float] = {}
        self._pair_distances: Dict[Tuple[str, str], List[float]] = defaultdict(list)

    def learn_from_matches(
        self,
        similarities: Dict[Tuple[int, int], float],
        camera_ids: List[str],
        clusters: List[Set[int]],
    ):
        """Learn bias from established identity clusters.

        For each cluster (assumed to be correct identity groupings),
        collect cross-camera similarity scores and compute the median
        as the bias for that camera pair.

        Args:
            similarities: Pairwise similarity scores.
            camera_ids: Camera ID for each tracklet index.
            clusters: Identity clusters from initial association.
        """
        self._pair_distances.clear()

        for cluster in clusters:
            if len(cluster) < 2:
                continue
            members = list(cluster)
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    a, b = members[i], members[j]
                    ca, cb = camera_ids[a], camera_ids[b]
                    if ca == cb:
                        continue
                    pair = tuple(sorted([ca, cb]))
                    sim = similarities.get((a, b), similarities.get((b, a), None))
                    if sim is not None:
                        self._pair_distances[pair].append(sim)

        # Compute bias as the median similarity for each pair
        for pair, sims in self._pair_distances.items():
            if len(sims) >= 3:  # Need minimum samples
                self._bias[pair] = float(np.median(sims))

        logger.info(
            f"Camera distance bias learned for {len(self._bias)} camera pairs "
            f"(from {len(clusters)} clusters)"
        )

    def get_bias(self, cam_a: str, cam_b: str) -> float:
        """Get bias (median similarity) for a camera pair.

        Returns 0.0 if no bias is learned for this pair.
        """
        pair = tuple(sorted([cam_a, cam_b]))
        return self._bias.get(pair, 0.0)

    def adjust_similarity(
        self,
        similarity: float,
        cam_a: str,
        cam_b: str,
        global_mean: float = 0.5,
    ) -> float:
        """Adjust a similarity score using camera-pair bias.

        Normalizes the similarity relative to the expected baseline for
        this camera pair, then re-centers around the global mean.

        Args:
            similarity: Raw similarity score.
            cam_a: First camera ID.
            cam_b: Second camera ID.
            global_mean: Target mean for normalized scores.

        Returns:
            Adjusted similarity score.
        """
        bias = self.get_bias(cam_a, cam_b)
        if bias > 0:
            # Shift similarity so that the pair's median maps to global_mean
            adjusted = similarity - bias + global_mean
            return max(0.0, min(1.0, adjusted))
        return similarity

    def adjust_similarity_matrix(
        self,
        similarities: Dict[Tuple[int, int], float],
        camera_ids: List[str],
    ) -> Dict[Tuple[int, int], float]:
        """Adjust all pairwise similarities using camera bias.

        Args:
            similarities: Original similarity dict.
            camera_ids: Camera ID for each tracklet index.

        Returns:
            New similarity dict with bias-adjusted scores.
        """
        if not self._bias:
            return similarities

        # Compute global mean
        all_sims = list(similarities.values())
        global_mean = float(np.median(all_sims)) if all_sims else 0.5

        adjusted = {}
        for (i, j), sim in similarities.items():
            adjusted[(i, j)] = self.adjust_similarity(
                sim, camera_ids[i], camera_ids[j], global_mean
            )
        return adjusted

    def save(self, path: str | Path):
        """Save learned biases to JSON."""
        data = {
            "bias": {f"{k[0]}|{k[1]}": v for k, v in self._bias.items()},
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Camera distance bias saved to {path}")

    def load(self, path: str | Path):
        """Load biases from JSON."""
        with open(path) as f:
            data = json.load(f)
        self._bias = {
            tuple(k.split("|")): v for k, v in data["bias"].items()
        }
        logger.info(f"Camera distance bias loaded: {len(self._bias)} pairs")


class ZoneTransitionModel:
    """Zone-based spatio-temporal transition model for MTMC tracking.

    Key technique from AIC21 1st place solution:
    - Each camera has entry/exit zones (regions where vehicles appear/disappear)
    - Transition rules specify which zone pairs are valid across cameras
    - Each valid transition has an expected time range

    For overlapping cameras (WILDTRACK), zones are less critical since
    multiple cameras see the same location simultaneously.
    """

    def __init__(self):
        self._zones: Dict[str, Dict[str, dict]] = {}  # cam -> zone_id -> zone_info
        self._transitions: Dict[Tuple[str, str, str, str], dict] = {}
        # (src_cam, src_zone, dst_cam, dst_zone) -> {min_time, max_time, weight}

    def add_zone(
        self,
        camera_id: str,
        zone_id: str,
        bbox: Tuple[float, float, float, float],  # (x1, y1, x2, y2) normalized
        zone_type: str = "both",  # 'entry', 'exit', or 'both'
    ):
        """Register a zone for a camera.

        Args:
            camera_id: Camera identifier.
            zone_id: Zone identifier (e.g., 'zone_A', 'zone_B').
            bbox: Normalized bounding box of the zone in the image.
            zone_type: Whether this zone is entry, exit, or both.
        """
        if camera_id not in self._zones:
            self._zones[camera_id] = {}
        self._zones[camera_id][zone_id] = {
            "bbox": bbox,
            "type": zone_type,
        }

    def add_transition(
        self,
        src_cam: str,
        src_zone: str,
        dst_cam: str,
        dst_zone: str,
        min_time: float = 0.0,
        max_time: float = float("inf"),
        weight: float = 1.0,
    ):
        """Register a valid transition between zones.

        Args:
            src_cam: Source camera.
            src_zone: Source zone (exit zone).
            dst_cam: Destination camera.
            dst_zone: Destination zone (entry zone).
            min_time, max_time: Valid time range for this transition (seconds).
            weight: Confidence/weight for this transition route.
        """
        key = (src_cam, src_zone, dst_cam, dst_zone)
        self._transitions[key] = {
            "min_time": min_time,
            "max_time": max_time,
            "weight": weight,
        }

    def classify_zone(
        self,
        camera_id: str,
        x: float,
        y: float,
    ) -> Optional[str]:
        """Determine which zone a point belongs to.

        Args:
            camera_id: Camera identifier.
            x, y: Normalized coordinates in the image.

        Returns:
            Zone ID or None if not in any zone.
        """
        if camera_id not in self._zones:
            return None
        for zone_id, info in self._zones[camera_id].items():
            x1, y1, x2, y2 = info["bbox"]
            if x1 <= x <= x2 and y1 <= y <= y2:
                return zone_id
        return None

    def get_transition_score(
        self,
        src_cam: str,
        src_zone: Optional[str],
        dst_cam: str,
        dst_zone: Optional[str],
        time_gap: float,
    ) -> float:
        """Score a transition between two camera-zone pairs.

        Returns a weight in [0, 1] indicating how plausible this transition is.
        Returns 0 if the transition is invalid.
        """
        if src_zone is None or dst_zone is None:
            # If zones unknown, use a default permissive score
            return 0.5

        key = (src_cam, src_zone, dst_cam, dst_zone)
        if key not in self._transitions:
            return 0.0  # Unknown transition → not valid

        trans = self._transitions[key]
        if trans["min_time"] <= time_gap <= trans["max_time"]:
            return trans["weight"]
        return 0.0

    def load_from_config(self, config: dict):
        """Load zone and transition definitions from config dict.

        Expected format:
            zones:
              C001:
                zone_A: {bbox: [0.0, 0.7, 0.3, 1.0], type: exit}
                zone_B: {bbox: [0.7, 0.0, 1.0, 0.5], type: entry}
              C002: ...
            transitions:
              - {src_cam: C001, src_zone: zone_A, dst_cam: C002, dst_zone: zone_B,
                 min_time: 5, max_time: 60}
        """
        for cam_id, zones in config.get("zones", {}).items():
            for zone_id, zone_info in zones.items():
                self.add_zone(
                    cam_id, zone_id,
                    bbox=tuple(zone_info["bbox"]),
                    zone_type=zone_info.get("type", "both"),
                )

        for trans in config.get("transitions", []):
            self.add_transition(
                src_cam=trans["src_cam"],
                src_zone=trans["src_zone"],
                dst_cam=trans["dst_cam"],
                dst_zone=trans["dst_zone"],
                min_time=trans.get("min_time", 0),
                max_time=trans.get("max_time", float("inf")),
                weight=trans.get("weight", 1.0),
            )

        logger.info(
            f"Zone model loaded: {sum(len(z) for z in self._zones.values())} zones, "
            f"{len(self._transitions)} transitions"
        )

    def save(self, path: str | Path):
        """Save zone model to JSON."""
        data = {
            "zones": {},
            "transitions": [],
        }
        for cam_id, zones in self._zones.items():
            data["zones"][cam_id] = {
                zid: {"bbox": list(info["bbox"]), "type": info["type"]}
                for zid, info in zones.items()
            }
        for (sc, sz, dc, dz), info in self._transitions.items():
            data["transitions"].append({
                "src_cam": sc, "src_zone": sz,
                "dst_cam": dc, "dst_zone": dz,
                "min_time": info["min_time"],
                "max_time": info["max_time"],
                "weight": info["weight"],
            })
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str | Path):
        """Load zone model from JSON."""
        with open(path) as f:
            config = json.load(f)
        self.load_from_config(config)
