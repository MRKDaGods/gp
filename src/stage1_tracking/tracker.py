"""BoxMOT multi-object tracker wrapper."""

from __future__ import annotations

from typing import Optional

import numpy as np
from loguru import logger
from omegaconf import DictConfig

from src.core.data_models import Detection


class TrackerWrapper:
    """Wraps BoxMOT trackers (BoT-SORT, Deep-OCSORT, etc.) with a unified interface.

    Each instance tracks objects for a single camera. Create a new instance per camera.
    """

    SUPPORTED_ALGORITHMS = {
        "botsort", "deepocsort", "strongsort", "bytetrack", "ocsort", "hybridsort",
    }

    def __init__(
        self,
        algorithm: str = "botsort",
        reid_weights: Optional[str] = None,
        device: str = "cuda:0",
        half: bool = True,
        tracker_config: Optional[DictConfig] = None,
    ):
        if algorithm not in self.SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"Unknown tracker: {algorithm}. Supported: {self.SUPPORTED_ALGORITHMS}"
            )

        self.algorithm = algorithm
        self.device = device

        # Import and create the tracker
        tracker_cls = self._get_tracker_class(algorithm)

        kwargs = {"device": device, "half": half}
        if reid_weights:
            kwargs["reid_weights"] = reid_weights

        # Pass BoxMOT-specific config params
        if tracker_config:
            for key in ["track_high_thresh", "track_low_thresh", "new_track_thresh",
                        "track_buffer", "match_thresh", "proximity_thresh",
                        "appearance_thresh", "fuse_first_associate",
                        "max_age", "max_obs", "min_hits", "iou_threshold"]:
                if key in tracker_config:
                    kwargs[key] = tracker_config[key]

        self.tracker = tracker_cls(**kwargs)
        logger.info(f"Tracker initialized: {algorithm}, device={device}")

    @staticmethod
    def _get_tracker_class(algorithm: str):
        """Dynamically import the tracker class from boxmot."""
        from boxmot import BotSort, DeepOcSort, StrongSort, ByteTrack, OcSort, HybridSort

        mapping = {
            "botsort": BotSort,
            "deepocsort": DeepOcSort,
            "strongsort": StrongSort,
            "bytetrack": ByteTrack,
            "ocsort": OcSort,
            "hybridsort": HybridSort,
        }
        return mapping[algorithm]

    def update(
        self, detections: list[Detection] | np.ndarray, frame: np.ndarray
    ) -> np.ndarray:
        """Update tracker with new detections for one frame.

        Args:
            detections: Either a list of Detection objects or a (N, 6) numpy array
                       [x1, y1, x2, y2, confidence, class_id].
            frame: BGR uint8 numpy array (H, W, 3).

        Returns:
            np.ndarray of shape (M, 8): [x1, y1, x2, y2, track_id, confidence, class_id, det_idx]
            Returns empty array (0, 8) if no active tracks.
        """
        if isinstance(detections, list):
            if not detections:
                dets = np.empty((0, 6), dtype=np.float32)
            else:
                dets = np.array(
                    [[d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3], d.confidence, d.class_id]
                     for d in detections],
                    dtype=np.float32,
                )
        else:
            dets = detections

        tracks = self.tracker.update(dets, frame)

        if tracks is None or len(tracks) == 0:
            return np.empty((0, 8), dtype=np.float32)

        return np.array(tracks, dtype=np.float32)

    def reset(self) -> None:
        """Reset tracker state (for processing a new video)."""
        if hasattr(self.tracker, "reset"):
            self.tracker.reset()
