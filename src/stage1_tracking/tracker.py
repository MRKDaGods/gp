"""BoxMOT multi-object tracker wrapper."""

from __future__ import annotations

import inspect
from pathlib import Path
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
            kwargs["reid_weights"] = Path(reid_weights)

        # Pass BoxMOT-specific config params
        if tracker_config:
            for key in ["track_high_thresh", "track_low_thresh", "new_track_thresh",
                        "track_buffer", "match_thresh", "proximity_thresh",
                        "appearance_thresh", "fuse_first_associate",
                        "max_age", "max_obs", "min_hits", "iou_threshold",
                        "cmc_method", "frame_rate"]:
                if key in tracker_config:
                    kwargs[key] = tracker_config[key]

        sig = inspect.signature(tracker_cls.__init__)
        valid_params = set(sig.parameters.keys()) - {"self"}
        accepts_var_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in sig.parameters.values()
        )
        if not accepts_var_kwargs:
            filtered = {key: value for key, value in kwargs.items() if key in valid_params}
            dropped = set(kwargs.keys()) - set(filtered.keys())
            if dropped:
                logger.warning(f"Dropped unsupported tracker params: {dropped}")
            kwargs = filtered

        self.tracker = tracker_cls(**kwargs)
        logger.info(f"Tracker initialized: {algorithm}, device={device}")

    @staticmethod
    def _get_tracker_class(algorithm: str):
        """Dynamically import the tracker class from boxmot."""
        import boxmot

        candidates = {
            "botsort": ("BotSort", "BoTSORT", "BOTSORT"),
            "deepocsort": ("DeepOcSort", "DeepOCSort", "DEEPOCSORT"),
            "strongsort": ("StrongSort", "StrongSORT", "STRONGSORT"),
            "bytetrack": ("ByteTrack", "BYTETrack", "BYTETRACK"),
            "ocsort": ("OcSort", "OCSort", "OCSORT"),
            "hybridsort": ("HybridSort", "HybridSORT", "HYBRIDSORT"),
        }

        for class_name in candidates[algorithm]:
            tracker_cls = getattr(boxmot, class_name, None)
            if tracker_cls is not None:
                return tracker_cls

        raise ImportError(
            f"boxmot does not expose a supported class for '{algorithm}'. "
            f"Tried: {candidates[algorithm]}"
        )

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
                # Guard against NaN/inf from detector
                if not np.isfinite(dets).all():
                    from loguru import logger
                    bad_count = (~np.isfinite(dets)).any(axis=1).sum()
                    logger.warning(f"Dropping {bad_count} non-finite detections")
                    dets = dets[np.isfinite(dets).all(axis=1)]
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
