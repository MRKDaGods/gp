"""BoxMOT multi-object tracker wrapper."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import logging

logger = logging.getLogger(__name__)

from athar.core.data_models import Detection


def _patch_boxmot_unfreeze_for_numpy_arrays() -> None:
    """Patch DeepOCSORT KF unfreeze to handle vector-shaped history entries."""
    try:
        from boxmot.motion.kalman_filters.aabb.xysr_kf import KalmanFilterXYSR
    except Exception:
        return

    if getattr(KalmanFilterXYSR, "_mtmc_unfreeze_patched", False):
        return

    original_unfreeze = KalmanFilterXYSR.unfreeze

    def _normalize_history(history_obs):
        from collections import deque

        norm = []
        for item in list(history_obs):
            if item is None:
                norm.append(None)
                continue
            arr = np.asarray(item)
            # Flatten (4,1) / (1,4) / nested shapes to (4,)
            norm.append(arr.reshape(-1))
        return deque(norm, maxlen=history_obs.maxlen)

    def patched_unfreeze(self):
        # Normalize both live history and frozen snapshot before delegating.
        if getattr(self, "history_obs", None) is not None:
            self.history_obs = _normalize_history(self.history_obs)
        if getattr(self, "attr_saved", None) is not None and isinstance(self.attr_saved, dict):
            saved_hist = self.attr_saved.get("history_obs")
            if saved_hist is not None:
                self.attr_saved["history_obs"] = _normalize_history(saved_hist)
        return original_unfreeze(self)

    KalmanFilterXYSR.unfreeze = patched_unfreeze
    KalmanFilterXYSR._mtmc_unfreeze_patched = True


class TrackerWrapper:
    """Wraps BoxMOT trackers (BoT-SORT, Deep-OCSORT, etc.) with a unified interface.

    Each instance tracks objects for a single camera. Create a new instance per camera.
    """

    SUPPORTED_ALGORITHMS = {
        "botsort", "deepocsort", "strongsort", "bytetrack", "ocsort", "hybridsort",
        "boosttrack",  # boxmot >= 20 only (D18 production-profile candidate)
    }

    def __init__(
        self,
        algorithm: str = "botsort",
        reid_weights: Optional[str] = None,
        device: str = "cuda:0",
        half: bool = True,
        tracker_config: Optional[Mapping] = None,
    ):
        if algorithm not in self.SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"Unknown tracker: {algorithm}. Supported: {self.SUPPORTED_ALGORITHMS}"
            )

        _patch_boxmot_unfreeze_for_numpy_arrays()

        self.algorithm = algorithm
        self.device = device

        # Import and create the tracker
        tracker_cls = self._get_tracker_class(algorithm)

        kwargs = {"device": device, "half": half}
        # Always provide reid_weights as Path (required in newer boxmot)
        _reid = reid_weights or "models/tracker/osnet_x0_25_msmt17.pt"
        kwargs["reid_weights"] = Path(_reid)

        # Pass BoxMOT-specific config params - filter to only those accepted by this tracker version
        if tracker_config:
            valid_params = set(inspect.signature(tracker_cls.__init__).parameters.keys()) - {"self"}
            for key in ["track_high_thresh", "track_low_thresh", "new_track_thresh",
                        "track_buffer", "match_thresh", "proximity_thresh",
                        "appearance_thresh", "fuse_first_associate",
                        "max_age", "max_obs", "min_hits", "iou_threshold",
                        "cmc_method", "frame_rate",
                        # DeepOCSort-specific params
                        "per_class", "cmc_off", "Q_xy_scaling", "Q_s_scaling",
                        "delta_t", "inertia", "w_association_emb",
                        "alpha_fixed_emb", "aw_param", "embedding_off", "aw_off"]:
                if key in tracker_config and key in valid_params:
                    kwargs[key] = tracker_config[key]

        sig = inspect.signature(tracker_cls.__init__)
        valid_params = set(sig.parameters.keys()) - {"self"}

        # boxmot >= 20 trackers take a pre-built reid backend (`reid_model`)
        # instead of reid_weights/device/half. Build it ONLY from local
        # weights — never let boxmot fall back to downloading (air-gap).
        if "reid_model" in valid_params and "reid_weights" not in valid_params:
            reid_path = Path(_reid)
            if reid_path.is_file():
                try:
                    from boxmot.reid.core.reid import ReID

                    kwargs["reid_model"] = ReID(reid_path, device=device, half=half).model
                except Exception as exc:  # noqa: BLE001 - degrade, don't die
                    logger.warning(
                        "could not build boxmot ReID backend from %s (%s); "
                        "tracking without appearance features", reid_path, exc,
                    )
                    if "with_reid" in valid_params:
                        kwargs["with_reid"] = False
            else:
                logger.warning(
                    "reid weights not found at %s; tracking without appearance features",
                    reid_path,
                )
                if "with_reid" in valid_params:
                    kwargs["with_reid"] = False

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
        """Dynamically import the tracker class from boxmot.

        boxmot >= 20 ships an authoritative registry
        (boxmot.trackers.registry.TRACKER_DEFINITIONS) mapping algorithm name
        to class path — use it when present (validated against boxmot 22).
        Older boxmot (11-12, the v1 parity env) exposes top-level classes with
        unstable casing; fall back to trying known casings per algorithm, and
        tolerate a getattr that raises: those builds lazily import submodules
        on attribute access, so we must NOT scan the whole namespace (that
        triggers EVERY lazy import, including unrelated broken ones) and must
        catch import errors per-candidate rather than crash.
        """
        import importlib

        import boxmot

        try:
            from boxmot.trackers.registry import TRACKER_DEFINITIONS
        except ImportError:
            pass
        else:
            definition = TRACKER_DEFINITIONS.get(algorithm)
            if definition is not None:
                module_name, class_name = definition.class_path.rsplit(".", 1)
                return getattr(importlib.import_module(module_name), class_name)

        candidates = {
            "botsort": ("BotSort", "BoTSORT", "BOTSORT", "BotSORT", "Botsort"),
            "deepocsort": ("DeepOcSort", "DeepOCSORT", "DeepOCSort", "DEEPOCSORT", "DeepOcSORT", "Deepocsort"),
            "strongsort": ("StrongSort", "StrongSORT", "STRONGSORT", "Strongsort"),
            "bytetrack": ("ByteTrack", "BYTETrack", "BYTETRACK", "Bytetrack"),
            "ocsort": ("OcSort", "OCSort", "OCSORT", "Ocsort"),
            "hybridsort": ("HybridSort", "HybridSORT", "HYBRIDSORT", "Hybridsort"),
        }
        for class_name in candidates.get(algorithm, ()):
            try:
                tracker_cls = getattr(boxmot, class_name)
            except Exception:
                # Lazy-import of this attribute failed in this boxmot build - skip it.
                continue
            if isinstance(tracker_cls, type):
                return tracker_cls

        raise ImportError(
            f"boxmot {getattr(boxmot, '__version__', '?')} does not expose a class for "
            f"tracker '{algorithm}'. Validated generations: boxmot 11-12 (top-level "
            "classes, v1 parity env) and boxmot >= 20 (trackers.registry, v2 env)."
        )

    def update(
        self, detections: list[Detection] | np.ndarray, frame: np.ndarray
    ) -> np.ndarray:
        """Update tracker with new detections for one frame."""
        if isinstance(detections, list):
            if not detections:
                dets = np.empty((0, 6), dtype=np.float32)
            else:
                rows: list[list[float]] = []
                dropped = 0
                for d in detections:
                    # Some upstream libraries occasionally return shape (1, 4)
                    # boxes. Flatten aggressively so BoxMOT always gets scalars.
                    bbox = np.asarray(d.bbox, dtype=np.float32).reshape(-1)
                    if bbox.size < 4:
                        dropped += 1
                        continue
                    row = [
                        float(bbox[0]),
                        float(bbox[1]),
                        float(bbox[2]),
                        float(bbox[3]),
                        float(d.confidence),
                        float(d.class_id),
                    ]
                    rows.append(row)
                dets = (
                    np.asarray(rows, dtype=np.float32)
                    if rows
                    else np.empty((0, 6), dtype=np.float32)
                )
                if dropped:
                    logger.warning(f"Dropped {dropped} malformed detections (invalid bbox shape)")
        else:
            dets = np.asarray(detections, dtype=np.float32)
            if dets.ndim == 1:
                dets = dets.reshape(1, -1) if dets.size else np.empty((0, 6), dtype=np.float32)
            elif dets.ndim > 2:
                dets = dets.reshape(dets.shape[0], -1)
            if dets.shape[1] < 6:
                logger.warning(f"Received detection array with invalid shape {dets.shape}; dropping frame")
                dets = np.empty((0, 6), dtype=np.float32)
            else:
                dets = dets[:, :6]

        # Guard against NaN/inf from detector
        if dets.size and not np.isfinite(dets).all():
            bad_count = int((~np.isfinite(dets)).any(axis=1).sum())
            logger.warning(f"Dropping {bad_count} non-finite detections")
            dets = dets[np.isfinite(dets).all(axis=1)]

        tracks = self.tracker.update(dets, frame)

        if tracks is None or len(tracks) == 0:
            return np.empty((0, 8), dtype=np.float32)

        return np.array(tracks, dtype=np.float32)

    def reset(self) -> None:
        """Reset tracker state (for processing a new video)."""
        if hasattr(self.tracker, "reset"):
            self.tracker.reset()
