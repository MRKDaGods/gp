"""SAM2-based foreground masking for Stage 2 vehicle crops."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)


def _iter_model_name_candidates(model_name: str) -> list[str]:
    candidates = [model_name]

    if model_name.startswith("facebook/sam2.1-"):
        candidates.append(model_name.replace("facebook/sam2.1-", "facebook/sam2-", 1))
    elif model_name.startswith("facebook/sam2-"):
        candidates.append(model_name.replace("facebook/sam2-", "facebook/sam2.1-", 1))

    deduped: list[str] = []
    for candidate in candidates:
        if candidate not in deduped:
            deduped.append(candidate)
    return deduped


class ForegroundMasker:
    """Masks background in vehicle crops using SAM2 with a center-point prompt."""

    def __init__(
        self,
        model_name: str = "facebook/sam2.1-hiera-tiny",
        min_crop_size: int = 48,
        fill_value: str | Sequence[int] = "mean",
        device: str = "cuda:0",
    ):
        self.min_crop_size = int(min_crop_size)
        self.fill_value_mode = fill_value
        self.device = device
        self.predictor = self._load_model(model_name)

    def _load_model(self, model_name: str):
        try:
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as exc:
            logger.error("sam2 package not installed. Install with: pip install sam-2")
            raise ImportError("sam2 package not installed") from exc

        last_error: Exception | None = None
        for candidate in _iter_model_name_candidates(model_name):
            try:
                predictor = SAM2ImagePredictor.from_pretrained(candidate, device=self.device)
                logger.info("SAM2 loaded: %s", candidate)
                return predictor
            except Exception as exc:  # pragma: no cover - depends on external package/runtime
                last_error = exc

        raise RuntimeError(f"Unable to load SAM2 model '{model_name}'") from last_error

    def _resolve_fill_value(self, crop_bgr: np.ndarray) -> np.ndarray:
        if isinstance(self.fill_value_mode, str):
            if self.fill_value_mode == "mean":
                return np.rint(crop_bgr.mean(axis=(0, 1))).astype(np.uint8)
            if self.fill_value_mode == "zero":
                return np.zeros(3, dtype=np.uint8)
            raise ValueError(
                "fill_value must be 'mean', 'zero', or an RGB sequence of length 3"
            )

        if isinstance(self.fill_value_mode, Sequence) and len(self.fill_value_mode) == 3:
            return np.asarray(list(self.fill_value_mode), dtype=np.uint8)

        raise ValueError("fill_value must be 'mean', 'zero', or an RGB sequence of length 3")

    def mask_crop(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Apply foreground masking to a single BGR crop."""
        if crop_bgr is None or crop_bgr.ndim != 3 or crop_bgr.shape[2] != 3:
            return crop_bgr

        height, width = crop_bgr.shape[:2]
        if min(height, width) < self.min_crop_size:
            return crop_bgr

        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        center_point = np.array([[width // 2, height // 2]], dtype=np.float32)
        center_label = np.array([1], dtype=np.int32)

        with torch.inference_mode():
            self.predictor.set_image(crop_rgb)
            masks, scores, _ = self.predictor.predict(
                point_coords=center_point,
                point_labels=center_label,
                multimask_output=True,
            )

        if masks is None or len(masks) == 0:
            return crop_bgr

        best_idx = int(np.argmax(np.asarray(scores))) if scores is not None else 0
        binary_mask = np.asarray(masks[best_idx], dtype=bool)
        if binary_mask.shape != (height, width):
            binary_mask = cv2.resize(
                binary_mask.astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        if not np.any(binary_mask):
            return crop_bgr

        masked = crop_bgr.copy()
        masked[~binary_mask] = self._resolve_fill_value(crop_bgr)
        return masked

    def mask_crops(self, crops: list[Any]) -> list[Any]:
        """Apply foreground masking to quality-scored crops in place."""
        for crop in crops:
            crop.image = self.mask_crop(crop.image)
        return crops