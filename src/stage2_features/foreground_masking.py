"""SAM2 foreground masking for vehicle ReID crops.

Removes background clutter (road markings, adjacent vehicles, trees) from crops
before ReID feature extraction. This should improve cross-camera feature consistency
by eliminating camera-specific background signals.
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_IMAGENET_MEAN_RGB = np.array([0.485, 0.456, 0.406], dtype=np.float32)

_SAM2_MODEL_SPECS = {
    "sam2_hiera_tiny": {
        "hf_model_id": "facebook/sam2-hiera-tiny",
        "config_candidates": (
            "sam2_hiera_t.yaml",
            "configs/sam2.1/sam2.1_hiera_t.yaml",
            "configs/sam2/sam2_hiera_t.yaml",
        ),
        "checkpoint_candidates": (
            "sam2_hiera_tiny.pt",
            "sam2_hiera_t.pt",
            "sam2.1_hiera_tiny.pt",
            "sam2.1_hiera_t.pt",
        ),
    },
    "sam2_hiera_small": {
        "hf_model_id": "facebook/sam2-hiera-small",
        "config_candidates": (
            "sam2_hiera_s.yaml",
            "configs/sam2.1/sam2.1_hiera_s.yaml",
            "configs/sam2/sam2_hiera_s.yaml",
        ),
        "checkpoint_candidates": (
            "sam2_hiera_small.pt",
            "sam2_hiera_s.pt",
            "sam2.1_hiera_small.pt",
            "sam2.1_hiera_s.pt",
        ),
    },
    "sam2_hiera_base_plus": {
        "hf_model_id": "facebook/sam2-hiera-base-plus",
        "config_candidates": (
            "sam2_hiera_b+.yaml",
            "configs/sam2.1/sam2.1_hiera_b+.yaml",
            "configs/sam2/sam2_hiera_b+.yaml",
        ),
        "checkpoint_candidates": (
            "sam2_hiera_base_plus.pt",
            "sam2_hiera_b+.pt",
            "sam2.1_hiera_base_plus.pt",
            "sam2.1_hiera_b+.pt",
        ),
    },
    "sam2_hiera_large": {
        "hf_model_id": "facebook/sam2-hiera-large",
        "config_candidates": (
            "sam2_hiera_l.yaml",
            "configs/sam2.1/sam2.1_hiera_l.yaml",
            "configs/sam2/sam2_hiera_l.yaml",
        ),
        "checkpoint_candidates": (
            "sam2_hiera_large.pt",
            "sam2_hiera_l.pt",
            "sam2.1_hiera_large.pt",
            "sam2.1_hiera_l.pt",
        ),
    },
}


class ForegroundMasker:
    """Masks vehicle crop backgrounds using SAM2 automatic segmentation."""

    def __init__(
        self,
        model_type: str = "sam2_hiera_tiny",
        device: str = "cuda:0",
        min_crop_size: int = 64,
        fill_value: Optional[tuple[int, int, int]] = None,
    ):
        """Initialize SAM2 model for foreground masking.

        Args:
            model_type: SAM2 model variant (sam2_hiera_tiny recommended for speed)
            device: CUDA device
            min_crop_size: Skip crops smaller than this (pixels)
            fill_value: RGB fill for background. None uses ImageNet mean.
        """
        if model_type not in _SAM2_MODEL_SPECS:
            supported = ", ".join(sorted(_SAM2_MODEL_SPECS))
            raise ValueError(f"Unsupported SAM2 model_type '{model_type}'. Supported: {supported}")

        self.model_type = model_type
        self.device = device
        self.min_crop_size = int(min_crop_size)
        self.fill_value = fill_value
        self._fill_value_bgr = self._resolve_fill_value_bgr(fill_value)
        self._mask_generator = None
        self._availability_error: Optional[str] = None
        self._warned_unavailable = False

        self._mask_generator = self._build_mask_generator()

    @property
    def is_available(self) -> bool:
        """Return True when SAM2 masking is operational in the current environment."""
        return self._mask_generator is not None

    def mask_crops(
        self,
        crops: list[np.ndarray],
        batch_size: int = 32,
    ) -> list[np.ndarray]:
        """Apply foreground masking to a batch of crops.

        For each crop:
        1. Run SAM2 auto-mask generation
        2. Select the largest mask (assumed to be the vehicle)
        3. Apply mask: keep foreground, fill background with mean color

        Falls back to original crop on errors or tiny crops.

        Returns:
            List of masked crops (same size and format as input)
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be >= 1")

        if not crops:
            return []

        if not self.is_available:
            if not self._warned_unavailable:
                logger.warning(
                    "SAM2 foreground masking unavailable (%s); returning original crops",
                    self._availability_error or "unknown reason",
                )
                self._warned_unavailable = True
            logger.info(
                "Foreground masking skipped: total=%d masked=0 skipped=%d failed=0",
                len(crops),
                len(crops),
            )
            return list(crops)

        masked_crops: list[np.ndarray] = []
        masked_count = 0
        skipped_count = 0
        failed_count = 0

        for start in range(0, len(crops), batch_size):
            batch = crops[start:start + batch_size]
            for crop in batch:
                if crop is None or crop.ndim != 3 or crop.shape[2] != 3:
                    failed_count += 1
                    masked_crops.append(crop)
                    continue

                height, width = crop.shape[:2]
                if min(height, width) < self.min_crop_size:
                    skipped_count += 1
                    masked_crops.append(crop)
                    continue

                try:
                    masks = self._generate_masks(crop)
                    if not masks:
                        failed_count += 1
                        masked_crops.append(crop)
                        continue

                    mask = self._get_largest_mask(masks)
                    if mask.shape != crop.shape[:2]:
                        mask = cv2.resize(
                            mask.astype(np.uint8),
                            (width, height),
                            interpolation=cv2.INTER_NEAREST,
                        ).astype(bool)

                    if not np.any(mask):
                        failed_count += 1
                        masked_crops.append(crop)
                        continue

                    masked_crop = crop.copy()
                    masked_crop[~mask] = self._background_fill(crop.dtype)
                    masked_crops.append(masked_crop)
                    masked_count += 1
                except Exception as exc:
                    failed_count += 1
                    logger.debug("SAM2 masking failed for crop %s: %s", getattr(crop, "shape", None), exc)
                    masked_crops.append(crop)

        logger.info(
            "Foreground masking complete: total=%d masked=%d skipped=%d failed=%d",
            len(crops),
            masked_count,
            skipped_count,
            failed_count,
        )
        return masked_crops

    def _get_largest_mask(self, masks: list) -> np.ndarray:
        """Select the mask with the largest area."""
        best_mask: Optional[np.ndarray] = None
        best_area = -1

        for mask_entry in masks:
            area: Optional[float] = None

            if isinstance(mask_entry, dict):
                raw_mask = mask_entry.get("segmentation")
                area = float(mask_entry.get("area", 0.0))
            else:
                raw_mask = mask_entry

            if raw_mask is None:
                continue

            mask = np.asarray(raw_mask, dtype=bool)
            if mask.ndim > 2:
                mask = np.squeeze(mask)
            if mask.ndim != 2:
                continue

            if area is None or area <= 0:
                area = float(mask.sum())

            if area > best_area:
                best_area = area
                best_mask = mask

        if best_mask is None:
            raise ValueError("SAM2 produced no valid segmentation masks")

        return best_mask

    def _build_mask_generator(self):
        try:
            sam2_mask_module = importlib.import_module("sam2.automatic_mask_generator")
            SAM2AutomaticMaskGenerator = getattr(sam2_mask_module, "SAM2AutomaticMaskGenerator")
        except Exception as exc:
            self._availability_error = f"sam2 import failed: {exc}"
            logger.warning("SAM2 foreground masking disabled: %s", self._availability_error)
            return None

        spec = _SAM2_MODEL_SPECS[self.model_type]
        load_errors: list[str] = []

        try:
            sam2_build_module = importlib.import_module("sam2.build_sam")
            build_sam2_hf = getattr(sam2_build_module, "build_sam2_hf")

            sam_model = build_sam2_hf(spec["hf_model_id"], device=self.device)
            logger.info(
                "Loaded SAM2 foreground masking model via Hugging Face: type=%s device=%s",
                self.model_type,
                self.device,
            )
            return SAM2AutomaticMaskGenerator(sam_model)
        except Exception as exc:
            load_errors.append(f"HF loader failed: {exc}")

        try:
            sam2_build_module = importlib.import_module("sam2.build_sam")
            build_sam2 = getattr(sam2_build_module, "build_sam2")

            config_name = spec["config_candidates"][0]
            checkpoint_path = self._find_checkpoint(spec["checkpoint_candidates"])
            sam_model = build_sam2(
                config_name,
                str(checkpoint_path),
                device=self.device,
                apply_postprocessing=False,
            )
            logger.info(
                "Loaded SAM2 foreground masking model from local checkpoint: type=%s checkpoint=%s",
                self.model_type,
                checkpoint_path,
            )
            return SAM2AutomaticMaskGenerator(sam_model)
        except Exception as exc:
            load_errors.append(f"local loader failed: {exc}")

        self._availability_error = "; ".join(load_errors)
        logger.warning("SAM2 foreground masking disabled: %s", self._availability_error)
        return None

    def _find_checkpoint(self, checkpoint_names: tuple[str, ...]) -> Path:
        search_roots = (
            Path("models"),
            Path("models/sam2"),
            Path("checkpoints"),
            Path("checkpoints/sam2"),
        )

        for root in search_roots:
            for checkpoint_name in checkpoint_names:
                candidate = root / checkpoint_name
                if candidate.exists():
                    return candidate

        searched = [str(root / checkpoint_name) for root in search_roots for checkpoint_name in checkpoint_names]
        raise FileNotFoundError(f"No SAM2 checkpoint found. Searched: {searched}")

    def _generate_masks(self, crop: np.ndarray) -> list:
        rgb_crop = crop[:, :, ::-1]
        masks = self._mask_generator.generate(rgb_crop)
        return masks if masks is not None else []

    def _background_fill(self, dtype: np.dtype) -> np.ndarray:
        if np.issubdtype(dtype, np.floating):
            return self._fill_value_bgr.astype(dtype)
        return np.rint(self._fill_value_bgr).astype(dtype)

    def _resolve_fill_value_bgr(self, fill_value: Optional[tuple[int, int, int]]) -> np.ndarray:
        if fill_value is None:
            rgb_value = np.rint(_IMAGENET_MEAN_RGB * 255.0)
        else:
            if len(fill_value) != 3:
                raise ValueError("fill_value must be a 3-tuple of RGB channel values")
            rgb_value = np.array(fill_value, dtype=np.float32)

        return rgb_value[::-1].astype(np.float32)