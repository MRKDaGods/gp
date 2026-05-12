"""SAM2-based foreground masking for Stage 2 vehicle crops."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

_SAM2_MODEL_SPECS = {
    "sam2_hiera_tiny": {
        "hf_model_id": "facebook/sam2-hiera-tiny",
        "config": "configs/sam2/sam2_hiera_t.yaml",
        "config_21": "configs/sam2.1/sam2.1_hiera_t.yaml",
        "checkpoint_names": (
            "sam2_hiera_tiny.pt",
            "sam2_hiera_t.pt",
            "sam2.1_hiera_tiny.pt",
            "sam2.1_hiera_t.pt",
        ),
    },
    "sam2_hiera_small": {
        "hf_model_id": "facebook/sam2-hiera-small",
        "config": "configs/sam2/sam2_hiera_s.yaml",
        "config_21": "configs/sam2.1/sam2.1_hiera_s.yaml",
        "checkpoint_names": (
            "sam2_hiera_small.pt",
            "sam2_hiera_s.pt",
            "sam2.1_hiera_small.pt",
            "sam2.1_hiera_s.pt",
        ),
    },
    "sam2_hiera_base_plus": {
        "hf_model_id": "facebook/sam2-hiera-base-plus",
        "config": "configs/sam2/sam2_hiera_b+.yaml",
        "config_21": "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "checkpoint_names": (
            "sam2_hiera_base_plus.pt",
            "sam2_hiera_b+.pt",
            "sam2.1_hiera_base_plus.pt",
            "sam2.1_hiera_b+.pt",
        ),
    },
    "sam2_hiera_large": {
        "hf_model_id": "facebook/sam2-hiera-large",
        "config": "configs/sam2/sam2_hiera_l.yaml",
        "config_21": "configs/sam2.1/sam2.1_hiera_l.yaml",
        "checkpoint_names": (
            "sam2_hiera_large.pt",
            "sam2_hiera_l.pt",
            "sam2.1_hiera_large.pt",
            "sam2.1_hiera_l.pt",
        ),
    },
}


def _normalise_model_name(model_name: str) -> str | None:
    name = str(model_name).lower().replace("\\", "/")
    name = Path(name).name
    if name.endswith((".pt", ".pth")):
        name = name.removesuffix(".pth").removesuffix(".pt")
    name = name.replace("facebook/", "")
    name = name.replace("sam2.1-", "sam2_").replace("sam2-", "sam2_")
    name = name.replace("hiera-base-plus", "hiera_base_plus")
    name = name.replace("hiera-tiny", "hiera_tiny")
    name = name.replace("hiera-small", "hiera_small")
    name = name.replace("hiera-large", "hiera_large")

    aliases = {
        "hiera_tiny": "sam2_hiera_tiny",
        "tiny": "sam2_hiera_tiny",
        "t": "sam2_hiera_tiny",
        "hiera_small": "sam2_hiera_small",
        "small": "sam2_hiera_small",
        "s": "sam2_hiera_small",
        "hiera_base_plus": "sam2_hiera_base_plus",
        "base_plus": "sam2_hiera_base_plus",
        "b+": "sam2_hiera_base_plus",
        "hiera_large": "sam2_hiera_large",
        "large": "sam2_hiera_large",
        "l": "sam2_hiera_large",
    }
    return name if name in _SAM2_MODEL_SPECS else aliases.get(name)


def _iter_model_name_candidates(model_name: str) -> list[str]:
    model_key = _normalise_model_name(model_name)
    candidates = [model_name]

    if model_key is not None:
        candidates.append(_SAM2_MODEL_SPECS[model_key]["hf_model_id"])

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

        local_model = self._load_local_model(model_name, SAM2ImagePredictor)
        if local_model is not None:
            return local_model

        last_error: Exception | None = None
        for candidate in _iter_model_name_candidates(model_name):
            try:
                predictor = SAM2ImagePredictor.from_pretrained(candidate, device=self.device)
                logger.info("SAM2 loaded: %s", candidate)
                return predictor
            except Exception as exc:  # pragma: no cover - depends on external package/runtime
                last_error = exc

        raise RuntimeError(f"Unable to load SAM2 model '{model_name}'") from last_error

    def _load_local_model(self, model_name: str, predictor_cls: type):
        checkpoint_path = self._resolve_local_checkpoint(model_name)
        if checkpoint_path is None:
            return None

        model_key = _normalise_model_name(str(checkpoint_path)) or _normalise_model_name(model_name)
        if model_key is None:
            raise RuntimeError(f"Unable to infer SAM2 model type from checkpoint '{checkpoint_path}'")

        spec = _SAM2_MODEL_SPECS[model_key]
        config_name = spec["config_21"] if "sam2.1" in checkpoint_path.name else spec["config"]

        try:
            from sam2.build_sam import build_sam2

            model = build_sam2(
                config_name,
                str(checkpoint_path),
                device=self.device,
                apply_postprocessing=False,
            )
            logger.info(
                "SAM2 loaded from local checkpoint: model=%s config=%s checkpoint=%s",
                model_key,
                config_name,
                checkpoint_path,
            )
            return predictor_cls(model)
        except Exception as exc:  # pragma: no cover - depends on external package/runtime
            raise RuntimeError(
                f"Unable to load SAM2 local checkpoint '{checkpoint_path}' with config '{config_name}'"
            ) from exc

    def _resolve_local_checkpoint(self, model_name: str) -> Path | None:
        explicit_path = Path(model_name).expanduser()
        if explicit_path.suffix.lower() in {".pt", ".pth"}:
            if explicit_path.exists():
                return explicit_path
            if explicit_path.parent != Path("."):
                raise FileNotFoundError(f"SAM2 checkpoint does not exist: {explicit_path}")

        model_key = _normalise_model_name(model_name)
        if model_key is None:
            return None

        checkpoint_names = _SAM2_MODEL_SPECS[model_key]["checkpoint_names"]
        search_roots = (
            Path("models/sam2"),
            Path("models"),
            Path("checkpoints/sam2"),
            Path("checkpoints"),
            Path("/kaggle/working/gp/models/sam2"),
        )

        for root in search_roots:
            for checkpoint_name in checkpoint_names:
                candidate = root / checkpoint_name
                if candidate.exists():
                    return candidate

        return None

    def _resolve_fill_value(self, crop_bgr: np.ndarray) -> np.ndarray:
        if isinstance(self.fill_value_mode, str):
            fill_value = self.fill_value_mode.lower()
            if fill_value == "mean":
                return np.rint(crop_bgr.mean(axis=(0, 1))).astype(np.uint8)
            if fill_value in {"zero", "black"}:
                return np.zeros(3, dtype=np.uint8)
            raise ValueError(
                "fill_value must be 'mean', 'zero', a numeric scalar, or an RGB sequence of length 3"
            )

        if isinstance(self.fill_value_mode, (int, float)):
            value = np.clip(float(self.fill_value_mode), 0.0, 255.0)
            return np.full(3, value, dtype=np.uint8)

        if isinstance(self.fill_value_mode, Sequence) and len(self.fill_value_mode) == 3:
            return np.asarray(list(self.fill_value_mode), dtype=np.uint8)

        raise ValueError("fill_value must be 'mean', 'zero', a numeric scalar, or an RGB sequence of length 3")

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