"""v2 Embedder protocol over the ported v1 embedding kernels.

Each adapter satisfies the flat protocol (``embed(crops) -> (N, dim)`` rows,
L2-normed) and additionally exposes ``embed_tracklet(scored_crops)`` — the
byte-faithful v1 tracklet-level path (flip augment + softmax-quality
attention pooling) that the embed stage prefers when present.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from athar.components.embedders.crop_extractor import QualityScoredCrop


class TransReidEmbedderAdapter:
    """TransReID stream via the ported v1 ``ReIDModel`` wrapper (parity
    component, D18). ``input_size``/``num_cameras`` must match the
    checkpoint recipe (see tests/test_checkpoint_contract.py)."""

    def __init__(
        self,
        weights_path: str,
        stream_name: str = "transreid_primary",
        input_size: Sequence[int] = (224, 224),  # (H, W) — VeRi recipe
        num_cameras: int = 20,
        device: str = "cpu",
        half: bool = False,
        flip_augment: bool = True,
        quality_temperature: float = 3.0,
        batch_size: int = 16,
    ) -> None:
        from athar.components.embedders.reid_model import ReIDModel

        self.stream_name = stream_name
        self.dim = 768
        self.model_id: Optional[str] = None
        self._quality_temperature = quality_temperature
        self._batch_size = batch_size
        self._model = ReIDModel(
            model_name="transreid",
            weights_path=weights_path,
            embedding_dim=self.dim,
            input_size=tuple(input_size),
            device=device,
            half=half,
            flip_augment=flip_augment,
            num_cameras=num_cameras,
        )

    def embed(self, crops: np.ndarray) -> np.ndarray:
        crop_list = [crops[i] for i in range(crops.shape[0])]
        return self._model.extract_features(crop_list, batch_size=self._batch_size)

    def embed_tracklet(
        self, scored_crops: "list[QualityScoredCrop]", cam_index: Optional[int] = None
    ) -> Optional[np.ndarray]:
        return self._model.get_tracklet_embedding_from_scored_crops(
            scored_crops,
            cam_id=cam_index,
            quality_temperature=self._quality_temperature,
        )


class HsvEmbedderAdapter:
    """Striped HSV color histogram stream (pure numpy/cv2 — always available).

    Color is a weak identity cue on its own but a strong complementary
    score term; IR/grayscale segments de-weight it dynamically (D15).
    """

    def __init__(
        self,
        stream_name: str = "hsv",
        h_bins: int = 32,
        s_bins: int = 16,
        v_bins: int = 16,
        n_stripes: int = 3,
        device: str = "cpu",  # accepted for slot uniformity; histograms are CPU
    ) -> None:
        from athar.components.embedders.hsv_extractor import HSVExtractor

        self._extractor = HSVExtractor(
            h_bins=h_bins, s_bins=s_bins, v_bins=v_bins, n_stripes=n_stripes
        )
        self.stream_name = stream_name
        self.dim = self._extractor.total_bins

    def embed(self, crops: np.ndarray) -> np.ndarray:
        rows = [self._extractor.extract_histogram(crops[i]) for i in range(crops.shape[0])]
        return np.stack(rows).astype(np.float32)

    def embed_tracklet(
        self, scored_crops: "list[QualityScoredCrop]", cam_index: Optional[int] = None
    ) -> Optional[np.ndarray]:
        if not scored_crops:
            return None
        return self._extractor.extract_tracklet_histogram_from_scored_crops(scored_crops)
