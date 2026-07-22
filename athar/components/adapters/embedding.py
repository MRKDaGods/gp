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
        vit_model: str = "vit_base_patch16_clip_224.openai",
        clip_normalization: Optional[bool] = None,
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
            vit_model=vit_model,
            clip_normalization=clip_normalization,
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


class ClipSenetEmbedderAdapter:
    """CLIP-SENet v6 appearance stream (the 91.36/93.3 VeRi arch).

    Constructed OFFLINE by design (air-gap): vendored IBN-a appearance
    backbone + timm TinyCLIP semantic branch, both ``pretrained=False``,
    then a strict checkpoint load. Bitwise equivalence of this construction
    vs the original pretrained-download path is established by
    ``scripts/eval/check_clipsenet_offline_build.py`` (state dicts
    identical; forward drift ~6e-8 from kernel-order differences).

    Tracklet pooling mirrors the v1 TransReID semantics exactly:
    ``softmax(quality * temperature)`` weights over per-crop L2-normed
    features, weighted sum, no final re-norm.
    """

    def __init__(
        self,
        weights_path: str,
        stream_name: str = "clipsenet",
        device: str = "cpu",
        batch_size: int = 16,
        quality_temperature: float = 3.0,
        image_size: Optional[Sequence[int]] = None,  # default: canonical (320, 320)
    ) -> None:
        from pathlib import Path

        from athar.components.embedders.clip_senet_v6 import (
            IMAGE_SIZE,
            build_clip_senet,
            build_transform,
            load_checkpoint,
        )

        state_dict, _kind, num_classes = load_checkpoint(
            Path(weights_path), map_location="cpu"
        )
        model = build_clip_senet(
            num_classes=num_classes,
            appearance_pretrained=False,
            semantic_pretrained=False,
        )
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "CLIP-SENet checkpoint load was not strict: "
                f"missing={missing[:5]}, unexpected={unexpected[:5]}"
            )
        self._device = device
        self._model = model.to(device).eval()
        self._transform = build_transform(
            tuple(image_size) if image_size else IMAGE_SIZE
        )
        self.stream_name = stream_name
        self.dim = 2048
        self.model_id: Optional[str] = None
        self._batch_size = batch_size
        self._quality_temperature = quality_temperature

    def _embed_bgr(self, crops: "list[np.ndarray]") -> np.ndarray:
        import torch
        import torch.nn.functional as F
        from PIL import Image

        features: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(crops), self._batch_size):
                chunk = crops[start:start + self._batch_size]
                batch = torch.stack(
                    [self._transform(Image.fromarray(c[:, :, ::-1])) for c in chunk]
                ).to(self._device)
                out = self._model(batch)
                if isinstance(out, (tuple, list)):
                    out = out[-1]
                features.append(
                    F.normalize(out.float(), p=2, dim=1).cpu().numpy()
                )
        return np.concatenate(features, axis=0).astype(np.float32)

    def embed(self, crops: np.ndarray) -> np.ndarray:
        return self._embed_bgr([crops[i] for i in range(crops.shape[0])])

    def embed_tracklet(
        self, scored_crops: "list[QualityScoredCrop]", cam_index: Optional[int] = None
    ) -> Optional[np.ndarray]:
        if not scored_crops:
            return None
        features = self._embed_bgr([c.image for c in scored_crops])
        qualities = np.asarray([c.quality for c in scored_crops], dtype=np.float32)
        weights = np.exp(qualities * self._quality_temperature)
        weights = weights / weights.sum()
        return (features * weights[:, np.newaxis]).sum(axis=0)


class Dinov2EmbedderAdapter:
    """DINOv2-L TransReID tertiary stream (14e B1 recipe, CityFlowV2 ckpt).

    Arch is the verbatim 09s-kernel port
    (:mod:`athar.components.embedders.transreid_dinov2_09s`); constructed
    OFFLINE (``pretrained=False`` + strict load of the full finetuned
    state dict — ``num_classes``/``num_cameras`` are inferred from the
    checkpoint shapes). Eval mirrors the kernel's ``extract_features``:
    horizontal-flip TTA averaging, then a final L2 re-norm.

    Camera ids are NOT passed at inference (SIE skipped): the checkpoint's
    camera vocabulary is its CityFlowV2 train split, which is meaningless
    for deployment footage — same convention as the other v1 streams.
    """

    def __init__(
        self,
        weights_path: str,
        stream_name: str = "dinov2",
        device: str = "cpu",
        batch_size: int = 8,
        quality_temperature: float = 3.0,
        flip_augment: bool = True,
        image_size: Optional[int] = None,  # default: canonical 252
    ) -> None:
        import torch

        from athar.components.embedders.transreid_dinov2_09s import (
            EMBED_DIM,
            IMG_SIZE,
            STRIDE_SIZE,
            VIT_MODEL,
            TransReID,
            build_test_transform,
            infer_checkpoint_dims,
        )

        img_size = int(image_size) if image_size else IMG_SIZE
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        num_classes, num_cameras = infer_checkpoint_dims(state_dict)
        model = TransReID(
            num_classes=num_classes,
            num_cameras=num_cameras,
            embed_dim=EMBED_DIM,
            vit_model=VIT_MODEL,
            pretrained=False,
            sie_camera=num_cameras > 0,
            jpm=True,
            img_size=img_size,
            stride_size=STRIDE_SIZE,
        )
        model.load_state_dict(state_dict, strict=True)
        self._device = device
        self._model = model.to(device).eval()
        self._transform = build_test_transform(img_size)
        self.stream_name = stream_name
        self.dim = EMBED_DIM
        self.model_id: Optional[str] = None
        self._batch_size = batch_size
        self._quality_temperature = quality_temperature
        self._flip_augment = flip_augment

    def _embed_bgr(self, crops: "list[np.ndarray]") -> np.ndarray:
        import torch
        import torch.nn.functional as F
        from PIL import Image

        features: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(crops), self._batch_size):
                chunk = crops[start:start + self._batch_size]
                batch = torch.stack(
                    [self._transform(Image.fromarray(c[:, :, ::-1])) for c in chunk]
                ).to(self._device)
                feats = self._model(batch)
                if self._flip_augment:
                    flipped = self._model(torch.flip(batch, [3]))
                    feats = (feats + flipped) / 2.0
                features.append(
                    F.normalize(feats.float(), p=2, dim=1).cpu().numpy()
                )
        return np.concatenate(features, axis=0).astype(np.float32)

    def embed(self, crops: np.ndarray) -> np.ndarray:
        return self._embed_bgr([crops[i] for i in range(crops.shape[0])])

    def embed_tracklet(
        self, scored_crops: "list[QualityScoredCrop]", cam_index: Optional[int] = None
    ) -> Optional[np.ndarray]:
        if not scored_crops:
            return None
        features = self._embed_bgr([c.image for c in scored_crops])
        qualities = np.asarray([c.quality for c in scored_crops], dtype=np.float32)
        weights = np.exp(qualities * self._quality_temperature)
        weights = weights / weights.sum()
        return (features * weights[:, np.newaxis]).sum(axis=0)


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
