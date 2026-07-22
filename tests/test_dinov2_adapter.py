"""Dinov2EmbedderAdapter tests (09s kernel port).

Logic tests run without the 1.2GB checkpoint (bypassed construction);
the real-checkpoint smoke — offline construction + STRICT full-state
load + flip-TTA forward — is gated behind ATHAR_RUN_PARITY like the
other heavy checks.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from athar.components.adapters.embedding import Dinov2EmbedderAdapter
from athar.components.embedders.crop_extractor import QualityScoredCrop

CHECKPOINT = Path("models/reid/vehicle_transreid_dinov2_large_cityflowv2_final.pth")


def _bare_adapter(features: np.ndarray, temperature: float = 3.0):
    adapter = object.__new__(Dinov2EmbedderAdapter)
    adapter.stream_name = "dinov2"
    adapter.dim = features.shape[1]
    adapter._quality_temperature = temperature
    adapter._batch_size = 8
    adapter._flip_augment = True
    adapter.calls = []

    def fake_embed(crops):
        adapter.calls.append(len(crops))
        return features[: len(crops)]

    adapter._embed_bgr = fake_embed
    return adapter


def _scored(qualities: list[float]) -> list[QualityScoredCrop]:
    crop = np.zeros((8, 8, 3), dtype=np.uint8)
    return [
        QualityScoredCrop(image=crop, quality=q, frame_id=i, confidence=0.9)
        for i, q in enumerate(qualities)
    ]


class TestPooling:
    def test_matches_v1_softmax_semantics(self):
        features = np.eye(3, 5, dtype=np.float32)
        qualities = [0.8, 0.2, 0.6]
        adapter = _bare_adapter(features, temperature=3.0)
        vec = adapter.embed_tracklet(_scored(qualities))
        weights = np.exp(np.asarray(qualities, np.float32) * 3.0)
        weights = weights / weights.sum()
        expected = (features * weights[:, None]).sum(axis=0)
        np.testing.assert_allclose(vec, expected, rtol=1e-6)

    def test_empty_tracklet_returns_none(self):
        adapter = _bare_adapter(np.zeros((1, 4), np.float32))
        assert adapter.embed_tracklet([]) is None


class TestCheckpointIntrospection:
    def test_dims_inferred_from_shapes(self):
        torch = pytest.importorskip("torch")
        from athar.components.embedders.transreid_dinov2_09s import (
            infer_checkpoint_dims,
        )

        state = {
            "cls_head.weight": torch.zeros(93, 1024),
            "sie_embed": torch.zeros(38, 1024),
        }
        assert infer_checkpoint_dims(state) == (93, 38)
        assert infer_checkpoint_dims({"cls_head.weight": torch.zeros(7, 8)}) == (7, 0)
        with pytest.raises(KeyError, match="cls_head"):
            infer_checkpoint_dims({"bn.weight": torch.zeros(4)})


class TestRecipe:
    def test_test_transform_matches_09s(self):
        pytest.importorskip("torch")
        from athar.components.embedders.transreid_dinov2_09s import (
            CLIP_MEAN,
            IMG_SIZE,
            VIT_MODEL,
            build_test_transform,
        )

        assert IMG_SIZE == 252
        assert VIT_MODEL == "vit_large_patch14_dinov2"
        assert CLIP_MEAN[0] == pytest.approx(0.48145466)
        transform = build_test_transform()
        from PIL import Image

        out = transform(Image.new("RGB", (100, 50)))
        assert tuple(out.shape) == (3, 252, 252)


class TestRegistry:
    def test_registered_as_dinov2_v1(self):
        import athar.components.adapters  # noqa: F401 — registration side effect
        from athar.components.protocols import ComponentKindName
        from athar.components.registry import registry

        assert "dinov2_v1" in set(registry.names(ComponentKindName.EMBEDDER))


@pytest.mark.skipif(
    not os.environ.get("ATHAR_RUN_PARITY") or not CHECKPOINT.is_file(),
    reason="real-checkpoint smoke: set ATHAR_RUN_PARITY=1 (needs the 1.2GB ckpt)",
)
class TestRealModel:
    def test_offline_strict_load_and_shapes(self):
        adapter = Dinov2EmbedderAdapter(str(CHECKPOINT))
        rng = np.random.default_rng(0)
        crops = np.stack(
            [rng.integers(0, 255, (96, 128, 3), dtype=np.uint8) for _ in range(2)]
        )
        features = adapter.embed(crops)
        assert features.shape == (2, 1024)
        np.testing.assert_allclose(np.linalg.norm(features, axis=1), 1.0, rtol=1e-4)
        # distinct crops must not collapse to one point
        assert float(features[0] @ features[1]) < 0.999
