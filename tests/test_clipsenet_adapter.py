"""ClipSenetEmbedderAdapter tests.

Logic tests run without the 370MB checkpoint (bypassed construction);
the real-model smoke is gated behind ATHAR_RUN_PARITY like the other
heavy checks (the offline-equivalence script is the deeper validation:
scripts/eval/check_clipsenet_offline_build.py).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from athar.components.adapters.embedding import ClipSenetEmbedderAdapter
from athar.components.embedders.crop_extractor import QualityScoredCrop

CHECKPOINT = Path("models/reid/clipsenet_v6_veri776_best.pth")


def _bare_adapter(features: np.ndarray, temperature: float = 3.0):
    """Adapter with construction bypassed and _embed_bgr canned."""
    adapter = object.__new__(ClipSenetEmbedderAdapter)
    adapter.stream_name = "clipsenet"
    adapter.dim = features.shape[1]
    adapter._quality_temperature = temperature
    adapter._batch_size = 16
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
        features = np.eye(3, 4, dtype=np.float32)
        qualities = [0.9, 0.1, 0.5]
        adapter = _bare_adapter(features, temperature=3.0)
        vec = adapter.embed_tracklet(_scored(qualities))
        weights = np.exp(np.asarray(qualities, np.float32) * 3.0)
        weights = weights / weights.sum()
        expected = (features * weights[:, None]).sum(axis=0)
        np.testing.assert_allclose(vec, expected, rtol=1e-6)

    def test_empty_tracklet_returns_none(self):
        adapter = _bare_adapter(np.zeros((1, 4), np.float32))
        assert adapter.embed_tracklet([]) is None

    def test_embed_unpacks_stacked_crops(self):
        adapter = _bare_adapter(np.zeros((5, 4), np.float32))
        out = adapter.embed(np.zeros((5, 8, 8, 3), dtype=np.uint8))
        assert out.shape == (5, 4)
        assert adapter.calls == [5]


class TestRegistry:
    def test_registered_as_clipsenet_v1(self):
        import athar.components.adapters  # noqa: F401 — registration side effect
        from athar.components.protocols import ComponentKindName
        from athar.components.registry import registry

        assert "clipsenet_v1" in set(registry.names(ComponentKindName.EMBEDDER))


@pytest.mark.skipif(
    not os.environ.get("ATHAR_RUN_PARITY") or not CHECKPOINT.is_file(),
    reason="real-checkpoint smoke: set ATHAR_RUN_PARITY=1 (needs the frozen ckpt)",
)
class TestRealModel:
    def test_offline_construction_and_shapes(self):
        adapter = ClipSenetEmbedderAdapter(str(CHECKPOINT))
        rng = np.random.default_rng(0)
        crops = np.stack(
            [rng.integers(0, 255, (64, 80, 3), dtype=np.uint8) for _ in range(2)]
        )
        features = adapter.embed(crops)
        assert features.shape == (2, 2048)
        np.testing.assert_allclose(np.linalg.norm(features, axis=1), 1.0, rtol=1e-4)
