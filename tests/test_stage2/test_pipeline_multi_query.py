"""Tests for Stage 2 pipeline multi-query integration."""

from __future__ import annotations

import numpy as np
from omegaconf import OmegaConf

from src.core.data_models import Tracklet, TrackletFrame
from src.stage2_features.crop_extractor import QualityScoredCrop
from src.stage2_features.pipeline import run_stage2


class _FakeCropExtractor:
    def __init__(self, *args, **kwargs):
        pass

    def extract_crops(self, tracklet, video_path):
        return [
            QualityScoredCrop(
                image=np.zeros((8, 8, 3), dtype=np.uint8),
                quality=0.9,
                frame_id=tracklet.frames[0].frame_id,
                confidence=0.95,
            )
        ]


class _FakeHSVExtractor:
    def __init__(self, *args, **kwargs):
        pass

    def extract_tracklet_histogram_from_scored_crops(self, scored_crops):
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)


class _FakeReIDModel:
    def __init__(self, *args, **kwargs):
        pass

    def get_tracklet_embedding_from_scored_crops(self, scored_crops, cam_id=None, quality_temperature=3.0):
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)

    def get_tracklet_multi_query_embeddings(self, *args, **kwargs):
        raise AssertionError("MQ extraction should not be called when stage2.multi_query.k=0")


class _MaskAwareFakeReIDModel:
    def __init__(self, *args, **kwargs):
        pass

    def get_tracklet_embedding_from_scored_crops(
        self,
        scored_crops,
        cam_id=None,
        quality_temperature=3.0,
    ):
        if np.any(scored_crops[0].image):
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)


class _FakeForegroundMasker:
    instances = []

    def __init__(self, *args, **kwargs):
        self.calls = 0
        _FakeForegroundMasker.instances.append(self)

    def mask_crops(self, scored_crops):
        self.calls += 1
        for crop in scored_crops:
            crop.image = np.full_like(crop.image, 7)
        return scored_crops


def test_run_stage2_with_multi_query_disabled_is_backward_compatible(monkeypatch, tmp_path):
    monkeypatch.setattr("src.stage2_features.pipeline.CropExtractor", _FakeCropExtractor)
    monkeypatch.setattr("src.stage2_features.pipeline.HSVExtractor", _FakeHSVExtractor)
    monkeypatch.setattr("src.stage2_features.pipeline.ReIDModel", _FakeReIDModel)

    cfg = OmegaConf.create(
        {
            "dataset": {"target_classes": [2]},
            "stage2": {
                "reid": {
                    "device": "cpu",
                    "half": False,
                    "vehicle": {
                        "model_name": "fake",
                        "weights_path": "",
                        "embedding_dim": 3,
                        "input_size": [256, 256],
                        "num_cameras": 0,
                        "vit_model": "fake",
                        "clip_normalization": False,
                        "sie_camera_map": {},
                    },
                    "vehicle2": {"enabled": False, "save_separate": False},
                    "flip_augment": False,
                    "color_augment": False,
                    "multiscale_sizes": [],
                    "quality_temperature": 3.0,
                },
                "hsv": {"h_bins": 1, "s_bins": 1, "v_bins": 1, "n_stripes": 1},
                "pca": {"enabled": False, "n_components": 3},
                "multi_query": {"k": 0},
                "crop": {
                    "min_area": 1,
                    "padding_ratio": 0.1,
                    "samples_per_tracklet": 1,
                    "min_quality": 0.0,
                    "laplacian_min_var": 0.0,
                },
                "camera_bn": {"enabled": False},
                "power_norm": {"alpha": 0.0},
                "camera_tta": {"enabled": False},
            },
        }
    )

    tracklets_by_camera = {
        "cam01": [
            Tracklet(
                track_id=1,
                camera_id="cam01",
                class_id=2,
                class_name="car",
                frames=[TrackletFrame(frame_id=0, timestamp=0.0, bbox=(0, 0, 10, 10), confidence=0.9)],
            )
        ]
    }

    features = run_stage2(
        cfg=cfg,
        tracklets_by_camera=tracklets_by_camera,
        video_paths={"cam01": "dummy.mp4"},
        output_dir=tmp_path,
    )

    assert len(features) == 1
    assert features[0].multi_query_embeddings is None
    assert not (tmp_path / "multi_query_embeddings.npz").exists()


def test_run_stage2_applies_foreground_masking_to_vehicle_crops(monkeypatch, tmp_path):
    _FakeForegroundMasker.instances.clear()
    monkeypatch.setattr("src.stage2_features.pipeline.CropExtractor", _FakeCropExtractor)
    monkeypatch.setattr("src.stage2_features.pipeline.HSVExtractor", _FakeHSVExtractor)
    monkeypatch.setattr("src.stage2_features.pipeline.ReIDModel", _MaskAwareFakeReIDModel)
    monkeypatch.setattr(
        "src.stage2_features.pipeline.ForegroundMasker",
        _FakeForegroundMasker,
    )

    cfg = OmegaConf.create(
        {
            "dataset": {"target_classes": [2]},
            "stage2": {
                "reid": {
                    "device": "cpu",
                    "half": False,
                    "vehicle": {
                        "model_name": "fake",
                        "weights_path": "",
                        "embedding_dim": 3,
                        "input_size": [256, 256],
                        "num_cameras": 0,
                        "vit_model": "fake",
                        "clip_normalization": False,
                        "sie_camera_map": {},
                    },
                    "vehicle2": {"enabled": False, "save_separate": False},
                    "flip_augment": False,
                    "color_augment": False,
                    "multiscale_sizes": [],
                    "quality_temperature": 3.0,
                },
                "hsv": {"h_bins": 1, "s_bins": 1, "v_bins": 1, "n_stripes": 1},
                "pca": {"enabled": False, "n_components": 3},
                "multi_query": {"k": 0},
                "crop": {
                    "min_area": 1,
                    "padding_ratio": 0.1,
                    "samples_per_tracklet": 1,
                    "min_quality": 0.0,
                    "laplacian_min_var": 0.0,
                    "foreground_masking": {
                        "enabled": True,
                        "model_name": "facebook/sam2.1-hiera-tiny",
                        "min_crop_size": 48,
                        "fill_value": "mean",
                    },
                },
                "camera_bn": {"enabled": False},
                "power_norm": {"alpha": 0.0},
                "camera_tta": {"enabled": False},
            },
        }
    )

    tracklets_by_camera = {
        "cam01": [
            Tracklet(
                track_id=1,
                camera_id="cam01",
                class_id=2,
                class_name="car",
                frames=[
                    TrackletFrame(
                        frame_id=0,
                        timestamp=0.0,
                        bbox=(0, 0, 10, 10),
                        confidence=0.9,
                    )
                ],
            )
        ]
    }

    features = run_stage2(
        cfg=cfg,
        tracklets_by_camera=tracklets_by_camera,
        video_paths={"cam01": "dummy.mp4"},
        output_dir=tmp_path,
    )

    assert len(features) == 1
    assert _FakeForegroundMasker.instances[0].calls == 1
    np.testing.assert_array_equal(
        features[0].raw_embedding,
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
    )