"""Shared pytest fixtures for the MTMC tracker test suite."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

from src.core.data_models import (
    Detection,
    FrameInfo,
    GlobalTrajectory,
    Tracklet,
    TrackletFeatures,
    TrackletFrame,
)


@pytest.fixture
def sample_frame() -> np.ndarray:
    """A synthetic 480x640 BGR image."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_detection() -> Detection:
    return Detection(
        bbox=(100.0, 50.0, 200.0, 300.0),
        confidence=0.85,
        class_id=0,
        class_name="person",
    )


@pytest.fixture
def sample_detections() -> List[Detection]:
    return [
        Detection(bbox=(100, 50, 200, 300), confidence=0.9, class_id=0, class_name="person"),
        Detection(bbox=(300, 100, 450, 400), confidence=0.85, class_id=2, class_name="car"),
        Detection(bbox=(500, 200, 600, 350), confidence=0.7, class_id=0, class_name="person"),
    ]


@pytest.fixture
def sample_tracklet_frames() -> List[TrackletFrame]:
    """30 frames of synthetic tracklet data."""
    frames = []
    for i in range(30):
        frames.append(TrackletFrame(
            frame_id=i * 3,
            timestamp=i * 0.1,
            bbox=(100.0 + i, 50.0 + i * 0.5, 200.0 + i, 300.0 + i * 0.5),
            confidence=0.8 + np.random.uniform(-0.1, 0.1),
        ))
    return frames


@pytest.fixture
def sample_tracklet(sample_tracklet_frames) -> Tracklet:
    return Tracklet(
        track_id=1,
        camera_id="cam01",
        class_id=0,
        class_name="person",
        frames=sample_tracklet_frames,
    )


@pytest.fixture
def sample_tracklets_by_camera() -> Dict[str, List[Tracklet]]:
    """Two cameras with 3 tracklets each."""
    result = {}
    for cam_id in ["cam01", "cam02"]:
        tracklets = []
        for tid in range(3):
            frames = [
                TrackletFrame(
                    frame_id=i * 3,
                    timestamp=i * 0.1 + tid * 5.0,
                    bbox=(100.0 + i, 50.0, 200.0 + i, 300.0),
                    confidence=0.85,
                )
                for i in range(20)
            ]
            tracklets.append(Tracklet(
                track_id=tid,
                camera_id=cam_id,
                class_id=0 if tid < 2 else 2,
                class_name="person" if tid < 2 else "car",
                frames=frames,
            ))
        result[cam_id] = tracklets
    return result


@pytest.fixture
def sample_embeddings() -> np.ndarray:
    """(100, 512) random L2-normalized embeddings."""
    emb = np.random.randn(100, 512).astype(np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / norms


@pytest.fixture
def sample_hsv_features() -> np.ndarray:
    """(100, 32) random L2-normalized HSV histograms."""
    hsv = np.random.rand(100, 32).astype(np.float32)
    norms = np.linalg.norm(hsv, axis=1, keepdims=True)
    return hsv / norms


@pytest.fixture
def sample_tracklet_features(sample_embeddings, sample_hsv_features) -> List[TrackletFeatures]:
    """100 TrackletFeatures split across 2 cameras."""
    features = []
    for i in range(100):
        cam_id = "cam01" if i < 50 else "cam02"
        features.append(TrackletFeatures(
            track_id=i % 50,
            camera_id=cam_id,
            class_id=0,
            embedding=sample_embeddings[i],
            hsv_histogram=sample_hsv_features[i],
        ))
    return features


@pytest.fixture
def sample_global_trajectory(sample_tracklets_by_camera) -> GlobalTrajectory:
    """A trajectory spanning 2 cameras."""
    t1 = sample_tracklets_by_camera["cam01"][0]
    t2 = sample_tracklets_by_camera["cam02"][0]
    return GlobalTrajectory(global_id=0, tracklets=[t1, t2])


@pytest.fixture
def tmp_output_dir(tmp_path) -> Path:
    """Temporary output directory for test artifacts."""
    d = tmp_path / "test_output"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# Timeline / Phase-3 fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fixture_timeline_dir() -> Path:
    """Absolute path to the pre-generated timeline fixture data."""
    return Path(__file__).parent / "fixtures" / "timeline"


@pytest.fixture
def probe_artifact(fixture_timeline_dir):
    """EmbeddingArtifact loaded from the probe fixture directory."""
    from backend.models.embedding import EmbeddingArtifact
    return EmbeddingArtifact.load(fixture_timeline_dir / "probe", "probe_run")


@pytest.fixture
def gallery_artifact(fixture_timeline_dir):
    """EmbeddingArtifact loaded from the gallery fixture directory."""
    from backend.models.embedding import EmbeddingArtifact
    return EmbeddingArtifact.load(fixture_timeline_dir / "gallery", "gallery_run")


@pytest.fixture
def sample_trajectories(fixture_timeline_dir) -> List[Dict]:
    """3 synthetic global trajectories referencing gallery tracks."""
    import json
    return json.loads((fixture_timeline_dir / "global_trajectories.json").read_text())
