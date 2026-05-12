"""Tests for ``backend.services.timeline_service.TimelineService`` (Phase 3).

All tests use a ``StubRepository`` that injects fixture data without touching
the filesystem, keeping tests fast and hermetic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import numpy as np
import pytest

from backend.models.embedding import EmbeddingArtifact
from backend.repositories.dataset_repository import DatasetRepository
from backend.services.timeline_service import TimelineService


# ---------------------------------------------------------------------------
# Stub repository
# ---------------------------------------------------------------------------

class StubRepository:
    """In-memory DatasetRepository for tests.  No filesystem I/O."""

    def __init__(
        self,
        videos: Dict[str, Dict] | None = None,
        latest_runs: Dict[str, str] | None = None,
        trajectories: Optional[List[Dict]] = None,   # None → stage4 missing
        probe_artifact: Optional[EmbeddingArtifact] = None,
        gallery_artifact: Optional[EmbeddingArtifact] = None,
    ) -> None:
        self._videos = videos or {}
        self._latest_runs = latest_runs or {}
        self._trajectories = trajectories
        self._probe_artifact = probe_artifact
        self._gallery_artifact = gallery_artifact

    def get_video(self, video_id: str) -> Optional[Dict]:
        return self._videos.get(video_id)

    def get_latest_run(self, video_id: str) -> Optional[str]:
        return self._latest_runs.get(video_id)

    def list_trajectories(self, run_id: str) -> Optional[List[Dict]]:
        return self._trajectories

    def load_embedding_artifact(self, run_id: str) -> Optional[EmbeddingArtifact]:
        if run_id == "probe_run":
            return self._probe_artifact
        return self._gallery_artifact


assert isinstance(StubRepository(), DatasetRepository), (
    "StubRepository must satisfy the DatasetRepository Protocol"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_request(**kwargs):
    """Build a TimelineQueryRequest with sensible defaults."""
    from backend.models.requests import TimelineQueryRequest
    defaults = {
        "videoId": "vid1",
        "runId": "gallery_run",
        "selectedTrackIds": [],
    }
    defaults.update(kwargs)
    return TimelineQueryRequest(**defaults)


# Patch heavy I/O helpers so tests don't need real pipeline output dirs
_NOOP_PATCHES = [
    "backend.services.timeline_service._build_selected_tracklet_summaries",
    "backend.services.timeline_service._resolve_probe_run_id_for_video",
    "backend.services.timeline_service._run_dir_for_video",
    "backend.services.timeline_service._detect_camera_for_video",
]


# ---------------------------------------------------------------------------
# Test 1 — EmbeddingArtifact.load from fixture directory
# ---------------------------------------------------------------------------

def test_embedding_artifact_load(fixture_timeline_dir):
    """EmbeddingArtifact loaded from fixture probe dir has shape (10, 384)."""
    art = EmbeddingArtifact.load(fixture_timeline_dir / "probe", "probe_run")
    assert art is not None
    assert art.embeddings.shape == (10, 384)
    assert art.dim == 384
    assert len(art.index) == 10
    assert art.run_id == "probe_run"


# ---------------------------------------------------------------------------
# Test 2 — InMemoryDatasetRepository smoke test
# ---------------------------------------------------------------------------

def test_dataset_repository_get_video():
    """InMemoryDatasetRepository.get_video returns the expected record."""
    from backend.repositories import InMemoryDatasetRepository

    videos = {"vid1": {"filename": "clip.mp4", "camera": "c001"}}
    repo = InMemoryDatasetRepository(videos, {}, Path("/no/such/dir"))
    assert repo.get_video("vid1") == {"filename": "clip.mp4", "camera": "c001"}
    assert repo.get_video("missing") is None


# ---------------------------------------------------------------------------
# Test 3 — no_selection fast path
# ---------------------------------------------------------------------------

def test_query_no_selection():
    """Empty selectedTrackIds returns mode='no_selection' without any I/O."""
    repo = StubRepository(videos={"vid1": {}})
    service = TimelineService(repo)
    req = _make_request(selectedTrackIds=[])

    result = service.query(req, {"vid1": {}})

    assert result["success"] is True
    data = result["data"]
    assert data["mode"] == "no_selection"
    assert data["trajectories"] == []
    assert data["selectedTracklets"] == []


# ---------------------------------------------------------------------------
# Test 4 — Stage 4 missing → needs_association
# ---------------------------------------------------------------------------

def test_query_no_stage4():
    """When list_trajectories returns None the service returns mode='needs_association'."""
    repo = StubRepository(
        videos={"vid1": {}},
        trajectories=None,   # stage4 missing
    )
    service = TimelineService(repo)
    req = _make_request(selectedTrackIds=["1", "2"])

    with (
        patch("backend.services.timeline_service._resolve_probe_run_id_for_video", return_value=None),
        patch("backend.services.timeline_service._detect_camera_for_video", return_value="c001"),
        patch("backend.services.timeline_service._build_selected_tracklet_summaries", return_value=[]),
        patch("backend.services.timeline_service._run_dir_for_video", return_value=None),
    ):
        result = service.query(req, {"vid1": {}})

    assert result["success"] is True
    assert result["data"]["mode"] == "needs_association"
    assert result["data"]["stage4Available"] is False


# ---------------------------------------------------------------------------
# Test 5 — visual match (probe rows ≡ gallery rows 0-9, similarity ≈ 1.0)
# ---------------------------------------------------------------------------

def test_query_visual_match(probe_artifact, gallery_artifact, sample_trajectories):
    """Probe rows matching gallery rows should score above thresholds."""
    repo = StubRepository(
        videos={"vid1": {}},
        trajectories=sample_trajectories,
        probe_artifact=probe_artifact,
        gallery_artifact=gallery_artifact,
    )
    service = TimelineService(repo)
    # Select probe tracks 1+2 only — their rows (0-3) are identical to gallery
    # rows 0-3 (trajectory 0, c002 tracks 10+11), giving similarity ≈ 1.0.
    # Selecting all 5 tracks would reduce p25 below threshold because the other
    # trajectories don't cover all probe tracks simultaneously.
    req = _make_request(
        runId="gallery_run",
        selectedTrackIds=["1", "2"],
    )

    with (
        patch("backend.services.timeline_service._resolve_probe_run_id_for_video", return_value="probe_run"),
        patch("backend.services.timeline_service._detect_camera_for_video", return_value="c001"),
        patch("backend.services.timeline_service._build_selected_tracklet_summaries", return_value=[]),
        patch("backend.services.timeline_service._run_dir_for_video", return_value=None),
    ):
        result = service.query(req, {"vid1": {}})

    assert result["success"] is True
    data = result["data"]
    assert data["stage4Available"] is True
    assert data["mode"] == "matched", f"Expected 'matched', got {data['mode']!r}. diag={data.get('diagnostics')}"
    assert len(data["trajectories"]) >= 1
    # Confidence key must be present on matched trajectories
    for traj in data["trajectories"]:
        assert "confidence" in traj


# ---------------------------------------------------------------------------
# Test 6 — embedding dim mismatch (no PCA file)
# ---------------------------------------------------------------------------

def test_query_dim_mismatch(fixture_timeline_dir):
    """Probe and gallery with different dims and no PCA pkl → dim_mismatch mode."""
    probe = EmbeddingArtifact.load(fixture_timeline_dir / "probe", "probe_run")
    assert probe is not None

    # Build a gallery with different dim (128D) to trigger mismatch
    rng = np.random.default_rng(1)
    raw128 = rng.standard_normal((5, 128)).astype(np.float32)
    norms = np.linalg.norm(raw128, axis=1, keepdims=True)
    gallery_128 = EmbeddingArtifact(
        run_id="gallery_128",
        embeddings=raw128 / np.maximum(norms, 1e-8),
        index=[{"track_id": i, "camera_id": "c002", "class_id": 2} for i in range(5)],
    )

    trajectories = [
        {"global_id": 0, "tracklets": [{"camera_id": "c002", "track_id": 0, "class_id": 2}]}
    ]
    repo = StubRepository(
        videos={"vid1": {}},
        trajectories=trajectories,
        probe_artifact=probe,
        gallery_artifact=gallery_128,
    )
    service = TimelineService(repo)
    req = _make_request(selectedTrackIds=["1", "2"])

    with (
        patch("backend.services.timeline_service._resolve_probe_run_id_for_video", return_value="probe_run"),
        patch("backend.services.timeline_service._detect_camera_for_video", return_value="c001"),
        patch("backend.services.timeline_service._build_selected_tracklet_summaries", return_value=[]),
        patch("backend.services.timeline_service._run_dir_for_video", return_value=None),
        # Ensure PCA model path does not exist so projection is skipped
        patch("backend.services.timeline_service.PCA_MODEL_PATH", Path("/no/such/pca.pkl")),
    ):
        result = service.query(req, {"vid1": {}})

    assert result["success"] is True
    diag = result["data"]["diagnostics"]
    assert diag.get("search_mode") == "embedding_dim_mismatch"
