"""Tests for AppState containerisation (Phase 4)."""

import pytest
from fastapi.testclient import TestClient

from backend.dependencies import get_app_state
from backend.state import AppState, active_runs, app_state, uploaded_videos, video_to_latest_run


def test_app_state_reset_clears_all():
    """reset() clears all 3 dicts in-place without replacing the objects."""
    app_state.active_runs["r1"] = {"status": "running"}
    app_state.uploaded_videos["v1"] = {"name": "test.mp4"}
    app_state.video_to_latest_run["v1"] = "r1"

    app_state.reset()

    assert app_state.active_runs == {}
    assert app_state.uploaded_videos == {}
    assert app_state.video_to_latest_run == {}


def test_backward_compat_aliases_share_same_dict():
    """Module-level aliases point to the SAME dict objects as app_state attrs."""
    assert active_runs is app_state.active_runs
    assert uploaded_videos is app_state.uploaded_videos
    assert video_to_latest_run is app_state.video_to_latest_run


def test_get_app_state_returns_singleton():
    """get_app_state() always returns the same AppState singleton."""
    assert get_app_state() is app_state
    assert get_app_state() is get_app_state()


def test_dependency_override_in_test_client():
    """app.dependency_overrides[get_app_state] isolates route state in tests."""
    from backend.app import app

    isolated = AppState()
    isolated.uploaded_videos["demo"] = {
        "id": "demo",
        "name": "demo.mp4",
        "path": "/tmp/demo.mp4",
        "size": 0,
        "width": 1920,
        "height": 1080,
        "duration": 10.0,
        "fps": 30.0,
        "cameraId": None,
        "status": "ready",
        "uploadedAt": "2024-01-01T00:00:00",
    }

    app.dependency_overrides[get_app_state] = lambda: isolated
    try:
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/api/videos")
        assert resp.status_code == 200
        ids = [v["id"] for v in resp.json().get("data", [])]
        assert "demo" in ids

        # Global singleton is NOT affected
        assert "demo" not in app_state.uploaded_videos
    finally:
        app.dependency_overrides.pop(get_app_state, None)
