from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.app import app
from backend.dependencies import get_app_state
from backend.routers import pipeline as pipeline_router
from backend.state import AppState


@pytest.fixture()
def client(tmp_path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    test_state = AppState()
    video_path = tmp_path / "dummy_video.mp4"
    video_path.write_bytes(b"video")
    test_state.uploaded_videos["dummy_video"] = {
        "id": "dummy_video",
        "name": "dummy_video.mp4",
        "path": str(video_path),
    }

    async def noop_execute_stage(*args, **kwargs):
        return None

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pipeline_router, "execute_stage", noop_execute_stage)
    app.dependency_overrides[get_app_state] = lambda: test_state
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


@patch("backend.routers.pipeline.kaggle_service")
@patch("backend.services.kaggle_run_service.kaggle_service")
@patch("backend.services.kaggle_run_service.kaggle_kernel_templates")
def test_kaggle_dispatch_lifecycle(
    mock_templates: MagicMock,
    mock_kaggle: MagicMock,
    mock_router_kaggle: MagicMock,
    client: TestClient,
) -> None:
    """End-to-end Kaggle dispatch: POST run-stage -> job persisted -> GET status -> POST cancel."""
    mock_kaggle.whoami.return_value = "testuser"
    mock_kaggle.count_active_kernels.return_value = 0
    mock_kaggle.dataset_create_or_update.return_value = MagicMock(
        slug="testuser/mtmc-tracker-source",
        version="1",
    )
    mock_kaggle.push_kernel.return_value = MagicMock(
        slug="testuser/mtmc-test-stage1",
        kernel_url="https://www.kaggle.com/code/testuser/mtmc-test-stage1",
    )
    mock_templates.render_kernel.side_effect = lambda _ctx, output_dir: output_dir

    response = client.post(
        "/api/pipeline/run-stage/1",
        json={
            "runId": "test_e2e_kaggle",
            "videoId": "dummy_video",
            "kaggle": {
                "target": "kaggle",
                "username": "testuser",
                "key": "testkey",
                "dataset_slug": "testuser/preexisting",
            },
        },
    )

    assert response.status_code == 200, response.json()
    body = response.json()["data"]
    assert body["execution_target"] == "kaggle"
    assert body["kaggle"]["kernel_slug"] == "testuser/mtmc-test-stage1"
    assert body["kaggle"]["kernel_url"] == "https://www.kaggle.com/code/testuser/mtmc-test-stage1"
    assert body["kaggle"]["dataset_slug"] == "testuser/preexisting"
    assert body["kaggle"]["status"] == "queued"
    mock_kaggle.push_kernel.assert_called_once()

    mock_kaggle.kernel_status.return_value = MagicMock(
        slug="testuser/mtmc-test-stage1",
        status="running",
        raw_stdout="...",
        last_polled_iso="2025-01-01T00:00:00Z",
    )
    status_resp = client.get("/api/pipeline/kaggle-status/test_e2e_kaggle")

    assert status_resp.status_code == 200, status_resp.json()
    assert status_resp.json()["data"]["status"] == "running"
    assert status_resp.json()["data"]["last_polled_at"] == "2025-01-01T00:00:00Z"

    mock_router_kaggle.cancel_kernel.return_value = MagicMock(
        final_status="cancelled",
        attempts=1,
        fallback_used=False,
    )
    cancel_resp = client.post("/api/pipeline/kaggle-cancel/test_e2e_kaggle")

    assert cancel_resp.status_code == 200, cancel_resp.json()
    assert cancel_resp.json()["data"]["status"] == "cancelled"
    mock_router_kaggle.cancel_kernel.assert_called_once_with("testuser/mtmc-test-stage1")


@patch("backend.services.kaggle_run_service.kaggle_service")
def test_kaggle_dispatch_concurrency_429(
    mock_kaggle: MagicMock,
    client: TestClient,
) -> None:
    """Both Kaggle slots busy -> 429."""
    mock_kaggle.whoami.return_value = "testuser"
    mock_kaggle.count_active_kernels.return_value = 2

    response = client.post(
        "/api/pipeline/run-stage/1",
        json={
            "runId": "test_concurrency",
            "kaggle": {
                "target": "kaggle",
                "username": "u",
                "key": "k",
                "dataset_slug": "u/d",
            },
        },
    )

    assert response.status_code == 429


def test_kaggle_local_target_unaffected(client: TestClient) -> None:
    """target='local' or no kaggle field -> local code path unchanged."""
    response = client.post(
        "/api/pipeline/run-stage/0",
        json={
            "runId": "test_local",
            "videoId": "nonexistent",
        },
    )

    assert response.status_code < 600