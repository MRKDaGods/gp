from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest
from fastapi.testclient import TestClient

from backend.app import app
from backend.dependencies import get_app_state
from backend.routers import pipeline as pipeline_router
from backend.services.kaggle_run_service import KaggleRunResult
from backend.services.kaggle_service import KaggleAuthError, KaggleConcurrencyError
from backend.state import AppState


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    test_state = AppState()
    video_path = tmp_path / "S02_c008.avi"
    video_path.write_bytes(b"video")
    test_state.uploaded_videos["vid1"] = {
        "id": "vid1",
        "name": "S02_c008.avi",
        "path": str(video_path),
    }
    stage_calls = []

    async def noop_execute_stage(*args, **kwargs):
        stage_calls.append((args, kwargs))
        return None

    monkeypatch.setattr(pipeline_router, "execute_stage", noop_execute_stage)
    app.dependency_overrides[get_app_state] = lambda: test_state
    try:
        test_client = TestClient(app)
        test_client.stage_calls = stage_calls
        yield test_client
    finally:
        app.dependency_overrides.clear()


def _dispatch_result(run_id: str = "kaggle-run") -> KaggleRunResult:
    return KaggleRunResult(
        run_id=run_id,
        kernel_slug="gumfreddy/mtmc-kaggle-run-stage-1",
        kernel_url="https://www.kaggle.com/code/gumfreddy/mtmc-kaggle-run-stage-1",
        dataset_slug="owner/dataset",
        project_dataset_slug="gumfreddy/mtmc-tracker-source",
        status="queued",
        metadata_path=f"data/outputs/{run_id}/kaggle_job.json",
    )


def test_run_stage_with_kaggle_target_returns_kaggle_block(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatch = mock.Mock(return_value=_dispatch_result())
    monkeypatch.setattr(pipeline_router, "dispatch_stage_to_kaggle", dispatch)

    response = client.post(
        "/pipeline/run-stage/1",
        json={
            "runId": "kaggle-run",
            "videoId": "vid1",
            "kaggle": {
                "target": "kaggle",
                "username": "u",
                "key": "k",
                "datasetSlug": "owner/dataset",
            },
        },
    )

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["runId"] == "kaggle-run"
    assert data["execution_target"] == "kaggle"
    assert data["kaggle"] == {
        "kernel_slug": "gumfreddy/mtmc-kaggle-run-stage-1",
        "kernel_url": "https://www.kaggle.com/code/gumfreddy/mtmc-kaggle-run-stage-1",
        "dataset_slug": "owner/dataset",
        "project_dataset_slug": "gumfreddy/mtmc-tracker-source",
        "status": "queued",
    }
    assert "fusion_resolved" in data
    assert "applied_overrides" in data
    assert client.stage_calls == []
    dispatch.assert_called_once()


def test_run_stage_kaggle_dataset_slug_does_not_require_video_id(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatch = mock.Mock(return_value=_dispatch_result("dataset-run"))
    monkeypatch.setattr(pipeline_router, "dispatch_stage_to_kaggle", dispatch)

    response = client.post(
        "/api/pipeline/run-stage/1",
        json={
            "runId": "dataset-run",
            "kaggle": {
                "target": "kaggle",
                "username": "u",
                "key": "k",
                "datasetSlug": "owner/dataset",
            },
        },
    )

    assert response.status_code == 200
    assert response.json()["data"]["execution_target"] == "kaggle"
    dispatch.assert_called_once()
    assert dispatch.call_args.kwargs["user_video_path"] is None


def test_run_stage_with_local_target_uses_existing_local_path(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatch = mock.Mock()
    monkeypatch.setattr(pipeline_router, "dispatch_stage_to_kaggle", dispatch)

    response = client.post(
        "/api/pipeline/run-stage/1",
        json={"runId": "local-run", "videoId": "vid1", "kaggle": {"target": "local"}},
    )

    assert response.status_code == 200
    assert response.json()["data"]["status"] == "running"
    assert client.stage_calls
    dispatch.assert_not_called()


def test_run_stage_without_kaggle_field_uses_existing_local_path(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatch = mock.Mock()
    monkeypatch.setattr(pipeline_router, "dispatch_stage_to_kaggle", dispatch)

    response = client.post(
        "/api/pipeline/run-stage/1",
        json={"runId": "local-regression", "videoId": "vid1"},
    )

    assert response.status_code == 200
    assert response.json()["data"]["status"] == "running"
    assert client.stage_calls
    dispatch.assert_not_called()


def test_run_stage_kaggle_auth_error_maps_to_401(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pipeline_router,
        "dispatch_stage_to_kaggle",
        mock.Mock(side_effect=KaggleAuthError("bad creds")),
    )

    response = client.post(
        "/api/pipeline/run-stage/1",
        json={
            "runId": "auth-error",
            "videoId": "vid1",
            "kaggle": {
                "target": "kaggle",
                "username": "u",
                "key": "k",
                "datasetSlug": "owner/dataset",
            },
        },
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "bad creds"


def test_run_stage_kaggle_concurrency_error_maps_to_429(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pipeline_router,
        "dispatch_stage_to_kaggle",
        mock.Mock(side_effect=KaggleConcurrencyError("busy")),
    )

    response = client.post(
        "/api/pipeline/run-stage/1",
        json={
            "runId": "busy-error",
            "videoId": "vid1",
            "kaggle": {
                "target": "kaggle",
                "username": "u",
                "key": "k",
                "datasetSlug": "owner/dataset",
            },
        },
    )

    assert response.status_code == 429
    assert response.json()["detail"] == "busy"


def test_kaggle_status_unknown_run_returns_404(client: TestClient) -> None:
    response = client.get("/pipeline/kaggle-status/not-found")

    assert response.status_code == 404
    assert "No Kaggle job found" in response.json()["detail"]


def test_kaggle_status_existing_run_returns_persisted_state(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    persisted = {
        "run_id": "status-run",
        "kernel_slug": "gumfreddy/status-run",
        "status": "running",
    }
    monkeypatch.setattr(pipeline_router, "get_kaggle_job_state", mock.Mock(return_value=persisted))
    monkeypatch.setattr(
        pipeline_router,
        "refresh_kaggle_job_status",
        mock.Mock(return_value=persisted),
    )

    response = client.get("/api/pipeline/kaggle-status/status-run")

    assert response.status_code == 200
    assert response.json()["data"] == persisted
