from __future__ import annotations

from unittest import mock

import pytest
from fastapi.testclient import TestClient

from backend.app import app
from backend.routers import pipeline as pipeline_router
from backend.services.kaggle_service import KaggleAuthError


@pytest.fixture()
def client():
    with TestClient(app) as test_client:
        yield test_client


def _state(status: str = "running") -> dict[str, object]:
    return {
        "run_id": "cancel-run",
        "kernel_slug": "gumfreddy/cancel-run",
        "kernel_url": "https://www.kaggle.com/code/gumfreddy/cancel-run",
        "dataset_slug": "owner/dataset",
        "project_dataset_slug": "gumfreddy/mtmc-tracker-source",
        "status": status,
        "stages": [1],
        "started_at": "2026-05-20T00:00:00Z",
        "last_polled_at": None,
        "exit_code": None,
        "outputs_downloaded_to": None,
    }


def test_cancel_active_job_marks_cancelled(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state("running")
    cancel_kernel = mock.Mock()
    persist = mock.Mock()
    monkeypatch.setattr(pipeline_router, "get_kaggle_job_state", mock.Mock(return_value=state))
    monkeypatch.setattr(pipeline_router.kaggle_service, "cancel_kernel", cancel_kernel)
    monkeypatch.setattr(pipeline_router, "_persist_kaggle_state", persist)

    response = client.post("/pipeline/kaggle-cancel/cancel-run")

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["status"] == "cancelled"
    assert data["last_polled_at"].endswith("Z")
    cancel_kernel.assert_called_once_with("gumfreddy/cancel-run")
    persist.assert_called_once()
    assert persist.call_args.args[0] == "cancel-run"
    assert persist.call_args.args[1]["status"] == "cancelled"


def test_cancel_terminal_job_returns_message_without_kaggle_call(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state("complete")
    cancel_kernel = mock.Mock()
    monkeypatch.setattr(pipeline_router, "get_kaggle_job_state", mock.Mock(return_value=state))
    monkeypatch.setattr(pipeline_router.kaggle_service, "cancel_kernel", cancel_kernel)

    response = client.post("/api/pipeline/kaggle-cancel/cancel-run")

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["status"] == "complete"
    assert data["message"] == "Job already in terminal state"
    cancel_kernel.assert_not_called()


def test_cancel_nonexistent_run_returns_404(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pipeline_router, "get_kaggle_job_state", mock.Mock(return_value=None))

    response = client.post("/pipeline/kaggle-cancel/missing-run")

    assert response.status_code == 404
    assert "No Kaggle job found" in response.json()["detail"]


def test_cancel_auth_error_maps_to_401(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pipeline_router, "get_kaggle_job_state", mock.Mock(return_value=_state()))
    monkeypatch.setattr(
        pipeline_router.kaggle_service,
        "cancel_kernel",
        mock.Mock(side_effect=KaggleAuthError("bad creds")),
    )

    response = client.post("/pipeline/kaggle-cancel/cancel-run")

    assert response.status_code == 401
    assert response.json()["detail"] == "bad creds"


def test_cancel_generic_error_maps_to_500(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pipeline_router, "get_kaggle_job_state", mock.Mock(return_value=_state()))
    monkeypatch.setattr(
        pipeline_router.kaggle_service,
        "cancel_kernel",
        mock.Mock(side_effect=RuntimeError("cancel failed")),
    )

    response = client.post("/pipeline/kaggle-cancel/cancel-run")

    assert response.status_code == 500
    assert response.json()["detail"] == "cancel failed"
