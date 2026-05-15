from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.app import app
from backend.dependencies import get_app_state
from backend.routers import pipeline as pipeline_router
from backend.services.model_registry import get_registry
from backend.state import AppState


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch):
    test_state = AppState()
    full_pipeline_calls = []

    async def noop_execute_full_pipeline(*args, **kwargs):
        full_pipeline_calls.append((args, kwargs))
        return None

    monkeypatch.setattr(pipeline_router, "execute_full_pipeline", noop_execute_full_pipeline)
    app.dependency_overrides[get_app_state] = lambda: test_state
    try:
        test_client = TestClient(app)
        test_client.full_pipeline_calls = full_pipeline_calls
        yield test_client
    finally:
        app.dependency_overrides.clear()


@pytest.fixture(scope="module")
def registry_entries():
    get_registry.cache_clear()
    return list(get_registry().models)


def _required_response_fields() -> set[str]:
    return {
        "id",
        "name",
        "task_type",
        "dataset",
        "description",
        "metrics",
        "pipeline_config",
        "model_overrides",
        "checkpoint_refs",
        "requirements",
        "status",
        "runnable_locally",
        "notebook_or_kernel_ref",
        "provenance",
        "tombstone",
        "missing_checkpoints",
    }


@pytest.mark.parametrize("entry", get_registry().models, ids=lambda entry: entry.id)
def test_get_models_includes_registry_entry(client: TestClient, entry) -> None:
    response = client.get("/api/models", params={"include_dead_ends": True})

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    ids = {item["id"] for item in payload["data"]}
    assert entry.id in ids


@pytest.mark.parametrize("entry", get_registry().models, ids=lambda entry: entry.id)
def test_get_model_detail_returns_expected_fields(client: TestClient, entry) -> None:
    response = client.get(f"/api/models/{entry.id}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    data = payload["data"]
    assert data["id"] == entry.id
    assert data["name"] == entry.name
    assert data["task_type"] == entry.task_type
    assert data["dataset"] == entry.dataset
    assert data["status"] == entry.status
    assert data["runnable_locally"] is entry.runnable_locally
    assert _required_response_fields().issubset(data)


@pytest.mark.parametrize("entry", get_registry().models, ids=lambda entry: entry.id)
def test_pipeline_run_respects_registry_runnable_policy(client: TestClient, entry) -> None:
    response = client.post(
        "/api/pipeline/run",
        json={"runId": f"e2e-{entry.id}", "model_id": entry.id},
    )

    if entry.runnable_locally:
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["model_id"] == entry.id
        assert data["resolved_config"] == entry.pipeline_config
        assert data["applied_overrides"] == entry.model_overrides
        assert client.full_pipeline_calls
        return

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert entry.id in detail
    assert "not runnable" in detail
    assert (entry.notebook_or_kernel_ref or "no Kaggle kernel reference recorded") in detail
    assert client.full_pipeline_calls == []