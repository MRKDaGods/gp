from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.app import app
from backend.dependencies import get_app_state
from backend.routers import pipeline as pipeline_router
from backend.state import AppState


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch):
    test_state = AppState()

    async def noop_execute_full_pipeline(*args, **kwargs):
        return None

    monkeypatch.setattr(pipeline_router, "execute_full_pipeline", noop_execute_full_pipeline)
    app.dependency_overrides[get_app_state] = lambda: test_state
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


def test_pipeline_run_with_vehicle_model_id_resolves_cityflow_config(client: TestClient) -> None:
    response = client.post(
        "/api/pipeline/run",
        json={"runId": "test-vehicle-model", "model_id": "vehicle_mtmc_14e_b1"},
    )

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["model_id"] == "vehicle_mtmc_14e_b1"
    assert data["dataset"] == "cityflowv2"
    assert data["resolved_config"] == "configs/datasets/cityflowv2.yaml"
    assert "stage4.association.query_expansion.k=2" in data["applied_overrides"]
    assert "stage4.association.graph.similarity_threshold=0.48" in data["applied_overrides"]
    assert data["warnings"] == []


def test_pipeline_run_with_person_model_id_resolves_wildtrack_config(client: TestClient) -> None:
    response = client.post(
        "/api/pipeline/run",
        json={"runId": "test-person-model", "model_id": "person_mtmc_12b"},
    )

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["model_id"] == "person_mtmc_12b"
    assert data["dataset"] == "wildtrack"
    assert data["resolved_config"] == "configs/datasets/wildtrack.yaml"
    assert "stage1.tracker.min_hits=2" in data["applied_overrides"]


def test_pipeline_run_dataset_only_keeps_backward_compatible_resolution(client: TestClient) -> None:
    response = client.post(
        "/api/pipeline/run",
        json={"runId": "test-dataset-only", "dataset": "cityflowv2"},
    )

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["model_id"] is None
    assert data["dataset"] == "cityflowv2"
    assert data["resolved_config"] == "configs/datasets/cityflowv2.yaml"
    assert data["applied_overrides"] == []
    assert data["warnings"] == []


def test_pipeline_run_model_id_wins_over_conflicting_dataset(client: TestClient) -> None:
    response = client.post(
        "/api/pipeline/run",
        json={
            "runId": "test-conflict-model-wins",
            "dataset": "wildtrack",
            "model_id": "vehicle_mtmc_14e_b1",
        },
    )

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["model_id"] == "vehicle_mtmc_14e_b1"
    assert data["dataset"] == "cityflowv2"
    assert data["resolved_config"] == "configs/datasets/cityflowv2.yaml"
    assert data["warnings"]
    assert "overrides requested dataset 'wildtrack'" in data["warnings"][0]


def test_pipeline_run_unknown_model_id_returns_400(client: TestClient) -> None:
    response = client.post(
        "/api/pipeline/run",
        json={"runId": "test-unknown-model", "model_id": "not_a_model"},
    )

    assert response.status_code == 400
    assert "Unknown model_id" in response.json()["detail"]


def test_pipeline_run_rejects_non_local_model_with_kernel_hint(client: TestClient) -> None:
    response = client.post(
        "/api/pipeline/run",
        json={"runId": "test-nonlocal-model", "model_id": "veri776_14t_fusion"},
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "not runnable" in detail
    assert "yahiaakhalafallah/14t-veri-fusion-clip-senet-x-transreid" in detail


def test_pipeline_run_rejects_task_type_mismatch(client: TestClient) -> None:
    response = client.post(
        "/api/pipeline/run",
        json={"runId": "test-task-mismatch", "model_id": "person_detector_12a_mvdetr"},
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "task_type 'detector_only'" in detail
    assert "not compatible" in detail
