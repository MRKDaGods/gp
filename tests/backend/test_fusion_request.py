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
    stage_calls = []

    async def noop_execute_full_pipeline(*args, **kwargs):
        return None

    async def noop_execute_stage(*args, **kwargs):
        stage_calls.append((args, kwargs))
        return None

    monkeypatch.setattr(pipeline_router, "execute_full_pipeline", noop_execute_full_pipeline)
    monkeypatch.setattr(pipeline_router, "execute_stage", noop_execute_stage)
    app.dependency_overrides[get_app_state] = lambda: test_state
    try:
        test_client = TestClient(app)
        test_client.stage_calls = stage_calls
        yield test_client
    finally:
        app.dependency_overrides.clear()


def _fusion_payload(models: list[dict[str, object]]) -> dict[str, object]:
    return {
        "runId": "test-fusion",
        "dataset": "cityflowv2",
        "fusion": {
            "models": models,
            "aqe_k": 3,
            "k1": 80,
            "k2": 15,
            "lambda": 0.2,
            "rerank": True,
        },
    }


def test_valid_two_model_fusion_produces_secondary_overrides(client: TestClient) -> None:
    response = client.post(
        "/api/pipeline/run-stage/2",
        json=_fusion_payload(
            [
                {"model_id": "vehicle_mtmc_14e_b1", "weight": 0.7},
                {"model_id": "veri776_09v_v17_transreid", "weight": 0.3},
            ]
        ),
    )

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["model_id"] == "vehicle_mtmc_14e_b1"
    assert data["fusion_resolved"]["primary_model_id"] == "vehicle_mtmc_14e_b1"
    assert data["fusion_resolved"]["models"][1]["model_id"] == "veri776_09v_v17_transreid"

    overrides = data["applied_overrides"]
    assert "stage2.reid.vehicle2.enabled=true" in overrides
    assert "stage2.reid.vehicle2.save_separate=true" in overrides
    assert "stage2.reid.vehicle2.model_name=transreid" in overrides
    assert "stage2.reid.vehicle2.weights_path=models/reid/vehicle_transreid_vit_base_veri776.pth" in overrides
    assert "stage4.association.secondary_embeddings.enabled=true" in overrides
    assert "stage4.association.secondary_embeddings.weight=0.3" in overrides
    assert "stage4.association.query_expansion.k=3" in overrides
    assert "stage4.association.reranking.enabled=true" in overrides
    assert "stage4.association.reranking.k1=80" in overrides
    assert "stage4.association.reranking.k2=15" in overrides
    assert "stage4.association.reranking.lambda_value=0.2" in overrides

    assert client.stage_calls
    args, _ = client.stage_calls[-1]
    assert args[2]["fusionResolved"] == data["fusion_resolved"]


def test_valid_three_model_fusion_produces_secondary_and_tertiary_overrides(client: TestClient) -> None:
    response = client.post(
        "/api/pipeline/run-stage/2",
        json=_fusion_payload(
            [
                {"model_id": "vehicle_mtmc_14e_b1", "weight": 0.5},
                {"model_id": "veri776_09v_v17_transreid", "weight": 0.3},
                {"model_id": "veri776_clipsenet_v6", "weight": 0.2},
            ]
        ),
    )

    assert response.status_code == 200
    data = response.json()["data"]
    overrides = data["applied_overrides"]

    assert "stage2.reid.vehicle2.enabled=true" in overrides
    assert "stage2.reid.vehicle3.enabled=true" in overrides
    assert "stage2.reid.vehicle3.model_name=clip_senet" in overrides
    assert "stage2.reid.vehicle3.weights_path=models/reid/clipsenet_v6_veri776_best.pth" in overrides
    assert "stage4.association.secondary_embeddings.enabled=true" in overrides
    assert "stage4.association.secondary_embeddings.weight=0.3" in overrides
    assert "stage4.association.tertiary_embeddings.enabled=true" in overrides
    assert "stage4.association.tertiary_embeddings.weight=0.2" in overrides
    assert data["fusion_resolved"]["models"][2]["role"] == "tertiary"


def test_unknown_model_id_in_fusion_models_returns_400(client: TestClient) -> None:
    response = client.post(
        "/api/pipeline/run-stage/2",
        json=_fusion_payload(
            [
                {"model_id": "vehicle_mtmc_14e_b1", "weight": 0.7},
                {"model_id": "not_a_model", "weight": 0.3},
            ]
        ),
    )

    assert response.status_code == 400
    assert "Unknown model_id in fusion.models" in response.json()["detail"]


def test_fusion_model_missing_architecture_returns_clear_error(client: TestClient) -> None:
    response = client.post(
        "/api/pipeline/run-stage/2",
        json=_fusion_payload(
            [
                {"model_id": "vehicle_mtmc_14e_b1", "weight": 0.7},
                {"model_id": "vehicle_mtmc_14k_v1_k7", "weight": 0.3},
            ]
        ),
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "missing the 'architecture' block in model_registry.yaml" in detail
    assert "Fusion requires arch metadata" in detail


def test_fusion_weights_outside_tolerance_returns_validation_error(client: TestClient) -> None:
    response = client.post(
        "/api/pipeline/run-stage/2",
        json=_fusion_payload(
            [
                {"model_id": "vehicle_mtmc_14e_b1", "weight": 0.25},
                {"model_id": "veri776_09v_v17_transreid", "weight": 0.25},
            ]
        ),
    )

    assert response.status_code == 422
    assert "weights must sum to 1.0" in response.text


def test_fusion_weights_inside_tolerance_are_accepted_and_normalized(client: TestClient) -> None:
    response = client.post(
        "/api/pipeline/run-stage/2",
        json=_fusion_payload(
            [
                {"model_id": "vehicle_mtmc_14e_b1", "weight": 0.701},
                {"model_id": "veri776_09v_v17_transreid", "weight": 0.300},
            ]
        ),
    )

    assert response.status_code == 200
    models = response.json()["data"]["fusion_resolved"]["models"]
    assert sum(model["weight"] for model in models) == pytest.approx(1.0)
    assert models[0]["weight"] == pytest.approx(0.701 / 1.001)
    assert models[1]["weight"] == pytest.approx(0.300 / 1.001)


def test_one_model_fusion_returns_validation_error(client: TestClient) -> None:
    response = client.post(
        "/api/pipeline/run-stage/2",
        json=_fusion_payload([
            {"model_id": "vehicle_mtmc_14e_b1", "weight": 1.0},
        ]),
    )

    assert response.status_code == 422


def test_four_model_fusion_returns_validation_error(client: TestClient) -> None:
    response = client.post(
        "/api/pipeline/run-stage/2",
        json=_fusion_payload(
            [
                {"model_id": "vehicle_mtmc_14e_b1", "weight": 0.4},
                {"model_id": "veri776_09v_v17_transreid", "weight": 0.2},
                {"model_id": "veri776_clipsenet_v6", "weight": 0.2},
                {"model_id": "cityflow_transreid", "weight": 0.2},
            ]
        ),
    )

    assert response.status_code == 422


def test_duplicate_model_ids_return_validation_error(client: TestClient) -> None:
    response = client.post(
        "/api/pipeline/run-stage/2",
        json=_fusion_payload(
            [
                {"model_id": "vehicle_mtmc_14e_b1", "weight": 0.5},
                {"model_id": "vehicle_mtmc_14e_b1", "weight": 0.5},
            ]
        ),
    )

    assert response.status_code == 422
    assert "Fusion model IDs must be unique" in response.text


def test_single_mode_request_without_fusion_keeps_existing_resolution(client: TestClient) -> None:
    response = client.post(
        "/api/pipeline/run-stage/4",
        json={"runId": "test-single-mode", "model_id": "vehicle_mtmc_14e_b1"},
    )

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["model_id"] == "vehicle_mtmc_14e_b1"
    assert data["dataset"] == "cityflowv2"
    assert data["resolved_config"] == "configs/datasets/cityflowv2.yaml"
    assert "stage4.association.query_expansion.k=2" in data["applied_overrides"]
    assert data["warnings"] == []
    assert data["fusion_resolved"] is None
