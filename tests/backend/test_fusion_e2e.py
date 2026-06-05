from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.app import app
from backend.dependencies import get_app_state
from backend.models.requests import FusionConfig, FusionModel
from backend.routers import pipeline as pipeline_router
from backend.services.pipeline_service import resolve_pipeline_model
from backend.state import AppState


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    test_state = AppState()

    async def noop_execute_stage(*args, **kwargs):
        return None

    monkeypatch.setattr(pipeline_router, "execute_stage", noop_execute_stage)
    app.dependency_overrides[get_app_state] = lambda: test_state
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


def _data(body: dict) -> dict:
    return body.get("data", body)


def test_fusion_request_full_flow(client: TestClient) -> None:
    """POST /api/pipeline/run-stage/2 resolves canonical fusion metadata."""
    resp = client.post(
        "/api/pipeline/run-stage/2",
        json={
            "runId": "test_fusion_e2e",
            "videoId": None,
            "fusion": {
                "models": [
                    {"model_id": "veri776_09v_v17_transreid", "weight": 0.30},
                    {"model_id": "veri776_clipsenet_v6", "weight": 0.70},
                ],
                "aqe_k": 3,
                "k1": 80,
                "k2": 15,
                "lambda": 0.2,
                "rerank": True,
            },
        },
    )
    body = resp.json()
    print(body)

    assert resp.status_code == 200
    data = _data(body)
    fr = data["fusion_resolved"]
    assert fr["aqe_k"] == 3
    assert any(
        model["model_id"] == "veri776_clipsenet_v6" and model["primary"] is True
        for model in fr["models"]
    )
    weight_sum = sum(model["weight"] for model in fr["models"])
    assert 0.99 <= weight_sum <= 1.01


def test_fusion_request_3_models(client: TestClient) -> None:
    """3-model fusion produces secondary and tertiary Stage 2/4 overrides."""
    resp = client.post(
        "/api/pipeline/run-stage/2",
        json={
            "runId": "test_fusion_3",
            "fusion": {
                "models": [
                    {"model_id": "veri776_09v_v17_transreid", "weight": 0.30},
                    {"model_id": "veri776_clipsenet_v6", "weight": 0.50},
                    {"model_id": "vehicle_mtmc_14e_b1", "weight": 0.20},
                ],
                "aqe_k": 3,
                "k1": 80,
                "k2": 15,
                "lambda": 0.2,
                "rerank": True,
            },
        },
    )
    body = resp.json()
    print(body)

    assert resp.status_code == 200
    overrides = _data(body)["applied_overrides"]
    overrides_str = " ".join(overrides)
    assert "stage2.reid.vehicle2" in overrides_str
    assert "stage2.reid.vehicle3" in overrides_str
    assert "stage4.association.secondary_embeddings" in overrides_str
    assert "stage4.association.tertiary_embeddings" in overrides_str


def test_bundled_model_resolves_fusion_via_endpoint(client: TestClient) -> None:
    """A registered bundled-fusion model (single modelId) resolves dynamic
    Stage-4 ensemble paths + fusion_resolved through the run-stage endpoint."""
    resp = client.post(
        "/api/pipeline/run-stage/4",
        json={"runId": "test_bundled_14e", "modelId": "vehicle_mtmc_14e_b1"},
    )
    body = resp.json()
    assert resp.status_code == 200, body
    data = _data(body)
    overrides = data["applied_overrides"]
    overrides_str = " ".join(overrides)
    # Dynamic, run-scoped tertiary path (NOT the stale run_latest baked in YAML).
    assert (
        "stage4.association.tertiary_embeddings.path="
        "${project.output_dir}/${project.run_name}/stage2/embeddings_tertiary.npy"
        in overrides
    )
    assert "run_latest" not in overrides_str
    assert data["fusion_resolved"] is not None
    assert data["fusion_resolved"]["mode"] == "bundled"


def test_fusion_request_invalid_model_id(client: TestClient) -> None:
    """Unknown model_id should be rejected cleanly."""
    resp = client.post(
        "/api/pipeline/run-stage/2",
        json={
            "runId": "test_fusion_invalid",
            "fusion": {
                "models": [
                    {"model_id": "veri776_09v_v17_transreid", "weight": 0.30},
                    {"model_id": "this_model_does_not_exist_in_registry", "weight": 0.70},
                ],
                "aqe_k": 3,
                "k1": 80,
                "k2": 15,
                "lambda": 0.2,
                "rerank": True,
            },
        },
    )

    assert resp.status_code >= 400, (
        f"Expected error for invalid model_id, got {resp.status_code}: {resp.json()}"
    )


def test_fusion_resolved_overrides_snapshot() -> None:
    """Lock down the exact override list for 14t canonical fusion."""
    fusion = FusionConfig(
        models=[
            FusionModel(model_id="veri776_09v_v17_transreid", weight=0.30),
            FusionModel(model_id="veri776_clipsenet_v6", weight=0.70),
        ],
        aqe_k=3,
        k1=80,
        k2=15,
        **{"lambda": 0.2},
        rerank=True,
    )

    result = resolve_pipeline_model(model_id=None, dataset=None, fusion=fusion)
    overrides = result.applied_overrides

    assert result.model_id == "veri776_clipsenet_v6"
    assert any("stage2.reid.vehicle.model_name=clip_senet" in override for override in overrides)
    assert any("vehicle2" in override and "transreid" in override.lower() for override in overrides), (
        f"vehicle2 transreid override missing. Got: {overrides}"
    )
    assert any(
        "stage4.association.secondary_embeddings.enabled=true" in override.lower()
        for override in overrides
    ), f"secondary_embeddings.enabled missing. Got: {overrides}"
    assert any("query_expansion.k=3" in override for override in overrides)
    assert any("reranking.enabled=true" in override.lower() for override in overrides)

    print("FUSION OVERRIDES SNAPSHOT:")
    for override in overrides:
        print(f"  {override}")