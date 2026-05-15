from __future__ import annotations

import copy

import pytest
from fastapi.testclient import TestClient
from jsonschema import ValidationError

from backend.app import app
from backend.services.model_registry import get_registry, list_models, validate_registry_data


REQUIRED_MODEL_IDS = {
    "vehicle_mtmc_14e_b1",
    "vehicle_mtmc_14k_v1_k7",
    "person_mtmc_12b",
    "person_detector_12a_mvdetr",
    "veri776_14t_fusion",
    "veri776_09v_v17_transreid",
    "veri776_clipsenet_v6",
}


def test_registry_loads_and_validates() -> None:
    get_registry.cache_clear()
    registry = get_registry()

    assert registry.version == 1
    assert len(registry.models) >= 9
    assert REQUIRED_MODEL_IDS.issubset({model.id for model in registry.models})


def test_required_entries_have_required_fields() -> None:
    registry = get_registry()
    entries = {model.id: model for model in registry.models}

    for model_id in REQUIRED_MODEL_IDS:
        entry = entries[model_id]
        assert entry.name
        assert entry.description
        assert entry.task_type
        assert entry.dataset
        assert entry.metrics
        assert entry.requirements is not None
        assert entry.provenance.verified_by


def test_filter_by_task_type_works() -> None:
    models = list_models(task_type="single_cam_reid")

    assert models
    assert {model.task_type for model in models} == {"single_cam_reid"}
    assert "veri776_14t_fusion" in {model.id for model in models}


def test_filter_by_status_works() -> None:
    models = list_models(status="production")

    assert models
    assert {model.status for model in models} == {"production"}
    assert "vehicle_mtmc_14e_b1" in {model.id for model in models}


def test_dead_ends_hidden_by_default() -> None:
    default_ids = {model.id for model in list_models()}
    all_ids = {model.id for model in list_models(include_dead_ends=True)}

    assert "deadend_vehicle_csls" not in default_ids
    assert "deadend_vehicle_aflink" not in default_ids
    assert "deadend_vehicle_csls" in all_ids
    assert "deadend_vehicle_aflink" in all_ids


def test_get_api_models_returns_expected_shape() -> None:
    client = TestClient(app)

    response = client.get("/api/models")

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert isinstance(payload["data"], list)
    assert payload["data"]
    first = payload["data"][0]
    assert {"id", "name", "task_type", "metrics", "status", "missing_checkpoints"}.issubset(first)
    assert all(model["status"] != "dead_end" for model in payload["data"])


def test_get_api_models_filters_and_includes_dead_ends() -> None:
    client = TestClient(app)

    response = client.get("/api/models", params={"task_type": "mtmc_vehicle", "include_dead_ends": True})

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["data"]
    assert all(model["task_type"] == "mtmc_vehicle" for model in payload["data"])
    assert any(model["status"] == "dead_end" for model in payload["data"])


def test_get_api_model_by_id_and_404() -> None:
    client = TestClient(app)

    response = client.get("/api/models/vehicle_mtmc_14e_b1")
    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["data"]["id"] == "vehicle_mtmc_14e_b1"

    missing = client.get("/api/models/not_a_model")
    assert missing.status_code == 404


def test_schema_rejects_invalid_registry() -> None:
    registry = get_registry()
    invalid = copy.deepcopy(registry.model_dump())
    invalid["models"][0]["unexpected"] = "nope"

    with pytest.raises(ValidationError):
        validate_registry_data(invalid)
