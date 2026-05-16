from __future__ import annotations

import base64
from io import BytesIO

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from backend.app import app
from src.serving.reid_loaders import LoadedReIDModel


def _png_b64(color: tuple[int, int, int]) -> str:
    image = Image.new("RGB", (32, 32), color=color)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _install_fake_reid(monkeypatch) -> None:
    def fake_load_reid_model(model_id: str, device: str) -> LoadedReIDModel:
        return LoadedReIDModel(
            model_id=model_id,
            model=object(),
            device=device,
            checkpoint_path=__file__,
            feature_dim=2,
            loader="fake",
            loaded_at=0.0,
        )

    def fake_extract_features(loaded_model: LoadedReIDModel, images):
        if len(images) == 1:
            return np.asarray([[1.0, 0.0]], dtype=np.float32)
        if loaded_model.model_id == "veri776_09v_v17_transreid":
            return np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        return np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    monkeypatch.setattr("backend.services.reid_service.load_reid_model", fake_load_reid_model)
    monkeypatch.setattr("backend.services.reid_service.extract_features", fake_extract_features)


def _fusion_payload(weights: tuple[float, float] = (0.75, 0.25), *, rerank: bool = False, aqe_k: int = 0) -> dict:
    return {
        "models": [
            {"modelId": "veri776_09v_v17_transreid", "weight": weights[0]},
            {"modelId": "veri776_clipsenet_v6", "weight": weights[1]},
        ],
        "queries": [{"id": "q0", "image_base64": _png_b64((255, 0, 0))}],
        "gallery": [
            {"id": "g0", "image_base64": _png_b64((0, 255, 0))},
            {"id": "g1", "image_base64": _png_b64((255, 0, 0))},
        ],
        "topK": 2,
        "rerank": rerank,
        "aqeK": aqe_k,
    }


def test_fusion_reid_happy_path_with_two_models(monkeypatch) -> None:
    _install_fake_reid(monkeypatch)

    response = TestClient(app).post("/api/v1/reid/fusion", json=_fusion_payload())

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["modelIds"] == ["veri776_09v_v17_transreid", "veri776_clipsenet_v6"]
    assert payload["weights"] == [0.75, 0.25]
    assert [match["galleryId"] for match in payload["results"][0]["matches"]] == ["g1", "g0"]
    assert len(payload["components"]) == 2


def test_fusion_reid_rejects_single_model() -> None:
    payload = _fusion_payload()
    payload["models"] = payload["models"][:1]

    response = TestClient(app).post("/api/v1/reid/fusion", json=payload)

    assert response.status_code == 422


def test_fusion_reid_auto_normalizes_weights(monkeypatch) -> None:
    _install_fake_reid(monkeypatch)

    response = TestClient(app).post("/api/v1/reid/fusion", json=_fusion_payload(weights=(3.0, 1.0)))

    assert response.status_code == 200
    payload = response.json()
    assert payload["weights"] == [0.75, 0.25]
    assert payload["warnings"]


def test_fusion_reid_runs_with_rerank_and_aqe(monkeypatch) -> None:
    _install_fake_reid(monkeypatch)

    response = TestClient(app).post("/api/v1/reid/fusion", json=_fusion_payload(rerank=True, aqe_k=2))

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert len(payload["results"][0]["matches"]) == 2


def test_fusion_reid_rejects_unserved_reid_model(monkeypatch) -> None:
    def fake_load_reid_model(model_id: str, device: str) -> LoadedReIDModel:
        if model_id == "veri776_14t_fusion":
            raise ValueError("No Phase 2b serving loader is registered for veri776_14t_fusion")
        return LoadedReIDModel(
            model_id=model_id,
            model=object(),
            device=device,
            checkpoint_path=__file__,
            feature_dim=2,
            loader="fake",
            loaded_at=0.0,
        )

    monkeypatch.setattr("backend.services.reid_service.load_reid_model", fake_load_reid_model)
    payload = _fusion_payload()
    payload["models"][1]["modelId"] = "veri776_14t_fusion"

    response = TestClient(app).post("/api/v1/reid/fusion", json=payload)

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "unsupported_reid_model"