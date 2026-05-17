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


def test_single_cam_reid_happy_path(monkeypatch) -> None:
    calls = {"extract": 0}

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

    def fake_extract_features(_loaded_model, images):
        calls["extract"] += 1
        if calls["extract"] == 1:
            assert len(images) == 1
            return np.asarray([[1.0, 0.0]], dtype=np.float32)
        assert len(images) == 2
        return np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)

    monkeypatch.setattr("backend.services.reid_service.load_reid_model", fake_load_reid_model)
    monkeypatch.setattr("backend.services.reid_service.extract_features", fake_extract_features)

    client = TestClient(app)
    response = client.post(
        "/api/v1/reid/single_cam",
        json={
            "modelId": "veri776_09v_v17_transreid",
            "queries": [{"id": "q0", "image_base64": _png_b64((255, 0, 0))}],
            "gallery": [
                {"id": "g0", "image_base64": _png_b64((0, 255, 0))},
                {"id": "g1", "image_base64": _png_b64((255, 0, 0))},
            ],
            "topK": 2,
            "rerank": False,
            "aqeK": 0,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["modelId"] == "veri776_09v_v17_transreid"
    assert payload["featureDim"] == 2
    assert payload["queryCount"] == 1
    assert payload["galleryCount"] == 2
    matches = payload["results"][0]["matches"]
    assert [match["galleryId"] for match in matches] == ["g1", "g0"]
    assert matches[0]["score"] == 1.0
    assert matches[1]["score"] == 0.0


def test_single_cam_reid_cityflow_transreid_happy_path(monkeypatch) -> None:
    seen_model_ids = []
    calls = {"extract": 0}

    def fake_load_reid_model(model_id: str, device: str) -> LoadedReIDModel:
        seen_model_ids.append(model_id)
        return LoadedReIDModel(
            model_id=model_id,
            model=object(),
            device=device,
            checkpoint_path=__file__,
            feature_dim=768,
            loader="transreid_cityflow",
            loaded_at=0.0,
        )

    def fake_extract_features(_loaded_model, images):
        calls["extract"] += 1
        assert len(images) == 1
        features = np.zeros((1, 768), dtype=np.float32)
        if calls["extract"] == 1:
            features[0, 0] = 1.0
        else:
            features[0, 0] = 0.25
            features[0, 1] = 0.75
        return features

    monkeypatch.setattr("backend.services.reid_service.load_reid_model", fake_load_reid_model)
    monkeypatch.setattr("backend.services.reid_service.extract_features", fake_extract_features)

    client = TestClient(app)
    response = client.post(
        "/api/v1/reid/single_cam",
        json={
            "modelId": "cityflow_transreid",
            "queries": [{"id": "q0", "image_base64": _png_b64((255, 0, 0))}],
            "gallery": [{"id": "g0", "image_base64": _png_b64((255, 0, 0))}],
            "topK": 1,
            "rerank": False,
            "aqeK": 0,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["modelId"] == "cityflow_transreid"
    assert payload["featureDim"] == 768
    assert payload["queryCount"] == 1
    assert payload["galleryCount"] == 1
    assert seen_model_ids == ["cityflow_transreid"]
    assert calls["extract"] == 2


def test_single_cam_reid_model_not_found() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/v1/reid/single_cam",
        json={
            "modelId": "fake",
            "queries": [{"id": "q0", "image_base64": _png_b64((255, 0, 0))}],
            "gallery": [{"id": "g0", "image_base64": _png_b64((255, 0, 0))}],
        },
    )

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "model_not_found"


def test_single_cam_reid_rejects_path_traversal() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/v1/reid/single_cam",
        json={
            "modelId": "veri776_09v_v17_transreid",
            "queries": [{"id": "q0", "path": "../etc/passwd"}],
            "gallery": [{"id": "g0", "image_base64": _png_b64((255, 0, 0))}],
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "invalid_image_input"
