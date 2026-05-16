"""Single-camera ReID serving service."""

from __future__ import annotations

import base64
import binascii
import os
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from backend.config import UPLOAD_DIR
from backend.models.reid import ReIDQueryResult, ReIDRankedMatch, SingleCamReIDResponse
from backend.models.requests import ReIDImageInput, SingleCamReIDRequest
from backend.services.model_registry import PROJECT_ROOT, get_model, get_registry
from src.serving.reid_loaders import extract_features, load_reid_model

MAX_BASE64_BYTES_PER_IMAGE = 25 * 1024 * 1024
MAX_IMAGES_PER_REQUEST = 50
DATA_UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"


class ReIDServiceError(Exception):
    status_code = 500
    code = "inference_failed"

    def __init__(self, message: str, *, status_code: int | None = None, code: str | None = None) -> None:
        super().__init__(message)
        if status_code is not None:
            self.status_code = status_code
        if code is not None:
            self.code = code


class PayloadTooLargeError(ReIDServiceError):
    status_code = 413
    code = "payload_too_large"


class InvalidImageInputError(ReIDServiceError):
    status_code = 400
    code = "invalid_image_input"


@dataclass(frozen=True)
class DecodedImage:
    id: str
    image: Image.Image
    source: str
    metadata: dict[str, Any]


class ReIDService:
    def __init__(self, *, upload_roots: Iterable[Path] | None = None) -> None:
        roots = list(upload_roots or [DATA_UPLOAD_DIR, PROJECT_ROOT / UPLOAD_DIR])
        roots.extend(self._registered_checkpoint_dirs())
        self.upload_roots = [root.resolve() for root in roots]

    def single_cam_reid(
        self,
        query_images: list[ReIDImageInput],
        gallery_images: list[ReIDImageInput],
        model_id: str,
        rerank: bool,
        aqe_k: int,
        top_k: int = 20,
        normalize: bool = True,
    ) -> SingleCamReIDResponse:
        started = time.perf_counter()
        if rerank:
            raise ReIDServiceError("rerank is not supported by the Phase 2a serving endpoint", status_code=422, code="rerank_not_supported")
        if aqe_k != 0:
            raise ReIDServiceError("aqe_k is not supported by the Phase 2a serving endpoint", status_code=422, code="aqe_not_supported")

        model_entry = get_model(model_id)
        if model_entry is None:
            raise ReIDServiceError("model_not_found", status_code=404, code="model_not_found")
        if model_entry.status == "dead_end":
            raise ReIDServiceError("dead_end_model_not_served", status_code=422, code="dead_end_model_not_served")
        if model_entry.task_type != "single_cam_reid":
            raise ReIDServiceError("unsupported_task_type", status_code=422, code="unsupported_task_type")

        self._check_request_size(query_images, gallery_images)
        queries = self._decode_images(query_images, prefix="q")
        gallery = self._decode_images(gallery_images, prefix="g")
        device = self._select_device()

        try:
            loaded_model = load_reid_model(model_id, device)
            query_features = extract_features(loaded_model, [item.image for item in queries])
            gallery_features = extract_features(loaded_model, [item.image for item in gallery])
        except FileNotFoundError as exc:
            raise ReIDServiceError(str(exc), status_code=503, code="checkpoint_missing") from exc
        except ReIDServiceError:
            raise
        except Exception as exc:  # noqa: BLE001 - sanitize API response at the router boundary
            raise ReIDServiceError(str(exc), status_code=500, code="inference_failed") from exc

        if normalize:
            query_features = self._l2_normalize(query_features)
            gallery_features = self._l2_normalize(gallery_features)
        scores = query_features @ gallery_features.T
        if not np.all(np.isfinite(scores)):
            raise ReIDServiceError("Non-finite ReID score produced", status_code=500, code="non_finite_scores")
        scores = np.clip(scores, -1.0, 1.0)

        results: list[ReIDQueryResult] = []
        limit = min(int(top_k), len(gallery))
        for query_index, query in enumerate(queries):
            query_started = time.perf_counter()
            order = np.argsort(-scores[query_index], kind="mergesort")[:limit]
            matches = [
                ReIDRankedMatch(
                    galleryId=gallery[gallery_index].id,
                    rank=rank,
                    score=float(scores[query_index, gallery_index]),
                    distance=float(1.0 - scores[query_index, gallery_index]),
                    metadata=gallery[gallery_index].metadata,
                )
                for rank, gallery_index in enumerate(order, start=1)
            ]
            results.append(
                ReIDQueryResult(
                    queryId=query.id,
                    matches=matches,
                    latencyMs=(time.perf_counter() - query_started) * 1000.0,
                )
            )

        return SingleCamReIDResponse(
            success=True,
            modelId=model_id,
            device=loaded_model.device,
            featureDim=int(query_features.shape[1]),
            queryCount=len(queries),
            galleryCount=len(gallery),
            results=results,
            latencyMs=(time.perf_counter() - started) * 1000.0,
        )

    def _check_request_size(self, query_images: list[ReIDImageInput], gallery_images: list[ReIDImageInput]) -> None:
        total = len(query_images) + len(gallery_images)
        if total > MAX_IMAGES_PER_REQUEST:
            raise PayloadTooLargeError(f"At most {MAX_IMAGES_PER_REQUEST} images are accepted per request")
        for image_input in [*query_images, *gallery_images]:
            if image_input.image_base64 is None:
                continue
            encoded = self._strip_data_url_prefix(image_input.image_base64)
            estimated_bytes = (len(encoded) * 3) // 4
            if estimated_bytes > MAX_BASE64_BYTES_PER_IMAGE:
                raise PayloadTooLargeError("Base64 image payload exceeds 25MB")

    def _decode_images(self, image_inputs: list[ReIDImageInput], *, prefix: str) -> list[DecodedImage]:
        decoded: list[DecodedImage] = []
        for index, image_input in enumerate(image_inputs):
            image_id = image_input.id or f"{prefix}{index}"
            if image_input.image_base64 is not None:
                image = self._decode_base64_image(image_input.image_base64)
                decoded.append(DecodedImage(image_id, image, "base64", dict(image_input.metadata)))
            elif image_input.path is not None:
                path = self._resolve_safe_path(image_input.path)
                try:
                    image = Image.open(path).convert("RGB")
                except (OSError, UnidentifiedImageError) as exc:
                    raise InvalidImageInputError(f"Could not load image path: {image_input.path}") from exc
                decoded.append(DecodedImage(image_id, image, str(path), dict(image_input.metadata)))
            else:
                raise InvalidImageInputError("Each ReID image requires image_base64 or path")
        return decoded

    def _decode_base64_image(self, value: str) -> Image.Image:
        encoded = self._strip_data_url_prefix(value)
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise InvalidImageInputError("Invalid base64 image payload") from exc
        if len(raw) > MAX_BASE64_BYTES_PER_IMAGE:
            raise PayloadTooLargeError("Base64 image payload exceeds 25MB")
        try:
            return Image.open(BytesIO(raw)).convert("RGB")
        except (OSError, UnidentifiedImageError) as exc:
            raise InvalidImageInputError("Base64 payload is not a supported image") from exc

    @staticmethod
    def _strip_data_url_prefix(value: str) -> str:
        if "," in value and value.lstrip().lower().startswith("data:"):
            return value.split(",", 1)[1]
        return value

    def _resolve_safe_path(self, raw_path: str) -> Path:
        candidate = Path(raw_path)
        if ".." in candidate.parts:
            raise InvalidImageInputError("path_traversal_rejected")
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        resolved = candidate.resolve()
        if not any(self._is_relative_to(resolved, root) for root in self.upload_roots):
            raise InvalidImageInputError("path_outside_upload_dir")
        return resolved

    @staticmethod
    def _is_relative_to(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

    @staticmethod
    def _select_device() -> str:
        use_cpu = os.getenv("USE_CPU", "").strip().lower() in {"1", "true", "yes", "on"}
        if use_cpu or not torch.cuda.is_available():
            return "cpu"
        return "cuda:0"

    @staticmethod
    def _registered_checkpoint_dirs() -> list[Path]:
        roots: list[Path] = []
        try:
            registry = get_registry()
        except Exception:
            return roots
        for model in registry.models:
            for checkpoint in model.checkpoint_refs:
                roots.append((PROJECT_ROOT / checkpoint.local_path).parent)
        return roots

    @staticmethod
    def _l2_normalize(features: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        array = np.asarray(features, dtype=np.float32)
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        return array / np.maximum(norms, eps)
