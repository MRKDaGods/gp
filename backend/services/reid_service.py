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
from backend.models.reid import FusionReIDResponse, ReIDComponentResult, ReIDQueryResult, ReIDRankedMatch, SingleCamReIDResponse
from backend.models.requests import FusionReIDModelWeight, ReIDImageInput, SingleCamReIDRequest
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
        model_entry = get_model(model_id)
        self._validate_model_entry(model_id, model_entry)

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
        except ValueError as exc:
            raise ReIDServiceError(str(exc), status_code=422, code="unsupported_reid_model") from exc
        except ReIDServiceError:
            raise
        except Exception as exc:  # noqa: BLE001 - sanitize API response at the router boundary
            raise ReIDServiceError(str(exc), status_code=500, code="inference_failed") from exc

        scores, ranking_values, rank_by_distance = self._score_features(
            query_features,
            gallery_features,
            normalize=normalize,
            rerank=rerank,
            aqe_k=aqe_k,
        )

        results = self._build_results(queries, gallery, scores, ranking_values, rank_by_distance, top_k)

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

    def fusion_reid(
        self,
        query_images: list[ReIDImageInput],
        gallery_images: list[ReIDImageInput],
        models: list[FusionReIDModelWeight],
        rerank: bool,
        aqe_k: int,
        top_k: int = 20,
        normalize: bool = True,
        warnings: list[str] | None = None,
    ) -> FusionReIDResponse:
        started = time.perf_counter()
        if len(models) < 2:
            raise ReIDServiceError("fusion_requires_at_least_two_models", status_code=422, code="fusion_requires_at_least_two_models")

        for model in models:
            self._validate_model_entry(model.model_id, get_model(model.model_id))

        self._check_request_size(query_images, gallery_images)
        queries = self._decode_images(query_images, prefix="q")
        gallery = self._decode_images(gallery_images, prefix="g")
        device = self._select_device()

        weights = np.asarray([model.weight for model in models], dtype=np.float32)
        weights = weights / weights.sum()
        model_ids = [model.model_id for model in models]
        per_model_scores: list[np.ndarray] = []
        per_model_all_similarity: list[np.ndarray] = []
        components: list[ReIDComponentResult] = []

        for model_spec, weight in zip(models, weights):
            try:
                loaded_model = load_reid_model(model_spec.model_id, device)
                query_features = extract_features(loaded_model, [item.image for item in queries])
                gallery_features = extract_features(loaded_model, [item.image for item in gallery])
            except FileNotFoundError as exc:
                raise ReIDServiceError(str(exc), status_code=503, code="checkpoint_missing") from exc
            except ValueError as exc:
                raise ReIDServiceError(str(exc), status_code=422, code="unsupported_reid_model") from exc
            except ReIDServiceError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise ReIDServiceError(str(exc), status_code=500, code="inference_failed") from exc

            q_model, g_model, all_model = self._prepare_query_gallery_features(
                query_features,
                gallery_features,
                normalize=normalize,
                aqe_k=aqe_k,
            )
            score_matrix = np.clip(q_model @ g_model.T, -1.0, 1.0)
            all_similarity = np.clip(all_model @ all_model.T, -1.0, 1.0)
            self._assert_finite_scores(score_matrix)
            per_model_scores.append(score_matrix)
            per_model_all_similarity.append(all_similarity)
            component_results = self._build_results(
                queries,
                gallery,
                score_matrix,
                score_matrix,
                False,
                top_k,
            )
            components.append(
                ReIDComponentResult(
                    modelId=model_spec.model_id,
                    weight=float(weight),
                    featureDim=int(q_model.shape[1]),
                    results=component_results,
                )
            )

        fused_scores = np.zeros_like(per_model_scores[0], dtype=np.float32)
        fused_all_similarity = np.zeros_like(per_model_all_similarity[0], dtype=np.float32)
        for weight, score_matrix, all_similarity in zip(weights, per_model_scores, per_model_all_similarity):
            fused_scores += float(weight) * score_matrix.astype(np.float32, copy=False)
            fused_all_similarity += float(weight) * all_similarity.astype(np.float32, copy=False)
        fused_scores = np.clip(fused_scores, -1.0, 1.0)
        self._assert_finite_scores(fused_scores)

        if rerank:
            rerank_dist = self._rerank_distances_from_similarity(
                fused_all_similarity,
                query_count=len(queries),
            )
            ranking_values = rerank_dist
            response_scores = np.clip(1.0 - rerank_dist, -1.0, 1.0)
            rank_by_distance = True
        else:
            ranking_values = fused_scores
            response_scores = fused_scores
            rank_by_distance = False

        results = self._build_results(queries, gallery, response_scores, ranking_values, rank_by_distance, top_k)
        return FusionReIDResponse(
            success=True,
            modelIds=model_ids,
            weights=[float(weight) for weight in weights],
            device=device,
            queryCount=len(queries),
            galleryCount=len(gallery),
            results=results,
            components=components,
            warnings=warnings or [],
            latencyMs=(time.perf_counter() - started) * 1000.0,
        )

    def _validate_model_entry(self, model_id: str, model_entry: Any) -> None:
        if model_entry is None:
            raise ReIDServiceError("model_not_found", status_code=404, code="model_not_found")
        if model_entry.status == "dead_end":
            raise ReIDServiceError("dead_end_model_not_served", status_code=422, code="dead_end_model_not_served")
        if model_entry.task_type != "single_cam_reid":
            raise ReIDServiceError("unsupported_task_type", status_code=422, code="unsupported_task_type")
        if not model_id:
            raise ReIDServiceError("model_not_found", status_code=404, code="model_not_found")

    def _score_features(
        self,
        query_features: np.ndarray,
        gallery_features: np.ndarray,
        *,
        normalize: bool,
        rerank: bool,
        aqe_k: int,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        q_feats, g_feats, all_features = self._prepare_query_gallery_features(
            query_features,
            gallery_features,
            normalize=normalize,
            aqe_k=aqe_k,
        )
        scores = np.clip(q_feats @ g_feats.T, -1.0, 1.0)
        self._assert_finite_scores(scores)
        if not rerank:
            return scores, scores, False
        all_similarity = np.clip(all_features @ all_features.T, -1.0, 1.0)
        rerank_dist = self._rerank_distances_from_similarity(all_similarity, query_count=q_feats.shape[0])
        return np.clip(1.0 - rerank_dist, -1.0, 1.0), rerank_dist, True

    def _prepare_query_gallery_features(
        self,
        query_features: np.ndarray,
        gallery_features: np.ndarray,
        *,
        normalize: bool,
        aqe_k: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        q_feats = np.asarray(query_features, dtype=np.float32)
        g_feats = np.asarray(gallery_features, dtype=np.float32)
        if normalize:
            q_feats = self._l2_normalize(q_feats)
            g_feats = self._l2_normalize(g_feats)
        all_features = np.concatenate([q_feats, g_feats], axis=0)
        if aqe_k > 0:
            all_features = self._average_query_expansion(all_features, aqe_k)
        elif normalize:
            all_features = self._l2_normalize(all_features)
        query_count = q_feats.shape[0]
        return all_features[:query_count], all_features[query_count:], all_features

    def _build_results(
        self,
        queries: list[DecodedImage],
        gallery: list[DecodedImage],
        response_scores: np.ndarray,
        ranking_values: np.ndarray,
        rank_by_distance: bool,
        top_k: int,
    ) -> list[ReIDQueryResult]:
        results: list[ReIDQueryResult] = []
        limit = min(int(top_k), len(gallery))
        for query_index, query in enumerate(queries):
            query_started = time.perf_counter()
            if rank_by_distance:
                order = np.argsort(ranking_values[query_index], kind="mergesort")[:limit]
            else:
                order = np.argsort(-ranking_values[query_index], kind="mergesort")[:limit]
            matches = [
                ReIDRankedMatch(
                    galleryId=gallery[gallery_index].id,
                    rank=rank,
                    score=float(response_scores[query_index, gallery_index]),
                    distance=float(1.0 - response_scores[query_index, gallery_index]),
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
        return results

    @staticmethod
    def _assert_finite_scores(scores: np.ndarray) -> None:
        if not np.all(np.isfinite(scores)):
            raise ReIDServiceError("Non-finite ReID score produced", status_code=500, code="non_finite_scores")

    def _rerank_distances_from_similarity(self, similarity: np.ndarray, *, query_count: int) -> np.ndarray:
        original_dist, initial_rank = self._build_rerank_state_from_similarity(similarity, max_k1=80)
        return self._compute_reranking(
            original_dist,
            initial_rank,
            query_num=query_count,
            k1=80,
            k2=15,
            lambda_value=0.2,
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

    def _average_query_expansion(self, features: np.ndarray, k: int, iterations: int = 1) -> np.ndarray:
        current = self._l2_normalize(features)
        if k <= 1:
            return current
        for _ in range(iterations):
            sim = current @ current.T
            topk = min(int(k), sim.shape[1])
            kth = max(topk - 1, 0)
            topk_idx = np.argpartition(-sim, kth=kth, axis=1)[:, :topk]
            expanded = np.zeros_like(current, dtype=np.float32)
            for index in range(current.shape[0]):
                expanded[index] = current[topk_idx[index]].mean(axis=0)
            current = self._l2_normalize(expanded)
        return current

    @staticmethod
    def _build_rerank_state_from_similarity(similarity: np.ndarray, max_k1: int) -> tuple[np.ndarray, np.ndarray]:
        sim = np.asarray(similarity, dtype=np.float32)
        original_dist = (1.0 - sim).astype(np.float32, copy=False)
        k = min(int(max_k1) + 1, sim.shape[1])
        initial_rank = np.argsort(-sim, axis=1)[:, :k].astype(np.int32, copy=False)
        return original_dist, initial_rank

    @staticmethod
    def _compute_reranking(
        original_dist: np.ndarray,
        initial_rank: np.ndarray,
        query_num: int,
        k1: int = 80,
        k2: int = 15,
        lambda_value: float = 0.2,
    ) -> np.ndarray:
        all_num = original_dist.shape[0]
        effective_k1 = min(int(k1), max(1, initial_rank.shape[1] - 1))
        effective_k2 = min(int(k2), initial_rank.shape[1])
        V = np.zeros((all_num, all_num), dtype=np.float16)
        half_k1 = int(np.round(effective_k1 / 2.0))

        for index in range(all_num):
            forward = initial_rank[index, : effective_k1 + 1]
            backward = initial_rank[forward, : effective_k1 + 1]
            reciprocal = forward[np.any(backward == index, axis=1)]
            reciprocal_expansion = reciprocal.copy()

            for candidate in reciprocal:
                candidate_forward = initial_rank[candidate, : half_k1 + 1]
                candidate_backward = initial_rank[candidate_forward, : half_k1 + 1]
                candidate_reciprocal = candidate_forward[np.any(candidate_backward == candidate, axis=1)]
                if candidate_reciprocal.size == 0:
                    continue
                overlap = np.intersect1d(candidate_reciprocal, reciprocal)
                if overlap.size > (2.0 / 3.0) * candidate_reciprocal.size:
                    reciprocal_expansion = np.concatenate((reciprocal_expansion, candidate_reciprocal))

            reciprocal_expansion = np.unique(reciprocal_expansion)
            weights = np.exp(-original_dist[index, reciprocal_expansion]).astype(np.float32)
            V[index, reciprocal_expansion] = (weights / (weights.sum() + 1e-12)).astype(np.float16)

        if effective_k2 > 1:
            V_qe = np.zeros_like(V, dtype=np.float16)
            for index in range(all_num):
                V_qe[index] = V[initial_rank[index, :effective_k2]].mean(axis=0)
            V = V_qe

        inv_index = [np.flatnonzero(V[:, column]) for column in range(all_num)]
        jaccard_dist = np.zeros((query_num, all_num), dtype=np.float32)

        for index in range(query_num):
            temp_min = np.zeros(all_num, dtype=np.float32)
            non_zero = np.flatnonzero(V[index])
            for nz in non_zero:
                related = inv_index[nz]
                temp_min[related] += np.minimum(np.float32(V[index, nz]), V[related, nz].astype(np.float32))
            jaccard_dist[index] = 1.0 - temp_min / (2.0 - temp_min)

        final_dist = jaccard_dist * (1.0 - lambda_value) + original_dist[:query_num] * lambda_value
        return final_dist[:, query_num:]
