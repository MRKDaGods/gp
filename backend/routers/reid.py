"""ReID serving API routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from backend.models.reid import FusionReIDResponse, SingleCamReIDResponse
from backend.models.requests import FusionReIDModelWeight, FusionReIDRequest, SingleCamReIDRequest

router = APIRouter(prefix="/api/v1", tags=["reid"])
_reid_service: Any | None = None


def _get_reid_service() -> Any:
    global _reid_service
    if _reid_service is None:
        try:
            from backend.services.reid_service import ReIDService
        except ModuleNotFoundError as exc:
            if exc.name in {"torch", "torchvision"}:
                raise HTTPException(
                    status_code=503,
                    detail={
                        "code": "reid_dependency_missing",
                        "message": f"ReID inference requires optional dependency: {exc.name}",
                    },
                ) from exc
            raise
        _reid_service = ReIDService()
    return _reid_service


def _raise_service_http_error(exc: Exception) -> None:
    status_code = getattr(exc, "status_code", None)
    code = getattr(exc, "code", None)
    if status_code is None or code is None:
        raise exc
    raise HTTPException(status_code=status_code, detail={"code": code, "message": str(exc)}) from exc


@router.post("/reid/single_cam", response_model=SingleCamReIDResponse, response_model_by_alias=True)
def single_cam_reid(request: SingleCamReIDRequest) -> SingleCamReIDResponse:
    try:
        reid_service = _get_reid_service()
        return reid_service.single_cam_reid(
            query_images=request.queries,
            gallery_images=request.gallery,
            model_id=request.model_id,
            rerank=request.rerank,
            aqe_k=request.aqe_k,
            top_k=request.top_k,
            normalize=request.normalize,
        )
    except HTTPException:
        raise
    except Exception as exc:
        _raise_service_http_error(exc)
        raise


@router.post("/reid/fusion", response_model=FusionReIDResponse, response_model_by_alias=True)
def fusion_reid(request: FusionReIDRequest) -> FusionReIDResponse:
    warnings: list[str] = []
    weight_sum = sum(model.weight for model in request.models)
    models = request.models
    if abs(weight_sum - 1.0) > 1e-6:
        warnings.append(f"Fusion weights were normalized from sum={weight_sum:.6g} to 1.0")
        models = [FusionReIDModelWeight(modelId=model.model_id, weight=model.weight / weight_sum) for model in request.models]

    try:
        reid_service = _get_reid_service()
        return reid_service.fusion_reid(
            query_images=request.queries,
            gallery_images=request.gallery,
            models=models,
            rerank=request.rerank,
            aqe_k=request.aqe_k,
            top_k=request.top_k,
            normalize=request.normalize,
            warnings=warnings,
        )
    except HTTPException:
        raise
    except Exception as exc:
        _raise_service_http_error(exc)
        raise
