"""ReID serving API routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.models.reid import FusionReIDResponse, SingleCamReIDResponse
from backend.models.requests import FusionReIDModelWeight, FusionReIDRequest, SingleCamReIDRequest
from backend.services.reid_service import ReIDService, ReIDServiceError

router = APIRouter(prefix="/api/v1", tags=["reid"])
reid_service = ReIDService()


@router.post("/reid/single_cam", response_model=SingleCamReIDResponse, response_model_by_alias=True)
def single_cam_reid(request: SingleCamReIDRequest) -> SingleCamReIDResponse:
    try:
        return reid_service.single_cam_reid(
            query_images=request.queries,
            gallery_images=request.gallery,
            model_id=request.model_id,
            rerank=request.rerank,
            aqe_k=request.aqe_k,
            top_k=request.top_k,
            normalize=request.normalize,
        )
    except ReIDServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail={"code": exc.code, "message": str(exc)}) from exc


@router.post("/reid/fusion", response_model=FusionReIDResponse, response_model_by_alias=True)
def fusion_reid(request: FusionReIDRequest) -> FusionReIDResponse:
    warnings: list[str] = []
    weight_sum = sum(model.weight for model in request.models)
    models = request.models
    if abs(weight_sum - 1.0) > 1e-6:
        warnings.append(f"Fusion weights were normalized from sum={weight_sum:.6g} to 1.0")
        models = [FusionReIDModelWeight(modelId=model.model_id, weight=model.weight / weight_sum) for model in request.models]

    try:
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
    except ReIDServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail={"code": exc.code, "message": str(exc)}) from exc
