"""ReID serving API routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.models.reid import SingleCamReIDResponse
from backend.models.requests import SingleCamReIDRequest
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
