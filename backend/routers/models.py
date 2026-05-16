"""Read-only model registry API routes."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from backend.models.registry import ModelDetailResponse, ModelListResponse
from backend.services.model_registry import get_model, list_models

router = APIRouter()


@router.get("/api/models", response_model=ModelListResponse)
def get_models(
    task_type: Optional[str] = Query(default=None),
    dataset: Optional[str] = Query(default=None),
    status: Optional[str] = Query(default=None),
    include_dead_ends: bool = Query(default=False),
) -> ModelListResponse:
    """Return registry entries, hiding dead-end tombstones by default."""
    return ModelListResponse(
        success=True,
        data=list_models(
            task_type=task_type,
            dataset=dataset,
            status=status,
            include_dead_ends=include_dead_ends,
        ),
    )


@router.get("/api/models/{model_id}", response_model=ModelDetailResponse)
def get_model_entry(model_id: str) -> ModelDetailResponse:
    """Return a single registry entry by stable model id."""
    model = get_model(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return ModelDetailResponse(success=True, data=model)
