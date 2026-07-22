"""Model lifecycle over HTTP: list/inspect for everyone with access,
promote/retire/rollback for admins only — always eval-gated (D5)."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, status

from athar.api import audit
from athar.api.deps import AdminUser, DbDep, RequireViewer, ServicesDep
from athar.api.schemas import PromoteRequest, RollbackRequest
from athar.serving.lifecycle import LifecycleError, ModelNotFound
from athar.serving.registry import EvalReportRef, ModelEntry, ModelStage, ModelTask

router = APIRouter(prefix="/models", tags=["models"], dependencies=[RequireViewer])


@router.get("")
def list_models(
    services: ServicesDep, task: Optional[str] = None, stage: Optional[str] = None
) -> list[ModelEntry]:
    try:
        task_filter = ModelTask(task) if task else None
        stage_filter = ModelStage(stage) if stage else None
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from None
    return services.lifecycle.list(task=task_filter, stage=stage_filter)


@router.get("/{model_id}")
def get_model(services: ServicesDep, model_id: str) -> ModelEntry:
    try:
        return services.lifecycle.get(model_id)
    except ModelNotFound:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"model {model_id!r} not found") from None


@router.get("/{model_id}/history")
def model_history(services: ServicesDep, model_id: str) -> list[dict]:
    try:
        services.lifecycle.get(model_id)
    except ModelNotFound:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"model {model_id!r} not found") from None
    return services.lifecycle.events(model_id)


@router.post("/{model_id}/promote")
def promote_model(
    body: PromoteRequest, services: ServicesDep, db: DbDep, user: AdminUser,
    model_id: str,
) -> ModelEntry:
    report = None
    if body.eval_run_id:
        report = EvalReportRef(
            run_id=body.eval_run_id, benchmark=body.benchmark, metrics=body.metrics
        )
    try:
        entry = services.lifecycle.promote(
            model_id, ModelStage(body.to),
            eval_report=report, actor=user.username, notes=body.notes,
        )
    except ModelNotFound:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"model {model_id!r} not found") from None
    except (LifecycleError, ValueError) as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from None
    audit.append(db, user.username, "model_promoted",
                 model_id=model_id, to=body.to, eval_run_id=body.eval_run_id)
    return entry


@router.post("/{model_id}/retire")
def retire_model(
    services: ServicesDep, db: DbDep, user: AdminUser, model_id: str
) -> ModelEntry:
    try:
        entry = services.lifecycle.retire(model_id, actor=user.username)
    except ModelNotFound:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"model {model_id!r} not found") from None
    audit.append(db, user.username, "model_retired", model_id=model_id)
    return entry


@router.post("/rollback")
def rollback_production(
    body: RollbackRequest, services: ServicesDep, db: DbDep, user: AdminUser
) -> ModelEntry:
    try:
        entry = services.lifecycle.rollback(ModelTask(body.task), actor=user.username)
    except (LifecycleError, ValueError) as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from None
    audit.append(db, user.username, "model_rollback", task=body.task,
                 production_now=entry.model_id if entry.stage is ModelStage.PRODUCTION else None)
    return entry
