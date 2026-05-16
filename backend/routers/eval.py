"""Evaluation job submission and polling routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from backend.models.reid import EvalJobResponse, EvalJobResultResponse, EvalJobStatusResponse
from backend.services.eval_service import EVAL_SPECS, eval_service, summarize_eval_result
from backend.services.job_service import get_result, get_status

router = APIRouter(prefix="/api/v1/eval", tags=["eval"])


class EvalRunRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    eval_type: str = Field(alias="evalType")
    config_overrides: dict[str, Any] = Field(default_factory=dict, alias="configOverrides")


@router.post("/run", response_model=EvalJobResponse, response_model_by_alias=True)
def run_eval(request: EvalRunRequest, background_tasks: BackgroundTasks) -> EvalJobResponse:
    if request.eval_type not in EVAL_SPECS:
        raise HTTPException(status_code=422, detail={"code": "unsupported_eval_type", "message": "Unsupported eval_type"})
    try:
        job_id = eval_service.submit_eval(request.eval_type, request.config_overrides, background_tasks=background_tasks)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={"code": "invalid_eval_request", "message": str(exc)}) from exc
    return EvalJobResponse(jobId=job_id, status="queued")


@router.get("/{job_id}/status", response_model=EvalJobStatusResponse, response_model_by_alias=True)
def eval_status(job_id: str) -> EvalJobStatusResponse:
    status = get_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail={"code": "job_not_found", "message": "Eval job not found"})
    return EvalJobStatusResponse.model_validate(status)


@router.get("/{job_id}", response_model=EvalJobStatusResponse, response_model_by_alias=True)
def eval_status_alias(job_id: str) -> EvalJobStatusResponse:
    return eval_status(job_id)


@router.get("/{job_id}/result", response_model=EvalJobResultResponse, response_model_by_alias=True)
def eval_result(job_id: str) -> EvalJobResultResponse:
    result = get_result(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail={"code": "job_not_found", "message": "Eval job not found"})
    status = str(result.get("status"))
    if status in {"queued", "running"}:
        raise HTTPException(status_code=409, detail={"code": "job_not_finished", "message": "Eval job is not finished"})
    if status == "failed":
        raise HTTPException(status_code=500, detail={"code": "job_failed", "message": "Eval job failed"})

    payload = result.get("result") or {}
    if isinstance(payload, dict) and "summary" in payload and "result" in payload:
        response_result = payload
    else:
        response_result = {"summary": summarize_eval_result(payload if isinstance(payload, dict) else {"value": payload}), "result": payload}
    return EvalJobResultResponse(jobId=job_id, status=status, result=response_result)