import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, BackgroundTasks, Body, Depends, HTTPException, WebSocket

from backend.dependencies import get_app_state
from backend.models.requests import PipelineRunRequest, StageExecutionTarget
from backend.services.kaggle_run_service import (
    dispatch_stage_to_kaggle,
    get_kaggle_job_state,
    refresh_kaggle_job_status,
)
from backend.services.kaggle_service import (
    KaggleAuthError,
    KaggleConcurrencyError,
    KaggleValidationError,
)
from backend.services.pipeline_service import (
    PipelineModelValidationError,
    _resolve_run_id,
    _write_run_context,
    execute_full_pipeline,
    execute_stage,
    resolve_pipeline_model,
)
from backend.services.video_service import _detect_camera_for_video
from backend.state import AppState

router = APIRouter()


def _payload_model_id(payload: PipelineRunRequest, config: Dict[str, Any]) -> Optional[str]:
    return payload.model_id or config.get("model_id") or config.get("modelId")


def _payload_dataset(payload: PipelineRunRequest, config: Dict[str, Any]) -> Optional[str]:
    return payload.dataset or config.get("dataset") or config.get("datasetName")


def _get_user_video_path(video_id: Optional[str], state: AppState) -> Optional[Path]:
    if not video_id:
        return None
    record = state.uploaded_videos.get(video_id)
    if not record:
        return None
    path = record.get("path")
    return Path(path) if path else None


@router.post("/api/pipeline/run-stage/{stage}")
@router.post("/pipeline/run-stage/{stage}")
async def run_stage(
    stage: int,
    background_tasks: BackgroundTasks,
    request: Optional[PipelineRunRequest] = Body(default=None),
    state: AppState = Depends(get_app_state),
):
    """Run a specific pipeline stage"""
    try:
        request_dump = request.model_dump() if request else "None"
        print(f"\n[UI Request] Run Stage {stage} Payload: {request_dump}")
        payload = request or PipelineRunRequest()
        requested_run_id = payload.runId or (payload.config or {}).get("runId")
        run_id = _resolve_run_id(str(requested_run_id) if requested_run_id is not None else None)

        config = payload.config or {}
        video_id = payload.videoId or config.get("videoId")
        camera_id = payload.cameraId or config.get("cameraId")
        smoke_test = bool(payload.smokeTest or config.get("smokeTest", False))
        use_cpu = bool(payload.useCpu or config.get("useCpu", False))
        resolution = resolve_pipeline_model(
            model_id=_payload_model_id(payload, config),
            dataset=_payload_dataset(payload, config),
            fusion=payload.fusion,
        )

        kaggle_dataset_input = bool(
            payload.kaggle
            and payload.kaggle.target == StageExecutionTarget.KAGGLE
            and payload.kaggle.dataset_slug
        )
        if stage == 1 and not video_id and not kaggle_dataset_input:
            raise HTTPException(status_code=400, detail="videoId is required for stage 1")

        resolved_camera_id = None
        video_name = None
        if video_id:
            if video_id not in state.uploaded_videos:
                raise HTTPException(status_code=404, detail="Video not found")
            resolved_camera_id = _detect_camera_for_video(
                state.uploaded_videos[video_id],
                camera_id,
            )
            video_name = state.uploaded_videos[video_id].get("name")

        dataset_name = config.get("datasetName") or None

        state.active_runs[run_id] = {
            "id": run_id,
            "runId": run_id,
            "stage": stage,
            "status": "running",
            "progress": 0,
            "message": "Queued",
            "startedAt": datetime.now().isoformat(),
            "videoId": video_id,
            "videoName": dataset_name or video_name,
            "cameraId": resolved_camera_id,
            "smokeTest": smoke_test,
            "useCpu": use_cpu,
            "dataset": resolution.dataset,
            "model_id": resolution.model_id,
            "resolved_config": resolution.resolved_config,
            "applied_overrides": resolution.applied_overrides,
            "warnings": resolution.warnings,
            "fusion_resolved": resolution.fusion_resolved,
        }

        _write_run_context(
            run_id,
            {
                "source": "pipeline-run-stage",
                "stage": stage,
                "videoId": video_id,
                "cameraId": resolved_camera_id,
                "datasetName": dataset_name,
                "dataset": resolution.dataset,
                "model_id": resolution.model_id,
                "resolved_config": resolution.resolved_config,
                "applied_overrides": resolution.applied_overrides,
                "warnings": resolution.warnings,
                "fusion_resolved": resolution.fusion_resolved,
            },
        )

        if payload.kaggle and payload.kaggle.target == StageExecutionTarget.KAGGLE:
            try:
                result = dispatch_stage_to_kaggle(
                    run_id=run_id,
                    stages=[stage],
                    config_path=resolution.resolved_config,
                    config_overrides=resolution.applied_overrides,
                    model_id=resolution.model_id,
                    fusion=payload.fusion.model_dump(by_alias=True) if payload.fusion else None,
                    kaggle_cfg=payload.kaggle,
                    user_video_path=_get_user_video_path(video_id, state),
                )
            except KaggleAuthError as e:
                raise HTTPException(status_code=401, detail=str(e))
            except KaggleConcurrencyError as e:
                raise HTTPException(status_code=429, detail=str(e))
            except KaggleValidationError as e:
                raise HTTPException(status_code=400, detail=str(e))

            state.active_runs[run_id].update(
                {
                    "status": "queued",
                    "message": "Kaggle kernel queued",
                    "execution_target": "kaggle",
                    "kaggle": {
                        "kernel_slug": result.kernel_slug,
                        "kernel_url": result.kernel_url,
                        "dataset_slug": result.dataset_slug,
                        "project_dataset_slug": result.project_dataset_slug,
                        "status": result.status,
                    },
                }
            )
            return {"success": True, "data": state.active_runs[run_id]}

        reid_model_path = config.get("reid_model_path") or None
        tracker = config.get("tracker") or None

        background_tasks.add_task(
            execute_stage,
            run_id,
            stage,
            {
                "videoId": video_id,
                "cameraId": resolved_camera_id,
                "smokeTest": smoke_test,
                "useCpu": use_cpu,
                "reidModelPath": reid_model_path,
                "tracker": tracker,
                "dataset": resolution.dataset,
                "resolvedConfig": resolution.resolved_config,
                "appliedOverrides": resolution.applied_overrides,
                "fusionResolved": resolution.fusion_resolved,
            },
        )
        return {"success": True, "data": state.active_runs[run_id]}
    except PipelineModelValidationError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/pipeline/run")
async def run_full_pipeline(
    background_tasks: BackgroundTasks,
    request: Optional[PipelineRunRequest] = Body(default=None),
    state: AppState = Depends(get_app_state),
):
    """Run full pipeline"""
    try:
        payload = request or PipelineRunRequest()
        config = payload.config or {}
        resolution = resolve_pipeline_model(
            model_id=_payload_model_id(payload, config),
            dataset=_payload_dataset(payload, config),
            fusion=payload.fusion,
        )
    except PipelineModelValidationError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    run_id = _resolve_run_id(payload.runId or config.get("runId"))
    video_id = payload.videoId or config.get("videoId")
    camera_id = payload.cameraId or config.get("cameraId")
    smoke_test = bool(payload.smokeTest or config.get("smokeTest", False))
    use_cpu = bool(payload.useCpu or config.get("useCpu", False))

    state.active_runs[run_id] = {
        "id": run_id,
        "runId": run_id,
        "status": "running",
        "progress": 0,
        "currentStage": 0,
        "stages": [
            {"stage": i, "status": "pending", "progress": 0, "message": f"Stage {i}"}
            for i in range(7)
        ],
        "startedAt": datetime.now().isoformat(),
        "videoId": video_id,
        "cameraId": camera_id,
        "smokeTest": smoke_test,
        "useCpu": use_cpu,
        "dataset": resolution.dataset,
        "model_id": resolution.model_id,
        "resolved_config": resolution.resolved_config,
        "applied_overrides": resolution.applied_overrides,
        "warnings": resolution.warnings,
        "fusion_resolved": resolution.fusion_resolved,
    }
    _write_run_context(
        run_id,
        {
            "source": "pipeline-run-full",
            "dataset": resolution.dataset,
            "model_id": resolution.model_id,
            "resolved_config": resolution.resolved_config,
            "applied_overrides": resolution.applied_overrides,
            "warnings": resolution.warnings,
            "fusion_resolved": resolution.fusion_resolved,
        },
    )
    background_tasks.add_task(
        execute_full_pipeline,
        run_id,
        {
            "videoId": video_id,
            "cameraId": camera_id,
            "smokeTest": smoke_test,
            "useCpu": use_cpu,
            "dataset": resolution.dataset,
            "resolvedConfig": resolution.resolved_config,
            "appliedOverrides": resolution.applied_overrides,
            "fusionResolved": resolution.fusion_resolved,
            "reidModelPath": config.get("reid_model_path") or None,
        },
    )
    return {"success": True, "data": state.active_runs[run_id]}


@router.get("/api/pipeline/status/{run_id}")
async def get_pipeline_status(run_id: str, state: AppState = Depends(get_app_state)):
    """Get pipeline execution status"""
    if run_id not in state.active_runs:
        raise HTTPException(status_code=404, detail="Run not found")
    return {"success": True, "data": state.active_runs[run_id]}


@router.get("/api/pipeline/kaggle-status/{run_id}")
@router.get("/pipeline/kaggle-status/{run_id}")
async def get_kaggle_status(run_id: str):
    state = get_kaggle_job_state(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"No Kaggle job found for run_id {run_id}")
    try:
        refreshed = refresh_kaggle_job_status(run_id)
    except KaggleAuthError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except KaggleValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"success": True, "data": refreshed}


@router.post("/api/pipeline/cancel/{run_id}")
async def cancel_pipeline(run_id: str, state: AppState = Depends(get_app_state)):
    """Cancel pipeline execution"""
    if run_id not in state.active_runs:
        raise HTTPException(status_code=404, detail="Run not found")
    state.active_runs[run_id]["status"] = "cancelled"
    return {"success": True, "data": None}


@router.websocket("/api/ws/pipeline/{run_id}")
async def websocket_pipeline_updates(
    websocket: WebSocket,
    run_id: str,
    state: AppState = Depends(get_app_state),
):
    """WebSocket for pipeline progress updates"""
    await websocket.accept()
    try:
        while True:
            if run_id in state.active_runs:
                await websocket.send_json(state.active_runs[run_id])
            await asyncio.sleep(1)
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()
