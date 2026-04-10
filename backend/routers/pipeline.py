import asyncio
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter, BackgroundTasks, Body, HTTPException, WebSocket

from backend.config import OUTPUT_DIR
from backend.models.requests import PipelineRunRequest
from backend.services.pipeline_service import (
    _resolve_run_id,
    _write_run_context,
    execute_full_pipeline,
    execute_stage,
)
from backend.services.video_service import _detect_camera_for_video
from backend.state import active_runs, uploaded_videos

router = APIRouter()


@router.post("/api/pipeline/run-stage/{stage}")
async def run_stage(
    stage: int,
    background_tasks: BackgroundTasks,
    request: Optional[PipelineRunRequest] = Body(default=None),
):
    """Run a specific pipeline stage"""
    try:
        print(f"\n[UI Request] Run Stage {stage} Payload: {request.dict() if request else 'None'}")
        payload = request or PipelineRunRequest()
        requested_run_id = payload.runId or (payload.config or {}).get("runId")
        run_id = _resolve_run_id(str(requested_run_id) if requested_run_id is not None else None)

        config = payload.config or {}
        video_id = payload.videoId or config.get("videoId")
        camera_id = payload.cameraId or config.get("cameraId")
        smoke_test = bool(payload.smokeTest or config.get("smokeTest", False))
        use_cpu = bool(payload.useCpu or config.get("useCpu", False))

        if stage == 1 and not video_id:
            raise HTTPException(status_code=400, detail="videoId is required for stage 1")

        resolved_camera_id = None
        video_name = None
        if video_id:
            if video_id not in uploaded_videos:
                raise HTTPException(status_code=404, detail="Video not found")
            resolved_camera_id = _detect_camera_for_video(uploaded_videos[video_id], camera_id)
            video_name = uploaded_videos[video_id].get("name")

        dataset_name = config.get("datasetName") or None

        active_runs[run_id] = {
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
        }

        _write_run_context(
            run_id,
            {
                "source": "pipeline-run-stage",
                "stage": stage,
                "videoId": video_id,
                "cameraId": resolved_camera_id,
                "datasetName": dataset_name,
            },
        )

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
            },
        )
        return {"success": True, "data": active_runs[run_id]}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/pipeline/run")
async def run_full_pipeline(background_tasks: BackgroundTasks):
    """Run full pipeline"""
    run_id = _resolve_run_id(None)
    active_runs[run_id] = {
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
    }
    _write_run_context(run_id, {"source": "pipeline-run-full"})
    background_tasks.add_task(execute_full_pipeline, run_id, {})
    return {"success": True, "data": active_runs[run_id]}


@router.get("/api/pipeline/status/{run_id}")
async def get_pipeline_status(run_id: str):
    """Get pipeline execution status"""
    if run_id not in active_runs:
        raise HTTPException(status_code=404, detail="Run not found")
    return {"success": True, "data": active_runs[run_id]}


@router.post("/api/pipeline/cancel/{run_id}")
async def cancel_pipeline(run_id: str):
    """Cancel pipeline execution"""
    if run_id not in active_runs:
        raise HTTPException(status_code=404, detail="Run not found")
    active_runs[run_id]["status"] = "cancelled"
    return {"success": True, "data": None}


@router.websocket("/api/ws/pipeline/{run_id}")
async def websocket_pipeline_updates(websocket: WebSocket, run_id: str):
    """WebSocket for pipeline progress updates"""
    await websocket.accept()
    try:
        while True:
            if run_id in active_runs:
                await websocket.send_json(active_runs[run_id])
            await asyncio.sleep(1)
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()
