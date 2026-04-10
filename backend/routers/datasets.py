import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from backend.config import DATASET_DIR, OUTPUT_DIR, VIDEO_EXTENSIONS
from backend.dependencies import get_app_state
from backend.services.pipeline_service import (
    _execute_dataset_pipeline,
    _resolve_run_id,
    _write_run_context,
)
from backend.state import AppState

router = APIRouter()


@router.get("/api/datasets")
async def list_datasets(state: AppState = Depends(get_app_state)):
    """List available dataset folders under dataset/ with camera info."""
    results: List[Dict[str, Any]] = []
    if not DATASET_DIR.exists():
        return {"success": True, "data": results}

    for folder in sorted(DATASET_DIR.iterdir()):
        if not folder.is_dir():
            continue
        cameras: List[Dict[str, Any]] = []
        for cam_dir in sorted(folder.iterdir()):
            if not cam_dir.is_dir():
                continue
            has_video = any((cam_dir / f"vdo{ext}").exists() for ext in VIDEO_EXTENSIONS)
            cameras.append({"id": cam_dir.name, "hasVideo": has_video})
        dataset_key = folder.name.lower()
        candidate_runs: List[tuple] = []
        if OUTPUT_DIR.exists():
            for run_dir in OUTPUT_DIR.iterdir():
                if not run_dir.is_dir():
                    continue
                run_id = run_dir.name

                matched = False
                if run_id == f"dataset_precompute_{dataset_key}":
                    matched = True

                if not matched:
                    ctx_path = run_dir / "run_context.json"
                    if ctx_path.exists():
                        try:
                            ctx = json.loads(ctx_path.read_text(encoding="utf-8"))
                            if str(ctx.get("source", "")).startswith("dataset") and str(
                                ctx.get("datasetFolder", "")
                            ).lower() == dataset_key:
                                matched = True
                        except Exception:
                            pass

                if matched:
                    candidate_runs.append((run_dir.stat().st_mtime, run_id, run_dir))

        candidate_runs.sort(key=lambda x: x[0], reverse=True)
        latest_run_id = candidate_runs[0][1] if candidate_runs else None
        latest_run_dir = candidate_runs[0][2] if candidate_runs else None

        already_processed = False
        has_gallery = False
        if latest_run_dir is not None:
            already_processed = (latest_run_dir / "stage1").exists() and any(
                (latest_run_dir / "stage1").glob("tracklets_*.json")
            )
            has_gallery = (
                already_processed
                and (latest_run_dir / "stage2" / "embeddings.npy").exists()
                and (latest_run_dir / "stage2" / "embedding_index.json").exists()
                and (latest_run_dir / "stage4" / "global_trajectories.json").exists()
            )

        is_processing = any(
            r.get("status") == "running"
            and str(r.get("datasetFolder", "")).lower() == dataset_key
            for r in state.active_runs.values()
        )

        results.append(
            {
                "name": folder.name,
                "path": str(folder),
                "cameras": cameras,
                "cameraCount": len(cameras),
                "videosFound": sum(1 for c in cameras if c["hasVideo"]),
                "alreadyProcessed": already_processed,
                "hasGallery": has_gallery,
                "isProcessing": is_processing,
                "runId": latest_run_id
                if (latest_run_id and (already_processed or is_processing))
                else None,
                "galleryRunId": latest_run_id if (latest_run_id and has_gallery) else None,
            }
        )

    return {"success": True, "data": results}


@router.post("/api/datasets/{folder}/process")
async def process_dataset(folder: str, background_tasks: BackgroundTasks, state: AppState = Depends(get_app_state)):
    """Trigger full pipeline (stages 0-4) on a dataset folder."""
    dataset_path = DATASET_DIR / folder
    if not dataset_path.exists() or not dataset_path.is_dir():
        raise HTTPException(status_code=404, detail=f"Dataset folder '{folder}' not found")

    run_id = _resolve_run_id(None)

    for run in state.active_runs.values():
        if run.get("status") == "running" and str(run.get("datasetFolder", "")).lower() == folder.lower():
            return {"success": True, "data": run, "message": "Already processing"}

    state.active_runs[run_id] = {
        "id": run_id,
        "runId": run_id,
        "status": "running",
        "progress": 0,
        "message": f"Starting pipeline on {folder}...",
        "startedAt": datetime.now().isoformat(),
        "datasetFolder": folder,
        "totalStages": 5,
        "completedStages": 0,
    }

    _write_run_context(
        run_id,
        {
            "source": "dataset-process",
            "datasetFolder": folder,
            "datasetPath": str(dataset_path),
        },
    )

    background_tasks.add_task(_execute_dataset_pipeline, run_id, dataset_path, folder)
    return {"success": True, "data": state.active_runs[run_id]}
