import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from backend.config import ENABLE_KAGGLE_IMPORT, OUTPUT_DIR
from backend.services.clip_service import _transcode_to_mp4
from backend.services.pipeline_service import (
    _materialize_import_tree,
    _resolve_run_id,
    _write_run_context,
)
from backend.services.video_service import _detect_camera_for_video
from backend.state import active_runs, uploaded_videos, video_to_latest_run

router = APIRouter()


@router.get("/api/runs/{run_id}/matched_summary")
async def get_matched_summary(run_id: str):
    """Return the matched/summary.json for a probe run (fallback for UI rendering)."""
    summary_path = OUTPUT_DIR / run_id / "matched" / "summary.json"
    if not summary_path.exists():
        raise HTTPException(status_code=404, detail=f"No matched summary for run {run_id}")
    import json

    return json.loads(summary_path.read_text(encoding="utf-8"))


@router.get("/api/runs/{run_id}/matched_clips/{filename}")
async def get_matched_clip(run_id: str, filename: str):
    """Serve a matched clip mp4 from outputs/{run_id}/matched/ for in-browser playback."""
    safe_name = Path(filename).name
    if safe_name != filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    clip_path = OUTPUT_DIR / run_id / "matched" / safe_name
    if not clip_path.exists() or not clip_path.is_file():
        raise HTTPException(status_code=404, detail=f"Clip not found: {filename}")
    cache_dir = OUTPUT_DIR / run_id / "matched" / ".browser_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / safe_name
    if not cached.exists():
        if not _transcode_to_mp4(clip_path, cached):
            return FileResponse(str(clip_path), media_type="video/mp4")
    return FileResponse(str(cached), media_type="video/mp4")


@router.post("/api/runs/import-kaggle")
async def import_kaggle_run_artifacts(
    artifactsZip: UploadFile = File(...),
    runId: Optional[str] = Form(default=None),
    videoId: Optional[str] = Form(default=None),
    cameraId: Optional[str] = Form(default=None),
):
    """Import Kaggle-generated artifacts zip into local outputs for demo visualization."""
    if not ENABLE_KAGGLE_IMPORT:
        raise HTTPException(
            status_code=403, detail="Kaggle artifact import is disabled on this server"
        )

    if not artifactsZip.filename or not artifactsZip.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="artifactsZip must be a .zip file")

    if videoId and videoId not in uploaded_videos:
        raise HTTPException(status_code=404, detail="Video not found")

    run_id = _resolve_run_id(runId)
    run_dir = OUTPUT_DIR / run_id

    with tempfile.TemporaryDirectory(prefix="kaggle_import_") as tmp_dir:
        zip_path = Path(tmp_dir) / artifactsZip.filename
        with open(zip_path, "wb") as f:
            f.write(await artifactsZip.read())

        extract_root = Path(tmp_dir) / "extracted"
        extract_root.mkdir(parents=True, exist_ok=True)

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_root)
        except zipfile.BadZipFile as exc:
            raise HTTPException(status_code=400, detail=f"Invalid zip file: {exc}")

        _materialize_import_tree(extract_root, run_dir)

    resolved_camera = None
    if videoId:
        resolved_camera = _detect_camera_for_video(uploaded_videos[videoId], cameraId)
        video_to_latest_run[videoId] = run_id

    active_runs[run_id] = {
        "id": run_id,
        "runId": run_id,
        "stage": 6,
        "status": "completed",
        "progress": 100,
        "message": "Imported Kaggle artifacts",
        "startedAt": datetime.now().isoformat(),
        "completedAt": datetime.now().isoformat(),
        "videoId": videoId,
        "cameraId": resolved_camera,
        "runDir": str(run_dir),
        "source": "kaggle-import",
    }

    _write_run_context(
        run_id,
        {
            "source": "kaggle-import",
            "videoId": videoId,
            "cameraId": resolved_camera,
            "importFile": artifactsZip.filename,
        },
    )

    return {"success": True, "data": active_runs[run_id]}
