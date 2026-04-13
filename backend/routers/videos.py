import os
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from backend.config import UPLOAD_DIR, VIDEO_EXTENSIONS
from backend.dependencies import get_app_state
from backend.services.clip_service import _transcode_to_mp4
from backend.services.video_service import _build_video_record
from backend.state import AppState

router = APIRouter()


def _video_payload(video_id: str, record: dict, state: AppState) -> dict:
    """Attach latest pipeline run id for Output/export when client state is stale."""
    payload = dict(record)
    rid = state.video_to_latest_run.get(video_id)
    payload["latestRunId"] = rid if rid else None
    return payload


@router.post("/api/videos/upload")
async def upload_video(
    video: UploadFile = File(...),
    state: AppState = Depends(get_app_state),
):
    """Upload a video file"""
    try:
        video_id = str(uuid.uuid4())
        video_path = UPLOAD_DIR / f"{video_id}_{video.filename}"

        with open(video_path, "wb") as f:
            content = await video.read()
            f.write(content)

        state.uploaded_videos[video_id] = _build_video_record(video_id, video_path)

        return {
            "success": True,
            "data": _video_payload(video_id, state.uploaded_videos[video_id], state),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/videos")
async def get_videos(state: AppState = Depends(get_app_state)):
    """Get all uploaded videos"""
    data = [
        _video_payload(vid, rec, state) for vid, rec in state.uploaded_videos.items()
    ]
    return {"success": True, "data": data}


@router.get("/api/videos/{video_id}")
async def get_video(video_id: str, state: AppState = Depends(get_app_state)):
    """Get video details"""
    if video_id not in state.uploaded_videos:
        raise HTTPException(status_code=404, detail="Video not found")
    return {
        "success": True,
        "data": _video_payload(video_id, state.uploaded_videos[video_id], state),
    }


@router.delete("/api/videos/{video_id}")
async def delete_video(video_id: str, state: AppState = Depends(get_app_state)):
    """Delete a video"""
    if video_id not in state.uploaded_videos:
        raise HTTPException(status_code=404, detail="Video not found")

    video = state.uploaded_videos.pop(video_id)
    if os.path.exists(video["path"]):
        os.remove(video["path"])

    return {"success": True, "data": None}


@router.get("/api/videos/stream/{video_id}")
async def stream_video(video_id: str, state: AppState = Depends(get_app_state)):
    """Stream video file (transcodes AVI to MP4 for browser playback)"""
    if video_id not in state.uploaded_videos:
        raise HTTPException(status_code=404, detail="Video not found")
    video_path = Path(state.uploaded_videos[video_id]["path"])
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not available for streaming")
    if video_path.suffix.lower() in {".avi", ".mkv", ".mov", ".m4v"}:
        cache_dir = Path("uploads/.transcode_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        mp4_path = cache_dir / f"{video_id}.mp4"
        marker = cache_dir / f"{video_id}.ok"
        if not mp4_path.exists() or not marker.exists():
            mp4_path.unlink(missing_ok=True)
            marker.unlink(missing_ok=True)
            ok = _transcode_to_mp4(video_path, mp4_path)
            if ok and mp4_path.exists():
                marker.write_text("ok")
            else:
                mp4_path.unlink(missing_ok=True)
                return FileResponse(str(video_path), media_type="video/x-msvideo")
        return FileResponse(str(mp4_path), media_type="video/mp4")
    return FileResponse(str(video_path), media_type="video/mp4")
