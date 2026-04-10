import os
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from backend.config import UPLOAD_DIR, VIDEO_EXTENSIONS
from backend.services.clip_service import _transcode_to_mp4
from backend.services.video_service import _build_video_record
from backend.state import uploaded_videos

router = APIRouter()


@router.post("/api/videos/upload")
async def upload_video(video: UploadFile = File(...)):
    """Upload a video file"""
    try:
        video_id = str(uuid.uuid4())
        video_path = UPLOAD_DIR / f"{video_id}_{video.filename}"

        with open(video_path, "wb") as f:
            content = await video.read()
            f.write(content)

        uploaded_videos[video_id] = _build_video_record(video_id, video_path)

        return {"success": True, "data": uploaded_videos[video_id]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/videos")
async def get_videos():
    """Get all uploaded videos"""
    return {"success": True, "data": list(uploaded_videos.values())}


@router.get("/api/videos/{video_id}")
async def get_video(video_id: str):
    """Get video details"""
    if video_id not in uploaded_videos:
        raise HTTPException(status_code=404, detail="Video not found")
    return {"success": True, "data": uploaded_videos[video_id]}


@router.delete("/api/videos/{video_id}")
async def delete_video(video_id: str):
    """Delete a video"""
    if video_id not in uploaded_videos:
        raise HTTPException(status_code=404, detail="Video not found")

    video = uploaded_videos.pop(video_id)
    if os.path.exists(video["path"]):
        os.remove(video["path"])

    return {"success": True, "data": None}


@router.get("/api/videos/stream/{video_id}")
async def stream_video(video_id: str):
    """Stream video file (transcodes AVI to MP4 for browser playback)"""
    if video_id not in uploaded_videos:
        raise HTTPException(status_code=404, detail="Video not found")
    video_path = Path(uploaded_videos[video_id]["path"])
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not available for streaming")
    if video_path.suffix.lower() in {".avi", ".mkv", ".mov", ".m4v"} and video_path.suffix.lower() != ".mp4":
        cache_dir = Path("uploads/.transcode_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        mp4_path = cache_dir / f"{video_id}.mp4"
        if not mp4_path.exists():
            ok = _transcode_to_mp4(video_path, mp4_path)
            if not ok:
                return FileResponse(str(video_path), media_type="video/x-msvideo")
        return FileResponse(str(mp4_path), media_type="video/mp4")
    return FileResponse(str(video_path), media_type="video/mp4")
