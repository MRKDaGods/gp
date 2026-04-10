import io
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

from backend.config import _HAS_CV2, OUTPUT_DIR
from backend.dependencies import get_app_state
from backend.services.clip_service import (
    _frame_image_path_in_dir,
    _resolve_stage0_camera_dir,
    _stage0_frame_path,
)
from backend.services.tracklet_service import _load_all_stage1_tracklets
from backend.services.video_service import _detect_camera_for_video, _normalize_camera_id
from backend.state import AppState

if _HAS_CV2:
    import cv2

router = APIRouter()


@router.get("/api/frames/{video_id}/{frame_id}/detections")
async def get_frame_with_detections(video_id: str, frame_id: int, state: AppState = Depends(get_app_state)):
    """Get frame with detections"""
    from backend.routers.detections import get_detections  # local import to avoid circular

    detections_response = await get_detections(video_id, frame_id, state=state)
    camera_id = _detect_camera_for_video(state.uploaded_videos.get(video_id, {}), None)
    return {
        "success": True,
        "data": {
            "frame": {
                "frameId": frame_id,
                "cameraId": camera_id,
                "timestamp": frame_id * 0.033,
                "framePath": f"/api/frames/{video_id}/{frame_id}",
                "width": state.uploaded_videos[video_id]["width"],
                "height": state.uploaded_videos[video_id]["height"],
            },
            "detections": detections_response["data"],
        },
    }


@router.get("/api/frames/{video_id}/{frame_id}")
async def get_frame_image(video_id: str, frame_id: int, state: AppState = Depends(get_app_state)):
    """Serve a single video frame as a JPEG image."""
    if not _HAS_CV2:
        raise HTTPException(status_code=500, detail="OpenCV not available")
    if video_id not in state.uploaded_videos:
        raise HTTPException(status_code=404, detail="Video not found")

    run_id = state.video_to_latest_run.get(video_id)
    if run_id:
        camera_id = _detect_camera_for_video(state.uploaded_videos.get(video_id, {}), None)
        fp = _stage0_frame_path(run_id, camera_id or "", frame_id)
        if fp and fp.exists():
            return FileResponse(
                str(fp),
                media_type="image/jpeg",
                headers={"Cache-Control": "public, max-age=3600"},
            )

    video_path = Path(state.uploaded_videos[video_id]["path"])
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file missing")

    cap = cv2.VideoCapture(str(video_path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if not ret or frame is None:
            raise HTTPException(status_code=404, detail=f"Frame {frame_id} not found")
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return StreamingResponse(
            io.BytesIO(buf.tobytes()),
            media_type="image/jpeg",
            headers={"Cache-Control": "public, max-age=3600"},
        )
    finally:
        cap.release()


@router.get("/api/runs/{run_id}/full_frame")
async def get_run_full_frame(
    run_id: str,
    cameraId: str = "",
    frameId: int = 0,
):
    """Serve a full stage0 frame (not cropped) for timeline tracklet overlay."""
    if not cameraId:
        raise HTTPException(status_code=400, detail="cameraId is required")
    d = _resolve_stage0_camera_dir(run_id, cameraId)
    if d is None:
        raise HTTPException(
            status_code=404,
            detail=f"No stage0 directory for run '{run_id}' and camera '{cameraId}'",
        )
    fp = _frame_image_path_in_dir(d, int(frameId))
    if fp is None:
        raise HTTPException(status_code=404, detail=f"Frame {frameId} not found under {d}")
    mt = "image/jpeg" if fp.suffix.lower() == ".jpg" else "image/png"
    return FileResponse(
        str(fp),
        media_type=mt,
        headers={"Cache-Control": "public, max-age=86400"},
    )


@router.get("/api/runs/{run_id}/tracklet_sequence")
async def get_tracklet_sequence(
    run_id: str,
    cameraId: str = "",
    trackId: int = -1,
    max_frames: int = 64,
):
    """Sampled frame ids + bboxes for a stage1 tracklet (timeline preview sync)."""
    if not cameraId or int(trackId) < 0:
        raise HTTPException(status_code=400, detail="cameraId and trackId are required")
    run_dir = OUTPUT_DIR / run_id
    if not run_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    cam_norm = _normalize_camera_id(cameraId)
    tracklet: Optional[Dict[str, Any]] = None
    for t in _load_all_stage1_tracklets(run_dir):
        if int(t.get("track_id", -1)) != int(trackId):
            continue
        if _normalize_camera_id(str(t.get("camera_id", ""))) == cam_norm:
            tracklet = t
            break
    if tracklet is None:
        raise HTTPException(
            status_code=404,
            detail=f"No tracklet track_id={trackId} for camera {cam_norm} in run {run_id}",
        )

    frames = (
        tracklet.get("frames") if isinstance(tracklet.get("frames"), list) else []
    )
    if not frames:
        return {
            "width": 0,
            "height": 0,
            "cameraId": cam_norm,
            "trackId": int(trackId),
            "frames": [],
        }

    mf = max(8, min(int(max_frames), 120))
    step = max(1, (len(frames) + mf - 1) // mf)
    sampled = frames[::step]
    n = len(sampled)

    stage_dir = _resolve_stage0_camera_dir(run_id, cam_norm)
    img_w, img_h = 0, 0
    if stage_dir is not None and _HAS_CV2:
        for fr in sampled:
            fid = int(fr.get("frame_id", fr.get("frameId", -1)))
            if fid < 0:
                continue
            p = _frame_image_path_in_dir(stage_dir, fid)
            if p is None:
                continue
            img = cv2.imread(str(p))
            if img is not None:
                img_h, img_w = img.shape[:2]
                break

    out_frames: List[Dict[str, Any]] = []
    for i, fr in enumerate(sampled):
        bbox = fr.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            bbox = [0.0, 0.0, 0.0, 0.0]
        fid = int(fr.get("frame_id", fr.get("frameId", -1)))
        ts = fr.get("timestamp")
        time_rel = 0.0 if n <= 1 else float(i) / float(n - 1)
        out_frames.append(
            {
                "frameId": fid,
                "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                "timeRel": time_rel,
                "timestamp": float(ts) if ts is not None else None,
            }
        )

    return {
        "width": int(img_w),
        "height": int(img_h),
        "cameraId": cam_norm,
        "trackId": int(trackId),
        "frames": out_frames,
    }
