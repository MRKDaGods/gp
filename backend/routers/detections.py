from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException

from backend.config import OUTPUT_DIR
from backend.services.tracklet_service import (
    _confidence_for_tracklet_frame,
    _load_tracklets,
    _run_dir_for_video,
    _tracklets_to_detections,
)
from backend.services.video_service import (
    _detect_camera_for_video,
    _parse_gt_detections,
)
from backend.state import active_runs, uploaded_videos, video_to_latest_run

router = APIRouter()


@router.get("/api/detections/{video_id}")
async def get_detections(video_id: str, frameId: Optional[int] = None):
    """Get detections for a video/frame from real stage1 outputs."""
    if video_id not in uploaded_videos:
        raise HTTPException(status_code=404, detail="Video not found")

    run_dir = _run_dir_for_video(video_id)
    run_id = video_to_latest_run.get(video_id)
    camera_id = None
    if run_id and run_id in active_runs:
        camera_id = active_runs[run_id].get("cameraId")
    if camera_id is None:
        camera_id = _detect_camera_for_video(uploaded_videos[video_id], None)

    if run_dir is None:
        gt_dets = _parse_gt_detections(camera_id, frameId)
        if gt_dets:
            return {"success": True, "data": gt_dets, "message": "CityFlowV2 ground truth demo data"}
        return {
            "success": True,
            "data": [],
            "message": "No stage1 run found for this video yet.",
        }

    tracklets = _load_tracklets(camera_id, run_dir)
    detections = _tracklets_to_detections(tracklets, frameId)
    return {"success": True, "data": detections}


@router.get("/api/detections/{video_id}/all")
async def get_all_detections(video_id: str):
    """Return every detection for every frame, grouped by frame number."""
    if video_id not in uploaded_videos:
        raise HTTPException(status_code=404, detail="Video not found")

    run_dir = _run_dir_for_video(video_id)
    run_id = video_to_latest_run.get(video_id)
    camera_id = None
    if run_id and run_id in active_runs:
        camera_id = active_runs[run_id].get("cameraId")
    if camera_id is None:
        camera_id = _detect_camera_for_video(uploaded_videos[video_id], None)

    if run_dir is None:
        return {"success": True, "data": {}}

    tracklets = _load_tracklets(camera_id, run_dir)
    grouped: Dict[str, list] = {}
    for tracklet in tracklets:
        track_id = tracklet.get("track_id")
        class_id = tracklet.get("class_id")
        class_name = tracklet.get("class_name")
        for frame in tracklet.get("frames", []):
            fid = str(int(frame.get("frame_id", 0)))
            det = {
                "id": f"det-{track_id}-{fid}",
                "bbox": frame.get("bbox", [0, 0, 0, 0]),
                "classId": class_id,
                "className": class_name,
                "confidence": _confidence_for_tracklet_frame(frame, tracklet),
                "frameId": int(fid),
                "trackId": track_id,
            }
            grouped.setdefault(fid, []).append(det)

    return {"success": True, "data": grouped}
