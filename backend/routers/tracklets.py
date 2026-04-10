import json
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException

from backend.config import OUTPUT_DIR
from backend.services.tracklet_service import (
    _load_all_stage1_tracklets,
    _load_tracklets,
    _run_dir_for_video,
)
from backend.services.video_service import _detect_camera_for_video
from backend.state import uploaded_videos, video_to_latest_run

router = APIRouter()


@router.get("/api/tracklets")
async def get_tracklets(cameraId: Optional[str] = None, videoId: Optional[str] = None):
    """Get tracklets from latest real stage1 output."""
    print(f"\n[UI Request] Get Tracklets: cameraId={cameraId}, videoId={videoId}")
    if not videoId:
        return {"success": True, "data": []}
    if videoId not in uploaded_videos:
        raise HTTPException(status_code=404, detail="Video not found")

    run_dir = _run_dir_for_video(videoId)
    if run_dir is None:
        return {"success": True, "data": []}

    resolved_camera_id = cameraId or _detect_camera_for_video(uploaded_videos[videoId], None)
    tracklets = _load_tracklets(resolved_camera_id, run_dir)

    if not tracklets:
        tracklets = _load_all_stage1_tracklets(run_dir)

    summary = []
    for t in tracklets:
        frames = t.get("frames", [])
        if not frames:
            continue
        mid_frame = frames[len(frames) // 2]

        _max_samples = 6
        if len(frames) <= _max_samples:
            sample_frames_data = frames
        else:
            step = (len(frames) - 1) / (_max_samples - 1)
            sample_frames_data = [frames[round(i * step)] for i in range(_max_samples)]

        sample_frames = [
            {
                "frameId": int(sf.get("frame_id", 0)),
                "bbox": sf.get("bbox", [0, 0, 0, 0]),
            }
            for sf in sample_frames_data
        ]

        summary.append(
            {
                "id": t.get("track_id"),
                "cameraId": t.get("camera_id"),
                "startFrame": frames[0].get("frame_id"),
                "endFrame": frames[-1].get("frame_id"),
                "numFrames": len(frames),
                "duration": float(frames[-1].get("timestamp", 0.0))
                - float(frames[0].get("timestamp", 0.0)),
                "className": t.get("class_name"),
                "classId": t.get("class_id"),
                "confidence": (
                    sum(float(f.get("confidence", 0.0)) for f in frames) / max(len(frames), 1)
                ),
                "representativeFrame": int(mid_frame.get("frame_id", 0)),
                "representativeBbox": mid_frame.get("bbox", [0, 0, 0, 0]),
                "sampleFrames": sample_frames,
            }
        )

    return {"success": True, "data": summary}


@router.get("/api/trajectories/{run_id}")
async def get_trajectories(run_id: str):
    """Get global trajectories from stage4 artifact if available."""
    traj_path = OUTPUT_DIR / run_id / "stage4" / "global_trajectories.json"
    if not traj_path.exists():
        return {"success": True, "data": []}

    return {"success": True, "data": json.loads(traj_path.read_text())}
