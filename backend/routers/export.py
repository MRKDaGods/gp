import json
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from backend.config import OUTPUT_DIR
from backend.services.tracklet_service import _load_all_stage1_tracklets

router = APIRouter()


@router.get("/api/export/{run_id}")
async def export_trajectories(run_id: str, format: str = "json"):
    """Export trajectories or tracklets in json/csv/mot formats."""
    fmt = format.lower()
    if fmt not in {"json", "csv", "mot"}:
        raise HTTPException(status_code=400, detail="format must be one of: json, csv, mot")

    run_dir = OUTPUT_DIR / run_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Run not found")

    export_dir = run_dir / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)

    trajectories_path = run_dir / "stage4" / "global_trajectories.json"
    trajectories = json.loads(trajectories_path.read_text()) if trajectories_path.exists() else []
    tracklets = _load_all_stage1_tracklets(run_dir)

    if fmt == "json":
        output_path = export_dir / "trajectories.json"
        payload = trajectories if trajectories else tracklets
        output_path.write_text(json.dumps(payload, indent=2))
    elif fmt == "csv":
        output_path = export_dir / "trajectories.csv"
        lines: List[str] = []
        if trajectories:
            lines.append("globalId,cameraCount,totalDuration,confidence")
            for item in trajectories:
                global_id = item.get("global_id", item.get("globalId", ""))
                cameras = item.get("camera_sequence", item.get("cameraSequence", [])) or []
                duration = item.get("total_duration", item.get("totalDuration", 0))
                confidence = item.get("confidence", 0)
                lines.append(f"{global_id},{len(cameras)},{duration},{confidence}")
        else:
            lines.append("trackId,cameraId,numFrames,startFrame,endFrame")
            for t in tracklets:
                frames = t.get("frames", [])
                start_frame = frames[0].get("frame_id") if frames else ""
                end_frame = frames[-1].get("frame_id") if frames else ""
                lines.append(
                    f"{t.get('track_id','')},{t.get('camera_id','')},{len(frames)},{start_frame},{end_frame}"
                )
        output_path.write_text("\n".join(lines) + "\n")
    else:
        output_path = export_dir / "trajectories.mot"
        lines = []
        for t in tracklets:
            track_id = int(t.get("track_id", -1))
            for frame in t.get("frames", []):
                frame_id = int(frame.get("frame_id", 0))
                bbox = frame.get("bbox", [0, 0, 0, 0])
                if isinstance(bbox, list) and len(bbox) >= 4:
                    x1, y1, x2, y2 = bbox[:4]
                    w = float(x2) - float(x1)
                    h = float(y2) - float(y1)
                else:
                    x1 = y1 = w = h = 0.0
                conf = float(frame.get("confidence", 1.0))
                lines.append(
                    f"{frame_id},{track_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf:.4f},-1,-1,-1"
                )
        output_path.write_text("\n".join(lines) + ("\n" if lines else ""))

    return {
        "success": True,
        "data": {"downloadUrl": f"/api/download/{run_id}/{output_path.name}"},
    }


@router.get("/api/download/{run_id}/{filename}")
async def download_export_file(run_id: str, filename: str):
    """Download an exported or generated run artifact by filename."""
    safe_name = Path(filename).name
    run_dir = OUTPUT_DIR / run_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Run not found")

    candidates = [
        run_dir / "exports" / safe_name,
        run_dir / "stage6" / safe_name,
    ]
    for path in candidates:
        if path.exists() and path.is_file():
            media_type = "application/octet-stream"
            if safe_name.endswith(".json"):
                media_type = "application/json"
            elif safe_name.endswith(".csv"):
                media_type = "text/csv"
            elif safe_name.endswith(".mp4"):
                media_type = "video/mp4"
            elif safe_name.endswith(".mot"):
                media_type = "text/plain"
            return FileResponse(path, media_type=media_type, filename=safe_name)

    raise HTTPException(status_code=404, detail="File not found")
