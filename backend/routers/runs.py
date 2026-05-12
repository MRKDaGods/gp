import json
import re
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from backend.config import ENABLE_KAGGLE_IMPORT, OUTPUT_DIR
from backend.dependencies import get_app_state
from backend.services.clip_service import _export_tracklet_clip, _transcode_to_mp4
from backend.services.pipeline_service import (
    _materialize_import_tree,
    _resolve_run_id,
    _write_run_context,
)
from backend.services.tracklet_service import (
    _build_tracklet_embedding_bank,
    _build_tracklet_global_map,
    _build_tracklet_lookup,
)
from backend.services.video_service import (
    _detect_camera_for_video,
    _normalize_camera_id,
    _safe_name_token,
)
from backend.state import AppState

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
        # Lazy-generate missing matched clip when possible.
        m = re.match(r"^global_(.+?)_cam_(.+?)_track_(\d+)\.mp4$", safe_name)
        if m:
            cam_token = str(m.group(2))
            tid = int(m.group(3))
            cam_norm = _normalize_camera_id(cam_token)

            gallery_run_id = run_id
            summary_path = OUTPUT_DIR / run_id / "matched" / "summary.json"
            if summary_path.exists():
                try:
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    ds_run = str(summary.get("datasetRunId") or "").strip()
                    if ds_run:
                        gallery_run_id = ds_run
                except Exception:
                    pass

            lookup = _build_tracklet_lookup(gallery_run_id)
            src_tracklet = lookup.get((cam_norm, tid))
            if src_tracklet is not None:
                try:
                    clip_path.parent.mkdir(parents=True, exist_ok=True)
                    ok, _msg = _export_tracklet_clip(gallery_run_id, src_tracklet, clip_path)
                    if not ok:
                        raise HTTPException(status_code=404, detail=f"Clip not found: {filename}")
                except HTTPException:
                    raise
                except Exception:
                    raise HTTPException(status_code=404, detail=f"Clip not found: {filename}")

        if not clip_path.exists() or not clip_path.is_file():
            raise HTTPException(status_code=404, detail=f"Clip not found: {filename}")
    cache_dir = OUTPUT_DIR / run_id / "matched" / ".browser_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / safe_name
    if not cached.exists():
        if not _transcode_to_mp4(clip_path, cached):
            return FileResponse(str(clip_path), media_type="video/mp4")
    return FileResponse(str(cached), media_type="video/mp4")


def _validate_run_id(run_id: str) -> None:
    """Reject run_id values that could escape OUTPUT_DIR."""
    if not run_id or ".." in run_id or "/" in run_id or "\\" in run_id:
        raise HTTPException(status_code=400, detail="Invalid run_id")
    resolved = (OUTPUT_DIR / run_id).resolve()
    if not str(resolved).startswith(str(OUTPUT_DIR.resolve())):
        raise HTTPException(status_code=403, detail="Access denied")


@router.get("/api/runs/{run_id}/matched_alternatives")
async def get_matched_alternatives(
    run_id: str,
    topK: int = 5,
    anchorCameraId: str = "",
    anchorTrackId: Optional[int] = None,
    excludeGlobalId: Optional[int] = None,
    excludeCameraId: str = "",
    excludeTrackId: Optional[int] = None,
):
    """Return top alternatives for timeline selection."""
    _validate_run_id(run_id)
    summary_path = OUTPUT_DIR / run_id / "matched" / "summary.json"
    if not summary_path.exists():
        raise HTTPException(status_code=404, detail=f"No matched summary for run {run_id}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    k = max(1, min(int(topK), 20))
    anchor_cam_norm = _normalize_camera_id(anchorCameraId) if anchorCameraId else ""
    anchor_tid = int(anchorTrackId) if anchorTrackId is not None else None

    # --- Per-anchor mode: compute from stage2 embeddings ---
    if anchor_cam_norm and anchor_tid is not None and anchor_tid >= 0:
        gallery_run_id = str(summary.get("datasetRunId") or "").strip() or run_id
        bank = _build_tracklet_embedding_bank(gallery_run_id)
        anchor_key = (anchor_cam_norm, anchor_tid)
        anchor_vec = bank.get(anchor_key)
        if anchor_vec is None:
            return {
                "runId": run_id,
                "totalCameras": int(summary.get("totalCameras", 0) or 0),
                "cameras": summary.get("cameras", []),
                "subfolder": "top5_alternatives/by_track",
                "alternatives": [],
                "mode": "per_track",
                "message": "Anchor track embedding not found in gallery stage2 index",
            }

        key_to_gid, gid_to_cam_count, key_to_span = _build_tracklet_global_map(gallery_run_id)
        lookup = _build_tracklet_lookup(gallery_run_id)
        scored: List[Tuple[float, Tuple[str, int], Optional[int]]] = []

        for key, vec in bank.items():
            if key == anchor_key:
                continue
            gid = key_to_gid.get(key)
            if excludeGlobalId is not None and gid is not None and int(gid) == int(excludeGlobalId):
                continue
            score = float(np.dot(anchor_vec, vec))
            scored.append((score, key, gid))

        scored.sort(key=lambda x: x[0], reverse=True)

        top_root = OUTPUT_DIR / run_id / "matched" / "top5_alternatives"
        anchor_folder = f"{_safe_name_token(anchor_cam_norm)}_{int(anchor_tid)}"
        anchor_dir = top_root / "by_track" / anchor_folder
        anchor_dir.mkdir(parents=True, exist_ok=True)

        alternatives: List[Dict[str, Any]] = []
        for score, key, gid in scored:
            if len(alternatives) >= k:
                break
            cam, tid = key
            tracklet = lookup.get(key)
            if tracklet is None:
                continue

            gid_token = f"_g{gid}" if gid is not None else ""
            filename = (
                f"alt_{len(alternatives) + 1:02d}_cam_"
                f"{_safe_name_token(cam)}_track_{int(tid)}{gid_token}.mp4"
            )
            clip_rel = (Path("by_track") / anchor_folder / filename).as_posix()
            out_file = top_root / clip_rel

            if out_file.exists():
                ok, msg = True, "cached"
            else:
                ok, msg = _export_tracklet_clip(gallery_run_id, tracklet, out_file)

            if not ok:
                continue

            frames = tracklet.get("frames") if isinstance(tracklet.get("frames"), list) else []
            rep_frame = None
            rep_bbox = None
            if frames:
                mid = frames[len(frames) // 2]
                rep_frame = int(mid.get("frame_id", mid.get("frameId", -1)))
                bbox = mid.get("bbox")
                if isinstance(bbox, list) and len(bbox) == 4:
                    rep_bbox = [float(v) for v in bbox]

            if (cam, tid) in key_to_span:
                start_s = float(key_to_span[(cam, tid)]["start"])
                end_s = float(key_to_span[(cam, tid)]["end"])
            elif frames:
                ts = [float(fr.get("timestamp", 0.0) or 0.0) for fr in frames]
                start_s = float(min(ts)) if ts else 0.0
                end_s = float(max(ts)) if ts else start_s + 0.1
                end_s = max(end_s, start_s + 0.1)
            else:
                start_s = 0.0
                end_s = 0.1

            gid_label = int(gid) if gid is not None else 0
            label = f"G-{str(gid_label).zfill(4)} \u00b7 {cam} \u00b7 track {int(tid)}"

            alternatives.append({
                "rank": len(alternatives) + 1,
                "global_id": gid,
                "camera_id": cam,
                "track_id": int(tid),
                "score": round(float(score), 4),
                "confidence": round(float(score), 4),
                "num_cameras": gid_to_cam_count.get(int(gid), 1) if gid is not None else 1,
                "class_name": str(tracklet.get("class_name", "vehicle")),
                "start_time_s": round(float(start_s), 4),
                "end_time_s": round(float(end_s), 4),
                "representative_frame": rep_frame,
                "representative_bbox": rep_bbox,
                "label": label,
                "clip_path": clip_rel,
                "ok": True,
                "msg": msg,
            })

        return {
            "runId": run_id,
            "totalCameras": int(summary.get("totalCameras", 0) or 0),
            "cameras": summary.get("cameras", []),
            "subfolder": "top5_alternatives/by_track",
            "alternatives": alternatives,
            "mode": "per_track",
            "anchor": {
                "cameraId": anchor_cam_norm,
                "trackId": int(anchor_tid),
            },
        }

    # --- Legacy mode: run-level alternatives list (fallback) ---
    alternatives_raw = (
        summary.get("topAlternatives")
        if isinstance(summary.get("topAlternatives"), list)
        else []
    )
    if not alternatives_raw:
        raw_clips = summary.get("clips") if isinstance(summary.get("clips"), list) else []
        alternatives_raw = [
            {
                "rank": idx + 1,
                "global_id": c.get("global_id"),
                "camera_id": c.get("camera_id"),
                "track_id": c.get("track_id"),
                "score": float(c.get("confidence") or 0),
                "confidence": float(c.get("confidence") or 0),
                "num_cameras": 1,
                "clip_path": c.get("file"),
                "ok": bool(c.get("ok")),
                "msg": c.get("msg"),
            }
            for idx, c in enumerate(raw_clips)
            if isinstance(c, dict)
        ]

    exclude_cam_norm = _normalize_camera_id(excludeCameraId) if excludeCameraId else ""
    filtered: List[Dict[str, Any]] = []
    for alt in alternatives_raw:
        try:
            alt_gid = alt.get("global_id")
            alt_cam = _normalize_camera_id(str(alt.get("camera_id", "")))
            alt_tid = int(alt.get("track_id", -1))
        except Exception:
            continue

        if excludeGlobalId is not None and alt_gid is not None:
            try:
                if int(alt_gid) == int(excludeGlobalId):
                    continue
            except Exception:
                if str(alt_gid) == str(excludeGlobalId):
                    continue
        if excludeTrackId is not None and alt_tid == int(excludeTrackId):
            if not exclude_cam_norm or alt_cam == exclude_cam_norm:
                continue

        filtered.append(alt)

    filtered.sort(
        key=lambda x: float(x.get("score", x.get("confidence", 0)) or 0),
        reverse=True,
    )

    return {
        "runId": run_id,
        "totalCameras": int(summary.get("totalCameras", 0) or 0),
        "cameras": summary.get("cameras", []),
        "subfolder": str(summary.get("topAlternativesSubfolder", "top5_alternatives")),
        "alternatives": filtered[:k],
        "mode": "legacy",
    }


@router.get("/api/runs/{run_id}/matched_alternatives/{clip_relpath:path}")
async def get_matched_alternative_clip(run_id: str, clip_relpath: str):
    """Serve one top-alternative clip from outputs/{run_id}/matched/top5_alternatives/."""
    _validate_run_id(run_id)
    rel = Path(str(clip_relpath))
    if rel.is_absolute() or ".." in rel.parts:
        raise HTTPException(status_code=400, detail="Invalid filename")

    alt_dir = (OUTPUT_DIR / run_id / "matched" / "top5_alternatives").resolve()
    clip_path = (alt_dir / rel).resolve()
    if alt_dir not in clip_path.parents:
        raise HTTPException(status_code=400, detail="Invalid filename")
    if not clip_path.exists() or not clip_path.is_file():
        raise HTTPException(status_code=404, detail=f"Alternative clip not found: {clip_relpath}")

    cache_dir = clip_path.parent / ".browser_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / clip_path.name
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
    state: AppState = Depends(get_app_state),
):
    """Import Kaggle-generated artifacts zip into local outputs for demo visualization."""
    if not ENABLE_KAGGLE_IMPORT:
        raise HTTPException(
            status_code=403, detail="Kaggle artifact import is disabled on this server"
        )

    if not artifactsZip.filename or not artifactsZip.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="artifactsZip must be a .zip file")

    if videoId and videoId not in state.uploaded_videos:
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
        resolved_camera = _detect_camera_for_video(state.uploaded_videos[videoId], cameraId)
        state.video_to_latest_run[videoId] = run_id

    state.active_runs[run_id] = {
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

    return {"success": True, "data": state.active_runs[run_id]}
