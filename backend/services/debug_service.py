"""Timeline debug bundle export."""
import json
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from backend.config import OUTPUT_DIR
from backend.services.clip_service import _export_tracklet_clip
from backend.services.logging_service import _timeline_debug
from backend.services.tracklet_service import _find_tracklet_in_run
from backend.state import uploaded_videos


def _export_timeline_debug_bundle(
    request_payload: Dict[str, Any],
    timeline_payload: Dict[str, Any],
) -> Optional[Path]:
    """Persist a timeline-debug bundle under outputs/ for backend vs frontend triage."""
    selected_ids = request_payload.get("selectedTrackIds") or []
    if not isinstance(selected_ids, list) or not selected_ids:
        return None

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bundle_root = OUTPUT_DIR / "timeline_debug_exports" / f"{stamp}_{uuid.uuid4().hex[:8]}"
    bundle_root.mkdir(parents=True, exist_ok=True)

    (bundle_root / "request.json").write_text(json.dumps(request_payload, indent=2), encoding="utf-8")
    (bundle_root / "timeline_response.json").write_text(json.dumps(timeline_payload, indent=2), encoding="utf-8")

    data = timeline_payload.get("data", {}) if isinstance(timeline_payload, dict) else {}
    diagnostics = data.get("diagnostics", {}) if isinstance(data, dict) else {}

    video_id = str(request_payload.get("videoId", ""))
    if video_id in uploaded_videos:
        src_path = Path(str(uploaded_videos[video_id].get("path", "")))
        if src_path.exists() and src_path.is_file():
            probe_dir = bundle_root / "probe_video"
            probe_dir.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(src_path, probe_dir / src_path.name)
            except Exception:
                pass

    clip_manifest: Dict[str, Any] = {
        "selected": [],
        "timeline_candidates": [],
    }

    selected_summaries = data.get("selectedTracklets", []) if isinstance(data, dict) else []
    probe_run_id = str(diagnostics.get("selectedTrackletsSourceRun") or request_payload.get("runId") or "")
    if probe_run_id and isinstance(selected_summaries, list):
        out_selected = bundle_root / "selected_tracklets"
        out_selected.mkdir(parents=True, exist_ok=True)
        for idx, item in enumerate(selected_summaries[:20], start=1):
            try:
                tid = int(item.get("id", -1))
            except Exception:
                continue
            cam = str(item.get("cameraId", ""))
            t = _find_tracklet_in_run(probe_run_id, tid, cam)
            if t is None:
                continue
            out_file = out_selected / f"selected_{idx:02d}_track_{tid}_{str(t.get('camera_id', 'unknown'))}.mp4"
            ok, note = _export_tracklet_clip(probe_run_id, t, out_file)
            clip_manifest["selected"].append({
                "trackId": tid,
                "cameraId": str(t.get("camera_id", "")),
                "runId": probe_run_id,
                "file": str(out_file.relative_to(bundle_root).as_posix()),
                "ok": ok,
                "note": note,
            })

    trajectories = data.get("trajectories", []) if isinstance(data, dict) else []
    gallery_run_id = str(request_payload.get("runId", ""))
    if gallery_run_id and isinstance(trajectories, list):
        out_timeline = bundle_root / "timeline_candidates"
        out_timeline.mkdir(parents=True, exist_ok=True)
        exported = 0
        for traj_idx, traj in enumerate(trajectories[:20], start=1):
            tracklets = traj.get("tracklets", []) if isinstance(traj, dict) else []
            if not isinstance(tracklets, list):
                continue
            for tr in tracklets:
                if exported >= 30:
                    break
                try:
                    tid = int(tr.get("track_id") or tr.get("trackId") or -1)
                except Exception:
                    continue
                if tid < 0:
                    continue
                cam = str(tr.get("camera_id") or tr.get("cameraId") or "")
                t = _find_tracklet_in_run(gallery_run_id, tid, cam)
                if t is None:
                    continue
                out_file = out_timeline / f"traj_{traj_idx:02d}_track_{tid}_{str(t.get('camera_id', 'unknown'))}.mp4"
                ok, note = _export_tracklet_clip(gallery_run_id, t, out_file)
                clip_manifest["timeline_candidates"].append({
                    "trajectoryIndex": traj_idx,
                    "trackId": tid,
                    "cameraId": str(t.get("camera_id", "")),
                    "runId": gallery_run_id,
                    "file": str(out_file.relative_to(bundle_root).as_posix()),
                    "ok": ok,
                    "note": note,
                })
                exported += 1
            if exported >= 30:
                break

    (bundle_root / "clip_manifest.json").write_text(json.dumps(clip_manifest, indent=2), encoding="utf-8")
    return bundle_root
