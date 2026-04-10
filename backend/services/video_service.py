"""Video metadata, camera ID utilities, GT parsing, and startup scanning."""
import configparser
import re
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.config import (
    _FFPROBE,
    _HAS_CV2,
    CITYFLOW_DIR,
    DATASET_DIR,
    DEMO_VIDEO_FALLBACK,
    OUTPUT_DIR,
    UPLOAD_DIR,
    VIDEO_EXTENSIONS,
)
from backend.state import uploaded_videos, video_to_latest_run

if _HAS_CV2:
    import cv2  # noqa: F401 — imported for type-checker; used via _HAS_CV2 guard


def _safe_reid_batch_size() -> int:
    """Determine a safe ReID batch size based on available GPU VRAM."""
    try:
        import torch
        if not torch.cuda.is_available():
            return 4
        total_vram_gb = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
        free_vram_gb = torch.cuda.mem_get_info(0)[0] / (1024 ** 3)
        available = min(free_vram_gb, total_vram_gb - 2.0)
        if available >= 8.0:
            return 32
        elif available >= 4.0:
            return 16
        elif available >= 2.0:
            return 8
        else:
            return 4
    except Exception:
        return 8


def _probe_video_metadata(file_path: Path) -> Dict[str, Any]:
    """Probe actual video duration/fps/resolution.
    Tries OpenCV first, then ffprobe, then returns safe defaults."""
    defaults = {"duration": 0.0, "fps": 30.0, "width": 1920, "height": 1080}
    if not file_path.exists():
        return defaults

    if _HAS_CV2:
        try:
            import cv2 as _cv2
            cap = _cv2.VideoCapture(str(file_path))
            if cap.isOpened():
                fps = cap.get(_cv2.CAP_PROP_FPS) or 30.0
                frame_count = cap.get(_cv2.CAP_PROP_FRAME_COUNT) or 0
                width = int(cap.get(_cv2.CAP_PROP_FRAME_WIDTH) or 1920)
                height = int(cap.get(_cv2.CAP_PROP_FRAME_HEIGHT) or 1080)
                cap.release()
                duration = frame_count / fps if fps > 0 else 0.0
                if duration > 0:
                    return {"duration": round(duration, 2), "fps": round(fps, 2), "width": width, "height": height}
        except Exception:
            pass

    if _FFPROBE:
        try:
            import json as _json
            result = subprocess.run(
                [
                    _FFPROBE, "-v", "quiet", "-print_format", "json",
                    "-show_streams", "-show_format", str(file_path),
                ],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode == 0:
                info = _json.loads(result.stdout)
                duration = float(info.get("format", {}).get("duration", 0) or 0)
                video_stream = next(
                    (s for s in info.get("streams", []) if s.get("codec_type") == "video"),
                    {}
                )
                width = int(video_stream.get("width", 1920) or 1920)
                height = int(video_stream.get("height", 1080) or 1080)
                fps_raw = video_stream.get("r_frame_rate", "30/1")
                try:
                    num, den = fps_raw.split("/")
                    fps = round(float(num) / float(den), 3) if float(den) else 30.0
                except Exception:
                    fps = 30.0
                if duration > 0:
                    return {"duration": round(duration, 2), "fps": fps, "width": width, "height": height}
        except Exception:
            pass

    return defaults


def _build_video_record(video_id: str, file_path: Path) -> Dict[str, Any]:
    """Build API-safe metadata for a discovered video file."""
    stat = file_path.stat()
    display_name = file_path.name
    _uuid_prefix = re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}_', display_name)
    if _uuid_prefix:
        display_name = display_name[_uuid_prefix.end():]
    parts = file_path.parts
    for i, part in enumerate(parts):
        if part.startswith("S0") and i + 1 < len(parts) and parts[i + 1].startswith("c"):
            display_name = f"{part}_{parts[i+1]}"
            break
    meta = _probe_video_metadata(file_path)
    return {
        "id": video_id,
        "name": display_name,
        "filename": file_path.name,
        "path": str(file_path),
        "size": stat.st_size,
        "duration": meta["duration"],
        "fps": meta["fps"],
        "width": meta["width"],
        "height": meta["height"],
        "uploadedAt": datetime.fromtimestamp(stat.st_mtime).isoformat(),
    }


def _register_video_path(file_path: Path) -> None:
    if not file_path.exists() or file_path.suffix.lower() not in VIDEO_EXTENSIONS:
        return
    video_id = str(uuid.uuid5(uuid.NAMESPACE_URL, str(file_path.resolve())))
    uploaded_videos[video_id] = _build_video_record(video_id, file_path)


def _build_virtual_video_record(camera_id: str, seqinfo_path: Path, fallback_video: Path) -> Dict[str, Any]:
    """Build a virtual video record from seqinfo.ini for a CityFlowV2 camera."""
    cp = configparser.ConfigParser()
    cp.read(str(seqinfo_path))
    fps = int(cp.get("Sequence", "frameRate", fallback="10"))
    width = int(cp.get("Sequence", "imWidth", fallback="1920"))
    height = int(cp.get("Sequence", "imHeight", fallback="1080"))
    seq_length = int(cp.get("Sequence", "seqLength", fallback="1955"))

    video_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"cityflowv2-demo-{camera_id}"))
    return {
        "id": video_id,
        "name": f"{camera_id}.avi",
        "filename": f"{camera_id}.avi",
        "path": str(fallback_video.resolve()) if fallback_video.exists() else "",
        "size": fallback_video.stat().st_size if fallback_video.exists() else 0,
        "duration": seq_length / max(fps, 1),
        "fps": fps,
        "width": width,
        "height": height,
        "uploadedAt": datetime.now().isoformat(),
        "_camera_id": camera_id,
        "_demo": True,
    }


def _parse_gt_detections(camera_id: str, frame_id: Optional[int] = None) -> List[Dict[str, Any]]:
    """Parse CityFlowV2 ground truth annotations as detections."""
    gt_path = CITYFLOW_DIR / camera_id / "gt" / "gt.txt"
    if not gt_path.exists():
        gt_path = CITYFLOW_DIR / camera_id.lower() / "gt" / "gt.txt"
    if not gt_path.exists():
        return []

    detections = []
    seen_tracks = set()
    for line in gt_path.read_text().strip().splitlines():
        parts = line.strip().split(",")
        if len(parts) < 6:
            continue
        fid = int(parts[0])
        if frame_id is not None and fid != frame_id:
            continue
        tid = int(parts[1])
        x1, y1, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])

        class_id = 2
        class_name = "car"
        if tid % 7 == 0:
            class_id = 7
            class_name = "truck"
        elif tid % 11 == 0:
            class_id = 5
            class_name = "bus"

        detections.append({
            "id": f"gt-{tid}-{fid}",
            "bbox": [x1, y1, x1 + w, y1 + h],
            "classId": class_id,
            "className": class_name,
            "confidence": round(0.82 + (tid % 17) * 0.01, 2),
            "frameId": fid,
            "trackId": tid,
        })

        if frame_id is None:
            if tid in seen_tracks:
                continue
            seen_tracks.add(tid)

    return detections


def _scan_startup_videos() -> None:
    """Load existing local videos so UI can show real footage after restart."""
    uploaded_videos.clear()

    for file_path in UPLOAD_DIR.glob("*"):
        _register_video_path(file_path)

    if CITYFLOW_DIR.exists():
        for file_path in CITYFLOW_DIR.rglob("*"):
            _register_video_path(file_path)

    if DATASET_DIR.exists():
        for file_path in DATASET_DIR.rglob("*"):
            _register_video_path(file_path)

    if DEMO_VIDEO_FALLBACK.exists():
        _register_video_path(DEMO_VIDEO_FALLBACK)

    if CITYFLOW_DIR.exists():
        fallback = DEMO_VIDEO_FALLBACK if DEMO_VIDEO_FALLBACK.exists() else Path("")
        registered_cameras = set()
        for v in uploaded_videos.values():
            cam = _extract_camera_id(str(v.get("name", ""))) or _extract_camera_id(str(v.get("path", "")))
            if cam:
                registered_cameras.add(cam)

        for cam_dir in sorted(CITYFLOW_DIR.iterdir()):
            if not cam_dir.is_dir():
                continue
            seqinfo = cam_dir / "seqinfo.ini"
            if not seqinfo.exists():
                continue
            camera_id = cam_dir.name.upper()
            if camera_id not in registered_cameras:
                rec = _build_virtual_video_record(camera_id, seqinfo, fallback)
                uploaded_videos[rec["id"]] = rec

    if OUTPUT_DIR.exists():
        latest_by_video: Dict[str, tuple] = {}
        for link_file in OUTPUT_DIR.glob("*/probe_video_id.txt"):
            try:
                vid_id = link_file.read_text().strip()
                run_id = link_file.parent.name
                mtime = float(link_file.stat().st_mtime)
                if not vid_id or not run_id:
                    continue
                prev = latest_by_video.get(vid_id)
                if prev is None or mtime > prev[0]:
                    latest_by_video[vid_id] = (mtime, run_id)
            except Exception:
                pass

        for vid_id, (_, run_id) in latest_by_video.items():
            video_to_latest_run[vid_id] = run_id


def _extract_camera_id(raw: str) -> Optional[str]:
    match = re.search(r"S\d{2}_c\d{3}", raw, flags=re.IGNORECASE)
    if match:
        return match.group(0).upper()
    match2 = re.search(r"(S\d{2})[/\\](c\d{3})", raw, flags=re.IGNORECASE)
    if match2:
        return f"{match2.group(1).upper()}_{match2.group(2).lower()}"
    return None


def _detect_camera_for_video(video_meta: Dict[str, Any], requested_camera_id: Optional[str]) -> str:
    if requested_camera_id:
        return requested_camera_id.upper()

    path_hint = str(video_meta.get("path", ""))
    name_hint = str(video_meta.get("name", ""))

    camera_id = _extract_camera_id(path_hint) or _extract_camera_id(name_hint)
    if camera_id:
        return camera_id

    vid_id = str(video_meta.get("id", "unknown"))
    return f"upload_{vid_id[:8]}"


def _normalize_camera_id(camera_id: str) -> str:
    cam = str(camera_id or "").strip()
    if cam.lower().startswith("query_"):
        cam = cam[6:]

    m = re.search(r"c\d{3}", cam, flags=re.IGNORECASE)
    if m:
        return m.group(0).lower()

    return cam.lower()


def _parse_selected_track_nums(raw_ids: List[str]) -> set:
    selected: set = set()
    if not raw_ids:
        return selected

    for raw in raw_ids:
        txt = str(raw).strip()
        if not txt:
            continue
        try:
            direct = int(txt)
            selected.add(direct)
            continue
        except Exception:
            pass

        m = re.match(r"^det-(\d+)(?:-|$)", txt)
        if m:
            selected.add(int(m.group(1)))
    return selected
