"""Frame path resolution, clip export, and video transcoding helpers."""
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from backend.config import _HAS_CV2, OUTPUT_DIR
from backend.services.tracklet_service import (
    _load_all_stage1_tracklets,
    _normalize_camera_id,
)

if _HAS_CV2:
    import cv2  # noqa: F401


def _transcode_to_mp4(src: Path, dst: Path) -> bool:
    """Transcode a non-MP4 video to browser-friendly H.264 MP4."""
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        try:
            r = subprocess.run(
                [ffmpeg_bin, "-y", "-i", str(src), "-c:v", "libx264",
                 "-preset", "fast", "-crf", "23", "-an", str(dst)],
                capture_output=True, timeout=600,
            )
            if r.returncode == 0 and dst.exists() and dst.stat().st_size > 0:
                return True
            dst.unlink(missing_ok=True)
        except Exception:
            dst.unlink(missing_ok=True)

    if not _HAS_CV2:
        return False
    try:
        import cv2 as _cv2
        cap = _cv2.VideoCapture(str(src))
        if not cap.isOpened():
            return False
        fps = cap.get(_cv2.CAP_PROP_FPS) or 10
        w = int(cap.get(_cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(_cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = _cv2.VideoWriter_fourcc(*"mp4v")
        tmp = dst.with_suffix(".tmp.mp4")
        writer = _cv2.VideoWriter(str(tmp), fourcc, fps, (w, h))
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
        cap.release()
        writer.release()
        if tmp.exists() and tmp.stat().st_size > 0:
            tmp.replace(dst)
            return True
        tmp.unlink(missing_ok=True)
    except Exception:
        dst.unlink(missing_ok=True)
    return False


def _stage0_frame_path(run_id: str, camera_id: str, frame_id: int) -> Optional[Path]:
    """Resolve a frame image path from stage0 artifacts for run/camera/frame."""
    candidates = [
        OUTPUT_DIR / run_id / "stage0" / camera_id,
        OUTPUT_DIR / "dataset_precompute_s01" / "stage0" / camera_id,
    ]
    for run_stage0 in candidates:
        jpg = run_stage0 / f"frame_{frame_id:06d}.jpg"
        png = run_stage0 / f"frame_{frame_id:06d}.png"
        if jpg.exists():
            return jpg
        if png.exists():
            return png
    return None


def _resolve_stage0_camera_dir(run_id: str, camera_id: str) -> Optional[Path]:
    """Directory with extracted stage0 frames for a camera."""
    cam = _normalize_camera_id(str(camera_id))
    primary = OUTPUT_DIR / run_id / "stage0" / cam
    if primary.is_dir():
        return primary
    if OUTPUT_DIR.exists():
        for pre_root in sorted(OUTPUT_DIR.glob("dataset_precompute_s*"), reverse=True):
            d = pre_root / "stage0" / cam
            if d.is_dir():
                return d
    legacy = OUTPUT_DIR / "dataset_precompute_s01" / "stage0" / cam
    if legacy.is_dir():
        return legacy
    return None


def _frame_image_path_in_dir(stage0_cam_dir: Path, frame_id: int) -> Optional[Path]:
    jpg = stage0_cam_dir / f"frame_{frame_id:06d}.jpg"
    if jpg.exists():
        return jpg
    png = stage0_cam_dir / f"frame_{frame_id:06d}.png"
    if png.exists():
        return png
    return None


def _export_tracklet_clip(
    run_id: str,
    tracklet: Dict[str, Any],
    out_path: Path,
    max_frames: int = 180,
    target_fps: float = 10.0,
) -> Tuple[bool, str]:
    """Export a cropped mp4 clip for a tracklet from stage0 frame artifacts."""
    if not _HAS_CV2:
        return False, "opencv_not_available"

    frames = tracklet.get("frames", []) if isinstance(tracklet, dict) else []
    if not frames:
        return False, "tracklet_has_no_frames"

    import cv2 as _cv2

    step = max(1, int(np.ceil(len(frames) / max_frames)))
    sampled = frames[::step]

    writer = None
    written = 0
    clip_w = clip_h = 0
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        for fr in sampled:
            try:
                frame_id = int(fr.get("frame_id", -1))
                bbox = fr.get("bbox", [0, 0, 0, 0])
                if frame_id < 0 or not isinstance(bbox, list) or len(bbox) != 4:
                    continue
                frame_path = _stage0_frame_path(run_id, str(tracklet.get("camera_id", "")), frame_id)
                if frame_path is None:
                    continue
                img = _cv2.imread(str(frame_path))
                if img is None:
                    continue

                h, w = img.shape[:2]
                bx1, by1, bx2, by2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                bw = max(1, bx2 - bx1)
                bh = max(1, by2 - by1)
                pad_x = int(bw * 0.5)
                pad_y = int(bh * 0.5)
                x1 = max(0, bx1 - pad_x)
                y1 = max(0, by1 - pad_y)
                x2 = min(w, bx2 + pad_x)
                y2 = min(h, by2 + pad_y)
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                if writer is None:
                    clip_h, clip_w = crop.shape[:2]
                    fourcc = _cv2.VideoWriter_fourcc(*"mp4v")
                    writer = _cv2.VideoWriter(str(out_path), fourcc, target_fps, (clip_w, clip_h))

                if crop.shape[1] != clip_w or crop.shape[0] != clip_h:
                    crop = _cv2.resize(crop, (clip_w, clip_h), interpolation=_cv2.INTER_AREA)
                writer.write(crop)
                written += 1
            except Exception:
                continue
    finally:
        if writer is not None:
            writer.release()

    if written <= 0:
        return False, "no_frames_written"

    _ffmpeg = shutil.which("ffmpeg")
    if _ffmpeg and out_path.exists():
        tmp = out_path.with_suffix(".tmp.mp4")
        try:
            subprocess.run(
                [
                    _ffmpeg, "-y", "-i", str(out_path),
                    "-c:v", "libx264", "-preset", "fast",
                    "-crf", "23", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    "-an", str(tmp),
                ],
                check=True, capture_output=True, timeout=60,
            )
            tmp.replace(out_path)
        except Exception as exc:
            if tmp.exists():
                tmp.unlink()
            print(f"[matched] ffmpeg re-encode failed: {exc}", flush=True)

    return True, f"frames_written={written}"


def _export_selected_clips(run_id: str, selected_ids: set) -> None:
    """Export a cropped mp4 clip for each user-selected tracklet."""
    run_dir = OUTPUT_DIR / run_id
    if not (run_dir / "stage1").exists():
        return

    out_dir = run_dir / "selected"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_tracklets = _load_all_stage1_tracklets(run_dir)
    chosen = [t for t in all_tracklets if int(t.get("track_id", -1)) in selected_ids]
    if not chosen:
        print(f"[selected] No tracklets matching IDs {selected_ids} found in run {run_id}", flush=True)
        return

    manifest = []
    for t in chosen:
        tid = t.get("track_id", "?")
        cam = t.get("camera_id", "unknown")
        out_file = out_dir / f"track_{tid}_{cam}.mp4"
        ok, msg = _export_tracklet_clip(run_id, t, out_file)
        print(f"[selected] track_{tid} ({cam}): {msg}", flush=True)
        manifest.append({"track_id": tid, "camera_id": cam, "file": out_file.name, "ok": ok, "msg": msg})

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[selected] Exported {len(manifest)} clip(s) to {out_dir}", flush=True)


def _export_matched_clips(
    probe_run_id: str,
    gallery_run_id: str,
    trajectories: List[Dict[str, Any]],
) -> None:
    """Export cropped mp4 clips for every tracklet in matched trajectories."""
    if not trajectories:
        return

    from datetime import datetime

    out_dir = OUTPUT_DIR / probe_run_id / "matched"
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_name: str = gallery_run_id
    gallery_ctx_path = OUTPUT_DIR / gallery_run_id / "run_context.json"
    if gallery_ctx_path.exists():
        try:
            ctx = json.loads(gallery_ctx_path.read_text(encoding="utf-8"))
            dataset_name = ctx.get("datasetFolder") or ctx.get("datasetName") or gallery_run_id
        except Exception:
            pass
    if dataset_name == gallery_run_id:
        try:
            import re as _re
            cfg_text = (OUTPUT_DIR / gallery_run_id / "config.yaml").read_text(encoding="utf-8")
            m = _re.search(r"run_name\s*:\s*(\S+)", cfg_text)
            if m:
                dataset_name = m.group(1)
        except Exception:
            pass

    gallery_run_dir = OUTPUT_DIR / gallery_run_id
    gallery_tracklets: Dict[tuple, Dict[str, Any]] = {}
    for t in _load_all_stage1_tracklets(gallery_run_dir):
        cam = _normalize_camera_id(str(t.get("camera_id", "")))
        tid = int(t.get("track_id", -1))
        if tid >= 0 and cam:
            gallery_tracklets[(cam, tid)] = t

    clips: List[Dict[str, Any]] = []
    cameras_seen: set = set()

    for traj in trajectories:
        gid = traj.get("global_id", traj.get("id", "?"))
        confidence = traj.get("confidence") or traj.get("matchEvidence", {}).get("meanBestFrameSimilarity")

        time_by_key: Dict[tuple, Dict[str, Any]] = {}
        for tl in (traj.get("timeline") or []):
            tl_cam = _normalize_camera_id(str(tl.get("camera_id") or ""))
            tl_tid = int(tl.get("track_id") or -1)
            if tl_cam and tl_tid >= 0 and not tl_cam.startswith("query_"):
                time_by_key.setdefault((tl_cam, tl_tid), tl)

        seen_keys: set = set()
        tracklets_list = traj.get("tracklets") or []
        for tr in tracklets_list:
            cam_raw = str(tr.get("camera_id") or tr.get("cameraId") or "")
            if cam_raw.startswith("query_"):
                continue
            cam = _normalize_camera_id(cam_raw)
            tid = int(tr.get("track_id") or tr.get("trackId") or -1)
            if tid < 0 or not cam:
                continue
            key = (cam, tid)
            if key in seen_keys:
                continue
            seen_keys.add(key)

            cameras_seen.add(cam)
            tracklet = gallery_tracklets.get((cam, tid))
            if tracklet is None:
                clips.append({
                    "global_id": gid, "camera_id": cam, "track_id": tid,
                    "ok": False, "msg": "tracklet_not_found_in_gallery_stage1",
                })
                continue

            tl_info = time_by_key.get(key, {})
            safe_cam = cam.replace("/", "_").replace("\\", "_")
            out_file = out_dir / f"global_{gid}_cam_{safe_cam}_track_{tid}.mp4"
            ok, msg = _export_tracklet_clip(gallery_run_id, tracklet, out_file)
            print(f"[matched] global_{gid} cam={cam} track={tid}: {msg}", flush=True)
            clips.append({
                "global_id": gid,
                "camera_id": cam,
                "track_id": tid,
                "confidence": round(float(confidence), 4) if confidence is not None else None,
                "start_time_s": tl_info.get("start"),
                "end_time_s": tl_info.get("end"),
                "duration_s": tl_info.get("duration_s"),
                "file": out_file.name,
                "ok": ok,
                "msg": msg,
            })

    ok_clips = [c for c in clips if c.get("ok")]
    summary = {
        "generatedAt": datetime.now().isoformat(),
        "probeRunId": probe_run_id,
        "datasetRunId": gallery_run_id,
        "datasetName": dataset_name,
        "totalMatchedTrajectories": len(trajectories),
        "totalMatchedTracklets": len(clips),
        "totalClipsExported": len(ok_clips),
        "totalCameras": len(cameras_seen),
        "cameras": sorted(cameras_seen),
        "clips": clips,
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"[matched] {len(ok_clips)} clip(s) across {len(cameras_seen)} camera(s) → {out_dir}",
        flush=True,
    )
