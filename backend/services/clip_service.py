"""Frame path resolution, clip export, and video transcoding helpers."""
import hashlib
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


def _find_ffmpeg() -> Optional[str]:
    """Return an ffmpeg binary path, checking PATH then imageio_ffmpeg bundle."""
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _transcode_to_mp4(src: Path, dst: Path) -> bool:
    """Transcode a non-MP4 video to browser-friendly H.264 MP4."""
    ffmpeg_bin = _find_ffmpeg()
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
        tmp = dst.with_suffix(".tmp.mp4")
        # Try H.264 codecs that browsers can actually play, then fall back
        for tag in ("avc1", "H264", "X264", "mp4v"):
            fourcc = _cv2.VideoWriter_fourcc(*tag)
            writer = _cv2.VideoWriter(str(tmp), fourcc, fps, (w, h))
            if writer.isOpened():
                break
            writer.release()
        else:
            cap.release()
            return False
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


def _allowed_clip_keys_from_request(
    include_clips: List[Dict[str, Any]],
) -> set:
    """Normalize (camera_id, track_id) keys the same way matched/summary.json clips do."""
    allowed: set = set()
    for item in include_clips:
        raw_cam = str(item.get("camera_id") or item.get("cameraId") or "").strip()
        tid_raw = item.get("track_id")
        if tid_raw is None:
            tid_raw = item.get("trackletId")
        try:
            tid = int(tid_raw)
        except (TypeError, ValueError):
            continue
        cam = _normalize_camera_id(raw_cam)
        if cam and tid >= 0:
            allowed.add((cam, tid))
    return allowed


def _generate_annotated_summary_video(
    run_id: str,
    target_fps: float = 10.0,
    include_clips: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Path]:
    """Generate a stitched full-frame annotated video for a run's matched trajectory.

    Reads matched/summary.json, loads stage1 tracklets, draws bounding boxes
    on stage0 frames, and concatenates all camera segments with transition cards.
    Saves to stage6/summary.mp4 (full) or stage6/summary_sel_<hash>.mp4 (subset).
    Optional ``include_clips`` limits which cameras/tracks are stitched (timeline selection).
    Returns the path on success, None on failure.
    """
    if not _HAS_CV2:
        return None
    import cv2 as _cv2

    run_dir = OUTPUT_DIR / run_id
    summary_path = run_dir / "matched" / "summary.json"
    if not summary_path.exists():
        return None

    try:
        summary = json.loads(summary_path.read_text())
    except Exception:
        return None

    clips = summary.get("clips", [])
    if not clips:
        return None

    allowed: Optional[set] = None
    if include_clips is not None:
        if len(include_clips) == 0:
            return None
        allowed = _allowed_clip_keys_from_request(include_clips)
        if not allowed:
            return None

    dataset_run_id = str(summary.get("datasetRunId", run_id))
    probe_run_id = str(summary.get("probeRunId", run_id))

    out_dir = run_dir / "stage6"
    out_dir.mkdir(parents=True, exist_ok=True)
    if include_clips is None:
        out_path = out_dir / "summary.mp4"
    else:
        blob = json.dumps(include_clips, sort_keys=True, default=str).encode("utf-8")
        h = hashlib.sha256(blob).hexdigest()[:24]
        out_path = out_dir / f"summary_sel_{h}.mp4"

    if include_clips is not None and out_path.exists():
        return out_path

    tmp_path = out_dir / f"_tmp_{out_path.stem}_frames.mp4"

    writer = None
    vid_w, vid_h = 0, 0
    total_written = 0
    transition_frames = int(target_fps * 0.5)

    BOX_COLOR = (0, 220, 0)
    LABEL_BG = (0, 0, 0)
    LABEL_FG = (255, 255, 255)
    FONT = _cv2.FONT_HERSHEY_SIMPLEX

    try:
        for clip_idx, clip_info in enumerate(clips):
            camera_id = str(clip_info.get("camera_id", ""))
            track_id = int(clip_info.get("track_id", -1))
            confidence = float(clip_info.get("confidence", 0))

            if not camera_id or track_id < 0:
                continue

            cam_key = _normalize_camera_id(camera_id)
            if allowed is not None and (cam_key, track_id) not in allowed:
                continue

            tracklet = _find_tracklet_for_clip(
                probe_run_id, dataset_run_id, camera_id, track_id
            )
            if tracklet is None:
                continue
            frames = tracklet.get("frames", [])
            if not frames:
                continue

            stage0_dir = _resolve_stage0_camera_dir(dataset_run_id, camera_id)
            if stage0_dir is None:
                stage0_dir = _resolve_stage0_camera_dir(probe_run_id, camera_id)
            if stage0_dir is None:
                continue

            if writer is not None and transition_frames > 0:
                card = np.zeros((vid_h, vid_w, 3), dtype=np.uint8)
                label = f"Camera: {camera_id}"
                font_scale = min(vid_w, vid_h) / 500.0
                thickness = max(1, int(font_scale * 2))
                (tw, th), _ = _cv2.getTextSize(label, FONT, font_scale, thickness)
                tx = (vid_w - tw) // 2
                ty = (vid_h + th) // 2
                _cv2.putText(card, label, (tx, ty), FONT, font_scale, LABEL_FG, thickness, _cv2.LINE_AA)
                for _ in range(transition_frames):
                    writer.write(card)
                    total_written += 1

            step = max(1, int(np.ceil(len(frames) / 300)))
            sampled = frames[::step]

            for fr in sampled:
                try:
                    frame_id = int(fr.get("frame_id", -1))
                    bbox = fr.get("bbox", [0, 0, 0, 0])
                    if frame_id < 0 or not isinstance(bbox, list) or len(bbox) != 4:
                        continue

                    fp = _frame_image_path_in_dir(stage0_dir, frame_id)
                    if fp is None:
                        continue
                    img = _cv2.imread(str(fp))
                    if img is None:
                        continue

                    h, w = img.shape[:2]

                    if writer is None:
                        vid_w, vid_h = w, h
                        fourcc = _cv2.VideoWriter_fourcc(*"mp4v")
                        writer = _cv2.VideoWriter(str(tmp_path), fourcc, target_fps, (vid_w, vid_h))
                        if not writer.isOpened():
                            return None

                    if w != vid_w or h != vid_h:
                        img = _cv2.resize(img, (vid_w, vid_h), interpolation=_cv2.INTER_AREA)
                        h, w = vid_h, vid_w

                    bx1 = max(0, int(bbox[0]))
                    by1 = max(0, int(bbox[1]))
                    bx2 = min(w, int(bbox[2]))
                    by2 = min(h, int(bbox[3]))
                    _cv2.rectangle(img, (bx1, by1), (bx2, by2), BOX_COLOR, 2)

                    label = f"Cam: {camera_id} | Track #{track_id} | {confidence*100:.0f}%"
                    font_scale = 0.55
                    thickness = 1
                    (tw, th), baseline = _cv2.getTextSize(label, FONT, font_scale, thickness)
                    lx = bx1
                    ly = max(th + baseline + 4, by1 - 6)
                    _cv2.rectangle(img, (lx, ly - th - baseline - 2), (lx + tw + 4, ly + 2), LABEL_BG, -1)
                    _cv2.putText(img, label, (lx + 2, ly - baseline), FONT, font_scale, LABEL_FG, thickness, _cv2.LINE_AA)

                    writer.write(img)
                    total_written += 1
                except Exception:
                    continue
    finally:
        if writer is not None:
            writer.release()

    if total_written == 0:
        tmp_path.unlink(missing_ok=True)
        return None

    ffmpeg_bin = _find_ffmpeg()
    if ffmpeg_bin and tmp_path.exists():
        try:
            subprocess.run(
                [ffmpeg_bin, "-y", "-i", str(tmp_path),
                 "-c:v", "libx264", "-preset", "fast",
                 "-crf", "23", "-pix_fmt", "yuv420p",
                 "-movflags", "+faststart", "-an", str(out_path)],
                check=True, capture_output=True, timeout=300,
            )
            tmp_path.unlink(missing_ok=True)
        except Exception as exc:
            print(f"[summary] ffmpeg re-encode failed: {exc}", flush=True)
            if out_path.exists():
                out_path.unlink(missing_ok=True)
            tmp_path.rename(out_path)
    else:
        tmp_path.rename(out_path)

    print(f"[summary] Generated annotated summary: {out_path} ({total_written} frames)", flush=True)
    return out_path


def _find_tracklet_for_clip(
    probe_run_id: str,
    dataset_run_id: str,
    camera_id: str,
    track_id: int,
) -> Optional[Dict[str, Any]]:
    """Locate the stage1 tracklet for a matched clip across probe and dataset runs."""
    cam = _normalize_camera_id(camera_id)
    for rid in (probe_run_id, dataset_run_id):
        stage1_dir = OUTPUT_DIR / rid / "stage1"
        if not stage1_dir.exists():
            continue
        for p in sorted(stage1_dir.glob("tracklets_*.json")):
            if cam not in _normalize_camera_id(p.stem):
                continue
            try:
                tracklets = json.loads(p.read_text())
                for t in tracklets:
                    if int(t.get("track_id", -1)) == track_id:
                        return t
            except Exception:
                continue
    all_tracklets = _load_all_stage1_tracklets(OUTPUT_DIR / dataset_run_id)
    for t in all_tracklets:
        if int(t.get("track_id", -1)) == track_id and _normalize_camera_id(str(t.get("camera_id", ""))) == cam:
            return t
    return None


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
    ranked_candidates: Optional[List[Tuple[float, Dict[str, Any]]]] = None,
    top_k_alternatives: int = 5,
) -> None:
    """Export cropped mp4 clips for every tracklet in matched trajectories."""
    if not trajectories and not ranked_candidates:
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

    # ── Export top-k "near miss" alternatives ──────────────────────────
    matched_keys = {
        (_normalize_camera_id(str(c.get("camera_id", ""))), int(c.get("track_id", -1)))
        for c in clips
        if int(c.get("track_id", -1)) >= 0
    }
    matched_global_ids = {
        str(tr.get("global_id", tr.get("id", "")))
        for tr in trajectories
    }

    alternatives_subfolder = "top5_alternatives"
    alternatives_dir = out_dir / alternatives_subfolder
    alternatives_dir.mkdir(parents=True, exist_ok=True)
    alternatives: List[Dict[str, Any]] = []

    target_alts = max(1, min(int(top_k_alternatives), 20))
    if ranked_candidates:
        ranked_sorted = sorted(ranked_candidates, key=lambda x: float(x[0]), reverse=True)
        for score, candidate in ranked_sorted:
            ok_count = sum(1 for a in alternatives if a.get("ok"))
            if ok_count >= target_alts:
                break

            gid_raw = candidate.get("global_id", candidate.get("id", ""))
            gid = str(gid_raw)
            if gid in matched_global_ids:
                continue

            candidate_tracklets = candidate.get("tracklets") if isinstance(candidate.get("tracklets"), list) else []
            exported = False
            for tr in candidate_tracklets:
                cam_raw = str(tr.get("camera_id") or tr.get("cameraId") or "")
                if cam_raw.startswith("query_"):
                    continue
                cam = _normalize_camera_id(cam_raw)
                tid = int(tr.get("track_id") or tr.get("trackId") or -1)
                if tid < 0 or not cam:
                    continue

                key = (cam, tid)
                if key in matched_keys:
                    continue

                gallery_tracklet = gallery_tracklets.get(key)
                if gallery_tracklet is None:
                    continue

                safe_cam = cam.replace("/", "_").replace("\\", "_")
                rank_num = ok_count + 1
                out_file = alternatives_dir / f"alt_{rank_num:02d}_global_{gid}_cam_{safe_cam}_track_{tid}.mp4"
                ok, msg = _export_tracklet_clip(gallery_run_id, gallery_tracklet, out_file)

                cam_set = {
                    _normalize_camera_id(str(tt.get("camera_id") or tt.get("cameraId") or ""))
                    for tt in candidate_tracklets
                    if str(tt.get("camera_id") or tt.get("cameraId") or "")
                }
                cam_set = {c for c in cam_set if c and not c.startswith("query_")}

                alternatives.append({
                    "rank": rank_num,
                    "global_id": int(gid_raw) if str(gid_raw).isdigit() else gid_raw,
                    "camera_id": cam,
                    "track_id": tid,
                    "score": round(float(score), 4),
                    "confidence": round(float(candidate.get("confidence", score)), 4),
                    "num_cameras": len(cam_set),
                    "file": out_file.name,
                    "ok": ok,
                    "msg": msg,
                })

                if ok:
                    exported = True
                break

            if not exported:
                continue

    summary["topAlternativesSubfolder"] = alternatives_subfolder
    summary["topAlternativesCount"] = sum(1 for a in alternatives if a.get("ok"))
    summary["topAlternatives"] = alternatives

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"[matched] {len(ok_clips)} clip(s) across {len(cameras_seen)} camera(s) -> {out_dir}",
        flush=True,
    )
