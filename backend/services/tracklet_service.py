"""Tracklet loading, run resolution, and selected-tracklet summaries."""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from backend.config import OUTPUT_DIR
from backend.services.logging_service import _timeline_debug
from backend.services.video_service import (
    _detect_camera_for_video,
    _extract_camera_id,
    _normalize_camera_id,
)
from backend.state import uploaded_videos, video_to_latest_run


def _load_tracklets(camera_id: str, run_dir: Path) -> List[Dict[str, Any]]:
    path = run_dir / "stage1" / f"tracklets_{camera_id}.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())


def _load_all_stage1_tracklets(run_dir: Path) -> List[Dict[str, Any]]:
    stage1_dir = run_dir / "stage1"
    if not stage1_dir.exists():
        return []

    all_tracklets: List[Dict[str, Any]] = []
    for file_path in sorted(stage1_dir.glob("tracklets_*.json")):
        try:
            payload = json.loads(file_path.read_text())
            if isinstance(payload, list):
                all_tracklets.extend(payload)
        except Exception:
            continue

    return all_tracklets


def _find_tracklet_in_run(
    run_id: str,
    track_id: int,
    preferred_camera_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Find a stage1 tracklet by id (and optionally camera) inside a run."""
    run_dir = OUTPUT_DIR / run_id
    tracklets = _load_all_stage1_tracklets(run_dir)
    preferred_norm = _normalize_camera_id(preferred_camera_id) if preferred_camera_id else None

    best: Optional[Dict[str, Any]] = None
    for t in tracklets:
        try:
            tid = int(t.get("track_id", -1))
        except Exception:
            continue
        if tid != track_id:
            continue
        if preferred_norm is None:
            return t
        cam_norm = _normalize_camera_id(str(t.get("camera_id", "")))
        if cam_norm == preferred_norm:
            return t
        if best is None:
            best = t
    return best


def _run_dir_for_video(video_id: str) -> Optional[Path]:
    run_id = video_to_latest_run.get(video_id)
    if not run_id:
        return None
    run_dir = OUTPUT_DIR / run_id
    if not run_dir.exists():
        return None
    return run_dir


def _persist_probe_link(video_id: str, run_id: str) -> None:
    """Write video_id → run_id mapping to disk so it survives server restarts."""
    try:
        link_path = OUTPUT_DIR / run_id / "probe_video_id.txt"
        link_path.parent.mkdir(parents=True, exist_ok=True)
        link_path.write_text(video_id)
    except Exception as exc:
        print(f"[WARN] _persist_probe_link failed: {exc}")


def _confidence_for_tracklet_frame(frame: Dict[str, Any], tracklet: Dict[str, Any]) -> float:
    """Return detection confidence for API/UI.

    Interpolated gap-fill frames are stored with confidence 0 in stage1 JSON; for display
    we substitute the mean confidence of real (non-zero) frames in the same tracklet.
    """
    c = float(frame.get("confidence", 0.0))
    if c > 1e-6:
        return c
    frames = tracklet.get("frames") or []
    vals = [
        float(f.get("confidence", 0.0))
        for f in frames
        if float(f.get("confidence", 0.0)) > 1e-6
    ]
    if vals:
        return float(sum(vals) / len(vals))
    return 0.0


def _tracklets_to_detections(
    tracklets: List[Dict[str, Any]],
    frame_id: Optional[int],
) -> List[Dict[str, Any]]:
    detections: List[Dict[str, Any]] = []

    for tracklet in tracklets:
        track_id = tracklet.get("track_id")
        class_id = tracklet.get("class_id")
        class_name = tracklet.get("class_name")

        frames = tracklet.get("frames", [])
        for frame in frames:
            this_frame_id = int(frame.get("frame_id", 0))
            if frame_id is not None and this_frame_id != frame_id:
                continue

            detections.append(
                {
                    "id": f"det-{track_id}-{this_frame_id}",
                    "bbox": frame.get("bbox", [0, 0, 0, 0]),
                    "classId": class_id,
                    "className": class_name,
                    "confidence": _confidence_for_tracklet_frame(frame, tracklet),
                    "frameId": this_frame_id,
                    "trackId": track_id,
                }
            )

            if frame_id is None:
                break

    return detections


def _build_selected_tracklet_summaries(
    probe_run_id: str, selected_nums: set
) -> List[Dict[str, Any]]:
    """Load stage-1 tracklets for *probe_run_id* and return summary dicts for
    any tracklet whose track_id is in *selected_nums*.  The shape matches what
    ``buildTracksFromSummary`` in the frontend expects."""
    run_dir = OUTPUT_DIR / probe_run_id
    tracklets = _load_all_stage1_tracklets(run_dir)
    _timeline_debug(
        "[Timeline Fallback] Building selected summaries:",
        {
            "probeRunId": probe_run_id,
            "selectedNums": sorted(list(selected_nums)),
            "stage1TrackletCount": len(tracklets),
            "stage1Path": str((run_dir / "stage1").as_posix()),
        },
    )
    summaries: List[Dict[str, Any]] = []
    for t in tracklets:
        try:
            track_id = int(t.get("track_id", -1))
        except Exception:
            continue

        if track_id not in selected_nums:
            continue
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
        summaries.append({
            "id": t.get("track_id"),
            "cameraId": str(t.get("camera_id", "unknown")),
            "startFrame": frames[0].get("frame_id", 0),
            "endFrame": frames[-1].get("frame_id", 0),
            "numFrames": len(frames),
            "className": t.get("class_name"),
            "representativeFrame": int(mid_frame.get("frame_id", 0)),
            "representativeBbox": mid_frame.get("bbox", [0, 0, 0, 0]),
            "sampleFrames": [
                {"frameId": int(sf.get("frame_id", 0)), "bbox": sf.get("bbox", [0, 0, 0, 0])}
                for sf in sample_frames_data
            ],
        })

    _timeline_debug(
        "[Timeline Fallback] Selected summaries built:",
        {
            "probeRunId": probe_run_id,
            "selectedNums": sorted(list(selected_nums)),
            "summaryCount": len(summaries),
        },
    )
    return summaries


def _resolve_probe_run_id_for_video(
    video_id: str, selected_nums: set
) -> Optional[str]:
    """Resolve the best probe run for a video id.
    Prefers a run that actually contains the selected track ids in stage-1
    tracklets. This avoids stale in-memory mappings after backend restarts.
    """
    _timeline_debug(
        "[Timeline Resolve] Resolving probe run:",
        {"videoId": video_id, "selectedNums": sorted(list(selected_nums))},
    )

    candidate_meta: Dict[str, Dict[str, Any]] = {}

    def _candidate_mtime(run_id: str) -> float:
        run_dir = OUTPUT_DIR / run_id
        link_path = run_dir / "probe_video_id.txt"
        try:
            if link_path.exists():
                return float(link_path.stat().st_mtime)
            return float(run_dir.stat().st_mtime)
        except Exception:
            return 0.0

    def _upsert_candidate(run_id: str, source: str, mtime_hint: Optional[float] = None) -> None:
        if not run_id:
            return
        item = candidate_meta.get(run_id)
        if item is None:
            candidate_meta[run_id] = {
                "runId": run_id,
                "mtime": float(mtime_hint if mtime_hint is not None else _candidate_mtime(run_id)),
                "sources": [source],
            }
            return

        item["mtime"] = max(
            float(item.get("mtime", 0.0)),
            float(mtime_hint if mtime_hint is not None else _candidate_mtime(run_id)),
        )
        sources = list(item.get("sources", []))
        if source not in sources:
            sources.append(source)
        item["sources"] = sources

    mapped = video_to_latest_run.get(video_id)
    if mapped:
        _upsert_candidate(mapped, "memory_map")

    if OUTPUT_DIR.exists():
        linked: List[tuple] = []
        for link_file in OUTPUT_DIR.glob("*/probe_video_id.txt"):
            try:
                vid_id = link_file.read_text().strip()
                if vid_id == video_id:
                    linked.append((link_file.stat().st_mtime, link_file.parent.name))
            except Exception:
                continue
        linked.sort(key=lambda x: x[0], reverse=True)
        for mtime, run_id in linked:
            _upsert_candidate(run_id, "probe_link", float(mtime))

    ordered_candidate_ids = [
        x["runId"]
        for x in sorted(
            candidate_meta.values(),
            key=lambda x: float(x.get("mtime", 0.0)),
            reverse=True,
        )
    ]
    _timeline_debug(
        "[Timeline Resolve] Candidate runs:",
        {
            "videoId": video_id,
            "candidateCount": len(ordered_candidate_ids),
            "candidates": [
                {
                    "runId": rid,
                    "sources": candidate_meta.get(rid, {}).get("sources", []),
                    "mtime": candidate_meta.get(rid, {}).get("mtime", 0.0),
                }
                for rid in ordered_candidate_ids
            ],
        },
    )

    for run_id in ordered_candidate_ids:
        run_dir = OUTPUT_DIR / run_id
        if not (run_dir / "stage1").exists():
            _timeline_debug(
                "[Timeline Resolve] Reject candidate (missing stage1):",
                {"runId": run_id},
            )
            continue
        tracklets = _load_all_stage1_tracklets(run_dir)
        track_ids = {
            int(t.get("track_id", -1))
            for t in tracklets
            if int(t.get("track_id", -1)) >= 0
        }
        if not selected_nums or bool(track_ids.intersection(selected_nums)):
            _timeline_debug(
                "[Timeline Resolve] Accepted candidate:",
                {
                    "runId": run_id,
                    "stage1TrackletCount": len(tracklets),
                    "matchedSelected": sorted(list(track_ids.intersection(selected_nums))),
                },
            )
            return run_id
        _timeline_debug(
            "[Timeline Resolve] Reject candidate (selected IDs missing):",
            {
                "runId": run_id,
                "selectedNums": sorted(list(selected_nums)),
                "sampleTrackIds": sorted(list(track_ids))[:10],
            },
        )

    if selected_nums and OUTPUT_DIR.exists():
        max_scan = 40
        preferred_cam = None
        if video_id in uploaded_videos:
            preferred_cam = _normalize_camera_id(
                _detect_camera_for_video(uploaded_videos[video_id], None)
            )

        run_dirs = [p for p in OUTPUT_DIR.iterdir() if p.is_dir()]
        run_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        best_pick: Optional[tuple] = None

        for run_dir in run_dirs[:max_scan]:
            stage1_dir = run_dir / "stage1"
            if not stage1_dir.exists():
                continue
            run_id = run_dir.name
            tracklets = _load_all_stage1_tracklets(run_dir)
            if not tracklets:
                continue

            matched_count = 0
            cam_match = False
            for t in tracklets:
                try:
                    tid = int(t.get("track_id", -1))
                except Exception:
                    continue
                if tid in selected_nums:
                    matched_count += 1
                if preferred_cam and _normalize_camera_id(str(t.get("camera_id", ""))) == preferred_cam:
                    cam_match = True

            if matched_count <= 0:
                continue

            score = matched_count + (1000 if cam_match else 0)
            mtime = float(run_dir.stat().st_mtime)
            if best_pick is None or score > best_pick[0] or (score == best_pick[0] and mtime > best_pick[1]):
                best_pick = (score, mtime, run_id)

        if best_pick is not None:
            _timeline_debug(
                "[Timeline Resolve] Selected by broad scan:",
                {
                    "runId": best_pick[2],
                    "score": best_pick[0],
                    "preferredCamera": preferred_cam,
                    "scanLimit": max_scan,
                },
            )
            return best_pick[2]

    fallback = ordered_candidate_ids[0] if ordered_candidate_ids else None
    _timeline_debug(
        "[Timeline Resolve] No exact selected-id candidate; using fallback:",
        {"fallbackRunId": fallback},
    )
    return fallback


def _build_tracklet_lookup(run_id: str) -> Dict[Tuple[str, int], Dict[str, Any]]:
    """Index all stage1 tracklets by (normalised_camera, track_id). First-seen wins."""
    lookup: Dict[Tuple[str, int], Dict[str, Any]] = {}
    run_dir = OUTPUT_DIR / run_id
    for t in _load_all_stage1_tracklets(run_dir):
        cam = _normalize_camera_id(str(t.get("camera_id", "")))
        tid = int(t.get("track_id", -1))
        if cam and tid >= 0 and (cam, tid) not in lookup:
            lookup[(cam, tid)] = t
    return lookup


def _build_tracklet_embedding_bank(run_id: str) -> Dict[Tuple[str, int], np.ndarray]:
    """Mean-pool stage2 embeddings per (camera, track_id) and L2-normalise."""
    bank: Dict[Tuple[str, int], np.ndarray] = {}
    stage2_dir = OUTPUT_DIR / run_id / "stage2"
    emb_path = stage2_dir / "embeddings.npy"
    idx_path = stage2_dir / "embedding_index.json"
    if not emb_path.exists() or not idx_path.exists():
        return bank

    emb = np.load(emb_path)
    with open(idx_path, "r", encoding="utf-8") as f:
        idx = json.load(f)

    if not isinstance(idx, list) or emb.ndim != 2 or emb.shape[0] == 0:
        return bank

    groups: Dict[Tuple[str, int], List[int]] = {}
    for i, row in enumerate(idx):
        cam = _normalize_camera_id(str(row.get("camera_id", "")))
        tid = int(row.get("track_id", -1))
        if not cam or tid < 0:
            continue
        groups.setdefault((cam, tid), []).append(i)

    for key, rows in groups.items():
        vec = emb[rows].mean(axis=0).astype(np.float32, copy=False)
        n = float(np.linalg.norm(vec))
        if n <= 1e-8:
            continue
        bank[key] = vec / n

    return bank


def _build_tracklet_global_map(
    run_id: str,
) -> Tuple[
    Dict[Tuple[str, int], Optional[int]],
    Dict[int, int],
    Dict[Tuple[str, int], Dict[str, float]],
]:
    """Map tracklets to global IDs, camera counts, and time spans.

    Returns ``(key_to_gid, gid_to_cam_count, key_to_span)``.
    """
    key_to_gid: Dict[Tuple[str, int], Optional[int]] = {}
    gid_to_cam_count: Dict[int, int] = {}
    key_to_span: Dict[Tuple[str, int], Dict[str, float]] = {}
    traj_path = OUTPUT_DIR / run_id / "stage4" / "global_trajectories.json"
    if not traj_path.exists():
        return key_to_gid, gid_to_cam_count, key_to_span

    try:
        trajectories = json.loads(traj_path.read_text(encoding="utf-8"))
    except Exception:
        return key_to_gid, gid_to_cam_count, key_to_span

    if not isinstance(trajectories, list):
        return key_to_gid, gid_to_cam_count, key_to_span

    for traj in trajectories:
        try:
            gid_raw = traj.get("global_id", traj.get("globalId"))
            gid = int(gid_raw) if gid_raw is not None else None
        except Exception:
            gid = None

        tracklets = traj.get("tracklets") if isinstance(traj.get("tracklets"), list) else []
        timeline = traj.get("timeline") if isinstance(traj.get("timeline"), list) else []
        cams: set = set()
        for tr in tracklets:
            cam = _normalize_camera_id(str(tr.get("camera_id") or tr.get("cameraId") or ""))
            tid = int(tr.get("track_id") or tr.get("trackId") or -1)
            if not cam or tid < 0:
                continue
            key_to_gid[(cam, tid)] = gid
            cams.add(cam)

        for tl in timeline:
            cam = _normalize_camera_id(str(tl.get("camera_id") or tl.get("cameraId") or ""))
            tid = int(tl.get("track_id") or tl.get("trackId") or -1)
            if not cam or tid < 0:
                continue
            start = float(tl.get("start", 0.0) or 0.0)
            end = float(tl.get("end", start) or start)
            key_to_span[(cam, tid)] = {
                "start": start,
                "end": max(end, start + 0.1),
            }

        if gid is not None and cams:
            gid_to_cam_count[gid] = len(cams)

    return key_to_gid, gid_to_cam_count, key_to_span
