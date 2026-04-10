import copy
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastapi import APIRouter, HTTPException

from backend.config import OUTPUT_DIR
from backend.models.requests import TimelineQueryRequest
from backend.services.clip_service import _export_matched_clips, _export_selected_clips
from backend.services.debug_service import _export_timeline_debug_bundle
from backend.services.logging_service import _timeline_debug
from backend.services.tracklet_service import (
    _build_selected_tracklet_summaries,
    _resolve_probe_run_id_for_video,
    _run_dir_for_video,
)
from backend.services.video_service import (
    _detect_camera_for_video,
    _normalize_camera_id,
    _parse_selected_track_nums,
)
from backend.state import uploaded_videos, video_to_latest_run

router = APIRouter()


@router.post("/api/timeline/query")
async def query_timeline(request: TimelineQueryRequest):
    """Resolve selected Stage-2 tracklets into Stage-4 matched trajectories."""
    _timeline_debug("[UI Request] Timeline Query payload:", request.dict())

    if request.videoId not in uploaded_videos:
        raise HTTPException(status_code=404, detail="Video not found")

    request_payload = request.dict()

    selected_nums = _parse_selected_track_nums(request.selectedTrackIds)
    _timeline_debug(
        "[UI Request] Timeline Query extracted selected track IDs:",
        {"selectedNums": sorted(list(selected_nums))},
    )

    resolved_probe_run_id = _resolve_probe_run_id_for_video(request.videoId, selected_nums)
    probe_run_id_for_summaries = resolved_probe_run_id or request.runId

    def _ensure_selected_summaries_nonempty(
        summaries: List[Dict[str, Any]],
        selected_ids: set,
        current_probe_run_id: str,
    ) -> Tuple[List[Dict[str, Any]], str, bool]:
        """Final guard to prevent blank timeline when a valid selection exists."""
        if not selected_ids or summaries:
            return summaries, current_probe_run_id, False

        video_run_dir = _run_dir_for_video(request.videoId)
        if video_run_dir is None:
            return summaries, current_probe_run_id, False

        video_run_id = video_run_dir.name
        if video_run_id == current_probe_run_id:
            return summaries, current_probe_run_id, False

        _timeline_debug(
            "[UI Request] Timeline Query final fallback using video-mapped run:",
            {
                "previousProbeRunId": current_probe_run_id,
                "videoMappedRunId": video_run_id,
                "selectedNums": sorted(list(selected_ids)),
            },
        )
        fallback_summaries = _build_selected_tracklet_summaries(video_run_id, selected_ids)
        if fallback_summaries:
            return fallback_summaries, video_run_id, True
        return summaries, current_probe_run_id, False

    _timeline_debug(
        "[UI Request] Timeline Query resolved probe run:",
        {
            "resolvedProbeRunId": resolved_probe_run_id,
            "mappedProbeRunId": video_to_latest_run.get(request.videoId),
            "queryRunId": request.runId,
        },
    )

    if not selected_nums:
        response_payload = {
            "success": True,
            "data": {
                "stage4Available": False,
                "mode": "no_selection",
                "message": "No selected tracklets were provided",
                "trajectories": [],
                "selectedTracklets": [],
                "diagnostics": {
                    "selectedCount": 0,
                    "selectedKeyCount": 0,
                    "trajectoryCount": 0,
                    "matchedTrajectoryCount": 0,
                },
            },
        }
        return response_payload

    video_info = uploaded_videos[request.videoId]
    resolved_cam = _normalize_camera_id(_detect_camera_for_video(video_info, None))

    diag: Dict[str, Any] = {
        "selectedCount": len(selected_nums),
        "trajectoryCount": 0,
        "matchedTrajectoryCount": 0,
        "parsedNums": list(selected_nums),
        "rawIdsReceived": request.selectedTrackIds,
        "resolvedCamera": resolved_cam,
        "resolvedProbeRunId": resolved_probe_run_id,
        "selectedTrackletsSourceRun": probe_run_id_for_summaries,
    }

    traj_path = OUTPUT_DIR / request.runId / "stage4" / "global_trajectories.json"
    if not traj_path.exists():
        selected_summaries = _build_selected_tracklet_summaries(
            probe_run_id_for_summaries, selected_nums
        )
        if selected_nums and not selected_summaries:
            retry_probe_run = _resolve_probe_run_id_for_video(request.videoId, selected_nums)
            if retry_probe_run and retry_probe_run != probe_run_id_for_summaries:
                _timeline_debug(
                    "[UI Request] Timeline Query retrying selected fallback with alternate run:",
                    {
                        "previousProbeRunId": probe_run_id_for_summaries,
                        "retryProbeRunId": retry_probe_run,
                    },
                )
                probe_run_id_for_summaries = retry_probe_run
                selected_summaries = _build_selected_tracklet_summaries(
                    probe_run_id_for_summaries, selected_nums
                )

        selected_summaries, probe_run_id_for_summaries, final_fallback_used = (
            _ensure_selected_summaries_nonempty(
                selected_summaries,
                selected_nums,
                probe_run_id_for_summaries,
            )
        )

        diag["selectedTrackletsSourceRun"] = probe_run_id_for_summaries
        diag["selectedTrackletsReturned"] = len(selected_summaries)
        diag["selectedTrackletsFinalFallbackUsed"] = bool(final_fallback_used)
        _timeline_debug(
            "[UI Request] Timeline Query fallback summaries (needs_association):",
            {
                "count": len(selected_summaries),
                "probeRunId": probe_run_id_for_summaries,
                "selectedNums": sorted(list(selected_nums)),
            },
        )
        response_payload = {
            "success": True,
            "data": {
                "stage4Available": False,
                "mode": "needs_association",
                "message": "Stage 4 artifacts missing for this run",
                "trajectories": [],
                "selectedTracklets": selected_summaries,
                "diagnostics": diag,
            },
        }

        debug_bundle_path = _export_timeline_debug_bundle(request_payload, response_payload)
        if debug_bundle_path is not None:
            response_payload["data"].setdefault("diagnostics", {})[
                "debugExportPath"
            ] = str(debug_bundle_path.as_posix())
            _timeline_debug(
                "[UI Request] Timeline debug bundle exported:",
                {"path": str(debug_bundle_path.as_posix())},
            )
        if selected_nums and probe_run_id_for_summaries:
            try:
                _export_selected_clips(probe_run_id_for_summaries, selected_nums)
            except Exception as _sc_err:
                print(f"[selected] clip export failed: {_sc_err}", flush=True)
        return response_payload

    trajectories: List[Dict[str, Any]] = json.loads(traj_path.read_text())
    if not isinstance(trajectories, list):
        trajectories = []

    diag["trajectoryCount"] = len(trajectories)

    filtered: List[Dict[str, Any]] = []
    if selected_nums:
        probe_run_id = probe_run_id_for_summaries
        probe_stage2_dir = OUTPUT_DIR / probe_run_id / "stage2"
        gallery_stage2_dir = OUTPUT_DIR / request.runId / "stage2"

        probe_emb_path = probe_stage2_dir / "embeddings.npy"
        probe_idx_path = probe_stage2_dir / "embedding_index.json"
        gallery_emb_path = gallery_stage2_dir / "embeddings.npy"
        gallery_idx_path = gallery_stage2_dir / "embedding_index.json"

        diag["probeRunId"] = probe_run_id
        diag["probeEmbeddingsAvailable"] = (
            probe_emb_path.exists() and probe_idx_path.exists()
        )
        diag["galleryEmbeddingsAvailable"] = (
            gallery_emb_path.exists() and gallery_idx_path.exists()
        )

        if diag["probeEmbeddingsAvailable"] and diag["galleryEmbeddingsAvailable"]:
            diag["search_mode"] = "visual_reid_strict"
            try:
                probe_emb = np.load(probe_emb_path)
                with open(probe_idx_path) as f:
                    probe_idx = json.load(f)

                gallery_emb = np.load(gallery_emb_path)
                with open(gallery_idx_path) as f:
                    gallery_idx = json.load(f)

                probe_dim = (
                    int(probe_emb.shape[1])
                    if probe_emb.ndim == 2 and probe_emb.shape[0] > 0
                    else None
                )
                gallery_dim = (
                    int(gallery_emb.shape[1])
                    if gallery_emb.ndim == 2 and gallery_emb.shape[0] > 0
                    else None
                )
                diag["probeEmbeddingDim"] = probe_dim
                diag["galleryEmbeddingDim"] = gallery_dim

                if probe_dim is not None and gallery_dim is not None and probe_dim != gallery_dim:
                    if probe_dim > gallery_dim:
                        pca_model_path = Path("models/reid/pca_transform.pkl")
                        try:
                            with open(pca_model_path, "rb") as _pf:
                                _pca_obj = pickle.load(_pf)
                            projected = _pca_obj.transform(probe_emb.astype(np.float32))
                            if projected.shape[1] == gallery_dim:
                                probe_emb = projected.astype(np.float32)
                                probe_dim = gallery_dim
                                diag["probeEmbeddingDim"] = probe_dim
                                diag["pcaProjectionApplied"] = True
                                print(
                                    f"[timeline] PCA projection applied: probe {projected.shape}",
                                    flush=True,
                                )
                            else:
                                diag["pcaProjectionApplied"] = False
                                diag["pcaProjectedDim"] = projected.shape[1]
                        except Exception as _pca_err:
                            diag["pcaProjectionError"] = str(_pca_err)
                            print(f"[timeline] PCA projection failed: {_pca_err}", flush=True)

                if probe_dim is None or gallery_dim is None or probe_dim != gallery_dim:
                    diag["search_mode"] = "embedding_dim_mismatch"
                    diag["search_error"] = (
                        f"Embedding dimension mismatch: probe={probe_dim}, gallery={gallery_dim}"
                    )
                    _timeline_debug(
                        "[UI Request] Timeline Query embedding dimension mismatch:",
                        {
                            "probeRunId": probe_run_id,
                            "galleryRunId": request.runId,
                            "probeDim": probe_dim,
                            "galleryDim": gallery_dim,
                        },
                    )
                    filtered = []
                else:
                    gallery_map: Dict[tuple, List[int]] = {}
                    for i, x in enumerate(gallery_idx):
                        cam = _normalize_camera_id(str(x.get("camera_id", "")))
                        tid = int(x.get("track_id", -1))
                        if tid < 0 or not cam:
                            continue
                        gallery_map.setdefault((cam, tid), []).append(i)

                    probe_indices = [
                        i
                        for i, x in enumerate(probe_idx)
                        if int(x.get("track_id", -1)) in selected_nums
                    ]
                    diag["probeFrameCount"] = len(probe_indices)

                    if probe_indices:
                        probe_feats = probe_emb[probe_indices].astype(np.float32, copy=False)
                        probe_norms = np.linalg.norm(probe_feats, axis=1, keepdims=True)
                        probe_feats = probe_feats / np.maximum(probe_norms, 1e-8)

                        probe_classes: List[int] = []
                        for i in probe_indices:
                            c = probe_idx[i].get("class_id")
                            if c is not None:
                                probe_classes.append(int(c))
                        dominant_probe_class = None
                        if probe_classes:
                            dominant_probe_class = max(set(probe_classes), key=probe_classes.count)
                        diag["probeClassId"] = dominant_probe_class

                        scored_trajectories: List[Tuple[float, Dict[str, Any]]] = []
                        for traj in trajectories:
                            tracklets = (
                                traj.get("tracklets", []) if isinstance(traj, dict) else []
                            )
                            t_indices: List[int] = []
                            t_classes: List[int] = []
                            for tr in tracklets:
                                cam = _normalize_camera_id(
                                    str(tr.get("camera_id") or tr.get("cameraId") or "")
                                )
                                tid = int(tr.get("track_id") or tr.get("trackId") or -1)
                                rows = gallery_map.get((cam, tid), [])
                                if rows:
                                    t_indices.extend(rows)
                                class_id = tr.get("class_id")
                                if class_id is not None:
                                    t_classes.append(int(class_id))

                            if not t_indices:
                                continue

                            if dominant_probe_class is not None and t_classes:
                                dominant_traj_class = max(set(t_classes), key=t_classes.count)
                                if dominant_traj_class != dominant_probe_class:
                                    continue

                            t_feats = gallery_emb[t_indices].astype(np.float32, copy=False)
                            t_norms = np.linalg.norm(t_feats, axis=1, keepdims=True)
                            t_feats = t_feats / np.maximum(t_norms, 1e-8)

                            sim_mat = np.dot(probe_feats, t_feats.T)
                            best_per_probe = sim_mat.max(axis=1)
                            mean_best = float(np.mean(best_per_probe))
                            p25_best = float(np.percentile(best_per_probe, 25))

                            if mean_best >= 0.82 and p25_best >= 0.74:
                                score = mean_best
                                traj["confidence"] = score
                                traj["matchEvidence"] = {
                                    "meanBestFrameSimilarity": round(mean_best, 4),
                                    "p25BestFrameSimilarity": round(p25_best, 4),
                                    "probeFrames": int(probe_feats.shape[0]),
                                    "trajectoryFrames": int(t_feats.shape[0]),
                                }
                                scored_trajectories.append((score, traj))

                        scored_trajectories.sort(key=lambda x: x[0], reverse=True)
                        diag["visual_matches_scored"] = len(scored_trajectories)
                        filtered = [traj for _, traj in scored_trajectories]
                    else:
                        diag["search_mode"] = "probe_not_found"

            except Exception as e:
                diag["search_error"] = str(e)
                print(f"Visual search failed: {e}")

        elif not diag["probeEmbeddingsAvailable"]:
            diag["search_mode"] = "missing_probe_features"
        else:
            diag["search_mode"] = "missing_gallery_features"

        if not filtered and request.runId == probe_run_id:
            for traj in trajectories:
                tracklets = traj.get("tracklets", []) if isinstance(traj, dict) else []
                found = False
                for t in tracklets:
                    cam = _normalize_camera_id(
                        str(t.get("camera_id") or t.get("cameraId") or "")
                    )
                    tid = int(t.get("track_id") or t.get("trackId") or -1)
                    if tid in selected_nums:
                        found = True
                        break
                if found:
                    filtered.append(traj)
            if filtered:
                diag["search_mode"] = "exact_id_same_run"

    if filtered:
        mode = "matched"
        message = "Association loaded (query-matched)"
    else:
        mode = "empty"
        if diag.get("search_mode") == "missing_probe_features":
            message = "Probe embeddings are missing for this uploaded video run. Run Stage 2 on the probe video first."
        elif diag.get("search_mode") == "missing_gallery_features":
            message = "Gallery embeddings are missing for this run. Run Stage 2/4 for the gallery run first."
        elif diag.get("search_mode") == "probe_not_found":
            message = "Selected tracklets were not found in probe embeddings for this camera context."
        else:
            message = "Selected tracklets could not be resolved in current video/run context"

    diag["matchedTrajectoryCount"] = len(filtered)

    selected_summaries = _build_selected_tracklet_summaries(
        probe_run_id_for_summaries, selected_nums
    )
    if selected_nums and not selected_summaries:
        retry_probe_run = _resolve_probe_run_id_for_video(request.videoId, selected_nums)
        if retry_probe_run and retry_probe_run != probe_run_id_for_summaries:
            _timeline_debug(
                "[UI Request] Timeline Query retrying selected fallback with alternate run:",
                {
                    "previousProbeRunId": probe_run_id_for_summaries,
                    "retryProbeRunId": retry_probe_run,
                },
            )
            probe_run_id_for_summaries = retry_probe_run
            selected_summaries = _build_selected_tracklet_summaries(
                probe_run_id_for_summaries, selected_nums
            )

    selected_summaries, probe_run_id_for_summaries, final_fallback_used = (
        _ensure_selected_summaries_nonempty(
            selected_summaries,
            selected_nums,
            probe_run_id_for_summaries,
        )
    )

    if selected_nums and not selected_summaries:
        _timeline_debug(
            "[UI Request] Timeline Query warning: selected IDs present but selectedTracklets empty",
            {
                "selectedNums": sorted(list(selected_nums)),
                "probeRunId": probe_run_id_for_summaries,
                "queryRunId": request.runId,
                "resolvedProbeRunId": resolved_probe_run_id,
            },
        )

    diag["selectedTrackletsSourceRun"] = probe_run_id_for_summaries
    diag["selectedTrackletsReturned"] = len(selected_summaries)
    diag["selectedTrackletsFinalFallbackUsed"] = bool(final_fallback_used)
    _timeline_debug(
        "[UI Request] Timeline Query selected fallback summaries:",
        {
            "count": len(selected_summaries),
            "probeRunId": probe_run_id_for_summaries,
            "selectedNums": sorted(list(selected_nums)),
            "mode": mode,
            "matchedTrajectoryCount": len(filtered),
        },
    )

    def _clean_trajectory_for_ui(traj: Dict[str, Any]) -> Dict[str, Any]:
        t = copy.copy(traj)
        for field in ("tracklets", "timeline"):
            entries = t.get(field)
            if isinstance(entries, list):
                seen: set = set()
                clean = []
                for entry in entries:
                    cam_raw = str(entry.get("camera_id") or entry.get("cameraId") or "")
                    if cam_raw.startswith("query_"):
                        continue
                    key = (cam_raw, entry.get("track_id") or entry.get("trackId"))
                    if key in seen:
                        continue
                    seen.add(key)
                    clean.append(entry)
                t[field] = clean
        return t

    cleaned_filtered = [_clean_trajectory_for_ui(t) for t in filtered]

    response_payload = {
        "success": True,
        "data": {
            "stage4Available": True,
            "mode": mode,
            "message": message,
            "trajectories": cleaned_filtered,
            "selectedTracklets": selected_summaries,
            "diagnostics": diag,
        },
    }

    debug_bundle_path = _export_timeline_debug_bundle(request_payload, response_payload)
    if debug_bundle_path is not None:
        response_payload["data"].setdefault("diagnostics", {})[
            "debugExportPath"
        ] = str(debug_bundle_path.as_posix())
        _timeline_debug(
            "[UI Request] Timeline debug bundle exported:",
            {"path": str(debug_bundle_path.as_posix())},
        )

    if selected_nums and probe_run_id_for_summaries:
        try:
            _export_selected_clips(probe_run_id_for_summaries, selected_nums)
        except Exception as _sc_err:
            print(f"[selected] clip export failed: {_sc_err}", flush=True)

    if filtered and probe_run_id_for_summaries:
        try:
            _export_matched_clips(probe_run_id_for_summaries, request.runId, filtered)
        except Exception as _mc_err:
            print(f"[matched] clip export failed: {_mc_err}", flush=True)

    return response_payload
