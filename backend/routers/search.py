import json
from typing import Any, Dict, List, Tuple

import numpy as np
from fastapi import APIRouter, HTTPException

from backend.config import OUTPUT_DIR
from backend.models.requests import SearchRequest
from backend.services.video_service import _normalize_camera_id
from backend.state import uploaded_videos, video_to_latest_run

router = APIRouter()


@router.post("/api/search/tracklet")
async def search_by_tracklet(request: SearchRequest):
    """Search the gallery for vehicles visually similar to the selected probe tracklet."""
    print(f"\n[UI Request] Search tracklet payload: {request.dict()}")
    top_k = max(1, min(request.topK, 200))

    probe_video_id = request.probeVideoId
    if probe_video_id and probe_video_id in uploaded_videos:
        probe_run_id = video_to_latest_run.get(probe_video_id)
    else:
        probe_run_id = None

    if not probe_run_id:
        raise HTTPException(
            status_code=400,
            detail="Probe video has not been processed yet (run Stage 1 first).",
        )

    probe_stage2_dir = OUTPUT_DIR / probe_run_id / "stage2"
    probe_emb_path = probe_stage2_dir / "embeddings.npy"
    probe_idx_path = probe_stage2_dir / "embedding_index.json"

    if not probe_emb_path.exists() or not probe_idx_path.exists():
        raise HTTPException(
            status_code=400,
            detail="Probe embeddings not found. Run Stage 2 on the probe video first.",
        )

    gallery_run_id = request.galleryRunId
    if not gallery_run_id:
        raise HTTPException(
            status_code=400,
            detail="No galleryRunId provided. Select a preprocessed dataset first.",
        )

    gallery_stage2_dir = OUTPUT_DIR / gallery_run_id / "stage2"
    gallery_emb_path = gallery_stage2_dir / "embeddings.npy"
    gallery_idx_path = gallery_stage2_dir / "embedding_index.json"
    traj_path = OUTPUT_DIR / gallery_run_id / "stage4" / "global_trajectories.json"

    if not gallery_emb_path.exists() or not gallery_idx_path.exists():
        raise HTTPException(
            status_code=400,
            detail="Gallery embeddings not found. Process the dataset (Stage 2-4) first.",
        )

    try:
        probe_emb = np.load(probe_emb_path)
        with open(probe_idx_path) as f:
            probe_idx = json.load(f)

        gallery_emb = np.load(gallery_emb_path)
        with open(gallery_idx_path) as f:
            gallery_idx = json.load(f)

        probe_rows = [
            i
            for i, x in enumerate(probe_idx)
            if int(x.get("track_id", -1)) == request.trackletId
        ]

        if not probe_rows:
            return {
                "success": True,
                "data": [],
                "message": "No embeddings found for that track ID in the probe run.",
            }

        probe_feats = probe_emb[probe_rows].astype(np.float32, copy=False)
        probe_norms = np.linalg.norm(probe_feats, axis=1, keepdims=True)
        probe_feats = probe_feats / np.maximum(probe_norms, 1e-8)

        gallery_groups: Dict[tuple, List[int]] = {}
        for i, x in enumerate(gallery_idx):
            key = (
                _normalize_camera_id(str(x.get("camera_id", ""))),
                int(x.get("track_id", -1)),
            )
            if key[1] < 0:
                continue
            gallery_groups.setdefault(key, []).append(i)

        scored: List[Tuple[float, str, int, str]] = []
        for (cam, tid), rows in gallery_groups.items():
            g_feats = gallery_emb[rows].astype(np.float32, copy=False)
            g_norms = np.linalg.norm(g_feats, axis=1, keepdims=True)
            g_feats = g_feats / np.maximum(g_norms, 1e-8)

            sim_mat = np.dot(probe_feats, g_feats.T)
            best_per_probe = sim_mat.max(axis=1)
            score = float(np.mean(best_per_probe))
            scored.append((score, cam, tid, gallery_run_id))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:top_k]

        traj_by_tracklet: Dict[tuple, int] = {}
        if traj_path.exists():
            try:
                trajectories = json.loads(traj_path.read_text())
                for traj in trajectories:
                    g_id = traj.get("global_id", -1)
                    for tr in traj.get("tracklets", []):
                        cam = _normalize_camera_id(
                            str(tr.get("camera_id") or tr.get("cameraId") or "")
                        )
                        tid = int(tr.get("track_id") or tr.get("trackId") or -1)
                        if tid >= 0:
                            traj_by_tracklet[(cam, tid)] = g_id
            except Exception:
                pass

        results = []
        for rank, (score, cam, tid, run_id_ref) in enumerate(top):
            global_id = traj_by_tracklet.get((cam, tid))
            results.append(
                {
                    "rank": rank + 1,
                    "score": round(score, 4),
                    "cameraId": cam,
                    "trackletId": tid,
                    "globalId": global_id,
                    "runId": run_id_ref,
                }
            )

        return {"success": True, "data": results}

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}")
