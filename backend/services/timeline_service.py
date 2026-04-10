"""TimelineService — business logic for the timeline query endpoint.

Extracted from ``backend/routers/timeline.py`` (Phase 3).  The router
retains only:
  - FastAPI boundary concerns (HTTP 404 guard, response shaping)
  - I/O side-effects (debug bundle export, clip exports)

This module raises ``ValueError`` on invalid input; never ``HTTPException``.
"""

from __future__ import annotations

import copy
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from backend.config import (
    OUTPUT_DIR,
    PCA_MODEL_PATH,
    SIMILARITY_THRESHOLD_MEAN,
    SIMILARITY_THRESHOLD_P25,
)
from backend.models.embedding import EmbeddingArtifact
from backend.models.requests import TimelineQueryRequest
from backend.repositories.dataset_repository import DatasetRepository
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


class TimelineService:
    """Resolves Stage-2 tracklets into Stage-4 matched trajectories.

    Args:
        repo: Repository used to read pipeline artefacts.  In production
              this is an ``InMemoryDatasetRepository``; in tests it is
              a mock that returns fixture data.
    """

    def __init__(self, repo: DatasetRepository) -> None:
        self._repo = repo

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def query(
        self,
        request: TimelineQueryRequest,
        uploaded_videos: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Execute the timeline query and return the response payload.

        Args:
            request:         Validated query request.
            uploaded_videos: Current in-memory video catalogue.

        Returns:
            Response payload dict with keys ``success`` and ``data``.

        Raises:
            ValueError: If the requested video_id is not in uploaded_videos.
        """
        if request.videoId not in uploaded_videos:
            raise ValueError(f"Video not found: {request.videoId!r}")

        selected_nums = _parse_selected_track_nums(request.selectedTrackIds)

        # ── No-selection fast path ───────────────────────────────────────
        if not selected_nums:
            return {
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

        # ── Resolve probe run ────────────────────────────────────────────
        resolved_probe_run_id = _resolve_probe_run_id_for_video(
            request.videoId, selected_nums
        )
        probe_run_id = resolved_probe_run_id or request.runId

        video_info = uploaded_videos[request.videoId]
        resolved_cam = _normalize_camera_id(
            _detect_camera_for_video(video_info, None)
        )

        diag: Dict[str, Any] = {
            "selectedCount": len(selected_nums),
            "trajectoryCount": 0,
            "matchedTrajectoryCount": 0,
            "parsedNums": list(selected_nums),
            "rawIdsReceived": request.selectedTrackIds,
            "resolvedCamera": resolved_cam,
            "resolvedProbeRunId": resolved_probe_run_id,
            "selectedTrackletsSourceRun": probe_run_id,
        }

        # ── Stage-4 missing ──────────────────────────────────────────────
        trajectories = self._repo.list_trajectories(request.runId)
        if trajectories is None:
            selected_summaries, probe_run_id = self._resolve_selected_summaries(
                request, selected_nums, probe_run_id
            )
            diag["selectedTrackletsSourceRun"] = probe_run_id
            diag["selectedTrackletsReturned"] = len(selected_summaries)
            return {
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

        diag["trajectoryCount"] = len(trajectories)

        # ── Visual ReID scoring ──────────────────────────────────────────
        filtered: List[Dict[str, Any]] = []
        filtered, diag = self._run_visual_search(
            request, selected_nums, probe_run_id, trajectories, diag
        )

        # Fallback to exact-ID match within the same run
        if not filtered and request.runId == probe_run_id:
            filtered = self._exact_id_fallback(
                selected_nums, trajectories
            )
            if filtered:
                diag["search_mode"] = "exact_id_same_run"

        # ── Resolve selected tracklet summaries ──────────────────────────
        selected_summaries, probe_run_id = self._resolve_selected_summaries(
            request, selected_nums, probe_run_id
        )
        diag["selectedTrackletsSourceRun"] = probe_run_id
        diag["selectedTrackletsReturned"] = len(selected_summaries)
        diag["matchedTrajectoryCount"] = len(filtered)

        # ── Build response ───────────────────────────────────────────────
        if filtered:
            mode = "matched"
            message = "Association loaded (query-matched)"
        else:
            mode = "empty"
            message = self._empty_message(diag)

        cleaned_filtered = [self._clean_trajectory_for_ui(t) for t in filtered]

        return {
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

    # ------------------------------------------------------------------
    # Embedding pair loading + PCA projection
    # ------------------------------------------------------------------

    def _load_embedding_pair(
        self,
        probe_run_id: str,
        gallery_run_id: str,
    ) -> Tuple[Optional[EmbeddingArtifact], Optional[EmbeddingArtifact]]:
        """Load probe and gallery embedding artefacts from the repository."""
        probe = self._repo.load_embedding_artifact(probe_run_id)
        gallery = (
            self._repo.load_embedding_artifact(gallery_run_id)
            if gallery_run_id != probe_run_id
            else probe
        )
        return probe, gallery

    def _apply_pca_projection(
        self,
        probe: EmbeddingArtifact,
        gallery: EmbeddingArtifact,
    ) -> EmbeddingArtifact:
        """Project probe embeddings to gallery dimensionality using the saved PCA.

        Returns the (possibly unchanged) probe artifact.  Only applies when
        ``probe.dim > gallery.dim`` and PCA model file exists.
        """
        if probe.dim <= gallery.dim:
            return probe
        pca_path = PCA_MODEL_PATH
        if not pca_path.exists():
            return probe
        try:
            with open(pca_path, "rb") as fh:
                pca_obj = pickle.load(fh)
            projected = pca_obj.transform(probe.embeddings.astype(np.float32))
            if projected.shape[1] == gallery.dim:
                return EmbeddingArtifact(
                    run_id=probe.run_id,
                    embeddings=projected.astype(np.float32),
                    index=probe.index,
                )
        except Exception:
            pass
        return probe

    # ------------------------------------------------------------------
    # Trajectory scoring
    # ------------------------------------------------------------------

    def _score_trajectories(
        self,
        probe: EmbeddingArtifact,
        gallery: EmbeddingArtifact,
        selected_nums: set,
        trajectories: List[Dict[str, Any]],
    ) -> Tuple[List[Tuple[float, Dict[str, Any]]], Dict[str, Any]]:
        """Cosine similarity scoring of trajectories against selected probes.

        Returns ``(scored_trajectories, extra_diag_fields)`` where
        ``scored_trajectories`` is sorted descending by score.
        """
        extra: Dict[str, Any] = {}

        # Build gallery row-index lookup: (cam, tid) → [row indices]
        gallery_map: Dict[Tuple[str, int], List[int]] = {}
        for i, x in enumerate(gallery.index):
            cam = _normalize_camera_id(str(x.get("camera_id", "")))
            tid = int(x.get("track_id", -1))
            if tid < 0 or not cam:
                continue
            gallery_map.setdefault((cam, tid), []).append(i)

        # Collect probe row indices matching selected tracks
        probe_indices = [
            i
            for i, x in enumerate(probe.index)
            if int(x.get("track_id", -1)) in selected_nums
        ]
        extra["probeFrameCount"] = len(probe_indices)

        if not probe_indices:
            extra["search_mode"] = "probe_not_found"
            return [], extra

        probe_feats = probe.embeddings[probe_indices].astype(np.float32, copy=False)
        probe_norms = np.linalg.norm(probe_feats, axis=1, keepdims=True)
        probe_feats = probe_feats / np.maximum(probe_norms, 1e-8)

        # Determine dominant class of probe for class-gating
        probe_classes = [
            int(probe.index[i]["class_id"])
            for i in probe_indices
            if "class_id" in probe.index[i]
        ]
        dominant_probe_class = (
            max(set(probe_classes), key=probe_classes.count) if probe_classes else None
        )
        extra["probeClassId"] = dominant_probe_class

        scored: List[Tuple[float, Dict[str, Any]]] = []

        for traj in trajectories:
            tracklets = traj.get("tracklets", []) if isinstance(traj, dict) else []
            t_indices: List[int] = []
            t_classes: List[int] = []

            for tr in tracklets:
                cam = _normalize_camera_id(
                    str(tr.get("camera_id") or tr.get("cameraId") or "")
                )
                tid = int(tr.get("track_id") or tr.get("trackId") or -1)
                rows = gallery_map.get((cam, tid), [])
                t_indices.extend(rows)
                if (c := tr.get("class_id")) is not None:
                    t_classes.append(int(c))

            if not t_indices:
                continue

            # Class gate: skip cross-class matches
            if dominant_probe_class is not None and t_classes:
                dominant_traj_class = max(set(t_classes), key=t_classes.count)
                if dominant_traj_class != dominant_probe_class:
                    continue

            t_feats = gallery.embeddings[t_indices].astype(np.float32, copy=False)
            t_norms = np.linalg.norm(t_feats, axis=1, keepdims=True)
            t_feats = t_feats / np.maximum(t_norms, 1e-8)

            sim_mat = np.dot(probe_feats, t_feats.T)
            best_per_probe = sim_mat.max(axis=1)
            mean_best = float(np.mean(best_per_probe))
            p25_best = float(np.percentile(best_per_probe, 25))

            if mean_best >= SIMILARITY_THRESHOLD_MEAN and p25_best >= SIMILARITY_THRESHOLD_P25:
                traj = copy.deepcopy(traj)
                traj["confidence"] = mean_best
                traj["matchEvidence"] = {
                    "meanBestFrameSimilarity": round(mean_best, 4),
                    "p25BestFrameSimilarity": round(p25_best, 4),
                    "probeFrames": int(probe_feats.shape[0]),
                    "trajectoryFrames": int(t_feats.shape[0]),
                }
                scored.append((mean_best, traj))

        scored.sort(key=lambda x: x[0], reverse=True)
        extra["visual_matches_scored"] = len(scored)
        return scored, extra

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_visual_search(
        self,
        request: TimelineQueryRequest,
        selected_nums: set,
        probe_run_id: str,
        trajectories: List[Dict[str, Any]],
        diag: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Attempt visual ReID search; update diag in-place and return matches."""
        probe, gallery = self._load_embedding_pair(probe_run_id, request.runId)

        diag["probeEmbeddingsAvailable"] = probe is not None
        diag["galleryEmbeddingsAvailable"] = gallery is not None

        if probe is None:
            diag["search_mode"] = "missing_probe_features"
            return [], diag

        if gallery is None:
            diag["search_mode"] = "missing_gallery_features"
            return [], diag

        diag["probeEmbeddingDim"] = probe.dim
        diag["galleryEmbeddingDim"] = gallery.dim

        if probe.dim != gallery.dim:
            # Try PCA projection (probe_dim > gallery_dim case)
            original_probe_dim = probe.dim
            probe = self._apply_pca_projection(probe, gallery)
            diag["pcaProjectionApplied"] = probe.dim == gallery.dim
            diag["probeEmbeddingDim"] = probe.dim
            if probe.dim != gallery.dim:
                diag["search_mode"] = "embedding_dim_mismatch"
                diag["search_error"] = (
                    f"Embedding dimension mismatch: probe={original_probe_dim},"
                    f" gallery={gallery.dim}"
                )
                return [], diag

        diag["search_mode"] = "visual_reid_strict"
        scored, extra = self._score_trajectories(
            probe, gallery, selected_nums, trajectories
        )
        diag.update(extra)
        return [t for _, t in scored], diag

    def _exact_id_fallback(
        self,
        selected_nums: set,
        trajectories: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Return trajectories that contain any selected track ID directly."""
        result = []
        for traj in trajectories:
            tracklets = traj.get("tracklets", []) if isinstance(traj, dict) else []
            for t in tracklets:
                tid = int(t.get("track_id") or t.get("trackId") or -1)
                if tid in selected_nums:
                    result.append(traj)
                    break
        return result

    def _resolve_selected_summaries(
        self,
        request: TimelineQueryRequest,
        selected_nums: set,
        probe_run_id: str,
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Build selected tracklet summaries with retry + final fallback."""
        summaries = _build_selected_tracklet_summaries(probe_run_id, selected_nums)

        if selected_nums and not summaries:
            retry = _resolve_probe_run_id_for_video(request.videoId, selected_nums)
            if retry and retry != probe_run_id:
                probe_run_id = retry
                summaries = _build_selected_tracklet_summaries(probe_run_id, selected_nums)

        if selected_nums and not summaries:
            video_run_dir = _run_dir_for_video(request.videoId)
            if video_run_dir is not None:
                video_run_id = video_run_dir.name
                if video_run_id != probe_run_id:
                    fallback = _build_selected_tracklet_summaries(video_run_id, selected_nums)
                    if fallback:
                        probe_run_id = video_run_id
                        summaries = fallback

        return summaries, probe_run_id

    @staticmethod
    def _clean_trajectory_for_ui(traj: Dict[str, Any]) -> Dict[str, Any]:
        """Strip query_ pseudo-cameras and deduplicate tracklet entries."""
        t = copy.copy(traj)
        for field in ("tracklets", "timeline"):
            entries = t.get(field)
            if not isinstance(entries, list):
                continue
            seen: set = set()
            clean = []
            for entry in entries:
                cam_raw = str(
                    entry.get("camera_id") or entry.get("cameraId") or ""
                )
                if cam_raw.startswith("query_"):
                    continue
                key = (cam_raw, entry.get("track_id") or entry.get("trackId"))
                if key in seen:
                    continue
                seen.add(key)
                clean.append(entry)
            t[field] = clean
        return t

    @staticmethod
    def _empty_message(diag: Dict[str, Any]) -> str:
        """Human-readable reason why no trajectories were found."""
        mode = diag.get("search_mode", "")
        if mode == "missing_probe_features":
            return (
                "Probe embeddings are missing for this uploaded video run. "
                "Run Stage 2 on the probe video first."
            )
        if mode == "missing_gallery_features":
            return (
                "Gallery embeddings are missing for this run. "
                "Run Stage 2/4 for the gallery run first."
            )
        if mode == "probe_not_found":
            return (
                "Selected tracklets were not found in probe embeddings "
                "for this camera context."
            )
        return "Selected tracklets could not be resolved in current video/run context"
