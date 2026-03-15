"""Forensic Re-Identification Search and Watchlist Matching.

Provides:
- query_by_embedding   : find all tracklet appearances matching a query crop/vector
- query_by_image       : extract embedding from an image crop and search
- watchlist_scan       : scan all trajectories for matches against a watchlist of
                         known subjects (vehicle images / embedding vectors)
- export_forensic_report: produce a structured, auditable report of all cross-camera
                          identities above a confidence threshold

These capabilities are designed for:
  - Law enforcement / intelligence agencies needing to trace a specific vehicle
    through a monitored area
  - Forensic analysts reviewing post-incident footage
  - Real-time alert generation when a watchlisted vehicle is detected
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from src.core.data_models import GlobalTrajectory


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """A single tracklet hit from a Re-ID search query."""
    rank: int
    tracklet_id: int
    camera_id: str
    start_time: float
    end_time: float
    similarity: float
    global_id: Optional[int] = None   # trajectory this tracklet belongs to
    trajectory_confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "camera_id": self.camera_id,
            "tracklet_id": self.tracklet_id,
            "global_trajectory_id": self.global_id,
            "trajectory_confidence": round(self.trajectory_confidence, 4),
            "similarity": round(self.similarity, 4),
            "first_frame_time_s": round(self.start_time, 3),
            "last_frame_time_s": round(self.end_time, 3),
            "duration_s": round(self.end_time - self.start_time, 3),
        }


@dataclass
class WatchlistHit:
    """A match between a watchlist subject and a tracked trajectory."""
    subject_id: str
    global_id: int
    similarity: float
    trajectory_confidence: float
    cameras_seen: List[str] = field(default_factory=list)
    first_seen: float = 0.0
    last_seen: float = 0.0

    @property
    def alert_level(self) -> str:
        """Operational alert level based on similarity × trajectory confidence."""
        score = self.similarity * self.trajectory_confidence
        if score >= 0.60:
            return "HIGH"
        elif score >= 0.40:
            return "MEDIUM"
        return "LOW"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject_id": self.subject_id,
            "alert_level": self.alert_level,
            "global_trajectory_id": self.global_id,
            "match_similarity": round(self.similarity, 4),
            "trajectory_confidence": round(self.trajectory_confidence, 4),
            "cameras_seen": self.cameras_seen,
            "first_seen_s": round(self.first_seen, 3),
            "last_seen_s": round(self.last_seen, 3),
        }


# ---------------------------------------------------------------------------
# Core search functions
# ---------------------------------------------------------------------------

class ForensicSearchEngine:
    """Re-ID search and watchlist engine backed by the Stage 3 FAISS index.

    Typical usage::

        engine = ForensicSearchEngine(faiss_index, embeddings, index_map, trajectories)
        results = engine.query_by_embedding(query_vec, top_k=20)
        hits = engine.watchlist_scan(watchlist_embeddings, threshold=0.55)
        engine.export_forensic_report(output_dir)
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        index_map: List[Dict[str, Any]],
        trajectories: List[GlobalTrajectory],
        faiss_index=None,
    ):
        """
        Args:
            embeddings: (N, D) L2-normalised embedding matrix from Stage 2.
            index_map: List of dicts with keys 'track_id', 'camera_id', 'class_id'.
            trajectories: Stage 4 GlobalTrajectory results.
            faiss_index: Optional FAISSIndex for fast ANN search; falls back to
                         brute-force matrix multiplication if None.
        """
        self.embeddings = embeddings.astype(np.float32)
        self.index_map = index_map
        self.trajectories = trajectories
        self.faiss_index = faiss_index

        # Build reverse lookup: (camera_id, track_id) → global_id + confidence
        self._tracklet_to_traj: Dict[Tuple[str, int], Tuple[int, float]] = {}
        # Pre-built time lookup: (camera_id, track_id) → (start_time, end_time)
        self._tracklet_times: Dict[Tuple[str, int], Tuple[float, float]] = {}
        # Pre-built trajectory lookup: global_id → GlobalTrajectory
        self._traj_by_id: Dict[int, GlobalTrajectory] = {}
        for traj in trajectories:
            self._traj_by_id[traj.global_id] = traj
            for t in traj.tracklets:
                key = (t.camera_id, t.track_id)
                self._tracklet_to_traj[key] = (traj.global_id, traj.confidence)
                self._tracklet_times[key] = (t.start_time, t.end_time)

        logger.debug(
            f"ForensicSearchEngine ready: {len(self.embeddings)} tracklets, "
            f"{len(trajectories)} trajectories"
        )

    # ------------------------------------------------------------------
    # Query by embedding vector
    # ------------------------------------------------------------------

    def query_by_embedding(
        self,
        query_vec: np.ndarray,
        top_k: int = 20,
        min_similarity: float = 0.30,
    ) -> List[SearchResult]:
        """Find the top-K tracklets most similar to a query embedding.

        Args:
            query_vec: (D,) L2-normalised embedding of the probe image.
            top_k: Maximum number of results to return.
            min_similarity: Discard results below this cosine similarity.

        Returns:
            Sorted list of SearchResult (highest similarity first).
        """
        query_vec = query_vec.astype(np.float32)
        norm = np.linalg.norm(query_vec)
        if norm > 1e-8:
            query_vec = query_vec / norm

        # Brute-force cosine similarity (embeddings are L2-normed)
        sims = self.embeddings @ query_vec  # (N,)
        order = np.argsort(-sims)

        results = []
        for rank, idx in enumerate(order[:top_k * 3]):  # over-fetch then filter
            sim = float(sims[idx])
            if sim < min_similarity:
                break
            if len(results) >= top_k:
                break

            meta = self.index_map[idx]
            cam, tid = meta["camera_id"], meta["track_id"]
            traj_info = self._tracklet_to_traj.get((cam, tid), (None, 0.0))

            # Resolve tracklet times from trajectories
            start, end = self._get_tracklet_times(cam, tid)
            results.append(SearchResult(
                rank=rank + 1,
                tracklet_id=tid,
                camera_id=cam,
                start_time=start,
                end_time=end,
                similarity=sim,
                global_id=traj_info[0],
                trajectory_confidence=traj_info[1],
            ))

        logger.info(
            f"Re-ID query: {len(results)} results above sim={min_similarity:.2f}"
        )
        return results

    # ------------------------------------------------------------------
    # Query by raw image crop
    # ------------------------------------------------------------------

    def query_by_image(
        self,
        image_bgr: np.ndarray,
        reid_model,
        top_k: int = 20,
        min_similarity: float = 0.30,
    ) -> List[SearchResult]:
        """Extract embedding from a BGR crop and search the gallery.

        Args:
            image_bgr: H×W×3 uint8 BGR image of the probe vehicle/person.
            reid_model: ReIDModel instance (from stage2_features.reid_model).
            top_k: Maximum results.
            min_similarity: Minimum cosine similarity threshold.
        """
        import torch
        from torchvision import transforms
        # Delegate to the ReID model's preprocessing pipeline
        embedding = reid_model.extract_embedding(image_bgr)
        if embedding is None:
            logger.warning("Failed to extract embedding from probe image")
            return []
        return self.query_by_embedding(embedding, top_k=top_k, min_similarity=min_similarity)

    # ------------------------------------------------------------------
    # Watchlist scan
    # ------------------------------------------------------------------

    def watchlist_scan(
        self,
        watchlist: Dict[str, np.ndarray],
        threshold: float = 0.55,
    ) -> List[WatchlistHit]:
        """Scan all trajectories for subjects matching a watchlist.

        Each trajectory is represented by the mean embedding of its constituent
        tracklets. The maximum similarity across all per-tracklet embeddings to
        any watchlist entry triggers a hit.

        Args:
            watchlist: Mapping of subject_id → L2-normalised query embedding (D,).
            threshold: Minimum cosine similarity to report a hit.

        Returns:
            List of WatchlistHit sorted by alert level then similarity (descending).
        """
        # Build per-trajectory mean embeddings indexed by global_id
        # For each trajectory, collect its feature indices
        traj_feat_indices: Dict[int, List[int]] = {}
        for fi, meta in enumerate(self.index_map):
            cam, tid = meta["camera_id"], meta["track_id"]
            traj_info = self._tracklet_to_traj.get((cam, tid))
            if traj_info is not None:
                gid = traj_info[0]
                traj_feat_indices.setdefault(gid, []).append(fi)

        hits: List[WatchlistHit] = []

        for subject_id, query_vec in watchlist.items():
            query_vec = query_vec.astype(np.float32)
            q_norm = np.linalg.norm(query_vec)
            if q_norm > 1e-8:
                query_vec = query_vec / q_norm

            for gid, feat_idx_list in traj_feat_indices.items():
                traj = self._traj_by_id.get(gid)
                if traj is None:
                    continue
                # Best-match similarity across all tracklets in this trajectory
                traj_embs = self.embeddings[feat_idx_list]  # (K, D)
                sims = traj_embs @ query_vec  # (K,)
                best_sim = float(np.max(sims))

                if best_sim < threshold:
                    continue

                span = traj.time_span
                hits.append(WatchlistHit(
                    subject_id=subject_id,
                    global_id=gid,
                    similarity=best_sim,
                    trajectory_confidence=traj.confidence,
                    cameras_seen=list(dict.fromkeys(traj.camera_sequence)),
                    first_seen=span[0],
                    last_seen=span[1],
                ))

        # Sort: HIGH alerts first, then by similarity descending
        _level_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        hits.sort(key=lambda h: (_level_order[h.alert_level], -h.similarity))

        logger.info(
            f"Watchlist scan: {len(watchlist)} subjects, "
            f"{len(hits)} hits above threshold={threshold:.2f}"
        )
        return hits

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_forensic_report(
        self,
        output_dir: Path,
        min_confidence: float = 0.0,
        min_cameras: int = 1,
    ) -> Path:
        """Export a structured forensic report of all tracked identities.

        Produces ``forensic_report.json`` containing every trajectory with its
        full audit trail (confidence, evidence pairs, camera timeline).

        Args:
            output_dir: Directory to write the report.
            min_confidence: Only include trajectories at or above this confidence.
            min_cameras: Only include trajectories seen in at least N cameras.

        Returns:
            Path to the written report file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        eligible = [
            t for t in self.trajectories
            if t.confidence >= min_confidence and t.num_cameras >= min_cameras
        ]
        eligible.sort(key=lambda t: (-t.confidence, t.global_id))

        report = {
            "summary": {
                "total_trajectories": len(self.trajectories),
                "reported_trajectories": len(eligible),
                "cross_camera_trajectories": sum(
                    1 for t in self.trajectories if t.is_cross_camera
                ),
                "high_confidence_trajectories": sum(
                    1 for t in self.trajectories if t.confidence >= 0.70
                ),
                "filters": {
                    "min_confidence": min_confidence,
                    "min_cameras": min_cameras,
                },
            },
            "trajectories": [t.to_forensic_dict() for t in eligible],
        }

        report_path = output_dir / "forensic_report.json"
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8"
        )
        logger.info(
            f"Forensic report written: {report_path} "
            f"({len(eligible)} trajectories)"
        )
        return report_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_tracklet_times(
        self, camera_id: str, track_id: int
    ) -> Tuple[float, float]:
        return self._tracklet_times.get((camera_id, track_id), (0.0, 0.0))
