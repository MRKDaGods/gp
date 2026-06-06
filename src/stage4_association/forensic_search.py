"""Forensic Re-Identification Search and Watchlist Matching."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from src.core.data_models import GlobalTrajectory


# Data structures

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
    # Geospatial fields, set by query_by_embedding_geo.
    distance_from_query_m: Optional[float] = None  # great-circle camera distance
    geo_score: Optional[float] = None              # plausibility in [0, 1]
    required_speed_ms: Optional[float] = None      # distance / elapsed time
    combined_score: Optional[float] = None         # similarity * geo_score

    def to_dict(self) -> Dict[str, Any]:
        out = {
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
        if self.geo_score is not None:
            out["geo_score"] = round(self.geo_score, 4)
            out["combined_score"] = round(self.combined_score or 0.0, 4)
            if self.distance_from_query_m is not None:
                out["distance_from_query_m"] = round(self.distance_from_query_m, 1)
            if self.required_speed_ms is not None:
                out["required_speed_ms"] = round(self.required_speed_ms, 2)
        return out


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
        """Operational alert level based on similarity x trajectory confidence."""
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


# Core search functions

class ForensicSearchEngine:
    """Re-ID search and watchlist engine backed by the Stage 3 FAISS index."""

    def __init__(
        self,
        embeddings: np.ndarray,
        index_map: List[Dict[str, Any]],
        trajectories: List[GlobalTrajectory],
        faiss_index=None,
        geo_constraint=None,
    ):
        """Args:"""
        self.embeddings = embeddings.astype(np.float32)
        self.index_map = index_map
        self.trajectories = trajectories
        self.faiss_index = faiss_index
        self.geo_constraint = geo_constraint

        # Class id per embedding row, for geo speed-profile selection.
        self._class_by_index: List[int] = [int(m.get("class_id", -1)) for m in index_map]

        # Build reverse lookup: (camera_id, track_id) -> global_id + confidence
        self._tracklet_to_traj: Dict[Tuple[str, int], Tuple[int, float]] = {}
        # Pre-built time lookup: (camera_id, track_id) -> (start_time, end_time)
        self._tracklet_times: Dict[Tuple[str, int], Tuple[float, float]] = {}
        # Pre-built trajectory lookup: global_id -> GlobalTrajectory
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

    # Query by embedding vector

    def query_by_embedding(
        self,
        query_vec: np.ndarray,
        top_k: int = 20,
        min_similarity: float = 0.30,
    ) -> List[SearchResult]:
        """Find the top-K tracklets most similar to a query embedding."""
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

    # Query by raw image crop

    def query_by_image(
        self,
        image_bgr: np.ndarray,
        reid_model,
        top_k: int = 20,
        min_similarity: float = 0.30,
    ) -> List[SearchResult]:
        """Extract embedding from a BGR crop and search the gallery."""
        # Delegate to the ReID model's preprocessing pipeline
        embedding = reid_model.extract_embedding(image_bgr)
        if embedding is None:
            logger.warning("Failed to extract embedding from probe image")
            return []
        return self.query_by_embedding(embedding, top_k=top_k, min_similarity=min_similarity)

    # Geospatially-constrained expanding-radius search

    def query_by_embedding_geo(
        self,
        query_vec: np.ndarray,
        query_camera: str,
        query_time: float,
        query_class_id: int,
        top_k: int = 20,
        min_similarity: float = 0.30,
        ring_width_m: Optional[float] = None,
        confident_similarity: float = 0.55,
        min_confident_hits: int = 1,
    ) -> List[SearchResult]:
        """Expanding-radius Re-ID search constrained by camera geometry."""
        geo = self.geo_constraint
        if geo is None or not geo.is_active or not geo.has_coords(query_camera):
            logger.info("Geo search unavailable; falling back to plain embedding search")
            return self.query_by_embedding(
                query_vec, top_k=top_k, min_similarity=min_similarity
            )

        # Score the whole gallery once (embeddings are L2-normalised).
        query_vec = query_vec.astype(np.float32)
        norm = np.linalg.norm(query_vec)
        if norm > 1e-8:
            query_vec = query_vec / norm
        sims = self.embeddings @ query_vec

        # Candidates above the floor, grouped by camera (excluding the probe camera).
        from collections import defaultdict
        cam_candidates: Dict[str, List[int]] = defaultdict(list)
        for idx in range(len(sims)):
            if float(sims[idx]) < min_similarity:
                continue
            cam = self.index_map[idx]["camera_id"]
            if cam == query_camera:
                continue
            cam_candidates[cam].append(idx)

        # Expanding rings (nearest first), then a trailing group of cameras
        # without coordinates so the full set is still covered.
        if ring_width_m is None:
            ring_width_m = max(geo.overlap_fov_radius_m, 1.0)
        groups: List[List[str]] = [list(r) for r in geo.expanding_rings(query_camera, ring_width_m)]
        coordless = sorted(c for c in cam_candidates if not geo.has_coords(c))
        if coordless:
            groups.append(coordless)

        results: List[SearchResult] = []
        confident = 0
        rings_scanned = 0
        for group in groups:
            scanned_any = False
            for cam in group:
                tids = cam_candidates.get(cam, [])
                if tids:
                    scanned_any = True
                for idx in tids:
                    tid = self.index_map[idx]["track_id"]
                    start, end = self._get_tracklet_times(cam, tid)
                    gap = start - query_time
                    cls = self._class_by_index[idx]
                    if cls < 0:
                        cls = query_class_id
                    if not geo.is_reachable(query_camera, cam, gap, cls):
                        continue
                    g_score = geo.geo_score(query_camera, cam, gap, cls)
                    sim = float(sims[idx])
                    dist = geo.distance_m(query_camera, cam)
                    req_speed = None
                    if dist is not None and abs(gap) > geo.overlap_time_window_s:
                        req_speed = dist / max(abs(gap), geo.min_time_s)
                    traj_info = self._tracklet_to_traj.get((cam, tid), (None, 0.0))
                    results.append(SearchResult(
                        rank=0,
                        tracklet_id=tid,
                        camera_id=cam,
                        start_time=start,
                        end_time=end,
                        similarity=sim,
                        global_id=traj_info[0],
                        trajectory_confidence=traj_info[1],
                        distance_from_query_m=dist,
                        geo_score=g_score,
                        required_speed_ms=req_speed,
                        combined_score=sim * g_score,
                    ))
                    if sim >= confident_similarity and g_score > 0:
                        confident += 1
            if scanned_any:
                rings_scanned += 1
            if confident >= min_confident_hits and len(results) >= top_k:
                break

        results.sort(key=lambda r: (r.combined_score or 0.0, r.similarity), reverse=True)
        results = results[:top_k]
        for rank, r in enumerate(results, start=1):
            r.rank = rank

        logger.info(
            f"Geo search ({query_camera}@{query_time:.1f}s): {len(results)} hits, "
            f"{confident} confident, {rings_scanned} ring(s) scanned"
        )
        return results

    # Watchlist scan

    def watchlist_scan(
        self,
        watchlist: Dict[str, np.ndarray],
        threshold: float = 0.55,
    ) -> List[WatchlistHit]:
        """Scan all trajectories for subjects matching a watchlist."""
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

    # Export

    def export_forensic_report(
        self,
        output_dir: Path,
        min_confidence: float = 0.0,
        min_cameras: int = 1,
    ) -> Path:
        """Export a structured forensic report of all tracked identities."""
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

        # Attach a Maps/QR path link per trajectory from the GPS coordinates of
        # the cameras it visited, in chronological order.
        if self.geo_constraint is not None and self.geo_constraint.is_active:
            self._attach_geospatial_paths(eligible, report["trajectories"])

        report_path = output_dir / "forensic_report.json"
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8"
        )
        logger.info(
            f"Forensic report written: {report_path} "
            f"({len(eligible)} trajectories)"
        )
        return report_path

    # Internal helpers

    def _attach_geospatial_paths(
        self,
        trajectories: List[GlobalTrajectory],
        traj_dicts: List[Dict[str, Any]],
    ) -> None:
        """Add a GPS path and Maps link to each trajectory dict (parallel lists)."""
        from src.stage4_association.geospatial import build_maps_path_share_url

        geo = self.geo_constraint
        for traj, tdict in zip(trajectories, traj_dicts):
            path: List[Dict[str, Any]] = []
            latlng_path: List[Tuple[float, float]] = []
            for cam in traj.camera_sequence:
                coord = geo.coordinates.get(cam)
                if coord is None:
                    continue
                lat, lng = coord
                if latlng_path and latlng_path[-1] == (lat, lng):
                    continue
                path.append({"camera_id": cam, "lat": lat, "lng": lng})
                latlng_path.append((lat, lng))
            if not path:
                continue
            tdict["geospatial_path"] = path
            tdict["maps_path_url"] = build_maps_path_share_url(latlng_path)

    def _get_tracklet_times(
        self, camera_id: str, track_id: int
    ) -> Tuple[float, float]:
        return self._tracklet_times.get((camera_id, track_id), (0.0, 0.0))
