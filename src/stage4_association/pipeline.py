"""Stage 4 — Multi-Camera Association pipeline.

Performs cross-camera tracklet association using appearance, color, and
spatio-temporal cues. Produces global trajectories with unified IDs.

Improvements over baseline:
* Mutual nearest-neighbour filtering before re-ranking.
* FAISS index passed to sparse re-ranking (avoids full N²).
* Class-adaptive similarity weights (person vs vehicle).
* Iterative gallery expansion for orphan tracklets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Set, Tuple

import networkx as nx
import numpy as np
from loguru import logger
from omegaconf import DictConfig

from src.core.data_models import GlobalTrajectory, Tracklet, TrackletFeatures
from src.core.io_utils import save_global_trajectories
from src.stage3_indexing.faiss_index import FAISSIndex
from src.stage3_indexing.metadata_store import MetadataStore
from src.stage4_association.camera_bias import CameraDistanceBias, ZoneTransitionModel
from src.stage4_association.fic import per_camera_whiten, cross_camera_augment
from src.stage4_association.global_trajectories import merge_tracklets_to_trajectories
from src.stage4_association.graph_solver import GraphSolver
from src.stage4_association.query_expansion import average_query_expansion_batched
from src.stage4_association.reranking import k_reciprocal_rerank
from src.stage4_association.similarity import (
    compute_combined_similarity,
    mutual_nearest_neighbor_filter,
)
from src.stage4_association.spatial_temporal import SpatioTemporalValidator


def run_stage4(
    cfg: DictConfig,
    faiss_index: FAISSIndex,
    metadata_store: MetadataStore,
    features: List[TrackletFeatures],
    tracklets_by_camera: Dict[str, List[Tracklet]],
    output_dir: str | Path,
) -> List[GlobalTrajectory]:
    """Run cross-camera association.

    Args:
        cfg: Full pipeline config (uses cfg.stage4).
        faiss_index: Built FAISS index from Stage 3.
        metadata_store: Populated metadata store from Stage 3.
        features: TrackletFeatures from Stage 2.
        tracklets_by_camera: Tracklets from Stage 1.
        output_dir: Directory for stage4 outputs.

    Returns:
        List of GlobalTrajectory objects.
    """
    stage_cfg = cfg.stage4.association
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n = len(features)
    if n == 0:
        logger.warning("No features to associate")
        return []

    logger.info(f"Starting cross-camera association: {n} tracklets")

    # Build embedding matrix and metadata arrays
    embeddings = np.stack([f.embedding for f in features], axis=0)
    hsv_features = np.stack([f.hsv_histogram for f in features], axis=0)

    camera_ids = [f.camera_id for f in features]
    class_ids = [f.class_id for f in features]

    # Step 0: Per-camera feature whitening (FIC) — AIC21 1st-place technique.
    # Removes camera-specific bias (lighting, viewpoint) from embeddings.
    fic_cfg = stage_cfg.get("fic", {})
    if fic_cfg.get("enabled", False):
        embeddings = per_camera_whiten(
            embeddings,
            camera_ids,
            regularisation=float(fic_cfg.get("regularisation", 3.0)),
            min_samples=int(fic_cfg.get("min_samples", 5)),
        )

    # Step 0b: Cross-camera feature augmentation (FAC) — AIC21 technique.
    # Pulls each feature toward its cross-camera KNN consensus.
    fac_cfg = stage_cfg.get("fac", {})
    if fac_cfg.get("enabled", False):
        embeddings = cross_camera_augment(
            embeddings,
            camera_ids,
            knn=int(fac_cfg.get("knn", 20)),
            learning_rate=float(fac_cfg.get("learning_rate", 0.5)),
            beta=float(fac_cfg.get("beta", 0.08)),
        )

    # Get temporal info and frame counts from metadata store
    start_times = []
    end_times = []
    num_frames = []
    missing_meta_count = 0
    for i in range(n):
        meta = metadata_store.get_tracklet(i)
        if meta:
            st = meta["start_time"]
            et = meta["end_time"]
            if st > et:
                logger.warning(
                    f"Tracklet {i}: start_time ({st:.3f}) > end_time ({et:.3f}) — swapping"
                )
                st, et = et, st
            start_times.append(st)
            end_times.append(et)
            num_frames.append(meta.get("num_frames", 1))
        else:
            missing_meta_count += 1
            start_times.append(0.0)
            end_times.append(0.0)
            num_frames.append(1)

    if missing_meta_count > 0:
        logger.warning(
            f"Missing metadata for {missing_meta_count}/{n} tracklets — "
            f"temporal analysis may be degraded"
        )

    # Step 1: FAISS top-K retrieval
    # If FIC/FAC modified embeddings, rebuild FAISS from the updated features
    # so KNN retrieval (used by QE and reranking) is consistent.
    if fic_cfg.get("enabled", False) or fac_cfg.get("enabled", False):
        faiss_index = FAISSIndex(index_type="flat_ip")
        faiss_index.build(embeddings.astype(np.float32))
        logger.info("Rebuilt FAISS index from FIC/FAC-transformed embeddings")
    top_k = stage_cfg.top_k
    distances, indices = faiss_index.search(embeddings, top_k)
    logger.info(f"FAISS retrieval: top-{top_k} candidates per tracklet")

    # Step 1b: Average Query Expansion (AQE) + Database-side Augmentation (DBA)
    #
    # AQE: expand each query embedding by averaging with its k-NN, then L2-renorm.
    # DBA: after AQE, rebuild the FAISS index with the expanded embeddings so that
    #      retrieval is symmetric — expanded queries search an expanded gallery.
    #      Without DBA the second search still uses the original gallery vectors,
    #      which means QE only partially improves retrieval.
    #
    # Alpha interpretation: original weight = alpha / (alpha + k_valid).
    #   alpha=1.0, k=5 → 16.7 % original (aggressive, pushes embeddings toward centroid)
    #   alpha=5.0, k=5 → 50.0 % original (balanced — matches Radenović et al. α=0.5)
    #   alpha=10.0, k=5 → 66.7 % original (conservative)
    qe_cfg = stage_cfg.get("query_expansion", {})
    if qe_cfg.get("enabled", True):
        qe_k = qe_cfg.get("k", 5)
        qe_alpha = qe_cfg.get("alpha", 5.0)
        embeddings = average_query_expansion_batched(
            embeddings, indices, k=qe_k, alpha=qe_alpha,
        )
        # DBA: rebuild FAISS with expanded embeddings for symmetric retrieval.
        # Enabled by default; disable only for ablation studies.
        if qe_cfg.get("dba", True):
            dba_index = FAISSIndex(index_type="flat_ip")
            dba_index.build(embeddings.astype(np.float32))
            distances, indices = dba_index.search(embeddings, top_k)
            # Update faiss_index reference so reranking uses the DBA index
            # (AQE-expanded embeddings, consistent with the modified embeddings)
            faiss_index = dba_index
            logger.info(
                f"QE+DBA: rebuilt FAISS with expanded embeddings "
                f"(alpha={qe_alpha}, k={qe_k})"
            )
        else:
            distances, indices = faiss_index.search(embeddings, top_k)
            logger.info(
                f"QE (no DBA): re-retrieved with expanded queries "
                f"(alpha={qe_alpha}, k={qe_k})"
            )

    # Step 2: Build candidate pairs (cross-camera, same class)
    #
    # Two modes:
    #  - exhaustive (default): compute ALL cross-camera pairwise cosine similarities.
    #    SOTA practice for MTMC with <2000 tracklets — avoids top-K truncation errors
    #    where true matches fall below the FAISS cutoff due to visually-similar decoys.
    #  - topk: keep only FAISS top-K candidates (legacy; faster but misses long-tail matches).
    exhaustive_cfg = stage_cfg.get("exhaustive_cross_camera", True)
    if exhaustive_cfg:
        min_sim = float(stage_cfg.get("exhaustive_min_similarity", 0.0))
        candidate_pairs = _build_all_cross_camera_pairs(
            n, embeddings, camera_ids, class_ids, min_similarity=min_sim,
        )
    else:
        candidate_pairs = _build_candidate_pairs(
            n, indices, distances, top_k, camera_ids, class_ids,
        )
    logger.info(f"Candidate pairs after filtering: {len(candidate_pairs)}")

    if not candidate_pairs:
        logger.warning("No cross-camera candidate pairs found")
        all_tracklets = [t for tl in tracklets_by_camera.values() for t in tl]
        return _assign_individual_ids(all_tracklets)

    # Step 2a: Hard temporal constraint — pre-filter pairs where the two tracklets
    # overlap in time within the SAME camera.  These are provably different identities
    # and should never enter the graph, regardless of embedding similarity. Removing them
    # here prevents Louvain from creating false transitive links (A~B~C where A and C
    # share camera+time but B bridges them from another camera).
    pre_hard = len(candidate_pairs)
    candidate_pairs = [
        (i, j, sim) for i, j, sim in candidate_pairs
        if not (
            camera_ids[i] == camera_ids[j]
            and start_times[i] <= end_times[j]
            and start_times[j] <= end_times[i]
        )
    ]
    hard_removed = pre_hard - len(candidate_pairs)
    if hard_removed > 0:
        logger.info(
            f"Hard temporal constraint: removed {hard_removed} same-camera "
            f"overlapping pairs (impossible same-identity links)"
        )

    # Step 2b: Mutual nearest-neighbour filter
    mutual_nn_cfg = stage_cfg.get("mutual_nn", {})
    if mutual_nn_cfg.get("enabled", True):
        pre_filter = len(candidate_pairs)
        candidate_pairs = mutual_nearest_neighbor_filter(
            candidate_pairs,
            top_k_per_query=mutual_nn_cfg.get("top_k_per_query", 10),
        )
        logger.info(
            f"Mutual NN filter: {pre_filter} → {len(candidate_pairs)} pairs "
            f"({pre_filter - len(candidate_pairs)} pruned)"
        )

    # Step 3: k-reciprocal re-ranking (optional)
    if stage_cfg.reranking.enabled:
        logger.info("Applying sparse k-reciprocal re-ranking...")
        appearance_sim = k_reciprocal_rerank(
            embeddings=embeddings,
            candidate_pairs=candidate_pairs,
            k1=stage_cfg.reranking.k1,
            k2=stage_cfg.reranking.k2,
            lambda_value=stage_cfg.reranking.lambda_value,
            faiss_index=faiss_index.index if hasattr(faiss_index, "index") else None,
        )
    else:
        appearance_sim = {(i, j): sim for i, j, sim in candidate_pairs}

    # Step 4: Spatio-temporal validation
    st_validator = SpatioTemporalValidator(
        min_time_gap=stage_cfg.spatiotemporal.min_time_gap,
        max_time_gap=stage_cfg.spatiotemporal.max_time_gap,
        camera_transitions=stage_cfg.spatiotemporal.get("camera_transitions"),
    )

    # Step 5: Compute combined similarity with class-adaptive weights
    combined_sim = compute_combined_similarity(
        appearance_sim=appearance_sim,
        hsv_features=hsv_features,
        start_times=start_times,
        end_times=end_times,
        camera_ids=camera_ids,
        st_validator=st_validator,
        weights=stage_cfg.weights,
        class_ids=class_ids,
        num_frames=num_frames,
    )

    logger.info(f"Combined similarity pairs: {len(combined_sim)}")

    # Step 5b: Camera distance bias adjustment (iterative)
    camera_bias_cfg = stage_cfg.get("camera_bias", {})
    if camera_bias_cfg.get("enabled", False):
        cam_bias = CameraDistanceBias()

        # Load pre-learned bias if available
        bias_path = camera_bias_cfg.get("bias_path")
        if bias_path and Path(bias_path).exists():
            cam_bias.load(bias_path)
            logger.info(f"Loaded camera bias from {bias_path}")
            combined_sim = cam_bias.adjust_similarity_matrix(combined_sim, camera_ids)
            logger.info("Applied pre-learned camera distance bias")
        else:
            # Learn bias iteratively: cluster → learn bias → re-adjust → re-cluster
            n_iterations = camera_bias_cfg.get("iterations", 1)
            for iter_idx in range(n_iterations):
                # Initial clustering to discover identity groups
                tmp_solver = GraphSolver(
                    similarity_threshold=stage_cfg.graph.similarity_threshold,
                    algorithm=stage_cfg.graph.algorithm,
                    louvain_resolution=stage_cfg.graph.get("louvain_resolution", 1.0),
                    louvain_seed=int(stage_cfg.graph.get("louvain_seed", 42)),
                    bridge_prune_margin=float(stage_cfg.graph.get("bridge_prune_margin", 0.0)),
                    max_component_size=int(stage_cfg.graph.get("max_component_size", 0)),
                )
                tmp_clusters = tmp_solver.solve(combined_sim, n)
                # Only learn from multi-member clusters
                multi_clusters = [c for c in tmp_clusters if len(c) >= 2]
                if not multi_clusters:
                    break
                cam_bias.learn_from_matches(combined_sim, camera_ids, multi_clusters)
                combined_sim = cam_bias.adjust_similarity_matrix(combined_sim, camera_ids)
                logger.info(
                    f"Camera bias iteration {iter_idx + 1}/{n_iterations}: "
                    f"learned from {len(multi_clusters)} clusters"
                )

            # Save learned bias for future runs
            save_path = camera_bias_cfg.get("save_path")
            if save_path:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                cam_bias.save(save_path)

    # Step 5c: Zone-based transition scoring (for CityFlow-style datasets)
    zone_cfg = stage_cfg.get("zone_model", {})
    if zone_cfg.get("enabled", False):
        zone_model = ZoneTransitionModel()
        zone_model.load_from_config(zone_cfg)
        # TODO: integrate zone scores into combined_sim
        # This requires tracklet entry/exit positions which are dataset-specific
        logger.info("Zone transition model loaded (integration pending)")

    # Step 6: Build graph and solve
    if combined_sim:
        sim_vals = list(combined_sim.values())
        threshold = float(stage_cfg.graph.similarity_threshold)
        n_above = sum(1 for s in sim_vals if s >= threshold)
        logger.info(
            f"Combined sim stats: {len(sim_vals)} pairs, "
            f"min={min(sim_vals):.3f} median={np.median(sim_vals):.3f} "
            f"max={max(sim_vals):.3f}, {n_above} above threshold {threshold}"
        )
    solver = GraphSolver(
        similarity_threshold=stage_cfg.graph.similarity_threshold,
        algorithm=stage_cfg.graph.algorithm,
        louvain_resolution=stage_cfg.graph.get("louvain_resolution", 1.0),
        louvain_seed=int(stage_cfg.graph.get("louvain_seed", 42)),
        bridge_prune_margin=float(stage_cfg.graph.get("bridge_prune_margin", 0.0)),
        max_component_size=int(stage_cfg.graph.get("max_component_size", 0)),
    )

    clusters = solver.solve(combined_sim, n)
    logger.info(f"Graph solver found {len(clusters)} identity clusters")

    # Step 6b: Resolve same-camera conflicts
    clusters = _resolve_same_camera_conflicts(
        clusters, camera_ids, start_times, end_times, combined_sim,
    )

    # Step 6c: Gallery expansion — attempt to recover orphan tracklets
    gallery_cfg = stage_cfg.get("gallery_expansion", {})
    if gallery_cfg.get("enabled", False):
        clusters = _gallery_expansion(
            clusters=clusters,
            embeddings=embeddings,
            hsv_features=hsv_features,
            camera_ids=camera_ids,
            class_ids=class_ids,
            start_times=start_times,
            end_times=end_times,
            st_validator=st_validator,
            n=n,
            stage_cfg=stage_cfg,
            threshold=gallery_cfg.get("threshold", 0.5),
            max_rounds=gallery_cfg.get("max_rounds", 2),
            orphan_match_threshold=float(gallery_cfg.get("orphan_match_threshold", 0.0)),
            combined_sim=combined_sim,
        )

    # Step 6d: Re-resolve same-camera conflicts introduced by orphan-orphan matching.
    # Orphan pairs (A→X, B→X) can individually pass the cross-camera check yet
    # transitively place same-camera tracklets A and B in the same cluster.
    clusters = _resolve_same_camera_conflicts(
        clusters, camera_ids, start_times, end_times, combined_sim,
    )

    # Step 7: Build global trajectories
    tracklet_lookup: Dict[Tuple[str, int], Tracklet] = {}
    for cam_id, tracklets in tracklets_by_camera.items():
        for t in tracklets:
            tracklet_lookup[(cam_id, t.track_id)] = t

    feature_to_tracklet_key = [
        (f.camera_id, f.track_id) for f in features
    ]

    trajectories = merge_tracklets_to_trajectories(
        clusters=clusters,
        feature_to_tracklet_key=feature_to_tracklet_key,
        tracklet_lookup=tracklet_lookup,
        embeddings=embeddings,
        combined_sim=combined_sim,
    )

    # Log confidence distribution for diagnostics
    confidences = [t.confidence for t in trajectories if t.num_cameras > 1]
    if confidences:
        import statistics
        logger.info(
            f"Cross-camera trajectory confidence: "
            f"mean={statistics.mean(confidences):.3f}, "
            f"min={min(confidences):.3f}, "
            f"high(≥0.7)={sum(1 for c in confidences if c >= 0.7)}/{len(confidences)}"
        )
    else:
        logger.info("No cross-camera trajectories produced (all single-camera)")

    # Save
    save_global_trajectories(trajectories, output_dir / "global_trajectories.json")

    # Auto-generate forensic report alongside trajectory JSON
    try:
        from src.stage4_association.forensic_search import ForensicSearchEngine
        from src.core.io_utils import load_embeddings, load_hsv_features
        engine = ForensicSearchEngine(
            embeddings=embeddings,
            index_map=[{"camera_id": f.camera_id, "track_id": f.track_id, "class_id": f.class_id}
                       for f in features],
            trajectories=trajectories,
        )
        engine.export_forensic_report(output_dir, min_confidence=0.0, min_cameras=1)
    except Exception as exc:
        logger.warning(f"Forensic report generation failed (non-fatal): {exc}")

    logger.info(
        f"Stage 4 complete: {len(trajectories)} global trajectories, "
        f"covering {sum(len(t.tracklets) for t in trajectories)} tracklets"
    )

    return trajectories


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_candidate_pairs(
    n: int,
    indices: np.ndarray,
    distances: np.ndarray,
    top_k: int,
    camera_ids: List[str],
    class_ids: List[int],
) -> List[Tuple[int, int, float]]:
    """Build candidate pairs from FAISS results with cross-camera + same-class filter."""
    # Use actual result width from FAISS (may be < top_k when corpus is small)
    actual_k = indices.shape[1]
    candidate_pairs: List[Tuple[int, int, float]] = []
    for i in range(n):
        for j_idx in range(actual_k):
            j = int(indices[i, j_idx])
            if j < 0 or j >= n or i == j:
                continue
            if camera_ids[i] == camera_ids[j]:
                continue
            if class_ids[i] != class_ids[j]:
                continue
            candidate_pairs.append((i, j, float(distances[i, j_idx])))
    return candidate_pairs


def _extract_scene(camera_id: str) -> str:
    """Extract scene prefix from camera ID.

    E.g. 'S01_c001' -> 'S01', 'cam1' -> '' (no scene prefix).
    Used for scene blocking: cameras from different scenes should never
    be linked (they are physically separate locations).
    """
    parts = camera_id.split("_")
    if len(parts) >= 2 and parts[0][:1].upper() == "S" and parts[0][1:].isdigit():
        return parts[0]
    return ""  # No scene prefix — treat all cameras as same scene


def _build_all_cross_camera_pairs(
    n: int,
    embeddings: np.ndarray,
    camera_ids: List[str],
    class_ids: List[int],
    min_similarity: float = 0.0,
) -> List[Tuple[int, int, float]]:
    """Exhaustive cross-camera candidate pair generation via brute-force cosine similarity.

    This is SOTA practice for MTMC with ≤2000 tracklets: compute ALL cross-camera
    pairwise cosine similarities rather than relying on FAISS top-K retrieval.
    FAISS top-K can miss true matches when many visually-similar vehicles push
    genuine cross-camera pairs below the top-K cutoff.

    **Scene blocking**: Camera pairs from different scenes (e.g. S01 vs S02) are
    skipped entirely.  This prevents cross-scene vehicles from contaminating
    the mutual-NN neighbourhoods, which would otherwise displace valid same-scene
    matches from the top-K slots.

    Handles the matrix computation camera-pair-by-camera-pair to keep peak memory
    usage proportional to the largest camera's tracklet count (not N²).

    Args:
        n: Total number of tracklets.
        embeddings: (N, D) L2-normalised embedding matrix.
        camera_ids: Camera ID for each tracklet.
        class_ids: Class ID for each tracklet.
        min_similarity: Pairs below this threshold are discarded (reduces downstream
            memory; 0.0 keeps everything and lets the mutual-NN filter decide).

    Returns:
        List of (i, j, similarity) tuples, one per cross-camera same-class pair
        with similarity ≥ min_similarity.
    """
    # Group tracklet indices by camera
    from collections import defaultdict
    cam_to_idxs: Dict[str, List[int]] = defaultdict(list)
    for idx, cam in enumerate(camera_ids):
        cam_to_idxs[cam].append(idx)

    cameras = sorted(cam_to_idxs.keys())

    # Pre-compute scene for each camera for scene blocking
    cam_scenes = {cam: _extract_scene(cam) for cam in cameras}
    scene_set = set(cam_scenes.values()) - {""}
    scene_blocking_active = len(scene_set) > 1
    if scene_blocking_active:
        logger.info(
            f"Scene blocking: {len(scene_set)} scenes detected "
            f"({', '.join(sorted(scene_set))}). Cross-scene pairs will be skipped."
        )

    candidate_pairs: List[Tuple[int, int, float]] = []
    skipped_cross_scene = 0

    # Iterate over all ordered camera pairs (a_cam < b_cam to avoid duplicates)
    for a_idx, cam_a in enumerate(cameras):
        a_global = cam_to_idxs[cam_a]
        a_embs = embeddings[a_global]  # (N_a, D)
        scene_a = cam_scenes[cam_a]

        for cam_b in cameras[a_idx + 1:]:
            scene_b = cam_scenes[cam_b]

            # Scene blocking: skip camera pairs from different scenes
            if scene_blocking_active and scene_a and scene_b and scene_a != scene_b:
                skipped_cross_scene += len(a_global) * len(cam_to_idxs[cam_b])
                continue

            b_global = cam_to_idxs[cam_b]
            b_embs = embeddings[b_global]  # (N_b, D)

            # (N_a, N_b) cosine similarity (embeddings are L2-normalised)
            sim_mat = a_embs @ b_embs.T

            # Find all pairs above threshold
            a_local, b_local = np.where(sim_mat >= min_similarity)
            for al, bl in zip(a_local.tolist(), b_local.tolist()):
                gi = a_global[al]
                gj = b_global[bl]
                if class_ids[gi] != class_ids[gj]:
                    continue
                candidate_pairs.append((gi, gj, float(sim_mat[al, bl])))

    logger.info(
        f"Exhaustive cross-camera pairs: {len(candidate_pairs)} "
        f"(min_sim={min_similarity:.2f}, {len(cameras)} cameras"
        + (f", {skipped_cross_scene:,} cross-scene pairs blocked" if skipped_cross_scene else "")
        + ")"
    )
    return candidate_pairs


def _assign_individual_ids(tracklets: List[Tracklet]) -> List[GlobalTrajectory]:
    """Fallback: assign each tracklet its own global ID."""
    return [
        GlobalTrajectory(global_id=i, tracklets=[t])
        for i, t in enumerate(tracklets)
    ]


def _gallery_expansion(
    clusters: List[Set[int]],
    embeddings: np.ndarray,
    hsv_features: np.ndarray,
    camera_ids: List[str],
    class_ids: List[int],
    start_times: List[float],
    end_times: List[float],
    st_validator: SpatioTemporalValidator,
    n: int,
    stage_cfg: DictConfig = None,
    threshold: float = 0.5,
    max_rounds: int = 2,
    orphan_match_threshold: float = 0.0,
    combined_sim: Dict[Tuple[int, int], float] | None = None,
) -> List[Set[int]]:
    """Iteratively absorb orphan (singleton) tracklets into existing clusters.

    Phase 1 (orphan → cluster): For each orphan, compute average cosine
    similarity to every cluster centroid.  If the best match exceeds
    *threshold* and passes spatio-temporal + same-class + cross-camera checks,
    merge the orphan.  Runs for up to *max_rounds* iterations.

    Phase 2 (orphan ↔ orphan): After phase 1, attempt to link remaining orphans
    *to each other* at the lower *orphan_match_threshold*.  This catches medium-
    confidence cross-camera pairs whose similarity score fell below the main
    similarity_threshold (0.45) but still exceed a loose lower bound.
    Only cross-camera, same-class, ST-valid pairs are accepted.
    """
    for round_idx in range(max_rounds):
        # Identify orphans (singleton clusters)
        assigned = set()
        multi_clusters: List[Set[int]] = []
        orphan_indices: List[int] = []

        for cluster in clusters:
            if len(cluster) > 1:
                multi_clusters.append(cluster)
                assigned.update(cluster)
            else:
                orphan_indices.extend(cluster)

        if not orphan_indices or not multi_clusters:
            break

        # Build cluster member embeddings for max-member similarity matching.
        # Max-member is more robust than centroid when cluster members span
        # diverse viewpoints — an orphan matching one member strongly should
        # be absorbed even if the centroid (averaged over all viewpoints) is
        # only moderately similar.
        cluster_member_embs: List[np.ndarray] = []
        cluster_class_ids: List[int] = []
        cluster_cameras: List[Set[str]] = []
        cluster_time_ranges: List[Tuple[float, float]] = []

        for ci, cluster in enumerate(multi_clusters):
            members = list(cluster)
            cluster_member_embs.append(embeddings[members])  # (N_i, D)
            # Majority class
            cls_counts: Dict[int, int] = {}
            for m in members:
                cls_counts[class_ids[m]] = cls_counts.get(class_ids[m], 0) + 1
            cluster_class_ids.append(max(cls_counts, key=cls_counts.get))
            cluster_cameras.append({camera_ids[m] for m in members})
            cluster_time_ranges.append(
                (min(start_times[m] for m in members), max(end_times[m] for m in members))
            )

        merged_count = 0
        remaining_orphans: List[Set[int]] = []

        for orphan in orphan_indices:
            orphan_emb = embeddings[orphan]
            orphan_hsv = hsv_features[orphan]
            # Max-member similarity: highest cosine sim with any member.
            # If combined_sim (reranked + weighted) is available, prefer it
            # for pairs that were already scored in the main pipeline.
            max_sims_list = []
            for ci_idx, embs in enumerate(cluster_member_embs):
                if len(embs) == 0:
                    max_sims_list.append(-1.0)
                    continue
                # Raw cosine sim with all members
                raw_sim = float((embs @ orphan_emb).max())
                # Check combined_sim for refined scores
                if combined_sim:
                    members = list(multi_clusters[ci_idx])
                    cs_vals = []
                    for m in members:
                        cs_val = combined_sim.get((orphan, m)) or combined_sim.get((m, orphan))
                        if cs_val is not None:
                            cs_vals.append(cs_val)
                    if cs_vals:
                        raw_sim = max(raw_sim, max(cs_vals))
                max_sims_list.append(raw_sim)
            max_sims = np.array(max_sims_list)
            order = np.argsort(-max_sims)

            merged = False
            for ci in order:
                if max_sims[ci] < threshold:
                    break
                # Same class check
                if class_ids[orphan] != cluster_class_ids[ci]:
                    continue
                # HSV consistency gate: reject if color is too different
                members = list(multi_clusters[ci])
                hsv_sims = hsv_features[members] @ orphan_hsv
                if float(hsv_sims.max()) < 0.3:
                    continue
                # Cross-camera check (orphan must come from a different camera
                # than at least one cluster member, but must not violate
                # same-camera temporal overlap)
                if camera_ids[orphan] in cluster_cameras[ci]:
                    # Check temporal overlap with same-camera members
                    conflict = False
                    for m in multi_clusters[ci]:
                        if camera_ids[m] == camera_ids[orphan]:
                            if start_times[orphan] <= end_times[m] and start_times[m] <= end_times[orphan]:
                                conflict = True
                                break
                    if conflict:
                        continue
                # Spatio-temporal plausibility with any cluster member
                st_ok = any(
                    st_validator.is_valid_transition(
                        camera_ids[m], camera_ids[orphan],
                        end_times[m], start_times[orphan],
                    ) or st_validator.is_valid_transition(
                        camera_ids[orphan], camera_ids[m],
                        end_times[orphan], start_times[m],
                    )
                    for m in multi_clusters[ci]
                    if camera_ids[m] != camera_ids[orphan]
                )
                if not st_ok:
                    continue

                multi_clusters[ci].add(orphan)
                cluster_cameras[ci].add(camera_ids[orphan])
                # Add orphan embedding to cluster member list
                cluster_member_embs[ci] = np.vstack([
                    cluster_member_embs[ci],
                    embeddings[orphan].reshape(1, -1),
                ])
                merged_count += 1
                merged = True
                break

            if not merged:
                remaining_orphans.append({orphan})

        clusters = multi_clusters + remaining_orphans
        logger.info(
            f"Gallery expansion round {round_idx + 1}: "
            f"absorbed {merged_count} orphans, {len(remaining_orphans)} remain"
        )

        if merged_count == 0:
            break

    # ── Phase 2: orphan ↔ orphan matching at lower threshold ─────────────────
    if orphan_match_threshold > 0:
        # Re-collect current orphans
        final_multi: List[Set[int]] = []
        final_orphan_indices: List[int] = []
        for cluster in clusters:
            if len(cluster) > 1:
                final_multi.append(cluster)
            else:
                final_orphan_indices.extend(cluster)

        if len(final_orphan_indices) >= 2:
            orphan_embs = embeddings[final_orphan_indices]  # (O, D)
            # Pairwise cosine similarity matrix (O × O)
            pairwise_sim = orphan_embs @ orphan_embs.T  # already normalised from stage2/3

            n_orphans = len(final_orphan_indices)

            # Build similarity dict for valid cross-camera, same-class, ST-valid pairs
            orphan_sims: Dict[Tuple[int, int], float] = {}
            for oi in range(n_orphans):
                gi = final_orphan_indices[oi]
                for oj in range(oi + 1, n_orphans):
                    gj = final_orphan_indices[oj]
                    if pairwise_sim[oi, oj] < orphan_match_threshold:
                        continue
                    if class_ids[gi] != class_ids[gj]:
                        continue
                    if camera_ids[gi] == camera_ids[gj]:
                        continue
                    st_ok = (
                        st_validator.is_valid_transition(
                            camera_ids[gi], camera_ids[gj],
                            end_times[gi], start_times[gj],
                        ) or st_validator.is_valid_transition(
                            camera_ids[gj], camera_ids[gi],
                            end_times[gj], start_times[gi],
                        )
                    )
                    if not st_ok:
                        continue
                    orphan_sims[(gi, gj)] = float(pairwise_sim[oi, oj])

            if orphan_sims:
                # Use GraphSolver with bridge pruning + component cap
                # to prevent false transitive chains among orphans.
                orphan_solver = GraphSolver(
                    similarity_threshold=orphan_match_threshold,
                    algorithm="connected_components",
                    bridge_prune_margin=float(stage_cfg.get("graph", {}).get("bridge_prune_margin", 0.0)),
                    max_component_size=int(stage_cfg.get("graph", {}).get("max_component_size", 0)),
                )
                orphan_node_ids = set()
                for (i, j) in orphan_sims:
                    orphan_node_ids.add(i)
                    orphan_node_ids.add(j)
                # Include all orphans as nodes (even those with no valid edges)
                all_orphan_set = set(final_orphan_indices)
                orphan_clusters_raw = orphan_solver.solve(orphan_sims, max(all_orphan_set) + 1)

                # Filter to only include actual orphan tracklets
                new_orphans: List[Set[int]] = []
                new_multi: List[Set[int]] = []
                pairs_linked = sum(1 for c in orphan_clusters_raw if len(c & all_orphan_set) > 1)
                for cluster in orphan_clusters_raw:
                    members = cluster & all_orphan_set
                    if not members:
                        continue
                    if len(members) > 1:
                        new_multi.append(members)
                    else:
                        new_orphans.append(members)
                # Include orphans that had no valid edges
                linked_orphans = set()
                for c in new_multi + new_orphans:
                    linked_orphans |= c
                for gi in final_orphan_indices:
                    if gi not in linked_orphans:
                        new_orphans.append({gi})

                clusters = final_multi + new_multi + new_orphans
                logger.info(
                    f"Orphan↔orphan matching (threshold={orphan_match_threshold:.2f}): "
                    f"{len(orphan_sims)} candidate pairs → {len(new_multi)} new clusters, "
                    f"{len(new_orphans)} still orphaned"
                )
            else:
                clusters = final_multi + [set(f) for f in [[gi] for gi in final_orphan_indices]]
                logger.info("Orphan↔orphan matching: no valid pairs found")

    return clusters


def _resolve_same_camera_conflicts(
    clusters: List[Set[int]],
    camera_ids: List[str],
    start_times: List[float],
    end_times: List[float],
    similarities: Dict[Tuple[int, int], float],
) -> List[Set[int]]:
    """Split clusters that contain temporally-overlapping same-camera tracklets.

    Two tracklets from the same camera whose time spans overlap cannot be the
    same person. When Louvain merges them transitively (A-cam0 ~ B-cam1 ~ C-cam0),
    we need to split the cluster so A and C end up in separate identities.

    Uses graph coloring on the conflict sub-graph: nodes that conflict (same-camera
    temporal overlap) get different colors, and each color group becomes its own
    sub-cluster. Similarity-based ordering ensures the strongest links are preserved.
    """
    refined: List[Set[int]] = []
    total_splits = 0

    for cluster in clusters:
        if len(cluster) <= 1:
            refined.append(cluster)
            continue

        members = list(cluster)

        # Find conflict pairs: same camera + overlapping time
        conflicts = []
        for i_idx in range(len(members)):
            for j_idx in range(i_idx + 1, len(members)):
                a, b = members[i_idx], members[j_idx]
                if camera_ids[a] != camera_ids[b]:
                    continue
                # Overlap check: a's interval intersects b's interval
                if start_times[a] <= end_times[b] and start_times[b] <= end_times[a]:
                    conflicts.append((a, b))

        if not conflicts:
            refined.append(cluster)
            continue

        # Build conflict graph and use greedy coloring to find the minimum
        # number of sub-clusters needed to resolve all conflicts
        conflict_graph = nx.Graph()
        conflict_graph.add_nodes_from(members)
        conflict_graph.add_edges_from(conflicts)

        # Order nodes by their total similarity to others in the cluster
        # (strongest-linked nodes get colored first, keeping them together)
        node_strength = {}
        for node in members:
            total_sim = 0.0
            for other in members:
                if node == other:
                    continue
                total_sim += similarities.get((node, other), 0.0)
                total_sim += similarities.get((other, node), 0.0)
            node_strength[node] = total_sim

        ordered_nodes = sorted(members, key=lambda n: node_strength[n], reverse=True)

        # Use custom strategy that respects our similarity-based ordering:
        # strongest-linked nodes get colored first, keeping them together
        def _similarity_order(G, colors):
            return ordered_nodes

        coloring = nx.coloring.greedy_color(
            conflict_graph,
            strategy=_similarity_order,
        )

        # Group by color
        color_groups: Dict[int, Set[int]] = {}
        for node, color in coloring.items():
            color_groups.setdefault(color, set()).add(node)

        refined.extend(color_groups.values())
        total_splits += len(color_groups) - 1

    if total_splits > 0:
        logger.info(
            f"Conflict resolution: split {total_splits} clusters, "
            f"now {len(refined)} total clusters"
        )
    else:
        logger.info("Conflict resolution: no same-camera conflicts found")

    return refined
