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

import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
from loguru import logger
from omegaconf import DictConfig

from src.core.data_models import GlobalTrajectory, Tracklet, TrackletFeatures
from src.core.io_utils import load_multi_query_embeddings, save_global_trajectories
from src.stage3_indexing.faiss_index import FAISSIndex
from src.stage3_indexing.metadata_store import MetadataStore
from src.stage4_association.camera_bias import CameraDistanceBias, ZoneTransitionModel
from src.stage4_association.aflink import aflink_post_association
from src.stage4_association.fic import per_camera_whiten, cross_camera_augment, iterative_fac
from src.stage4_association.global_trajectories import merge_tracklets_to_trajectories
from src.stage4_association.graph_solver import GraphSolver
from src.stage4_association.query_expansion import average_query_expansion_batched
from src.stage4_association.reranking import k_reciprocal_rerank
from src.stage4_association.similarity import (
    compute_combined_similarity,
    compute_temporal_overlap_ratio,
    mutual_nearest_neighbor_filter,
)
from src.stage4_association.spatial_temporal import SpatioTemporalValidator
from src.stage4_association.zone_scoring import ZoneScorer


def _flatten_multi_query_embeddings(
    mq_embeddings: List[Optional[np.ndarray]],
    camera_ids: List[str],
) -> tuple[Optional[np.ndarray], List[int], List[str]]:
    """Flatten MQ arrays and repeat camera IDs per representative embedding."""
    flattened: List[np.ndarray] = []
    sizes: List[int] = []
    mq_camera_ids: List[str] = []

    for camera_id, mq in zip(camera_ids, mq_embeddings):
        if mq is None:
            continue
        flattened.append(mq)
        sizes.append(mq.shape[0])
        mq_camera_ids.extend([camera_id] * mq.shape[0])

    if not flattened:
        return None, [], []

    return np.concatenate(flattened, axis=0), sizes, mq_camera_ids


def _restore_multi_query_embeddings(
    mq_embeddings: List[Optional[np.ndarray]],
    mq_flat: np.ndarray,
    sizes: List[int],
) -> None:
    """Restore flattened MQ rows after FIC whitening."""
    offset = 0
    size_idx = 0
    for idx, mq in enumerate(mq_embeddings):
        if mq is None:
            continue
        size = sizes[size_idx]
        mq_embeddings[idx] = mq_flat[offset:offset + size]
        offset += size
        size_idx += 1


def _compute_multi_query_pair_similarity(
    avg_i: np.ndarray,
    avg_j: np.ndarray,
    mq_i: Optional[np.ndarray],
    mq_j: Optional[np.ndarray],
    mq_weight: float,
) -> float:
    """Blend average-embedding cosine with max-of-KxK multi-query cosine."""
    avg_sim = float(avg_i @ avg_j)
    if mq_i is None and mq_j is None:
        return avg_sim

    if mq_i is not None and mq_j is not None:
        mq_sim = float((mq_i @ mq_j.T).max())
    elif mq_i is not None:
        mq_sim = float((mq_i @ avg_j).max())
    else:
        mq_sim = float((avg_i @ mq_j.T).max())

    return (1.0 - mq_weight) * avg_sim + mq_weight * mq_sim


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

    # FIC config (used by both primary and secondary embeddings)
    fic_cfg = stage_cfg.get("fic", {})

    mq_cfg = stage_cfg.get("multi_query", {})
    mq_enabled = bool(mq_cfg.get("enabled", False))
    mq_weight = float(mq_cfg.get("weight", 0.5))
    mq_dir = mq_cfg.get("dir", "")
    mq_embeddings: List[Optional[np.ndarray]] = [None] * n
    if mq_enabled:
        mq_input_dir = Path(mq_dir) if mq_dir else output_dir.parent / "stage2"
        mq_embeddings = load_multi_query_embeddings(mq_input_dir, n)
        mq_count = sum(mq is not None for mq in mq_embeddings)
        logger.info(
            f"Multi-query embeddings loaded: {mq_count}/{n} tracklets "
            f"(weight={mq_weight:.2f})"
        )

    # Optional: load secondary embeddings for score-level fusion.
    # Instead of concatenating features (which mixes uncalibrated spaces),
    # we compute separate appearance similarities and blend them later.
    sec_cfg = stage_cfg.get("secondary_embeddings", {})
    sec_path = sec_cfg.get("path", "")
    sec_embeddings: Optional[np.ndarray] = None
    sec_weight = 0.0
    if sec_path and float(sec_cfg.get("weight", 0.3)) > 0 and Path(sec_path).exists():
        sec_raw = np.load(sec_path).astype(np.float32)
        if sec_raw.shape[0] == n:
            sec_weight = float(sec_cfg.get("weight", 0.3))
            # L2-normalize
            sec_norms = np.linalg.norm(sec_raw, axis=1, keepdims=True)
            sec_embeddings = sec_raw / np.maximum(sec_norms, 1e-8)
            logger.info(
                f"Secondary embeddings loaded: {sec_embeddings.shape[1]}D, "
                f"weight={sec_weight:.2f} (score-level fusion)"
            )
            # Apply FIC whitening to secondary embeddings separately
            if fic_cfg.get("enabled", False):
                sec_embeddings = per_camera_whiten(
                    sec_embeddings,
                    camera_ids,
                    regularisation=float(fic_cfg.get("regularisation", 3.0)),
                    min_samples=int(fic_cfg.get("min_samples", 5)),
                )
                logger.info("Applied FIC whitening to secondary embeddings")
        else:
            logger.warning(f"Secondary embeddings shape mismatch: {sec_raw.shape[0]} vs {n}")
    elif sec_path:
        logger.warning(
            f"Secondary embeddings file not found: {sec_path}. "
            f"Falling back to primary-only (sec_weight=0.0)."
        )

    tert_cfg = stage_cfg.get("tertiary_embeddings", {})
    tert_path = tert_cfg.get("path", "")
    tert_embeddings: Optional[np.ndarray] = None
    tert_weight = 0.0
    if tert_path and float(tert_cfg.get("weight", 0.0)) > 0 and Path(tert_path).exists():
        tert_raw = np.load(tert_path).astype(np.float32)
        if tert_raw.shape[0] == n:
            tert_weight = float(tert_cfg.get("weight", 0.0))
            tert_norms = np.linalg.norm(tert_raw, axis=1, keepdims=True)
            tert_embeddings = tert_raw / np.maximum(tert_norms, 1e-8)
            logger.info(
                f"Tertiary embeddings loaded: {tert_embeddings.shape[1]}D, "
                f"weight={tert_weight:.2f} (score-level fusion)"
            )
            if fic_cfg.get("enabled", False):
                tert_embeddings = per_camera_whiten(
                    tert_embeddings,
                    camera_ids,
                    regularisation=float(fic_cfg.get("regularisation", 3.0)),
                    min_samples=int(fic_cfg.get("min_samples", 5)),
                )
                logger.info("Applied FIC whitening to tertiary embeddings")
        else:
            logger.warning(f"Tertiary embeddings shape mismatch: {tert_raw.shape[0]} vs {n}")
    elif tert_path and float(tert_cfg.get("weight", 0.0)) > 0:
        logger.warning(
            f"Tertiary embeddings file not found: {tert_path}. "
            f"Falling back to primary/secondary fusion only (tert_weight=0.0)."
        )

    if sec_weight + tert_weight > 1.0 + 1e-8:
        raise ValueError(
            "stage4 association ensemble weights invalid: "
            f"secondary({sec_weight:.3f}) + tertiary({tert_weight:.3f}) must be <= 1.0"
        )

    # Step 0: Per-camera feature whitening (FIC) — AIC21 1st-place technique.
    # Removes camera-specific bias (lighting, viewpoint) from embeddings.
    if fic_cfg.get("enabled", False):
        embeddings = per_camera_whiten(
            embeddings,
            camera_ids,
            regularisation=float(fic_cfg.get("regularisation", 3.0)),
            min_samples=int(fic_cfg.get("min_samples", 5)),
        )

        mq_flat, mq_sizes, mq_camera_ids = _flatten_multi_query_embeddings(
            mq_embeddings,
            camera_ids,
        )
        if mq_flat is not None:
            mq_flat = per_camera_whiten(
                mq_flat,
                mq_camera_ids,
                regularisation=float(fic_cfg.get("regularisation", 3.0)),
                min_samples=int(fic_cfg.get("min_samples", 5)),
            )
            _restore_multi_query_embeddings(mq_embeddings, mq_flat, mq_sizes)

    # Step 0b: Cross-camera feature augmentation (FAC) — AIC21 technique.
    # Pulls each feature toward its cross-camera KNN consensus.
    fac_cfg = stage_cfg.get("fac", {})
    if fac_cfg.get("enabled", False):
        fac_epochs = int(fac_cfg.get("epochs", 1))
        fac_knn = int(fac_cfg.get("knn", 20))
        fac_lr = float(fac_cfg.get("learning_rate", 0.5))
        fac_beta = float(fac_cfg.get("beta", 0.08))
        if fac_epochs > 1:
            embeddings = iterative_fac(
                embeddings, camera_ids,
                epochs=fac_epochs, knn=fac_knn,
                learning_rate=fac_lr, beta=fac_beta,
            )
        else:
            embeddings = cross_camera_augment(
                embeddings, camera_ids,
                knn=fac_knn, learning_rate=fac_lr, beta=fac_beta,
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
        if mq_enabled and any(mq is not None for mq in mq_embeddings):
            candidate_pairs = _build_all_cross_camera_pairs_multi_query(
                n,
                embeddings,
                mq_embeddings,
                camera_ids,
                class_ids,
                min_similarity=min_sim,
                mq_weight=mq_weight,
            )
        else:
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

    # Step 3b: Score-level fusion with secondary/tertiary embeddings
    if (sec_embeddings is not None and sec_weight > 0) or (tert_embeddings is not None and tert_weight > 0):
        primary_weight = 1.0 - sec_weight - tert_weight
        logger.info(
            "Blending ensemble appearance sim "
            f"(w_pri={primary_weight:.2f}, w_sec={sec_weight:.2f}, w_tert={tert_weight:.2f})..."
        )
        blended = 0
        for (i, j), pri_sim in list(appearance_sim.items()):
            sim = primary_weight * pri_sim
            if sec_embeddings is not None and sec_weight > 0:
                sim += sec_weight * float(np.dot(sec_embeddings[i], sec_embeddings[j]))
            if tert_embeddings is not None and tert_weight > 0:
                sim += tert_weight * float(np.dot(tert_embeddings[i], tert_embeddings[j]))
            appearance_sim[(i, j)] = sim
            blended += 1
        logger.info(f"Blended {blended} pairs with ensemble embeddings")

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
        temporal_overlap_cfg=stage_cfg.get("temporal_overlap"),
    )

    logger.info(f"Combined similarity pairs: {len(combined_sim)}")

    # Step 5a: Per-camera-pair similarity normalization.
    # Center each camera-pair distribution on the global mean of eligible pairs
    # to reduce systematic cross-camera score bias before graph construction.
    pair_norm_cfg = stage_cfg.get("camera_pair_norm", {})
    if pair_norm_cfg.get("enabled", False) and combined_sim:
        from collections import defaultdict

        min_pairs = int(pair_norm_cfg.get("min_pairs", 10))
        pair_sims: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        for (i, j), sim in combined_sim.items():
            pair_key = tuple(sorted((camera_ids[i], camera_ids[j])))
            pair_sims[pair_key].append(sim)

        pair_means: Dict[Tuple[str, str], float] = {}
        eligible_means: List[float] = []
        for pair_key, sims in pair_sims.items():
            if len(sims) >= min_pairs:
                mean_sim = float(np.mean(sims))
                pair_means[pair_key] = mean_sim
                eligible_means.append(mean_sim)

        if eligible_means:
            global_mean = float(np.mean(eligible_means))
            adjusted = 0
            for (i, j), sim in list(combined_sim.items()):
                pair_key = tuple(sorted((camera_ids[i], camera_ids[j])))
                pair_mean = pair_means.get(pair_key)
                if pair_mean is None:
                    continue
                combined_sim[(i, j)] = sim + (global_mean - pair_mean)
                adjusted += 1

            logger.info(
                "Camera-pair normalization: adjusted "
                f"{adjusted} pairs across {len(pair_means)} camera combos "
                f"(global_mean={global_mean:.3f}, min_pairs={min_pairs})"
            )
        else:
            logger.info(
                "Camera-pair normalization skipped: no camera pairs met "
                f"min_pairs={min_pairs}"
            )

    # Step 5b: Camera distance bias adjustment (iterative)
    camera_bias_cfg = stage_cfg.get("camera_bias", {})
    if camera_bias_cfg.get("enabled", False):
        cid_bias_path = camera_bias_cfg.get("cid_bias_npy_path", "")
        cid_bias_applied = False
        if cid_bias_path and Path(cid_bias_path).exists():
            cid_bias_matrix = np.load(cid_bias_path).astype(np.float32)
            cid_mapping_path = Path(cid_bias_path).with_suffix(".json")
            if cid_mapping_path.exists():
                with open(cid_mapping_path, encoding="utf-8") as f:
                    cam_names = json.load(f).get("cameras", [])
            else:
                cam_names = sorted(set(camera_ids))
            cam2idx = {camera_name: idx for idx, camera_name in enumerate(cam_names)}

            adjusted_count = 0
            missing_count = 0
            for (i, j), sim in list(combined_sim.items()):
                ci = cam2idx.get(camera_ids[i])
                cj = cam2idx.get(camera_ids[j])
                if ci is None or cj is None:
                    missing_count += 1
                    continue
                if ci >= cid_bias_matrix.shape[0] or cj >= cid_bias_matrix.shape[1]:
                    missing_count += 1
                    continue
                combined_sim[(i, j)] = sim + float(cid_bias_matrix[ci, cj])
                adjusted_count += 1

            logger.info(
                f"CID_BIAS: adjusted {adjusted_count} pairs from {cid_bias_path}"
                + (f" ({missing_count} pairs skipped: unmapped cameras)" if missing_count else "")
            )
            cid_bias_applied = True

        cam_bias = CameraDistanceBias()

        # Load pre-learned JSON bias if available
        bias_path = camera_bias_cfg.get("bias_path")
        if bias_path and Path(bias_path).exists():
            cam_bias.load(bias_path)
            logger.info(f"Loaded camera bias from {bias_path}")
            combined_sim = cam_bias.adjust_similarity_matrix(combined_sim, camera_ids)
            logger.info("Applied pre-learned camera distance bias")
        elif not cid_bias_applied:
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
                tmp_clusters = tmp_solver.solve(combined_sim, n, camera_ids, start_times, end_times)
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
        zone_data_path = zone_cfg.get("zone_data_path")
        if zone_data_path and Path(zone_data_path).exists():
            zone_scorer = ZoneScorer(
                zone_data_path,
                min_count=int(zone_cfg.get("min_count", 2)),
            )
            # Extract entry/exit positions from Stage 1 tracklets
            tracklet_lookup = {}
            for cam_id, tracklets in tracklets_by_camera.items():
                for t in tracklets:
                    tracklet_lookup[(t.camera_id, t.track_id)] = t
            entry_positions = []
            exit_positions = []
            for feat in features:
                t = tracklet_lookup.get((feat.camera_id, feat.track_id))
                if t and t.frames:
                    fb = t.frames[0].bbox  # (x1, y1, x2, y2)
                    entry_positions.append(((fb[0] + fb[2]) / 2, (fb[1] + fb[3]) / 2))
                    lb = t.frames[-1].bbox
                    exit_positions.append(((lb[0] + lb[2]) / 2, (lb[1] + lb[3]) / 2))
                else:
                    entry_positions.append(None)
                    exit_positions.append(None)
            entry_zones, exit_zones = zone_scorer.assign_zones(
                entry_positions, exit_positions, camera_ids,
            )
            combined_sim = zone_scorer.apply_to_similarities(
                combined_sim, camera_ids, entry_zones, exit_zones,
                bonus=float(zone_cfg.get("bonus", 0.03)),
                penalty=float(zone_cfg.get("penalty", 0.03)),
            )
        else:
            logger.warning(f"Zone data file not found: {zone_data_path}")

    # Step 5d: Per-camera-pair similarity boost (targeted threshold lowering)
    # Instead of a single global threshold, boost similarities for specific
    # camera pairs that have high true-match rate but low raw similarity.
    # This effectively lowers the threshold for those pairs while keeping
    # the global threshold conservative for well-connected pairs.
    pair_boost_cfg = stage_cfg.get("camera_pair_boost", {})
    if pair_boost_cfg.get("enabled", False):
        boosts = pair_boost_cfg.get("boosts", {})
        boost_count = 0
        for (i, j), sim in list(combined_sim.items()):
            cam_i, cam_j = camera_ids[i], camera_ids[j]
            key = f"{cam_i}-{cam_j}"
            key_rev = f"{cam_j}-{cam_i}"
            boost_val = boosts.get(key, boosts.get(key_rev, 0.0))
            if boost_val != 0.0:
                combined_sim[(i, j)] = sim + boost_val
                boost_count += 1
        if boost_count > 0:
            logger.info(f"Camera pair boost: adjusted {boost_count} pairs")

    # Step 6: Build graph and solve
    # Phase 1: Reciprocal best-match seeding (SOTA technique, threshold-free).
    # For each tracklet, find its best match in each other camera.  If two
    # tracklets are each other's best cross-camera match AND their similarity
    # exceeds a loose floor, seed them as an initial pair.  This is robust
    # because it's rank-based — works regardless of absolute similarity scale.
    rbm_cfg = stage_cfg.get("reciprocal_best_match", {})
    rbm_edges: Dict[Tuple[int, int], float] = {}
    if rbm_cfg.get("enabled", False):
        rbm_floor = float(rbm_cfg.get("min_similarity", 0.20))
        rbm_edges = _reciprocal_best_match(
            combined_sim, camera_ids, class_ids,
            start_times, end_times, st_validator,
            min_similarity=rbm_floor,
        )
        # Inject RBM edges into combined_sim (they bypass threshold)
        for (i, j), sim in rbm_edges.items():
            if (i, j) not in combined_sim:
                combined_sim[(i, j)] = sim
        logger.info(
            f"Reciprocal best-match: {len(rbm_edges)} seed pairs "
            f"(floor={rbm_floor:.2f})"
        )

    if combined_sim:
        sim_vals = list(combined_sim.values())
        threshold = float(stage_cfg.graph.similarity_threshold)
        n_above = sum(1 for s in sim_vals if s >= threshold)
        logger.info(
            f"Combined sim stats: {len(sim_vals)} pairs, "
            f"min={min(sim_vals):.3f} median={np.median(sim_vals):.3f} "
            f"max={max(sim_vals):.3f}, {n_above} above threshold {threshold}"
        )

    # CSLS hubness reduction: penalize vectors with high average similarity
    # to their neighbors (reduces "universal hub" false positives)
    csls_cfg = stage_cfg.get("csls", {})
    if csls_cfg.get("enabled", False) and combined_sim:
        csls_k = int(csls_cfg.get("k", 10))
        # Collect top-K similarities per tracklet
        from collections import defaultdict
        per_node_sims: Dict[int, List[float]] = defaultdict(list)
        for (i, j), s in combined_sim.items():
            per_node_sims[i].append(s)
            per_node_sims[j].append(s)
        # Compute mean of top-K for each node
        hub_penalty: Dict[int, float] = {}
        for node, sims in per_node_sims.items():
            topk = sorted(sims, reverse=True)[:csls_k]
            hub_penalty[node] = np.mean(topk) if topk else 0.0
        # Apply CSLS: sim_adj = 2*sim - penalty_i - penalty_j
        adjusted = {}
        for (i, j), s in combined_sim.items():
            adjusted[(i, j)] = 2.0 * s - hub_penalty.get(i, 0.0) - hub_penalty.get(j, 0.0)
        combined_sim = adjusted
        logger.info(
            f"CSLS hubness reduction (k={csls_k}): "
            f"adjusted {len(combined_sim)} pairs"
        )

    # Phase 2: Graph solving. RBM edges that are below the normal threshold
    # get boosted above the bridge pruning threshold so they survive as
    # anchor connections.  Without this, bridge pruning (threshold + margin)
    # would remove the very edges RBM identified as high-confidence.
    graph_threshold = float(stage_cfg.graph.similarity_threshold)
    bridge_margin = float(stage_cfg.graph.get("bridge_prune_margin", 0.0))
    bridge_threshold = graph_threshold + bridge_margin
    solve_sim = dict(combined_sim)
    if rbm_edges:
        boosted = 0
        for (i, j), sim in rbm_edges.items():
            if sim < bridge_threshold:
                # Boost above bridge prune threshold so RBM links survive
                solve_sim[(i, j)] = bridge_threshold + 0.01
                boosted += 1
        if boosted > 0:
            logger.info(
                f"RBM boost: {boosted} seed edges boosted above bridge threshold "
                f"({bridge_threshold:.3f})"
            )

    # Step 5e: Per-camera-pair adaptive thresholds.
    # Different camera pairs have different similarity distributions.
    # Apply pair-specific thresholds to reduce fragmentation on hard pairs
    # while maintaining precision on easy pairs.
    pair_thresholds_cfg = stage_cfg.get("pair_thresholds", {})
    if pair_thresholds_cfg.get("enabled", False):
        pair_thresh_map = pair_thresholds_cfg.get("thresholds", {})
        if pair_thresh_map:
            removed = 0
            keys_to_remove = []
            for (i, j), sim in solve_sim.items():
                cam_pair = tuple(sorted([camera_ids[i], camera_ids[j]]))
                pair_key = f"{cam_pair[0]}-{cam_pair[1]}"
                pair_thresh = pair_thresh_map.get(pair_key, graph_threshold)
                if sim < pair_thresh:
                    keys_to_remove.append((i, j))
            for key in keys_to_remove:
                del solve_sim[key]
                removed += 1
            # Set graph threshold to minimum pair threshold so pre-filtered
            # edges are not double-filtered by the graph solver.
            min_pair_thresh = min(pair_thresh_map.values()) if pair_thresh_map else graph_threshold
            graph_threshold = min(graph_threshold, min_pair_thresh)
            logger.info(
                f"Per-pair thresholds: removed {removed} sub-threshold edges, "
                f"effective graph threshold={graph_threshold:.3f}"
            )

    # Step 5f: Intra-camera ReID merge — add same-camera, non-overlapping
    # tracklet pairs with high cosine similarity to the graph.  This catches
    # vehicles that exit and re-enter a camera (occlusion, stop, or tracking
    # loss) which stage 1 IoU-based merge can't handle.
    intra_merge_cfg = stage_cfg.get("intra_camera_merge", {})
    if intra_merge_cfg.get("enabled", False):
        intra_threshold = float(intra_merge_cfg.get("threshold", 0.70))
        max_time_gap = float(intra_merge_cfg.get("max_time_gap", 120.0))
        # Group tracklets by camera
        cam_to_indices: Dict[str, List[int]] = {}
        for idx in range(n):
            cam_to_indices.setdefault(camera_ids[idx], []).append(idx)
        intra_added = 0
        for cam, indices_list in cam_to_indices.items():
            for ii in range(len(indices_list)):
                for jj in range(ii + 1, len(indices_list)):
                    a, b = indices_list[ii], indices_list[jj]
                    # Skip if temporally overlapping (different identities)
                    if start_times[a] <= end_times[b] and start_times[b] <= end_times[a]:
                        continue
                    # Skip if time gap is too large
                    time_gap = max(start_times[a] - end_times[b],
                                   start_times[b] - end_times[a])
                    if time_gap > max_time_gap:
                        continue
                    # Cosine similarity from FIC+QE embeddings
                    sim = float(embeddings[a] @ embeddings[b])
                    if sim >= intra_threshold:
                        key = (min(a, b), max(a, b))
                        if key not in solve_sim or solve_sim[key] < sim:
                            solve_sim[key] = sim
                            intra_added += 1
        if intra_added > 0:
            logger.info(
                f"Intra-camera ReID merge: added {intra_added} same-camera pairs "
                f"(threshold={intra_threshold:.2f}, max_gap={max_time_gap:.0f}s)"
            )

    solver = GraphSolver(
        similarity_threshold=graph_threshold,
        algorithm=stage_cfg.graph.algorithm,
        louvain_resolution=stage_cfg.graph.get("louvain_resolution", 1.0),
        louvain_seed=int(stage_cfg.graph.get("louvain_seed", 42)),
        bridge_prune_margin=float(stage_cfg.graph.get("bridge_prune_margin", 0.0)),
        max_component_size=int(stage_cfg.graph.get("max_component_size", 0)),
    )

    clusters = solver.solve(solve_sim, n, camera_ids, start_times, end_times)
    logger.info(f"Graph solver found {len(clusters)} identity clusters")

    # Step 6b: Resolve same-camera conflicts
    clusters = _resolve_same_camera_conflicts(
        clusters, camera_ids, start_times, end_times, combined_sim,
    )

    # Step 6c: Hierarchical centroid-based expansion (SOTA technique).
    # After initial clustering, compute cluster centroids (averaged + L2-normed
    # embeddings) which are more robust than individual embeddings because noise
    # averages out. Then try to merge orphans at a lower threshold.
    hierarch_cfg = stage_cfg.get("hierarchical", {})
    if hierarch_cfg.get("enabled", False):
        clusters = _hierarchical_centroid_expansion(
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
            combined_sim=combined_sim,
            hierarch_cfg=hierarch_cfg,
        )

        # Re-resolve same-camera conflicts after hierarchical expansion
        clusters = _resolve_same_camera_conflicts(
            clusters, camera_ids, start_times, end_times, combined_sim,
        )

    # Step 6c-legacy: Gallery expansion (if hierarchical is disabled, use legacy)
    # Also run after hierarchical as a second absorption pass with max-member sim
    gallery_cfg = stage_cfg.get("gallery_expansion", {})
    run_gallery = gallery_cfg.get("enabled", False)
    if run_gallery:
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

    # Step 6d: Re-resolve same-camera conflicts introduced by orphan-orphan matching
    # or gallery expansion.  Cross-camera merges can transitively place
    # same-camera tracklets in the same cluster.
    if run_gallery or not hierarch_cfg.get("enabled", False):
        clusters = _resolve_same_camera_conflicts(
            clusters, camera_ids, start_times, end_times, combined_sim,
        )

    # Step 6e-pre: Sub-cluster temporal splitting (AIC21 technique).
    # Splits clusters that contain a "silence" gap — a time window where NO member
    # tracklet is active — longer than min_gap seconds, AND where the cross-gap
    # average cosine similarity is below split_threshold.
    # Targets conflation: same-looking vehicles driving through the scene at
    # different times incorrectly merged into one trajectory.
    temporal_split_cfg = stage_cfg.get("temporal_split", {})
    if temporal_split_cfg.get("enabled", False):
        ts_min_gap = float(temporal_split_cfg.get("min_gap", 60.0))
        ts_split_thresh = float(temporal_split_cfg.get("split_threshold", 0.50))
        clusters_before = len(clusters)
        clusters = _temporal_split_clusters(
            clusters, start_times, end_times, embeddings,
            min_gap=ts_min_gap,
            split_threshold=ts_split_thresh,
        )
        logger.info(
            f"Temporal split: {clusters_before} → {len(clusters)} clusters "
            f"(+{len(clusters) - clusters_before} splits, "
            f"min_gap={ts_min_gap:.0f}s, threshold={ts_split_thresh:.2f})"
        )

    # Step 6e: Post-cluster verification — check internal connectivity and
    # eject members whose maximum cosine similarity to any other cluster member
    # is below a minimum threshold.  This catches false transitive merges where
    # A→B→C but A-C similarity is very low.
    verify_cfg = stage_cfg.get("cluster_verify", {})
    if verify_cfg.get("enabled", False):
        min_connectivity = float(verify_cfg.get("min_connectivity", 0.30))
        ejected_total = 0
        new_clusters = []
        for cluster in clusters:
            if len(cluster) <= 2:
                new_clusters.append(cluster)
                continue
            members = list(cluster)
            eject = set()
            for m in members:
                # Check max cosine sim to any other member
                max_sim = -1.0
                for m2 in members:
                    if m2 == m:
                        continue
                    sim = float(embeddings[m] @ embeddings[m2])
                    if sim > max_sim:
                        max_sim = sim
                if max_sim < min_connectivity:
                    eject.add(m)
            if eject:
                remaining = cluster - eject
                if len(remaining) >= 2:
                    new_clusters.append(remaining)
                else:
                    for m in remaining:
                        new_clusters.append({m})
                for m in eject:
                    new_clusters.append({m})
                ejected_total += len(eject)
            else:
                new_clusters.append(cluster)
        if ejected_total > 0:
            logger.info(
                f"Cluster verification: ejected {ejected_total} weakly-connected "
                f"members (min_connectivity={min_connectivity:.2f})"
            )
        clusters = new_clusters

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

    aflink_cfg = stage_cfg.get("aflink", {})
    if aflink_cfg.get("enabled", False):
        trajectories = aflink_post_association(
            trajectories=trajectories,
            feature_to_tracklet_key=feature_to_tracklet_key,
            tracklet_lookup=tracklet_lookup,
            embeddings=embeddings,
            combined_sim=combined_sim,
            max_time_gap_frames=int(aflink_cfg.get("max_time_gap_frames", 150)),
            max_spatial_gap_px=float(aflink_cfg.get("max_spatial_gap_px", 200.0)),
            min_direction_cos=float(aflink_cfg.get("min_direction_cos", 0.7)),
            min_velocity_ratio=float(aflink_cfg.get("min_velocity_ratio", 0.5)),
            velocity_window=int(aflink_cfg.get("velocity_window", 5)),
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


def _build_all_cross_camera_pairs_multi_query(
    n: int,
    embeddings: np.ndarray,
    mq_embeddings: List[Optional[np.ndarray]],
    camera_ids: List[str],
    class_ids: List[int],
    min_similarity: float = 0.0,
    mq_weight: float = 0.5,
) -> List[Tuple[int, int, float]]:
    """Exhaustive cross-camera candidate generation with MQ-aware appearance.

    Each pair blends the original averaged similarity with a max-of-KxK
    multi-query similarity. If only one side has MQ embeddings, the MQ branch
    falls back to a Kx1 max against the other tracklet's averaged embedding.
    """
    from collections import defaultdict

    cam_to_idxs: Dict[str, List[int]] = defaultdict(list)
    for idx, cam in enumerate(camera_ids):
        cam_to_idxs[cam].append(idx)

    cameras = sorted(cam_to_idxs.keys())
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

    for a_idx, cam_a in enumerate(cameras):
        a_global = cam_to_idxs[cam_a]
        scene_a = cam_scenes[cam_a]

        for cam_b in cameras[a_idx + 1:]:
            scene_b = cam_scenes[cam_b]
            if scene_blocking_active and scene_a and scene_b and scene_a != scene_b:
                skipped_cross_scene += len(a_global) * len(cam_to_idxs[cam_b])
                continue

            b_global = cam_to_idxs[cam_b]
            for gi in a_global:
                for gj in b_global:
                    if class_ids[gi] != class_ids[gj]:
                        continue
                    sim = _compute_multi_query_pair_similarity(
                        embeddings[gi],
                        embeddings[gj],
                        mq_embeddings[gi],
                        mq_embeddings[gj],
                        mq_weight,
                    )
                    if sim >= min_similarity:
                        candidate_pairs.append((gi, gj, sim))

    logger.info(
        f"Exhaustive cross-camera MQ pairs: {len(candidate_pairs)} "
        f"(min_sim={min_similarity:.2f}, mq_weight={mq_weight:.2f}, {len(cameras)} cameras"
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


# ---------------------------------------------------------------------------
# Reciprocal best-match seeding (SOTA threshold-free matching)
# ---------------------------------------------------------------------------

def _reciprocal_best_match(
    combined_sim: Dict[Tuple[int, int], float],
    camera_ids: List[str],
    class_ids: List[int],
    start_times: List[float],
    end_times: List[float],
    st_validator: SpatioTemporalValidator,
    min_similarity: float = 0.20,
) -> Dict[Tuple[int, int], float]:
    """Find reciprocal best-match pairs across cameras (threshold-free).

    For each tracklet i, find its highest-similarity match j in each other
    camera.  If j's best match in camera(i) is also i, and the similarity
    exceeds a loose floor, they form a reciprocal best-match pair.

    This is robust to absolute similarity compression because it's rank-based:
    even if max cross-camera similarity is only 0.35, that's still good enough
    if it's the BEST match on both sides.

    Args:
        combined_sim: Pairwise combined similarity scores.
        camera_ids: Camera ID for each tracklet.
        class_ids: Class ID for each tracklet.
        start_times, end_times: Temporal bounds per tracklet.
        st_validator: Spatio-temporal validator.
        min_similarity: Absolute floor — pairs below this are never linked.

    Returns:
        Dict of reciprocal best-match pairs and their similarities.
    """
    from collections import defaultdict

    # Build per-tracklet best match in each other camera
    # best_match[i][cam_j] = (j, similarity)
    best_match: Dict[int, Dict[str, Tuple[int, float]]] = defaultdict(dict)

    for (i, j), sim in combined_sim.items():
        if sim < min_similarity:
            continue
        cam_j = camera_ids[j]
        cam_i = camera_ids[i]
        if cam_i == cam_j:
            continue

        # Update best match for i in cam_j's direction
        if cam_j not in best_match[i] or sim > best_match[i][cam_j][1]:
            best_match[i][cam_j] = (j, sim)
        # Update best match for j in cam_i's direction
        if cam_i not in best_match[j] or sim > best_match[j][cam_i][1]:
            best_match[j][cam_i] = (i, sim)

    # Find reciprocal pairs
    rbm_pairs: Dict[Tuple[int, int], float] = {}

    seen = set()
    for i, per_cam in best_match.items():
        for cam_j, (j, sim_ij) in per_cam.items():
            if (i, j) in seen or (j, i) in seen:
                continue
            # Check reciprocity: j's best match in camera(i) must be i
            cam_i = camera_ids[i]
            if cam_i in best_match.get(j, {}):
                best_from_j, sim_ji = best_match[j][cam_i]
                if best_from_j == i:
                    # Reciprocal! Use the minimum similarity as the edge weight
                    pair_sim = min(sim_ij, sim_ji)
                    if pair_sim >= min_similarity:
                        # Same-camera temporal overlap check (should be cross-cam but verify)
                        if camera_ids[i] != camera_ids[j]:
                            key = (min(i, j), max(i, j))
                            rbm_pairs[key] = pair_sim
                            seen.add(key)

    return rbm_pairs


# ---------------------------------------------------------------------------
# Hierarchical centroid-based expansion (SOTA multi-pass matching)
# ---------------------------------------------------------------------------

def _compute_cluster_centroids(
    clusters: List[Set[int]],
    embeddings: np.ndarray,
) -> np.ndarray:
    """Compute L2-normalised centroid for each cluster.

    Centroids average out viewpoint-specific noise, making them more
    discriminative than individual embeddings for cross-camera matching.

    Returns:
        (C, D) matrix of cluster centroids, one per cluster.
    """
    centroids = np.zeros((len(clusters), embeddings.shape[1]), dtype=np.float32)
    for ci, cluster in enumerate(clusters):
        members = list(cluster)
        centroid = embeddings[members].mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 1e-8:
            centroid /= norm
        centroids[ci] = centroid
    return centroids


def _hierarchical_centroid_expansion(
    clusters: List[Set[int]],
    embeddings: np.ndarray,
    hsv_features: np.ndarray,
    camera_ids: List[str],
    class_ids: List[int],
    start_times: List[float],
    end_times: List[float],
    st_validator: SpatioTemporalValidator,
    n: int,
    stage_cfg: DictConfig,
    combined_sim: Dict[Tuple[int, int], float],
    hierarch_cfg: dict,
) -> List[Set[int]]:
    """Hierarchical multi-pass association (AIC21/22 SOTA technique).

    After initial graph clustering:
    1. **Centroid expansion**: Compute cluster centroids (averaged embeddings),
       then absorb orphans whose centroid similarity exceeds a lower threshold.
       Centroids are more robust because noise averages out across viewpoints.
    2. **Cluster-to-cluster merging**: Try to merge small clusters together
       using centroid-to-centroid cosine similarity.
    3. **Orphan-to-orphan recovery**: Link remaining orphans at the loosest
       threshold with strict constraints.

    Each pass resolves same-camera conflicts before proceeding to the next.
    """
    centroid_threshold = float(hierarch_cfg.get("centroid_threshold", 0.35))
    merge_threshold = float(hierarch_cfg.get("merge_threshold", 0.35))
    orphan_threshold = float(hierarch_cfg.get("orphan_threshold", 0.30))
    max_merge_size = int(hierarch_cfg.get("max_merge_size", 12))
    hsv_gate = float(hierarch_cfg.get("hsv_gate", 0.2))

    total_absorbed = 0
    total_merged = 0

    # ── Pass 1: Centroid-based orphan absorption ────────────────────────
    for round_idx in range(1):  # Single round — prevents centroid drift
        multi_clusters = [c for c in clusters if len(c) > 1]
        orphan_sets = [c for c in clusters if len(c) == 1]
        orphan_indices = [list(c)[0] for c in orphan_sets]

        if not orphan_indices or not multi_clusters:
            break

        # Compute cluster centroids
        centroids = _compute_cluster_centroids(multi_clusters, embeddings)

        # Compute cluster metadata
        cluster_meta = []
        for ci, cluster in enumerate(multi_clusters):
            members = list(cluster)
            cls_counts: Dict[int, int] = {}
            for m in members:
                cls_counts[class_ids[m]] = cls_counts.get(class_ids[m], 0) + 1
            majority_class = max(cls_counts, key=cls_counts.get)
            cams = {camera_ids[m] for m in members}
            scenes = {_extract_scene(camera_ids[m]) for m in members} - {""}
            cluster_meta.append({
                "class": majority_class,
                "cameras": cams,
                "scenes": scenes,
                "members": members,
            })

        # For each orphan, find best matching cluster via centroid similarity.
        # Use centroid-only (no max-member shortcut) — centroids average out
        # viewpoint noise, while max-member allows single spurious matches.
        orphan_embs = embeddings[orphan_indices]  # (O, D)
        # Centroid similarities: (O, C)
        centroid_sims = orphan_embs @ centroids.T

        merged_this_round = 0
        remaining_orphans = []

        for oi, orphan_idx in enumerate(orphan_indices):
            best_ci = -1
            best_sim = -1.0

            # Sort cluster candidates by centroid similarity (descending)
            order = np.argsort(-centroid_sims[oi])
            for rank, ci in enumerate(order[:10]):
                c_sim = float(centroid_sims[oi, ci])
                # Centroid-only gating — no max-member shortcut
                if c_sim < centroid_threshold:
                    break
                sim = c_sim

                meta = cluster_meta[ci]

                # Same class check
                if class_ids[orphan_idx] != meta["class"]:
                    continue

                # Scene blocking
                orphan_scene = _extract_scene(camera_ids[orphan_idx])
                if orphan_scene and meta["scenes"] and orphan_scene not in meta["scenes"]:
                    continue

                # Size cap
                if len(multi_clusters[ci]) >= max_merge_size:
                    continue

                # Same-camera temporal overlap check
                if camera_ids[orphan_idx] in meta["cameras"]:
                    conflict = False
                    for m in meta["members"]:
                        if camera_ids[m] == camera_ids[orphan_idx]:
                            if (start_times[orphan_idx] <= end_times[m] and
                                    start_times[m] <= end_times[orphan_idx]):
                                conflict = True
                                break
                    if conflict:
                        continue

                # Spatio-temporal check with at least one cluster member
                st_ok = any(
                    st_validator.is_valid_transition(
                        camera_ids[m], camera_ids[orphan_idx],
                        end_times[m], start_times[orphan_idx],
                    ) or st_validator.is_valid_transition(
                        camera_ids[orphan_idx], camera_ids[m],
                        end_times[orphan_idx], start_times[m],
                    )
                    for m in meta["members"]
                    if camera_ids[m] != camera_ids[orphan_idx]
                )
                if not st_ok:
                    continue

                # HSV consistency: orphan's HSV must be plausible
                member_hsv = hsv_features[meta["members"]]
                hsv_sims = member_hsv @ hsv_features[orphan_idx]
                if hsv_gate > 0 and float(hsv_sims.max()) < hsv_gate:
                    continue

                best_ci = ci
                best_sim = sim
                break

            if best_ci >= 0:
                multi_clusters[best_ci].add(orphan_idx)
                meta = cluster_meta[best_ci]
                meta["cameras"].add(camera_ids[orphan_idx])
                orphan_scene = _extract_scene(camera_ids[orphan_idx])
                if orphan_scene:
                    meta["scenes"].add(orphan_scene)
                meta["members"].append(orphan_idx)
                merged_this_round += 1
            else:
                remaining_orphans.append({orphan_idx})

        total_absorbed += merged_this_round
        clusters = multi_clusters + remaining_orphans

        # Resolve same-camera conflicts after each round
        clusters = _resolve_same_camera_conflicts(
            clusters, camera_ids, start_times, end_times, combined_sim,
        )

        logger.info(
            f"Hierarchical pass 1 round {round_idx + 1}: "
            f"absorbed {merged_this_round} orphans via centroids "
            f"(threshold={centroid_threshold:.2f})"
        )
        if merged_this_round == 0:
            break

    # ── Pass 2: Cluster-to-cluster merging via centroids ───────────────
    multi_clusters = [c for c in clusters if len(c) > 1]
    orphan_sets = [c for c in clusters if len(c) == 1]

    if len(multi_clusters) >= 2:
        nc = len(multi_clusters)
        centroids = _compute_cluster_centroids(multi_clusters, embeddings)

        # Cluster metadata
        c_classes = []
        c_scenes = []
        c_cameras = []
        for cluster in multi_clusters:
            members = list(cluster)
            cls_counts: Dict[int, int] = {}
            for m in members:
                cls_counts[class_ids[m]] = cls_counts.get(class_ids[m], 0) + 1
            c_classes.append(max(cls_counts, key=cls_counts.get))
            c_scenes.append({_extract_scene(camera_ids[m]) for m in members} - {""})
            c_cameras.append({camera_ids[m] for m in members})

        # Centroid-to-centroid similarity matrix
        cc_sim = centroids @ centroids.T

        # Build merge candidate pairs using centroid-centroid similarity only.
        # No max-member shortcut — centroids are inherently robust.
        merge_pairs = []
        for ci in range(nc):
            for cj in range(ci + 1, nc):
                sim = float(cc_sim[ci, cj])
                if sim < merge_threshold:
                    continue
                # Same class
                if c_classes[ci] != c_classes[cj]:
                    continue
                # Scene compatibility
                if c_scenes[ci] and c_scenes[cj]:
                    if not c_scenes[ci] & c_scenes[cj]:
                        continue
                # Size cap
                if len(multi_clusters[ci]) + len(multi_clusters[cj]) > max_merge_size:
                    continue
                # Same-camera temporal overlap: check ALL pairs would be valid
                conflict = False
                for mi in multi_clusters[ci]:
                    for mj in multi_clusters[cj]:
                        if camera_ids[mi] == camera_ids[mj]:
                            if (start_times[mi] <= end_times[mj] and
                                    start_times[mj] <= end_times[mi]):
                                conflict = True
                                break
                    if conflict:
                        break
                if conflict:
                    continue
                merge_pairs.append((ci, cj, sim))

        # Greedily merge highest-similarity pairs
        merge_pairs.sort(key=lambda x: x[2], reverse=True)
        merged_into: Dict[int, int] = {}  # ci -> canonical ci

        def _find_canonical(ci: int) -> int:
            while ci in merged_into:
                ci = merged_into[ci]
            return ci

        for ci, cj, sim in merge_pairs:
            ci_canon = _find_canonical(ci)
            cj_canon = _find_canonical(cj)
            if ci_canon == cj_canon:
                continue
            # Verify size cap after transitive merges
            merged_size = len(multi_clusters[ci_canon]) + len(multi_clusters[cj_canon])
            if merged_size > max_merge_size:
                continue
            # Re-check same-camera temporal overlap after transitive merges
            conflict = False
            for mi in multi_clusters[ci_canon]:
                for mj in multi_clusters[cj_canon]:
                    if camera_ids[mi] == camera_ids[mj]:
                        if (start_times[mi] <= end_times[mj] and
                                start_times[mj] <= end_times[mi]):
                            conflict = True
                            break
                if conflict:
                    break
            if conflict:
                continue
            # Merge cj into ci
            multi_clusters[ci_canon].update(multi_clusters[cj_canon])
            merged_into[cj_canon] = ci_canon
            total_merged += 1

        # Collect surviving clusters
        surviving = [multi_clusters[i] for i in range(nc) if i not in merged_into]
        clusters = surviving + orphan_sets

        if total_merged > 0:
            logger.info(
                f"Hierarchical pass 2: merged {total_merged} cluster pairs "
                f"(threshold={merge_threshold:.2f})"
            )

    # ── Pass 3: Orphan-to-orphan recovery at loose threshold ───────────
    if orphan_threshold > 0:
        final_multi = [c for c in clusters if len(c) > 1]
        final_orphan_indices = []
        for c in clusters:
            if len(c) == 1:
                final_orphan_indices.extend(c)

        if len(final_orphan_indices) >= 2:
            orphan_embs = embeddings[final_orphan_indices]
            pairwise_sim = orphan_embs @ orphan_embs.T

            n_orphans = len(final_orphan_indices)
            orphan_sims: Dict[Tuple[int, int], float] = {}

            for oi in range(n_orphans):
                gi = final_orphan_indices[oi]
                for oj in range(oi + 1, n_orphans):
                    gj = final_orphan_indices[oj]
                    pair_sim = float(pairwise_sim[oi, oj])
                    if pair_sim < orphan_threshold:
                        continue
                    if class_ids[gi] != class_ids[gj]:
                        continue
                    if camera_ids[gi] == camera_ids[gj]:
                        continue
                    # Scene blocking
                    scene_gi = _extract_scene(camera_ids[gi])
                    scene_gj = _extract_scene(camera_ids[gj])
                    if scene_gi and scene_gj and scene_gi != scene_gj:
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
                    orphan_sims[(gi, gj)] = pair_sim

            if orphan_sims:
                orphan_solver = GraphSolver(
                    similarity_threshold=orphan_threshold,
                    algorithm="connected_components",
                    bridge_prune_margin=float(
                        stage_cfg.get("graph", {}).get("bridge_prune_margin", 0.0)
                    ),
                    max_component_size=max_merge_size,
                )
                all_orphan_set = set(final_orphan_indices)
                orphan_clusters_raw = orphan_solver.solve(
                    orphan_sims, max(all_orphan_set) + 1,
                )

                new_orphans = []
                new_multi = []
                for cluster in orphan_clusters_raw:
                    members = cluster & all_orphan_set
                    if not members:
                        continue
                    if len(members) > 1:
                        new_multi.append(members)
                    else:
                        new_orphans.append(members)
                linked_orphans = set()
                for c in new_multi + new_orphans:
                    linked_orphans |= c
                for gi in final_orphan_indices:
                    if gi not in linked_orphans:
                        new_orphans.append({gi})

                clusters = final_multi + new_multi + new_orphans
                logger.info(
                    f"Hierarchical pass 3: orphan matching "
                    f"(threshold={orphan_threshold:.2f}): "
                    f"{len(orphan_sims)} pairs -> {len(new_multi)} new clusters"
                )
            else:
                clusters = final_multi + [{gi} for gi in final_orphan_indices]
                logger.info("Hierarchical pass 3: no valid orphan pairs")

    logger.info(
        f"Hierarchical expansion complete: "
        f"absorbed {total_absorbed} orphans, merged {total_merged} clusters, "
        f"{len(clusters)} final clusters"
    )
    return clusters


# ---------------------------------------------------------------------------
# Sub-cluster temporal splitting (AIC21 technique)
# ---------------------------------------------------------------------------

def _try_temporal_split(
    members: List[int],
    start_times: List[float],
    end_times: List[float],
    embeddings: np.ndarray,
    min_gap: float,
    split_threshold: float,
) -> List[Set[int]]:
    """Recursively split a member group at the largest temporal silence gap.

    A silence gap is a time interval [gap_start, gap_end] where every member
    that started before gap_start has also ended before gap_start (i.e., no
    member is active during the gap).  The gap is found via a start-time-sorted
    sweep: after processing member m_i, ``max_end_so_far`` is the latest end
    time of all members m_0..m_i.  If m_{i+1}.start > max_end_so_far, the
    interval [max_end_so_far, m_{i+1}.start] is a true silence gap.

    A split is only applied when:
    1. The largest silence gap ≥ min_gap seconds.
    2. The average cross-gap cosine similarity between group_a (members that
       ended before the gap) and group_b (members that start after the gap)
       is strictly less than split_threshold.  High similarity means same
       vehicle — in that case we keep the cluster intact.

    Returns a list of sub-cluster sets (possibly [set(members)] if no split).
    """
    if len(members) < 2:
        return [set(members)]

    # Sort by start time to enable the sweep
    members_sorted = sorted(members, key=lambda m: start_times[m])

    # Sweep to find the largest silence gap
    best_gap_size = 0.0
    best_gap_start = 0.0
    best_gap_end = 0.0

    max_end_so_far = end_times[members_sorted[0]]
    for m in members_sorted[1:]:
        gap_size = start_times[m] - max_end_so_far
        if gap_size > best_gap_size:
            best_gap_size = gap_size
            best_gap_start = max_end_so_far
            best_gap_end = start_times[m]
        max_end_so_far = max(max_end_so_far, end_times[m])

    if best_gap_size < min_gap:
        return [set(members)]

    # Partition members into groups relative to the gap
    group_a = [m for m in members if end_times[m] <= best_gap_start]
    group_b = [m for m in members if start_times[m] >= best_gap_end]
    # Safety: members spanning the gap (start < gap_start AND end > gap_end)
    # should not exist in a true silence gap, but keep them separately
    group_c = [m for m in members if m not in group_a and m not in group_b]

    if not group_a or not group_b:
        return [set(members)]

    # Compute average cross-gap cosine similarity
    embs_a = embeddings[group_a]   # (|A|, D)
    embs_b = embeddings[group_b]   # (|B|, D)
    avg_sim = float((embs_a @ embs_b.T).mean())

    if avg_sim >= split_threshold:
        # Cross-gap similarity is high → same vehicle, do not split
        return [set(members)]

    # Split is warranted — recurse on each group independently
    result: List[Set[int]] = []
    result.extend(_try_temporal_split(
        group_a, start_times, end_times, embeddings, min_gap, split_threshold,
    ))
    result.extend(_try_temporal_split(
        group_b, start_times, end_times, embeddings, min_gap, split_threshold,
    ))
    if group_c:
        result.append(set(group_c))

    return result


def _temporal_split_clusters(
    clusters: List[Set[int]],
    start_times: List[float],
    end_times: List[float],
    embeddings: np.ndarray,
    min_gap: float = 60.0,
    split_threshold: float = 0.50,
) -> List[Set[int]]:
    """Apply sub-cluster temporal splitting to all clusters.

    For each cluster, attempts to split at temporal silence gaps using
    `_try_temporal_split`.  Single-member clusters are passed through unchanged.

    Args:
        clusters: List of cluster sets.
        start_times: Tracklet start times.
        end_times: Tracklet end times.
        embeddings: L2-normalised embedding matrix (N, D).
        min_gap: Minimum silence duration in seconds to trigger a split check.
        split_threshold: Maximum average cross-gap cosine similarity to allow a
            split.  Pairs above this are assumed to be the same vehicle.

    Returns:
        Updated cluster list (may contain more clusters than input).
    """
    new_clusters: List[Set[int]] = []
    for cluster in clusters:
        if len(cluster) < 2:
            new_clusters.append(cluster)
            continue
        sub_clusters = _try_temporal_split(
            list(cluster), start_times, end_times, embeddings,
            min_gap, split_threshold,
        )
        new_clusters.extend(sub_clusters)
    return new_clusters


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
        gallery_hsv_gate = float((stage_cfg.get("gallery_expansion", {}) or {}).get("hsv_gate", 0.3))

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
        cluster_scenes: List[Set[str]] = []
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
            cluster_scenes.append({_extract_scene(camera_ids[m]) for m in members} - {""})
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
                # Scene blocking: orphan must belong to same scene as cluster
                orphan_scene = _extract_scene(camera_ids[orphan])
                if orphan_scene and cluster_scenes[ci] and orphan_scene not in cluster_scenes[ci]:
                    continue
                # HSV consistency gate: reject if color is too different
                members = list(multi_clusters[ci])
                hsv_sims = hsv_features[members] @ orphan_hsv
                if gallery_hsv_gate > 0 and float(hsv_sims.max()) < gallery_hsv_gate:
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
                orphan_scene_val = _extract_scene(camera_ids[orphan])
                if orphan_scene_val:
                    cluster_scenes[ci].add(orphan_scene_val)
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
                    # Use combined_sim if available (includes ST/HSV weighting),
                    # fall back to raw cosine similarity
                    raw_cos = float(pairwise_sim[oi, oj])
                    if combined_sim:
                        cs_val = combined_sim.get((gi, gj)) or combined_sim.get((gj, gi))
                        pair_sim = max(raw_cos, cs_val) if cs_val is not None else raw_cos
                    else:
                        pair_sim = raw_cos
                    if pair_sim < orphan_match_threshold:
                        continue
                    if class_ids[gi] != class_ids[gj]:
                        continue
                    if camera_ids[gi] == camera_ids[gj]:
                        continue
                    # Scene blocking: different scenes should never be linked
                    scene_gi = _extract_scene(camera_ids[gi])
                    scene_gj = _extract_scene(camera_ids[gj])
                    if scene_gi and scene_gj and scene_gi != scene_gj:
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
                    orphan_sims[(gi, gj)] = pair_sim

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
