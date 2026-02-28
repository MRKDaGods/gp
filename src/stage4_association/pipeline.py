"""Stage 4 — Multi-Camera Association pipeline.

Performs cross-camera tracklet association using appearance, color, and
spatio-temporal cues. Produces global trajectories with unified IDs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from loguru import logger
from omegaconf import DictConfig

from src.core.data_models import GlobalTrajectory, Tracklet, TrackletFeatures
from src.core.io_utils import save_global_trajectories
from src.stage3_indexing.faiss_index import FAISSIndex
from src.stage3_indexing.metadata_store import MetadataStore
from src.stage4_association.global_trajectories import merge_tracklets_to_trajectories
from src.stage4_association.graph_solver import GraphSolver
from src.stage4_association.reranking import k_reciprocal_rerank
from src.stage4_association.similarity import compute_combined_similarity
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

    # Get temporal info from metadata store
    start_times = []
    end_times = []
    for i in range(n):
        meta = metadata_store.get_tracklet(i)
        if meta:
            start_times.append(meta["start_time"])
            end_times.append(meta["end_time"])
        else:
            start_times.append(0.0)
            end_times.append(0.0)

    # Step 1: FAISS top-K retrieval
    top_k = stage_cfg.top_k
    distances, indices = faiss_index.search(embeddings, top_k)
    logger.info(f"FAISS retrieval: top-{top_k} candidates per tracklet")

    # Step 2: Build candidate pairs (cross-camera, same class)
    candidate_pairs = []
    for i in range(n):
        for j_idx in range(top_k):
            j = int(indices[i, j_idx])
            if j < 0 or j >= n:
                continue
            if i == j:
                continue
            # Cross-camera constraint
            if camera_ids[i] == camera_ids[j]:
                continue
            # Same class constraint
            if class_ids[i] != class_ids[j]:
                continue
            candidate_pairs.append((i, j, float(distances[i, j_idx])))

    logger.info(f"Candidate pairs after filtering: {len(candidate_pairs)}")

    if not candidate_pairs:
        logger.warning("No cross-camera candidate pairs found")
        all_tracklets = [t for tl in tracklets_by_camera.values() for t in tl]
        return _assign_individual_ids(all_tracklets)

    # Step 3: k-reciprocal re-ranking (optional)
    if stage_cfg.reranking.enabled:
        logger.info("Applying k-reciprocal re-ranking...")
        appearance_sim = k_reciprocal_rerank(
            embeddings=embeddings,
            candidate_pairs=candidate_pairs,
            k1=stage_cfg.reranking.k1,
            k2=stage_cfg.reranking.k2,
            lambda_value=stage_cfg.reranking.lambda_value,
        )
    else:
        appearance_sim = {(i, j): sim for i, j, sim in candidate_pairs}

    # Step 4: Spatio-temporal validation
    st_validator = SpatioTemporalValidator(
        min_time_gap=stage_cfg.spatiotemporal.min_time_gap,
        max_time_gap=stage_cfg.spatiotemporal.max_time_gap,
        camera_transitions=stage_cfg.spatiotemporal.get("camera_transitions"),
    )

    # Step 5: Compute combined similarity
    combined_sim = compute_combined_similarity(
        appearance_sim=appearance_sim,
        hsv_features=hsv_features,
        start_times=start_times,
        end_times=end_times,
        camera_ids=camera_ids,
        st_validator=st_validator,
        weights=stage_cfg.weights,
    )

    logger.info(f"Combined similarity pairs: {len(combined_sim)}")

    # Step 6: Build graph and solve
    solver = GraphSolver(
        similarity_threshold=stage_cfg.graph.similarity_threshold,
        algorithm=stage_cfg.graph.algorithm,
    )

    clusters = solver.solve(combined_sim, n)
    logger.info(f"Graph solver found {len(clusters)} identity clusters")

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
    )

    # Save
    save_global_trajectories(trajectories, output_dir / "global_trajectories.json")
    logger.info(
        f"Stage 4 complete: {len(trajectories)} global trajectories, "
        f"covering {sum(len(t.tracklets) for t in trajectories)} tracklets"
    )

    return trajectories


def _assign_individual_ids(tracklets: List[Tracklet]) -> List[GlobalTrajectory]:
    """Fallback: assign each tracklet its own global ID."""
    return [
        GlobalTrajectory(global_id=i, tracklets=[t])
        for i, t in enumerate(tracklets)
    ]
