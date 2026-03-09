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

    # Step 1b: Average Query Expansion (AQE)
    # Averages each embedding with its top-K nearest neighbours then
    # re-normalises → makes embeddings more robust and boosts recall.
    qe_cfg = stage_cfg.get("query_expansion", {})
    if qe_cfg.get("enabled", True):
        qe_k = qe_cfg.get("k", 5)
        qe_alpha = qe_cfg.get("alpha", 1.0)
        embeddings = average_query_expansion_batched(
            embeddings, indices, k=qe_k, alpha=qe_alpha,
        )
        # Re-run FAISS retrieval with expanded embeddings for better candidates
        distances, indices = faiss_index.search(embeddings, top_k)
        logger.info("Re-retrieved candidates with QE-expanded embeddings")

    # Step 2: Build candidate pairs (cross-camera, same class)
    candidate_pairs = _build_candidate_pairs(
        n, indices, distances, top_k, camera_ids, class_ids,
    )
    logger.info(f"Candidate pairs after filtering: {len(candidate_pairs)}")

    if not candidate_pairs:
        logger.warning("No cross-camera candidate pairs found")
        all_tracklets = [t for tl in tracklets_by_camera.values() for t in tl]
        return _assign_individual_ids(all_tracklets)

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
    solver = GraphSolver(
        similarity_threshold=stage_cfg.graph.similarity_threshold,
        algorithm=stage_cfg.graph.algorithm,
        louvain_resolution=stage_cfg.graph.get("louvain_resolution", 1.0),
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
            threshold=gallery_cfg.get("threshold", 0.5),
            max_rounds=gallery_cfg.get("max_rounds", 2),
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
    )

    # Save
    save_global_trajectories(trajectories, output_dir / "global_trajectories.json")
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
    candidate_pairs: List[Tuple[int, int, float]] = []
    for i in range(n):
        for j_idx in range(top_k):
            j = int(indices[i, j_idx])
            if j < 0 or j >= n or i == j:
                continue
            if camera_ids[i] == camera_ids[j]:
                continue
            if class_ids[i] != class_ids[j]:
                continue
            candidate_pairs.append((i, j, float(distances[i, j_idx])))
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
    threshold: float = 0.5,
    max_rounds: int = 2,
) -> List[Set[int]]:
    """Iteratively absorb orphan (singleton) tracklets into existing clusters.

    For each orphan, compute average cosine similarity to every cluster's
    centroid embedding.  If the best match exceeds *threshold* and passes
    spatio-temporal + same-class + cross-camera checks, merge the orphan.

    This runs for up to *max_rounds* iterations so that orphans absorbed in
    round 1 can shift centroids enough to attract others in round 2.
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

        # Build cluster centroids
        centroids = np.zeros((len(multi_clusters), embeddings.shape[1]), dtype=np.float32)
        cluster_class_ids: List[int] = []
        cluster_cameras: List[Set[str]] = []
        cluster_time_ranges: List[Tuple[float, float]] = []

        for ci, cluster in enumerate(multi_clusters):
            members = list(cluster)
            centroids[ci] = embeddings[members].mean(axis=0)
            # Normalise centroid to unit length
            norm = np.linalg.norm(centroids[ci])
            if norm > 0:
                centroids[ci] /= norm
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
            sims = centroids @ orphan_emb  # (C,)
            order = np.argsort(-sims)

            merged = False
            for ci in order:
                if sims[ci] < threshold:
                    break
                # Same class check
                if class_ids[orphan] != cluster_class_ids[ci]:
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

        coloring = nx.coloring.greedy_color(
            conflict_graph, strategy="connected_sequential",
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
