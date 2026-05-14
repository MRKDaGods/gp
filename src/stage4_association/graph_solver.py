"""Graph-based solver for cross-camera identity clustering."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
from loguru import logger
from scipy.optimize import linear_sum_assignment


class GraphSolver:
    """Builds a similarity graph and finds identity clusters.

    Nodes represent tracklets, edges represent plausible same-identity links
    weighted by combined similarity. Connected components or community detection
    identifies global identities.

    Bridge pruning (optional): After building the graph, identifies bridge edges
    (the sole connection between two sub-graphs).  If a bridge's weight is below
    ``bridge_prune_threshold``, it is removed.  This prevents false transitive
    merges (A~B~C where the A-B link is weak but B-C is strong).
    """

    def __init__(
        self,
        similarity_threshold: float = 0.5,
        algorithm: str = "connected_components",
        louvain_resolution: float = 1.0,
        louvain_seed: int = 42,
        bridge_prune_margin: float = 0.0,
        max_component_size: int = 0,
        merge_verify_threshold: Optional[float] = None,
    ):
        self.similarity_threshold = similarity_threshold
        self.algorithm = algorithm
        self.louvain_resolution = louvain_resolution
        self.louvain_seed = louvain_seed
        self.bridge_prune_margin = bridge_prune_margin
        self.max_component_size = max_component_size
        self.merge_verify_threshold = merge_verify_threshold

    def solve(
        self,
        similarities: Dict[Tuple[int, int], float],
        num_nodes: int,
        camera_ids: Optional[List[str]] = None,
        start_times: Optional[List[float]] = None,
        end_times: Optional[List[float]] = None,
    ) -> List[Set[int]]:
        """Build graph and find clusters.

        Args:
            similarities: Dict[(i, j)] -> similarity score for each candidate pair.
            num_nodes: Total number of tracklets (nodes).

        Returns:
            List of sets, each set containing tracklet indices belonging
            to the same identity.
        """
        # Build graph
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))

        edges_added = 0
        for (i, j), sim in similarities.items():
            if sim >= self.similarity_threshold:
                G.add_edge(i, j, weight=sim)
                edges_added += 1

        logger.info(
            f"Similarity graph: {num_nodes} nodes, {edges_added} edges "
            f"(threshold={self.similarity_threshold})"
        )

        # Bridge pruning: remove weak bridges that are the sole connection
        # between two sub-graphs.  These are the most dangerous edges for
        # false transitive merges.
        # Recompute bridges after each removal since removing one bridge
        # can expose new bridges in the remaining graph.
        if self.bridge_prune_margin > 0:
            bridge_threshold = self.similarity_threshold + self.bridge_prune_margin
            pruned = 0
            total_bridges = 0
            while True:
                bridges = [
                    (u, v) for u, v in nx.bridges(G)
                    if G[u][v].get("weight", 0.0) < bridge_threshold
                ]
                if not bridges:
                    break
                total_bridges += len(bridges)
                for u, v in bridges:
                    if G.has_edge(u, v):
                        G.remove_edge(u, v)
                        pruned += 1
            if pruned > 0:
                logger.info(
                    f"Bridge pruning: removed {pruned} weak bridges "
                    f"(threshold={bridge_threshold:.3f})"
                )

        # Find clusters
        if self.algorithm == "connected_components":
            clusters = list(nx.connected_components(G))
        elif self.algorithm == "conflict_free_cc":
            clusters = self._conflict_free_greedy(
                similarities, num_nodes, camera_ids, start_times, end_times,
            )
        elif self.algorithm == "network_flow":
            clusters = self._network_flow_solver(
                similarities, num_nodes, camera_ids, start_times, end_times,
            )
        elif self.algorithm == "community_detection":
            clusters = self._community_detection(G, self.louvain_resolution, self.louvain_seed)
        elif self.algorithm == "agglomerative":
            clusters = self._agglomerative_clustering(G, num_nodes, self.similarity_threshold)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        # Split oversized components by iteratively removing weakest edges.
        # Remove only ONE weakest edge per iteration then recheck, since
        # removing an edge can split a component into valid-sized pieces.
        if self.max_component_size > 0:
            split_count = 0
            final_clusters = []
            for cluster in clusters:
                if len(cluster) <= self.max_component_size:
                    final_clusters.append(cluster)
                    continue
                # Extract subgraph and iteratively remove weakest edge
                sub = G.subgraph(cluster).copy()
                while sub.number_of_edges() > 0:
                    components = list(nx.connected_components(sub))
                    oversized = [c for c in components if len(c) > self.max_component_size]
                    if not oversized:
                        break
                    # Pick the largest oversized component and remove its weakest edge
                    largest = max(oversized, key=len)
                    comp_sub = sub.subgraph(largest)
                    edges = list(comp_sub.edges(data=True))
                    if not edges:
                        break
                    weakest = min(
                        edges,
                        key=lambda e: e[2].get("weight", 0.0),
                    )
                    sub.remove_edge(weakest[0], weakest[1])
                    split_count += 1
                final_clusters.extend(list(nx.connected_components(sub)))
            if split_count > 0:
                logger.info(
                    f"Component size cap ({self.max_component_size}): "
                    f"split {split_count} edges, {len(clusters)} -> {len(final_clusters)} clusters"
                )
            clusters = final_clusters

        # Filter out single-node clusters (keep them but they represent
        # tracklets seen on only one camera)
        logger.info(
            f"Found {len(clusters)} clusters: "
            f"{sum(1 for c in clusters if len(c) > 1)} multi-tracklet, "
            f"{sum(1 for c in clusters if len(c) == 1)} singleton"
        )

        return clusters

    @staticmethod
    def _community_detection(G: nx.Graph, resolution: float = 1.0, seed: int = 42) -> List[Set[int]]:
        """Run Louvain community detection on the similarity graph."""
        from networkx.algorithms.community import louvain_communities

        communities = louvain_communities(G, weight="weight", resolution=resolution, seed=seed)
        return [set(c) for c in communities]

    @staticmethod
    def _agglomerative_clustering(
        G: nx.Graph, num_nodes: int, sim_threshold: float,
    ) -> List[Set[int]]:
        """Agglomerative clustering with complete linkage on precomputed distance.

        This is the approach used by the AIC21 1st-place MTMC solution.
        Complete linkage means two clusters merge only when ALL inter-cluster
        distances are below the threshold — conservative and avoids runaway
        transitive merges.

        For nodes not connected by edges, distance is set to 1.0 (maximum
        dissimilarity for cosine-based features).
        """
        import numpy as np

        # Get only nodes that participate in at least one edge
        active_nodes = sorted(set(u for u, v in G.edges()) | set(v for u, v in G.edges()))
        if len(active_nodes) < 2:
            # No edges or only one node with edges — return connected components
            return list(nx.connected_components(G))

        # Build node index mapping
        node_to_idx = {n: i for i, n in enumerate(active_nodes)}
        m = len(active_nodes)

        # Build distance matrix: distance = 1 - similarity
        dist_matrix = np.ones((m, m), dtype=np.float32)
        np.fill_diagonal(dist_matrix, 0.0)
        for u, v, data in G.edges(data=True):
            if u in node_to_idx and v in node_to_idx:
                d = 1.0 - data.get("weight", 0.0)
                dist_matrix[node_to_idx[u], node_to_idx[v]] = d
                dist_matrix[node_to_idx[v], node_to_idx[u]] = d

        from sklearn.cluster import AgglomerativeClustering

        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1.0 - sim_threshold,
            metric="precomputed",
            linkage="complete",
        )
        labels = clustering.fit_predict(dist_matrix)

        # Group by label
        clusters_dict: dict = {}
        for i, label in enumerate(labels):
            clusters_dict.setdefault(label, set()).add(active_nodes[i])

        clusters = list(clusters_dict.values())

        # Add isolated nodes (no edges) as singletons
        active_set = set(active_nodes)
        for node in range(num_nodes):
            if node not in active_set:
                clusters.append({node})

        logger.info(
            f"Agglomerative clustering (complete linkage): "
            f"{m} active nodes → {len(clusters_dict)} clusters "
            f"+ {num_nodes - m} singletons"
        )
        return clusters

    def _conflict_free_greedy(
        self,
        similarities: Dict[Tuple[int, int], float],
        num_nodes: int,
        camera_ids: Optional[List[str]],
        start_times: Optional[List[float]],
        end_times: Optional[List[float]],
    ) -> List[Set[int]]:
        """Conflict-free greedy matching: add edges in descending similarity
        order, skipping any edge that would create a same-camera temporal
        overlap within the resulting cluster.

        This is fundamentally better than plain connected_components +
        post-hoc conflict resolution because it prevents false transitive
        chains from forming in the first place.  When a bad merge is blocked,
        the tracklet stays unmatched and gets a second chance via gallery
        expansion.

        Falls back to connected_components when temporal info is unavailable.
        """
        if camera_ids is None or start_times is None or end_times is None:
            logger.warning("conflict_free_cc: missing temporal info, falling back to connected_components")
            G = nx.Graph()
            G.add_nodes_from(range(num_nodes))
            for (i, j), sim in similarities.items():
                if sim >= self.similarity_threshold:
                    G.add_edge(i, j, weight=sim)
            return list(nx.connected_components(G))

        # Sort edges above threshold in descending similarity
        edges = [
            (i, j, sim) for (i, j), sim in similarities.items()
            if sim >= self.similarity_threshold
        ]
        edges.sort(key=lambda e: e[2], reverse=True)

        # Union-Find with cluster membership tracking
        parent = list(range(num_nodes))
        rank = [0] * num_nodes
        # Track all members of each cluster root
        cluster_members: Dict[int, Set[int]] = {i: {i} for i in range(num_nodes)}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]  # path compression
                x = parent[x]
            return x

        def _has_conflict(members_a: Set[int], members_b: Set[int]) -> bool:
            """Check if merging two clusters would create a same-camera
            temporal overlap."""
            for a in members_a:
                cam_a = camera_ids[a]
                for b in members_b:
                    if camera_ids[b] != cam_a:
                        continue
                    # Same camera — check temporal overlap
                    if start_times[a] <= end_times[b] and start_times[b] <= end_times[a]:
                        return True
            return False

        edges_added = 0
        edges_blocked = 0

        for i, j, sim in edges:
            ri, rj = find(i), find(j)
            if ri == rj:
                continue  # already in same cluster

            # Check for conflicts before merging
            if _has_conflict(cluster_members[ri], cluster_members[rj]):
                edges_blocked += 1
                continue

            # Merge: union by rank
            if rank[ri] < rank[rj]:
                ri, rj = rj, ri
            parent[rj] = ri
            if rank[ri] == rank[rj]:
                rank[ri] += 1

            # Merge member sets
            cluster_members[ri] = cluster_members[ri] | cluster_members[rj]
            del cluster_members[rj]
            edges_added += 1

        # Collect final clusters
        clusters_dict: Dict[int, Set[int]] = {}
        for node in range(num_nodes):
            root = find(node)
            clusters_dict.setdefault(root, set()).add(node)
        clusters = list(clusters_dict.values())

        logger.info(
            f"Conflict-free greedy: {edges_added} edges added, "
            f"{edges_blocked} blocked by conflicts, "
            f"{len(clusters)} clusters"
        )
        return clusters

    @staticmethod
    def _pair_key(i: int, j: int) -> Tuple[int, int]:
        return (i, j) if i < j else (j, i)

    @staticmethod
    def _clusters_have_temporal_conflict(
        members_a: Set[int],
        members_b: Set[int],
        camera_ids: List[str],
        start_times: Optional[List[float]],
        end_times: Optional[List[float]],
    ) -> bool:
        if start_times is None or end_times is None:
            return False

        for a in members_a:
            cam_a = camera_ids[a]
            for b in members_b:
                if camera_ids[b] != cam_a:
                    continue
                if start_times[a] <= end_times[b] and start_times[b] <= end_times[a]:
                    return True
        return False

    def _clusters_pass_merge_verification(
        self,
        members_a: Set[int],
        members_b: Set[int],
        similarities: Dict[Tuple[int, int], float],
        camera_ids: List[str],
    ) -> bool:
        verify_threshold = self.merge_verify_threshold
        if verify_threshold is None:
            verify_threshold = self.similarity_threshold

        cameras_a = {camera_ids[node] for node in members_a}
        cameras_b = {camera_ids[node] for node in members_b}

        for cam_a in cameras_a:
            for cam_b in cameras_b:
                if cam_a == cam_b:
                    continue

                supported = False
                for node_a in members_a:
                    if camera_ids[node_a] != cam_a:
                        continue
                    for node_b in members_b:
                        if camera_ids[node_b] != cam_b:
                            continue
                        sim = similarities.get(self._pair_key(node_a, node_b), float("-inf"))
                        if sim >= verify_threshold:
                            supported = True
                            break
                    if supported:
                        break

                if not supported:
                    return False

        return True

    def _network_flow_solver(
        self,
        similarities: Dict[Tuple[int, int], float],
        num_nodes: int,
        camera_ids: Optional[List[str]],
        start_times: Optional[List[float]],
        end_times: Optional[List[float]],
    ) -> List[Set[int]]:
        """Solve association with pairwise Hungarian matching plus merge verification.

        The solver first computes globally optimal 1:1 matches for each camera pair
        using linear assignment. It then merges those matches only when the merge
        would remain temporally valid and every newly connected camera pair has a
        direct edge above the verification threshold. This blocks A-B-C transitive
        chains when the implied A-C link is weak or absent.
        """
        if camera_ids is None:
            logger.warning("network_flow: missing camera IDs, falling back to connected_components")
            G = nx.Graph()
            G.add_nodes_from(range(num_nodes))
            for (i, j), sim in similarities.items():
                if sim >= self.similarity_threshold:
                    G.add_edge(i, j, weight=sim)
            return list(nx.connected_components(G))

        node_cameras: Dict[str, List[int]] = defaultdict(list)
        for idx, camera_id in enumerate(camera_ids):
            node_cameras[camera_id].append(idx)

        similarity_lookup: Dict[Tuple[int, int], float] = {}
        edges_by_camera_pair: Dict[Tuple[str, str], Dict[Tuple[int, int], float]] = defaultdict(dict)
        for (i, j), sim in similarities.items():
            key = self._pair_key(i, j)
            similarity_lookup[key] = sim
            if sim < self.similarity_threshold:
                continue
            cam_i = camera_ids[i]
            cam_j = camera_ids[j]
            if cam_i == cam_j:
                continue
            cam_pair = tuple(sorted((cam_i, cam_j)))
            left, right = key
            if camera_ids[left] != cam_pair[0]:
                left, right = right, left
            edges_by_camera_pair[cam_pair][(left, right)] = sim

        accepted_matches: List[Tuple[int, int, float]] = []
        total_pair_matches = 0

        for (cam_a, cam_b), pair_sims in sorted(edges_by_camera_pair.items()):
            left_nodes = sorted(node_cameras[cam_a])
            right_nodes = sorted(node_cameras[cam_b])
            if not left_nodes or not right_nodes:
                continue

            left_index = {node: idx for idx, node in enumerate(left_nodes)}
            right_index = {node: idx for idx, node in enumerate(right_nodes)}
            matrix_size = max(len(left_nodes), len(right_nodes))
            profit = np.zeros((matrix_size, matrix_size), dtype=np.float32)

            for (left_node, right_node), sim in pair_sims.items():
                row = left_index[left_node]
                col = right_index[right_node]
                profit[row, col] = max(sim - self.similarity_threshold, 0.0) + 1e-6

            row_ind, col_ind = linear_sum_assignment(-profit)

            matched_here = 0
            for row, col in zip(row_ind.tolist(), col_ind.tolist()):
                if row >= len(left_nodes) or col >= len(right_nodes):
                    continue
                left_node = left_nodes[row]
                right_node = right_nodes[col]
                sim = pair_sims.get((left_node, right_node))
                if sim is None or sim < self.similarity_threshold:
                    continue
                accepted_matches.append((left_node, right_node, sim))
                matched_here += 1

            total_pair_matches += matched_here
            logger.info(
                f"Network flow pair {cam_a}↔{cam_b}: {matched_here} matches "
                f"above threshold {self.similarity_threshold:.3f}"
            )

        parent = list(range(num_nodes))
        rank = [0] * num_nodes
        cluster_members: Dict[int, Set[int]] = {idx: {idx} for idx in range(num_nodes)}

        def find(node: int) -> int:
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        def union(root_a: int, root_b: int) -> int:
            if rank[root_a] < rank[root_b]:
                root_a, root_b = root_b, root_a
            parent[root_b] = root_a
            if rank[root_a] == rank[root_b]:
                rank[root_a] += 1
            cluster_members[root_a] |= cluster_members[root_b]
            del cluster_members[root_b]
            return root_a

        merges_added = 0
        blocked_temporal = 0
        blocked_verify = 0
        for i, j, sim in sorted(accepted_matches, key=lambda item: item[2], reverse=True):
            root_i = find(i)
            root_j = find(j)
            if root_i == root_j:
                continue

            members_i = cluster_members[root_i]
            members_j = cluster_members[root_j]
            if self._clusters_have_temporal_conflict(
                members_i, members_j, camera_ids, start_times, end_times,
            ):
                blocked_temporal += 1
                continue

            if not self._clusters_pass_merge_verification(
                members_i, members_j, similarity_lookup, camera_ids,
            ):
                blocked_verify += 1
                continue

            union(root_i, root_j)
            merges_added += 1

        clusters_dict: Dict[int, Set[int]] = {}
        for node in range(num_nodes):
            clusters_dict.setdefault(find(node), set()).add(node)

        clusters = list(clusters_dict.values())
        logger.info(
            f"Network flow: {total_pair_matches} pairwise assignments, "
            f"{merges_added} merges accepted, {blocked_temporal} blocked by conflicts, "
            f"{blocked_verify} blocked by merge verification"
        )
        return clusters

    def get_graph_stats(
        self,
        similarities: Dict[Tuple[int, int], float],
        num_nodes: int,
    ) -> Dict:
        """Compute statistics about the similarity graph for analysis."""
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))

        for (i, j), sim in similarities.items():
            if sim >= self.similarity_threshold:
                G.add_edge(i, j, weight=sim)

        components = list(nx.connected_components(G))
        component_sizes = [len(c) for c in components]

        edge_weights = [d["weight"] for _, _, d in G.edges(data=True)]

        return {
            "num_nodes": num_nodes,
            "num_edges": G.number_of_edges(),
            "num_components": len(components),
            "max_component_size": max(component_sizes) if component_sizes else 0,
            "mean_component_size": sum(component_sizes) / len(component_sizes) if component_sizes else 0,
            "mean_edge_weight": sum(edge_weights) / len(edge_weights) if edge_weights else 0,
            "density": nx.density(G),
        }
