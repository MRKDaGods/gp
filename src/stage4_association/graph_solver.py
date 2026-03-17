"""Graph-based solver for cross-camera identity clustering."""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

import networkx as nx
from loguru import logger


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
    ):
        self.similarity_threshold = similarity_threshold
        self.algorithm = algorithm
        self.louvain_resolution = louvain_resolution
        self.louvain_seed = louvain_seed
        self.bridge_prune_margin = bridge_prune_margin
        self.max_component_size = max_component_size

    def solve(
        self,
        similarities: Dict[Tuple[int, int], float],
        num_nodes: int,
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
