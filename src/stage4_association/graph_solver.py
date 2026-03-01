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
    """

    def __init__(
        self,
        similarity_threshold: float = 0.5,
        algorithm: str = "connected_components",
        louvain_resolution: float = 1.0,
    ):
        self.similarity_threshold = similarity_threshold
        self.algorithm = algorithm
        self.louvain_resolution = louvain_resolution

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

        # Find clusters
        if self.algorithm == "connected_components":
            clusters = list(nx.connected_components(G))
        elif self.algorithm == "community_detection":
            clusters = self._community_detection(G, self.louvain_resolution)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        # Filter out single-node clusters (keep them but they represent
        # tracklets seen on only one camera)
        logger.info(
            f"Found {len(clusters)} clusters: "
            f"{sum(1 for c in clusters if len(c) > 1)} multi-tracklet, "
            f"{sum(1 for c in clusters if len(c) == 1)} singleton"
        )

        return clusters

    @staticmethod
    def _community_detection(G: nx.Graph, resolution: float = 1.0) -> List[Set[int]]:
        """Run Louvain community detection on the similarity graph."""
        from networkx.algorithms.community import louvain_communities

        communities = louvain_communities(G, weight="weight", resolution=resolution)
        return [set(c) for c in communities]

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
