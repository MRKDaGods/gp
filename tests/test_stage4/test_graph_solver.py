"""Tests for graph solver."""

from src.stage4_association.graph_solver import GraphSolver


def test_connected_components_basic():
    solver = GraphSolver(similarity_threshold=0.5, algorithm="connected_components")
    similarities = {
        (0, 1): 0.8,  # 0 and 1 are same identity
        (2, 3): 0.7,  # 2 and 3 are same identity
        (0, 2): 0.3,  # below threshold, should not connect
    }
    clusters = solver.solve(similarities, num_nodes=4)

    # Should find at least 2 clusters: {0,1} and {2,3}
    multi_clusters = [c for c in clusters if len(c) > 1]
    assert len(multi_clusters) == 2

    # Check correct grouping
    cluster_sorted = sorted(multi_clusters, key=lambda c: min(c))
    assert cluster_sorted[0] == {0, 1}
    assert cluster_sorted[1] == {2, 3}


def test_all_below_threshold():
    solver = GraphSolver(similarity_threshold=0.9)
    similarities = {(0, 1): 0.5, (1, 2): 0.6}
    clusters = solver.solve(similarities, num_nodes=3)
    # All singletons
    assert all(len(c) == 1 for c in clusters)


def test_single_cluster():
    solver = GraphSolver(similarity_threshold=0.3)
    similarities = {(0, 1): 0.5, (1, 2): 0.6, (0, 2): 0.4}
    clusters = solver.solve(similarities, num_nodes=3)
    multi = [c for c in clusters if len(c) > 1]
    assert len(multi) == 1
    assert multi[0] == {0, 1, 2}


def test_graph_stats():
    solver = GraphSolver(similarity_threshold=0.5)
    similarities = {(0, 1): 0.8, (2, 3): 0.7}
    stats = solver.get_graph_stats(similarities, num_nodes=4)
    assert stats["num_nodes"] == 4
    assert stats["num_edges"] == 2
    # nodes 0,1,2,3. Edges: 0-1, 2-3. Components: {0,1}, {2,3} = 2 components
    assert stats["num_components"] == 2


def test_bridge_pruning():
    """Bridge pruning should remove weak bridge edges."""
    solver = GraphSolver(
        similarity_threshold=0.4,
        algorithm="connected_components",
        bridge_prune_margin=0.2,  # prune bridges < 0.4 + 0.2 = 0.6
    )
    # 0-1 strong (0.8), 1-2 weak bridge (0.5), 2-3 strong (0.9)
    similarities = {(0, 1): 0.8, (1, 2): 0.5, (2, 3): 0.9}
    clusters = solver.solve(similarities, num_nodes=4)
    multi = sorted([c for c in clusters if len(c) > 1], key=lambda c: min(c))
    # Bridge between 1-2 (0.5 < 0.6) should be removed
    assert len(multi) == 2
    assert multi[0] == {0, 1}
    assert multi[1] == {2, 3}


def test_bridge_pruning_keeps_strong_bridges():
    """Strong bridges should NOT be pruned."""
    solver = GraphSolver(
        similarity_threshold=0.4,
        algorithm="connected_components",
        bridge_prune_margin=0.1,  # prune bridges < 0.4 + 0.1 = 0.5
    )
    # 0-1 strong (0.8), 1-2 strong bridge (0.7 > 0.5), 2-3 strong (0.9)
    similarities = {(0, 1): 0.8, (1, 2): 0.7, (2, 3): 0.9}
    clusters = solver.solve(similarities, num_nodes=4)
    multi = [c for c in clusters if len(c) > 1]
    # All connected — bridge is strong enough
    assert len(multi) == 1
    assert multi[0] == {0, 1, 2, 3}


def test_max_component_size():
    """Oversized components should be split."""
    solver = GraphSolver(
        similarity_threshold=0.3,
        algorithm="connected_components",
        max_component_size=2,
    )
    # 3-node chain: 0-1 (0.9), 1-2 (0.5), 0-2 (0.4)
    similarities = {(0, 1): 0.9, (1, 2): 0.5, (0, 2): 0.4}
    clusters = solver.solve(similarities, num_nodes=3)
    # Component of 3 exceeds max_size=2, weakest edge removed
    for c in clusters:
        assert len(c) <= 2


def test_max_component_size_no_split_needed():
    """Already-small components should not be affected."""
    solver = GraphSolver(
        similarity_threshold=0.3,
        algorithm="connected_components",
        max_component_size=5,
    )
    similarities = {(0, 1): 0.8, (2, 3): 0.7}
    clusters = solver.solve(similarities, num_nodes=4)
    multi = sorted([c for c in clusters if len(c) > 1], key=lambda c: min(c))
    assert len(multi) == 2
    assert multi[0] == {0, 1}
    assert multi[1] == {2, 3}
