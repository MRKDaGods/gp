from __future__ import annotations

import numpy as np

from scripts.eval.eval_14t_fusion_veri776 import (
    average_query_expansion,
    build_rerank_state_from_similarity,
    compute_reranking_torch,
    l2_normalize,
    score_similarity,
)
from src.training.evaluate_reid import eval_market1501


def test_score_fusion_formula_is_deterministic() -> None:
    rng = np.random.default_rng(1234)
    q_tr = l2_normalize(rng.normal(size=(4, 8)).astype(np.float32))
    g_tr = l2_normalize(rng.normal(size=(12, 8)).astype(np.float32))
    q_cs = l2_normalize(rng.normal(size=(4, 8)).astype(np.float32))
    g_cs = l2_normalize(rng.normal(size=(12, 8)).astype(np.float32))

    weight = 0.7
    fused = score_similarity(q_tr, g_tr, q_cs, g_cs, weight)
    expected = weight * (q_cs[0] @ g_cs[1]) + (1.0 - weight) * (q_tr[0] @ g_tr[1])

    assert fused.shape == (4, 12)
    assert np.isclose(fused[0, 1], expected, atol=1e-7)
    assert np.allclose(fused, score_similarity(q_tr, g_tr, q_cs, g_cs, weight))


def test_aqe_k1_preserves_self_top1_identity() -> None:
    rng = np.random.default_rng(5678)
    features = l2_normalize(rng.normal(size=(16, 8)).astype(np.float32))

    expanded = average_query_expansion(features, k=1, iterations=1)

    assert np.allclose(expanded, features, atol=1e-7)


def test_rerank_prefers_identical_gallery_self_match() -> None:
    query = l2_normalize(np.eye(2, 4, dtype=np.float32))
    gallery = l2_normalize(
        np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
    )
    all_features = np.concatenate([query, gallery], axis=0)
    similarity = all_features @ all_features.T
    original_dist, initial_rank = build_rerank_state_from_similarity(similarity, max_k1=2)

    rerank_dist = compute_reranking_torch(
        original_dist,
        initial_rank,
        query_num=2,
        k1=2,
        k2=1,
        lambda_value=0.2,
    )

    assert rerank_dist[0, 0] <= rerank_dist[0, 1]
    assert rerank_dist[1, 1] <= rerank_dist[1, 0]


def test_eval_market1501_perfect_match() -> None:
    distmat = np.array(
        [
            [0.0, 2.0, 3.0],
            [2.0, 0.0, 3.0],
        ],
        dtype=np.float32,
    )
    q_pids = np.array([10, 20])
    g_pids = np.array([10, 20, 30])
    q_camids = np.array([0, 0])
    g_camids = np.array([1, 1, 1])

    mAP, cmc = eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids)

    assert mAP == 1.0
    assert cmc[0] == 1.0