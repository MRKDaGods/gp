"""Robust aggregation for per-tracklet multi-query embeddings."""

from __future__ import annotations

from typing import Callable, Iterable

import numpy as np


DEFAULT_MIN_K = 8
GEOMETRIC_MEDIAN_MAX_ITER = 20
GEOMETRIC_MEDIAN_EPS = 1e-6


def l2_normalize_vector(vector: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Return a float32 L2-normalized copy of one vector."""
    array = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    if norm <= eps:
        return np.zeros_like(array, dtype=np.float32)
    return (array / norm).astype(np.float32)


def l2_normalize_rows(embeddings: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Return row-wise L2-normalized embeddings."""
    array = np.asarray(embeddings, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D embedding block, got shape {array.shape}")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return (array / np.maximum(norms, eps)).astype(np.float32)


def trim_stage2_padding(embeddings: np.ndarray, atol: float = 1e-6) -> np.ndarray:
    """Remove the repeated pooled suffix used to pad short multi-query tracks."""
    rows = l2_normalize_rows(embeddings)
    if rows.shape[0] <= 1:
        return rows

    last = rows[-1]
    suffix_start = rows.shape[0] - 1
    while suffix_start > 0 and np.allclose(rows[suffix_start - 1], last, atol=atol):
        suffix_start -= 1

    suffix_count = rows.shape[0] - suffix_start
    if suffix_count <= 1:
        return rows
    return rows[:suffix_start]


def softmax_quality_mean(
    embeddings: np.ndarray,
    qualities: np.ndarray | None = None,
    temperature: float = 3.0,
) -> np.ndarray:
    """Quality-weighted fallback matching Stage 2's softmax pooling shape."""
    rows = l2_normalize_rows(embeddings)
    if qualities is None:
        return mean_pool(rows)

    quality_array = np.asarray(qualities, dtype=np.float32)
    if quality_array.ndim != 1 or quality_array.shape[0] != rows.shape[0]:
        raise ValueError(
            f"qualities must have shape ({rows.shape[0]},), got {quality_array.shape}"
        )
    logits = quality_array * float(temperature)
    logits = logits - np.max(logits)
    weights = np.exp(logits).astype(np.float32)
    weights = weights / max(float(weights.sum()), 1e-8)
    return l2_normalize_vector((rows * weights[:, np.newaxis]).sum(axis=0))


def mean_pool(embeddings: np.ndarray) -> np.ndarray:
    """Arithmetic mean followed by L2 renormalization."""
    rows = l2_normalize_rows(embeddings)
    return l2_normalize_vector(rows.mean(axis=0))


def median_pool(embeddings: np.ndarray) -> np.ndarray:
    """Per-dimension median followed by L2 renormalization."""
    rows = l2_normalize_rows(embeddings)
    return l2_normalize_vector(np.median(rows, axis=0))


def geometric_median_pool(
    embeddings: np.ndarray,
    max_iter: int = GEOMETRIC_MEDIAN_MAX_ITER,
    eps: float = GEOMETRIC_MEDIAN_EPS,
) -> np.ndarray:
    """Weiszfeld geometric median followed by L2 renormalization."""
    rows = l2_normalize_rows(embeddings)
    estimate = rows.mean(axis=0)

    for _ in range(max_iter):
        distances = np.linalg.norm(rows - estimate[np.newaxis, :], axis=1)
        exact = distances < eps
        if np.any(exact):
            estimate = rows[int(np.argmax(exact))]
            break

        weights = 1.0 / np.maximum(distances, eps)
        next_estimate = (rows * weights[:, np.newaxis]).sum(axis=0) / weights.sum()
        if float(np.linalg.norm(next_estimate - estimate)) < eps:
            estimate = next_estimate
            break
        estimate = next_estimate

    return l2_normalize_vector(estimate)


def medoid_pool(embeddings: np.ndarray) -> np.ndarray:
    """Return the input row with highest summed cosine similarity to all rows."""
    rows = l2_normalize_rows(embeddings)
    similarity = rows @ rows.T
    medoid_index = int(np.argmax(similarity.sum(axis=1)))
    return rows[medoid_index].astype(np.float32)


def _top_by_similarity_to_reference(
    rows: np.ndarray,
    reference: np.ndarray,
    keep_count: int,
) -> np.ndarray:
    keep = max(1, min(int(keep_count), rows.shape[0]))
    scores = rows @ l2_normalize_vector(reference)
    indices = np.argsort(-scores)[:keep]
    return rows[indices]


def trimmed_mean_pool(embeddings: np.ndarray, trim_fraction: float) -> np.ndarray:
    """Drop the least central rows by cosine-to-mean, then average."""
    rows = l2_normalize_rows(embeddings)
    trim_count = int(np.floor(rows.shape[0] * float(trim_fraction)))
    keep_count = max(1, rows.shape[0] - trim_count)
    reference = mean_pool(rows)
    kept = _top_by_similarity_to_reference(rows, reference, keep_count)
    return mean_pool(kept)


def top_to_mean_pool(embeddings: np.ndarray, keep_count: int = 12) -> np.ndarray:
    """Average the rows nearest to the arithmetic-mean direction."""
    rows = l2_normalize_rows(embeddings)
    kept = _top_by_similarity_to_reference(rows, mean_pool(rows), keep_count)
    return mean_pool(kept)


def top_to_medoid_pool(embeddings: np.ndarray, keep_count: int = 12) -> np.ndarray:
    """Average the rows nearest to the medoid direction."""
    rows = l2_normalize_rows(embeddings)
    kept = _top_by_similarity_to_reference(rows, medoid_pool(rows), keep_count)
    return mean_pool(kept)


AGGREGATION_MODES: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "mean": mean_pool,
    "median": median_pool,
    "geo_median": geometric_median_pool,
    "geometric_median": geometric_median_pool,
    "medoid": medoid_pool,
    "trimmed_mean_10": lambda rows: trimmed_mean_pool(rows, 0.10),
    "trimmed_mean_25": lambda rows: trimmed_mean_pool(rows, 0.25),
    "top12_to_mean": lambda rows: top_to_mean_pool(rows, 12),
    "top12_to_medoid": lambda rows: top_to_medoid_pool(rows, 12),
}

SWEEP_MODES = (
    "mean",
    "median",
    "geo_median",
    "medoid",
    "trimmed_mean_10",
    "trimmed_mean_25",
    "top12_to_mean",
    "top12_to_medoid",
)


def aggregate_tracklet_embeddings(
    embeddings: np.ndarray,
    mode: str,
    fallback_embedding: np.ndarray | None = None,
    min_k: int = DEFAULT_MIN_K,
    trim_padding: bool = True,
) -> tuple[np.ndarray, bool]:
    """Aggregate one tracklet's multi-query rows.

    Returns ``(embedding, used_fallback)``. If fewer than ``min_k`` rows are
    available, the provided fallback embedding is returned, or a mean pool when
    no fallback is available.
    """
    rows = trim_stage2_padding(embeddings) if trim_padding else l2_normalize_rows(embeddings)
    if rows.shape[0] == 0:
        if fallback_embedding is None:
            raise ValueError("Cannot aggregate an empty embedding block without fallback")
        return l2_normalize_vector(fallback_embedding), True

    if rows.shape[0] < int(min_k):
        if fallback_embedding is not None:
            return l2_normalize_vector(fallback_embedding), True
        return mean_pool(rows), True

    try:
        aggregator = AGGREGATION_MODES[mode]
    except KeyError as exc:
        choices = ", ".join(sorted(AGGREGATION_MODES))
        raise ValueError(f"Unknown robust pooling mode '{mode}'. Choices: {choices}") from exc

    return aggregator(rows), False


def aggregate_embedding_matrix(
    multi_query_embeddings: np.ndarray,
    mode: str,
    fallback_embeddings: np.ndarray | None = None,
    min_k: int = DEFAULT_MIN_K,
    trim_padding: bool = True,
) -> tuple[np.ndarray, int]:
    """Aggregate an ``(N, K, D)`` multi-query tensor into ``(N, D)``."""
    tensor = np.asarray(multi_query_embeddings, dtype=np.float32)
    if tensor.ndim != 3:
        raise ValueError(f"Expected multi-query tensor shape (N, K, D), got {tensor.shape}")
    if fallback_embeddings is not None:
        fallback_embeddings = np.asarray(fallback_embeddings, dtype=np.float32)
        if fallback_embeddings.shape != (tensor.shape[0], tensor.shape[2]):
            raise ValueError(
                "fallback_embeddings must have shape "
                f"{(tensor.shape[0], tensor.shape[2])}, got {fallback_embeddings.shape}"
            )

    pooled: list[np.ndarray] = []
    fallback_count = 0
    for index, rows in enumerate(tensor):
        fallback = None if fallback_embeddings is None else fallback_embeddings[index]
        embedding, used_fallback = aggregate_tracklet_embeddings(
            rows,
            mode=mode,
            fallback_embedding=fallback,
            min_k=min_k,
            trim_padding=trim_padding,
        )
        pooled.append(embedding)
        fallback_count += int(used_fallback)

    return np.stack(pooled, axis=0).astype(np.float32), fallback_count


def available_modes() -> Iterable[str]:
    """Return the public sweep modes in execution order."""
    return SWEEP_MODES
