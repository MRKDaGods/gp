"""FAISS vector index for efficient embedding similarity search."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import faiss
import numpy as np
from loguru import logger


class FAISSIndex:
    """Wraps a FAISS index for building, searching, and persisting embedding indices.

    Uses IndexFlatIP (inner product) by default, which is equivalent to
    cosine similarity when embeddings are L2-normalized.
    """

    def __init__(self, index_type: str = "flat_ip"):
        self.index_type = index_type
        self.index: Optional[faiss.Index] = None
        self.id_map: Optional[List[int]] = None

    def build(self, embeddings: np.ndarray, ids: Optional[List[int]] = None) -> None:
        """Build the FAISS index from a matrix of embeddings.

        Args:
            embeddings: (N, D) float32 matrix (should be L2-normalized for cosine).
            ids: Optional list of integer IDs mapping to each row. If None, uses 0..N-1.
        """
        n, d = embeddings.shape
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

        if self.index_type == "flat_ip":
            self.index = faiss.IndexFlatIP(d)
        elif self.index_type == "flat_l2":
            self.index = faiss.IndexFlatL2(d)
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")

        self.index.add(embeddings)
        self.id_map = ids if ids is not None else list(range(n))

        logger.debug(f"FAISS index built: {n} vectors, {d}D, type={self.index_type}")

    def search(
        self,
        query: np.ndarray,
        top_k: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search for the top-K nearest neighbors.

        Args:
            query: (Q, D) float32 query matrix.
            top_k: Number of results per query.

        Returns:
            (distances, indices) both of shape (Q, top_k).
            For IndexFlatIP, distances are similarity scores (higher = more similar).
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build() first.")

        query = np.ascontiguousarray(query, dtype=np.float32)
        top_k = min(top_k, self.index.ntotal)

        distances, indices = self.index.search(query, top_k)
        return distances, indices

    def search_single(
        self,
        query_vec: np.ndarray,
        top_k: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search with a single query vector.

        Args:
            query_vec: (D,) float32 vector.
            top_k: Number of results.

        Returns:
            (distances, indices) both of shape (top_k,).
        """
        query = query_vec.reshape(1, -1)
        distances, indices = self.search(query, top_k)
        return distances[0], indices[0]

    def save(self, path: str | Path) -> None:
        """Save index to disk."""
        if self.index is None:
            raise RuntimeError("Nothing to save — index not built.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))

    def load(self, path: str | Path) -> None:
        """Load index from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"FAISS index not found: {path}")
        self.index = faiss.read_index(str(path))
        self.id_map = list(range(self.index.ntotal))
        logger.debug(f"FAISS index loaded: {self.index.ntotal} vectors from {path}")

    @property
    def size(self) -> int:
        return self.index.ntotal if self.index else 0
