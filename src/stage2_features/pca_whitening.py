"""PCA whitening for embedding refinement."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from loguru import logger
from sklearn.decomposition import PCA


class PCAWhitener:
    """PCA dimensionality reduction and whitening for ReID embeddings.

    Fits PCA on gallery embeddings and transforms query/gallery embeddings
    to a lower-dimensional, decorrelated space.
    """

    def __init__(self, n_components: int = 512):
        self.n_components = n_components
        self.pca: PCA | None = None

    def fit(self, embeddings: np.ndarray) -> None:
        """Fit PCA on a matrix of embeddings.

        Args:
            embeddings: (N, D) float32 matrix where N >= n_components.
        """
        n_samples, n_features = embeddings.shape
        actual_components = min(self.n_components, n_features, n_samples)

        if actual_components < self.n_components:
            logger.warning(
                f"Reducing PCA components from {self.n_components} to {actual_components} "
                f"(samples={n_samples}, features={n_features})"
            )

        self.pca = PCA(n_components=actual_components, whiten=True)
        self.pca.fit(embeddings)

        explained = self.pca.explained_variance_ratio_.sum()
        logger.info(
            f"PCA fitted: {n_features}D -> {actual_components}D, "
            f"explained variance: {explained:.3f}"
        )

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Transform embeddings using the fitted PCA.

        Args:
            embeddings: (N, D) float32 matrix.

        Returns:
            (N, n_components) transformed embeddings.
        """
        if self.pca is None:
            raise RuntimeError("PCA not fitted. Call fit() first or load() a saved model.")
        return self.pca.transform(embeddings).astype(np.float32)

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(embeddings)
        return self.transform(embeddings)

    def save(self, path: str | Path) -> None:
        """Save fitted PCA model to disk."""
        if self.pca is None:
            raise RuntimeError("Nothing to save — PCA not fitted.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.pca, f)
        logger.info(f"PCA model saved to {path}")

    def load(self, path: str | Path) -> None:
        """Load a previously fitted PCA model."""
        with open(path, "rb") as f:
            self.pca = pickle.load(f)
        self.n_components = self.pca.n_components_
        logger.info(f"PCA model loaded from {path} (n_components={self.n_components})")
