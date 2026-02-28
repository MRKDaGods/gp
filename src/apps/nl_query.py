"""Natural Language Query Engine for trajectory search.

Uses sentence-transformers to embed trajectory descriptions and match
against user queries via cosine similarity.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from loguru import logger

from src.core.data_models import GlobalTrajectory


class NLQueryEngine:
    """Search global trajectories using natural language queries.

    Generates text descriptions for each trajectory, embeds them using
    sentence-transformers, and matches against user queries.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.trajectories: List[GlobalTrajectory] = []
        self.descriptions: List[str] = []
        self.description_embeddings: Optional[np.ndarray] = None

    def _load_model(self):
        """Lazy-load the sentence transformer model."""
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(f"Loaded sentence-transformer: {self.model_name}")

    def build_index(self, trajectories: List[GlobalTrajectory]) -> None:
        """Build the search index from trajectories.

        Generates a text description for each trajectory and computes embeddings.

        Args:
            trajectories: Global trajectories to index.
        """
        self._load_model()
        self.trajectories = trajectories
        self.descriptions = [
            self._describe_trajectory(traj) for traj in trajectories
        ]

        self.description_embeddings = self.model.encode(
            self.descriptions,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        logger.info(f"NL query index built: {len(trajectories)} trajectories")

    def query(
        self,
        text: str,
        top_k: int = 10,
    ) -> List[Tuple[GlobalTrajectory, float, str]]:
        """Search trajectories using a natural language query.

        Args:
            text: Natural language query, e.g. "red car seen on camera 1 and 3".
            top_k: Number of results to return.

        Returns:
            List of (trajectory, similarity_score, description) tuples,
            sorted by descending similarity.
        """
        if self.description_embeddings is None:
            logger.warning("Index not built. Call build_index() first.")
            return []

        self._load_model()

        query_embedding = self.model.encode(
            [text], normalize_embeddings=True
        )

        # Cosine similarity (embeddings are already L2-normalized)
        similarities = np.dot(query_embedding, self.description_embeddings.T)[0]

        # Get top-K indices
        top_indices = np.argsort(-similarities)[:top_k]

        results = []
        for idx in top_indices:
            results.append((
                self.trajectories[idx],
                float(similarities[idx]),
                self.descriptions[idx],
            ))

        return results

    @staticmethod
    def _describe_trajectory(traj: GlobalTrajectory) -> str:
        """Generate a natural language description of a trajectory."""
        parts = []

        # Object type
        parts.append(f"{traj.class_name}")

        # Camera sequence
        cameras = traj.camera_sequence
        if len(cameras) == 1:
            parts.append(f"seen on camera {cameras[0]}")
        else:
            parts.append(f"seen on cameras {', '.join(cameras[:-1])} and {cameras[-1]}")

        # Time info
        start, end = traj.time_span
        parts.append(f"from {start:.1f}s to {end:.1f}s")
        parts.append(f"total duration {traj.total_duration:.1f} seconds")

        # Number of cameras
        parts.append(f"across {traj.num_cameras} camera(s)")

        # Tracklet details
        for t in sorted(traj.tracklets, key=lambda x: x.start_time):
            parts.append(
                f"on camera {t.camera_id} from {t.start_time:.1f}s to {t.end_time:.1f}s "
                f"({t.num_frames} frames)"
            )

        return " ".join(parts)
