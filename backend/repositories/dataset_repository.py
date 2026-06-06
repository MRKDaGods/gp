"""DatasetRepository - protocol and in-memory implementation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from backend.models.embedding import EmbeddingArtifact


@runtime_checkable
class DatasetRepository(Protocol):
    """Read-only interface over pipeline artefacts for query-time services."""

    def get_video(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Return the video record for *video_id*, or ``None`` if unknown."""
        ...

    def get_latest_run(self, video_id: str) -> Optional[str]:
        """Return the latest pipeline run_id that processed *video_id*.

        Returns ``None`` if no run has been recorded for the video.
        """
        ...

    def list_trajectories(self, run_id: str) -> Optional[List[Dict[str, Any]]]:
        """Load global trajectories for *run_id* from Stage-4 output."""
        ...

    def load_embedding_artifact(self, run_id: str) -> Optional[EmbeddingArtifact]:
        """Load the Stage-2 embedding artefact for *run_id*.

        Returns ``None`` if the artefact files are absent.
        """
        ...


class InMemoryDatasetRepository:
    """Concrete ``DatasetRepository`` backed by the in-memory state dicts."""

    def __init__(
        self,
        uploaded_videos: Dict[str, Dict[str, Any]],
        video_to_latest_run: Dict[str, str],
        output_dir: Path,
    ) -> None:
        self._videos = uploaded_videos
        self._latest_runs = video_to_latest_run
        self._output_dir = output_dir

    # Protocol implementation

    def get_video(self, video_id: str) -> Optional[Dict[str, Any]]:
        return self._videos.get(video_id)

    def get_latest_run(self, video_id: str) -> Optional[str]:
        return self._latest_runs.get(video_id)

    def list_trajectories(self, run_id: str) -> Optional[List[Dict[str, Any]]]:
        traj_path = self._output_dir / run_id / "stage4" / "global_trajectories.json"
        if not traj_path.exists():
            return None
        data = json.loads(traj_path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []

    def load_embedding_artifact(self, run_id: str) -> Optional[EmbeddingArtifact]:
        return EmbeddingArtifact.load_for_run(run_id, self._output_dir)
