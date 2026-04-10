"""DatasetRepository — protocol and in-memory implementation.

The protocol defines the read-only view over pipeline artefacts needed by
``TimelineService`` and other query-time services.  The in-memory
implementation delegates directly to the global state dicts and the
filesystem under ``OUTPUT_DIR``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from backend.models.embedding import EmbeddingArtifact


@runtime_checkable
class DatasetRepository(Protocol):
    """Read-only interface over pipeline artefacts for query-time services.

    Implementations must be safe to call from both sync and async contexts.
    All methods that touch the filesystem should be inexpensive enough to
    call inline (they read small JSON/npy files, not large videos).
    """

    def get_video(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Return the video record for *video_id*, or ``None`` if unknown."""
        ...

    def get_latest_run(self, video_id: str) -> Optional[str]:
        """Return the latest pipeline run_id that processed *video_id*.

        Returns ``None`` if no run has been recorded for the video.
        """
        ...

    def list_trajectories(self, run_id: str) -> Optional[List[Dict[str, Any]]]:
        """Load global trajectories for *run_id* from Stage-4 output.

        Returns:
            ``None``  — if ``outputs/{run_id}/stage4/global_trajectories.json``
                        does not exist (Stage 4 has not run for this run).
            ``[]``    — if the file exists but is empty or non-list.
            List[…]  — parsed trajectory list otherwise.
        """
        ...

    def load_embedding_artifact(self, run_id: str) -> Optional[EmbeddingArtifact]:
        """Load the Stage-2 embedding artefact for *run_id*.

        Returns ``None`` if the artefact files are absent.
        """
        ...


class InMemoryDatasetRepository:
    """Concrete ``DatasetRepository`` backed by the in-memory state dicts.

    The three constructor arguments correspond directly to the module-level
    globals in ``backend.state`` and the constant in ``backend.config``.

    Args:
        uploaded_videos:     Mapping of ``video_id`` → video-record dict.
        video_to_latest_run: Mapping of ``video_id`` → latest ``run_id``.
        output_dir:          Root outputs directory (``backend.config.OUTPUT_DIR``).
    """

    def __init__(
        self,
        uploaded_videos: Dict[str, Dict[str, Any]],
        video_to_latest_run: Dict[str, str],
        output_dir: Path,
    ) -> None:
        self._videos = uploaded_videos
        self._latest_runs = video_to_latest_run
        self._output_dir = output_dir

    # ------------------------------------------------------------------
    # Protocol implementation
    # ------------------------------------------------------------------

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
