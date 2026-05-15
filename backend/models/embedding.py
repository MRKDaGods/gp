"""Embedding artifact model used by backend timeline services."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


class EmbeddingArtifact:
    """Stage-2 embedding matrix plus its row-level metadata index."""

    def __init__(self, run_id: str, embeddings: np.ndarray, index: List[Dict[str, Any]]) -> None:
        self.run_id = run_id
        self.embeddings = embeddings
        self.index = index

    @property
    def dim(self) -> int:
        if self.embeddings.ndim < 2:
            return 0
        return int(self.embeddings.shape[1])

    @classmethod
    def load(cls, directory: Path, run_id: str) -> Optional["EmbeddingArtifact"]:
        embeddings_path = directory / "embeddings.npy"
        index_path = directory / "embedding_index.json"
        if not embeddings_path.exists() or not index_path.exists():
            return None
        embeddings = np.load(embeddings_path).astype(np.float32, copy=False)
        index = json.loads(index_path.read_text(encoding="utf-8"))
        if not isinstance(index, list):
            return None
        return cls(run_id=run_id, embeddings=embeddings, index=index)

    @classmethod
    def load_for_run(cls, run_id: str, output_dir: Path) -> Optional["EmbeddingArtifact"]:
        return cls.load(output_dir / run_id / "stage2", run_id)
