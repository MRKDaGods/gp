"""Embedding artifact model for timeline visual search."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class EmbeddingArtifact:
    """Stage-2 embedding matrix plus its row-level metadata index."""

    run_id: str
    embeddings: np.ndarray
    index: List[Dict[str, Any]]

    @property
    def dim(self) -> int:
        if self.embeddings.ndim != 2:
            return 0
        return int(self.embeddings.shape[1])

    @classmethod
    def load(cls, stage2_dir: str | Path, run_id: str) -> Optional["EmbeddingArtifact"]:
        stage2_dir = Path(stage2_dir)
        embeddings_path = stage2_dir / "embeddings.npy"
        index_path = stage2_dir / "embedding_index.json"
        if not embeddings_path.exists() or not index_path.exists():
            return None

        embeddings = np.load(embeddings_path).astype(np.float32, copy=False)
        index = json.loads(index_path.read_text(encoding="utf-8"))
        if not isinstance(index, list):
            return None
        return cls(run_id=run_id, embeddings=embeddings, index=index)

    @classmethod
    def load_for_run(cls, run_id: str, output_dir: str | Path) -> Optional["EmbeddingArtifact"]:
        return cls.load(Path(output_dir) / run_id / "stage2", run_id)
