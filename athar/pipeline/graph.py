"""The stage DAG.

One graph for every profile::

    ingest -> detect_track -> embed -> index -> associate -> package

Multi-view (WILDTRACK) runs bind a MultiViewDetector into detect_track;
runs without embeddings-based search skip embed/index by profile, not by
code fork. Stages communicate ONLY through manifest-registered artifacts.
"""

from __future__ import annotations

from typing import Protocol

STAGES: tuple[str, ...] = (
    "ingest",
    "detect_track",
    "embed",
    "index",
    "associate",
    "package",
)


class StageContext(Protocol):
    """What the runner hands each stage: manifest, store, profile, components,
    event emitter, and a cancellation token. Concrete class arrives with the
    runner in Phase 2."""

    ...


class Stage(Protocol):
    name: str

    def run(self, ctx: "StageContext") -> None: ...

    def is_complete(self, ctx: "StageContext") -> bool:
        """Resume support: True when this stage's artifacts already exist and
        validate — the runner skips completed stages after a crash."""
        ...
