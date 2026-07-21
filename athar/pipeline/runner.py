"""The pipeline runner: executes the stage DAG against one run manifest.

Guarantees (the v1 failure modes this closes):

- **Config is frozen before anything runs.** A manifest without a
  ``ResolvedConfig`` is refused — no run ever executes with ambient config.
- **State lives in exactly two places**: the manifest (what was produced)
  and per-stage checkpoint files (where an interrupted stage got to). Both
  are written atomically; there is no third store to drift.
- **Resume is two-level.** Stage-level: ``Stage.is_complete`` lets finished
  stages skip (their artifacts validate). Chunk-level: a stage saves a small
  JSON cursor via ``StageContext.save_checkpoint`` as it processes chunks
  and resumes from it after a crash; the runner clears it on completion.
- **Every observable fact is a typed event** (events.py). No stdout parsing.
- **Failure is a recorded outcome, not a stack trace to nowhere**: the
  manifest ends FAILED/CANCELLED with the error captured, then re-raises.
"""

from __future__ import annotations

import json
import logging
import threading
import traceback as _traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

from pydantic import BaseModel

from athar.components.registry import ComponentRegistry
from athar.components.registry import registry as default_registry
from athar.contracts.manifest import ArtifactRecord, RunManifest, RunStatus
from athar.contracts.store import FilesystemRunStore
from athar.pipeline.events import (
    ArtifactWritten,
    RunCancelled,
    RunCompleted,
    RunFailed,
    StageCompleted,
    StageProgress,
    StageSkipped,
    StageStarted,
    dump_event,
)
from athar.pipeline.graph import Stage
from athar.profiles.base import RunProfile

logger = logging.getLogger(__name__)

CHECKPOINT_DIRNAME = "checkpoints"
EVENTS_FILENAME = "events.jsonl"

EventSink = Callable[[BaseModel], None]


class RunnerError(RuntimeError):
    pass


class RunCancelledError(RuntimeError):
    """Raised inside a stage when the cancellation token fires."""


class CancellationToken:
    """Cooperative cancellation; checked by the runner between stages and by
    stages between chunks (``raise_if_cancelled``)."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise RunCancelledError()


def jsonl_event_sink(path: Path) -> EventSink:
    """Append events as JSON lines, flushed per event (crash-readable)."""

    def sink(event: BaseModel) -> None:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(dump_event(event) + "\n")

    return sink


@dataclass
class StageContext:
    """Everything the runner hands a stage. Stages touch the world ONLY
    through this object — no ambient globals, no direct path building."""

    manifest: RunManifest
    store: FilesystemRunStore
    profile: RunProfile
    registry: ComponentRegistry
    cancel: CancellationToken
    _emit: EventSink
    current_stage: str = field(default="", init=False)

    # -- paths -------------------------------------------------------------
    @property
    def run_dir(self) -> Path:
        return self.store.run_dir(self.manifest.run_id)

    def artifact_path(self, name: str) -> Path:
        return self.store.artifact_path(self.manifest, name)

    # -- events ------------------------------------------------------------
    def emit(self, event: BaseModel) -> None:
        self._emit(event)

    def progress(self, done: int, total: int, camera_id: Optional[str] = None) -> None:
        self.emit(
            StageProgress(
                run_id=self.manifest.run_id,
                stage=self.current_stage,
                camera_id=camera_id,
                done=done,
                total=total,
            )
        )

    # -- artifacts ---------------------------------------------------------
    def register_artifact(self, record: ArtifactRecord) -> None:
        """Register an artifact the stage has finished writing; persists the
        manifest immediately so a later crash cannot orphan the file."""
        self.manifest.register_artifact(record)
        self.store.save(self.manifest)
        self.emit(
            ArtifactWritten(
                run_id=self.manifest.run_id,
                stage=self.current_stage,
                artifact_name=record.name,
            )
        )

    # -- chunk-level resume ------------------------------------------------
    def _checkpoint_path(self, stage_name: str) -> Path:
        return self.run_dir / CHECKPOINT_DIRNAME / f"{stage_name}.json"

    def save_checkpoint(self, state: dict) -> None:
        """Persist the current stage's chunk cursor (atomic replace)."""
        path = self._checkpoint_path(self.current_stage)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state), encoding="utf-8")
        tmp.replace(path)

    def load_checkpoint(self) -> Optional[dict]:
        path = self._checkpoint_path(self.current_stage)
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def clear_checkpoint(self, stage_name: Optional[str] = None) -> None:
        path = self._checkpoint_path(stage_name or self.current_stage)
        path.unlink(missing_ok=True)


class PipelineRunner:
    """Executes stages in order against one manifest, with resume."""

    def __init__(
        self,
        store: FilesystemRunStore,
        stages: Sequence[Stage],
        registry: ComponentRegistry = default_registry,
        extra_sinks: Sequence[EventSink] = (),
    ) -> None:
        if not stages:
            raise RunnerError("runner needs at least one stage")
        names = [s.name for s in stages]
        if len(set(names)) != len(names):
            raise RunnerError(f"duplicate stage names: {names}")
        self.store = store
        self.stages = list(stages)
        self.registry = registry
        self.extra_sinks = list(extra_sinks)

    def run(
        self,
        manifest: RunManifest,
        profile: RunProfile,
        cancel: Optional[CancellationToken] = None,
    ) -> RunManifest:
        if manifest.config is None:
            raise RunnerError(
                f"run {manifest.run_id} has no ResolvedConfig — refusing to execute "
                "with ambient configuration (freeze config before running)"
            )
        if manifest.status == RunStatus.COMPLETED:
            raise RunnerError(f"run {manifest.run_id} already completed")
        if manifest.profile_name != profile.name:
            raise RunnerError(
                f"manifest bound to profile {manifest.profile_name!r}, got {profile.name!r}"
            )

        cancel = cancel or CancellationToken()
        run_dir = self.store.run_dir(manifest.run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        sinks = [jsonl_event_sink(run_dir / EVENTS_FILENAME), *self.extra_sinks]

        def emit(event: BaseModel) -> None:
            for sink in sinks:
                sink(event)

        ctx = StageContext(
            manifest=manifest,
            store=self.store,
            profile=profile,
            registry=self.registry,
            cancel=cancel,
            _emit=emit,
        )

        manifest.status = RunStatus.RUNNING
        manifest.error = None
        self.store.save(manifest)

        current: Optional[str] = None
        try:
            for stage in self.stages:
                current = ctx.current_stage = stage.name
                cancel.raise_if_cancelled()
                if stage.is_complete(ctx):
                    logger.info("run %s: stage %s already complete, skipping",
                                manifest.run_id, stage.name)
                    emit(StageSkipped(run_id=manifest.run_id, stage=stage.name))
                    continue
                emit(StageStarted(run_id=manifest.run_id, stage=stage.name))
                stage.run(ctx)
                ctx.clear_checkpoint(stage.name)
                self.store.save(manifest)
                emit(StageCompleted(run_id=manifest.run_id, stage=stage.name))
        except RunCancelledError:
            manifest.status = RunStatus.CANCELLED
            manifest.error = f"cancelled during stage {current!r}"
            self.store.save(manifest)
            emit(RunCancelled(run_id=manifest.run_id, stage=current))
            return manifest
        except Exception as exc:
            manifest.status = RunStatus.FAILED
            manifest.error = f"{current}: {exc}"
            self.store.save(manifest)
            emit(
                RunFailed(
                    run_id=manifest.run_id,
                    stage=current,
                    error=str(exc),
                    traceback=_traceback.format_exc(),
                )
            )
            raise

        manifest.status = RunStatus.COMPLETED
        self.store.save(manifest)
        emit(RunCompleted(run_id=manifest.run_id))
        return manifest
