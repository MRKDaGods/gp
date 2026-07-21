"""PipelineRunner tests: ordering, events, failure capture, two-level resume,
cancellation — all with fake stages (no ML deps)."""

from __future__ import annotations

import pytest

from athar.contracts.config import ConfigLayer, ResolvedConfig
from athar.contracts.manifest import ArtifactRecord, RunManifest, RunRole, RunStatus
from athar.contracts.store import FilesystemRunStore
from athar.core.ids import new_run_id
from athar.core.types import EntityClass
from athar.pipeline.events import parse_event
from athar.pipeline.runner import (
    EVENTS_FILENAME,
    CancellationToken,
    PipelineRunner,
    RunnerError,
    StageContext,
    jsonl_event_sink,
)
from athar.profiles.base import ClassBranch, ComponentSpec, RunProfile


def _profile(name: str = "test-profile") -> RunProfile:
    spec = ComponentSpec(name="x")
    return RunProfile(
        name=name,
        detector=spec,
        branches=[
            ClassBranch(
                entity_classes=[EntityClass.CAR],
                tracker=spec,
                embedders=[spec],
                score_terms=[spec],
                solver=spec,
            )
        ],
    )


def _manifest(profile_name: str = "test-profile", with_config: bool = True) -> RunManifest:
    manifest = RunManifest(
        run_id=new_run_id(), role=RunRole.GALLERY, profile_name=profile_name
    )
    if with_config:
        manifest.config = ResolvedConfig.resolve(
            [(ConfigLayer.PROFILE_DEFAULT, {"a": 1})]
        )
    return manifest


class FakeStage:
    """Records calls; completion is 'my marker artifact exists'."""

    def __init__(self, name: str, fail_times: int = 0):
        self.name = name
        self.runs = 0
        self._fail_times = fail_times

    def is_complete(self, ctx: StageContext) -> bool:
        return f"{self.name}.done" in ctx.manifest.artifacts

    def run(self, ctx: StageContext) -> None:
        self.runs += 1
        ctx.cancel.raise_if_cancelled()
        if self._fail_times > 0:
            self._fail_times -= 1
            raise RuntimeError(f"{self.name} exploded")
        marker = ctx.run_dir / f"{self.name}.done"
        marker.write_text("ok", encoding="utf-8")
        ctx.register_artifact(
            ArtifactRecord(
                name=f"{self.name}.done",
                relpath=marker.name,
                schema_version=1,
                producer=f"fake/{self.name}",
            )
        )


class ChunkedStage:
    """Processes 10 chunks with checkpointing; optionally crashes at chunk 5
    on the first attempt. Records which chunks each attempt processed."""

    name = "chunked"
    TOTAL = 10

    def __init__(self, crash_once_at: int | None = None):
        self._crash_at = crash_once_at
        self.attempts: list[list[int]] = []

    def is_complete(self, ctx: StageContext) -> bool:
        return "chunked.done" in ctx.manifest.artifacts

    def run(self, ctx: StageContext) -> None:
        state = ctx.load_checkpoint() or {"next_chunk": 0}
        processed: list[int] = []
        self.attempts.append(processed)
        for chunk in range(state["next_chunk"], self.TOTAL):
            if self._crash_at is not None and chunk == self._crash_at:
                self._crash_at = None
                raise RuntimeError("power cut")
            processed.append(chunk)
            ctx.progress(done=chunk + 1, total=self.TOTAL)
            ctx.save_checkpoint({"next_chunk": chunk + 1})
        marker = ctx.run_dir / "chunked.done"
        marker.write_text("ok", encoding="utf-8")
        ctx.register_artifact(
            ArtifactRecord(
                name="chunked.done", relpath=marker.name,
                schema_version=1, producer="fake/chunked",
            )
        )


def _events(store: FilesystemRunStore, run_id: str) -> list:
    path = store.run_dir(run_id) / EVENTS_FILENAME
    return [
        parse_event(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


@pytest.fixture()
def store(tmp_path):
    return FilesystemRunStore(tmp_path / "runs")


class TestHappyPath:
    def test_stages_run_in_order_and_events_stream(self, store):
        stages = [FakeStage("alpha"), FakeStage("beta")]
        manifest = _manifest()
        result = PipelineRunner(store, stages).run(manifest, _profile())

        assert result.status == RunStatus.COMPLETED
        assert [s.runs for s in stages] == [1, 1]
        assert set(result.artifacts) == {"alpha.done", "beta.done"}
        # persisted copy matches
        assert store.load(manifest.run_id).status == RunStatus.COMPLETED

        kinds = [e.event for e in _events(store, manifest.run_id)]
        assert kinds == [
            "stage_started", "artifact_written", "stage_completed",
            "stage_started", "artifact_written", "stage_completed",
            "run_completed",
        ]

    def test_extra_sink_receives_events(self, store):
        seen = []
        runner = PipelineRunner(store, [FakeStage("alpha")], extra_sinks=[seen.append])
        runner.run(_manifest(), _profile())
        assert [type(e).__name__ for e in seen][-1] == "RunCompleted"


class TestGuards:
    def test_config_required(self, store):
        with pytest.raises(RunnerError, match="no ResolvedConfig"):
            PipelineRunner(store, [FakeStage("a")]).run(
                _manifest(with_config=False), _profile()
            )

    def test_completed_run_refused(self, store):
        manifest = _manifest()
        manifest.status = RunStatus.COMPLETED
        with pytest.raises(RunnerError, match="already completed"):
            PipelineRunner(store, [FakeStage("a")]).run(manifest, _profile())

    def test_profile_mismatch_refused(self, store):
        with pytest.raises(RunnerError, match="bound to profile"):
            PipelineRunner(store, [FakeStage("a")]).run(
                _manifest(profile_name="other"), _profile()
            )

    def test_duplicate_stage_names_refused(self, store):
        with pytest.raises(RunnerError, match="duplicate stage"):
            PipelineRunner(store, [FakeStage("a"), FakeStage("a")])


class TestFailureAndResume:
    def test_failure_recorded_then_stage_level_resume(self, store):
        alpha, beta = FakeStage("alpha"), FakeStage("beta", fail_times=1)
        runner = PipelineRunner(store, [alpha, beta])
        manifest = _manifest()

        with pytest.raises(RuntimeError, match="beta exploded"):
            runner.run(manifest, _profile())
        persisted = store.load(manifest.run_id)
        assert persisted.status == RunStatus.FAILED
        assert "beta" in (persisted.error or "")
        failed = [e for e in _events(store, manifest.run_id) if e.event == "run_failed"]
        assert failed and "beta exploded" in failed[0].error
        assert failed[0].traceback

        # resume: alpha skips (artifact exists), beta re-runs and succeeds
        result = runner.run(persisted, _profile())
        assert result.status == RunStatus.COMPLETED
        assert alpha.runs == 1  # never re-ran
        kinds = [e.event for e in _events(store, manifest.run_id)]
        assert "stage_skipped" in kinds

    def test_chunk_level_resume(self, store):
        stage = ChunkedStage(crash_once_at=5)
        runner = PipelineRunner(store, [stage])
        manifest = _manifest()

        with pytest.raises(RuntimeError, match="power cut"):
            runner.run(manifest, _profile())
        result = runner.run(store.load(manifest.run_id), _profile())

        assert result.status == RunStatus.COMPLETED
        assert stage.attempts[0] == [0, 1, 2, 3, 4]
        assert stage.attempts[1] == [5, 6, 7, 8, 9]  # resumed, not restarted
        # checkpoint cleared after completion
        assert not (store.run_dir(manifest.run_id) / "checkpoints" / "chunked.json").exists()

    def test_progress_events_carry_totals(self, store):
        stage = ChunkedStage()
        manifest = _manifest()
        PipelineRunner(store, [stage]).run(manifest, _profile())
        progress = [e for e in _events(store, manifest.run_id) if e.event == "stage_progress"]
        assert (progress[0].done, progress[0].total) == (1, 10)
        assert (progress[-1].done, progress[-1].total) == (10, 10)


class TestCancellation:
    def test_cancel_between_stages(self, store):
        cancel = CancellationToken()

        class CancellingStage(FakeStage):
            def run(self, ctx: StageContext) -> None:
                super().run(ctx)
                cancel.cancel()

        stages = [CancellingStage("alpha"), FakeStage("beta")]
        manifest = _manifest()
        result = PipelineRunner(store, stages).run(manifest, _profile(), cancel=cancel)

        assert result.status == RunStatus.CANCELLED
        assert stages[1].runs == 0
        kinds = [e.event for e in _events(store, manifest.run_id)]
        assert kinds[-1] == "run_cancelled"


class TestJsonlSink:
    def test_lines_parse_back(self, tmp_path):
        from athar.pipeline.events import StageStarted

        path = tmp_path / "e.jsonl"
        sink = jsonl_event_sink(path)
        sink(StageStarted(run_id="r1", stage="embed"))
        (event,) = [parse_event(l) for l in path.read_text().splitlines()]
        assert event.event == "stage_started" and event.stage == "embed"
