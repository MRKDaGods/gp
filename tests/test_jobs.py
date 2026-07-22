"""Job queue + worker + service tests.

The worker integration tests run the REAL claim -> setup -> DAG-runner
path with fake stages monkeypatched into ``athar.pipeline.setup`` — no ML
components load, but ingest, manifests, checkpoints, events and resume are
all genuine.
"""

from __future__ import annotations

import threading

import cv2
import numpy as np
import pytest

from athar.contracts.manifest import ArtifactRecord, RunStatus
from athar.jobs.queue import (
    JobError,
    JobExecutor,
    JobNotFound,
    JobQueue,
    JobStatus,
)
from athar.jobs.service import JobService
from athar.jobs.worker import LocalRunExecutor, run_worker
from athar.pipeline.runner import EVENTS_FILENAME


@pytest.fixture()
def queue(tmp_path):
    q = JobQueue(tmp_path / "jobs.db")
    yield q
    q.close()


class TestQueue:
    def test_submit_get_roundtrip(self, queue):
        job = queue.submit("run_pipeline", {"answer": 42}, priority=3)
        loaded = queue.get(job.job_id)
        assert loaded.payload == {"answer": 42}
        assert loaded.status is JobStatus.QUEUED
        assert loaded.priority == 3
        assert loaded.created_at is not None

    def test_get_missing_raises(self, queue):
        with pytest.raises(JobNotFound):
            queue.get("ghost")

    def test_claim_priority_then_fifo(self, queue):
        low = queue.submit("k", {}, priority=0)
        high = queue.submit("k", {}, priority=5)
        low2 = queue.submit("k", {}, priority=0)
        order = [queue.claim("w").job_id for _ in range(3)]
        assert order == [high.job_id, low.job_id, low2.job_id]
        assert queue.claim("w") is None  # drained

    def test_claim_sets_bookkeeping(self, queue):
        queue.submit("k", {})
        job = queue.claim("worker-1")
        assert job.status is JobStatus.CLAIMED
        assert job.worker_id == "worker-1"
        assert job.attempt == 1
        assert job.claimed_at is not None and job.heartbeat_at is not None

    def test_claim_respects_executor(self, queue):
        kaggle_job = queue.submit("k", {}, executor=JobExecutor.KAGGLE)
        assert queue.claim("w", executor=JobExecutor.LOCAL) is None
        claimed = queue.claim("w", executor=JobExecutor.KAGGLE)
        assert claimed.job_id == kaggle_job.job_id

    def test_concurrent_claims_never_double_claim(self, tmp_path):
        seed = JobQueue(tmp_path / "jobs.db")
        jobs = [seed.submit("k", {"i": i}) for i in range(8)]
        seed.close()
        claimed: list[str] = []
        lock = threading.Lock()
        barrier = threading.Barrier(8)

        def worker(n: int) -> None:
            q = JobQueue(tmp_path / "jobs.db")
            try:
                barrier.wait()
                job = q.claim(f"w{n}")
                if job is not None:
                    with lock:
                        claimed.append(job.job_id)
            finally:
                q.close()

        threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert sorted(claimed) == sorted(j.job_id for j in jobs)  # each exactly once

    def test_cancel_queued_is_immediate(self, queue):
        job = queue.submit("k", {})
        assert queue.request_cancel(job.job_id) is JobStatus.CANCELLED
        assert queue.get(job.job_id).finished_at is not None
        assert queue.claim("w") is None  # not claimable

    def test_cancel_running_sets_flag(self, queue):
        job = queue.submit("k", {})
        queue.claim("w")
        queue.mark_running(job.job_id, run_id="run-x")
        assert queue.request_cancel(job.job_id) is JobStatus.RUNNING
        assert queue.is_cancel_requested(job.job_id)

    def test_cancel_terminal_is_noop(self, queue):
        job = queue.submit("k", {})
        queue.claim("w")
        queue.finish(job.job_id, JobStatus.COMPLETED)
        assert queue.request_cancel(job.job_id) is JobStatus.COMPLETED
        assert not queue.is_cancel_requested(job.job_id)

    def test_finish_requires_terminal_status(self, queue):
        job = queue.submit("k", {})
        with pytest.raises(JobError, match="terminal"):
            queue.finish(job.job_id, JobStatus.RUNNING)

    def test_requeue_stale_keeps_run_id(self, queue):
        job = queue.submit("k", {})
        queue.claim("w")
        queue.mark_running(job.job_id, run_id="run-abc")
        # backdate the heartbeat as if the worker died an hour ago
        queue.conn.execute(
            "UPDATE jobs SET heartbeat_at = '2000-01-01T00:00:00+00:00' "
            "WHERE job_id = ?", (job.job_id,),
        )
        queue.conn.commit()
        assert queue.requeue_stale(60.0) == [job.job_id]
        requeued = queue.get(job.job_id)
        assert requeued.status is JobStatus.QUEUED
        assert requeued.run_id == "run-abc"  # next worker RESUMES
        assert requeued.worker_id is None

    def test_requeue_stale_ignores_fresh(self, queue):
        job = queue.submit("k", {})
        queue.claim("w")
        assert queue.requeue_stale(3600.0) == []
        assert queue.get(job.job_id).status is JobStatus.CLAIMED


# ---------------------------------------------------------------------------
# Worker integration (fake stages, real everything else)
# ---------------------------------------------------------------------------


class MarkerStage:
    """Writes one marker artifact; completion = artifact registered."""

    def __init__(self, name: str, fail_times: int = 0):
        self.name = name
        self.runs = 0
        self._fail_times = fail_times

    def is_complete(self, ctx) -> bool:
        return f"{self.name}.done" in ctx.manifest.artifacts

    def run(self, ctx) -> None:
        self.runs += 1
        ctx.cancel.raise_if_cancelled()
        if self._fail_times > 0:
            self._fail_times -= 1
            raise RuntimeError(f"{self.name} exploded")
        marker = ctx.run_dir / f"{self.name}.done"
        marker.write_text("ok", encoding="utf-8")
        ctx.register_artifact(
            ArtifactRecord(
                name=f"{self.name}.done", relpath=marker.name,
                schema_version=1, producer=f"fake/{self.name}",
            )
        )


@pytest.fixture()
def image_dir(tmp_path):
    root = tmp_path / "cam_frames"
    root.mkdir()
    rng = np.random.default_rng(7)
    for i in range(4):
        img = rng.integers(0, 255, (24, 32, 3), dtype=np.uint8)
        cv2.imwrite(str(root / f"{i:08d}.png"), img)
    return root


@pytest.fixture()
def service(tmp_path):
    svc = JobService(tmp_path / "jobs.db", tmp_path / "runs", spawn_worker=False)
    yield svc
    svc.queue.close()


def _patch_stages(monkeypatch, stages) -> None:
    from athar.pipeline import setup

    monkeypatch.setattr(setup, "default_stages", lambda: stages)


class TestWorker:
    def test_executes_run_to_completion(self, service, image_dir, tmp_path, monkeypatch):
        _patch_stages(monkeypatch, [MarkerStage("s1"), MarkerStage("s2")])
        job = service.submit_run(videos={"c1": str(image_dir)}, fps=2.0)
        assert run_worker(tmp_path / "jobs.db", once=True, poll_s=0.01) == 0
        done = service.get(job.job_id)
        assert done.status is JobStatus.COMPLETED
        assert done.run_id is not None
        from athar.contracts.store import FilesystemRunStore

        manifest = FilesystemRunStore(tmp_path / "runs").load(done.run_id)
        assert manifest.status is RunStatus.COMPLETED
        assert "s1.done" in manifest.artifacts and "s2.done" in manifest.artifacts
        events = service.events_path(done)
        assert events is not None and events.name == EVENTS_FILENAME
        assert "run_completed" in events.read_text(encoding="utf-8")

    def test_cancel_before_worker_lands_as_cancelled(
        self, service, image_dir, tmp_path, monkeypatch
    ):
        _patch_stages(monkeypatch, [MarkerStage("s1")])
        job = service.submit_run(videos={"c1": str(image_dir)}, fps=2.0)
        # worker hasn't claimed yet -> flag rides along; poll interval 0 makes
        # the first event trip the token deterministically
        service.queue.conn.execute(
            "UPDATE jobs SET cancel_requested = 1 WHERE job_id = ?", (job.job_id,)
        )
        service.queue.conn.commit()
        executors = {"run_pipeline": LocalRunExecutor(cancel_poll_s=0.0)}
        run_worker(tmp_path / "jobs.db", once=True, poll_s=0.01, executors=executors)
        done = service.get(job.job_id)
        assert done.status is JobStatus.CANCELLED
        from athar.contracts.store import FilesystemRunStore

        manifest = FilesystemRunStore(tmp_path / "runs").load(done.run_id)
        assert manifest.status is RunStatus.CANCELLED  # resumable later

    def test_failed_stage_records_error_then_resume_completes(
        self, service, image_dir, tmp_path, monkeypatch
    ):
        s1 = MarkerStage("s1")
        s2 = MarkerStage("s2", fail_times=1)
        _patch_stages(monkeypatch, [s1, s2])
        job = service.submit_run(videos={"c1": str(image_dir)}, fps=2.0)
        run_worker(tmp_path / "jobs.db", once=True, poll_s=0.01)
        failed = service.get(job.job_id)
        assert failed.status is JobStatus.FAILED
        assert "exploded" in failed.error
        assert failed.run_id is not None

        # resubmit as a resume job: s1 skips (artifact present), s2 succeeds
        retry = service.submit_run(videos={}, resume_run_id=failed.run_id)
        run_worker(tmp_path / "jobs.db", once=True, poll_s=0.01)
        assert service.get(retry.job_id).status is JobStatus.COMPLETED
        assert s1.runs == 1  # never re-ran
        assert s2.runs == 2

    def test_unknown_kind_fails_cleanly(self, tmp_path):
        queue = JobQueue(tmp_path / "jobs.db")
        job = queue.submit("teleport", {})
        queue.close()
        run_worker(tmp_path / "jobs.db", once=True, poll_s=0.01)
        check = JobQueue(tmp_path / "jobs.db")
        try:
            done = check.get(job.job_id)
            assert done.status is JobStatus.FAILED
            assert "teleport" in done.error
        finally:
            check.close()

    def test_missing_runs_root_fails_cleanly(self, tmp_path):
        queue = JobQueue(tmp_path / "jobs.db")
        job = queue.submit("run_pipeline", {"videos": {"c1": "x"}})
        queue.close()
        run_worker(tmp_path / "jobs.db", once=True, poll_s=0.01)
        check = JobQueue(tmp_path / "jobs.db")
        try:
            assert check.get(job.job_id).status is JobStatus.FAILED
            assert "runs_root" in check.get(job.job_id).error
        finally:
            check.close()

    def test_idle_once_returns(self, tmp_path):
        assert run_worker(tmp_path / "jobs.db", once=True, poll_s=0.01) == 0


class TestService:
    def test_submit_requires_videos_or_resume(self, service):
        with pytest.raises(ValueError, match="videos or resume"):
            service.submit_run(videos={})

    def test_ensure_worker_spawns_once(self, tmp_path, monkeypatch):
        spawned: list[list[str]] = []

        class FakeProc:
            pid = 4242

            def poll(self):
                return None

        import subprocess as _subprocess

        monkeypatch.setattr(
            _subprocess, "Popen", lambda cmd, **kw: spawned.append(cmd) or FakeProc()
        )
        svc = JobService(tmp_path / "jobs.db", tmp_path / "runs", spawn_worker=False)
        try:
            svc.ensure_worker()
            svc.ensure_worker()  # alive -> no second spawn
            assert len(spawned) == 1
            assert "worker" in spawned[0]
            assert svc.worker_alive()
        finally:
            svc.queue.close()
