"""Job worker: claims queued jobs and executes them in THIS process.

Deployment runs one or more worker processes (``athar worker``) next to
the API; the queue's atomic claims coordinate them. Executors are pluggable
by job kind — ``run_pipeline`` executes the offline DAG locally; a Kaggle
executor (D13: non-sensitive training/eval jobs only) plugs into the same
seam when the training campaign lands.

Cancellation: the API sets ``cancel_requested`` on the row; the executor's
event sink polls it (throttled) and trips the runner's CancellationToken —
the run ends CANCELLED with its checkpoints intact, resumable later.
"""

from __future__ import annotations

import logging
import os
import socket
import time
from pathlib import Path
from typing import Optional, Protocol

from athar.contracts.manifest import RunStatus
from athar.jobs.queue import Job, JobExecutor, JobQueue, JobStatus

logger = logging.getLogger(__name__)

DEFAULT_POLL_S = 1.0
DEFAULT_CANCEL_POLL_S = 1.0
DEFAULT_STALE_TIMEOUT_S = 300.0


class ExecutorProtocol(Protocol):
    def execute(self, job: Job, queue: JobQueue) -> RunStatus: ...


class LocalRunExecutor:
    """Executes ``run_pipeline`` jobs through the shared run setup + DAG
    runner. Payload::

        {
          "runs_root": "data/runs",
          "profile": "multiclass",
          "videos": {"c017": "path/to/video-or-image-dir", ...},
          "role": "gallery",
          "fps": 2.0,                  # image dirs only
          "overrides": ["k.e.y=v"],
          "resume_run_id": "run-..."   # resume instead of create
        }

    If the JOB row already carries a run_id (a requeued crash), that run is
    resumed regardless of payload — work done before the crash is kept.
    """

    def __init__(self, cancel_poll_s: float = DEFAULT_CANCEL_POLL_S) -> None:
        self.cancel_poll_s = cancel_poll_s

    def execute(self, job: Job, queue: JobQueue) -> RunStatus:
        from athar.contracts.store import FilesystemRunStore
        from athar.pipeline import setup
        from athar.pipeline.runner import CancellationToken, PipelineRunner

        payload = job.payload
        runs_root = payload.get("runs_root")
        if not runs_root:
            raise ValueError("run_pipeline payload requires 'runs_root'")
        store = FilesystemRunStore(runs_root)
        profile_name = payload.get("profile", "multiclass")
        resume_id = job.run_id or payload.get("resume_run_id")
        if resume_id:
            manifest, profile = setup.resume_run(store, resume_id, profile_name)
        else:
            manifest, profile = setup.create_run(
                profile_name=profile_name,
                videos=payload.get("videos") or {},
                role=payload.get("role", "gallery"),
                fps=payload.get("fps"),
                overrides=payload.get("overrides"),
            )
        queue.mark_running(job.job_id, run_id=manifest.run_id)

        token = CancellationToken()
        last_poll = 0.0

        def control_sink(_event) -> None:
            nonlocal last_poll
            now = time.monotonic()
            if now - last_poll >= self.cancel_poll_s:
                last_poll = now
                queue.heartbeat(job.job_id)
                if queue.is_cancel_requested(job.job_id):
                    token.cancel()

        runner = PipelineRunner(store, setup.default_stages(), extra_sinks=[control_sink])
        result = runner.run(manifest, profile, cancel=token)  # raises on stage failure
        return result.status


class KaggleExecutor:
    """Seam for D13: push non-sensitive jobs to a Kaggle kernel and poll.
    Lands with the training campaign (Phase 6); the queue already routes by
    ``executor`` so kaggle jobs wait for a kaggle-capable worker."""

    def execute(self, job: Job, queue: JobQueue) -> RunStatus:  # pragma: no cover
        raise NotImplementedError(
            "kaggle executor arrives with the Phase 6 training campaign; "
            "submit with executor=local for now"
        )


def default_executors() -> dict[str, ExecutorProtocol]:
    return {"run_pipeline": LocalRunExecutor()}


def execute_one(
    queue: JobQueue, job: Job, executors: dict[str, ExecutorProtocol]
) -> JobStatus:
    """Run one claimed job to a terminal status (always finishes the row)."""
    executor = executors.get(job.kind)
    if executor is None:
        queue.finish(job.job_id, JobStatus.FAILED,
                     error=f"no executor for job kind {job.kind!r}")
        return JobStatus.FAILED
    try:
        run_status = executor.execute(job, queue)
    except Exception as exc:  # noqa: BLE001 — job outcome, not worker crash
        logger.exception("job %s failed", job.job_id)
        queue.finish(job.job_id, JobStatus.FAILED, error=str(exc))
        return JobStatus.FAILED
    status = {
        RunStatus.COMPLETED: JobStatus.COMPLETED,
        RunStatus.CANCELLED: JobStatus.CANCELLED,
    }.get(run_status, JobStatus.FAILED)
    error = None if status is not JobStatus.FAILED else (
        f"run ended {run_status.value!r}"
    )
    queue.finish(job.job_id, status, error=error)
    return status


def run_worker(
    queue_path: Path | str,
    once: bool = False,
    poll_s: float = DEFAULT_POLL_S,
    worker_id: Optional[str] = None,
    executors: Optional[dict[str, ExecutorProtocol]] = None,
    executor_kind: JobExecutor = JobExecutor.LOCAL,
    stale_timeout_s: float = DEFAULT_STALE_TIMEOUT_S,
) -> int:
    """Worker loop: requeue stale -> claim -> execute. ``once`` processes at
    most one job then returns (tests, cron); otherwise loops forever."""
    queue = JobQueue(queue_path)
    worker = worker_id or f"{socket.gethostname()}-{os.getpid()}"
    executors = executors if executors is not None else default_executors()
    logger.info("worker %s polling %s", worker, queue.db_path)
    try:
        while True:
            stale = queue.requeue_stale(stale_timeout_s)
            if stale:
                logger.warning("requeued stale jobs: %s", ", ".join(stale))
            job = queue.claim(worker, executor=executor_kind)
            if job is None:
                if once:
                    return 0
                time.sleep(poll_s)
                continue
            logger.info("claimed %s (%s)", job.job_id, job.kind)
            execute_one(queue, job, executors)
            if once:
                return 0
    finally:
        queue.close()
