"""JobService: the facade the API layer talks to.

Owns the queue handle, validates submissions into typed payloads, resolves
the event pipe for a job (its run's ``events.jsonl``), and supervises one
local worker subprocess. Routers never touch sqlite or subprocess handles
directly.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence

from athar.jobs.queue import (
    Job,
    JobExecutor,
    JobQueue,
    JobStatus,
)
from athar.pipeline.runner import EVENTS_FILENAME

logger = logging.getLogger(__name__)

RUN_PIPELINE = "run_pipeline"


class JobService:
    def __init__(
        self,
        queue_path: Path | str,
        runs_root: Path | str,
        spawn_worker: bool = True,
    ) -> None:
        self.queue = JobQueue(queue_path)
        self.runs_root = Path(runs_root)
        self._spawn_worker = spawn_worker
        self._worker: Optional[subprocess.Popen] = None

    # -- submissions -----------------------------------------------------------
    def submit_run(
        self,
        videos: dict[str, str],
        profile: str = "multiclass",
        role: str = "gallery",
        fps: Optional[float] = None,
        overrides: Optional[Sequence[str]] = None,
        resume_run_id: Optional[str] = None,
        executor: JobExecutor = JobExecutor.LOCAL,
        priority: int = 0,
    ) -> Job:
        if not videos and not resume_run_id:
            raise ValueError("submit_run needs videos or resume_run_id")
        payload = {
            "runs_root": str(self.runs_root),
            "profile": profile,
            "videos": videos,
            "role": role,
            "fps": fps,
            "overrides": list(overrides or []),
            "resume_run_id": resume_run_id,
        }
        job = self.queue.submit(RUN_PIPELINE, payload, executor=executor,
                                priority=priority)
        if self._spawn_worker:
            self.ensure_worker()
        return job

    # -- queries ---------------------------------------------------------------
    def get(self, job_id: str) -> Job:
        return self.queue.get(job_id)

    def list(self, status: Optional[JobStatus] = None, limit: int = 100) -> list[Job]:
        return self.queue.list(status=status, limit=limit)

    def cancel(self, job_id: str) -> JobStatus:
        return self.queue.request_cancel(job_id)

    def events_path(self, job: Job) -> Optional[Path]:
        """The typed event pipe: the job's run's events.jsonl (None until
        the worker has attached a run)."""
        if not job.run_id:
            return None
        path = self.runs_root / job.run_id / EVENTS_FILENAME
        return path if path.exists() else None

    # -- worker supervision ------------------------------------------------------
    def worker_alive(self) -> bool:
        return self._worker is not None and self._worker.poll() is None

    def ensure_worker(self) -> None:
        """Start the local worker subprocess if none is running."""
        if self.worker_alive():
            return
        cmd = [
            sys.executable, "-m", "athar.cli.main", "worker",
            "--queue", str(self.queue.db_path),
        ]
        self._worker = subprocess.Popen(cmd)
        logger.info("started worker subprocess pid=%d", self._worker.pid)

    def shutdown(self, timeout_s: float = 10.0) -> None:
        if self._worker is not None and self._worker.poll() is None:
            self._worker.terminate()
            try:
                self._worker.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                self._worker.kill()
        self._worker = None
        self.queue.close()
