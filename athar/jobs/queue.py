"""SQLite job queue (D19): own DB file, WAL, ``UPDATE...RETURNING`` claims.

Jobs are rows; workers are separate processes that atomically claim one
queued job at a time (the claim UPDATE's subquery re-evaluates under
SQLite's writer serialization, so two workers can never claim the same
job). The **typed event pipe** for a pipeline job is the run's
``events.jsonl`` — the job row carries ``run_id`` and consumers tail the
run's event log; there is no second event store to drift (the v1 lesson).

Crash recovery: workers heartbeat while executing; ``requeue_stale``
returns abandoned jobs to the queue with their ``run_id`` intact, so the
next worker RESUMES the interrupted run instead of starting over.
"""

from __future__ import annotations

import enum
import json
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from athar.core.ids import new_job_id

DEFAULT_QUEUE_DB = Path("data/jobs/jobs.db")


class JobError(RuntimeError):
    pass


class JobNotFound(JobError):
    pass


class JobStatus(str, enum.Enum):
    QUEUED = "queued"
    CLAIMED = "claimed"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_STATUSES = {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}


class JobExecutor(str, enum.Enum):
    """Where a job runs (D13). Sensitive footage never leaves premises:
    the executor is a property of the JOB, chosen at submit time."""

    LOCAL = "local"
    KAGGLE = "kaggle"


class Job(BaseModel):
    job_id: str = Field(default_factory=new_job_id)
    kind: str
    payload: dict[str, Any] = Field(default_factory=dict)
    executor: JobExecutor = JobExecutor.LOCAL
    status: JobStatus = JobStatus.QUEUED
    priority: int = 0
    run_id: Optional[str] = None
    worker_id: Optional[str] = None
    error: Optional[str] = None
    cancel_requested: bool = False
    attempt: int = 0
    created_at: Optional[str] = None
    claimed_at: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    heartbeat_at: Optional[str] = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobQueue:
    """One connection per instance; safe for multi-threaded use (internal
    lock) and multi-process use (WAL + busy_timeout + atomic claims)."""

    def __init__(self, db_path: Path | str = DEFAULT_QUEUE_DB) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock, self.conn:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA busy_timeout=5000")
            self.conn.execute(
                """CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    executor TEXT NOT NULL,
                    status TEXT NOT NULL,
                    priority INTEGER NOT NULL DEFAULT 0,
                    run_id TEXT,
                    worker_id TEXT,
                    error TEXT,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    attempt INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    claimed_at TEXT,
                    started_at TEXT,
                    finished_at TEXT,
                    heartbeat_at TEXT
                )"""
            )
            self.conn.execute(
                """CREATE INDEX IF NOT EXISTS idx_jobs_claim
                   ON jobs(status, executor, priority DESC, created_at)"""
            )

    def close(self) -> None:
        self.conn.close()

    @staticmethod
    def _to_job(row: sqlite3.Row) -> Job:
        data = dict(row)
        data["payload"] = json.loads(data["payload"])
        data["cancel_requested"] = bool(data["cancel_requested"])
        return Job.model_validate(data)

    # -- producer side -------------------------------------------------------
    def submit(
        self,
        kind: str,
        payload: dict[str, Any],
        executor: JobExecutor = JobExecutor.LOCAL,
        priority: int = 0,
    ) -> Job:
        job = Job(
            kind=kind, payload=payload, executor=executor,
            priority=priority, created_at=_now(),
        )
        with self._lock, self.conn:
            self.conn.execute(
                """INSERT INTO jobs
                   (job_id, kind, payload, executor, status, priority,
                    cancel_requested, attempt, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, 0, 0, ?)""",
                (job.job_id, job.kind, json.dumps(job.payload),
                 job.executor.value, job.status.value, job.priority,
                 job.created_at),
            )
        return job

    def get(self, job_id: str) -> Job:
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        if row is None:
            raise JobNotFound(job_id)
        return self._to_job(row)

    def list(
        self,
        status: Optional[JobStatus] = None,
        limit: int = 100,
    ) -> list[Job]:
        query = "SELECT * FROM jobs"
        params: list = []
        if status is not None:
            query += " WHERE status = ?"
            params.append(status.value)
        query += " ORDER BY created_at DESC, job_id DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self.conn.execute(query, params).fetchall()
        return [self._to_job(r) for r in rows]

    # -- worker side -------------------------------------------------------------
    def claim(
        self, worker_id: str, executor: JobExecutor = JobExecutor.LOCAL
    ) -> Optional[Job]:
        """Atomically claim the highest-priority oldest queued job."""
        now = _now()
        with self._lock, self.conn:
            row = self.conn.execute(
                """UPDATE jobs
                   SET status = ?, worker_id = ?, claimed_at = ?,
                       heartbeat_at = ?, attempt = attempt + 1
                   WHERE job_id = (
                       SELECT job_id FROM jobs
                       WHERE status = ? AND executor = ?
                       ORDER BY priority DESC, created_at, job_id
                       LIMIT 1
                   )
                   RETURNING *""",
                (JobStatus.CLAIMED.value, worker_id, now, now,
                 JobStatus.QUEUED.value, executor.value),
            ).fetchone()
        return self._to_job(row) if row else None

    def mark_running(self, job_id: str, run_id: Optional[str] = None) -> None:
        with self._lock, self.conn:
            changed = self.conn.execute(
                """UPDATE jobs
                   SET status = ?, started_at = COALESCE(started_at, ?),
                       heartbeat_at = ?, run_id = COALESCE(?, run_id)
                   WHERE job_id = ?""",
                (JobStatus.RUNNING.value, _now(), _now(), run_id, job_id),
            ).rowcount
        if not changed:
            raise JobNotFound(job_id)

    def heartbeat(self, job_id: str) -> None:
        with self._lock, self.conn:
            self.conn.execute(
                "UPDATE jobs SET heartbeat_at = ? WHERE job_id = ?", (_now(), job_id)
            )

    def is_cancel_requested(self, job_id: str) -> bool:
        with self._lock:
            row = self.conn.execute(
                "SELECT cancel_requested FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        if row is None:
            raise JobNotFound(job_id)
        return bool(row["cancel_requested"])

    def finish(
        self,
        job_id: str,
        status: JobStatus,
        error: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> None:
        if status not in TERMINAL_STATUSES:
            raise JobError(f"finish() requires a terminal status, got {status.value}")
        with self._lock, self.conn:
            changed = self.conn.execute(
                """UPDATE jobs
                   SET status = ?, error = ?, finished_at = ?,
                       run_id = COALESCE(?, run_id)
                   WHERE job_id = ?""",
                (status.value, error, _now(), run_id, job_id),
            ).rowcount
        if not changed:
            raise JobNotFound(job_id)

    # -- control plane ---------------------------------------------------------
    def request_cancel(self, job_id: str) -> JobStatus:
        """Queued jobs cancel immediately; claimed/running jobs get the flag
        (the worker's event sink turns it into a CancellationToken).
        Returns the job's status after the request."""
        with self._lock, self.conn:
            job = self.get(job_id)
            if job.status is JobStatus.QUEUED:
                self.conn.execute(
                    """UPDATE jobs SET status = ?, cancel_requested = 1,
                       finished_at = ? WHERE job_id = ? AND status = ?""",
                    (JobStatus.CANCELLED.value, _now(), job_id,
                     JobStatus.QUEUED.value),
                )
            elif job.status in (JobStatus.CLAIMED, JobStatus.RUNNING):
                self.conn.execute(
                    "UPDATE jobs SET cancel_requested = 1 WHERE job_id = ?",
                    (job_id,),
                )
        return self.get(job_id).status

    def requeue_stale(self, heartbeat_timeout_s: float) -> list[str]:
        """Return abandoned claimed/running jobs (dead worker) to the queue.
        ``run_id`` is kept so the next worker resumes the interrupted run."""
        cutoff = (
            datetime.now(timezone.utc) - timedelta(seconds=heartbeat_timeout_s)
        ).isoformat()
        with self._lock, self.conn:
            rows = self.conn.execute(
                """UPDATE jobs
                   SET status = ?, worker_id = NULL, claimed_at = NULL,
                       heartbeat_at = NULL
                   WHERE status IN (?, ?) AND heartbeat_at < ?
                   RETURNING job_id""",
                (JobStatus.QUEUED.value, JobStatus.CLAIMED.value,
                 JobStatus.RUNNING.value, cutoff),
            ).fetchall()
        return [r["job_id"] for r in rows]
