"""Small crash-safe in-memory job queue for local backend tasks.

Phase 2a adds the infrastructure only. Eval submission endpoints and concrete
script runners are intentionally left for Phase 2c.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
JOB_DIR = PROJECT_ROOT / "data" / "jobs"
JobHandler = Callable[[dict[str, Any], "Job"], Any]


@dataclass
class Job:
    id: str
    eval_type: str
    payload: dict[str, Any]
    status: str = "queued"
    created_at: str = field(default_factory=lambda: _now_iso())
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    progress: dict[str, Any] = field(default_factory=dict)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobService:
    def __init__(self, job_dir: Path = JOB_DIR) -> None:
        self.job_dir = job_dir
        self.jobs: Dict[str, Job] = {}
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self.handlers: Dict[str, JobHandler] = {}
        self._worker_task: Optional[asyncio.Task[None]] = None

    def load_jobs(self) -> None:
        self.job_dir.mkdir(parents=True, exist_ok=True)
        self.jobs.clear()
        for path in sorted(self.job_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                job = Job(**data)
            except Exception:
                continue
            if job.status == "running":
                job.status = "failed"
                job.finished_at = _now_iso()
                job.error = "Backend restarted before the job finished"
                self._persist(job)
            self.jobs[job.id] = job

    def register_handler(self, eval_type: str, handler: JobHandler) -> None:
        self.handlers[eval_type] = handler

    def start_worker(self) -> None:
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._worker(), name="job-service-worker")

    def submit_job(self, eval_type: str, payload: dict[str, Any], background_tasks: Any | None = None) -> str:
        job_id = f"{eval_type}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        job = Job(id=job_id, eval_type=eval_type, payload=dict(payload))
        self.jobs[job_id] = job
        self._persist(job)
        if background_tasks is not None:
            background_tasks.add_task(self._run_job, job_id)
        else:
            self.queue.put_nowait(job_id)
        return job_id

    def get_status(self, job_id: str) -> Optional[dict[str, Any]]:
        job = self.jobs.get(job_id) or self._load_one(job_id)
        if job is None:
            return None
        return {
            "jobId": job.id,
            "status": job.status,
            "createdAt": job.created_at,
            "startedAt": job.started_at,
            "finishedAt": job.finished_at,
            "error": job.error,
            "progress": job.progress,
        }

    def get_result(self, job_id: str) -> Optional[dict[str, Any]]:
        job = self.jobs.get(job_id) or self._load_one(job_id)
        if job is None:
            return None
        return {"jobId": job.id, "status": job.status, "result": job.result}

    async def _worker(self) -> None:
        while True:
            job_id = await self.queue.get()
            try:
                await self._run_job(job_id)
            finally:
                self.queue.task_done()

    async def _run_job(self, job_id: str) -> None:
        job = self.jobs.get(job_id)
        if job is None:
            return
        job.status = "running"
        job.started_at = _now_iso()
        job.progress = {"stage": "started"}
        self._persist(job)
        try:
            handler = self.handlers.get(job.eval_type)
            if handler is None:
                raise ValueError(f"No job handler registered for eval_type={job.eval_type!r}")
            output = handler(job.payload, job)
            if asyncio.iscoroutine(output):
                output = await output
            job.result = output if isinstance(output, dict) else {"value": output}
            job.status = "succeeded"
            job.progress = {"stage": "finished"}
        except Exception as exc:  # noqa: BLE001 - persisted error is intentionally sanitized
            job.status = "failed"
            job.error = str(exc)
        finally:
            job.finished_at = _now_iso()
            self._persist(job)

    def _persist(self, job: Job) -> None:
        self.job_dir.mkdir(parents=True, exist_ok=True)
        path = self.job_dir / f"{job.id}.json"
        path.write_text(json.dumps(asdict(job), indent=2, sort_keys=True), encoding="utf-8")

    def _load_one(self, job_id: str) -> Optional[Job]:
        path = self.job_dir / f"{job_id}.json"
        if not path.exists():
            return None
        try:
            job = Job(**json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            return None
        self.jobs[job.id] = job
        return job


job_service = JobService()


def submit_job(eval_type: str, payload: dict[str, Any], background_tasks: Any | None = None) -> str:
    return job_service.submit_job(eval_type, payload, background_tasks)


def get_status(job_id: str) -> Optional[dict[str, Any]]:
    return job_service.get_status(job_id)


def get_result(job_id: str) -> Optional[dict[str, Any]]:
    return job_service.get_result(job_id)
