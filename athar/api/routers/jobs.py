"""Jobs: submit pipeline runs, watch them, cancel them.

Submission is the ONLY write path into the pipeline from the API; the
worker process executes. Viewer reads, investigator mutates."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.sse import EventSourceResponse

from athar.api import audit
from athar.api.deps import DbDep, InvestigatorUser, RequireViewer, ServicesDep
from athar.api.schemas import CancelOut, JobSubmitRequest
from athar.api.sse import tail_events
from athar.jobs.queue import Job, JobExecutor, JobNotFound, JobStatus

router = APIRouter(prefix="/jobs", tags=["jobs"], dependencies=[RequireViewer])


def _get(services: ServicesDep, job_id: str) -> Job:
    try:
        return services.jobs.get(job_id)
    except JobNotFound:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"job {job_id!r} not found") from None


@router.get("")
def list_jobs(services: ServicesDep, job_status: Optional[str] = None) -> list[Job]:
    parsed = JobStatus(job_status) if job_status else None
    return services.jobs.list(status=parsed)


@router.get("/{job_id}")
def get_job(services: ServicesDep, job_id: str) -> Job:
    return _get(services, job_id)


@router.post("", status_code=status.HTTP_202_ACCEPTED)
def submit_job(
    body: JobSubmitRequest, services: ServicesDep, db: DbDep, user: InvestigatorUser
) -> Job:
    try:
        job = services.jobs.submit_run(
            videos=body.videos,
            profile=body.profile,
            role=body.role,
            fps=body.fps,
            overrides=body.overrides,
            resume_run_id=body.resume_run_id,
            executor=JobExecutor(body.executor),
            priority=body.priority,
        )
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from None
    audit.append(
        db, user.username, "job_submitted",
        job_id=job.job_id, profile=body.profile, role=body.role,
        cameras=sorted(body.videos), resume_run_id=body.resume_run_id,
    )
    return job


@router.post("/{job_id}/cancel")
def cancel_job(
    services: ServicesDep, db: DbDep, user: InvestigatorUser, job_id: str
) -> CancelOut:
    _get(services, job_id)  # 404 before mutating
    result = services.jobs.cancel(job_id)
    audit.append(db, user.username, "job_cancel_requested",
                 job_id=job_id, resulting_status=result.value)
    return CancelOut(job_id=job_id, status=result.value)


@router.get("/{job_id}/events/stream")
def stream_job_events(
    services: ServicesDep, job_id: str, request: Request
) -> EventSourceResponse:
    job = _get(services, job_id)
    path = services.jobs.events_path(job)
    if path is None:  # 404 BEFORE the stream starts
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"job {job_id!r} has no run events yet (status {job.status.value})",
        )
    return EventSourceResponse(tail_events(path, request))
