"""Runs: read-only views over the run store (manifests, events, report,
artifact downloads). Mutations happen through jobs, never here."""

from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import FileResponse
from fastapi.sse import EventSourceResponse

from athar.api.deps import RequireViewer, ServicesDep
from athar.api.schemas import RunSummary
from athar.api.sse import tail_events
from athar.contracts.manifest import RunManifest, RunRole
from athar.contracts.store import RunNotFound
from athar.pipeline.runner import EVENTS_FILENAME

router = APIRouter(prefix="/runs", tags=["runs"], dependencies=[RequireViewer])


def _load(services: ServicesDep, run_id: str) -> RunManifest:
    try:
        return services.store.load(run_id)
    except RunNotFound:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"run {run_id!r} not found") from None


@router.get("")
def list_runs(services: ServicesDep, role: Optional[str] = None) -> list[RunSummary]:
    role_filter = RunRole(role) if role else None
    return [
        RunSummary(
            run_id=m.run_id,
            role=m.role.value,
            status=m.status.value,
            profile_name=m.profile_name,
            created_at=m.created_at,
            config_hash=m.config.config_hash if m.config else None,
            num_artifacts=len(m.artifacts),
            error=m.error,
        )
        for m in services.store.list(role=role_filter)
    ]


@router.get("/{run_id}")
def get_run(services: ServicesDep, run_id: str) -> RunManifest:
    return _load(services, run_id)


@router.get("/{run_id}/events")
def list_events(services: ServicesDep, run_id: str, limit: int = 1000) -> list[dict]:
    _load(services, run_id)
    path = services.store.run_dir(run_id) / EVENTS_FILENAME
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines[-limit:] if line.strip()]


@router.get("/{run_id}/events/stream")
def stream_events(
    services: ServicesDep, run_id: str, request: Request
) -> EventSourceResponse:
    _load(services, run_id)  # 404 BEFORE the stream starts
    path = services.store.run_dir(run_id) / EVENTS_FILENAME
    return EventSourceResponse(tail_events(path, request))


@router.get("/{run_id}/report")
def get_report(services: ServicesDep, run_id: str) -> dict:
    manifest = _load(services, run_id)
    if "package.report" not in manifest.artifacts:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"run {run_id!r} has no package.report artifact (package stage not run)",
        )
    return json.loads(
        services.store.artifact_path(manifest, "package.report").read_text("utf-8")
    )


@router.get("/{run_id}/artifacts/{name}")
def download_artifact(services: ServicesDep, run_id: str, name: str) -> FileResponse:
    manifest = _load(services, run_id)
    if name not in manifest.artifacts:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f"run {run_id!r} has no artifact {name!r}"
        )
    path = services.store.artifact_path(manifest, name).resolve()
    run_dir = services.store.run_dir(run_id).resolve()
    if not path.is_relative_to(run_dir):  # manifest tampering guard
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "artifact path escapes run dir")
    if not path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "artifact file missing on disk")
    return FileResponse(path, filename=path.name)
