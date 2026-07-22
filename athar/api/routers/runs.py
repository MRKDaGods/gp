"""Runs: read-only views over the run store (manifests, events, report,
artifact downloads). Mutations happen through jobs, never here."""

from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, HTTPException, Request, Response, status
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.sse import EventSourceResponse

from athar.api import audit
from athar.api.deps import CurrentUser, DbDep, RequireViewer, ServicesDep
from athar.api.schemas import RunSummary
from athar.api.sse import tail_events
from athar.contracts.manifest import RunManifest, RunRole
from athar.contracts.store import RunNotFound
from athar.pipeline.runner import EVENTS_FILENAME
from athar.reporting import (
    ReportError,
    html_to_pdf,
    load_weight_shas,
    models_from_config,
    render_report_html,
)

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


def _report_html(services: ServicesDep, run_id: str, locale: str) -> str:
    manifest = _load(services, run_id)
    if "package.report" not in manifest.artifacts:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"run {run_id!r} has no package.report artifact (package stage not run)",
        )
    report = json.loads(
        services.store.artifact_path(manifest, "package.report").read_text("utf-8")
    )
    weight_shas = load_weight_shas(services.settings.weights_manifest)
    models = (
        models_from_config(manifest.config.values, weight_shas)
        if manifest.config
        else []
    )
    return render_report_html(
        report,
        models=models,
        run_dir=services.store.run_dir(run_id),
        locale=locale if locale in ("ar", "en") else "ar",
    )


@router.get("/{run_id}/report.html", response_class=HTMLResponse)
def export_report_html(
    services: ServicesDep, run_id: str, db: DbDep, user: CurrentUser,
    locale: str = "ar",
) -> HTMLResponse:
    """Chromium-free preview of the exact document the PDF prints."""
    html = _report_html(services, run_id, locale)
    audit.append(
        db, user.username, "report_exported",
        run_id=run_id, locale=locale, fmt="html",
    )
    return HTMLResponse(html)


@router.get("/{run_id}/report.pdf")
def export_report_pdf(
    services: ServicesDep, run_id: str, db: DbDep, user: CurrentUser,
    locale: str = "ar",
) -> Response:
    html = _report_html(services, run_id, locale)
    try:
        pdf = html_to_pdf(html)
    except ReportError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from None
    audit.append(
        db, user.username, "report_exported",
        run_id=run_id, locale=locale, fmt="pdf",
    )
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="athar-report-{run_id}.pdf"'
        },
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
