"""Runs: read-only views over the run store (manifests, events, report,
artifact downloads). Mutations happen through jobs, never here."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Request, Response, status
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.sse import EventSourceResponse

from athar.api import audit
from athar.api.deps import CurrentUser, DbDep, RequireViewer, ServicesDep
from athar.api.schemas import (
    RunSummary,
    TimelineCameraOut,
    TimelineIdentityOut,
    TimelineMemberOut,
    TimelineOut,
)
from athar.api.sse import tail_events
from athar.contracts.manifest import RunManifest, RunRole, VideoInput
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
    return _load_report(services, manifest)


def _report_html(services: ServicesDep, run_id: str, locale: str) -> str:
    manifest = _load(services, run_id)
    report = _load_report(services, manifest)
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


def _load_report(services: ServicesDep, manifest: RunManifest) -> dict:
    if "package.report" not in manifest.artifacts:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"run {manifest.run_id!r} has no package.report artifact "
            "(package stage not run)",
        )
    return json.loads(
        services.store.artifact_path(manifest, "package.report").read_text("utf-8")
    )


def _evidence_path(video: VideoInput, services: ServicesDep, manifest: RunManifest):
    """Playable evidence for a camera: the normalized copy if the run made
    one, else the original — which may legitimately be absent (run imported
    from another box)."""
    if video.normalized_artifact and video.normalized_artifact in manifest.artifacts:
        return services.store.artifact_path(manifest, video.normalized_artifact)
    return Path(video.original_path)


@router.get("/{run_id}/timeline")
def get_timeline(services: ServicesDep, run_id: str) -> TimelineOut:
    """Cross-camera timeline: per-camera scene-clock coverage + identity
    spans, derived from package.report and the run's time base."""
    manifest = _load(services, run_id)
    report = _load_report(services, manifest)

    cameras: list[TimelineCameraOut] = []
    on_disk: dict[str, bool] = {}
    span_end = 0.0
    for video in manifest.inputs:
        timebase = manifest.timebase.cameras.get(video.camera_id)
        offset = timebase.offset_s if timebase else 0.0
        scene_end = offset + video.duration_s if video.duration_s else None
        on_disk[video.camera_id] = _evidence_path(video, services, manifest).is_file()
        cameras.append(
            TimelineCameraOut(
                camera_id=video.camera_id,
                duration_s=video.duration_s,
                fps=video.fps,
                scene_start_s=offset,
                scene_end_s=scene_end,
                timebase_source=timebase.source.value if timebase else "assumed",
                timebase_confidence=timebase.confidence if timebase else 0.0,
                video_on_disk=on_disk[video.camera_id],
            )
        )
        if scene_end:
            span_end = max(span_end, scene_end)

    run_dir = services.store.run_dir(run_id)
    identities: list[TimelineIdentityOut] = []
    for identity in report.get("identities", []):
        members = [
            TimelineMemberOut(
                camera_id=m["camera_id"],
                track_id=m["track_id"],
                start_s=m.get("start_ts_scene_s"),
                end_s=m.get("end_ts_scene_s"),
                has_thumbnail=bool(m.get("thumbnail")),
                # footage on the box, or a clip pre-cut at package time
                clip_available=on_disk.get(m["camera_id"], False)
                or bool(m.get("clip") and (run_dir / m["clip"]).is_file()),
            )
            for m in identity["members"]
        ]
        for m in members:
            if m.end_s is not None:
                span_end = max(span_end, m.end_s)
        identities.append(
            TimelineIdentityOut(
                global_id=identity["global_id"],
                entity_class=identity["entity_class"],
                confidence=identity.get("confidence"),
                evidence=identity.get("evidence") or {},
                cross_camera=identity["cross_camera"],
                members=members,
            )
        )
    return TimelineOut(
        run_id=run_id, span_start_s=0.0, span_end_s=span_end,
        cameras=cameras, identities=identities,
    )


@router.get("/{run_id}/thumbs/{camera_id}/{track_id}")
def get_thumbnail(
    services: ServicesDep, run_id: str, camera_id: str, track_id: int
) -> FileResponse:
    _load(services, run_id)
    run_dir = services.store.run_dir(run_id).resolve()
    path = (run_dir / "thumbs" / camera_id / f"{track_id}.jpg").resolve()
    if not path.is_relative_to(run_dir):  # camera_id traversal guard
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "invalid camera id")
    if not path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "thumbnail not found")
    return FileResponse(path, media_type="image/jpeg")


@router.get("/{run_id}/clips/{camera_id}")
def get_clip(
    services: ServicesDep,
    run_id: str,
    camera_id: str,
    db: DbDep,
    user: CurrentUser,
    start_s: float,
    end_s: float,
) -> FileResponse:
    """Evidence clip for a scene-clock span of one camera, transcoded on
    demand (cached under the run dir)."""
    from athar.serving.clips import ClipError, cached_clip_path, clip_for_span

    manifest = _load(services, run_id)
    video = next((v for v in manifest.inputs if v.camera_id == camera_id), None)
    if video is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f"run has no camera {camera_id!r}"
        )
    cached = cached_clip_path(
        services.store.run_dir(run_id), camera_id, start_s, end_s,
        services.settings.clip_pad_s,
    )
    if cached.is_file():  # pre-cut at package time or a prior request —
        # serveable even when the source footage has left the box
        audit.append(
            db, user.username, "clip_exported",
            run_id=run_id, camera_id=camera_id, start_s=start_s, end_s=end_s,
        )
        return FileResponse(cached, media_type="video/mp4", filename=cached.name)
    source = _evidence_path(video, services, manifest)
    if not source.is_file():
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"evidence video for {camera_id!r} is not on disk "
            "(run imported without its source footage)",
        )
    timebase = manifest.timebase.cameras.get(camera_id)
    if timebase is None:
        raise HTTPException(
            status.HTTP_409_CONFLICT, f"run has no time base for {camera_id!r}"
        )
    try:
        path = clip_for_span(
            services.store.run_dir(run_id),
            source,
            camera_id,
            timebase,
            start_s,
            end_s,
            pad_s=services.settings.clip_pad_s,
            max_duration_s=services.settings.clip_max_duration_s,
        )
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from None
    except ClipError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from None
    audit.append(
        db, user.username, "clip_exported",
        run_id=run_id, camera_id=camera_id, start_s=start_s, end_s=end_s,
    )
    return FileResponse(path, media_type="video/mp4", filename=path.name)


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
