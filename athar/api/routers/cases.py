"""Case workspace: investigation cases, targets, and hypothesis decisions.

Need-to-know scoping: a case is visible to its OWNER and to admins only.
Anyone else gets 404 (not 403) so the API never confirms that a case
exists. Reads need any authenticated user (a viewer sees only cases they
own — in practice none, since creating one requires investigator); every
mutation needs investigator+ AND access, and lands in the audit chain.

Evidence integrity: a hypothesis may only reference a run that is attached
to the case, and a tracklet only ever joins a target's confirmed members
through an attributed confirm decision — never silently (D7).
"""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Response, status
from fastapi.responses import HTMLResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from athar.api import audit
from athar.api.db import (
    AuditRow,
    CaseRow,
    CaseRunRow,
    HypothesisRow,
    Role,
    TargetMemberRow,
    TargetRow,
    UserRow,
    utcnow,
)
from athar.api.deps import CurrentUser, DbDep, InvestigatorUser, ServicesDep
from athar.api.schemas import (
    AttachRunRequest,
    CaseCreateRequest,
    CaseDetail,
    CaseRunOut,
    CaseSummary,
    CaseUpdateRequest,
    DecideRequest,
    HypothesisCreateRequest,
    HypothesisOut,
    TargetCreateRequest,
    TargetOut,
    TrackRefOut,
)
from athar.contracts.store import RunNotFound
from athar.core.ids import new_case_id, new_target_id

router = APIRouter(prefix="/cases", tags=["cases"])


# --------------------------------------------------------------------------
# scoping + loading helpers


def _can_access(case: CaseRow, user: UserRow) -> bool:
    return case.owner == user.username or Role(user.role).covers(Role.ADMIN)


def _get_case(db: Session, case_id: str, user: UserRow) -> CaseRow:
    case = db.get(CaseRow, case_id)
    if case is None or not _can_access(case, user):
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"case {case_id!r} not found")
    return case


def _get_target(db: Session, case: CaseRow, target_id: str) -> TargetRow:
    target = db.get(TargetRow, target_id)
    if target is None or target.case_id != case.case_id:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f"target {target_id!r} not found"
        )
    return target


def _touch(case: CaseRow) -> None:
    case.updated_at = utcnow()


# --------------------------------------------------------------------------
# serializers


def _hypothesis_out(row: HypothesisRow) -> HypothesisOut:
    return HypothesisOut(
        hypothesis_id=row.id,
        kind=row.kind,
        run_id=row.run_id,
        camera_id=row.camera_id,
        track_id=row.track_id,
        raw_score=row.raw_score,
        probability=row.probability,
        stream=row.stream,
        status=row.status,
        proposed_by=row.proposed_by,
        created_at=row.created_at,
        decided_by=row.decided_by,
        decided_at=row.decided_at,
    )


def _target_out(db: Session, target: TargetRow) -> TargetOut:
    members = db.scalars(
        select(TargetMemberRow)
        .where(TargetMemberRow.target_id == target.target_id)
        .order_by(TargetMemberRow.id)
    ).all()
    hypotheses = db.scalars(
        select(HypothesisRow)
        .where(HypothesisRow.target_id == target.target_id)
        .order_by(HypothesisRow.id)
    ).all()
    return TargetOut(
        target_id=target.target_id,
        label=target.label,
        created_by=target.created_by,
        created_at=target.created_at,
        members=[
            TrackRefOut(run_id=m.run_id, camera_id=m.camera_id, track_id=m.track_id)
            for m in members
        ],
        hypotheses=[_hypothesis_out(h) for h in hypotheses],
    )


def _case_detail(db: Session, case: CaseRow) -> CaseDetail:
    runs = db.scalars(
        select(CaseRunRow)
        .where(CaseRunRow.case_id == case.case_id)
        .order_by(CaseRunRow.id)
    ).all()
    targets = db.scalars(
        select(TargetRow)
        .where(TargetRow.case_id == case.case_id)
        .order_by(TargetRow.created_at, TargetRow.target_id)
    ).all()
    return CaseDetail(
        case_id=case.case_id,
        title=case.title,
        status=case.status,
        owner=case.owner,
        created_at=case.created_at,
        updated_at=case.updated_at,
        runs=[
            CaseRunOut(
                run_id=r.run_id, role=r.role,
                attached_by=r.attached_by, attached_at=r.attached_at,
            )
            for r in runs
        ],
        targets=[_target_out(db, t) for t in targets],
    )


# --------------------------------------------------------------------------
# cases


@router.post("", status_code=status.HTTP_201_CREATED)
def create_case(body: CaseCreateRequest, db: DbDep, user: InvestigatorUser) -> CaseDetail:
    case = CaseRow(case_id=new_case_id(), title=body.title, owner=user.username)
    db.add(case)
    db.flush()
    audit.append(db, user.username, "case_created", case_id=case.case_id, title=body.title)
    return _case_detail(db, case)


@router.get("")
def list_cases(db: DbDep, user: CurrentUser) -> list[CaseSummary]:
    query = select(CaseRow).order_by(CaseRow.created_at.desc(), CaseRow.case_id.desc())
    if not Role(user.role).covers(Role.ADMIN):
        query = query.where(CaseRow.owner == user.username)
    cases = db.scalars(query).all()
    run_counts = dict(
        db.execute(
            select(CaseRunRow.case_id, func.count()).group_by(CaseRunRow.case_id)
        ).all()
    )
    target_counts = dict(
        db.execute(
            select(TargetRow.case_id, func.count()).group_by(TargetRow.case_id)
        ).all()
    )
    return [
        CaseSummary(
            case_id=c.case_id,
            title=c.title,
            status=c.status,
            owner=c.owner,
            num_runs=run_counts.get(c.case_id, 0),
            num_targets=target_counts.get(c.case_id, 0),
            created_at=c.created_at,
            updated_at=c.updated_at,
        )
        for c in cases
    ]


@router.get("/{case_id}")
def get_case(case_id: str, db: DbDep, user: CurrentUser) -> CaseDetail:
    return _case_detail(db, _get_case(db, case_id, user))


@router.patch("/{case_id}")
def update_case(
    case_id: str, body: CaseUpdateRequest, db: DbDep, user: InvestigatorUser
) -> CaseDetail:
    case = _get_case(db, case_id, user)
    changes: dict = {}
    if body.title is not None and body.title != case.title:
        changes["title"] = body.title
        case.title = body.title
    if body.status is not None and body.status != case.status:
        changes["status"] = body.status
        case.status = body.status
    if changes:
        _touch(case)
        audit.append(db, user.username, "case_updated", case_id=case.case_id, **changes)
    return _case_detail(db, case)


# --------------------------------------------------------------------------
# evidence runs


@router.post("/{case_id}/runs", status_code=status.HTTP_201_CREATED)
def attach_run(
    case_id: str,
    body: AttachRunRequest,
    services: ServicesDep,
    db: DbDep,
    user: InvestigatorUser,
) -> CaseDetail:
    case = _get_case(db, case_id, user)
    try:
        manifest = services.store.load(body.run_id)
    except RunNotFound:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f"run {body.run_id!r} not found"
        ) from None
    exists = db.scalar(
        select(CaseRunRow).where(
            CaseRunRow.case_id == case.case_id, CaseRunRow.run_id == body.run_id
        )
    )
    if exists is not None:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"run {body.run_id!r} is already attached to case {case_id!r}",
        )
    db.add(
        CaseRunRow(
            case_id=case.case_id,
            run_id=body.run_id,
            role=manifest.role.value,
            attached_by=user.username,
        )
    )
    _touch(case)
    audit.append(
        db, user.username, "case_run_attached",
        case_id=case.case_id, run_id=body.run_id, role=manifest.role.value,
    )
    return _case_detail(db, case)


@router.delete("/{case_id}/runs/{run_id}", status_code=status.HTTP_204_NO_CONTENT)
def detach_run(
    case_id: str, run_id: str, db: DbDep, user: InvestigatorUser
) -> None:
    case = _get_case(db, case_id, user)
    row = db.scalar(
        select(CaseRunRow).where(
            CaseRunRow.case_id == case.case_id, CaseRunRow.run_id == run_id
        )
    )
    if row is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"run {run_id!r} is not attached to case {case_id!r}",
        )
    db.delete(row)
    _touch(case)
    audit.append(
        db, user.username, "case_run_detached", case_id=case.case_id, run_id=run_id
    )


# --------------------------------------------------------------------------
# targets + hypotheses


@router.post("/{case_id}/targets", status_code=status.HTTP_201_CREATED)
def create_target(
    case_id: str, body: TargetCreateRequest, db: DbDep, user: InvestigatorUser
) -> TargetOut:
    case = _get_case(db, case_id, user)
    target = TargetRow(
        target_id=new_target_id(),
        case_id=case.case_id,
        label=body.label,
        created_by=user.username,
    )
    db.add(target)
    db.flush()
    _touch(case)
    audit.append(
        db, user.username, "target_created",
        case_id=case.case_id, target_id=target.target_id, label=body.label,
    )
    return _target_out(db, target)


@router.post(
    "/{case_id}/targets/{target_id}/hypotheses", status_code=status.HTTP_201_CREATED
)
def propose_hypothesis(
    case_id: str,
    target_id: str,
    body: HypothesisCreateRequest,
    db: DbDep,
    user: InvestigatorUser,
) -> HypothesisOut:
    case = _get_case(db, case_id, user)
    target = _get_target(db, case, target_id)
    attached = db.scalar(
        select(CaseRunRow).where(
            CaseRunRow.case_id == case.case_id, CaseRunRow.run_id == body.run_id
        )
    )
    if attached is None:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"run {body.run_id!r} is not attached to case {case_id!r}; "
            "attach the evidence run before referencing its tracklets",
        )
    row = HypothesisRow(
        target_id=target.target_id,
        kind=body.kind,
        run_id=body.run_id,
        camera_id=body.camera_id,
        track_id=body.track_id,
        raw_score=body.raw_score,
        probability=body.probability,
        stream=body.stream,
        proposed_by=user.username,
    )
    db.add(row)
    db.flush()
    _touch(case)
    audit.append(
        db, user.username, "hypothesis_proposed",
        case_id=case.case_id, target_id=target.target_id, hypothesis_id=row.id,
        kind=body.kind, run_id=body.run_id, camera_id=body.camera_id,
        track_id=body.track_id,
    )
    return _hypothesis_out(row)


@router.post("/{case_id}/targets/{target_id}/hypotheses/{hypothesis_id}/decide")
def decide_hypothesis(
    case_id: str,
    target_id: str,
    hypothesis_id: int,
    body: DecideRequest,
    db: DbDep,
    user: InvestigatorUser,
) -> HypothesisOut:
    case = _get_case(db, case_id, user)
    target = _get_target(db, case, target_id)
    row = db.get(HypothesisRow, hypothesis_id)
    if row is None or row.target_id != target.target_id:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f"hypothesis {hypothesis_id} not found"
        )
    if row.status != "proposed":
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"hypothesis {hypothesis_id} already decided: {row.status}",
        )
    row.status = body.status
    row.decided_by = user.username
    row.decided_at = utcnow()
    if body.status == "confirmed":
        member = db.scalar(
            select(TargetMemberRow).where(
                TargetMemberRow.target_id == target.target_id,
                TargetMemberRow.run_id == row.run_id,
                TargetMemberRow.camera_id == row.camera_id,
                TargetMemberRow.track_id == row.track_id,
            )
        )
        if member is None:
            db.add(
                TargetMemberRow(
                    target_id=target.target_id,
                    run_id=row.run_id,
                    camera_id=row.camera_id,
                    track_id=row.track_id,
                    added_by=user.username,
                )
            )
    _touch(case)
    audit.append(
        db, user.username, "hypothesis_decided",
        case_id=case.case_id, target_id=target.target_id, hypothesis_id=row.id,
        status=body.status, run_id=row.run_id, camera_id=row.camera_id,
        track_id=row.track_id,
    )
    return _hypothesis_out(row)


# --------------------------------------------------------------------------
# case report (dossier: decisions + audit slice)


def _case_audit_slice(db: Session, case_id: str) -> list[dict]:
    """Audit rows whose detail cites this case, in chain order. LIKE is a
    cheap prefilter; the JSON check is authoritative."""
    rows = db.scalars(
        select(AuditRow)
        .where(AuditRow.detail.like(f'%"case_id": "{case_id}"%'))
        .order_by(AuditRow.seq)
    ).all()
    out = []
    for row in rows:
        try:
            if json.loads(row.detail).get("case_id") != case_id:
                continue
        except (TypeError, ValueError):
            continue
        out.append(
            {"seq": row.seq, "ts": row.ts, "actor": row.actor,
             "action": row.action, "detail": row.detail, "hash": row.hash}
        )
    return out


def _run_evidence(services: ServicesDep, runs) -> list[dict]:
    """Manifest facts for each attached run; a run whose manifest is gone
    from the store is reported as missing, never silently dropped."""
    out = []
    for run in runs:
        try:
            manifest = services.store.load(run.run_id)
        except RunNotFound:
            out.append({"run_id": run.run_id, "missing": True})
            continue
        out.append(
            {
                "run_id": run.run_id,
                "profile": manifest.profile_name,
                "config_hash": manifest.config.config_hash if manifest.config else None,
                "cameras": [
                    {"camera_id": v.camera_id, "sha256": v.sha256}
                    for v in manifest.inputs
                ],
            }
        )
    return out


def _case_report_html(
    db: Session, services: ServicesDep, case: CaseRow, locale: str
) -> str:
    from athar.reporting import render_case_report_html

    detail = _case_detail(db, case)
    runs = db.scalars(
        select(CaseRunRow).where(CaseRunRow.case_id == case.case_id)
    ).all()
    return render_case_report_html(
        detail.model_dump(mode="json"),
        _case_audit_slice(db, case.case_id),
        run_evidence=_run_evidence(services, runs),
        locale=locale if locale in ("ar", "en") else "ar",
    )


@router.get("/{case_id}/report.html", response_class=HTMLResponse)
def export_case_report_html(
    case_id: str, db: DbDep, user: CurrentUser, services: ServicesDep,
    locale: str = "ar",
) -> HTMLResponse:
    """Chromium-free preview of the exact dossier the PDF prints."""
    case = _get_case(db, case_id, user)
    html = _case_report_html(db, services, case, locale)
    audit.append(
        db, user.username, "case_report_exported",
        case_id=case_id, locale=locale, fmt="html",
    )
    return HTMLResponse(html)


@router.get("/{case_id}/report.pdf")
def export_case_report_pdf(
    case_id: str, db: DbDep, user: CurrentUser, services: ServicesDep,
    locale: str = "ar",
) -> Response:
    from athar.reporting import ReportError, html_to_pdf

    case = _get_case(db, case_id, user)
    html = _case_report_html(db, services, case, locale)
    try:
        pdf = html_to_pdf(html)
    except ReportError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from None
    audit.append(
        db, user.username, "case_report_exported",
        case_id=case_id, locale=locale, fmt="pdf",
    )
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="athar-case-{case_id}.pdf"'
        },
    )
