"""Audit log inspection (admin): read the chain, verify its integrity,
anchor its head to WORM storage."""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Request
from sqlalchemy import select

from athar.api.audit import export_head, verify_anchors, verify_chain
from athar.api.db import AuditRow
from athar.api.deps import CurrentUser, DbDep, RequireAdmin
from athar.api.schemas import (
    AuditAnchorMismatchOut,
    AuditAnchorOut,
    AuditRecordOut,
    AuditVerifyOut,
)

router = APIRouter(prefix="/audit", tags=["audit"], dependencies=[RequireAdmin])


@router.get("")
def list_audit(db: DbDep, limit: int = 500) -> list[AuditRecordOut]:
    rows = db.scalars(
        select(AuditRow).order_by(AuditRow.seq.desc()).limit(limit)
    ).all()
    return [
        AuditRecordOut(
            seq=r.seq, ts=r.ts, actor=r.actor, action=r.action,
            detail=json.loads(r.detail), prev_hash=r.prev_hash, hash=r.hash,
        )
        for r in reversed(rows)
    ]


@router.get("/verify")
def verify(db: DbDep, request: Request) -> AuditVerifyOut:
    broken = verify_chain(db)
    out = AuditVerifyOut(intact=broken is None, first_broken_seq=broken)
    anchor_path = request.app.state.services.settings.audit_anchor_path
    if anchor_path is not None:
        checked, mismatches = verify_anchors(db, anchor_path)
        out.anchors_checked = checked
        out.anchors_intact = not mismatches
        out.anchor_mismatches = [AuditAnchorMismatchOut(**m) for m in mismatches]
    return out


@router.post("/anchor")
def anchor(db: DbDep, request: Request, user: CurrentUser) -> AuditAnchorOut:
    """Export the current chain head to the configured WORM anchor file.

    409 when anchoring is not configured. Idempotent: an unchanged head is
    reported as anchored=False, not re-written.
    """
    anchor_path = request.app.state.services.settings.audit_anchor_path
    if anchor_path is None:
        raise HTTPException(status_code=409, detail="audit anchoring not configured")
    record = export_head(db, anchor_path, exported_by=user.username)
    if record is None:
        return AuditAnchorOut(anchored=False, seq=None, hash=None, exported_at=None)
    return AuditAnchorOut(anchored=True, seq=record["seq"], hash=record["hash"],
                          exported_at=record["exported_at"])
