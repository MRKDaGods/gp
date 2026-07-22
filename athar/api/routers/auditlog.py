"""Audit log inspection (admin): read the chain, verify its integrity."""

from __future__ import annotations

import json

from fastapi import APIRouter
from sqlalchemy import select

from athar.api.audit import verify_chain
from athar.api.db import AuditRow
from athar.api.deps import DbDep, RequireAdmin
from athar.api.schemas import AuditRecordOut, AuditVerifyOut

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
def verify(db: DbDep) -> AuditVerifyOut:
    broken = verify_chain(db)
    return AuditVerifyOut(intact=broken is None, first_broken_seq=broken)
