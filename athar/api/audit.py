"""Tamper-evident audit log (D19): hash-chained append-only rows.

Each record's hash covers its content AND the previous record's hash, so
editing or deleting any historical row breaks every hash after it —
``verify_chain`` finds the first broken link. This is evidence-handling
hygiene, not cryptographic non-repudiation (that would need an external
anchor; Phase 7 can periodically export chain heads to WORM storage).
"""

from __future__ import annotations

import hashlib
import json
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from athar.api.db import AuditRow, utcnow

GENESIS = "0" * 64


def _record_hash(prev_hash: str, ts: str, actor: str, action: str, detail: str) -> str:
    payload = "|".join((prev_hash, ts, actor, action, detail))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def append(db: Session, actor: str, action: str, **detail) -> AuditRow:
    """Append one audit record, chaining onto the latest hash."""
    last = db.scalar(select(AuditRow).order_by(AuditRow.seq.desc()).limit(1))
    prev_hash = last.hash if last is not None else GENESIS
    ts = utcnow().isoformat()
    detail_json = json.dumps(detail, sort_keys=True, default=str)
    row = AuditRow(
        ts=ts,
        actor=actor,
        action=action,
        detail=detail_json,
        prev_hash=prev_hash,
        hash=_record_hash(prev_hash, ts, actor, action, detail_json),
    )
    db.add(row)
    db.flush()
    return row


def verify_chain(db: Session) -> Optional[int]:
    """Walk the whole chain; returns the seq of the FIRST broken record, or
    None when the chain is intact."""
    prev_hash = GENESIS
    for row in db.scalars(select(AuditRow).order_by(AuditRow.seq)):
        expected = _record_hash(prev_hash, row.ts, row.actor, row.action, row.detail)
        if row.prev_hash != prev_hash or row.hash != expected:
            return row.seq
        prev_hash = row.hash
    return None
