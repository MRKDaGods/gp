"""Tamper-evident audit log (D19): hash-chained append-only rows.

Each record's hash covers its content AND the previous record's hash, so
editing or deleting any historical row breaks every hash after it —
``verify_chain`` finds the first broken link.

The chain alone is evidence-handling hygiene, not non-repudiation: an
attacker with DB access can rewrite the WHOLE chain self-consistently.
``export_head`` closes that hole by appending the current head (seq + hash)
to an external append-only anchor file — deployments point it at WORM
storage (``ATHAR_AUDIT_ANCHOR_PATH``) — and ``verify_anchors`` checks every
past anchor against the current chain: a full rewrite can't fix hashes
already exported off-box.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
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


def export_head(db: Session, anchor_path: Path, *, exported_by: str = "system") -> Optional[dict]:
    """Append the current chain head to the anchor file (JSON Lines).

    Append-only by construction (WORM-friendly); fsynced so the anchor
    survives a crash. Returns the anchor record, or None when there is
    nothing new to anchor (empty chain, or the head is already the last
    anchor). Deliberately does NOT write an audit record itself — anchoring
    the act of anchoring would make every anchor stale the moment it lands.
    """
    last = db.scalar(select(AuditRow).order_by(AuditRow.seq.desc()).limit(1))
    if last is None:
        return None
    if anchor_path.exists():
        lines = [ln for ln in anchor_path.read_text("utf-8").splitlines() if ln.strip()]
        if lines:
            try:
                prev = json.loads(lines[-1])
            except json.JSONDecodeError:
                prev = {}
            if prev.get("seq") == last.seq and prev.get("hash") == last.hash:
                return None
    record = {
        "seq": last.seq,
        "hash": last.hash,
        "ts": last.ts,
        "exported_at": utcnow().isoformat(),
        "exported_by": exported_by,
    }
    anchor_path.parent.mkdir(parents=True, exist_ok=True)
    with open(anchor_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    return record


def verify_anchors(db: Session, anchor_path: Path) -> tuple[int, list[dict]]:
    """Check every exported anchor against the CURRENT chain.

    Returns (anchors_checked, mismatches). A mismatch means history changed
    after the anchor was exported: the anchored row vanished, its hash
    differs (whole-chain rewrite), or the anchor line itself is corrupt.
    """
    if not anchor_path.exists():
        return 0, []
    checked, mismatches = 0, []
    for line in anchor_path.read_text("utf-8").splitlines():
        if not line.strip():
            continue
        checked += 1
        try:
            anchor = json.loads(line)
            seq, anchored_hash = int(anchor["seq"]), str(anchor["hash"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            mismatches.append({"seq": None, "problem": "unparseable_anchor",
                               "anchored_hash": None, "current_hash": None})
            continue
        row = db.scalar(select(AuditRow).where(AuditRow.seq == seq))
        if row is None:
            mismatches.append({"seq": seq, "problem": "anchored_row_missing",
                               "anchored_hash": anchored_hash, "current_hash": None})
        elif row.hash != anchored_hash:
            mismatches.append({"seq": seq, "problem": "hash_mismatch",
                               "anchored_hash": anchored_hash, "current_hash": row.hash})
    return checked, mismatches
