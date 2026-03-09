"""Corrections database for forensic identity reassignment.

Stores analyst corrections (reassign, merge, split) with audit trail
in a SQLite database alongside the pipeline run outputs.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional


class CorrectionsStore:
    """Persistent store for analyst identity corrections."""

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS corrections (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   REAL    NOT NULL,
                action      TEXT    NOT NULL,
                source_gid  INTEGER,
                target_gid  INTEGER,
                tracklet_camera   TEXT,
                tracklet_track_id INTEGER,
                reason      TEXT,
                undone      INTEGER DEFAULT 0
            )
        """)
        self.conn.commit()

    def log_reassign(
        self,
        tracklet_cam: str,
        tracklet_tid: int,
        from_gid: int,
        to_gid: int,
        reason: str = "",
    ) -> int:
        cur = self.conn.execute(
            "INSERT INTO corrections "
            "(timestamp, action, source_gid, target_gid, tracklet_camera, tracklet_track_id, reason) "
            "VALUES (?, 'reassign', ?, ?, ?, ?, ?)",
            (time.time(), from_gid, to_gid, tracklet_cam, tracklet_tid, reason),
        )
        self.conn.commit()
        return cur.lastrowid

    def log_merge(
        self,
        gid_a: int,
        gid_b: int,
        merged_gid: int,
        reason: str = "",
    ) -> int:
        cur = self.conn.execute(
            "INSERT INTO corrections "
            "(timestamp, action, source_gid, target_gid, reason) "
            "VALUES (?, 'merge', ?, ?, ?)",
            (time.time(), gid_a, merged_gid, reason),
        )
        # Log second GID as well
        self.conn.execute(
            "INSERT INTO corrections "
            "(timestamp, action, source_gid, target_gid, reason) "
            "VALUES (?, 'merge', ?, ?, ?)",
            (time.time(), gid_b, merged_gid, reason),
        )
        self.conn.commit()
        return cur.lastrowid

    def log_split(
        self,
        source_gid: int,
        new_gid: int,
        tracklet_cam: str,
        tracklet_tid: int,
        reason: str = "",
    ) -> int:
        cur = self.conn.execute(
            "INSERT INTO corrections "
            "(timestamp, action, source_gid, target_gid, tracklet_camera, tracklet_track_id, reason) "
            "VALUES (?, 'split', ?, ?, ?, ?, ?)",
            (time.time(), source_gid, new_gid, tracklet_cam, tracklet_tid, reason),
        )
        self.conn.commit()
        return cur.lastrowid

    def undo_last(self) -> Optional[Dict]:
        row = self.conn.execute(
            "SELECT * FROM corrections WHERE undone = 0 ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        self.conn.execute("UPDATE corrections SET undone = 1 WHERE id = ?", (row["id"],))
        self.conn.commit()
        return dict(row)

    def get_all(self, include_undone: bool = False) -> List[Dict]:
        if include_undone:
            rows = self.conn.execute(
                "SELECT * FROM corrections ORDER BY id DESC"
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM corrections WHERE undone = 0 ORDER BY id DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def count(self) -> int:
        return self.conn.execute(
            "SELECT COUNT(*) FROM corrections WHERE undone = 0"
        ).fetchone()[0]

    def close(self) -> None:
        self.conn.close()

    def __del__(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass
