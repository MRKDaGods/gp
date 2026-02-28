"""SQLite metadata store for tracklet information."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger


class MetadataStore:
    """SQLite-backed storage for tracklet metadata.

    Stores track IDs, camera IDs, timestamps, class labels, and HSV histograms
    for each tracklet. Used by Stage 4 for spatio-temporal gating.
    """

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self._create_tables()

    def _create_tables(self) -> None:
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS tracklets (
                index_id INTEGER PRIMARY KEY,
                track_id INTEGER NOT NULL,
                camera_id TEXT NOT NULL,
                class_id INTEGER NOT NULL,
                start_time REAL NOT NULL,
                end_time REAL NOT NULL,
                num_frames INTEGER NOT NULL,
                hsv_histogram BLOB
            )
        """)
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_camera ON tracklets(camera_id)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_time ON tracklets(start_time, end_time)"
        )
        self.conn.commit()

    def insert_tracklet(
        self,
        index_id: int,
        track_id: int,
        camera_id: str,
        class_id: int,
        start_time: float,
        end_time: float,
        num_frames: int,
        hsv_histogram: Optional[np.ndarray] = None,
    ) -> None:
        """Insert a tracklet's metadata."""
        hsv_blob = hsv_histogram.tobytes() if hsv_histogram is not None else None
        self.conn.execute(
            """INSERT OR REPLACE INTO tracklets
               (index_id, track_id, camera_id, class_id, start_time, end_time, num_frames, hsv_histogram)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (index_id, track_id, camera_id, class_id, start_time, end_time, num_frames, hsv_blob),
        )
        self.conn.commit()

    def get_tracklet(self, index_id: int) -> Optional[Dict]:
        """Get metadata for a specific tracklet by its FAISS index ID."""
        row = self.conn.execute(
            "SELECT * FROM tracklets WHERE index_id = ?", (index_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_dict(row)

    def get_by_camera(self, camera_id: str) -> List[Dict]:
        """Get all tracklets for a specific camera."""
        rows = self.conn.execute(
            "SELECT * FROM tracklets WHERE camera_id = ?", (camera_id,)
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_by_time_range(self, start: float, end: float) -> List[Dict]:
        """Get tracklets active within a time range."""
        rows = self.conn.execute(
            "SELECT * FROM tracklets WHERE end_time >= ? AND start_time <= ?",
            (start, end),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_hsv_histogram(self, index_id: int) -> Optional[np.ndarray]:
        """Get the HSV histogram for a tracklet."""
        row = self.conn.execute(
            "SELECT hsv_histogram FROM tracklets WHERE index_id = ?", (index_id,)
        ).fetchone()
        if row is None or row[0] is None:
            return None
        return np.frombuffer(row[0], dtype=np.float32)

    def get_all(self) -> List[Dict]:
        """Get all tracklet metadata."""
        rows = self.conn.execute("SELECT * FROM tracklets").fetchall()
        return [self._row_to_dict(r) for r in rows]

    def count(self) -> int:
        """Get total number of tracklets."""
        row = self.conn.execute("SELECT COUNT(*) FROM tracklets").fetchone()
        return row[0]

    @staticmethod
    def _row_to_dict(row) -> Dict:
        return {
            "index_id": row[0],
            "track_id": row[1],
            "camera_id": row[2],
            "class_id": row[3],
            "start_time": row[4],
            "end_time": row[5],
            "num_frames": row[6],
        }

    def close(self) -> None:
        self.conn.close()

    def __del__(self):
        try:
            self.conn.close()
        except Exception:
            pass
