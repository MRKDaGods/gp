"""Model lifecycle registry — SQLite source of truth (D5).

Stages: candidate -> validated -> production -> retired. Every transition
is eval-gated (``ModelEntry.promote`` refuses without an evaluation report)
and recorded in an append-only ``lifecycle_events`` table, so "which model
was production when this case was processed, and why" is always answerable.

Authoring is YAML-only (``import_yaml``): new models enter as CANDIDATES;
promotions happen exclusively through this DB with eval evidence. Promoting
a model to production demotes the task's current production model to
validated (kept, so ``rollback`` can restore it).
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

from athar.serving.registry import EvalReportRef, ModelEntry, ModelStage, ModelTask

DEFAULT_DB = Path("data/registry/models.db")


class LifecycleError(RuntimeError):
    pass


class ModelNotFound(LifecycleError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ModelLifecycleDB:
    """SQLite-backed lifecycle store for :class:`ModelEntry` records."""

    def __init__(self, db_path: Path | str = DEFAULT_DB) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # served from FastAPI's threadpool -> cross-thread use behind _lock
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=5000")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()

    def _create_tables(self) -> None:
        with self._lock, self.conn:
            self.conn.execute(
                """CREATE TABLE IF NOT EXISTS models (
                    model_id TEXT PRIMARY KEY,
                    task TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    entry_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )"""
            )
            self.conn.execute(
                """CREATE TABLE IF NOT EXISTS lifecycle_events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    from_stage TEXT,
                    to_stage TEXT,
                    superseded_model TEXT,
                    eval_run_id TEXT,
                    actor TEXT NOT NULL DEFAULT '',
                    notes TEXT NOT NULL DEFAULT ''
                )"""
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_models_task_stage ON models(task, stage)"
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_model ON lifecycle_events(model_id)"
            )

    def close(self) -> None:
        self.conn.close()

    # -- write path ----------------------------------------------------------
    def _write_entry(self, entry: ModelEntry) -> None:
        self.conn.execute(
            """INSERT INTO models (model_id, task, stage, entry_json, updated_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(model_id) DO UPDATE SET
                 task=excluded.task, stage=excluded.stage,
                 entry_json=excluded.entry_json, updated_at=excluded.updated_at""",
            (
                entry.model_id,
                entry.task.value,
                entry.stage.value,
                entry.model_dump_json(),
                _now(),
            ),
        )

    def _event(
        self,
        model_id: str,
        action: str,
        from_stage: Optional[str] = None,
        to_stage: Optional[str] = None,
        superseded_model: Optional[str] = None,
        eval_run_id: Optional[str] = None,
        actor: str = "",
        notes: str = "",
    ) -> None:
        self.conn.execute(
            """INSERT INTO lifecycle_events
               (ts, model_id, action, from_stage, to_stage, superseded_model,
                eval_run_id, actor, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (_now(), model_id, action, from_stage, to_stage, superseded_model,
             eval_run_id, actor, notes),
        )

    def add(self, entry: ModelEntry, actor: str = "") -> None:
        with self._lock:
            existing = self.conn.execute(
                "SELECT 1 FROM models WHERE model_id = ?", (entry.model_id,)
            ).fetchone()
        if existing:
            raise LifecycleError(f"model {entry.model_id!r} already registered")
        with self._lock, self.conn:
            self._write_entry(entry)
            self._event(
                entry.model_id, "register", to_stage=entry.stage.value, actor=actor
            )

    # -- read path -------------------------------------------------------------
    def get(self, model_id: str) -> ModelEntry:
        with self._lock:
            row = self.conn.execute(
                "SELECT entry_json FROM models WHERE model_id = ?", (model_id,)
            ).fetchone()
        if row is None:
            raise ModelNotFound(model_id)
        return ModelEntry.model_validate_json(row["entry_json"])

    def list(
        self,
        task: Optional[ModelTask] = None,
        stage: Optional[ModelStage] = None,
    ) -> list[ModelEntry]:
        query = "SELECT entry_json FROM models"
        clauses, params = [], []
        if task is not None:
            clauses.append("task = ?")
            params.append(task.value)
        if stage is not None:
            clauses.append("stage = ?")
            params.append(stage.value)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY model_id"
        with self._lock:
            rows = self.conn.execute(query, params).fetchall()
        return [ModelEntry.model_validate_json(r["entry_json"]) for r in rows]

    def production(self, task: ModelTask) -> Optional[ModelEntry]:
        entries = self.list(task=task, stage=ModelStage.PRODUCTION)
        if len(entries) > 1:  # invariant guarded by promote(); belt-and-braces
            raise LifecycleError(
                f"multiple production models for {task.value}: "
                f"{[e.model_id for e in entries]}"
            )
        return entries[0] if entries else None

    def events(self, model_id: Optional[str] = None) -> list[dict]:
        query = "SELECT * FROM lifecycle_events"
        params: tuple = ()
        if model_id is not None:
            query += " WHERE model_id = ?"
            params = (model_id,)
        query += " ORDER BY seq"
        with self._lock:
            rows = self.conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    # -- lifecycle transitions ---------------------------------------------------
    def promote(
        self,
        model_id: str,
        to: ModelStage,
        eval_report: Optional[EvalReportRef] = None,
        actor: str = "",
        notes: str = "",
    ) -> ModelEntry:
        """Eval-gated promotion (D5). Promotion to PRODUCTION demotes the
        task's current production model to VALIDATED and records it as
        superseded so :meth:`rollback` can restore it."""
        entry = self.get(model_id)
        from_stage = entry.stage
        superseded: Optional[ModelEntry] = None
        if to is ModelStage.PRODUCTION:
            current = self.production(entry.task)
            if current is not None and current.model_id != model_id:
                superseded = current
        entry.promote(to, eval_report)  # raises on illegal/ungated transitions
        with self._lock, self.conn:
            if superseded is not None:
                superseded.stage = ModelStage.VALIDATED
                self._write_entry(superseded)
                self._event(
                    superseded.model_id, "demote",
                    from_stage=ModelStage.PRODUCTION.value,
                    to_stage=ModelStage.VALIDATED.value,
                    actor=actor, notes=f"superseded by {model_id}",
                )
            self._write_entry(entry)
            self._event(
                model_id, "promote",
                from_stage=from_stage.value, to_stage=to.value,
                superseded_model=superseded.model_id if superseded else None,
                eval_run_id=eval_report.run_id if eval_report else None,
                actor=actor, notes=notes,
            )
        return entry

    def retire(self, model_id: str, actor: str = "", notes: str = "") -> ModelEntry:
        entry = self.get(model_id)
        from_stage = entry.stage
        entry.promote(ModelStage.RETIRED)
        with self._lock, self.conn:
            self._write_entry(entry)
            self._event(
                model_id, "retire",
                from_stage=from_stage.value, to_stage=ModelStage.RETIRED.value,
                actor=actor, notes=notes,
            )
        return entry

    def rollback(self, task: ModelTask, actor: str = "", notes: str = "") -> ModelEntry:
        """Undo the latest production promotion for ``task``: the current
        production model returns to VALIDATED and the model it superseded
        (if any, and not retired since) returns to PRODUCTION."""
        current = self.production(task)
        if current is None:
            raise LifecycleError(f"no production model for {task.value} to roll back")
        with self._lock:
            promote_event = self.conn.execute(
                """SELECT * FROM lifecycle_events
               WHERE model_id = ? AND action = 'promote' AND to_stage = 'production'
               ORDER BY seq DESC LIMIT 1""",
            (current.model_id,),
        ).fetchone()
        restored: Optional[ModelEntry] = None
        if promote_event is not None and promote_event["superseded_model"]:
            candidate = self.get(promote_event["superseded_model"])
            if candidate.stage is ModelStage.VALIDATED:
                restored = candidate
        with self._lock, self.conn:
            current.stage = ModelStage.VALIDATED
            self._write_entry(current)
            if restored is not None:
                restored.stage = ModelStage.PRODUCTION
                self._write_entry(restored)
            self._event(
                current.model_id, "rollback",
                from_stage=ModelStage.PRODUCTION.value,
                to_stage=ModelStage.VALIDATED.value,
                superseded_model=restored.model_id if restored else None,
                actor=actor, notes=notes,
            )
        return restored if restored is not None else current

    # -- authoring ---------------------------------------------------------------
    def import_yaml(self, path: Path | str, actor: str = "") -> dict:
        """Import authoring YAML (``models:`` list of ModelEntry-shaped
        dicts). New models enter as CANDIDATES — a YAML claiming any other
        stage is refused (promotions are eval-gated DB operations, never
        authored). Already-registered ids are skipped (idempotent)."""
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        models = raw.get("models")
        if not isinstance(models, list):
            raise LifecycleError(f"{path}: expected a top-level 'models' list")
        added, skipped = [], []
        for item in models:
            declared = item.get("stage", ModelStage.CANDIDATE.value)
            if declared != ModelStage.CANDIDATE.value:
                raise LifecycleError(
                    f"{path}: model {item.get('model_id')!r} declares stage "
                    f"{declared!r} — authoring may only introduce candidates (D5)"
                )
            entry = ModelEntry.model_validate(item)
            try:
                self.add(entry, actor=actor)
                added.append(entry.model_id)
            except LifecycleError:
                skipped.append(entry.model_id)
        return {"added": added, "skipped": skipped}


def parse_metrics(pairs: list[str]) -> dict[str, float]:
    """``key=value`` CLI pairs -> metrics dict (for EvalReportRef)."""
    metrics: dict[str, float] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep:
            raise LifecycleError(f"expected METRIC=VALUE, got {pair!r}")
        try:
            metrics[key] = float(value)
        except ValueError:
            raise LifecycleError(f"metric {key!r} value {value!r} is not a number") from None
    return metrics


def entry_summary(entry: ModelEntry) -> str:
    """One line per model for CLI listings (ASCII-safe for cp1252 consoles)."""
    reports = len(entry.eval_reports)
    return (
        f"{entry.model_id:<40} {entry.task.value:<18} {entry.stage.value:<10} "
        f"{entry.architecture:<24} evals={reports}"
    )


def dump_entry(entry: ModelEntry) -> str:
    return json.dumps(entry.model_dump(mode="json"), indent=2)
