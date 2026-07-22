"""App database (SQLAlchemy 2 on SQLite WAL, D19): users, sessions, audit,
and case files.

Runs/jobs/models each have their own store (filesystem manifests, jobs.db,
models.db); this DB holds only what the APP owns — identities, sessions,
the tamper-evident audit chain, and investigation cases (cases are owned
by users and every decision on them is audited, so they live next to both).
"""

from __future__ import annotations

import enum
from datetime import datetime, timezone
from pathlib import Path

from typing import Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    event,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    sessionmaker,
)


class Role(str, enum.Enum):
    """Ordered RBAC roles: every admin is an investigator, every
    investigator is a viewer."""

    VIEWER = "viewer"
    INVESTIGATOR = "investigator"
    ADMIN = "admin"

    @property
    def rank(self) -> int:
        return {"viewer": 0, "investigator": 1, "admin": 2}[self.value]

    def covers(self, required: "Role") -> bool:
        return self.rank >= required.rank


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class UserRow(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(256))
    role: Mapped[str] = mapped_column(String(16), default=Role.VIEWER.value)
    disabled: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class SessionRow(Base):
    __tablename__ = "sessions"

    token_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    revoked: Mapped[bool] = mapped_column(Boolean, default=False)


class AuditRow(Base):
    __tablename__ = "audit_log"

    seq: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    ts: Mapped[str] = mapped_column(String(40))
    actor: Mapped[str] = mapped_column(String(64))
    action: Mapped[str] = mapped_column(String(64))
    detail: Mapped[str] = mapped_column(Text, default="{}")
    prev_hash: Mapped[str] = mapped_column(String(64))
    hash: Mapped[str] = mapped_column(String(64))


class CaseRow(Base):
    """An investigation case file. Need-to-know scoping: readable and
    writable by its owner and by admins only — a non-owner gets 404, never
    403, so the API does not even confirm the case exists."""

    __tablename__ = "cases"

    case_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    title: Mapped[str] = mapped_column(String(256))
    status: Mapped[str] = mapped_column(String(16), default="open")  # open|closed
    owner: Mapped[str] = mapped_column(String(64), index=True)  # username
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class CaseRunRow(Base):
    """A run attached to a case as evidence footage (gallery or probe —
    the role is copied from the run manifest at attach time)."""

    __tablename__ = "case_runs"
    __table_args__ = (UniqueConstraint("case_id", "run_id"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    case_id: Mapped[str] = mapped_column(ForeignKey("cases.case_id"), index=True)
    run_id: Mapped[str] = mapped_column(String(64))
    role: Mapped[str] = mapped_column(String(16))
    attached_by: Mapped[str] = mapped_column(String(64))
    attached_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class TargetRow(Base):
    """One case-level identity under investigation (D7)."""

    __tablename__ = "targets"

    target_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    case_id: Mapped[str] = mapped_column(ForeignKey("cases.case_id"), index=True)
    label: Mapped[str] = mapped_column(String(256))
    created_by: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class TargetMemberRow(Base):
    """A tracklet CONFIRMED to belong to a target. Rows are only ever
    created by confirming a hypothesis — never silently fused (D7)."""

    __tablename__ = "target_members"
    __table_args__ = (UniqueConstraint("target_id", "run_id", "camera_id", "track_id"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    target_id: Mapped[str] = mapped_column(ForeignKey("targets.target_id"), index=True)
    run_id: Mapped[str] = mapped_column(String(64))
    camera_id: Mapped[str] = mapped_column(String(64))
    track_id: Mapped[int] = mapped_column(Integer)
    added_by: Mapped[str] = mapped_column(String(64))
    added_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class HypothesisRow(Base):
    """A proposed identity link between a target and a tracklet, decided
    exactly once by an investigator (confirm/reject, attributed)."""

    __tablename__ = "hypotheses"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    target_id: Mapped[str] = mapped_column(ForeignKey("targets.target_id"), index=True)
    kind: Mapped[str] = mapped_column(String(16))  # HypothesisKind values
    run_id: Mapped[str] = mapped_column(String(64))
    camera_id: Mapped[str] = mapped_column(String(64))
    track_id: Mapped[int] = mapped_column(Integer)
    raw_score: Mapped[float] = mapped_column(Float)
    probability: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    stream: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="proposed")
    proposed_by: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    decided_by: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    decided_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


def make_engine(db_path: Path | str):
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(
        f"sqlite:///{path}", connect_args={"check_same_thread": False}
    )

    @event.listens_for(engine, "connect")
    def _pragmas(dbapi_connection, _record):  # noqa: ANN001
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    return engine


def make_session_factory(engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)
