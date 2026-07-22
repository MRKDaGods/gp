"""Passwords + server-side sessions (D19).

- Argon2id via pwdlib (passlib is dead upstream).
- Session tokens are 256-bit urlsafe secrets; ONLY their SHA-256 lands in
  the DB, so a database leak does not leak live sessions. The cookie is
  httponly; there is no JS-readable auth state.
"""

from __future__ import annotations

import hashlib
import secrets
from datetime import timedelta
from typing import Optional

from pwdlib import PasswordHash
from sqlalchemy import select
from sqlalchemy.orm import Session

from athar.api.db import Role, SessionRow, UserRow, utcnow

_password_hash = PasswordHash.recommended()  # argon2id


class AuthError(RuntimeError):
    pass


def hash_password(password: str) -> str:
    return _password_hash.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    return _password_hash.verify(password, password_hash)


def create_user(
    db: Session, username: str, password: str, role: Role = Role.VIEWER
) -> UserRow:
    if not username or not password:
        raise AuthError("username and password are required")
    if db.scalar(select(UserRow).where(UserRow.username == username)):
        raise AuthError(f"user {username!r} already exists")
    user = UserRow(
        username=username, password_hash=hash_password(password), role=role.value
    )
    db.add(user)
    db.flush()
    return user


def authenticate(db: Session, username: str, password: str) -> Optional[UserRow]:
    user = db.scalar(select(UserRow).where(UserRow.username == username))
    if user is None or user.disabled:
        # burn comparable time so missing users are not distinguishable
        _password_hash.verify(password, hash_password("timing-equalizer"))
        return None
    return user if verify_password(password, user.password_hash) else None


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def open_session(db: Session, user: UserRow, ttl_hours: float) -> str:
    """Create a session row; returns the RAW token (cookie value) — the DB
    only ever sees its hash."""
    token = secrets.token_urlsafe(32)
    db.add(
        SessionRow(
            token_hash=_token_hash(token),
            user_id=user.id,
            expires_at=utcnow() + timedelta(hours=ttl_hours),
        )
    )
    db.flush()
    return token


def resolve_session(db: Session, token: str) -> Optional[UserRow]:
    row = db.get(SessionRow, _token_hash(token))
    if row is None or row.revoked:
        return None
    expires = row.expires_at
    if expires.tzinfo is None:  # sqlite round-trips naive; stored values are UTC
        from datetime import timezone

        expires = expires.replace(tzinfo=timezone.utc)
    if expires < utcnow():
        return None
    user = db.get(UserRow, row.user_id)
    if user is None or user.disabled:
        return None
    return user


def revoke_session(db: Session, token: str) -> bool:
    row = db.get(SessionRow, _token_hash(token))
    if row is None or row.revoked:
        return False
    row.revoked = True
    db.flush()
    return True
