"""Shared dependencies: service container, DB session, auth, RBAC.

Routers never build paths or open stores themselves — they resolve
everything through the :class:`Services` container assembled once in
``create_app`` (thin routers -> services -> stores, D19).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Iterator

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.orm import Session, sessionmaker

from athar.api.db import Role, UserRow
from athar.api.security import resolve_session
from athar.api.settings import ApiSettings
from athar.contracts.store import FilesystemRunStore
from athar.jobs.service import JobService
from athar.search.calibration import StreamCalibrations
from athar.serving.lifecycle import ModelLifecycleDB


@dataclass
class Services:
    settings: ApiSettings
    store: FilesystemRunStore
    jobs: JobService
    lifecycle: ModelLifecycleDB
    session_factory: sessionmaker
    calibrations: StreamCalibrations


def get_services(request: Request) -> Services:
    return request.app.state.services


ServicesDep = Annotated[Services, Depends(get_services)]


def get_db(services: ServicesDep) -> Iterator[Session]:
    db = services.session_factory()
    try:
        yield db
        db.commit()
    except BaseException:
        db.rollback()
        raise
    finally:
        db.close()


DbDep = Annotated[Session, Depends(get_db)]


def get_current_user(request: Request, services: ServicesDep, db: DbDep) -> UserRow:
    token = request.cookies.get(services.settings.cookie_name)
    if not token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "not authenticated")
    user = resolve_session(db, token)
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "session invalid or expired")
    return user


CurrentUser = Annotated[UserRow, Depends(get_current_user)]


def require_role(required: Role):
    """Dependency factory: 403 unless the caller's role covers ``required``
    (ordered roles — every admin is an investigator is a viewer)."""

    def checker(user: CurrentUser) -> UserRow:
        if not Role(user.role).covers(required):
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                f"requires role {required.value} (you are {user.role})",
            )
        return user

    return checker


RequireViewer = Depends(require_role(Role.VIEWER))
RequireInvestigator = Depends(require_role(Role.INVESTIGATOR))
RequireAdmin = Depends(require_role(Role.ADMIN))

InvestigatorUser = Annotated[UserRow, Depends(require_role(Role.INVESTIGATOR))]
AdminUser = Annotated[UserRow, Depends(require_role(Role.ADMIN))]
