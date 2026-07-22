"""Authentication: login/logout/me over server-side sessions (D19)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, Response, status

from athar.api import audit
from athar.api.deps import CurrentUser, DbDep, ServicesDep
from athar.api.schemas import LoginRequest, UserOut
from athar.api.security import authenticate, open_session, revoke_session

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login")
def login(
    body: LoginRequest, services: ServicesDep, db: DbDep, response: Response
) -> UserOut:
    user = authenticate(db, body.username, body.password)
    if user is None:
        audit.append(db, body.username, "login_failed")
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid credentials")
    settings = services.settings
    token = open_session(db, user, settings.session_ttl_hours)
    audit.append(db, user.username, "login")
    response.set_cookie(
        settings.cookie_name,
        token,
        httponly=True,
        samesite="lax",
        secure=settings.cookie_secure,
        max_age=int(settings.session_ttl_hours * 3600),
    )
    return UserOut(username=user.username, role=user.role, created_at=user.created_at)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(
    request: Request,
    services: ServicesDep,
    db: DbDep,
    user: CurrentUser,
    response: Response,
) -> None:
    token = request.cookies.get(services.settings.cookie_name)
    if token:
        revoke_session(db, token)
    audit.append(db, user.username, "logout")
    response.delete_cookie(services.settings.cookie_name)


@router.get("/me")
def me(user: CurrentUser) -> UserOut:
    return UserOut(username=user.username, role=user.role, created_at=user.created_at)
