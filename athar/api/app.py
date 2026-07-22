"""ATHAR API application factory.

Thin routers -> :class:`~athar.api.deps.Services` -> stores (D19). The
factory takes explicit settings so tests build isolated apps; production
reads ``ATHAR_*`` env vars. Air-gap: the app serves and calls nothing
off-box.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI

from athar import __version__
from athar.api.db import make_engine, make_session_factory
from athar.api.deps import Services
from athar.api.settings import ApiSettings
from athar.contracts.store import FilesystemRunStore
from athar.jobs.service import JobService
from athar.search.calibration import StreamCalibrations
from athar.serving.lifecycle import ModelLifecycleDB

logger = logging.getLogger(__name__)


def _load_calibrations(settings: ApiSettings) -> StreamCalibrations:
    path = settings.calibration_path
    if path is None or not path.is_file():
        if path is not None:
            logger.warning("calibration file %s missing; scores stay uncalibrated", path)
        return StreamCalibrations()
    return StreamCalibrations.load(path)


def create_app(settings: Optional[ApiSettings] = None) -> FastAPI:
    settings = settings or ApiSettings()
    engine = make_engine(settings.app_db)
    services = Services(
        settings=settings,
        store=FilesystemRunStore(settings.runs_root),
        jobs=JobService(
            settings.jobs_db, settings.runs_root, spawn_worker=settings.spawn_worker
        ),
        lifecycle=ModelLifecycleDB(settings.registry_db),
        session_factory=make_session_factory(engine),
        calibrations=_load_calibrations(settings),
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        services.jobs.shutdown()
        services.lifecycle.close()
        engine.dispose()

    app = FastAPI(
        title="ATHAR API",
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        openapi_url="/openapi.json",
    )
    app.state.services = services

    from athar.api.routers import auditlog, auth, jobs, models, runs, search

    app.include_router(auth.router)
    app.include_router(runs.router)
    app.include_router(jobs.router)
    app.include_router(models.router)
    app.include_router(search.router)
    app.include_router(auditlog.router)

    @app.get("/health", tags=["meta"])
    def health() -> dict:
        return {"status": "ok", "version": __version__}

    return app
