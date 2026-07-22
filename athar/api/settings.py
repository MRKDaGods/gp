"""API settings (pydantic-settings, env prefix ``ATHAR_``).

Every path the app touches is explicit here — no ambient defaults inside
routers. Air-gap note: nothing in these settings may point off-box.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class ApiSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ATHAR_", extra="ignore")

    runs_root: Path = Path("data/runs")
    jobs_db: Path = Path("data/jobs/jobs.db")
    registry_db: Path = Path("data/registry/models.db")
    app_db: Path = Path("data/app/app.db")
    calibration_path: Path | None = None  # StreamCalibrations JSON (optional)

    session_ttl_hours: float = 12.0
    cookie_name: str = "athar_session"
    # Deployments behind TLS set ATHAR_COOKIE_SECURE=1. Default False because
    # first deployments are air-gapped LAN HTTP; Phase 7 hardening flips this
    # in the shipped deployment config.
    cookie_secure: bool = False

    spawn_worker: bool = True  # start a local job worker subprocess on demand
