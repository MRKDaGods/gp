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
    weights_manifest: Path = Path("configs/weights_manifest.yaml")
    # Deployment site plan: {camera_id: {lat, lng, label}} for the map view.
    # Optional — deployments without surveyed coordinates simply get no map.
    camera_locations: Path = Path("configs/camera_locations.json")

    # Evidence clips (serving/clips.py): context padding around a sighting
    # and the hard cap protecting the server from tape-length transcodes.
    clip_pad_s: float = 1.0
    clip_max_duration_s: float = 60.0

    session_ttl_hours: float = 12.0
    cookie_name: str = "athar_session"
    # Deployments behind TLS set ATHAR_COOKIE_SECURE=1. Default False because
    # first deployments are air-gapped LAN HTTP; Phase 7 hardening flips this
    # in the shipped deployment config.
    cookie_secure: bool = False

    spawn_worker: bool = True  # start a local job worker subprocess on demand

    # Dev origins for the web app (cookies -> allow_credentials, so origins
    # must be explicit, never "*"). Same-origin production needs none.
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
