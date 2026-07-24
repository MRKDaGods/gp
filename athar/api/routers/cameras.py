"""Cameras: deployment site-plan metadata (GPS locations for the map view).

Locations come from a surveyed JSON file on the box
(``settings.camera_locations``) — deployments without one simply get an
empty map. Air-gap note: coordinates are served to the frontend which
renders them over a locally-hosted PMTiles basemap; nothing here implies a
network tile fetch.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, status

from athar.api.deps import RequireViewer, ServicesDep
from athar.api.schemas import CameraLocationsOut

router = APIRouter(prefix="/cameras", tags=["cameras"], dependencies=[RequireViewer])


@router.get("/locations")
def camera_locations(services: ServicesDep) -> CameraLocationsOut:
    path = services.settings.camera_locations
    if not path.is_file():
        return CameraLocationsOut(cameras={})
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            f"camera locations file unreadable: {exc}",
        ) from None
    return CameraLocationsOut(cameras=raw)
