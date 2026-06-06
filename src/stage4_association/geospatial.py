"""Geospatial reachability constraints for cross-camera association."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.parse import quote, urlencode

from loguru import logger

from src.core.constants import PERSON_CLASSES

# Mean Earth radius (metres), WGS-84 authalic sphere.
EARTH_RADIUS_M = 6_371_008.8

# Class speed profiles (m/s). max_speed bounds reachability (the gate) and is
# kept generous so true matches survive; typical_speed centres the score.
DEFAULT_PERSON_MAX_SPEED_MS = 8.0        # ~29 km/h sprint
DEFAULT_PERSON_TYPICAL_SPEED_MS = 1.4    # ~5 km/h walking
DEFAULT_VEHICLE_MAX_SPEED_MS = 40.0      # ~144 km/h
DEFAULT_VEHICLE_TYPICAL_SPEED_MS = 11.0  # ~40 km/h urban

DEFAULT_OVERLAP_FOV_RADIUS_M = 150.0     # cameras this close can co-observe
DEFAULT_OVERLAP_TIME_WINDOW_S = 1.0      # gaps below this count as simultaneous
DEFAULT_REACH_MARGIN = 0.5               # slack on max_speed in the gate
DEFAULT_MIN_TIME_S = 0.5                 # floor on elapsed time for speed calc


@dataclass(frozen=True)
class GeoSpeedProfile:
    """Speed bounds for one object class."""

    max_speed_ms: float       # gates impossible matches
    typical_speed_ms: float   # centres the plausibility score


def haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Great-circle distance between two WGS-84 points, in metres."""
    if lat1 == lat2 and lng1 == lng2:
        return 0.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lng2 - lng1)
    a = (
        math.sin(d_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
    )
    c = 2.0 * math.asin(min(1.0, math.sqrt(a)))  # clamp guards FP overshoot
    return EARTH_RADIUS_M * c


def _coerce_lat_lng(raw: object) -> Optional[Tuple[float, float]]:
    """Return a valid (lat, lng) from a coordinate record, or None."""
    try:
        lat = float(raw["lat"])  # type: ignore[index]
        lng = float(raw["lng"])  # type: ignore[index]
    except (TypeError, ValueError, KeyError, IndexError):
        return None
    if not (math.isfinite(lat) and math.isfinite(lng)):
        return None
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lng <= 180.0):
        return None
    return (lat, lng)


def load_camera_coordinates(path: str | Path) -> Dict[str, Tuple[float, float]]:
    """Load {camera_id: (lat, lng)} from a camera_coordinates.json file."""
    path = Path(path)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(f"Could not read camera coordinates from {path}: {exc}")
        return {}
    if not isinstance(data, dict):
        logger.warning(f"Camera coordinates file {path} is not a JSON object")
        return {}

    coords: Dict[str, Tuple[float, float]] = {}
    skipped = 0
    for key, raw in data.items():
        cam_id = str(key).strip()
        latlng = _coerce_lat_lng(raw)
        if not cam_id or latlng is None:
            skipped += 1
            continue
        coords[cam_id] = latlng
    if skipped:
        logger.debug(f"camera_coordinates: skipped {skipped} invalid record(s) in {path}")
    return coords


class GeoSpatialConstraint:
    """Reachability gate, plausibility score and expanding-ring ordering."""

    def __init__(
        self,
        coordinates: Dict[str, Tuple[float, float]],
        *,
        person_profile: Optional[GeoSpeedProfile] = None,
        vehicle_profile: Optional[GeoSpeedProfile] = None,
        overlap_fov_radius_m: float = DEFAULT_OVERLAP_FOV_RADIUS_M,
        overlap_time_window_s: float = DEFAULT_OVERLAP_TIME_WINDOW_S,
        reach_margin: float = DEFAULT_REACH_MARGIN,
        min_time_s: float = DEFAULT_MIN_TIME_S,
    ):
        self.coordinates = dict(coordinates)
        self.person_profile = person_profile or GeoSpeedProfile(
            DEFAULT_PERSON_MAX_SPEED_MS, DEFAULT_PERSON_TYPICAL_SPEED_MS
        )
        self.vehicle_profile = vehicle_profile or GeoSpeedProfile(
            DEFAULT_VEHICLE_MAX_SPEED_MS, DEFAULT_VEHICLE_TYPICAL_SPEED_MS
        )
        self.overlap_fov_radius_m = max(float(overlap_fov_radius_m), 0.0)
        self.overlap_time_window_s = max(float(overlap_time_window_s), 0.0)
        self.reach_margin = max(float(reach_margin), 0.0)
        self.min_time_s = max(float(min_time_s), 1e-3)

        self._dist_cache: Dict[Tuple[str, str], float] = {}

    @property
    def is_active(self) -> bool:
        """True only when at least two cameras have coordinates."""
        return len(self.coordinates) >= 2

    def has_coords(self, camera_id: str) -> bool:
        return camera_id in self.coordinates

    def distance_m(self, cam_a: str, cam_b: str) -> Optional[float]:
        """Distance between two cameras in metres, or None if either is unknown."""
        if cam_a == cam_b:
            return 0.0
        key = (cam_a, cam_b) if cam_a < cam_b else (cam_b, cam_a)
        cached = self._dist_cache.get(key)
        if cached is not None:
            return cached
        pa = self.coordinates.get(cam_a)
        pb = self.coordinates.get(cam_b)
        if pa is None or pb is None:
            return None
        dist = haversine_m(pa[0], pa[1], pb[0], pb[1])
        self._dist_cache[key] = dist
        return dist

    def profile_for(self, class_id: int) -> GeoSpeedProfile:
        return self.person_profile if class_id in PERSON_CLASSES else self.vehicle_profile

    def reachable_radius_m(self, time_gap_s: float, class_id: int) -> float:
        """Max distance the object could travel in time_gap_s seconds."""
        dt = abs(time_gap_s)
        if dt <= self.overlap_time_window_s:
            return self.overlap_fov_radius_m * (1.0 + self.reach_margin)
        profile = self.profile_for(class_id)
        return profile.max_speed_ms * (1.0 + self.reach_margin) * dt

    def is_reachable(
        self,
        cam_a: str,
        cam_b: str,
        time_gap_s: float,
        class_id: int,
    ) -> bool:
        """Whether cam_b is reachable from cam_a within the elapsed time."""
        if cam_a == cam_b:
            return True
        dist = self.distance_m(cam_a, cam_b)
        if dist is None:
            return True
        return dist <= self.reachable_radius_m(time_gap_s, class_id)

    def geo_score(
        self,
        cam_a: str,
        cam_b: str,
        time_gap_s: float,
        class_id: int,
    ) -> float:
        """Plausibility of the transition in [0, 1]."""
        if cam_a == cam_b:
            return 1.0
        dist = self.distance_m(cam_a, cam_b)
        if dist is None:
            return 1.0
        if not self.is_reachable(cam_a, cam_b, time_gap_s, class_id):
            return 0.0

        dt = abs(time_gap_s)
        if dt <= self.overlap_time_window_s:
            # Simultaneous: governed by distance alone.
            sigma = max(self.overlap_fov_radius_m, 1.0)
            return math.exp(-0.5 * (dist / sigma) ** 2)

        profile = self.profile_for(class_id)
        required_speed = dist / max(dt, self.min_time_s)
        sigma = max(profile.typical_speed_ms, 1e-3)
        return math.exp(-0.5 * (required_speed / sigma) ** 2)

    def cameras_by_distance(
        self,
        origin_camera: str,
        *,
        include_origin: bool = False,
    ) -> List[Tuple[str, float]]:
        """[(camera_id, distance_m), ...] nearest-first from origin_camera."""
        out: List[Tuple[str, float]] = []
        for cam in self.coordinates:
            if cam == origin_camera and not include_origin:
                continue
            dist = self.distance_m(origin_camera, cam)
            if dist is None:
                continue
            out.append((cam, dist))
        out.sort(key=lambda cd: (cd[1], cd[0]))
        return out

    def expanding_rings(
        self,
        origin_camera: str,
        ring_width_m: float,
        *,
        include_origin: bool = False,
    ) -> List[List[str]]:
        """Cameras grouped into concentric rings of width ring_width_m."""
        if ring_width_m <= 0:
            raise ValueError(f"ring_width_m must be positive, got {ring_width_m}")
        ordered = self.cameras_by_distance(origin_camera, include_origin=include_origin)
        rings: Dict[int, List[str]] = {}
        for cam, dist in ordered:
            ring_idx = int(dist // ring_width_m)
            rings.setdefault(ring_idx, []).append(cam)
        return [rings[k] for k in sorted(rings.keys())]


def prune_unreachable_pairs(
    candidate_pairs: Sequence[Tuple[int, int, float]],
    camera_ids: Sequence[str],
    class_ids: Sequence[int],
    start_times: Sequence[float],
    end_times: Sequence[float],
    geo_constraint: GeoSpatialConstraint,
) -> Tuple[List[Tuple[int, int, float]], int]:
    """Drop candidate pairs whose cameras are unreachable in the elapsed time."""
    if not geo_constraint.is_active:
        return list(candidate_pairs), 0

    kept: List[Tuple[int, int, float]] = []
    removed = 0
    for i, j, sim in candidate_pairs:
        if camera_ids[i] == camera_ids[j]:
            kept.append((i, j, sim))
            continue
        later_start = max(start_times[i], start_times[j])
        earlier_end = min(end_times[i], end_times[j])
        min_gap = max(0.0, later_start - earlier_end)
        if geo_constraint.is_reachable(camera_ids[i], camera_ids[j], min_gap, class_ids[i]):
            kept.append((i, j, sim))
        else:
            removed += 1
    return kept, removed


# Google Maps path / QR link (port of frontend/src/lib/maps-share.ts).

GOOGLE_MAPS_DIR_BASE = "https://www.google.com/maps/dir/"
GOOGLE_MAPS_SEARCH_BASE = "https://www.google.com/maps/search/"
GOOGLE_MAPS_URL_MAX_LEN = 1800  # keep QR/deep links scannable


def _dedupe_consecutive(segments: Sequence[str]) -> List[str]:
    out: List[str] = []
    for s in segments:
        if not out or out[-1] != s:
            out.append(s)
    return out


def _pick_evenly_spaced_indices(total: int, pick_count: int) -> List[int]:
    """Pick pick_count evenly-spaced indices from range(total)."""
    if pick_count <= 0 or total <= 0:
        return []
    if pick_count >= total:
        return list(range(total))
    idxs: List[int] = []
    for k in range(pick_count):
        idx = round(((k + 1) / (pick_count + 1)) * (total + 1)) - 1
        idxs.append(min(max(idx, 0), total - 1))
    return sorted(set(idxs))


def build_google_maps_directions_url(coord_segments: Sequence[str]) -> str:
    """Multi-stop driving-directions URL from "lat,lng" segments."""
    origin = coord_segments[0]
    destination = coord_segments[-1]
    params = [
        ("api", "1"),
        ("origin", origin),
        ("destination", destination),
        ("travelmode", "driving"),
    ]
    if len(coord_segments) > 2:
        params.append(("waypoints", "|".join(coord_segments[1:-1])))
    return f"{GOOGLE_MAPS_DIR_BASE}?{urlencode(params)}"


def _shorten_directions_segments(segments: List[str]) -> List[str]:
    """Thin interior waypoints until the directions URL fits the length cap."""
    if len(segments) <= 1:
        return segments
    if len(build_google_maps_directions_url(segments)) <= GOOGLE_MAPS_URL_MAX_LEN:
        return segments
    if len(segments) == 2:
        return segments

    first = segments[0]
    last = segments[-1]
    full_inner = segments[1:-1]
    inner_count = len(full_inner)
    while inner_count >= 0:
        if inner_count == 0:
            inner: List[str] = []
        else:
            inner = [full_inner[j] for j in _pick_evenly_spaced_indices(len(full_inner), inner_count)]
        candidate = _dedupe_consecutive([first, *inner, last])
        if len(build_google_maps_directions_url(candidate)) <= GOOGLE_MAPS_URL_MAX_LEN:
            return candidate
        inner_count -= 1
    return [first, last]


def _maps_search_url(lat: float, lng: float) -> str:
    return f"{GOOGLE_MAPS_SEARCH_BASE}?api=1&query={quote(f'{lat},{lng}')}"


def build_maps_path_share_url(path: Sequence[Tuple[float, float]]) -> Optional[str]:
    """Google Maps share URL for an ordered (lat, lng) path."""
    if not path:
        return None

    segments = _dedupe_consecutive([f"{lat},{lng}" for lat, lng in path])

    if len(segments) <= 1:
        lat, lng = path[0]
        return _maps_search_url(lat, lng)

    segments = _shorten_directions_segments(segments)

    if len(segments) <= 1:
        lat_str, _, lng_str = segments[0].partition(",")
        try:
            return _maps_search_url(float(lat_str), float(lng_str))
        except ValueError:
            return None

    return build_google_maps_directions_url(segments)


def build_geo_constraint_from_config(
    geo_cfg: dict,
    *,
    coordinates: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Optional[GeoSpatialConstraint]:
    """Build a GeoSpatialConstraint from a stage-4 geospatial config block."""
    coords: Dict[str, Tuple[float, float]] = {}
    if coordinates:
        coords = dict(coordinates)
    elif geo_cfg.get("camera_coordinates"):
        for cam, raw in dict(geo_cfg["camera_coordinates"]).items():
            latlng = _coerce_lat_lng(raw)
            if latlng is not None:
                coords[str(cam)] = latlng
    else:
        path = geo_cfg.get("camera_coordinates_path")
        if path:
            coords = load_camera_coordinates(path)

    if len(coords) < 2:
        return None

    person_cfg = geo_cfg.get("person", {}) or {}
    vehicle_cfg = geo_cfg.get("vehicle", {}) or {}
    person_profile = GeoSpeedProfile(
        max_speed_ms=float(person_cfg.get("max_speed_ms", DEFAULT_PERSON_MAX_SPEED_MS)),
        typical_speed_ms=float(person_cfg.get("typical_speed_ms", DEFAULT_PERSON_TYPICAL_SPEED_MS)),
    )
    vehicle_profile = GeoSpeedProfile(
        max_speed_ms=float(vehicle_cfg.get("max_speed_ms", DEFAULT_VEHICLE_MAX_SPEED_MS)),
        typical_speed_ms=float(vehicle_cfg.get("typical_speed_ms", DEFAULT_VEHICLE_TYPICAL_SPEED_MS)),
    )
    return GeoSpatialConstraint(
        coords,
        person_profile=person_profile,
        vehicle_profile=vehicle_profile,
        overlap_fov_radius_m=float(geo_cfg.get("overlap_fov_radius_m", DEFAULT_OVERLAP_FOV_RADIUS_M)),
        overlap_time_window_s=float(geo_cfg.get("overlap_time_window_s", DEFAULT_OVERLAP_TIME_WINDOW_S)),
        reach_margin=float(geo_cfg.get("reach_margin", DEFAULT_REACH_MARGIN)),
        min_time_s=float(geo_cfg.get("min_time_s", DEFAULT_MIN_TIME_S)),
    )
