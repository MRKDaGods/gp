"""Short-lived cache for POST /api/timeline/query when skipExports=true.

Avoids recomputation + disk exports when the UI revisits Stage 4 with identical inputs.
"""

from __future__ import annotations

import copy
import time
from typing import Any, Dict, Tuple

from backend.models.requests import TimelineQueryRequest

_TIMELINE_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}
_TIMELINE_CACHE_TTL_SEC = 55.0


def timeline_query_cache_key(request: TimelineQueryRequest) -> str:
    ids = sorted(str(x) for x in (request.selectedTrackIds or []))
    return "|".join(
        [
            request.videoId,
            str(request.runId or ""),
            str(request.galleryRunId or ""),
            ",".join(ids),
        ]
    )


def timeline_query_cache_get(key: str) -> Dict[str, Any] | None:
    hit = _TIMELINE_CACHE.get(key)
    if hit is None:
        return None
    ts, payload = hit
    if time.time() - ts > _TIMELINE_CACHE_TTL_SEC:
        try:
            del _TIMELINE_CACHE[key]
        except KeyError:
            pass
        return None
    return copy.deepcopy(payload)


def timeline_query_cache_put(key: str, payload: Dict[str, Any]) -> None:
    _TIMELINE_CACHE[key] = (time.time(), copy.deepcopy(payload))
