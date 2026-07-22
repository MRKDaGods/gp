"""Probe-vs-gallery search with explicit compatibility semantics:

- 404 — a run id does not exist;
- 409 — the PAIR is incompatible (missing stream / dim / projection
  lineage): re-process one side, never silently degrade (v1 PCA bug);
- 400 — everything else the engine refuses (unindexed gallery, bad stream).

Scores come back with a calibrated probability when the stream has a
calibration artifact, null otherwise — the API never invents confidence.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from athar.api import audit
from athar.api.deps import DbDep, InvestigatorUser, ServicesDep
from athar.api.schemas import SearchHitOut, SearchRequest, SearchResponse
from athar.contracts.store import RunNotFound
from athar.search.engine import GallerySearcher, IncompatibleStreams, SearchError

router = APIRouter(prefix="/search", tags=["search"])


@router.post("")
def search(
    body: SearchRequest, services: ServicesDep, db: DbDep, user: InvestigatorUser
) -> SearchResponse:
    store = services.store
    try:
        gallery = store.load(body.gallery_run_id)
        probe = store.load(body.probe_run_id)
    except RunNotFound as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"run not found: {exc}") from None
    try:
        searcher = GallerySearcher(store, gallery)
        stream = body.stream or next(
            (s for s in searcher.streams() if s != "hsv"), None
        )
        if stream is None:
            raise SearchError(
                f"gallery {gallery.run_id} has no appearance stream to search"
            )
        hits = searcher.search_probe(
            store, probe, stream, top_k=body.top_k, min_score=body.min_score
        )
    except IncompatibleStreams as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from None
    except SearchError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from None

    calibrations = services.calibrations
    audit.append(
        db, user.username, "search",
        gallery_run_id=body.gallery_run_id, probe_run_id=body.probe_run_id,
        stream=stream, num_hits=len(hits),
    )
    return SearchResponse(
        gallery_run_id=body.gallery_run_id,
        probe_run_id=body.probe_run_id,
        stream=stream,
        calibrated=stream in calibrations.streams,
        hits=[
            SearchHitOut(
                probe_camera_id=h.probe_key.camera_id,
                probe_track_id=h.probe_key.track_id,
                gallery_camera_id=h.gallery_key.camera_id,
                gallery_track_id=h.gallery_key.track_id,
                stream=h.stream,
                score=h.score,
                probability=calibrations.probability(h.stream, h.score),
                gallery_entity_class=h.gallery_entity_class.value,
                gallery_start_ts_s=h.gallery_start_ts_s,
                gallery_end_ts_s=h.gallery_end_ts_s,
            )
            for h in hits
        ],
    )
