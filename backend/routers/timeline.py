from fastapi import APIRouter, Depends, HTTPException

from backend.models.requests import TimelineQueryRequest
from backend.repositories import InMemoryDatasetRepository
from backend.services.clip_service import _export_matched_clips, _export_selected_clips
from backend.services.debug_service import _export_timeline_debug_bundle
from backend.services.logging_service import _timeline_debug
from backend.services.timeline_service import TimelineService
from backend.services.video_service import _parse_selected_track_nums
from backend.dependencies import get_app_state
from backend.state import AppState
from backend.config import OUTPUT_DIR

router = APIRouter()


@router.post("/api/timeline/query")
async def query_timeline(request: TimelineQueryRequest, state: AppState = Depends(get_app_state)):
    """Resolve selected Stage-2 tracklets into Stage-4 matched trajectories."""
    _timeline_debug("[UI Request] Timeline Query payload:", request.dict())

    if request.videoId not in state.uploaded_videos:
        raise HTTPException(status_code=404, detail="Video not found")

    request_payload = request.dict()

    repo = InMemoryDatasetRepository(state.uploaded_videos, state.video_to_latest_run, OUTPUT_DIR)
    service = TimelineService(repo)
    response_payload, ranked_candidates = service.query_with_candidates(request, state.uploaded_videos)

    # ── I/O side-effects (stay in router) ───────────────────────────────
    debug_bundle_path = _export_timeline_debug_bundle(request_payload, response_payload)
    if debug_bundle_path is not None:
        response_payload["data"].setdefault("diagnostics", {})[
            "debugExportPath"
        ] = str(debug_bundle_path.as_posix())
        _timeline_debug(
            "[UI Request] Timeline debug bundle exported:",
            {"path": str(debug_bundle_path.as_posix())},
        )

    selected_nums = _parse_selected_track_nums(request.selectedTrackIds)
    probe_run_id = (
        response_payload.get("data", {}).get("diagnostics", {}).get("selectedTrackletsSourceRun")
        or request.runId
    )
    if selected_nums and probe_run_id:
        try:
            _export_selected_clips(probe_run_id, selected_nums)
        except Exception as _sc_err:
            print(f"[selected] clip export failed: {_sc_err}", flush=True)

    matched = response_payload.get("data", {}).get("trajectories", [])
    if matched and probe_run_id:
        try:
            _export_matched_clips(
                probe_run_id,
                request.runId,
                matched,
                ranked_candidates=ranked_candidates,
                top_k_alternatives=5,
            )
        except Exception as _mc_err:
            print(f"[matched] clip export failed: {_mc_err}", flush=True)

    return response_payload
