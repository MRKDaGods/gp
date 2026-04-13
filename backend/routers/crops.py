import io
import threading
from collections import OrderedDict
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from backend.config import _HAS_CV2, OUTPUT_DIR
from backend.dependencies import get_app_state
from backend.services.clip_service import _stage0_frame_path
from backend.services.video_service import _detect_camera_for_video
from backend.state import AppState

if _HAS_CV2:
    import cv2

router = APIRouter()

# ── Upload-video crop performance ───────────────────────────────────────────
# Each crop used to open a fresh VideoCapture; dozens of parallel sidebar
# requests exhausted the worker and stalled video streaming.  We keep one
# capture per video_id (serialized per video) and LRU-cache decoded full
# frames so multiple bboxes on the same frame share one decode.

_FRAME_LRU_MAX = 24
_frame_lru: "OrderedDict[tuple[str, int], object]" = OrderedDict()
_lru_lock = threading.Lock()
_caps: dict[str, object] = {}
_cap_locks: dict[str, threading.Lock] = {}


def _cap_lock(video_id: str) -> threading.Lock:
    with _lru_lock:
        if video_id not in _cap_locks:
            _cap_locks[video_id] = threading.Lock()
        return _cap_locks[video_id]


def _read_frame_bgr(video_id: str, video_path: Path, frame_id: int):
    """Return a BGR frame copy; uses persistent capture + LRU frame cache."""
    key = (video_id, frame_id)
    with _lru_lock:
        if key in _frame_lru:
            _frame_lru.move_to_end(key)
            cached = _frame_lru[key]
            return cached.copy()

    lock = _cap_lock(video_id)
    with lock:
        with _lru_lock:
            if key in _frame_lru:
                _frame_lru.move_to_end(key)
                return _frame_lru[key].copy()

        if video_id not in _caps:
            _caps[video_id] = cv2.VideoCapture(str(video_path))
        cap = _caps[video_id]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if not ret or frame is None:
            raise HTTPException(status_code=404, detail="Frame not found")

        store = frame.copy()
        with _lru_lock:
            while len(_frame_lru) >= _FRAME_LRU_MAX:
                _frame_lru.popitem(last=False)
            _frame_lru[key] = store
        return store.copy()


def _resolve_camera_id(state: AppState, video_id: str) -> str:
    """Match ``detections`` router: prefer run’s camera, else filename heuristic."""
    run_id = state.video_to_latest_run.get(video_id)
    camera_id = None
    if run_id and run_id in state.active_runs:
        camera_id = state.active_runs[run_id].get("cameraId")
    if camera_id is None:
        camera_id = _detect_camera_for_video(state.uploaded_videos[video_id], None)
    return str(camera_id or "")


def _load_frame_bgr_aligned_to_tracklets(
    state: AppState, video_id: str, video_path: Path, frame_id: int):
    """Prefer stage-0 PNG/JPG (same pixel space as tracklet bboxes); else raw video.

    Stage 1 bboxes are in the resolution of extracted frames (often resized in
    stage 0).  Cropping the *uploaded* video at full resolution misaligns
    coordinates → garbage / grey / white smears in thumbnails.
    """
    run_id = state.video_to_latest_run.get(video_id)
    cam = _resolve_camera_id(state, video_id)
    if run_id and cam:
        sp = _stage0_frame_path(run_id, cam, frame_id)
        if sp is not None and sp.exists():
            img = cv2.imread(str(sp))
            if img is not None:
                return img
    return _read_frame_bgr(video_id, video_path, frame_id)


def _pad_xyxy(x1: float, y1: float, x2: float, y2: float, w: int, h: int, pad_frac: float):
    """Expand bbox slightly for nicer crops (context around vehicle)."""
    if pad_frac <= 0:
        return int(x1), int(y1), int(x2), int(y2)
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    px = bw * pad_frac
    py = bh * pad_frac
    return (
        max(0, int(x1 - px)),
        max(0, int(y1 - py)),
        min(w, int(x2 + px)),
        min(h, int(y2 + py)),
    )


def _upscale_crop_min_edge(crop_bgr, min_edge: int):
    """Upscale small crops so UI thumbs (fixed CSS size) are not blocky."""
    if min_edge <= 0:
        return crop_bgr
    h, w = int(crop_bgr.shape[0]), int(crop_bgr.shape[1])
    if h <= 0 or w <= 0:
        return crop_bgr
    m = max(h, w)
    if m >= min_edge:
        return crop_bgr
    scale = min_edge / float(m)
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))
    return cv2.resize(crop_bgr, (nw, nh), interpolation=cv2.INTER_LANCZOS4)


@router.get("/api/crops/{video_id}")
async def get_crop(
    video_id: str,
    frameId: int = 0,
    x1: float = 0,
    y1: float = 0,
    x2: float = 0,
    y2: float = 0,
    quality: int = Query(82, ge=40, le=95, description="JPEG quality (smaller = faster)"),
    minEdge: int = Query(
        0,
        ge=0,
        le=640,
        description="If >0, upscale crop so max(w,h) is at least this (clearer tiny vehicles).",
    ),
    pad: float = Query(
        0.12,
        ge=0.0,
        le=0.5,
        description="Fractional padding around bbox before crop (adds context).",
    ),
    state: AppState = Depends(get_app_state),
):
    """Extract a cropped vehicle image from a video frame."""
    if not _HAS_CV2:
        raise HTTPException(status_code=500, detail="OpenCV not available")
    if video_id not in state.uploaded_videos:
        raise HTTPException(status_code=404, detail="Video not found")

    video_path = Path(state.uploaded_videos[video_id]["path"])
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file missing")

    frame = _load_frame_bgr_aligned_to_tracklets(state, video_id, video_path, frameId)

    h, w = frame.shape[:2]
    cx1, cy1, cx2, cy2 = _pad_xyxy(x1, y1, x2, y2, w, h, float(pad))
    cx1 = max(0, cx1)
    cy1 = max(0, cy1)
    cx2 = min(w, cx2)
    cy2 = min(h, cy2)

    if cx2 <= cx1 or cy2 <= cy1:
        raise HTTPException(status_code=400, detail="Invalid bbox")

    crop = frame[cy1:cy2, cx1:cx2]
    crop = _upscale_crop_min_edge(crop, int(minEdge))
    _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    return StreamingResponse(
        io.BytesIO(buf.tobytes()),
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@router.get("/api/crops/run/{run_id}")
async def get_crop_from_run(
    run_id: str,
    cameraId: str = "",
    frameId: int = 0,
    x1: float = 0,
    y1: float = 0,
    x2: float = 0,
    y2: float = 0,
):
    """Extract a cropped vehicle image from a pre-extracted frame in a run directory."""
    if not _HAS_CV2:
        raise HTTPException(status_code=500, detail="OpenCV not available")

    run_dir = OUTPUT_DIR / run_id / "stage0" / cameraId
    if not run_dir.exists():
        fallback_dir = OUTPUT_DIR / "dataset_precompute_s01" / "stage0" / cameraId
        if fallback_dir.exists():
            run_dir = fallback_dir
        else:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Run/camera frames not found. Looked in '{run_dir}' and fallback "
                    f"'{fallback_dir}' for cameraId '{cameraId}'."
                ),
            )

    frame_path_jpg = run_dir / f"frame_{frameId:06d}.jpg"
    frame_path_png = run_dir / f"frame_{frameId:06d}.png"
    if frame_path_jpg.exists():
        frame_path = frame_path_jpg
    elif frame_path_png.exists():
        frame_path = frame_path_png
    else:
        raise HTTPException(
            status_code=404,
            detail=f"Frame {frameId} not found. Looked for '{frame_path_jpg}' and '{frame_path_png}'.",
        )

    frame = cv2.imread(str(frame_path))
    if frame is None:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read frame image from '{frame_path}'",
        )

    h, w = frame.shape[:2]
    cx1 = max(0, int(x1))
    cy1 = max(0, int(y1))
    cx2 = min(w, int(x2))
    cy2 = min(h, int(y2))

    if cx2 <= cx1 or cy2 <= cy1:
        raise HTTPException(status_code=400, detail="Invalid bbox")

    crop = frame[cy1:cy2, cx1:cx2]
    _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return StreamingResponse(
        io.BytesIO(buf.tobytes()),
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=86400"},
    )
