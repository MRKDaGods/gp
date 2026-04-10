import io
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from backend.config import _HAS_CV2, OUTPUT_DIR
from backend.state import uploaded_videos

if _HAS_CV2:
    import cv2

router = APIRouter()


@router.get("/api/crops/{video_id}")
async def get_crop(
    video_id: str,
    frameId: int = 0,
    x1: float = 0,
    y1: float = 0,
    x2: float = 0,
    y2: float = 0,
):
    """Extract a cropped vehicle image from a video frame."""
    if not _HAS_CV2:
        raise HTTPException(status_code=500, detail="OpenCV not available")
    if video_id not in uploaded_videos:
        raise HTTPException(status_code=404, detail="Video not found")

    video_path = Path(uploaded_videos[video_id]["path"])
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file missing")

    cap = cv2.VideoCapture(str(video_path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frameId)
        ret, frame = cap.read()
        if not ret or frame is None:
            raise HTTPException(status_code=404, detail="Frame not found")

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
    finally:
        cap.release()


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
