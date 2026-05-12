"""FastAPI application factory for the MTMC Tracker backend.

All routers are registered here. The startup event initialises the
in-memory video catalogue and optionally triggers the S01 precompute.
"""
import asyncio
import sys as _sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import UPLOAD_DIR, OUTPUT_DIR
from backend.services.pipeline_service import _background_precompute_dataset
from backend.services.video_service import _scan_startup_videos

# ── Routers ────────────────────────────────────────────────────────────────
from backend.routers import (
    crops,
    datasets,
    detections,
    export,
    frames,
    health,
    locations,
    pipeline,
    results,
    runs,
    search,
    timeline,
    tracklets,
    videos,
)

# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(title="MTMC Tracker API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure upload and output directories exist at startup
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Include routers ────────────────────────────────────────────────────────
app.include_router(health.router)
app.include_router(locations.router)
app.include_router(results.router)
app.include_router(tracklets.router)
app.include_router(export.router)
app.include_router(crops.router)
app.include_router(videos.router)
app.include_router(detections.router)
app.include_router(frames.router)
app.include_router(runs.router)
app.include_router(datasets.router)
app.include_router(pipeline.router)
app.include_router(search.router)
app.include_router(timeline.router)


# ── Startup event ──────────────────────────────────────────────────────────
@app.on_event("startup")
async def _on_startup() -> None:
    # Suppress Windows-specific "ConnectionResetError: [WinError 10054]" spam
    # that fires when browsers close video range-request connections mid-stream.
    if _sys.platform == "win32":
        def _win_exc_handler(loop, context):
            exc = context.get("exception")
            if isinstance(exc, (ConnectionResetError, BrokenPipeError)):
                return  # harmless — browser closed the socket
            loop.default_exception_handler(context)

        asyncio.get_event_loop().set_exception_handler(_win_exc_handler)

    _scan_startup_videos()
    asyncio.create_task(_background_precompute_dataset())
