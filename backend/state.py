"""Global in-memory state for the MTMC Tracker backend.

Phase 4: All mutable state is encapsulated in ``AppState``.  The
module-level names ``active_runs``, ``uploaded_videos``,
``video_to_latest_run``, and ``RUN_ID_LOCK`` remain importable and
point to the **same** dict / lock objects held by the singleton
``app_state``.  Existing code that mutates these references in-place
(``d[key] = ...``, ``.clear()``, ``.pop()``) continues to work
unchanged.

For router-level dependency injection see ``backend.dependencies``.
"""
import threading
from typing import Any, Dict


class AppState:
    """Injectable container for all mutable in-process pipeline state.

    One singleton (``app_state``) is created at import time.  Routers
    receive it via ``Depends(get_app_state)`` from ``backend.dependencies``.
    Tests can instantiate a fresh ``AppState()`` and install it as an
    override or call ``app_state.reset()`` to clear all state in-place.
    """

    def __init__(self) -> None:
        # Active pipeline runs keyed by run_id
        self.active_runs: Dict[str, Dict[str, Any]] = {}
        # Registered video records keyed by video_id
        self.uploaded_videos: Dict[str, Dict[str, Any]] = {}
        # Latest run_id that processed each video_id
        self.video_to_latest_run: Dict[str, str] = {}
        # Lock for allocating new numeric run IDs
        self.run_id_lock: threading.Lock = threading.Lock()

    def reset(self) -> None:
        """Clear all state dicts in-place.  Safe to call between tests.

        The lock is intentionally preserved — resetting it while another
        thread holds it would cause a deadlock.
        """
        self.active_runs.clear()
        self.uploaded_videos.clear()
        self.video_to_latest_run.clear()


# ---------------------------------------------------------------------------
# Production singleton
# ---------------------------------------------------------------------------
app_state: AppState = AppState()

# ---------------------------------------------------------------------------
# Backward-compatibility aliases
#
# These names are direct references to the *same* dict / lock objects
# inside ``app_state``.  Any call site that imports these names and
# mutates them in-place continues to work without modification.
# ---------------------------------------------------------------------------
active_runs = app_state.active_runs
uploaded_videos = app_state.uploaded_videos
video_to_latest_run = app_state.video_to_latest_run
RUN_ID_LOCK = app_state.run_id_lock
