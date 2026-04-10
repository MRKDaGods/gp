"""Global in-memory state shared across all router and service modules.

Phase 4 will replace these plain dicts with injectable state containers.
For now (Phase 2) they remain module-level globals imported directly.
"""
import threading
from typing import Any, Dict

# Active pipeline runs keyed by run_id
active_runs: Dict[str, Dict[str, Any]] = {}

# Registered video records keyed by video_id
uploaded_videos: Dict[str, Dict[str, Any]] = {}

# Latest run_id that processed each video_id
video_to_latest_run: Dict[str, str] = {}

# Lock for allocating new numeric run IDs
RUN_ID_LOCK = threading.Lock()
