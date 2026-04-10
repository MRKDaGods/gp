"""Debug/logging helpers for the timeline query system.

This module is a dependency leaf — it imports only from backend.config
so other service modules can import _timeline_debug without circular deps.
"""
from datetime import datetime
from typing import Any, Dict, Optional

from backend.config import TIMELINE_DEBUG_LOG


def _timeline_debug(message: str, payload: Optional[Dict[str, Any]] = None) -> None:
    """Emit timeline query debug info to both stdout and a persistent log file."""
    text = message if payload is None else f"{message} {payload}"
    print(text, flush=True)
    try:
        TIMELINE_DEBUG_LOG.parent.mkdir(parents=True, exist_ok=True)
        with TIMELINE_DEBUG_LOG.open("a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat()} {text}\n")
    except Exception:
        pass
