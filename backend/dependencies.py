"""FastAPI dependency providers for the MTMC Tracker backend."""

from backend.state import AppState
from backend.state import app_state as _default_state


def get_app_state() -> AppState:
    """Return the active ``AppState`` for this process."""
    return _default_state
