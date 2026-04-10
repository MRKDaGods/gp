"""FastAPI dependency providers for the MTMC Tracker backend.

Usage in router handlers::

    from fastapi import Depends
    from backend.dependencies import get_app_state
    from backend.state import AppState

    @router.post("/api/something")
    async def handler(..., state: AppState = Depends(get_app_state)):
        state.uploaded_videos[video_id] = ...

Test isolation::

    from backend.dependencies import get_app_state
    from backend.state import AppState

    test_state = AppState()
    app.dependency_overrides[get_app_state] = lambda: test_state
    # ... run test ...
    app.dependency_overrides.clear()
"""

from backend.state import AppState
from backend.state import app_state as _default_state


def get_app_state() -> AppState:
    """Return the active ``AppState`` for this process.

    In production this always returns the module-level singleton defined
    in ``backend.state``.  In tests, override with::

        app.dependency_overrides[get_app_state] = lambda: my_test_state
    """
    return _default_state
