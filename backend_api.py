import sys as _sys
import asyncio

if _sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from backend.app import app  # noqa: F401
