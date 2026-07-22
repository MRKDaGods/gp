"""SSE tail over a run's ``events.jsonl`` — the typed event pipe.

The file is append-only JSON lines written flush-per-event by the runner;
tailing it is the ONE sanctioned way to watch progress (no stdout parsing,
no second event store). The stream closes itself after a terminal event.

``tail_events`` yields pre-formatted SSE frames so endpoints can validate
(404 etc.) BEFORE constructing the ``EventSourceResponse`` — raising inside
a streaming generator would be too late for a status code.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import AsyncIterable

import anyio
from fastapi import Request

TERMINAL_EVENTS = {"run_completed", "run_failed", "run_cancelled"}
POLL_INTERVAL_S = 0.5


def _read_new_lines(path: Path, offset: int) -> tuple[list[str], int]:
    if not path.is_file():
        return [], offset
    with open(path, "r", encoding="utf-8") as fh:
        fh.seek(offset)
        chunk = fh.read()
        new_offset = fh.tell()
    lines = [line for line in chunk.splitlines() if line.strip()]
    # a partially-flushed last line (no newline yet) is re-read next poll
    if chunk and not chunk.endswith("\n") and lines:
        new_offset -= len(lines[-1].encode("utf-8"))
        lines = lines[:-1]
    return lines, new_offset


def _frame(event_id: int, event: str, data_json: str) -> str:
    return f"event: {event}\nid: {event_id}\ndata: {data_json}\n\n"


async def tail_events(path: Path, request: Request) -> AsyncIterable[str]:
    """Yield each event line as one SSE frame; stop on terminal event or
    client disconnect. Works on completed runs (drains, then closes)."""
    offset = 0
    event_id = 0
    while True:
        if await request.is_disconnected():
            return
        lines, offset = _read_new_lines(path, offset)
        for line in lines:
            data = json.loads(line)
            event_id += 1
            yield _frame(event_id, data.get("event", "message"), json.dumps(data))
            if data.get("event") in TERMINAL_EVENTS:
                return
        await anyio.sleep(POLL_INTERVAL_S)
