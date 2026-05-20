from __future__ import annotations

import asyncio
import json
import logging
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from backend.services import kaggle_service
from backend.services.pipeline_service import _materialize_import_tree


log = logging.getLogger(__name__)

TERMINAL_STATUSES = {"complete", "error", "cancelled"}


class KagglePollingWorker:
    """Poll active Kaggle jobs and integrate completed outputs.

    Jobs are discovered from ``kaggle_job.json`` files under
    ``data/outputs/<run_id>/``. The worker does not read request-time Kaggle
    credentials from disk because Phase 10 deliberately never persists them;
    polling and output download therefore use server-default Kaggle credentials
    from ``~/.kaggle/kaggle.json`` after a backend restart.

    Idempotent: a job in terminal status with ``outputs_downloaded_to`` already
    set is skipped.
    """

    POLL_INTERVAL_SECONDS = 60
    BACKOFF_MAX_RETRIES = 5

    def __init__(self, output_root: Path, websocket_manager: Any) -> None:
        self.output_root = Path(output_root)
        self.websocket_manager = websocket_manager
        self._running = False
        self._task: Optional[asyncio.Task[None]] = None
        self._consecutive_failures: Dict[str, int] = {}

    async def start(self) -> None:
        """Start the polling task. Idempotent if already running."""
        if self._running and self._task is not None and not self._task.done():
            return
        self._running = True
        self._task = asyncio.create_task(self._loop(), name="kaggle-polling-worker")

    async def stop(self) -> None:
        """Cancel the polling task and wait for clean shutdown."""
        self._running = False
        if self._task is None:
            return
        task = self._task
        self._task = None
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def poll_once(self) -> Dict[str, str]:
        """Run one polling pass and return ``run_id -> new_status`` changes."""
        status_changes: Dict[str, str] = {}
        for metadata_path in sorted(self.output_root.glob("*/kaggle_job.json")):
            run_id = metadata_path.parent.name
            try:
                state = self._read_state(metadata_path)
            except (OSError, json.JSONDecodeError) as exc:
                log.warning("Skipping unreadable Kaggle job state at %s: %s", metadata_path, exc)
                continue

            if self._is_terminal_downloaded(state):
                continue

            try:
                new_status = await self._poll_job(run_id, state)
            except Exception as exc:
                failures = self._consecutive_failures.get(run_id, 0) + 1
                self._consecutive_failures[run_id] = failures
                log.exception("Polling Kaggle job failed for run_id=%s", run_id)
                if failures >= self.BACKOFF_MAX_RETRIES:
                    state["status"] = "error"
                    state["last_polled_at"] = _utc_now_iso()
                    state["error"] = f"Kaggle polling failed after {failures} attempts: {exc}"
                    self._write_state(run_id, state)
                    await self._emit_status(run_id, state)
                    status_changes[run_id] = "error"
                continue

            self._consecutive_failures[run_id] = 0
            if new_status is not None:
                status_changes[run_id] = new_status
        return status_changes

    async def _poll_job(self, run_id: str, state: Dict[str, Any]) -> Optional[str]:
        """Poll one job with server-default Kaggle credentials.

        Credentials are not stored in ``kaggle_job.json``. If the original push
        used request-scoped credentials, this polling pass still uses the
        backend server's ``~/.kaggle/kaggle.json`` fallback.
        """
        status = await asyncio.to_thread(
            kaggle_service.kernel_status,
            str(state["kernel_slug"]),
        )
        previous_status = state.get("status")
        state["last_polled_at"] = status.last_polled_iso

        if status.status == previous_status:
            if status.status in TERMINAL_STATUSES and not state.get("outputs_downloaded_to"):
                await self._on_terminal(run_id, state)
                self._write_state(run_id, state)
                await self._emit_status(run_id, state)
                if state.get("status") != previous_status:
                    return str(state["status"])
            self._write_state(run_id, state)
            return None

        state["status"] = status.status
        if status.status in TERMINAL_STATUSES:
            await self._on_terminal(run_id, state)
        self._write_state(run_id, state)
        await self._emit_status(run_id, state)
        return str(state["status"])

    async def _on_terminal(self, run_id: str, state: Dict[str, Any]) -> None:
        """Handle terminal Kaggle status by downloading and integrating outputs."""
        status = state.get("status")
        if status == "cancelled":
            return
        if status == "error":
            state.setdefault("error", "Kaggle kernel failed before producing outputs.")
            return
        if status != "complete":
            return

        with tempfile.TemporaryDirectory(prefix=f"mtmc-kaggle-{run_id}-") as tmp_dir:
            temp_dir = Path(tmp_dir)
            await asyncio.to_thread(
                kaggle_service.kernel_output,
                str(state["kernel_slug"]),
                temp_dir,
            )
            manifest_path = _find_manifest(temp_dir)
            output_zip_path = _find_output_zip(temp_dir, run_id)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            extract_root = temp_dir / "extracted"
            extract_root.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(output_zip_path, "r") as archive:
                archive.extractall(extract_root)

            run_dir = self.output_root / run_id
            _materialize_import_tree(extract_root, run_dir)

        state["outputs_downloaded_to"] = str((self.output_root / run_id).resolve())
        state["exit_code"] = manifest.get("exit_code")
        if state["exit_code"] != 0:
            state["status"] = "error"
            state["error"] = str(manifest.get("error") or "Kaggle kernel exited non-zero.")

    async def _emit_status(self, run_id: str, state: Dict[str, Any]) -> None:
        """Send a websocket status event to the run channel when available."""
        if self.websocket_manager is None:
            # TODO(Phase 14): integrate websocket events when ws manager is added.
            return
        send_to_run = getattr(self.websocket_manager, "send_to_run", None)
        if send_to_run is None:
            return
        await send_to_run(
            run_id,
            {
                "type": "kaggle_status",
                "run_id": run_id,
                "kernel_slug": state["kernel_slug"],
                "status": state["status"],
                "exit_code": state.get("exit_code"),
                "outputs_downloaded_to": state.get("outputs_downloaded_to"),
                "last_polled_at": state["last_polled_at"],
                "error": state.get("error"),
            },
        )

    async def _loop(self) -> None:
        while self._running:
            try:
                await self.poll_once()
            except Exception:
                log.exception("Polling pass failed")
            await asyncio.sleep(self.POLL_INTERVAL_SECONDS)

    def _read_state(self, metadata_path: Path) -> Dict[str, Any]:
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    def _write_state(self, run_id: str, state: Dict[str, Any]) -> None:
        run_dir = self.output_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "kaggle_job.json").write_text(
            json.dumps(state, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )

    def _is_terminal_downloaded(self, state: Dict[str, Any]) -> bool:
        return state.get("status") in TERMINAL_STATUSES and bool(state.get("outputs_downloaded_to"))


def _find_manifest(download_dir: Path) -> Path:
    matches = sorted(download_dir.rglob("manifest.json"))
    if not matches:
        raise FileNotFoundError(f"No manifest.json found in Kaggle output {download_dir}")
    return matches[0]


def _find_output_zip(download_dir: Path, run_id: str) -> Path:
    preferred_name = f"{run_id}_outputs.zip"
    preferred = sorted(path for path in download_dir.rglob(preferred_name) if path.is_file())
    if preferred:
        return preferred[0]
    zips = sorted(path for path in download_dir.rglob("*.zip") if path.is_file())
    if not zips:
        raise FileNotFoundError(f"No output zip found in Kaggle output {download_dir}")
    return zips[0]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")