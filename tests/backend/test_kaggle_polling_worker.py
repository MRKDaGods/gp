from __future__ import annotations

import asyncio
import json
import zipfile
from pathlib import Path
from unittest import mock

import pytest

from backend.services.kaggle_polling_worker import KagglePollingWorker
from backend.services.kaggle_service import KernelOutputResult, KernelStatusResult


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _job_state(run_id: str, status: str = "queued", **overrides: object) -> dict[str, object]:
    state: dict[str, object] = {
        "run_id": run_id,
        "kernel_slug": f"gumfreddy/{run_id}",
        "kernel_url": f"https://www.kaggle.com/code/gumfreddy/{run_id}",
        "dataset_slug": "owner/dataset",
        "project_dataset_slug": "gumfreddy/mtmc-tracker-source",
        "status": status,
        "stages": [1],
        "started_at": "2026-05-20T00:00:00Z",
        "last_polled_at": None,
        "exit_code": None,
        "outputs_downloaded_to": None,
    }
    state.update(overrides)
    return state


def _write_job(output_root: Path, run_id: str, state: dict[str, object]) -> Path:
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True)
    path = run_dir / "kaggle_job.json"
    path.write_text(json.dumps(state), encoding="utf-8")
    return path


def _read_job(output_root: Path, run_id: str) -> dict[str, object]:
    return json.loads((output_root / run_id / "kaggle_job.json").read_text(encoding="utf-8"))


def _status(run_id: str, status: str) -> KernelStatusResult:
    return KernelStatusResult(
        slug=f"gumfreddy/{run_id}",
        status=status,
        raw_stdout=status,
        last_polled_iso="2026-05-20T12:00:00Z",
    )


@pytest.mark.anyio
async def test_worker_start_is_idempotent(tmp_path: Path) -> None:
    worker = KagglePollingWorker(tmp_path, websocket_manager=None)

    await worker.start()
    first_task = worker._task
    await worker.start()

    assert worker._task is first_task
    await worker.stop()


@pytest.mark.anyio
async def test_worker_stop_cancels_task_cleanly(tmp_path: Path) -> None:
    worker = KagglePollingWorker(tmp_path, websocket_manager=None)

    await worker.start()
    task = worker._task
    await worker.stop()

    assert worker._running is False
    assert worker._task is None
    assert task is not None
    assert task.cancelled() or task.done()


@pytest.mark.anyio
async def test_poll_once_with_no_jobs_returns_empty_map(tmp_path: Path) -> None:
    worker = KagglePollingWorker(tmp_path, websocket_manager=None)

    assert await worker.poll_once() == {}


@pytest.mark.anyio
async def test_poll_once_running_update_persists_without_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_job(tmp_path, "run-42", _job_state("run-42", status="queued"))
    worker = KagglePollingWorker(tmp_path, websocket_manager=None)
    on_terminal = mock.AsyncMock()
    monkeypatch.setattr(worker, "_on_terminal", on_terminal)
    monkeypatch.setattr(
        "backend.services.kaggle_polling_worker.kaggle_service.kernel_status",
        mock.Mock(return_value=_status("run-42", "running")),
    )

    changes = await worker.poll_once()

    assert changes == {"run-42": "running"}
    persisted = _read_job(tmp_path, "run-42")
    assert persisted["status"] == "running"
    assert persisted["last_polled_at"] == "2026-05-20T12:00:00Z"
    on_terminal.assert_not_awaited()


@pytest.mark.anyio
async def test_poll_once_complete_downloads_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_job(tmp_path, "run-42", _job_state("run-42", status="running"))
    worker = KagglePollingWorker(tmp_path, websocket_manager=None)
    monkeypatch.setattr(
        "backend.services.kaggle_polling_worker.kaggle_service.kernel_status",
        mock.Mock(return_value=_status("run-42", "complete")),
    )

    def fake_kernel_output(slug: str, dest_dir: Path) -> KernelOutputResult:
        manifest = dest_dir / "manifest.json"
        manifest.write_text(json.dumps({"exit_code": 0}), encoding="utf-8")
        zip_path = dest_dir / "run-42_outputs.zip"
        with zipfile.ZipFile(zip_path, "w") as archive:
            archive.writestr("stage1/tracklets.json", "[]")
        return KernelOutputResult(
            downloaded_files=[manifest, zip_path],
            total_bytes=manifest.stat().st_size + zip_path.stat().st_size,
            raw_stdout="downloaded",
        )

    monkeypatch.setattr(
        "backend.services.kaggle_polling_worker.kaggle_service.kernel_output",
        fake_kernel_output,
    )

    changes = await worker.poll_once()

    assert changes == {"run-42": "complete"}
    persisted = _read_job(tmp_path, "run-42")
    assert persisted["status"] == "complete"
    assert persisted["exit_code"] == 0
    assert persisted["outputs_downloaded_to"] == str((tmp_path / "run-42").resolve())
    assert (tmp_path / "run-42" / "stage1" / "tracklets.json").read_text() == "[]"


@pytest.mark.anyio
async def test_poll_once_terminal_job_already_downloaded_is_skipped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_job(
        tmp_path,
        "run-42",
        _job_state("run-42", status="complete", outputs_downloaded_to=str(tmp_path / "run-42")),
    )
    kernel_status = mock.Mock()
    monkeypatch.setattr(
        "backend.services.kaggle_polling_worker.kaggle_service.kernel_status",
        kernel_status,
    )
    worker = KagglePollingWorker(tmp_path, websocket_manager=None)

    assert await worker.poll_once() == {}
    kernel_status.assert_not_called()


@pytest.mark.anyio
async def test_poll_once_complete_without_download_integrates_outputs_when_status_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_job(tmp_path, "run-42", _job_state("run-42", status="complete"))
    worker = KagglePollingWorker(tmp_path, websocket_manager=None)
    monkeypatch.setattr(
        "backend.services.kaggle_polling_worker.kaggle_service.kernel_status",
        mock.Mock(return_value=_status("run-42", "complete")),
    )

    def fake_kernel_output(slug: str, dest_dir: Path) -> KernelOutputResult:
        manifest = dest_dir / "manifest.json"
        manifest.write_text(json.dumps({"exit_code": 0}), encoding="utf-8")
        zip_path = dest_dir / "run-42_outputs.zip"
        with zipfile.ZipFile(zip_path, "w") as archive:
            archive.writestr("stage1/tracklets.json", "[]")
        return KernelOutputResult(downloaded_files=[manifest, zip_path], total_bytes=1, raw_stdout="")

    monkeypatch.setattr(
        "backend.services.kaggle_polling_worker.kaggle_service.kernel_output",
        fake_kernel_output,
    )

    assert await worker.poll_once() == {}
    persisted = _read_job(tmp_path, "run-42")
    assert persisted["outputs_downloaded_to"] == str((tmp_path / "run-42").resolve())
    assert (tmp_path / "run-42" / "stage1" / "tracklets.json").exists()


@pytest.mark.anyio
async def test_poll_job_exception_increments_consecutive_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_job(tmp_path, "run-42", _job_state("run-42"))
    worker = KagglePollingWorker(tmp_path, websocket_manager=None)
    monkeypatch.setattr(worker, "_poll_job", mock.AsyncMock(side_effect=RuntimeError("boom")))

    assert await worker.poll_once() == {}
    assert worker._consecutive_failures["run-42"] == 1
    assert _read_job(tmp_path, "run-42")["status"] == "queued"


@pytest.mark.anyio
async def test_five_consecutive_failures_mark_error_and_persist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_job(tmp_path, "run-42", _job_state("run-42"))
    worker = KagglePollingWorker(tmp_path, websocket_manager=None)
    monkeypatch.setattr(worker, "_poll_job", mock.AsyncMock(side_effect=RuntimeError("boom")))

    changes: dict[str, str] = {}
    for _ in range(worker.BACKOFF_MAX_RETRIES):
        changes = await worker.poll_once()

    persisted = _read_job(tmp_path, "run-42")
    assert changes == {"run-42": "error"}
    assert persisted["status"] == "error"
    assert "failed after 5 attempts" in str(persisted["error"])


@pytest.mark.anyio
async def test_websocket_event_sent_on_status_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_job(tmp_path, "run-42", _job_state("run-42", status="queued"))
    ws_manager = mock.Mock()
    ws_manager.send_to_run = mock.AsyncMock()
    worker = KagglePollingWorker(tmp_path, websocket_manager=ws_manager)
    monkeypatch.setattr(
        "backend.services.kaggle_polling_worker.kaggle_service.kernel_status",
        mock.Mock(return_value=_status("run-42", "running")),
    )

    await worker.poll_once()

    ws_manager.send_to_run.assert_awaited_once()
    run_id, event = ws_manager.send_to_run.await_args.args
    assert run_id == "run-42"
    assert event["type"] == "kaggle_status"
    assert event["status"] == "running"
    assert event["kernel_slug"] == "gumfreddy/run-42"


@pytest.mark.anyio
async def test_websocket_event_not_sent_when_status_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_job(tmp_path, "run-42", _job_state("run-42", status="running"))
    ws_manager = mock.Mock()
    ws_manager.send_to_run = mock.AsyncMock()
    worker = KagglePollingWorker(tmp_path, websocket_manager=ws_manager)
    monkeypatch.setattr(
        "backend.services.kaggle_polling_worker.kaggle_service.kernel_status",
        mock.Mock(return_value=_status("run-42", "running")),
    )

    assert await worker.poll_once() == {}
    persisted = _read_job(tmp_path, "run-42")
    assert persisted["last_polled_at"] == "2026-05-20T12:00:00Z"
    ws_manager.send_to_run.assert_not_awaited()


@pytest.mark.anyio
async def test_output_integration_extracts_zip_into_run_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = KagglePollingWorker(tmp_path, websocket_manager=None)
    state = _job_state("run-42", status="complete")

    def fake_kernel_output(slug: str, dest_dir: Path) -> KernelOutputResult:
        manifest = dest_dir / "manifest.json"
        manifest.write_text(json.dumps({"exit_code": 0}), encoding="utf-8")
        zip_path = dest_dir / "run-42_outputs.zip"
        with zipfile.ZipFile(zip_path, "w") as archive:
            archive.writestr("nested/stage2/features.json", "{\"ok\": true}")
        return KernelOutputResult(
            downloaded_files=[manifest, zip_path],
            total_bytes=manifest.stat().st_size + zip_path.stat().st_size,
            raw_stdout="downloaded",
        )

    monkeypatch.setattr(
        "backend.services.kaggle_polling_worker.kaggle_service.kernel_output",
        fake_kernel_output,
    )

    await worker._on_terminal("run-42", state)

    assert state["exit_code"] == 0
    assert state["outputs_downloaded_to"] == str((tmp_path / "run-42").resolve())
    assert json.loads((tmp_path / "run-42" / "stage2" / "features.json").read_text())["ok"] is True


@pytest.mark.anyio
async def test_complete_with_nonzero_manifest_marks_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = KagglePollingWorker(tmp_path, websocket_manager=None)
    state = _job_state("run-42", status="complete")

    def fake_kernel_output(slug: str, dest_dir: Path) -> KernelOutputResult:
        manifest = dest_dir / "manifest.json"
        manifest.write_text(json.dumps({"exit_code": 2, "error": "stage failed"}), encoding="utf-8")
        zip_path = dest_dir / "run-42_outputs.zip"
        with zipfile.ZipFile(zip_path, "w") as archive:
            archive.writestr("stage1/log.txt", "failed")
        return KernelOutputResult(downloaded_files=[manifest, zip_path], total_bytes=1, raw_stdout="")

    monkeypatch.setattr(
        "backend.services.kaggle_polling_worker.kaggle_service.kernel_output",
        fake_kernel_output,
    )

    await worker._on_terminal("run-42", state)

    assert state["status"] == "error"
    assert state["exit_code"] == 2
    assert state["error"] == "stage failed"


@pytest.mark.anyio
async def test_stop_without_start_is_noop(tmp_path: Path) -> None:
    worker = KagglePollingWorker(tmp_path, websocket_manager=None)

    await worker.stop()

    assert worker._running is False
