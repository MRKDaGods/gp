from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

from backend.models.requests import KaggleConfig, StageExecutionTarget
from backend.services.kaggle_run_service import (
    dispatch_stage_to_kaggle,
    get_kaggle_job_state,
    refresh_kaggle_job_status,
)
from backend.services.kaggle_service import (
    KaggleConcurrencyError,
    KernelPushResult,
    KernelStatusResult,
)


def _kaggle_cfg(**kwargs: object) -> KaggleConfig:
    return KaggleConfig(target=StageExecutionTarget.KAGGLE, **kwargs)


def _patch_dispatch_deps(monkeypatch: pytest.MonkeyPatch, parent: mock.Mock) -> None:
    whoami = mock.Mock(return_value="gumfreddy")
    count_active = mock.Mock(return_value=0)
    dataset_upsert = mock.Mock()
    render = mock.Mock(side_effect=lambda ctx, output_dir: output_dir)
    push = mock.Mock(
        return_value=KernelPushResult(
            slug="gumfreddy/mtmc-run-42-stage-1",
            kernel_url="https://www.kaggle.com/code/gumfreddy/mtmc-run-42-stage-1",
            raw_stdout="pushed",
        )
    )
    build_source = mock.Mock(
        side_effect=lambda dest_dir: dest_dir.mkdir(parents=True, exist_ok=True)
    )

    parent.attach_mock(whoami, "whoami")
    parent.attach_mock(count_active, "count_active")
    parent.attach_mock(dataset_upsert, "dataset_upsert")
    parent.attach_mock(render, "render")
    parent.attach_mock(push, "push")

    monkeypatch.setattr("backend.services.kaggle_run_service.kaggle_service.whoami", whoami)
    monkeypatch.setattr(
        "backend.services.kaggle_run_service.kaggle_service.count_active_kernels",
        count_active,
    )
    monkeypatch.setattr(
        "backend.services.kaggle_run_service.kaggle_service.dataset_create_or_update",
        dataset_upsert,
    )
    monkeypatch.setattr(
        "backend.services.kaggle_run_service.kaggle_kernel_templates.render_kernel",
        render,
    )
    monkeypatch.setattr("backend.services.kaggle_run_service.kaggle_service.push_kernel", push)
    monkeypatch.setattr(
        "backend.services.kaggle_run_service._build_project_source_archive",
        build_source,
    )


def test_dispatch_stage_to_kaggle_happy_path_writes_job_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = mock.Mock()
    _patch_dispatch_deps(monkeypatch, parent)

    result = dispatch_stage_to_kaggle(
        run_id="run-42",
        stages=[1],
        config_path="configs/datasets/cityflowv2.yaml",
        config_overrides=["stage4.association.query_expansion.k=2"],
        model_id="vehicle_mtmc_14e_b1",
        fusion={"models": []},
        kaggle_cfg=_kaggle_cfg(
            username="request-user",
            key="request-key",
            datasetSlug="owner/dataset",
        ),
        output_root=tmp_path,
    )

    assert result.run_id == "run-42"
    assert result.kernel_slug == "gumfreddy/mtmc-run-42-stage-1"
    assert result.dataset_slug == "owner/dataset"
    assert result.project_dataset_slug == "gumfreddy/mtmc-tracker-source"
    assert result.status == "queued"

    state = json.loads((tmp_path / "run-42" / "kaggle_job.json").read_text(encoding="utf-8"))
    assert state["status"] == "queued"
    assert state["stages"] == [1]
    assert state["last_polled_at"] is None
    assert state["outputs_downloaded_to"] is None

    assert [call[0] for call in parent.mock_calls] == [
        "whoami",
        "count_active",
        "dataset_upsert",
        "render",
        "push",
    ]


def test_dispatch_stage_to_kaggle_rejects_concurrency_before_push(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = mock.Mock()
    _patch_dispatch_deps(monkeypatch, parent)
    parent.count_active.return_value = 2

    with pytest.raises(KaggleConcurrencyError):
        dispatch_stage_to_kaggle(
            run_id="busy-run",
            stages=[1],
            config_path="configs/default.yaml",
            config_overrides=[],
            model_id=None,
            fusion=None,
            kaggle_cfg=_kaggle_cfg(username="u", key="k", datasetSlug="owner/dataset"),
            output_root=tmp_path,
        )

    parent.push.assert_not_called()
    parent.render.assert_not_called()


def test_preexisting_dataset_slug_skips_video_dataset_upload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = mock.Mock()
    _patch_dispatch_deps(monkeypatch, parent)

    dispatch_stage_to_kaggle(
        run_id="preexisting",
        stages=[4],
        config_path="configs/datasets/cityflowv2.yaml",
        config_overrides=[],
        model_id=None,
        fusion=None,
        kaggle_cfg=_kaggle_cfg(
            username="u",
            key="k",
            datasetSlug="thanhnguyenle/data-aicity-2023-track-2",
        ),
        output_root=tmp_path,
    )

    assert parent.dataset_upsert.call_count == 1
    assert parent.dataset_upsert.call_args.args[0] == "gumfreddy/mtmc-tracker-source"


def test_auto_upload_video_uses_deterministic_slug(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = mock.Mock()
    _patch_dispatch_deps(monkeypatch, parent)
    video_path = tmp_path / "S02_c008.avi"
    video_path.write_bytes(b"video")

    result = dispatch_stage_to_kaggle(
        run_id="Auto Run 42",
        stages=[1],
        config_path="configs/default.yaml",
        config_overrides=[],
        model_id=None,
        fusion=None,
        kaggle_cfg=_kaggle_cfg(username="u", key="k"),
        user_video_path=video_path,
        output_root=tmp_path,
    )

    assert result.dataset_slug == "gumfreddy/mtmc-user-video-auto-run-42"
    assert parent.dataset_upsert.call_count == 2
    assert (
        parent.dataset_upsert.call_args_list[1].args[0]
        == "gumfreddy/mtmc-user-video-auto-run-42"
    )


def test_missing_dataset_slug_and_video_path_raises_value_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = mock.Mock()
    _patch_dispatch_deps(monkeypatch, parent)

    with pytest.raises(ValueError, match="Need either dataset_slug or user_video_path"):
        dispatch_stage_to_kaggle(
            run_id="missing-input",
            stages=[1],
            config_path="configs/default.yaml",
            config_overrides=[],
            model_id=None,
            fusion=None,
            kaggle_cfg=_kaggle_cfg(username="u", key="k"),
            output_root=tmp_path,
        )


def test_request_credentials_win_over_kaggle_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = mock.Mock()
    _patch_dispatch_deps(monkeypatch, parent)
    home = tmp_path / "home"
    kaggle_dir = home / ".kaggle"
    kaggle_dir.mkdir(parents=True)
    (kaggle_dir / "kaggle.json").write_text(
        json.dumps({"username": "server-user", "key": "server-key"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(Path, "home", lambda: home)

    dispatch_stage_to_kaggle(
        run_id="creds-request",
        stages=[4],
        config_path="configs/default.yaml",
        config_overrides=[],
        model_id=None,
        fusion=None,
        kaggle_cfg=_kaggle_cfg(
            username="request-user",
            key="request-key",
            datasetSlug="owner/dataset",
        ),
        output_root=tmp_path,
    )

    assert parent.whoami.call_args.kwargs == {"username": "request-user", "key": "request-key"}


def test_kaggle_json_credentials_win_over_neither(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = mock.Mock()
    _patch_dispatch_deps(monkeypatch, parent)
    home = tmp_path / "home"
    kaggle_dir = home / ".kaggle"
    kaggle_dir.mkdir(parents=True)
    (kaggle_dir / "kaggle.json").write_text(
        json.dumps({"username": "server-user", "key": "server-key"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(Path, "home", lambda: home)

    dispatch_stage_to_kaggle(
        run_id="creds-server",
        stages=[4],
        config_path="configs/default.yaml",
        config_overrides=[],
        model_id=None,
        fusion=None,
        kaggle_cfg=_kaggle_cfg(datasetSlug="owner/dataset"),
        output_root=tmp_path,
    )

    assert parent.whoami.call_args.kwargs == {"username": "server-user", "key": "server-key"}


def test_get_kaggle_job_state_returns_none_for_unknown_run(tmp_path: Path) -> None:
    assert get_kaggle_job_state("unknown", output_root=tmp_path) is None


def test_refresh_kaggle_job_status_updates_and_persists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run-42"
    run_dir.mkdir()
    state = {
        "run_id": "run-42",
        "kernel_slug": "gumfreddy/mtmc-run-42-stage-1",
        "kernel_url": "https://www.kaggle.com/code/gumfreddy/mtmc-run-42-stage-1",
        "dataset_slug": "owner/dataset",
        "project_dataset_slug": "gumfreddy/mtmc-tracker-source",
        "status": "queued",
        "stages": [1],
        "started_at": "2026-05-20T00:00:00Z",
        "last_polled_at": None,
        "exit_code": None,
        "outputs_downloaded_to": None,
    }
    (run_dir / "kaggle_job.json").write_text(json.dumps(state), encoding="utf-8")

    monkeypatch.setattr(
        "backend.services.kaggle_run_service.kaggle_service.kernel_status",
        mock.Mock(
            return_value=KernelStatusResult(
                slug="gumfreddy/mtmc-run-42-stage-1",
                status="running",
                raw_stdout="running",
                last_polled_iso="2026-05-20T12:00:00Z",
            )
        ),
    )

    refreshed = refresh_kaggle_job_status("run-42", output_root=tmp_path)

    assert refreshed["status"] == "running"
    assert refreshed["last_polled_at"] == "2026-05-20T12:00:00Z"
    persisted = json.loads((run_dir / "kaggle_job.json").read_text(encoding="utf-8"))
    assert persisted == refreshed
