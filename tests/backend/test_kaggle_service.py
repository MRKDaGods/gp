from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

from backend.services.kaggle_service import (
    KaggleAuthError,
    KaggleConcurrencyError,
    KaggleValidationError,
    cancel_kernel,
    count_active_kernels,
    dataset_create_or_update,
    kaggle_env,
    kernel_output,
    kernel_status,
    push_kernel,
)


def _completed(returncode: int = 0, stdout: str = "", stderr: str = "") -> mock.MagicMock:
    completed = mock.MagicMock()
    completed.returncode = returncode
    completed.stdout = stdout
    completed.stderr = stderr
    return completed


def test_kaggle_env_with_creds_writes_temp_json_and_cleans_up() -> None:
    with kaggle_env(username="gumfreddy", key="secret-key") as env:
        config_dir = Path(env["KAGGLE_CONFIG_DIR"])
        credential_path = config_dir / "kaggle.json"
        assert credential_path.exists()
        assert json.loads(credential_path.read_text(encoding="utf-8")) == {
            "username": "gumfreddy",
            "key": "secret-key",
        }

    assert not config_dir.exists()


def test_kaggle_env_with_no_creds_passes_through_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MTMC_TEST_ENV", "present")
    monkeypatch.delenv("KAGGLE_CONFIG_DIR", raising=False)

    with kaggle_env() as env:
        assert env["MTMC_TEST_ENV"] == "present"
        assert "KAGGLE_CONFIG_DIR" not in env


def test_kaggle_env_with_only_username_raises_auth_error() -> None:
    with pytest.raises(KaggleAuthError):
        with kaggle_env(username="gumfreddy"):
            pass


def test_push_kernel_happy_path_parses_slug(tmp_path: Path) -> None:
    stdout = (
        "Kernel version 1 successfully pushed.\n"
        "Please check progress at https://www.kaggle.com/code/gumfreddy/mtmc-run-42\n"
    )
    with mock.patch(
        "backend.services.kaggle_service.subprocess.run",
        return_value=_completed(stdout=stdout),
    ) as run_mock:
        result = push_kernel(tmp_path)

    assert result.slug == "gumfreddy/mtmc-run-42"
    assert result.kernel_url == "https://www.kaggle.com/code/gumfreddy/mtmc-run-42"
    assert result.raw_stdout == stdout
    run_mock.assert_called_once()
    assert run_mock.call_args.args[0] == ["kaggle", "kernels", "push", "-p", str(tmp_path)]
    assert run_mock.call_args.kwargs["shell"] is False


def test_push_kernel_invalid_dataset_sources_raises_validation(tmp_path: Path) -> None:
    with mock.patch(
        "backend.services.kaggle_service.subprocess.run",
        return_value=_completed(stdout="The following are not valid dataset sources: bad/source"),
    ):
        with pytest.raises(KaggleValidationError):
            push_kernel(tmp_path)


def test_push_kernel_401_error_raises_auth(tmp_path: Path) -> None:
    with mock.patch(
        "backend.services.kaggle_service.subprocess.run",
        return_value=_completed(returncode=1, stderr="401 Unauthorized"),
    ):
        with pytest.raises(KaggleAuthError):
            push_kernel(tmp_path)


def test_push_kernel_concurrency_error_raises_concurrency(tmp_path: Path) -> None:
    with mock.patch(
        "backend.services.kaggle_service.subprocess.run",
        return_value=_completed(returncode=1, stderr="Maximum number of GPU sessions reached"),
    ):
        with pytest.raises(KaggleConcurrencyError):
            push_kernel(tmp_path)


@pytest.mark.parametrize(
    ("stdout", "expected"),
    [
        ("Kernel status: queued", "queued"),
        ("Kernel status: running", "running"),
        ("Kernel status: complete", "complete"),
        ("Kernel status: error", "error"),
        ("Kernel status: cancelled", "cancelled"),
    ],
)
def test_kernel_status_with_each_status_string(stdout: str, expected: str) -> None:
    with mock.patch(
        "backend.services.kaggle_service.subprocess.run",
        return_value=_completed(stdout=stdout),
    ):
        result = kernel_status("gumfreddy/mtmc-run-42")

    assert result.slug == "gumfreddy/mtmc-run-42"
    assert result.status == expected
    assert result.raw_stdout == stdout
    assert result.last_polled_iso.endswith("Z")


def test_kernel_status_with_garbage_output_returns_unknown() -> None:
    with mock.patch(
        "backend.services.kaggle_service.subprocess.run",
        return_value=_completed(stdout="what even is this"),
    ):
        result = kernel_status("gumfreddy/mtmc-run-42")

    assert result.status == "unknown"


def test_kernel_output_happy_path_lists_downloaded_files(tmp_path: Path) -> None:
    dest_dir = tmp_path / "downloads"
    nested_dir = dest_dir / "stage2"
    nested_dir.mkdir(parents=True)
    output_file = nested_dir / "embeddings.npy"
    output_file.write_bytes(b"12345")

    with mock.patch(
        "backend.services.kaggle_service.subprocess.run",
        return_value=_completed(stdout="Output downloaded"),
    ):
        result = kernel_output("gumfreddy/mtmc-run-42", dest_dir)

    assert result.downloaded_files == [output_file]
    assert result.total_bytes == 5
    assert result.raw_stdout == "Output downloaded"


def test_cancel_kernel_with_new_cli_uses_cancel_directly() -> None:
    with mock.patch(
        "backend.services.kaggle_service.subprocess.run",
        return_value=_completed(stdout="Kernel cancelled"),
    ) as run_mock:
        result = cancel_kernel("gumfreddy/mtmc-run-42")

    assert result.final_status == "cancelled"
    assert result.attempts == 1
    assert result.fallback_used is False
    run_mock.assert_called_once()
    assert run_mock.call_args.args[0] == ["kaggle", "kernels", "cancel", "gumfreddy/mtmc-run-42"]


def test_cancel_kernel_with_old_cli_falls_back_to_polling() -> None:
    responses = [
        _completed(returncode=2, stderr="Error: no such command 'cancel'"),
        _completed(stdout="Kernel status: complete"),
    ]
    with mock.patch(
        "backend.services.kaggle_service.subprocess.run",
        side_effect=responses,
    ) as run_mock:
        result = cancel_kernel("gumfreddy/mtmc-run-42")

    assert result.final_status == "complete"
    assert result.attempts == 1
    assert result.fallback_used is True
    assert run_mock.call_count == 2
    assert run_mock.call_args_list[1].args[0] == [
        "kaggle",
        "kernels",
        "status",
        "gumfreddy/mtmc-run-42",
    ]


def test_count_active_kernels_parses_tab_separated_output() -> None:
    stdout = (
        "ref\ttitle\tstatus\tlastRunTime\n"
        "gumfreddy/a\tA\trunning\t2026-05-20\n"
        "gumfreddy/b\tB\trunning\t2026-05-20\n"
    )
    with mock.patch(
        "backend.services.kaggle_service.subprocess.run",
        return_value=_completed(stdout=stdout),
    ):
        assert count_active_kernels() == 2


def test_count_active_kernels_with_parse_failure_returns_zero() -> None:
    with mock.patch(
        "backend.services.kaggle_service.subprocess.run",
        return_value=_completed(stdout="garbage output without separators"),
    ):
        assert count_active_kernels() == 0


def test_dataset_create_or_update_create_path_writes_metadata(tmp_path: Path) -> None:
    files_dir = tmp_path / "dataset"
    files_dir.mkdir()

    def run_side_effect(command: list[str], **kwargs: object) -> mock.MagicMock:
        if command[:4] == ["kaggle", "datasets", "list", "--mine"]:
            return _completed(stdout="ref\ttitle\n")
        if command[:3] == ["kaggle", "datasets", "create"]:
            metadata = json.loads((files_dir / "dataset-metadata.json").read_text(encoding="utf-8"))
            assert metadata["id"] == "gumfreddy/mtmc-user-video-42"
            assert metadata["title"] == "MTMC video 42"
            assert metadata["licenses"] == [{"name": "other"}]
            assert metadata["isPrivate"] is True
            return _completed(stdout="Dataset created")
        raise AssertionError(f"Unexpected command: {command}")

    with mock.patch(
        "backend.services.kaggle_service.subprocess.run",
        side_effect=run_side_effect,
    ) as run_mock:
        result = dataset_create_or_update(
            "gumfreddy/mtmc-user-video-42",
            files_dir,
            title="MTMC video 42",
            description="Auto-uploaded",
        )

    assert result.slug == "gumfreddy/mtmc-user-video-42"
    assert result.kaggle_url == "https://www.kaggle.com/datasets/gumfreddy/mtmc-user-video-42"
    assert not (files_dir / "dataset-metadata.json").exists()
    assert run_mock.call_args_list[1].args[0][:3] == ["kaggle", "datasets", "create"]


def test_dataset_create_or_update_update_path_detects_existing_dataset(tmp_path: Path) -> None:
    files_dir = tmp_path / "dataset"
    files_dir.mkdir()
    stdout = "ref\ttitle\ngumfreddy/mtmc-user-video-42\tMTMC video 42\n"

    with mock.patch(
        "backend.services.kaggle_service.subprocess.run",
        side_effect=[_completed(stdout=stdout), _completed(stdout="Dataset version 7 uploaded")],
    ) as run_mock:
        result = dataset_create_or_update(
            "gumfreddy/mtmc-user-video-42",
            files_dir,
            title="MTMC video 42",
        )

    assert result.version == "7"
    assert run_mock.call_args_list[1].args[0][:3] == ["kaggle", "datasets", "version"]
    assert any(arg.startswith("Auto-update ") for arg in run_mock.call_args_list[1].args[0])


def test_dataset_create_or_update_restores_existing_metadata(tmp_path: Path) -> None:
    files_dir = tmp_path / "dataset"
    files_dir.mkdir()
    metadata_path = files_dir / "dataset-metadata.json"
    original_metadata = b'{"title":"existing"}'
    metadata_path.write_bytes(original_metadata)

    def run_side_effect(command: list[str], **kwargs: object) -> mock.MagicMock:
        if command[:4] == ["kaggle", "datasets", "list", "--mine"]:
            return _completed(stdout="ref\ttitle\n")
        if command[:3] == ["kaggle", "datasets", "create"]:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            assert metadata["title"] == "Replacement title"
            return _completed(stdout="Dataset created")
        raise AssertionError(f"Unexpected command: {command}")

    with mock.patch(
        "backend.services.kaggle_service.subprocess.run",
        side_effect=run_side_effect,
    ):
        dataset_create_or_update(
            "gumfreddy/mtmc-user-video-42",
            files_dir,
            title="Replacement title",
        )

    assert metadata_path.read_bytes() == original_metadata
