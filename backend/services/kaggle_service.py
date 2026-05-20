from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Literal, Optional

KernelStatusValue = Literal["queued", "running", "complete", "error", "cancelled", "unknown"]

_TERMINAL_KERNEL_STATUSES: set[KernelStatusValue] = {"complete", "error", "cancelled"}
_CANCEL_FALLBACK_MAX_ATTEMPTS = 120
_CANCEL_FALLBACK_SLEEP_SECONDS = 60


class KaggleError(RuntimeError):
    """Base exception for Kaggle wrapper failures."""


class KaggleAuthError(KaggleError):
    """Raised when Kaggle credentials are missing or rejected."""


class KaggleValidationError(KaggleError):
    """Raised when Kaggle rejects kernel or dataset metadata validation."""


class KaggleConcurrencyError(KaggleError):
    """Raised when Kaggle rejects an operation because execution slots are full."""


class KaggleCliError(KaggleError):
    """Raised for non-auth, non-validation Kaggle CLI failures."""


@dataclass(frozen=True)
class SubprocessResult:
    """Captured Kaggle CLI subprocess result."""

    args: List[str]
    returncode: int
    stdout: str
    stderr: str

    @property
    def combined_output(self) -> str:
        """Return stdout and stderr as a single string for parsing."""
        return "\n".join(part for part in [self.stdout, self.stderr] if part)


@dataclass(frozen=True)
class KernelPushResult:
    """Result returned by a successful `kaggle kernels push`."""

    slug: str
    kernel_url: str
    raw_stdout: str

    def to_dict(self) -> Dict[str, str]:
        """Return a JSON-serializable dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class KernelStatusResult:
    """Normalized kernel status result."""

    slug: str
    status: KernelStatusValue
    raw_stdout: str
    last_polled_iso: str

    def to_dict(self) -> Dict[str, str]:
        """Return a JSON-serializable dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class KernelOutputResult:
    """Result returned after downloading kernel outputs."""

    downloaded_files: List[Path]
    total_bytes: int
    raw_stdout: str

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serializable dictionary."""
        return {
            "downloaded_files": [str(path) for path in self.downloaded_files],
            "total_bytes": self.total_bytes,
            "raw_stdout": self.raw_stdout,
        }


@dataclass(frozen=True)
class KernelCancelResult:
    """Result returned after attempting to cancel or observe a kernel to terminal state."""

    final_status: KernelStatusValue
    attempts: int
    fallback_used: bool
    raw_stdout: str

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serializable dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class DatasetPushResult:
    """Result returned after creating or versioning a Kaggle dataset."""

    slug: str
    version: Optional[str]
    kaggle_url: str
    raw_stdout: str

    def to_dict(self) -> Dict[str, Optional[str]]:
        """Return a JSON-serializable dictionary."""
        return asdict(self)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_temp_credentials(username: str, key: str) -> Path:
    """Write a temp kaggle.json. Returns the temp DIR path.

    The caller owns the returned directory and must clean it up with `shutil.rmtree`.
    """
    if not username or not key:
        raise KaggleAuthError("Both Kaggle username and key are required.")

    temp_dir = Path(tempfile.mkdtemp(prefix="mtmc-kaggle-"))
    credential_path = temp_dir / "kaggle.json"
    credential_path.write_text(
        json.dumps({"username": username, "key": key}, ensure_ascii=True),
        encoding="utf-8",
    )
    return temp_dir


@contextmanager
def kaggle_env(
    username: Optional[str] = None,
    key: Optional[str] = None,
) -> Iterator[Dict[str, str]]:
    """Yield environment variables suitable for Kaggle CLI subprocesses.

    If both `username` and `key` are provided, a temporary `kaggle.json` is created
    and `KAGGLE_CONFIG_DIR` points at its parent directory. If neither value is
    provided, the current process environment is passed through so Kaggle can use
    `~/.kaggle/kaggle.json`. The temporary directory is removed on exit.
    """
    if (username and not key) or (key and not username):
        raise KaggleAuthError("Kaggle username and key must be provided together.")

    temp_dir: Optional[Path] = None
    env = dict(os.environ)
    try:
        if username and key:
            temp_dir = write_temp_credentials(username, key)
            env["KAGGLE_CONFIG_DIR"] = str(temp_dir)
        yield env
    finally:
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)


def _run_kaggle_cli(args: List[str], env: Dict[str, str]) -> SubprocessResult:
    """Run the Kaggle CLI with captured text output and no shell interpolation."""
    command = ["kaggle", *args]
    try:
        completed = subprocess.run(
            command,
            env=env,
            text=True,
            capture_output=True,
            shell=False,
        )
    except FileNotFoundError as exc:
        raise KaggleCliError("Kaggle CLI is not installed or is not on PATH.") from exc

    return SubprocessResult(
        args=command,
        returncode=completed.returncode,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
    )


def _looks_like_auth_error(output: str) -> bool:
    lowered = output.lower()
    auth_markers = [
        "401",
        "unauthorized",
        "forbidden",
        "invalid api token",
        "could not find kaggle.json",
        "kaggle.json must contain",
        "authentication",
    ]
    return any(marker in lowered for marker in auth_markers)


def _looks_like_validation_error(output: str) -> bool:
    lowered = output.lower()
    validation_markers = [
        "not valid dataset sources",
        "invalid dataset sources",
        "invalid kernel metadata",
        "dataset sources are invalid",
    ]
    return any(marker in lowered for marker in validation_markers)


def _looks_like_concurrency_error(output: str) -> bool:
    lowered = output.lower()
    concurrency_markers = [
        "too many active",
        "maximum number of gpu",
        "maximum number of running",
        "concurrent gpu",
        "gpu session limit",
        "quota exceeded for gpu",
    ]
    return any(marker in lowered for marker in concurrency_markers)


def _looks_like_cancel_unsupported(output: str) -> bool:
    lowered = output.lower()
    unsupported_markers = [
        "invalid choice: 'cancel'",
        "no such command 'cancel'",
        "unrecognized arguments: cancel",
        "usage: kaggle kernels",
    ]
    return any(marker in lowered for marker in unsupported_markers)


def _raise_for_failed_result(result: SubprocessResult) -> None:
    if result.returncode == 0:
        return

    output = result.combined_output
    if _looks_like_auth_error(output):
        raise KaggleAuthError(output.strip() or "Kaggle authentication failed.")
    if _looks_like_validation_error(output):
        raise KaggleValidationError(output.strip() or "Kaggle validation failed.")
    if _looks_like_concurrency_error(output):
        raise KaggleConcurrencyError(output.strip() or "Kaggle execution slot limit reached.")
    raise KaggleCliError(output.strip() or f"Kaggle CLI failed: {' '.join(result.args)}")


def _parse_kernel_url(output: str) -> str:
    match = re.search(r"https://www\.kaggle\.com/(?:code|kernels)/(\S+)", output)
    if match:
        return match.group(0).rstrip(".)]")
    return ""


def _parse_slug_from_kernel_output(output: str, metadata_dir: Optional[Path] = None) -> str:
    url = _parse_kernel_url(output)
    if url:
        url_match = re.search(r"kaggle\.com/(?:code|kernels)/([^\s)\]]+)", url)
        if url_match:
            slug = url_match.group(1).strip("/")
            if slug:
                return slug

    slug_match = re.search(r"\b([A-Za-z0-9_-]+/[A-Za-z0-9_-]+)\b", output)
    if slug_match:
        return slug_match.group(1)

    if metadata_dir is not None:
        metadata_path = metadata_dir / "kernel-metadata.json"
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return ""
            return str(metadata.get("id") or "")
    return ""


def _normalize_status(raw_output: str) -> KernelStatusValue:
    lowered = raw_output.lower()
    status_patterns: list[tuple[KernelStatusValue, list[str]]] = [
        ("cancelled", ["cancelled", "canceled"]),
        ("complete", ["complete", "completed", "success"]),
        ("error", ["error", "failed", "failure"]),
        ("running", ["running"]),
        ("queued", ["queued", "pending"]),
    ]
    for status, markers in status_patterns:
        if any(re.search(rf"\b{re.escape(marker)}\b", lowered) for marker in markers):
            return status
    return "unknown"


def whoami(*, username: Optional[str] = None, key: Optional[str] = None) -> str:
    """Return the resolved Kaggle username from `kaggle config view` output.

    Raises `KaggleAuthError` when no usable credentials are available.
    """
    with kaggle_env(username=username, key=key) as env:
        result = _run_kaggle_cli(["config", "view"], env)
    if result.returncode != 0:
        output = result.combined_output
        if _looks_like_auth_error(output):
            raise KaggleAuthError(output.strip() or "Kaggle credentials are not available.")
        _raise_for_failed_result(result)

    for line in result.stdout.splitlines():
        match = re.match(r"\s*-?\s*username\s*[:=]\s*(\S+)\s*$", line, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    raise KaggleAuthError("Unable to resolve Kaggle username from `kaggle config view`.")


def push_kernel(
    metadata_dir: Path,
    *,
    username: Optional[str] = None,
    key: Optional[str] = None,
) -> KernelPushResult:
    """Run `kaggle kernels push -p <metadata_dir>` and return parsed push details.

    Raises `KaggleValidationError` for invalid dataset-source warnings,
    `KaggleConcurrencyError` for slot-limit failures, `KaggleAuthError` for 401-style
    responses, and `KaggleCliError` for other non-zero exits.
    """
    with kaggle_env(username=username, key=key) as env:
        result = _run_kaggle_cli(["kernels", "push", "-p", str(metadata_dir)], env)

    output = result.combined_output
    if _looks_like_validation_error(output):
        raise KaggleValidationError(output.strip())
    if result.returncode != 0:
        _raise_for_failed_result(result)

    kernel_url = _parse_kernel_url(output)
    slug = _parse_slug_from_kernel_output(output, metadata_dir=metadata_dir)
    return KernelPushResult(slug=slug, kernel_url=kernel_url, raw_stdout=result.stdout)


def kernel_status(
    slug: str,
    *,
    username: Optional[str] = None,
    key: Optional[str] = None,
) -> KernelStatusResult:
    """Run `kaggle kernels status <slug>` and return a normalized status result.

    Parse failures return `status='unknown'` rather than raising.
    """
    with kaggle_env(username=username, key=key) as env:
        result = _run_kaggle_cli(["kernels", "status", slug], env)
    if result.returncode != 0:
        _raise_for_failed_result(result)

    return KernelStatusResult(
        slug=slug,
        status=_normalize_status(result.stdout),
        raw_stdout=result.stdout,
        last_polled_iso=_utc_now_iso(),
    )


def kernel_output(
    slug: str,
    dest_dir: Path,
    *,
    username: Optional[str] = None,
    key: Optional[str] = None,
) -> KernelOutputResult:
    """Run `kaggle kernels output <slug> -p <dest_dir>` and list downloaded files."""
    with kaggle_env(username=username, key=key) as env:
        result = _run_kaggle_cli(["kernels", "output", slug, "-p", str(dest_dir)], env)
    if result.returncode != 0:
        _raise_for_failed_result(result)

    downloaded_files = sorted(path for path in dest_dir.rglob("*") if path.is_file())
    total_bytes = sum(path.stat().st_size for path in downloaded_files)
    return KernelOutputResult(
        downloaded_files=downloaded_files,
        total_bytes=total_bytes,
        raw_stdout=result.stdout,
    )


def cancel_kernel(
    slug: str,
    *,
    username: Optional[str] = None,
    key: Optional[str] = None,
) -> KernelCancelResult:
    """Cancel a running kernel, falling back to status polling if cancel is unsupported.

    Newer Kaggle CLI versions support `kaggle kernels cancel <slug>`. Older versions
    such as 2.0.1 do not; for those, this function polls `kernel_status` until the
    kernel reaches a terminal state.
    """
    with kaggle_env(username=username, key=key) as env:
        result = _run_kaggle_cli(["kernels", "cancel", slug], env)

    if result.returncode == 0:
        status = _normalize_status(result.stdout)
        return KernelCancelResult(
            final_status=status if status != "unknown" else "cancelled",
            attempts=1,
            fallback_used=False,
            raw_stdout=result.stdout,
        )

    if not _looks_like_cancel_unsupported(result.combined_output):
        _raise_for_failed_result(result)

    raw_parts = [result.combined_output]
    for attempt in range(1, _CANCEL_FALLBACK_MAX_ATTEMPTS + 1):
        status_result = kernel_status(slug, username=username, key=key)
        raw_parts.append(status_result.raw_stdout)
        if status_result.status in _TERMINAL_KERNEL_STATUSES:
            return KernelCancelResult(
                final_status=status_result.status,
                attempts=attempt,
                fallback_used=True,
                raw_stdout="\n".join(part for part in raw_parts if part),
            )
        if attempt < _CANCEL_FALLBACK_MAX_ATTEMPTS:
            time.sleep(_CANCEL_FALLBACK_SLEEP_SECONDS)

    return KernelCancelResult(
        final_status="unknown",
        attempts=_CANCEL_FALLBACK_MAX_ATTEMPTS,
        fallback_used=True,
        raw_stdout="\n".join(part for part in raw_parts if part),
    )


def count_active_kernels(*, username: Optional[str] = None, key: Optional[str] = None) -> int:
    """Query Kaggle for the user's currently running GPU kernels.

    Uses `kaggle kernels list --mine --status running -v`. If parsing fails, returns
    0 so the eventual push remains the authoritative concurrency check.
    """
    with kaggle_env(username=username, key=key) as env:
        result = _run_kaggle_cli(["kernels", "list", "--mine", "--status", "running", "-v"], env)
    if result.returncode != 0:
        _raise_for_failed_result(result)

    rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not rows or any("no kernels" in line.lower() for line in rows):
        return 0
    if not any("\t" in row for row in rows):
        return 0

    count = 0
    for row in rows:
        columns = [column.strip() for column in row.split("\t")]
        lowered_columns = [column.lower() for column in columns]
        if "ref" in lowered_columns or "title" in lowered_columns:
            continue
        if columns and "/" in columns[0]:
            count += 1
    return count


def _dataset_exists(slug: str, output: str) -> bool:
    slug_lower = slug.lower()
    for line in output.splitlines():
        if slug_lower in line.lower():
            return True
    return False


def _dataset_url(slug: str) -> str:
    return f"https://www.kaggle.com/datasets/{slug}"


def _parse_dataset_version(output: str) -> Optional[str]:
    version_match = re.search(r"\bversion\s+(\d+)\b", output, flags=re.IGNORECASE)
    if version_match:
        return version_match.group(1)
    return None


def dataset_create_or_update(
    slug: str,
    files_dir: Path,
    *,
    title: str,
    description: str = "",
    license_name: str = "other",
    is_private: bool = True,
    update_only: bool = False,
    username: Optional[str] = None,
    key: Optional[str] = None,
) -> DatasetPushResult:
    """Create a Kaggle dataset or push a new version of an existing dataset.

    The function checks for an existing dataset with `kaggle datasets list --mine -s
    <slug>`. It writes Kaggle's `dataset-metadata.json` schema into `files_dir`, runs
    either `datasets create` or `datasets version`, and restores or removes the
    metadata shim after the CLI call.
    """
    metadata_path = files_dir / "dataset-metadata.json"
    previous_metadata = metadata_path.read_bytes() if metadata_path.exists() else None
    metadata = {
        "title": title,
        "id": slug,
        "licenses": [{"name": license_name}],
        "description": description,
        "isPrivate": is_private,
    }

    with kaggle_env(username=username, key=key) as env:
        list_result = _run_kaggle_cli(["datasets", "list", "--mine", "-s", slug], env)
        if list_result.returncode != 0:
            _raise_for_failed_result(list_result)
        exists = _dataset_exists(slug, list_result.stdout)

        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        try:
            if exists or update_only:
                version_message = f"Auto-update {_utc_now_iso()}"
                result = _run_kaggle_cli(
                    ["datasets", "version", "-p", str(files_dir), "-m", version_message],
                    env,
                )
            else:
                args = ["datasets", "create", "-p", str(files_dir), "-u"]
                if is_private:
                    args.append("--private")
                result = _run_kaggle_cli(args, env)
            if result.returncode != 0:
                _raise_for_failed_result(result)
        finally:
            if previous_metadata is None:
                metadata_path.unlink(missing_ok=True)
            else:
                metadata_path.write_bytes(previous_metadata)

    return DatasetPushResult(
        slug=slug,
        version=_parse_dataset_version(result.combined_output),
        kaggle_url=_dataset_url(slug),
        raw_stdout=result.stdout,
    )
