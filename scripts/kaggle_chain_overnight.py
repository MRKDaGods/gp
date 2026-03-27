#!/usr/bin/env python
"""Overnight campaign: monitor 09d -> download weights -> upload -> 10a->10b->10c chain."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
LOG_FILE = DATA_DIR / "overnight_campaign.log"

YAHIA_TOKEN = "yahiaakhalafallah_access_token"
ALI_TOKEN = "ali_369_access_token"
MRK_TOKEN = "MRKDaGods__access_token"

KERNEL_09D = "yahiaakhalafallah/09d-vehicle-reid-resnet101-ibn-a-training"
WEIGHT_FILENAME = "resnet101ibn_cityflowv2_384px_best.pth"
WEIGHT_DESTINATION = ROOT / "models" / "reid" / "resnet101ibn_cityflowv2_veri_best.pth"
MODELS_DIR = ROOT / "models"
DOWNLOAD_DIR = DATA_DIR / "09d_v4_output"

PIPELINE = [
    (
        "10a",
        ROOT / "notebooks" / "kaggle" / "10a_stages012",
        "ali369/mtmc-10a-stages-0-2-tracking-reid-features",
        ALI_TOKEN,
        2.0,
    ),
    (
        "10b",
        ROOT / "notebooks" / "kaggle" / "10b_stage3",
        "ali369/mtmc-10b-stage-3-faiss-indexing",
        ALI_TOKEN,
        1.5,
    ),
    (
        "10c",
        ROOT / "notebooks" / "kaggle" / "10c_stages45",
        "ali369/mtmc-10c-stages-4-5-association-eval",
        ALI_TOKEN,
        1.5,
    ),
]

POLL_SECONDS = 120
STATUS_TIMEOUT_SECONDS = 60
PUSH_TIMEOUT_SECONDS = 180
DOWNLOAD_TIMEOUT_SECONDS = 1200
UPLOAD_TIMEOUT_SECONDS = 1800
DATASET_SETTLE_SECONDS = 300
KERNEL_REGISTER_SECONDS = 45
LOG_TAIL_CHARS = 4000
RETRYABLE_OUTPUT_MARKERS = (
    "429",
    "rate limit",
    "too many requests",
    "timed out",
    "timeout",
    "connection reset",
    "temporarily unavailable",
    "service unavailable",
)
TERMINAL_STATUSES = {"COMPLETE", "ERROR", "CANCEL", "CANCELLED", "TIMEOUT"}


def log(message: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {message}"
    print(line, flush=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def truncate(text: str, limit: int = LOG_TAIL_CHARS) -> str:
    cleaned = (text or "").strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[-limit:]


def set_kaggle_token(token_name: str) -> str:
    """Load a Kaggle token from ~/.kaggle and update env vars for CLI and SDK use."""
    token_path = Path.home() / ".kaggle" / token_name
    if not token_path.exists():
        raise FileNotFoundError(f"Kaggle token file not found: {token_path}")

    raw = token_path.read_text(encoding="utf-8").strip()
    payload = json.loads(raw)
    username = payload["username"]
    key = payload["key"]

    os.environ["KAGGLE_API_TOKEN"] = raw
    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = key
    return username


def is_retryable_output(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in RETRYABLE_OUTPUT_MARKERS)


def run_command(
    args: Sequence[str],
    *,
    timeout: int,
    retries: int = 3,
    retry_delay: int = 60,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a command with retry handling for transient Kaggle/API failures."""
    last_error: RuntimeError | None = None

    for attempt in range(1, retries + 1):
        try:
            result = subprocess.run(
                list(args),
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(cwd) if cwd else None,
                encoding="utf-8",
                errors="replace",
            )
        except subprocess.TimeoutExpired as exc:
            combined = f"TIMEOUT after {timeout}s\nSTDOUT:\n{exc.stdout or ''}\nSTDERR:\n{exc.stderr or ''}"
            last_error = RuntimeError(combined)
            if attempt == retries:
                raise RuntimeError(combined) from exc
            log(f"Command timeout on attempt {attempt}/{retries}: {' '.join(args)}")
            time.sleep(retry_delay)
            continue

        combined_output = (result.stdout or "") + "\n" + (result.stderr or "")
        if result.returncode == 0:
            return result

        last_error = RuntimeError(
            f"Command failed with exit code {result.returncode}: {' '.join(args)}\n{combined_output}"
        )
        if attempt == retries or not is_retryable_output(combined_output):
            return result

        log(
            f"Transient command failure on attempt {attempt}/{retries}: {' '.join(args)}"
        )
        time.sleep(retry_delay)

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Command failed unexpectedly: {' '.join(args)}")


def parse_status(output: str) -> str:
    upper = output.upper()
    for status in ("COMPLETE", "ERROR", "CANCELLED", "CANCEL", "RUNNING", "QUEUED"):
        if status in upper:
            return status
    return "UNKNOWN"


def get_status(kernel_slug: str) -> str:
    """Get kernel status via Kaggle CLI with conservative retries."""
    result = run_command(
        ["kaggle", "kernels", "status", kernel_slug],
        timeout=STATUS_TIMEOUT_SECONDS,
        retries=3,
        retry_delay=POLL_SECONDS,
    )
    return parse_status((result.stdout or "") + "\n" + (result.stderr or ""))


def wait_for(kernel_slug: str, *, poll_seconds: int = POLL_SECONDS, max_hours: float = 5.0) -> tuple[str, float]:
    """Poll a kernel until it reaches a terminal state or times out."""
    start = time.time()
    deadline = start + max_hours * 3600

    while time.time() < deadline:
        try:
            status = get_status(kernel_slug)
        except Exception as exc:
            elapsed = (time.time() - start) / 60
            log(f"  {kernel_slug}: STATUS_CHECK_ERROR ({elapsed:.0f}min) -> {exc}")
            time.sleep(poll_seconds)
            continue

        elapsed = (time.time() - start) / 60
        log(f"  {kernel_slug}: {status} ({elapsed:.0f}min)")
        if status == "COMPLETE":
            return status, elapsed
        if status in {"ERROR", "CANCEL", "CANCELLED"}:
            return status, elapsed
        time.sleep(poll_seconds)

    return "TIMEOUT", (time.time() - start) / 60


def download_09d_weights() -> Path | None:
    """Download 09d output files and locate the trained weight checkpoint."""
    set_kaggle_token(YAHIA_TOKEN)
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    log("Downloading 09d output files...")
    result = run_command(
        ["kaggle", "kernels", "output", KERNEL_09D, "-p", str(DOWNLOAD_DIR)],
        timeout=DOWNLOAD_TIMEOUT_SECONDS,
        retries=3,
        retry_delay=POLL_SECONDS,
    )
    if result.stdout.strip():
        log(f"Download stdout: {truncate(result.stdout)}")
    if result.stderr.strip():
        log(f"Download stderr: {truncate(result.stderr)}")

    candidates = [DOWNLOAD_DIR / WEIGHT_FILENAME]
    candidates.extend(sorted(DOWNLOAD_DIR.rglob(WEIGHT_FILENAME)))
    if not any(path.exists() for path in candidates):
        candidates.extend(sorted(DOWNLOAD_DIR.rglob("*.pth")))

    chosen: Path | None = None
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            chosen = candidate
            if candidate.name == WEIGHT_FILENAME:
                break

    if chosen is None:
        log("ERROR: Weight file not found in 09d output.")
        for file_path in sorted(DOWNLOAD_DIR.rglob("*")):
            if file_path.is_file():
                rel = file_path.relative_to(DOWNLOAD_DIR)
                log(f"  downloaded: {rel} ({file_path.stat().st_size} bytes)")
        return None

    size_mb = chosen.stat().st_size / 1024 / 1024
    log(f"Found weights: {chosen} ({size_mb:.1f} MB)")
    return chosen


def inspect_checkpoint(weight_path: Path) -> None:
    try:
        import torch
    except Exception as exc:
        log(f"Warning: torch unavailable for checkpoint inspection: {exc}")
        return

    try:
        checkpoint = torch.load(str(weight_path), map_location="cpu", weights_only=False)
    except Exception as exc:
        log(f"Warning: could not inspect checkpoint: {exc}")
        return

    if isinstance(checkpoint, dict) and "mAP" in checkpoint:
        log(f"Checkpoint mAP: {checkpoint['mAP']:.4f}")
        return
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        log(f"Checkpoint model_state_dict keys: {len(checkpoint['model_state_dict'])}")
        return

    top_level_keys = len(checkpoint) if isinstance(checkpoint, dict) else "N/A"
    log(f"Checkpoint loaded; top-level keys: {top_level_keys}")


def upload_weights(weight_path: Path) -> bool:
    """Copy the downloaded checkpoint into models/reid and publish the Kaggle dataset."""
    WEIGHT_DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(weight_path, WEIGHT_DESTINATION)
    size_mb = WEIGHT_DESTINATION.stat().st_size / 1024 / 1024
    log(f"Copied weights to {WEIGHT_DESTINATION} ({size_mb:.1f} MB)")

    inspect_checkpoint(WEIGHT_DESTINATION)

    set_kaggle_token(MRK_TOKEN)
    log("Uploading weights to mrkdagods/mtmc-weights...")
    result = run_command(
        [
            "kaggle",
            "datasets",
            "version",
            "-p",
            str(MODELS_DIR),
            "-m",
            "Add VeRi-pretrained ResNet101-IBN-a CityFlowV2 fine-tuned weights",
            "--dir-mode",
            "zip",
        ],
        timeout=UPLOAD_TIMEOUT_SECONDS,
        retries=2,
        retry_delay=POLL_SECONDS,
    )

    if result.returncode != 0:
        log(f"Upload stdout: {truncate(result.stdout)}")
        log(f"Upload stderr: {truncate(result.stderr)}")
        return False

    if result.stdout.strip():
        log(f"Upload stdout: {truncate(result.stdout)}")
    if result.stderr.strip():
        log(f"Upload stderr: {truncate(result.stderr)}")

    log(f"Waiting {DATASET_SETTLE_SECONDS // 60} minutes for Kaggle dataset processing...")
    time.sleep(DATASET_SETTLE_SECONDS)
    return True


def push_and_wait(
    stage_name: str,
    notebook_dir: Path,
    kernel_slug: str,
    token_name: str,
    *,
    max_hours: float,
) -> tuple[bool, float, str]:
    """Push a notebook, then wait for the Kaggle run to finish."""
    set_kaggle_token(token_name)
    log(f"Pushing {stage_name} from {notebook_dir}...")
    result = run_command(
        ["kaggle", "kernels", "push", "-p", str(notebook_dir)],
        timeout=PUSH_TIMEOUT_SECONDS,
        retries=2,
        retry_delay=POLL_SECONDS,
    )
    if result.stdout.strip():
        log(f"  Push stdout: {truncate(result.stdout)}")
    if result.stderr.strip():
        log(f"  Push stderr: {truncate(result.stderr)}")

    if result.returncode != 0:
        return False, 0.0, "PUSH_FAILED"

    time.sleep(KERNEL_REGISTER_SECONDS)
    status, elapsed = wait_for(kernel_slug, poll_seconds=POLL_SECONDS, max_hours=max_hours)
    return status == "COMPLETE", elapsed, status


def fetch_results(kernel_slug: str, token_name: str) -> str:
    """Fetch the recent kernel log tail via the existing helper script."""
    set_kaggle_token(token_name)
    helper = ROOT / "scripts" / "kaggle_logs.py"
    if not helper.exists():
        return "kaggle_logs.py not found"

    result = run_command(
        [sys.executable, str(helper), kernel_slug, "--tail", "200"],
        timeout=180,
        retries=2,
        retry_delay=60,
    )
    combined = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
    return combined.strip()


def main() -> int:
    summary: list[str] = []

    log("=" * 70)
    log("OVERNIGHT CAMPAIGN STARTED")
    log("=" * 70)

    try:
        log("")
        log("=== PHASE 1: Monitor 09d VeRi-pretrained ResNet101 ===")
        set_kaggle_token(YAHIA_TOKEN)
        status_09d, elapsed_09d = wait_for(KERNEL_09D, poll_seconds=POLL_SECONDS, max_hours=5.0)
        summary.append(f"09d={status_09d} ({elapsed_09d:.0f}min)")

        if status_09d == "COMPLETE":
            log(f"09d COMPLETE after {elapsed_09d:.0f}min")
            log("")
            log("=== PHASE 2: Download 09d weights ===")
            weight_path = download_09d_weights()

            if weight_path is not None:
                log("")
                log("=== PHASE 3: Upload weights to mtmc-weights ===")
                uploaded = upload_weights(weight_path)
                summary.append(f"weights_upload={'SUCCESS' if uploaded else 'FAILED'}")
                if not uploaded:
                    log("Weight upload failed; proceeding with current Kaggle dataset contents.")
            else:
                summary.append("weights_upload=SKIPPED_NO_FILE")
                log("Skipping upload because no weight file was found.")
        else:
            log(f"09d ended with {status_09d} after {elapsed_09d:.0f}min")
            log("Proceeding with existing weights in mtmc-weights.")
            summary.append("weights_upload=SKIPPED_EXISTING_DATASET")

        log("")
        log("=== PHASE 4: Pipeline Chain ===")
        for stage_name, notebook_dir, slug, token_name, max_hours in PIPELINE:
            ok, elapsed, stage_status = push_and_wait(
                stage_name,
                notebook_dir,
                slug,
                token_name,
                max_hours=max_hours,
            )
            summary.append(f"{stage_name}={stage_status} ({elapsed:.0f}min)")

            stage_logs = fetch_results(slug, token_name)
            if ok:
                log(f"  {stage_name} COMPLETE after {elapsed:.0f}min")
            else:
                log(f"  {stage_name} ended with {stage_status} after {elapsed:.0f}min")
            if stage_logs:
                log(f"  {stage_name} tail log:\n{truncate(stage_logs, limit=LOG_TAIL_CHARS)}")
            if not ok:
                break

        log("")
        log("=== PHASE 5: Summary ===")
        for item in summary:
            log(f"  {item}")
        log("=" * 70)
        log("OVERNIGHT CAMPAIGN COMPLETE")
        log(f"Full log: {LOG_FILE}")
        log("=" * 70)
        return 0
    except KeyboardInterrupt:
        log("Campaign interrupted by user.")
        return 130
    except Exception as exc:
        log(f"FATAL: {exc}")
        log("=" * 70)
        log("OVERNIGHT CAMPAIGN ABORTED")
        log(f"Full log: {LOG_FILE}")
        log("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())