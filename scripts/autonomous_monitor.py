#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, Sequence


ROOT = Path(__file__).resolve().parents[1]
KAGGLE_DIR = Path(r"C:\Users\mamar\.kaggle")
KAGGLE_JSON_PATH = KAGGLE_DIR / "kaggle.json"
ACCESS_TOKEN_PATH = KAGGLE_DIR / "access_token"

MRK_USERNAME = "mrkdagods"
MRK_TOKEN_FILE = KAGGLE_DIR / "MRKDaGods__access_token"
YAHIA_USERNAME = "yahiaakhalafallah"
YAHIA_TOKEN_FILE = KAGGLE_DIR / "yahiaakhalafallah_access_token"

KERNEL_09Q = f"{MRK_USERNAME}/09q-transreid-cityflow-v10"
KERNEL_10A = f"{YAHIA_USERNAME}/mtmc-10a-stages-0-2"
KERNEL_10B = f"{YAHIA_USERNAME}/mtmc-10b-stage-3-faiss-indexing"
KERNEL_10C = f"{YAHIA_USERNAME}/mtmc-10c-stages-4-5-association-eval"

BASELINE_MAP = 0.8014
POLL_INTERVAL_SECONDS = 300
MAX_POLLS_BEFORE_WARNING = 96

OUTPUT_DIR = Path(r"C:\temp\09q_v9_output")
UPLOAD_DIR = Path(r"C:\temp\09q_dataset_upload")
DATASET_ID = f"{YAHIA_USERNAME}/mtmc-10a-checkpoints"

NOTEBOOK_DIR_10A = ROOT / "notebooks" / "kaggle" / "10a_stages012"

EXP_A_CHECKPOINT = "vehicle_transreid_09q_expA_resume.pth"
EXP_B_CHECKPOINT = "vehicle_transreid_vit_base_cityflowv2.pth"
RENAMED_DATASET_CHECKPOINT = "vehicle_transreid_09q_expA.pth"

NEW_CANDIDATE_PATH = (
    '    Path("/kaggle/input/mtmc-10a-checkpoints/vehicle_transreid_09q_expA.pth"),\n'
)
NEW_CANDIDATE_COMMENT = (
    "    # Prefer the freshly uploaded 09q checkpoint when present.\n"
)

STATUS_KEYWORDS = (
    "RUNNING",
    "QUEUED",
    "COMPLETE",
    "SUCCESS",
    "ERROR",
    "CANCEL_ACKNOWLEDGED",
    "CANCELLED",
    "CANCEL",
)
TERMINAL_FAILURE_STATUSES = {"ERROR", "CANCEL_ACKNOWLEDGED", "CANCELLED", "CANCEL"}
TERMINAL_SUCCESS_STATUSES = {"COMPLETE", "SUCCESS"}

EXP_A_PATTERN = re.compile(r"Experiment A Done \| Best mAP: (\d+\.\d+)")
EXP_B_PATTERN = re.compile(r"Done in.*\| Best mAP: (\d+\.\d+)")
TRACEBACK_PATTERN = re.compile(r"Traceback \(most recent call last\):", re.IGNORECASE)
ERROR_LINE_PATTERN = re.compile(r"^.*\bERROR\b.*$", re.IGNORECASE | re.MULTILINE)
INVALID_DATASET_PATTERN = re.compile(r"not valid dataset sources", re.IGNORECASE)


@dataclass(frozen=True)
class ExperimentChoice:
    name: str
    best_map: float
    checkpoint_name: str


@dataclass(frozen=True)
class Phase2Decision:
    exp_a_map: float | None
    exp_b_map: float | None
    use_new_checkpoint: bool
    choice: ExperimentChoice | None


@dataclass(frozen=True)
class Stage:
    name: str
    kernel_ref: str
    notebook_dir: Path


STAGE_10A = Stage("10a", KERNEL_10A, ROOT / "notebooks" / "kaggle" / "10a_stages012")
STAGE_10B = Stage("10b", KERNEL_10B, ROOT / "notebooks" / "kaggle" / "10b_stage3")
STAGE_10C = Stage("10c", KERNEL_10C, ROOT / "notebooks" / "kaggle" / "10c_stages45")


def log(message: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)


def format_ratio(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f} ({value * 100:.2f}%)"


def load_token(token_file: Path) -> str:
    try:
        token = token_file.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise RuntimeError(f"Missing Kaggle token file: {token_file}") from exc
    if not token:
        raise RuntimeError(f"Kaggle token file is empty: {token_file}")
    return token


def switch_account(username: str, token_file: Path) -> None:
    token = load_token(token_file)
    KAGGLE_DIR.mkdir(parents=True, exist_ok=True)
    ACCESS_TOKEN_PATH.write_text(token, encoding="utf-8")
    KAGGLE_JSON_PATH.write_text(
        json.dumps({"username": username, "key": token}, ensure_ascii=True),
        encoding="utf-8",
    )
    log(f"Switched Kaggle account to {username}")


def restore_yahia() -> None:
    switch_account(YAHIA_USERNAME, YAHIA_TOKEN_FILE)


@contextmanager
def temporary_account(username: str, token_file: Path) -> Iterator[None]:
    switch_account(username, token_file)
    try:
        yield
    finally:
        if username != YAHIA_USERNAME or token_file != YAHIA_TOKEN_FILE:
            restore_yahia()


def command_text(command: Sequence[str]) -> str:
    return subprocess.list2cmdline([str(part) for part in command])


def run_command(
    command: Sequence[str],
    *,
    cwd: Path = ROOT,
    check: bool = True,
    description: str | None = None,
) -> subprocess.CompletedProcess[str]:
    log(f"Running: {command_text(command)}")
    completed = subprocess.run(
        [str(part) for part in command],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0 and check:
        details = (completed.stdout + completed.stderr).strip()
        raise RuntimeError(
            f"{description or 'Command'} failed with exit code {completed.returncode}.\n{details}"
        )
    return completed


def extract_status(text: str) -> str:
    upper_text = text.upper()
    for keyword in STATUS_KEYWORDS:
        if keyword in upper_text:
            if keyword == "SUCCESS":
                return "COMPLETE"
            if keyword == "CANCELLED":
                return "CANCEL"
            return keyword
    return "UNKNOWN"


def get_kernel_status(kernel_ref: str) -> tuple[str, str]:
    result = run_command(
        ["kaggle", "kernels", "status", kernel_ref],
        check=False,
        description=f"Status check for {kernel_ref}",
    )
    text = (result.stdout + result.stderr).strip()
    status = extract_status(text)
    return status, text


def fetch_logs(kernel_ref: str, tail: int, *, username: str, token_file: Path) -> str:
    with temporary_account(username, token_file):
        result = run_command(
            [
                sys.executable,
                str(ROOT / "scripts" / "kaggle_logs.py"),
                kernel_ref,
                "--tail",
                str(tail),
            ],
            check=True,
            description=f"Log fetch for {kernel_ref}",
        )
    log_text = result.stdout.strip()
    if not log_text:
        log_text = (result.stdout + result.stderr).strip()
    return log_text


def print_logs(title: str, log_text: str) -> None:
    separator = "=" * 90
    log(f"{title}")
    print(separator)
    print(log_text)
    print(separator)


def extract_last_float(pattern: re.Pattern[str], text: str) -> float | None:
    matches = pattern.findall(text)
    if not matches:
        return None
    return float(matches[-1])


def parse_idf1(log_text: str) -> float | None:
    lines = log_text.splitlines()
    for line in reversed(lines):
        if "mtmc" in line.lower() and "idf1" in line.lower():
            match = re.search(r"idf1[^0-9]*(\d+\.\d+)", line, re.IGNORECASE)
            if match:
                return float(match.group(1))
    for line in reversed(lines):
        if "idf1" in line.lower():
            match = re.search(r"idf1[^0-9]*(\d+\.\d+)", line, re.IGNORECASE)
            if match:
                return float(match.group(1))
    return None


def clean_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def monitor_kernel(
    kernel_ref: str,
    *,
    username: str,
    token_file: Path,
    error_tail: int,
) -> None:
    polls = 0
    warned = False
    while True:
        with temporary_account(username, token_file):
            status, raw_status = get_kernel_status(kernel_ref)
        log(f"{kernel_ref} status: {status}")
        if status in TERMINAL_SUCCESS_STATUSES:
            return
        if status in TERMINAL_FAILURE_STATUSES:
            error_logs = fetch_logs(
                kernel_ref,
                error_tail,
                username=username,
                token_file=token_file,
            )
            print_logs(f"{kernel_ref} failed logs", error_logs)
            raise RuntimeError(f"{kernel_ref} failed with status {status}. Manual fix needed.")
        if status == "UNKNOWN" and raw_status:
            log(f"Unparsed status response for {kernel_ref}: {raw_status}")
        polls += 1
        if polls >= MAX_POLLS_BEFORE_WARNING and not warned:
            log(
                f"{kernel_ref} has been running for more than 8 hours; continuing to poll every "
                f"{POLL_INTERVAL_SECONDS // 60} minutes."
            )
            warned = True
        time.sleep(POLL_INTERVAL_SECONDS)


def detect_log_failures(log_text: str) -> list[str]:
    findings: list[str] = []
    if TRACEBACK_PATTERN.search(log_text):
        findings.append("Python traceback found in logs")
    if ERROR_LINE_PATTERN.search(log_text):
        findings.append("ERROR lines found in logs")
    return findings


def choose_experiment(exp_a_map: float | None, exp_b_map: float | None) -> ExperimentChoice | None:
    if exp_a_map is not None and exp_a_map > BASELINE_MAP:
        return ExperimentChoice("Experiment A", exp_a_map, EXP_A_CHECKPOINT)
    if exp_b_map is not None and exp_b_map > BASELINE_MAP:
        return ExperimentChoice("Experiment B", exp_b_map, EXP_B_CHECKPOINT)
    return None


def find_checkpoint(root: Path, checkpoint_name: str) -> Path:
    matches = sorted(root.rglob(checkpoint_name))
    if not matches:
        raise RuntimeError(f"Checkpoint {checkpoint_name} was not found under {root}")
    return matches[0]


def file_size_text(path: Path) -> str:
    size_bytes = path.stat().st_size
    return f"{size_bytes / (1024 * 1024):.2f} MiB"


def write_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)
        handle.write("\n")


def find_10a_notebook() -> Path:
    notebooks = sorted(NOTEBOOK_DIR_10A.glob("*.ipynb"))
    if not notebooks:
        raise RuntimeError(f"No notebook found under {NOTEBOOK_DIR_10A}")
    if len(notebooks) > 1:
        metadata_path = NOTEBOOK_DIR_10A / "kernel-metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            preferred = NOTEBOOK_DIR_10A / metadata.get("code_file", "")
            if preferred.exists():
                return preferred
        raise RuntimeError(f"Multiple notebooks found under {NOTEBOOK_DIR_10A}: {notebooks}")
    return notebooks[0]


def normalize_source(source: list[str]) -> list[str]:
    if not source:
        return source
    normalized: list[str] = []
    for index, line in enumerate(source):
        if index < len(source) - 1:
            normalized.append(line if line.endswith("\n") else f"{line}\n")
        else:
            normalized.append(line[:-1] if line.endswith("\n") else line)
    return normalized


def update_10a_notebook() -> Path:
    notebook_path = find_10a_notebook()
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells = notebook.get("cells", [])

    target_cell: dict | None = None
    for cell in cells:
        source = cell.get("source", [])
        text = "".join(source)
        if (
            isinstance(source, list)
            and "LAION_TERTIARY_SRC_CANDIDATES" in text
            and "vehicle_transreid_vit_base_cityflowv2.pth" in text
        ):
            target_cell = cell
            break

    if target_cell is None:
        raise RuntimeError(
            "Could not find the 10a notebook cell that defines LAION_TERTIARY_SRC_CANDIDATES"
        )

    source_lines = list(target_cell.get("source", []))
    list_start = next(
        (index for index, line in enumerate(source_lines) if "LAION_TERTIARY_SRC_CANDIDATES" in line),
        None,
    )
    if list_start is None:
        raise RuntimeError("Could not locate the LAION_TERTIARY_SRC_CANDIDATES declaration")

    list_end = next(
        (index for index in range(list_start + 1, len(source_lines)) if source_lines[index].strip() == "]"),
        None,
    )
    if list_end is None:
        raise RuntimeError("Could not locate the end of the LAION_TERTIARY_SRC_CANDIDATES list")

    existing_items = source_lines[list_start + 1:list_end]
    rebuilt_items: list[str] = []
    for line in existing_items:
        if "vehicle_transreid_09q_expA.pth" in line:
            continue
        if line.strip() == NEW_CANDIDATE_COMMENT.strip():
            continue
        rebuilt_items.append(line)

    source_lines[list_start + 1:list_end] = [NEW_CANDIDATE_COMMENT, NEW_CANDIDATE_PATH, *rebuilt_items]
    target_cell["source"] = normalize_source(source_lines)

    with notebook_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(notebook, handle, ensure_ascii=True, indent=1)
        handle.write("\n")

    verified = json.loads(notebook_path.read_text(encoding="utf-8"))
    for cell in verified.get("cells", []):
        source = cell.get("source", [])
        if not isinstance(source, list):
            continue
        if not any("LAION_TERTIARY_SRC_CANDIDATES" in line for line in source):
            continue

        list_start = next(index for index, line in enumerate(source) if "LAION_TERTIARY_SRC_CANDIDATES" in line)
        list_end = next(
            index for index in range(list_start + 1, len(source)) if source[index].strip() == "]"
        )
        entries = [line.strip() for line in source[list_start + 1:list_end] if line.strip() and not line.strip().startswith("#")]
        if not entries or "vehicle_transreid_09q_expA.pth" not in entries[0]:
            raise RuntimeError(
                f"Notebook verification failed for {notebook_path}: the new checkpoint is not first"
            )
        break
    else:
        raise RuntimeError(f"Notebook verification failed for {notebook_path}")

    return notebook_path


def cancel_kernel(kernel_ref: str) -> None:
    result = run_command(
        ["kaggle", "kernels", "cancel", kernel_ref],
        check=False,
        description=f"Cancel {kernel_ref}",
    )
    details = (result.stdout + result.stderr).strip()
    if result.returncode == 0:
        log(f"Cancelled {kernel_ref} after invalid dataset-source warning.")
        if details:
            print(details)
        return
    raise RuntimeError(
        f"Push triggered an invalid dataset-source warning for {kernel_ref}, and cancellation failed.\n"
        f"Kernel URL: https://www.kaggle.com/code/{kernel_ref}\n{details}"
    )


def push_kernel(stage: Stage) -> None:
    restore_yahia()
    result = run_command(
        ["kaggle", "kernels", "push", "-p", str(stage.notebook_dir)],
        check=True,
        description=f"Push {stage.name}",
    )
    details = (result.stdout + result.stderr).strip()
    if details:
        print(details)
    if INVALID_DATASET_PATTERN.search(details):
        cancel_kernel(stage.kernel_ref)
        raise RuntimeError(
            f"{stage.name} push reported invalid dataset sources. The run was cancelled; fix metadata before retrying."
        )


def phase1_monitor_09q() -> None:
    log(f"Phase 1: monitoring {KERNEL_09Q}")
    monitor_kernel(
        KERNEL_09Q,
        username=MRK_USERNAME,
        token_file=MRK_TOKEN_FILE,
        error_tail=80,
    )
    log(f"{KERNEL_09Q} completed successfully")


def phase2_analyze_results() -> Phase2Decision:
    log("Phase 2: analyzing 09q results")
    logs = fetch_logs(KERNEL_09Q, 150, username=MRK_USERNAME, token_file=MRK_TOKEN_FILE)
    print_logs("09q completion logs", logs)

    findings = detect_log_failures(logs)
    for finding in findings:
        log(f"Warning: {finding}")

    exp_a_map = extract_last_float(EXP_A_PATTERN, logs)
    exp_b_map = extract_last_float(EXP_B_PATTERN, logs)

    log(
        f"Parsed results: Experiment A={format_ratio(exp_a_map)}, "
        f"Experiment B={format_ratio(exp_b_map)}, baseline={format_ratio(BASELINE_MAP)}"
    )

    if exp_a_map is None:
        raise RuntimeError("Could not parse Experiment A mAP from 09q logs")

    use_new_checkpoint = exp_a_map > BASELINE_MAP
    if not use_new_checkpoint:
        log(
            f"Exp A ({exp_a_map:.4f}) did not beat baseline ({BASELINE_MAP:.4f}). "
            "Skipping checkpoint update."
        )
        log("Proceeding with 10a→10b→10c chain using EXISTING baseline checkpoint.")
        return Phase2Decision(
            exp_a_map=exp_a_map,
            exp_b_map=exp_b_map,
            use_new_checkpoint=False,
            choice=None,
        )

    choice = choose_experiment(exp_a_map, exp_b_map)
    if choice is None:
        raise RuntimeError("Experiment A beat baseline, but no checkpoint choice could be determined")

    log(f"{choice.name} beat the baseline. Using checkpoint {choice.checkpoint_name}.")
    return Phase2Decision(
        exp_a_map=exp_a_map,
        exp_b_map=exp_b_map,
        use_new_checkpoint=True,
        choice=choice,
    )


def phase3_download_outputs(choice: ExperimentChoice) -> Path:
    log("Phase 3: downloading 09q kernel outputs")
    clean_directory(OUTPUT_DIR)
    with temporary_account(MRK_USERNAME, MRK_TOKEN_FILE):
        run_command(
            ["kaggle", "kernels", "output", KERNEL_09Q, "-p", str(OUTPUT_DIR)],
            check=True,
            description="Download 09q kernel output",
        )
    checkpoint_path = find_checkpoint(OUTPUT_DIR, choice.checkpoint_name)
    log(f"Downloaded checkpoint: {checkpoint_path} ({file_size_text(checkpoint_path)})")
    return checkpoint_path


def phase4_upload_dataset(checkpoint_path: Path, choice: ExperimentChoice) -> None:
    log("Phase 4: uploading checkpoint to yahiaakhalafallah/mtmc-10a-checkpoints")
    clean_directory(UPLOAD_DIR)
    target_checkpoint = UPLOAD_DIR / RENAMED_DATASET_CHECKPOINT
    shutil.copy2(checkpoint_path, target_checkpoint)
    write_json(
        UPLOAD_DIR / "dataset-metadata.json",
        {"id": DATASET_ID, "licenses": [{"name": "CC0-1.0"}]},
    )
    restore_yahia()
    run_command(
        [
            "kaggle",
            "datasets",
            "version",
            "-p",
            str(UPLOAD_DIR),
            "-m",
            f"09q v9 ExpA checkpoint: {choice.best_map:.4f} mAP",
        ],
        check=True,
        description="Upload checkpoint dataset version",
    )
    log("Dataset version uploaded successfully")


def phase5_update_notebook() -> Path:
    log("Phase 5: updating the 10a notebook to prefer the new 09q checkpoint")
    notebook_path = update_10a_notebook()
    log(f"Updated notebook: {notebook_path}")
    return notebook_path


def run_stage(stage: Stage, *, error_tail: int) -> None:
    log(f"Phase {stage.name}: pushing and monitoring {stage.kernel_ref}")
    push_kernel(stage)
    monitor_kernel(
        stage.kernel_ref,
        username=YAHIA_USERNAME,
        token_file=YAHIA_TOKEN_FILE,
        error_tail=error_tail,
    )
    log(f"{stage.kernel_ref} completed successfully")


def phase8_finish() -> float | None:
    log("Fetching final 10c logs")
    logs = fetch_logs(KERNEL_10C, 150, username=YAHIA_USERNAME, token_file=YAHIA_TOKEN_FILE)
    print_logs("10c completion logs", logs)
    findings = detect_log_failures(logs)
    for finding in findings:
        log(f"Warning: {finding}")
    idf1 = parse_idf1(logs)
    if idf1 is None:
        log("Could not parse MTMC IDF1 from 10c logs")
    else:
        log(f"Final MTMC IDF1: {idf1:.4f}")
    return idf1


def main() -> int:
    phase1_monitor_09q()
    decision = phase2_analyze_results()

    if decision.use_new_checkpoint:
        if decision.choice is None:
            raise RuntimeError("Phase 2 indicated a new checkpoint should be used, but no choice was returned")
        checkpoint_path = phase3_download_outputs(decision.choice)
        phase4_upload_dataset(checkpoint_path, decision.choice)
        phase5_update_notebook()
    else:
        log("Skipping Phases 3-5 and pushing 10a notebook as-is.")

    run_stage(STAGE_10A, error_tail=60)
    run_stage(STAGE_10B, error_tail=60)
    run_stage(STAGE_10C, error_tail=60)
    idf1 = phase8_finish()

    log("Autonomous monitoring pipeline completed")
    if decision.use_new_checkpoint and decision.choice is not None:
        winner_summary = f"{decision.choice.name} ({format_ratio(decision.choice.best_map)})"
        checkpoint_summary = decision.choice.checkpoint_name
    else:
        winner_summary = f"existing baseline ({format_ratio(BASELINE_MAP)})"
        checkpoint_summary = EXP_B_CHECKPOINT
    log(
        f"Summary: winner={winner_summary}, checkpoint={checkpoint_summary}, "
        f"final_mtmc_idf1={format_ratio(idf1) if idf1 is not None else 'n/a'}"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log("Interrupted by user")
        raise SystemExit(130)
    except Exception as exc:
        log(f"FATAL: {exc}")
        raise SystemExit(1)