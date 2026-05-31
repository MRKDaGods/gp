from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


RUNS = (
    ("base", "gumfreddy_access_token", "gumfreddy/citytrack-s02-base"),
    ("ssa", "gumfreddy_access_token", "gumfreddy/citytrack-s02-ssa"),
    ("bt", "MRKDaGods__access_token", "mrkdagods/citytrack-s02-bt"),
    ("all", "MRKDaGods__access_token", "mrkdagods/citytrack-s02-all"),
    ("occ", "ali_369_access_token", "ali369/citytrack-s02-occ"),
)

COMPLETE_STATUS = "COMPLETE"
DELTA = "d"
KEEP_PATTERNS = ("*_results.json", "*_evaluation_report.json", "*_stage5.tar.gz")
KNOWN_STATUSES = (
    "complete",
    "running",
    "queued",
    "pending",
    "error",
    "failed",
    "failure",
    "cancelled",
    "canceled",
)


@dataclass(frozen=True)
class RunConfig:
    label: str
    token_file: str
    slug: str


@dataclass
class RunResult:
    label: str
    status: str
    mtmc_idf1: float | None = None
    c006_idf1: float | None = None
    c006_mota: float | None = None
    id_switches: int | None = None
    results_path: Path | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect CityTrack S02 ablation kernel results.")
    parser.add_argument("--force", action="store_true", help="Re-download COMPLETE kernel outputs.")
    return parser.parse_args()


def kaggle_command() -> str:
    executable = "kaggle.exe" if os.name == "nt" else "kaggle"
    venv_candidate = Path(sys.executable).with_name(executable)
    if venv_candidate.exists():
        return str(venv_candidate)
    return "kaggle"


def read_token(token_file: str) -> str:
    token_path = Path.home() / ".kaggle" / token_file
    return token_path.read_text(encoding="utf-8").strip()


def run_kaggle(args: list[str], token_file: str) -> subprocess.CompletedProcess[str]:
    os.environ["KAGGLE_API_TOKEN"] = read_token(token_file)
    return subprocess.run(
        [kaggle_command(), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def command_text(process: subprocess.CompletedProcess[str]) -> str:
    return "\n".join(part for part in (process.stdout, process.stderr) if part).strip()


def parse_status(output: str) -> str:
    text = " ".join(output.split())
    if not text:
        return "UNKNOWN"

    try:
        parsed = json.loads(output)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        value = parsed.get("status") or parsed.get("state")
        if isinstance(value, str) and value.strip():
            return normalize_status(value)

    status_match = re.search(r"\bstatus\b\s*(?:is|:|=)?\s*[\"']?([A-Za-z_.-]+)", text, re.IGNORECASE)
    if status_match:
        return normalize_status(status_match.group(1))

    for status in KNOWN_STATUSES:
        if re.search(rf"\b{re.escape(status)}\b", text, re.IGNORECASE):
            return normalize_status(status)

    return "UNKNOWN"


def normalize_status(status: str) -> str:
    value = status.strip().rsplit(".", 1)[-1].replace("-", "_").upper()
    if value == "CANCELED":
        return "CANCELLED"
    if value == "FAILURE":
        return "FAILED"
    return value


def result_files(output_dir: Path) -> list[Path]:
    return sorted(output_dir.glob("*_results.json"))


def has_downloaded_result(output_dir: Path) -> bool:
    return any(result_files(output_dir))


def clean_output_dir(output_dir: Path) -> None:
    repo_clone = output_dir / "gp"
    if repo_clone.exists():
        shutil.rmtree(repo_clone)

    kept_files: set[Path] = set()
    for pattern in KEEP_PATTERNS:
        kept_files.update(path.resolve() for path in output_dir.glob(pattern) if path.is_file())

    for path in output_dir.iterdir() if output_dir.exists() else ():
        if path.is_file() and path.resolve() not in kept_files:
            path.unlink()


def download_if_needed(run: RunConfig, output_dir: Path, force: bool) -> bool:
    if has_downloaded_result(output_dir) and not force:
        clean_output_dir(output_dir)
        return True

    output_dir.mkdir(parents=True, exist_ok=True)
    process = run_kaggle(["kernels", "output", run.slug, "-p", str(output_dir), "--force"], run.token_file)
    clean_output_dir(output_dir)
    if process.returncode != 0:
        print(f"warning: download failed for {run.label}: {command_text(process)}", file=sys.stderr)
        return has_downloaded_result(output_dir)
    return True


def nested_number(data: dict[str, Any], *path: str) -> float | None:
    node: Any = data
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    if isinstance(node, bool) or node is None:
        return None
    if isinstance(node, (int, float)):
        return float(node)
    return None


def load_metrics(label: str, status: str, output_dir: Path) -> RunResult:
    files = result_files(output_dir)
    if not files:
        return RunResult(label=label, status=status)

    results_path = files[0]
    with results_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    mtmc_idf1 = (
        nested_number(data, "mtmc_idf1")
        or nested_number(data, "details_mtmc_idf1")
        or nested_number(data, "idf1")
    )
    c006_idf1 = nested_number(data, "per_camera", "S02_c006", "idf1") or nested_number(data, "s02_c006", "idf1")
    c006_mota = nested_number(data, "per_camera", "S02_c006", "mota") or nested_number(data, "s02_c006", "mota")
    id_switches_value = (
        nested_number(data, "mtmc_id_switches")
        or nested_number(data, "details_mtmc_id_switches")
        or nested_number(data, "id_switches")
    )
    id_switches = int(id_switches_value) if id_switches_value is not None else None

    return RunResult(
        label=label,
        status=status,
        mtmc_idf1=mtmc_idf1,
        c006_idf1=c006_idf1,
        c006_mota=c006_mota,
        id_switches=id_switches,
        results_path=results_path,
    )


def collect_run(run: RunConfig, output_root: Path, force: bool) -> RunResult:
    output_dir = output_root / run.label
    try:
        status_process = run_kaggle(["kernels", "status", run.slug], run.token_file)
    except FileNotFoundError as error:
        missing_name = Path(str(error.filename)).name if error.filename else ""
        if missing_name == run.token_file:
            print(f"warning: token file missing for {run.label}: {error.filename}", file=sys.stderr)
            return RunResult(label=run.label, status="TOKEN_MISSING")
        print(f"warning: Kaggle CLI not found for {run.label}: {error}", file=sys.stderr)
        return RunResult(label=run.label, status="CLI_MISSING")
    except OSError as error:
        print(f"warning: status check failed for {run.label}: {error}", file=sys.stderr)
        return RunResult(label=run.label, status="STATUS_ERROR")

    status_output = command_text(status_process)
    status = parse_status(status_output)
    if status_process.returncode != 0:
        print(f"warning: status check failed for {run.label}: {status_output}", file=sys.stderr)
        return RunResult(label=run.label, status="STATUS_ERROR")

    if status != COMPLETE_STATUS:
        return RunResult(label=run.label, status=status)

    downloaded = download_if_needed(run, output_dir, force)
    if not downloaded:
        return RunResult(label=run.label, status="DOWNLOAD_ERROR")

    try:
        return load_metrics(run.label, status, output_dir)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        print(f"warning: could not parse results for {run.label}: {error}", file=sys.stderr)
        return RunResult(label=run.label, status="PARSE_ERROR")


def format_score(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.3f}"


def format_delta(value: float | None, base_value: float | None, base_ready: bool) -> str:
    if not base_ready:
        return "pending base"
    if value is None or base_value is None:
        return "-"
    return f"{(value - base_value) * 100:+.3f}"


def render_table(results: list[RunResult]) -> str:
    base = next((result for result in results if result.label == "base"), None)
    base_ready = base is not None and base.status == COMPLETE_STATUS and base.mtmc_idf1 is not None
    base_c006_ready = base is not None and base.status == COMPLETE_STATUS and base.c006_idf1 is not None

    headers = [
        "run",
        "status",
        "S02 MTMC IDF1",
        f"{DELTA}IDF1 vs base",
        "S02_c006 IDF1",
        f"{DELTA}c006 vs base",
        "ID switches",
    ]
    rows = []
    for result in results:
        rows.append(
            [
                result.label,
                result.status,
                format_score(result.mtmc_idf1),
                format_delta(result.mtmc_idf1, base.mtmc_idf1 if base else None, base_ready),
                format_score(result.c006_idf1),
                format_delta(result.c006_idf1, base.c006_idf1 if base else None, base_c006_ready),
                str(result.id_switches) if result.id_switches is not None else "-",
            ]
        )

    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]

    lines = ["  ".join(header.ljust(width) for header, width in zip(headers, widths))]
    lines.append("  ".join("-" * width for width in widths))
    for row in rows:
        lines.append("  ".join(cell.ljust(width) for cell, width in zip(row, widths)))
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    runs = [RunConfig(*run) for run in RUNS]
    output_root = Path("tmp_ablation")
    results = [collect_run(run, output_root, args.force) for run in runs]
    print(render_table(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())