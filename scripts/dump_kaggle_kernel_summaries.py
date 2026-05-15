"""Download small Kaggle kernel summary artifacts for registry cross-checks."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = PROJECT_ROOT / "configs" / "model_registry.yaml"
MANIFEST_PATH = PROJECT_ROOT / "docs" / "_data" / "kaggle_kernel_summaries.json"
TMP_ROOT = PROJECT_ROOT / "_tmp_kernel_summaries"
MAX_KEEP_BYTES = 5 * 1024 * 1024
ALLOWED_FILE_PATTERN = r".*(summary|recipe|final_metrics|train_log|eval_results).*\.(json|txt|log|csv)$"
BLOCKED_EXTENSIONS = {".pth", ".pt", ".bin"}
ACCOUNT_TOKENS = [
    ("gumfreddy", ("gumfreddy_access_token",)),
    ("mrkdagods", ("mrkdagods_access_token", "MRKDaGods__access_token")),
    ("ali369", ("ali369_access_token", "ali_369_access_token")),
    ("yahiaakhalafallah", ("yahiaakhalafallah_access_token",)),
]


def load_registry() -> dict[str, Any]:
    loaded = OmegaConf.load(REGISTRY_PATH)
    data = OmegaConf.to_container(loaded, resolve=True)
    if not isinstance(data, dict):
        raise ValueError(f"Registry must be a mapping: {REGISTRY_PATH}")
    return data


def registry_kernel_slugs(registry: dict[str, Any]) -> list[str]:
    slugs: set[str] = set()
    for model in registry.get("models", []):
        notebook_ref = model.get("notebook_or_kernel_ref")
        if is_kernel_slug(notebook_ref):
            slugs.add(notebook_ref)
        for metric in model.get("metrics", []):
            source = metric.get("source") or {}
            if isinstance(source, dict) and is_kernel_slug(source.get("kernel")):
                slugs.add(source["kernel"])
            elif isinstance(source, str) and ":" in source:
                maybe_slug = source.split(":", 1)[0]
                if is_kernel_slug(maybe_slug):
                    slugs.add(maybe_slug)
    return sorted(slugs)


def is_kernel_slug(value: Any) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+", value))


def flatten_slug(slug: str) -> str:
    return slug.replace("/", "__")


def token_path(token_file: str) -> Path:
    return Path.home() / ".kaggle" / token_file


def resolve_token_file(token_files: tuple[str, ...]) -> str | None:
    for token_file in token_files:
        if token_path(token_file).exists():
            return token_file
    return None


def account_env(token_file: str) -> dict[str, str] | None:
    path = token_path(token_file)
    if not path.exists():
        return None
    token = path.read_text(encoding="utf-8").strip()
    if not token:
        return None
    env = os.environ.copy()
    env["KAGGLE_API_TOKEN"] = token
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def copy_token_to_kaggle_json(token_file: str) -> bool:
    path = token_path(token_file)
    if not path.exists():
        return False
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, kaggle_dir / "kaggle.json")
    return True


def run_kaggle(args: list[str], *, env: dict[str, str], timeout: int = 240) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["kaggle", *args],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def purge_large_or_binary_files(directory: Path) -> list[str]:
    removed: list[str] = []
    if not directory.exists():
        return removed
    for path in directory.rglob("*"):
        if not path.is_file():
            continue
        if path.stat().st_size > MAX_KEEP_BYTES or path.suffix.lower() in BLOCKED_EXTENSIONS:
            removed.append(str(path.relative_to(PROJECT_ROOT)))
            path.unlink(missing_ok=True)
    return removed


def load_json_file(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def walk_numeric(data: Any, prefix: tuple[str, ...] = ()) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            metrics.update(walk_numeric(value, (*prefix, str(key))))
    elif isinstance(data, list):
        for index, value in enumerate(data):
            metrics.update(walk_numeric(value, (*prefix, str(index))))
    elif isinstance(data, (int, float)) and not isinstance(data, bool):
        metrics[".".join(prefix)] = float(data)
    return metrics


def summarize_downloaded_files(out_dir: Path) -> tuple[list[dict[str, Any]], dict[str, float]]:
    summary_files: list[dict[str, Any]] = []
    metrics: dict[str, float] = {}
    for path in sorted(out_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.stat().st_size > MAX_KEEP_BYTES or path.suffix.lower() in BLOCKED_EXTENSIONS:
            continue
        relative = str(path.relative_to(out_dir)).replace("\\", "/")
        entry: dict[str, Any] = {"path": relative, "size_bytes": path.stat().st_size}
        if path.suffix.lower() == ".json":
            data = load_json_file(path)
            entry["data"] = data
            if data is not None:
                for key, value in walk_numeric(data).items():
                    metrics[f"{path.name}.{key}"] = value
        summary_files.append(entry)
    return summary_files, metrics


def parse_status(status_text: str) -> tuple[str | None, str | None]:
    status = None
    last_run = None
    for line in status_text.splitlines():
        lower = line.lower()
        if "status" in lower and ":" in line:
            status = line.split(":", 1)[1].strip()
        if ("last run" in lower or "last run time" in lower) and ":" in line:
            last_run = line.split(":", 1)[1].strip()
    return status, last_run


def fetch_kernel(slug: str, account_name: str, token_file: str) -> dict[str, Any] | None:
    env = account_env(token_file)
    if env is None:
        return None
    copy_token_to_kaggle_json(token_file)
    out_dir = TMP_ROOT / flatten_slug(slug)
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    status_proc = run_kaggle(["kernels", "status", slug], env=env, timeout=60)
    output_proc = run_kaggle(
        [
            "kernels",
            "output",
            slug,
            "-p",
            str(out_dir),
            "--force",
            "--quiet",
            "--file-pattern",
            ALLOWED_FILE_PATTERN,
        ],
        env=env,
        timeout=600,
    )
    removed = purge_large_or_binary_files(out_dir)
    if output_proc.returncode != 0:
        shutil.rmtree(out_dir, ignore_errors=True)
        reason = (output_proc.stderr or output_proc.stdout or "kaggle kernels output failed").strip()
        return {
            "slug": slug,
            "status": "UNAVAILABLE",
            "last_run": None,
            "account": account_name,
            "reason": reason,
            "metrics": {},
            "summary_files": [],
            "removed_files": removed,
        }

    summary_files, metrics = summarize_downloaded_files(out_dir)
    status, last_run = parse_status(status_proc.stdout + "\n" + status_proc.stderr)
    result = {
        "slug": slug,
        "status": status or "AVAILABLE",
        "last_run": last_run,
        "account": account_name,
        "reason": None,
        "metrics": metrics,
        "summary_files": summary_files,
        "removed_files": removed,
    }
    shutil.rmtree(out_dir, ignore_errors=True)
    return result


def fetch_with_all_accounts(slug: str) -> dict[str, Any]:
    unavailable: list[dict[str, Any]] = []
    for account_name, token_files in ACCOUNT_TOKENS:
        token_file = resolve_token_file(token_files)
        if token_file is None:
            unavailable.append({"account": account_name, "reason": f"missing token {token_files[0]}"})
            continue
        result = fetch_kernel(slug, account_name, token_file)
        if result is None:
            unavailable.append({"account": account_name, "reason": f"missing token {token_file}"})
            continue
        if result["status"] != "UNAVAILABLE":
            return result
        unavailable.append({"account": account_name, "reason": result.get("reason")})
    return {
        "slug": slug,
        "status": "UNAVAILABLE",
        "last_run": None,
        "account": None,
        "reason": "; ".join(f"{item['account']}: {item['reason']}" for item in unavailable),
        "metrics": {},
        "summary_files": [],
        "removed_files": [],
    }


def main() -> int:
    registry = load_registry()
    slugs = registry_kernel_slugs(registry)
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_entries: list[dict[str, Any]] = []
    try:
        for slug in slugs:
            print(f"Fetching {slug} ...")
            manifest_entries.append(fetch_with_all_accounts(slug))
    finally:
        shutil.rmtree(TMP_ROOT, ignore_errors=True)

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "file_pattern": ALLOWED_FILE_PATTERN,
        "max_keep_bytes": MAX_KEEP_BYTES,
        "kernels": manifest_entries,
        "summary": {
            "total": len(manifest_entries),
            "available": sum(1 for item in manifest_entries if item.get("status") != "UNAVAILABLE"),
            "unavailable": sum(1 for item in manifest_entries if item.get("status") == "UNAVAILABLE"),
        },
    }
    MANIFEST_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(
        "Wrote "
        f"{MANIFEST_PATH.relative_to(PROJECT_ROOT)} "
        f"({payload['summary']['available']} available, {payload['summary']['unavailable']} unavailable)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())