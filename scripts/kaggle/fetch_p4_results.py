"""Fetch the Gate P4 results from the Kaggle kernel output.

Downloads ONLY p4_metrics.json + p4_run_artifacts.tar.gz (the kernel's
early versions also persisted a whole source tree — never pull that) and
extracts the artifacts under data/goldens/wildtrack_p4_run/.

Windows note: the Kaggle CLI dies on cp1252 writing logs/filenames — this
script forces UTF-8 via the python API instead.

Usage: python scripts/kaggle/fetch_p4_results.py
"""

from __future__ import annotations

import json
import os
import sys
import tarfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
KERNEL = "mrkdagods/athar-p4-wildtrack-person-baseline"
DEST = PROJECT_ROOT / "data" / "goldens" / "wildtrack_p4_run"


def main() -> int:
    import subprocess

    DEST.mkdir(parents=True, exist_ok=True)
    # PYTHONUTF8=1 fixes the CLI's cp1252 crash writing the kernel log
    subprocess.run(
        ["kaggle", "kernels", "output", KERNEL,
         "--file-pattern", r"^p4_(metrics\.json|run_artifacts\.tar\.gz)$",
         "-p", str(DEST)],
        check=False, env={**os.environ, "PYTHONUTF8": "1"},
    )
    metrics_path = DEST / "p4_metrics.json"
    if not metrics_path.exists():
        print("p4_metrics.json not found — kernel not finished or failed", file=sys.stderr)
        return 1
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    print(json.dumps(metrics, indent=2))
    tarball = DEST / "p4_run_artifacts.tar.gz"
    if tarball.exists():
        with tarfile.open(tarball, "r:gz") as tar:
            tar.extractall(DEST / "run_artifacts")
        print(f"artifacts extracted to {DEST / 'run_artifacts'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
