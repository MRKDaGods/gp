"""Fetch the demo kernel's runs into the local run store.

Pulls ONLY the exported artifacts (never the kernel's whole output — the
raw runs/ mirror alone is gigabytes): shorouk_demo_summary.json +
gallery_run.tar.gz + probe_run.tar.gz, then extracts the run dirs under
data/runs/. Prints the acceptance verdict and the imported run ids.

Usage: python scripts/kaggle/shorouk_demo/fetch_results.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
KERNEL = "mrkdagods/athar-shorouk-demo"
SCRATCH = PROJECT_ROOT / "data" / "goldens" / "shorouk_demo_out"
RUNS_ROOT = PROJECT_ROOT / "data" / "runs"


def pull(pattern: str) -> None:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "kernels", "output", KERNEL,
         "--file-pattern", pattern, "-p", str(SCRATCH)],
        check=False, env={**os.environ, "PYTHONUTF8": "1"},
    )


def main() -> int:
    pull(r"^(shorouk_demo_summary\.json|gallery_run\.tar\.gz|probe_run\.tar\.gz)$")

    summary_path = SCRATCH / "shorouk_demo_summary.json"
    if not summary_path.exists():
        print("summary not found — kernel not finished or failed", file=sys.stderr)
        return 1
    summary = json.loads(summary_path.read_text("utf-8"))
    acceptance = summary.get("acceptance", {})
    print(f"gallery run: {summary['gallery_run_id']}  cams={summary['gallery_cams']}")
    print(f"probe run:   {summary['probe_run_id']}  cam={summary['probe_cam']}")
    print(f"identities: {summary['num_identities']}  "
          f"cross-camera: {summary['num_cross_camera']} "
          f"(vehicle {acceptance.get('cross_camera_vehicle_identities')}, "
          f"person {acceptance.get('cross_camera_person_identities')})")
    print(f"pre-cut clips: {acceptance.get('evidence_clips_precut')}")
    print(f"acceptance PASS: {acceptance.get('PASS')}")

    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    imported = []
    for name in ("gallery_run.tar.gz", "probe_run.tar.gz"):
        tar_path = SCRATCH / name
        if not tar_path.exists():
            print(f"{name} missing — not exported?", file=sys.stderr)
            return 1
        with tarfile.open(tar_path) as tar:
            top = {m.name.split("/", 1)[0] for m in tar.getmembers()}
            run_id = next(iter(top))
            if (RUNS_ROOT / run_id).exists():
                print(f"{run_id}: already imported — skipping extract")
            else:
                tar.extractall(RUNS_ROOT)
                print(f"{run_id}: extracted -> {RUNS_ROOT / run_id}")
            imported.append(run_id)

    print("\nnext: python scripts/kaggle/shorouk_demo/fetch_footage.py"
          "  (evidence for clip playback)")
    print(f"search smoke: .venv-v2/Scripts/python.exe -m athar.cli.main search "
          f"--gallery {imported[0]} --probe {imported[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
