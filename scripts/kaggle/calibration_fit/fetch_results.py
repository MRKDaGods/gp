"""Fetch + freeze the calibration-fit kernel results.

Pulls stream_calibrations.json and calibration_fit_report.json from the
kernel output, validates the calibrations through the serving-side
loader, and freezes them into the repo:

- configs/calibrations/cityflowv2_s02.json   (the deployable artifact)
- scripts/kaggle/calibration_fit/fit_report_<date>.json  (provenance)

Windows note: PYTHONUTF8=1 for the Kaggle CLI (cp1252 crash).

Usage: python scripts/kaggle/calibration_fit/fetch_results.py
"""

from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
KERNEL = "mrkdagods/athar-calibration-fit-cityflow"
SCRATCH = PROJECT_ROOT / "data" / "goldens" / "calibration_fit_out"
FROZEN = PROJECT_ROOT / "configs" / "calibrations" / "cityflowv2_s02.json"


def main() -> int:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "kernels", "output", KERNEL,
         "--file-pattern", r"^(stream_calibrations|calibration_fit_report)\.json$",
         "-p", str(SCRATCH)],
        check=False, env={**os.environ, "PYTHONUTF8": "1"},
    )
    calibrations_path = SCRATCH / "stream_calibrations.json"
    report_path = SCRATCH / "calibration_fit_report.json"
    if not calibrations_path.exists():
        print("stream_calibrations.json not found - kernel not finished or failed",
              file=sys.stderr)
        return 1

    from athar.search.calibration import StreamCalibrations

    calibrations = StreamCalibrations.load(calibrations_path)
    for name, calibration in calibrations.streams.items():
        assert calibration.probability(0.95) > calibration.probability(0.3), name
        print(f"{name}: midpoint {calibration.midpoint:.4f} "
              f"scale {calibration.scale:.4f} pairs {calibration.num_pairs} "
              f"({calibration.fitted_on})")

    FROZEN.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(calibrations_path, FROZEN)
    stamp = dt.date.today().strftime("%Y%m%d")
    frozen_report = Path(__file__).parent / f"fit_report_{stamp}.json"
    if report_path.exists():
        shutil.copy2(report_path, frozen_report)
        print(json.dumps(json.loads(report_path.read_text("utf-8")), indent=2))
    print(f"frozen: {FROZEN}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
