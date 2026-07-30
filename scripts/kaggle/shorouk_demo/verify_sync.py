"""Verify per-camera time alignment: local originals vs Kaggle rebuild.

The Kaggle dataset (mrkdagods/shorouk-dataset) was rebuilt from the raw
DVR exports with per-camera seek trims; the local data/raw/shorouk
originals came from an earlier trim of the same exports. Nothing
guarantees the two trims landed on the same frame — this script measures
the actual per-camera lag instead of assuming it.

Method: per camera, decode two 45s windows (sync-window start, and the
10-minute mark) from BOTH copies via the project's frame-exact
FrameSource, reduce each frame to a 4x4 grid of mean luminances, and
normalized-cross-correlate the signature series over integer frame lags
of +/-375 (15s @ 25fps). Reports best lag + correlation per window.

Pass criteria (calibrated 2026-07-30 against the real footage): |lag| <=
3 frames (120ms) per window, windows agreeing within 2 frames (no
drift), correlation >= 0.60. Cross-encoder NCC (local HEVC/NVENC vs
Kaggle libx264) tops out well below same-encoder levels — the LAG and
its inter-window consistency are the verdict, correlation is only a
peak-confidence floor. A low-motion window can produce a spurious peak
(weak corr + a lag the other window contradicts); re-check such cameras
at motion-rich timestamps before concluding a real offset.

Usage: python scripts/kaggle/shorouk_demo/verify_sync.py [cam ...]
       (default: the kernel's demo cameras; needs fetch_footage.py first)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from athar_shorouk_demo import ALL_CAMS  # noqa: E402

LOCAL = PROJECT_ROOT / "data" / "raw" / "shorouk"
KAGGLE = PROJECT_ROOT / "data" / "raw" / "shorouk_demo"

MAX_LAG = 375                     # +/- 15s @ 25fps
WIN = 375                         # correlation window: 15s of frames
WINDOWS = {"start": 0, "mid_10min": 15000}  # first frame of each probe span
GRID = 4

PASS_MAX_ABS_LAG = 3
PASS_MIN_CORR = 0.60
PASS_MAX_DRIFT = 2


def signatures(video: Path, first: int, count: int) -> np.ndarray:
    """(count, GRID*GRID) float32 mean-luminance signatures, frame-exact."""
    from athar.components.framesources.video import create_video_source

    indices = list(range(first, first + count))
    source = create_video_source("sync", video, indices=indices)
    out = np.zeros((count, GRID * GRID), dtype=np.float32)
    pos = 0
    for batch in source.batches(32):
        for image in batch.images():
            gray = image.mean(axis=2) if image.ndim == 3 else image
            h, w = gray.shape
            cells = (
                gray[: h - h % GRID, : w - w % GRID]
                .reshape(GRID, h // GRID, GRID, w // GRID)
                .mean(axis=(1, 3))
            )
            out[pos] = cells.reshape(-1)
            pos += 1
    assert pos == count, f"{video}: decoded {pos}/{count} frames"
    return out


def best_lag(local_sig: np.ndarray, kaggle_sig: np.ndarray) -> tuple[int, float]:
    """Lag k maximizing NCC of local[MAX_LAG:MAX_LAG+WIN] vs kaggle shifted
    by k. Positive lag = kaggle copy starts EARLIER (its frame MAX_LAG+k
    shows what local frame MAX_LAG shows)."""
    a = local_sig[MAX_LAG : MAX_LAG + WIN].astype(np.float64)
    a = a - a.mean(axis=0)
    an = np.sqrt((a * a).sum())
    best = (0, -2.0)
    for k in range(-MAX_LAG, MAX_LAG + 1):
        b = kaggle_sig[MAX_LAG + k : MAX_LAG + k + WIN].astype(np.float64)
        b = b - b.mean(axis=0)
        denom = an * np.sqrt((b * b).sum())
        corr = float((a * b).sum() / denom) if denom > 0 else -1.0
        if corr > best[1]:
            best = (k, corr)
    return best


def main() -> int:
    cams = sys.argv[1:] or ALL_CAMS
    span = 2 * MAX_LAG + WIN  # frames decoded per window per file
    report: dict[str, dict] = {}
    all_pass = True
    for cam in cams:
        local = LOCAL / cam / "vdo.mp4"
        kaggle = KAGGLE / cam / "vdo.mp4"
        assert local.is_file(), f"missing local original: {local}"
        assert kaggle.is_file(), f"missing Kaggle copy: {kaggle} (fetch_footage.py)"
        cam_report = {}
        for name, first in WINDOWS.items():
            lsig = signatures(local, first, span)
            ksig = signatures(kaggle, first, span)
            lag, corr = best_lag(lsig, ksig)
            ok = abs(lag) <= PASS_MAX_ABS_LAG and corr >= PASS_MIN_CORR
            cam_report[name] = {"lag_frames": lag, "lag_s": lag / 25.0,
                                "corr": round(corr, 4), "ok": ok}
            print(f"{cam} {name:>9}: lag={lag:+4d} frames ({lag / 25.0:+.2f}s) "
                  f"corr={corr:.4f} {'OK' if ok else 'MISMATCH'}", flush=True)
        lags = [w["lag_frames"] for w in cam_report.values()]
        cam_report["drift_frames"] = lags[1] - lags[0]
        cam_report["pass"] = (
            all(w["ok"] for w in cam_report.values() if isinstance(w, dict))
            and abs(lags[1] - lags[0]) <= PASS_MAX_DRIFT
        )
        all_pass &= cam_report["pass"]
        report[cam] = cam_report

    out = Path(__file__).parent / "sync_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")
    print("SYNC VERIFIED" if all_pass else "SYNC MISMATCH — do not mix copies")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
