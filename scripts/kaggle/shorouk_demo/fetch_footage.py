"""Download the demo cameras' EXACT Kaggle footage to data/raw/shorouk_demo/.

The demo kernel processes mrkdagods/shorouk-dataset files at the relative
path ``data/raw/shorouk_demo/<cam>/vdo.mp4``; this script materializes the
same layout locally with the same bytes, so the imported run manifests
resolve their evidence (clip playback, thumbnails re-cuts) against footage
that is bit-exact with what the pipeline saw — manifest sha256s verify.

We deliberately do NOT reuse data/raw/shorouk (the on-prem originals):
they are a different encode whose per-camera trim alignment against the
Kaggle rebuild is verified separately (verify_sync.py), not assumed.

Usage: python scripts/kaggle/shorouk_demo/fetch_footage.py [cam ...]
       (default: the kernel's GALLERY_CAMS + PROBE_CAM)
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET = "mrkdagods/shorouk-dataset"
DEST = PROJECT_ROOT / "data" / "raw" / "shorouk_demo"

sys.path.insert(0, str(Path(__file__).parent))
from athar_shorouk_demo import ALL_CAMS  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(cam: str) -> Path:
    out_dir = DEST / cam
    out = out_dir / "vdo.mp4"
    if out.is_file() and out.stat().st_size > 1e6:
        print(f"{cam}: already present ({out.stat().st_size / 1e6:.1f} MB)")
        return out
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "datasets", "download", DATASET,
         "-f", f"{cam}/vdo.mp4", "-p", str(out_dir)],
        check=True, env={**os.environ, "PYTHONUTF8": "1"},
    )
    # the CLI may deliver the file zipped; normalize to vdo.mp4
    zipped = next(out_dir.glob("*.zip"), None)
    if zipped is not None:
        with zipfile.ZipFile(zipped) as zf:
            member = next(m for m in zf.namelist() if m.endswith("vdo.mp4"))
            with zf.open(member) as src, open(out, "wb") as dst:
                while chunk := src.read(1 << 20):
                    dst.write(chunk)
        zipped.unlink()
    if not out.is_file():
        candidate = next(out_dir.glob("vdo.mp4*"), None)
        if candidate is not None and candidate != out:
            candidate.rename(out)
    assert out.is_file(), f"{cam}: download produced no vdo.mp4 under {out_dir}"
    print(f"{cam}: {out.stat().st_size / 1e6:.1f} MB  sha256={sha256_file(out)[:16]}...")
    return out


def main() -> int:
    cams = sys.argv[1:] or ALL_CAMS
    print(f"fetching {len(cams)} cameras from {DATASET} -> {DEST}")
    for cam in cams:
        fetch(cam)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
