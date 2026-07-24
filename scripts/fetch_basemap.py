"""Provision the offline PMTiles basemap for the web map view.

Extracts a small bounding-box slice of the Protomaps daily planet build
(OpenStreetMap data) into ``web/public/maps/basemap.pmtiles``, which the
frontend serves same-origin — the deployed app never fetches a tile from
the network (air-gap: this script runs on a connected ADMIN box; the
output file is carried onto the deployment alongside the app build).

Uses the official ``pmtiles`` CLI (go-pmtiles), downloaded on first use
into ``.tools/``. The bbox defaults to the extent of
``configs/camera_locations.json`` plus padding, so a new deployment site
just needs its camera survey before running this.

Usage (repo root):
    python scripts/fetch_basemap.py [--bbox MINLON,MINLAT,MAXLON,MAXLAT]
                                    [--build YYYYMMDD]
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import platform
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TOOLS = REPO / ".tools"
OUT = REPO / "web" / "public" / "maps" / "basemap.pmtiles"
CLI_VERSION = "1.31.2"
PAD_DEG = 0.012  # ~1.2 km of context around the outermost cameras


def cli_path() -> Path:
    exe = TOOLS / ("pmtiles.exe" if sys.platform == "win32" else "pmtiles")
    if exe.exists():
        return exe
    system = {"win32": "Windows", "darwin": "Darwin"}.get(sys.platform, "Linux")
    arch = "arm64" if platform.machine().lower() in ("arm64", "aarch64") else "x86_64"
    url = (
        f"https://github.com/protomaps/go-pmtiles/releases/download/"
        f"v{CLI_VERSION}/go-pmtiles_{CLI_VERSION}_{system}_{arch}.zip"
    )
    print(f"downloading pmtiles CLI: {url}")
    TOOLS.mkdir(exist_ok=True)
    with urllib.request.urlopen(url) as resp:
        archive = zipfile.ZipFile(io.BytesIO(resp.read()))
    member = next(n for n in archive.namelist() if n.startswith("pmtiles"))
    exe.write_bytes(archive.read(member))
    exe.chmod(0o755)
    return exe


def default_bbox() -> str:
    cams = json.loads(
        (REPO / "configs" / "camera_locations.json").read_text(encoding="utf-8")
    )
    lats = [c["lat"] for c in cams.values()]
    lngs = [c["lng"] for c in cams.values()]
    return (
        f"{min(lngs) - PAD_DEG},{min(lats) - PAD_DEG},"
        f"{max(lngs) + PAD_DEG},{max(lats) + PAD_DEG}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bbox", default=None, help="minlon,minlat,maxlon,maxlat")
    parser.add_argument(
        "--build", default=None,
        help="Protomaps daily build date YYYYMMDD (default: yesterday)",
    )
    args = parser.parse_args()

    bbox = args.bbox or default_bbox()
    build = args.build or (
        dt.date.today() - dt.timedelta(days=1)
    ).strftime("%Y%m%d")
    source = f"https://build.protomaps.com/{build}.pmtiles"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(cli_path()), "extract", source, str(OUT), f"--bbox={bbox}"]
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB, bbox {bbox})")


if __name__ == "__main__":
    main()
