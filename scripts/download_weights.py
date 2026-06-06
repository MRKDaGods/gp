#!/usr/bin/env python
"""Download pipeline + VeRi-776 model weights from the public Kaggle dataset.

All required checkpoints are consolidated into ONE public Kaggle dataset
(see ``configs/weights_manifest.yaml``). This script lets you fetch a single
model set or everything, then verifies each file against its pinned SHA-256.

Model sets:
  vehicle-mtmc-14e   Vehicle MTMC 14e B1 production (CityFlowV2)
  vehicle-mtmc-14k   Vehicle MTMC 14k v1 K7 R50-IBN research (superset of 14e)
  person-mtmc        Person MTMC 12b (WILDTRACK MVDeTr)
  veri               VeRi-776 paper two-stream fusion (93.32% mAP)
  all                Everything above

Usage:
    python scripts/download_weights.py                 # interactive picker
    python scripts/download_weights.py --set all
    python scripts/download_weights.py --set veri --set person-mtmc
    python scripts/download_weights.py --list          # show sets + files, no download
    python scripts/download_weights.py --set all --dry-run
    python scripts/download_weights.py --set veri --force   # re-download even if present

Requirements:
  - A (free) Kaggle account and API token at ~/.kaggle/kaggle.json
    (https://www.kaggle.com/docs/api). The dataset is public but the Kaggle
    API still requires an authenticated token.
  - The ``kaggle`` CLI (installed via requirements.txt) and PyYAML.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "configs" / "weights_manifest.yaml"
CHUNK = 8 * 1024 * 1024


def load_manifest() -> dict:
    with open(MANIFEST_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def human(n: int) -> str:
    f = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if f < 1024 or unit == "GB":
            return f"{f:.1f}{unit}" if unit != "B" else f"{int(f)}B"
        f /= 1024
    return f"{f:.1f}GB"


def files_for_sets(manifest: dict, set_names: list[str]) -> list[dict]:
    if "all" in set_names:
        return list(manifest["files"])
    valid = set(manifest["sets"])
    unknown = [s for s in set_names if s not in valid]
    if unknown:
        sys.exit(f"Unknown set(s): {', '.join(unknown)}. Valid: {', '.join(sorted(valid))}, all")
    wanted = set(set_names)
    selected, seen = [], set()
    for f in manifest["files"]:
        if wanted & set(f["sets"]) and f["name"] not in seen:
            selected.append(f)
            seen.add(f["name"])
    return selected


def print_listing(manifest: dict) -> None:
    print(f"Kaggle dataset: {manifest['kaggle_dataset']}  (license: {manifest.get('license', 'n/a')})\n")
    print("Model sets:")
    for name, spec in manifest["sets"].items():
        files = files_for_sets(manifest, [name])
        total = sum(f["size_bytes"] for f in files)
        desc = " ".join(spec["description"].split())
        print(f"  - {name:18s} {human(total):>8s} ({len(files)} files)")
        print(f"      {desc}")
    all_files = manifest["files"]
    print(f"\n  - {'all':18s} {human(sum(f['size_bytes'] for f in all_files)):>8s} ({len(all_files)} files)")


def prompt_for_set(manifest: dict) -> list[str]:
    names = list(manifest["sets"]) + ["all"]
    print("Select a model set to download:\n")
    for i, name in enumerate(names, 1):
        if name == "all":
            files = manifest["files"]
            desc = "Everything above"
        else:
            files = files_for_sets(manifest, [name])
            desc = " ".join(manifest["sets"][name]["description"].split())
        total = human(sum(f["size_bytes"] for f in files))
        print(f"  [{i}] {name:18s} {total:>8s} - {desc}")
    print()
    while True:
        choice = input(f"Enter choice [1-{len(names)}] (or 'q' to quit): ").strip().lower()
        if choice in ("q", "quit", "exit"):
            sys.exit(0)
        if choice.isdigit() and 1 <= int(choice) <= len(names):
            return [names[int(choice) - 1]]
        print("Invalid choice.")


def kaggle_available() -> bool:
    return shutil.which("kaggle") is not None


def download_one(dataset: str, name: str, dest: Path, tmp_root: Path) -> Path:
    """Download a single member from the Kaggle dataset and return its path."""
    with tempfile.TemporaryDirectory(dir=tmp_root) as td:
        td = Path(td)
        cmd = ["kaggle", "datasets", "download", "-d", dataset, "-f", name, "-p", str(td)]
        proc = subprocess.run(cmd, text=True, capture_output=True)
        if proc.returncode != 0:
            msg = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError(f"kaggle download failed for {name}: {msg}")
        # Kaggle may deliver the file directly or as <name>.zip
        direct = td / name
        zipped = td / f"{name}.zip"
        if zipped.exists():
            with zipfile.ZipFile(zipped) as zf:
                zf.extractall(td)
        if not direct.exists():
            # Some CLI versions name the single-file zip after the file with .zip
            candidates = [p for p in td.iterdir() if p.name == name]
            if not candidates:
                raise RuntimeError(f"downloaded archive did not contain {name}")
            direct = candidates[0]
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(direct), str(dest))
        return dest


def main() -> int:
    manifest = load_manifest()
    valid_sets = list(manifest["sets"]) + ["all"]

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", action="append", dest="sets", metavar="NAME",
                    help=f"Model set to download (repeatable). One of: {', '.join(valid_sets)}")
    ap.add_argument("--list", action="store_true", help="List sets and files, then exit.")
    ap.add_argument("--force", action="store_true", help="Re-download even if a valid file already exists.")
    ap.add_argument("--dry-run", action="store_true", help="Show what would be downloaded without fetching.")
    args = ap.parse_args()

    if args.list:
        print_listing(manifest)
        return 0

    set_names = args.sets
    if not set_names:
        if not sys.stdin.isatty():
            print_listing(manifest)
            print("\nNo --set given and not interactive. Re-run with e.g. --set all.")
            return 2
        set_names = prompt_for_set(manifest)

    selected = files_for_sets(manifest, set_names)
    dataset = manifest["kaggle_dataset"]
    total = sum(f["size_bytes"] for f in selected)
    print(f"\nSets: {', '.join(set_names)}  ->  {len(selected)} files, {human(total)} total")
    print(f"Source: kaggle datasets / {dataset}\n")

    if args.dry_run:
        for f in selected:
            print(f"  would fetch {f['name']:55s} -> {f['local_path']}  ({human(f['size_bytes'])})")
        return 0

    if not kaggle_available():
        print("ERROR: the `kaggle` CLI was not found. Install requirements.txt and place an")
        print("       API token at ~/.kaggle/kaggle.json (https://www.kaggle.com/docs/api).")
        return 2

    tmp_root = REPO_ROOT / "tmp_weight_downloads"
    tmp_root.mkdir(exist_ok=True)
    failures: list[str] = []
    try:
        for i, f in enumerate(selected, 1):
            dest = REPO_ROOT / f["local_path"]
            tag = f"[{i}/{len(selected)}] {f['name']}"
            if dest.exists() and not args.force:
                if sha256_of(dest) == f["sha256"]:
                    print(f"{tag}: already present and verified, skipping.")
                    continue
                print(f"{tag}: present but checksum mismatch -> re-downloading.")
            print(f"{tag}: downloading {human(f['size_bytes'])} -> {f['local_path']} ...")
            try:
                download_one(dataset, f["name"], dest, tmp_root)
            except Exception as exc:  # noqa: BLE001
                print(f"{tag}: FAILED ({exc})")
                failures.append(f["name"])
                continue
            got = sha256_of(dest)
            if got != f["sha256"]:
                print(f"{tag}: CHECKSUM MISMATCH (got {got[:12]}..., expected {f['sha256'][:12]}...)")
                failures.append(f["name"])
            else:
                print(f"{tag}: OK (sha256 {got[:12]}...)")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    print()
    if failures:
        print(f"Completed with {len(failures)} failure(s): {', '.join(failures)}")
        return 1
    print(f"All {len(selected)} file(s) downloaded and verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
