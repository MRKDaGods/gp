"""Fetch the Gate P2 CityFlow goldens from Kaggle into data/goldens/.

Usage (needs a Kaggle token that can see the kernel — the goldens kernel
lives on the yahiaakhalafallah account until its outputs are shared):

    python scripts/kaggle/fetch_p2_goldens.py [--owner yahiaakhalafallah]

Downloads cityflow_b1_goldens.tar.gz from the athar-p2-cityflow-goldens
kernel output, extracts to data/goldens/cityflow_b1/, and verifies every
file against the sha256 map in provenance.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOLDENS_DIR = PROJECT_ROOT / "data" / "goldens"
KERNEL_SLUG = "athar-p2-cityflow-goldens"
TARBALL = "cityflow_b1_goldens.tar.gz"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--owner", default="yahiaakhalafallah")
    parser.add_argument("--tarball", type=Path, default=None,
                        help="Skip the download; use an already-downloaded tarball")
    args = parser.parse_args()

    if args.tarball is not None:
        tar_path = args.tarball
    else:
        dl_dir = Path(tempfile.mkdtemp(prefix="p2_goldens_"))
        ref = f"{args.owner}/{KERNEL_SLUG}"
        print(f"downloading {TARBALL} from kernel {ref} ...")
        result = subprocess.run(
            ["kaggle", "kernels", "output", ref,
             "--file-pattern", "^" + TARBALL.replace(".", r"\.") + "$",
             "-p", str(dl_dir)],
            capture_output=True, text=True,
        )
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        tar_path = dl_dir / TARBALL
        if not tar_path.exists():
            print(
                "\nDownload failed. Either the kernel has not been run yet, or this "
                "Kaggle token cannot see it (it is private to the "
                f"{args.owner} account).", file=sys.stderr,
            )
            return 1

    GOLDENS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"extracting {tar_path} -> {GOLDENS_DIR}")
    with tarfile.open(str(tar_path), "r:gz") as tar:
        tar.extractall(str(GOLDENS_DIR))

    root = GOLDENS_DIR / "cityflow_b1_goldens"
    provenance = json.loads((root / "provenance.json").read_text(encoding="utf-8"))
    bad = []
    for rel, expected in provenance["file_sha256"].items():
        actual = sha256_of(root / rel)
        if actual != expected:
            bad.append(rel)
    if bad:
        print(f"sha256 MISMATCH for {len(bad)} files: {bad[:5]}", file=sys.stderr)
        return 1
    print(f"verified {len(provenance['file_sha256'])} files against provenance sha256s")
    print(f"baseline: IDF1={provenance['metrics']['mtmc_idf1']} "
          f"id_sw={provenance['metrics']['id_switches']} "
          f"v1_commit={provenance['v1_commit'][:12]}")
    print(f"goldens ready at {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
