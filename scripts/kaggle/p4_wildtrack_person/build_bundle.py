"""Assemble + upload the athar-p4-bundle Kaggle dataset (private, mrkdagods).

Contents:
- src/  — `git archive` of the CURRENT athar-v2 HEAD, extracted (a plain
  tree: Kaggle auto-decompresses dataset archives, so shipping the tree
  directly is the only deterministic layout). src/GIT_SHA.txt records
  provenance.
- models/yolo26m.pt, models/osnet_x0_25_msmt17.pt
- gt/C1..C7.txt — v1-recipe WILDTRACK MOT ground truth
- profile_p4_wildtrack_person.yaml

Usage:
    python scripts/kaggle/p4_wildtrack_person/build_bundle.py [--update]

--update pushes a new version of an existing dataset; without it the
dataset is created. Requires the mrkdagods KAGGLE_API_TOKEN.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DATASET_ID = "mrkdagods/athar-p4-bundle"


def run(cmd, cwd=None) -> str:
    print("$", " ".join(map(str, cmd)))
    return subprocess.check_output(list(map(str, cmd)), cwd=cwd, text=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action="store_true",
                        help="push a new version instead of creating the dataset")
    args = parser.parse_args()

    staging = Path(tempfile.mkdtemp(prefix="p4_bundle_"))
    print("staging:", staging)

    head = run(["git", "-C", PROJECT_ROOT, "rev-parse", "HEAD"]).strip()
    archive = staging / "src.tar"
    run(["git", "-C", PROJECT_ROOT, "archive", "-o", archive, "HEAD"])
    src_dir = staging / "src"
    src_dir.mkdir()
    with tarfile.open(archive) as tar:
        tar.extractall(src_dir)
    archive.unlink()
    (src_dir / "GIT_SHA.txt").write_text(head + "\n", encoding="utf-8")

    (staging / "models").mkdir()
    shutil.copy2(PROJECT_ROOT / "models" / "detection" / "yolo26m.pt",
                 staging / "models" / "yolo26m.pt")
    shutil.copy2(PROJECT_ROOT / "models" / "tracker" / "osnet_x0_25_msmt17.pt",
                 staging / "models" / "osnet_x0_25_msmt17.pt")

    gt_src = PROJECT_ROOT / "data" / "raw" / "wildtrack" / "manifests" / "ground_truth"
    shutil.copytree(gt_src, staging / "gt")
    shutil.copy2(HERE / "profile_p4_wildtrack_person.yaml", staging)

    (staging / "dataset-metadata.json").write_text(
        '{\n  "title": "athar-p4-bundle",\n'
        f'  "id": "{DATASET_ID}",\n'
        '  "licenses": [{"name": "CC0-1.0"}]\n}\n',
        encoding="utf-8",
    )

    verb = ["version", "-m", f"src @ {head[:12]}"] if args.update else ["create"]
    run(["kaggle", "datasets", *verb, "-p", staging, "--dir-mode", "zip"])
    print(f"bundle {'updated' if args.update else 'created'}: {DATASET_ID} @ {head[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
