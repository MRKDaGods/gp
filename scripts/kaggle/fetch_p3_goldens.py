"""Fetch the Gate P3 WILDTRACK goldens from Kaggle into data/goldens/.

The golden is the cached 12a MVDeTr detector output (test.txt) that the
whole v1 WILDTRACK headline chain (12b tracking + ground-plane eval, and the
14w verify kernel) consumed. It lives in the output of the public resume
kernel on the yahiaakhalafallah account.

Usage (needs a Kaggle token that can see the kernel):

    python scripts/kaggle/fetch_p3_goldens.py [--owner yahiaakhalafallah]

Downloads test.txt (+ wildtrack_gt.txt for reference) from the
12a-resume-emit-wildtrack-test-txt kernel output into
data/goldens/wildtrack_b1_goldens/ and verifies the pinned sha256s.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOLDENS_DIR = PROJECT_ROOT / "data" / "goldens" / "wildtrack_b1_goldens"
KERNEL_SLUG = "12a-resume-emit-wildtrack-test-txt"

EXPECTED_SHA256 = {
    "test.txt": "da47ba2cc29be21a405fd6d7bb06de5351ad61327029a6db77d44b2d9beffbe4",
    "wildtrack_gt.txt": "982be8b937f1d4681563b940676b612d43003ad5615ae530d0d395a4ffdc9cc6",
}


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--owner", default="yahiaakhalafallah")
    args = parser.parse_args()

    GOLDENS_DIR.mkdir(parents=True, exist_ok=True)
    ref = f"{args.owner}/{KERNEL_SLUG}"
    for name in EXPECTED_SHA256:
        print(f"downloading {name} from kernel {ref} ...")
        result = subprocess.run(
            ["kaggle", "kernels", "output", ref,
             "--file-pattern", "^" + name.replace(".", r"\.") + "$",
             "-p", str(GOLDENS_DIR)],
            capture_output=True, text=True,
        )
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        if not (GOLDENS_DIR / name).exists():
            print(
                f"\nDownload of {name} failed. Either the kernel output moved, or "
                f"this Kaggle token cannot see kernel {ref}.", file=sys.stderr,
            )
            return 1

    bad = []
    for name, expected in EXPECTED_SHA256.items():
        actual = sha256_of(GOLDENS_DIR / name)
        if actual != expected:
            bad.append(f"{name}: expected {expected[:12]}..., got {actual[:12]}...")
    if bad:
        print("sha256 MISMATCH:\n  " + "\n  ".join(bad), file=sys.stderr)
        return 1
    print(f"verified {len(EXPECTED_SHA256)} files against pinned sha256s")
    print(f"goldens ready at {GOLDENS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
