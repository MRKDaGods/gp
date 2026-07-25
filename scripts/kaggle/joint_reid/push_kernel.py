"""Push a joint_reid campaign kernel with the Kaggle token injected.

Kernel scripts in this directory that need in-kernel Kaggle auth (to create
output datasets) carry the placeholder __KGAT_INJECTED_AT_PUSH__ instead of a
real token, so nothing secret is ever committed. This helper copies the kernel
dir to a temp staging dir, substitutes the placeholder with the token from
~/.kaggle/kaggle.json, and pushes from there.

Windows note: PYTHONUTF8=1 for the Kaggle CLI (cp1252 crash).

Usage: python scripts/kaggle/joint_reid/push_kernel.py <kernel_dir_name>
e.g.:  python scripts/kaggle/joint_reid/push_kernel.py veriwild_prep
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

PLACEHOLDER = "__KGAT_INJECTED_AT_PUSH__"


def main() -> int:
    if len(sys.argv) not in (2, 3):
        print(__doc__, file=sys.stderr)
        return 2
    accelerator = sys.argv[2] if len(sys.argv) == 3 else None  # e.g. NvidiaTeslaT4
    kernel_dir = Path(__file__).parent / sys.argv[1]
    if not (kernel_dir / "kernel-metadata.json").exists():
        print(f"no kernel-metadata.json in {kernel_dir}", file=sys.stderr)
        return 2

    token = (Path.home() / ".kaggle" / "kaggle.json").read_text("utf-8").strip()
    if not token.startswith("KGAT_"):
        print("~/.kaggle/kaggle.json does not hold a raw KGAT token", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory() as staging:
        injected = False
        for src in kernel_dir.iterdir():
            if src.is_dir():
                continue
            text = src.read_text("utf-8")
            if PLACEHOLDER in text:
                text = text.replace(PLACEHOLDER, token)
                injected = True
            (Path(staging) / src.name).write_text(text, "utf-8", newline="\n")
        print(f"staged {kernel_dir.name} (token injected: {injected})")
        cmd = ["kaggle", "kernels", "push", "-p", staging]
        if accelerator:
            cmd += ["--accelerator", accelerator]
        result = subprocess.run(cmd, env={**os.environ, "PYTHONUTF8": "1"})
        return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
