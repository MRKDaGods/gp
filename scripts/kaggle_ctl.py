#!/usr/bin/env python
"""Thin Kaggle CLI wrapper that hot-selects an account token.

Reuses the canonical account->token map and KAGGLE_API_TOKEN env mechanism
from ``dump_kaggle_kernel_summaries.py`` so every call runs under the right
Kaggle identity without touching ~/.kaggle/kaggle.json.

Usage:
    python scripts/kaggle_ctl.py <account> <kaggle args...>

Examples:
    python scripts/kaggle_ctl.py gumfreddy kernels status gumfreddy/veri-canon-stream1-train
    python scripts/kaggle_ctl.py gumfreddy kernels push -p notebooks/kaggle/veri_canon_stream1_train
    python scripts/kaggle_ctl.py gumfreddy kernels output gumfreddy/veri-canon-fusion-eval -p outdir

Accounts: gumfreddy, mrkdagods, ali369, yahiaakhalafallah.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dump_kaggle_kernel_summaries import (  # noqa: E402
    ACCOUNT_TOKENS,
    account_env,
    resolve_token_file,
)


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    account = sys.argv[1]
    kaggle_args = sys.argv[2:]

    token_map = dict(ACCOUNT_TOKENS)
    if account not in token_map:
        print(f"Unknown account {account!r}; known: {list(token_map)}")
        return 2
    token_file = resolve_token_file(token_map[account])
    if not token_file:
        print(f"No token file found for {account}")
        return 2
    env = account_env(token_file)
    if env is None:
        print(f"Empty/missing token for {account}")
        return 2

    proc = subprocess.run(
        ["kaggle", *kaggle_args],
        env=env,
        text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
