"""Verify local MTMC Tracker checkpoints and datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.download_assets import ASSETS, markdown_table, verify_assets  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify downloaded MTMC Tracker assets")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail when optional datasets are missing. By default only required model files fail.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    results = verify_assets(ASSETS)
    print(markdown_table(results))

    failures = [result for result in results if not result.ok]
    if not args.strict:
        failures = [
            result
            for result in failures
            if not result.asset.optional and not result.asset.is_manual
        ]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())