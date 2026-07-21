"""`athar` CLI entry point.

Planned commands (wired as their subsystems land):
  athar run      — submit/execute a pipeline run from a profile + inputs
  athar models   — list/inspect/promote registry entries
  athar eval     — run a benchmark evaluation / parity gate
  athar migrate  — convert v1 run directories into v2 manifests
"""

from __future__ import annotations

import argparse

from athar import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="athar", description=__doc__)
    parser.add_argument("--version", action="version", version=f"athar {__version__}")
    parser.parse_args(argv)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
