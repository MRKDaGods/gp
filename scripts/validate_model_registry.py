"""Validate configs/model_registry.yaml for CI and local sanity checks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.services.model_registry import DEFAULT_REGISTRY_PATH, load_registry


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the MTMC model registry.")
    parser.add_argument(
        "registry",
        nargs="?",
        default=str(DEFAULT_REGISTRY_PATH),
        help="Path to a model registry YAML file.",
    )
    args = parser.parse_args()

    registry_path = Path(args.registry).resolve()
    try:
        registry = load_registry(registry_path)
    except Exception as exc:
        print(f"Model registry validation failed: {exc}", file=sys.stderr)
        return 1

    missing = sum(len(model.missing_checkpoints) for model in registry.models)
    print(
        f"Validated {len(registry.models)} model registry entries "
        f"({missing} checkpoint paths not present locally)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
