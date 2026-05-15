"""Generate a markdown snapshot from configs/model_registry.yaml.

Phase 1 writes docs/models.generated.md only; docs/models.md remains untouched.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.services.model_registry import DEFAULT_REGISTRY_PATH, load_registry

DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "docs" / "models.generated.md"


def _format_metrics(metrics) -> str:
    parts = []
    for metric in metrics:
        badge = "verified" if metric.verified else "unverified"
        parts.append(f"{metric.name}={metric.value:.5g} ({badge})")
    return "<br>".join(parts)


def generate_markdown(registry_path: Path) -> str:
    registry = load_registry(registry_path)
    lines = [
        "# Models & Checkpoints - Generated Snapshot",
        "",
        "> Generated from `configs/model_registry.yaml`. Do not edit this Phase 1 snapshot by hand.",
        "",
        "| ID | Name | Task | Dataset | Status | Runnable locally | Key metrics |",
        "|---|---|---|---|---|---:|---|",
    ]
    for model in registry.models:
        lines.append(
            "| "
            + " | ".join(
                [
                    model.id,
                    model.name,
                    model.task_type,
                    model.dataset,
                    model.status,
                    "yes" if model.runnable_locally else "no",
                    _format_metrics(model.metrics),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate docs/models.generated.md from registry YAML.")
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    args = parser.parse_args()

    registry_path = Path(args.registry).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(generate_markdown(registry_path), encoding="utf-8")
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
