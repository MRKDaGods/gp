"""Config authoring: YAML layer files + dotted overrides → ResolvedConfig.

YAML is the *authoring* format only (D5/D6): humans edit small per-layer
files; submission resolves them into the frozen, provenance-carrying
``ResolvedConfig`` stored in the run manifest. Unknown layers, malformed
files, and non-mapping roots fail at submit time — never mid-run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from athar.contracts.config import ConfigLayer, ResolvedConfig


class ConfigAuthoringError(ValueError):
    pass


def load_layer_file(path: Path | str) -> dict[str, Any]:
    """Load one YAML layer file; the root must be a mapping."""
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ConfigAuthoringError(f"config layer file not found: {path}") from None
    except yaml.YAMLError as exc:
        raise ConfigAuthoringError(f"invalid YAML in {path}: {exc}") from exc
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ConfigAuthoringError(f"config root must be a mapping, got {type(raw).__name__}: {path}")
    return dict(raw)


def parse_dotted_overrides(pairs: list[str]) -> dict[str, Any]:
    """Parse CLI ``key.path=value`` overrides into a nested mapping.

    Values are parsed as YAML scalars (so ``conf=0.4`` is a float,
    ``enabled=true`` a bool, ``size=x`` a string).
    """
    nested: dict[str, Any] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            raise ConfigAuthoringError(f"override must be key.path=value, got {pair!r}")
        parsed = yaml.safe_load(value) if value != "" else ""
        cursor = nested
        parts = key.split(".")
        for part in parts[:-1]:
            existing = cursor.setdefault(part, {})
            if not isinstance(existing, dict):
                raise ConfigAuthoringError(
                    f"override {key!r} conflicts with scalar at {part!r}"
                )
            cursor = existing
        cursor[parts[-1]] = parsed
    return nested


def resolve_from_files(
    profile_defaults: Path | str,
    deployment: Path | str | None = None,
    case: Path | str | None = None,
    overrides: list[str] | None = None,
) -> ResolvedConfig:
    """The standard submission path: layer files + CLI overrides → ResolvedConfig."""
    layers: list[tuple[ConfigLayer, Mapping[str, Any]]] = [
        (ConfigLayer.PROFILE_DEFAULT, load_layer_file(profile_defaults))
    ]
    if deployment is not None:
        layers.append((ConfigLayer.DEPLOYMENT, load_layer_file(deployment)))
    if case is not None:
        layers.append((ConfigLayer.CASE, load_layer_file(case)))
    if overrides:
        layers.append((ConfigLayer.RUN_OVERRIDE, parse_dotted_overrides(overrides)))
    return ResolvedConfig.resolve(layers)
