"""Layered configuration resolution with per-key provenance.

v1 failure mode: "we never knew which config entry or file was used in a run"
— values scattered across default.yaml, dataset yaml, registry model_overrides,
and CLI dotlists, merged invisibly. v2 rule: a run stores its FULLY RESOLVED
config (every key, every value), a provenance map naming the layer that set
each key, and a content hash. The app can show "exactly what this run used and
why" for any run, forever — which is also what court-grade reproducibility
requires.
"""

from __future__ import annotations

import enum
import hashlib
import json
from typing import Any, Mapping

from pydantic import BaseModel, Field


class ConfigLayer(str, enum.Enum):
    """Config layers, listed lowest → highest precedence."""

    PROFILE_DEFAULT = "profile_default"   # shipped tuned defaults per profile
    DEPLOYMENT = "deployment"             # site-level settings (installer)
    CASE = "case"                         # case-level settings (investigator)
    RUN_OVERRIDE = "run_override"         # explicit per-run overrides

    @classmethod
    def precedence(cls) -> list["ConfigLayer"]:
        return [cls.PROFILE_DEFAULT, cls.DEPLOYMENT, cls.CASE, cls.RUN_OVERRIDE]


def _flatten(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, Mapping):
        for k, v in value.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            _flatten(key, v, out)
    else:
        out[prefix] = value


class ResolvedConfig(BaseModel):
    """The frozen effective configuration of one run."""

    values: dict[str, Any] = Field(description="Flat dot-keyed key → value")
    provenance: dict[str, ConfigLayer] = Field(
        description="key → the layer that supplied its final value"
    )
    config_hash: str = Field(description="sha256 over canonical values JSON")

    @classmethod
    def resolve(cls, layers: list[tuple[ConfigLayer, Mapping[str, Any]]]) -> "ResolvedConfig":
        """Merge layers (later precedence wins) and record provenance.

        ``layers`` may arrive in any order; they are applied in canonical
        precedence order, so passing the same layer kind twice is an error.
        """
        seen: set[ConfigLayer] = set()
        for layer, _ in layers:
            if layer in seen:
                raise ValueError(f"duplicate config layer: {layer.value}")
            seen.add(layer)

        ordered = sorted(layers, key=lambda lv: ConfigLayer.precedence().index(lv[0]))
        values: dict[str, Any] = {}
        provenance: dict[str, ConfigLayer] = {}
        for layer, mapping in ordered:
            flat: dict[str, Any] = {}
            _flatten("", mapping, flat)
            for key, value in flat.items():
                values[key] = value
                provenance[key] = layer

        canonical = json.dumps(values, sort_keys=True, default=str)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return cls(values=values, provenance=provenance, config_hash=digest)

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)
