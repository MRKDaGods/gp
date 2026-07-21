"""Multi-class run profiles.

A profile binds ONE detection pass to N class branches. Each branch names the
tracker, embedder streams, score terms, and solver for its entity classes.
Person + vehicle tracking in the same run is the default shape (D3) — the v1
notion of "a vehicle run" vs "a person run" survives only as profiles with a
single branch. The WILDTRACK MVDeTr path is a profile whose detection slot is
a MultiViewDetector — same graph, no fork.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, model_validator

from athar.core.types import EntityClass


class ComponentSpec(BaseModel):
    """A (registry name, config) pair for one component slot."""

    name: str
    config: dict = Field(default_factory=dict)


class ClassBranch(BaseModel):
    """The per-class pipeline after detection."""

    entity_classes: list[EntityClass] = Field(min_length=1)
    tracker: ComponentSpec
    embedders: list[ComponentSpec] = Field(
        min_length=1, description="Embedding streams; first is primary"
    )
    refiners: list[ComponentSpec] = Field(default_factory=list)
    score_terms: list[ComponentSpec] = Field(min_length=1)
    solver: ComponentSpec


class RunProfile(BaseModel):
    """Complete component binding for a run."""

    name: str
    detector: ComponentSpec
    multi_view: bool = Field(
        default=False, description="True when detector slot is a MultiViewDetector"
    )
    branches: list[ClassBranch] = Field(min_length=1)
    spatial_model: Optional[ComponentSpec] = None
    ir_classifier: Optional[ComponentSpec] = None
    interaction_detectors: list[ComponentSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def _no_overlapping_classes(self) -> "RunProfile":
        seen: set[EntityClass] = set()
        for branch in self.branches:
            overlap = seen.intersection(branch.entity_classes)
            if overlap:
                raise ValueError(
                    f"entity classes bound to multiple branches: {sorted(c.value for c in overlap)}"
                )
            seen.update(branch.entity_classes)
        return self

    def branch_for(self, entity_class: EntityClass) -> ClassBranch:
        for branch in self.branches:
            if entity_class in branch.entity_classes:
                return branch
        raise KeyError(f"profile {self.name!r} has no branch for {entity_class.value}")
