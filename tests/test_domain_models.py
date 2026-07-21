"""Tests for profiles, events, case/target models, and the model lifecycle."""

from __future__ import annotations

import pytest

from athar.components.registry import ComponentRegistry, UnknownComponent
from athar.core.ids import TrackKey, new_target_id
from athar.core.types import BBox, EntityClass
from athar.pipeline.events import (
    RunFailed,
    StageProgress,
    dump_event,
    parse_event,
)
from athar.profiles.base import ClassBranch, ComponentSpec, RunProfile
from athar.search.case_models import (
    HypothesisEdge,
    HypothesisKind,
    HypothesisStatus,
    Target,
)
from athar.serving.registry import (
    CheckpointRef,
    EvalReportRef,
    ModelEntry,
    ModelStage,
    ModelTask,
)


def _branch(classes: list[EntityClass]) -> ClassBranch:
    spec = ComponentSpec(name="stub")
    return ClassBranch(
        entity_classes=classes, tracker=spec, embedders=[spec], score_terms=[spec], solver=spec
    )


class TestProfiles:
    def test_multiclass_profile_routes_by_class(self):
        profile = RunProfile(
            name="vehicle_person_v1",
            detector=ComponentSpec(name="yolo"),
            branches=[
                _branch([EntityClass.PERSON]),
                _branch([EntityClass.CAR, EntityClass.BUS, EntityClass.TRUCK]),
            ],
        )
        assert profile.branch_for(EntityClass.BUS) is profile.branches[1]
        assert profile.branch_for(EntityClass.PERSON) is profile.branches[0]
        with pytest.raises(KeyError):
            profile.branch_for(EntityClass.TUKTUK)

    def test_class_bound_to_two_branches_rejected(self):
        with pytest.raises(ValueError, match="multiple branches"):
            RunProfile(
                name="bad",
                detector=ComponentSpec(name="yolo"),
                branches=[_branch([EntityClass.CAR]), _branch([EntityClass.CAR])],
            )


class TestEvents:
    def test_roundtrip_discriminated_union(self):
        event = StageProgress(run_id="r1", stage="embed", camera_id="c1", done=3, total=10)
        parsed = parse_event(dump_event(event))
        assert isinstance(parsed, StageProgress)
        assert parsed.done == 3

    def test_failure_event_carries_error(self):
        parsed = parse_event(dump_event(RunFailed(run_id="r1", error="boom", stage="index")))
        assert isinstance(parsed, RunFailed)
        assert parsed.error == "boom"


class TestComponentRegistry:
    def test_register_create_and_unknown(self):
        reg = ComponentRegistry()

        @reg.register("detector", "stub")
        def make_stub(threshold: float = 0.5):
            return {"threshold": threshold}

        assert reg.create("detector", "stub", threshold=0.9) == {"threshold": 0.9}
        assert list(reg.names("detector")) == ["stub"]
        with pytest.raises(UnknownComponent, match="registered: stub"):
            reg.create("detector", "missing")

    def test_duplicate_registration_rejected(self):
        reg = ComponentRegistry()
        reg.register("solver", "cc")(lambda: None)
        with pytest.raises(ValueError, match="already registered"):
            reg.register("solver", "cc")(lambda: None)


class TestHypotheses:
    def _edge(self) -> HypothesisEdge:
        return HypothesisEdge(
            kind=HypothesisKind.APPEARANCE,
            track_key=TrackKey(run_id="r", camera_id="c", track_id=1),
            raw_score=0.83,
            calibrated_probability=0.71,
        )

    def test_decision_is_attributed_and_final(self):
        edge = self._edge()
        edge.decide(HypothesisStatus.CONFIRMED, operator="analyst_1")
        assert edge.decided_by == "analyst_1"
        assert edge.decided_at is not None
        with pytest.raises(ValueError, match="already decided"):
            edge.decide(HypothesisStatus.REJECTED, operator="analyst_2")

    def test_cannot_undecide(self):
        with pytest.raises(ValueError):
            self._edge().decide(HypothesisStatus.PROPOSED, operator="x")

    def test_target_holds_cross_entity_hypotheses(self):
        target = Target(target_id=new_target_id(), label="Suspect A")
        target.hypotheses.append(self._edge())
        boarding = HypothesisEdge(
            kind=HypothesisKind.BOARDING,
            track_key=TrackKey(run_id="r", camera_id="c", track_id=9),
            raw_score=0.6,
        )
        target.hypotheses.append(boarding)
        kinds = {h.kind for h in target.hypotheses}
        assert HypothesisKind.BOARDING in kinds


class TestModelLifecycle:
    def _entry(self) -> ModelEntry:
        return ModelEntry(
            model_id="reid_vehicle_test",
            task=ModelTask.REID_VEHICLE,
            architecture="transreid_vit_b16",
            checkpoint=CheckpointRef(sha256="ab" * 32, size_bytes=10, filename="w.pth"),
        )

    def test_promotion_requires_eval_report(self):
        entry = self._entry()
        with pytest.raises(ValueError, match="requires an evaluation report"):
            entry.promote(ModelStage.VALIDATED)

    def test_gated_promotion_path(self):
        entry = self._entry()
        report = EvalReportRef(run_id="run-x", benchmark="veri776", metrics={"mAP": 93.3})
        entry.promote(ModelStage.VALIDATED, report)
        entry.promote(ModelStage.PRODUCTION, report)
        assert entry.stage is ModelStage.PRODUCTION
        assert len(entry.eval_reports) == 2

    def test_no_stage_skipping(self):
        entry = self._entry()
        report = EvalReportRef(run_id="r", benchmark="b", metrics={})
        with pytest.raises(ValueError, match="illegal promotion"):
            entry.promote(ModelStage.PRODUCTION, report)

    def test_retire_always_allowed(self):
        entry = self._entry()
        entry.promote(ModelStage.RETIRED)
        assert entry.stage is ModelStage.RETIRED


class TestBBox:
    def test_degenerate_rejected(self):
        with pytest.raises(ValueError):
            BBox(x1=10, y1=0, x2=5, y2=5)

    def test_geometry(self):
        box = BBox(x1=0, y1=0, x2=4, y2=3)
        assert box.area == 12
