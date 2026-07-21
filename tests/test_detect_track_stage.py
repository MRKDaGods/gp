"""detect_track stage tests with fake components (no ML deps).

Covers: branch class filtering, per-camera artifacts + payload shape,
branch id namespacing, camera-level checkpoint resume, is_complete.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from athar.components.protocols import ComponentKindName
from athar.components.registry import ComponentRegistry
from athar.contracts.config import ConfigLayer, ResolvedConfig
from athar.contracts.manifest import RunManifest, RunRole, RunStatus
from athar.contracts.store import FilesystemRunStore
from athar.core.ids import TrackKey, new_run_id
from athar.core.timebase import CameraTimeBase, TimeBaseSource
from athar.core.types import (
    BBox,
    Detection,
    EntityClass,
    TrackObservation,
    Tracklet,
)
from athar.pipeline.runner import PipelineRunner
from athar.pipeline.stages.detect_track import BRANCH_ID_OFFSET, DetectTrackStage
from athar.profiles.base import ClassBranch, ComponentSpec, RunProfile

FPS = 10.0
FRAMES = 8


class FakeBatch:
    def __init__(self, camera_id: str, frame_indices: tuple[int, ...]):
        self.camera_id = camera_id
        self.frame_indices = frame_indices

    def images(self) -> np.ndarray:
        return np.zeros((len(self.frame_indices), 8, 8, 3), dtype=np.uint8)

    def __len__(self) -> int:
        return len(self.frame_indices)


class FakeFrameSource:
    def __init__(self, camera_id: str, path, **_):
        self.camera_id = camera_id
        self.frame_count = FRAMES

    def batches(self, batch_size: int):
        for start in range(0, FRAMES, batch_size):
            indices = tuple(range(start, min(start + batch_size, FRAMES)))
            yield FakeBatch(self.camera_id, indices)


class FakeDetector:
    """Every frame: one person + one car."""

    def __init__(self, scene_clock, **_):
        self._clock = scene_clock

    def detect(self, batch) -> list[Detection]:
        tb = self._clock.require(batch.camera_id)
        out = []
        for idx in batch.frame_indices:
            for entity in (EntityClass.PERSON, EntityClass.CAR):
                out.append(
                    Detection(
                        camera_id=batch.camera_id,
                        frame_index=idx,
                        ts_scene_s=tb.to_scene(idx),
                        bbox=BBox(x1=0, y1=0, x2=10, y2=10),
                        entity_class=entity,
                        confidence=0.9,
                    )
                )
        return out


class FakeTracker:
    """One track per entity class seen; records what it was fed."""

    fed_classes: list[set[EntityClass]] = []  # class-level probe, reset per test

    def __init__(self, run_id: str, camera_id: str, scene_clock, **_):
        self.run_id = run_id
        self.camera_id = camera_id
        self._seen: dict[EntityClass, list[Detection]] = {}
        FakeTracker.fed_classes.append(set())
        self._probe = FakeTracker.fed_classes[-1]

    def update(self, detections: list[Detection], batch) -> None:
        for d in detections:
            self._probe.add(d.entity_class)
            self._seen.setdefault(d.entity_class, []).append(d)

    def drain(self):
        tracklets, observations = [], {}
        for i, (entity, dets) in enumerate(sorted(self._seen.items()), start=1):
            obs = [
                TrackObservation(
                    frame_index=d.frame_index,
                    ts_scene_s=d.ts_scene_s,
                    bbox=d.bbox,
                    confidence=d.confidence,
                )
                for d in dets
            ]
            tracklets.append(
                Tracklet(
                    key=TrackKey(
                        run_id=self.run_id, camera_id=self.camera_id, track_id=i
                    ),
                    entity_class=entity,
                    start_ts_scene_s=obs[0].ts_scene_s,
                    end_ts_scene_s=obs[-1].ts_scene_s,
                    observation_count=len(obs),
                    mean_confidence=0.9,
                )
            )
            observations[i] = obs
        self._seen.clear()
        return tracklets, observations

    def flush(self):
        return self.drain()[0]


@pytest.fixture()
def registry():
    reg = ComponentRegistry()
    reg.register(ComponentKindName.FRAME_SOURCE, "fake")(FakeFrameSource)
    reg.register(ComponentKindName.DETECTOR, "fake")(FakeDetector)
    reg.register(ComponentKindName.TRACKER, "fake")(FakeTracker)
    FakeTracker.fed_classes = []
    return reg


def _profile() -> RunProfile:
    return RunProfile(
        name="fake-profile",
        frame_source=ComponentSpec(name="fake"),
        detector=ComponentSpec(name="fake"),
        branches=[
            ClassBranch(
                entity_classes=[EntityClass.PERSON],
                tracker=ComponentSpec(name="fake"),
                embedders=[ComponentSpec(name="x")],
                score_terms=[ComponentSpec(name="x")],
                solver=ComponentSpec(name="x"),
            ),
            ClassBranch(
                entity_classes=[EntityClass.CAR, EntityClass.BUS, EntityClass.TRUCK],
                tracker=ComponentSpec(name="fake"),
                embedders=[ComponentSpec(name="x")],
                score_terms=[ComponentSpec(name="x")],
                solver=ComponentSpec(name="x"),
            ),
        ],
    )


def _manifest(cameras: list[str]) -> RunManifest:
    manifest = RunManifest(
        run_id=new_run_id(), role=RunRole.GALLERY, profile_name="fake-profile"
    )
    manifest.config = ResolvedConfig.resolve(
        [(ConfigLayer.PROFILE_DEFAULT, {"detect_track": {"batch_size": 3}})]
    )
    for cam in cameras:
        from athar.contracts.manifest import VideoInput

        manifest.inputs.append(
            VideoInput(camera_id=cam, original_path=f"unused/{cam}.mp4", sha256="0" * 64)
        )
        manifest.timebase.cameras[cam] = CameraTimeBase(
            camera_id=cam, fps=FPS, source=TimeBaseSource.SYNCHRONIZED, confidence=1.0
        )
    return manifest


def _run(store, registry, manifest):
    runner = PipelineRunner(store, [DetectTrackStage()], registry=registry)
    return runner.run(manifest, _profile())


@pytest.fixture()
def store(tmp_path):
    return FilesystemRunStore(tmp_path / "runs")


class TestDetectTrack:
    def test_artifacts_and_branch_namespacing(self, store, registry):
        manifest = _manifest(["cam01", "cam02"])
        result = _run(store, registry, manifest)

        assert result.status == RunStatus.COMPLETED
        assert set(result.artifacts) == {"tracklets.cam01", "tracklets.cam02"}

        payload = json.loads(
            store.artifact_path(result, "tracklets.cam01").read_text(encoding="utf-8")
        )
        assert payload["camera_id"] == "cam01"
        # person branch: 1 track (id 1); vehicle branch: 1 CAR track (id 1 + offset)
        ids = sorted(t["key"]["track_id"] for t in payload["tracklets"])
        assert ids == [1, BRANCH_ID_OFFSET + 1]
        classes = {t["entity_class"] for t in payload["tracklets"]}
        assert classes == {"person", "car"}
        # observations keyed by offset ids, FRAMES observations each
        assert set(payload["observations"]) == {"1", str(BRANCH_ID_OFFSET + 1)}
        assert len(payload["observations"]["1"]) == FRAMES
        # scene time honored the timebase
        first = payload["observations"]["1"][0]
        assert first["ts_scene_s"] == pytest.approx(first["frame_index"] / FPS)

    def test_branch_class_filtering(self, store, registry):
        _run(store, registry, _manifest(["cam01"]))
        person_feed, vehicle_feed = FakeTracker.fed_classes
        assert person_feed == {EntityClass.PERSON}
        assert vehicle_feed == {EntityClass.CAR}

    def test_camera_checkpoint_resume(self, store, registry):
        crashed = {"done": False}

        class CrashingSecondCameraSource(FakeFrameSource):
            def batches(self, batch_size):
                if self.camera_id == "cam02" and not crashed["done"]:
                    crashed["done"] = True
                    raise RuntimeError("decoder died")
                yield from super().batches(batch_size)

        reg = ComponentRegistry()
        reg.register(ComponentKindName.FRAME_SOURCE, "fake")(CrashingSecondCameraSource)
        reg.register(ComponentKindName.DETECTOR, "fake")(FakeDetector)
        reg.register(ComponentKindName.TRACKER, "fake")(FakeTracker)
        FakeTracker.fed_classes = []

        manifest = _manifest(["cam01", "cam02"])
        with pytest.raises(RuntimeError, match="decoder died"):
            _run(store, reg, manifest)

        persisted = store.load(manifest.run_id)
        assert persisted.status == RunStatus.FAILED
        assert "tracklets.cam01" in persisted.artifacts  # cam01 survived

        # resume completes cam02 without redoing cam01 (checkpoint carries it)
        trackers_before = len(FakeTracker.fed_classes)
        result = _run(store, reg, persisted)
        assert result.status == RunStatus.COMPLETED
        assert set(result.artifacts) == {"tracklets.cam01", "tracklets.cam02"}
        # only cam02's two branch trackers were created on resume
        assert len(FakeTracker.fed_classes) == trackers_before + 2

    def test_is_complete_skips_everything(self, store, registry):
        manifest = _manifest(["cam01"])
        result = _run(store, registry, manifest)
        # wipe status back and re-run: stage must skip, artifacts unchanged
        result.status = RunStatus.CREATED
        rerun = _run(store, registry, result)
        assert rerun.status == RunStatus.COMPLETED
        events = (store.run_dir(result.run_id) / "events.jsonl").read_text()
        assert "stage_skipped" in events

    def test_no_inputs_rejected(self, store, registry):
        manifest = _manifest([])
        with pytest.raises(ValueError, match="no ingested videos"):
            _run(store, registry, manifest)
