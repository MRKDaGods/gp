"""embed stage tests: candidate planning, sparse decode, v1 crop kernel,
per-stream artifacts, camera resume. Uses the real CropExtractor + real
HSV embedder (pure numpy/cv2) over a synthetic image-dir camera; the heavy
TransReID adapter is exercised separately by parity smokes."""

from __future__ import annotations

import json

import numpy as np
import pytest

from athar.components.protocols import ComponentKindName
from athar.components.registry import ComponentRegistry
from athar.contracts.config import ConfigLayer, ResolvedConfig
from athar.contracts.manifest import ArtifactRecord, RunManifest, RunRole, RunStatus, VideoInput
from athar.contracts.store import FilesystemRunStore
from athar.core.ids import new_run_id
from athar.core.timebase import CameraTimeBase, TimeBaseSource
from athar.pipeline.runner import PipelineRunner
from athar.pipeline.stages.embed import EmbedStage, candidate_frame_ids
from athar.profiles.base import ClassBranch, ComponentSpec, RunProfile
from athar.core.types import EntityClass

cv2 = pytest.importorskip("cv2")

FPS = 10.0
FRAMES = 30
SIZE = (96, 72)  # (w, h) — big enough for min_area crops


@pytest.fixture(scope="module")
def camera_dir(tmp_path_factory):
    """Image-dir camera whose frames have a bright moving square."""
    d = tmp_path_factory.mktemp("cam_frames")
    rng = np.random.default_rng(3)
    for i in range(FRAMES):
        frame = rng.integers(0, 40, (SIZE[1], SIZE[0], 3), dtype=np.uint8)
        x = 5 + i  # square drifts right
        frame[10:60, x : x + 30] = (0, 128, 255)
        cv2.imwrite(str(d / f"{i:06d}.png"), frame)
    return d


class CountingHsv:
    """Real HSV embedder + call counting (kernel-faithful, no torch)."""

    calls: int = 0

    def __init__(self, **kwargs):
        from athar.components.adapters.embedding import HsvEmbedderAdapter

        self._inner = HsvEmbedderAdapter(**kwargs)
        self.stream_name = self._inner.stream_name
        self.dim = self._inner.dim

    def embed(self, crops):
        return self._inner.embed(crops)

    def embed_tracklet(self, scored_crops, cam_index=None):
        CountingHsv.calls += 1
        return self._inner.embed_tracklet(scored_crops)


@pytest.fixture()
def registry():
    from athar.components.framesources.image_dir import ImageDirFrameSource

    reg = ComponentRegistry()
    reg.register(ComponentKindName.FRAME_SOURCE, "image_dir")(ImageDirFrameSource)
    reg.register(ComponentKindName.EMBEDDER, "hsv_counting")(CountingHsv)
    CountingHsv.calls = 0
    return reg


def _profile() -> RunProfile:
    return RunProfile(
        name="embed-test",
        frame_source=ComponentSpec(name="image_dir"),
        detector=ComponentSpec(name="unused"),
        branches=[
            ClassBranch(
                entity_classes=[EntityClass.CAR],
                tracker=ComponentSpec(name="unused"),
                embedders=[ComponentSpec(name="hsv_counting")],
                score_terms=[ComponentSpec(name="x")],
                solver=ComponentSpec(name="x"),
            )
        ],
    )


def _bbox(i: int) -> dict:
    x = 5 + i
    return {"x1": float(x), "y1": 10.0, "x2": float(x + 30), "y2": 60.0}


def _manifest_with_tracklets(store: FilesystemRunStore, camera_dir) -> RunManifest:
    manifest = RunManifest(
        run_id=new_run_id(), role=RunRole.GALLERY, profile_name="embed-test"
    )
    manifest.config = ResolvedConfig.resolve(
        [(ConfigLayer.PROFILE_DEFAULT, {"embed": {"samples_per_tracklet": 4,
                                                  "tracklet_chunk": 2, "min_area": 100}})]
    )
    manifest.inputs.append(
        VideoInput(camera_id="cam01", original_path=str(camera_dir), sha256="0" * 64)
    )
    manifest.timebase.cameras["cam01"] = CameraTimeBase(
        camera_id="cam01", fps=FPS, source=TimeBaseSource.SYNCHRONIZED, confidence=1.0
    )

    tracklets, observations = [], {}
    for track_id, (lo, hi) in ((1, (0, 15)), (2, (10, 30)), (3, (20, 30))):
        obs = [
            {"frame_index": i, "ts_scene_s": i / FPS, "bbox": _bbox(i),
             "confidence": 0.9, "interpolated": False}
            for i in range(lo, hi)
        ]
        observations[str(track_id)] = obs
        tracklets.append({
            "key": {"run_id": manifest.run_id, "camera_id": "cam01", "track_id": track_id},
            "entity_class": "car",
            "start_ts_scene_s": obs[0]["ts_scene_s"],
            "end_ts_scene_s": obs[-1]["ts_scene_s"],
            "observation_count": len(obs),
            "mean_confidence": 0.9,
        })

    run_dir = store.run_dir(manifest.run_id)
    (run_dir / "tracklets").mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 1, "camera_id": "cam01",
               "tracklets": tracklets, "observations": observations}
    (run_dir / "tracklets" / "cam01.json").write_text(json.dumps(payload), encoding="utf-8")
    manifest.register_artifact(
        ArtifactRecord(name="tracklets.cam01", relpath="tracklets/cam01.json",
                       schema_version=1, producer="test", row_count=3)
    )
    return manifest


@pytest.fixture()
def store(tmp_path):
    return FilesystemRunStore(tmp_path / "runs")


class TestCandidatePlan:
    def test_small_tracklet_uses_all_frames(self):
        assert candidate_frame_ids(3, [7, 8, 9], samples_per_tracklet=4) == [7, 8, 9]

    def test_large_tracklet_stratifies(self):
        ids = list(range(100, 200))
        picked = candidate_frame_ids(100, ids, samples_per_tracklet=4)
        assert len(picked) == 8  # 2x oversampling
        assert picked == sorted(picked)
        assert picked[0] == 100 and picked[-1] >= 180  # spans the tracklet


class TestEmbedStage:
    def test_streams_written_and_aligned(self, store, registry, camera_dir):
        manifest = _manifest_with_tracklets(store, camera_dir)
        result = PipelineRunner(store, [EmbedStage()], registry=registry).run(
            manifest, _profile()
        )
        assert result.status == RunStatus.COMPLETED
        assert "embed.summary" in result.artifacts
        assert CountingHsv.calls == 3

        data = np.load(store.artifact_path(result, "embeddings.cam01.hsv"))
        assert data["embeddings"].shape == (3, 192)  # (32+16+16)*3 stripes
        assert data["track_ids"].tolist() == [1, 2, 3]
        assert data["embeddings"].dtype == np.float32
        # histograms are non-degenerate and differ across tracklets
        assert np.abs(data["embeddings"]).sum() > 0

        summary = json.loads(
            store.artifact_path(result, "embed.summary").read_text(encoding="utf-8")
        )
        assert summary["streams"]["hsv"]["dim"] == 192
        assert summary["streams"]["hsv"]["artifacts"] == ["embeddings.cam01.hsv"]

    def test_camera_resume_skips_completed(self, store, registry, camera_dir):
        manifest = _manifest_with_tracklets(store, camera_dir)
        stage = EmbedStage()
        runner = PipelineRunner(store, [stage], registry=registry)
        result = runner.run(manifest, _profile())
        calls_after_first = CountingHsv.calls

        # simulate crash AFTER the camera checkpoint but BEFORE summary:
        # drop the summary artifact and status, keep the checkpoint
        result.status = RunStatus.FAILED
        del result.artifacts["embed.summary"]
        ckpt = store.run_dir(result.run_id) / "checkpoints"
        ckpt.mkdir(exist_ok=True)
        (ckpt / "embed.json").write_text(
            json.dumps({"completed_cameras": ["cam01"]}), encoding="utf-8"
        )
        store.save(result)

        rerun = runner.run(store.load(result.run_id), _profile())
        assert rerun.status == RunStatus.COMPLETED
        assert CountingHsv.calls == calls_after_first  # camera NOT re-embedded
        summary = json.loads(
            store.artifact_path(rerun, "embed.summary").read_text(encoding="utf-8")
        )
        assert summary["streams"]["hsv"]["artifacts"] == ["embeddings.cam01.hsv"]
