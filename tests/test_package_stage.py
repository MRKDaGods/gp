"""package stage tests: thumbnails + report inputs from a synthetic run."""

from __future__ import annotations

import json

import numpy as np
import pytest

import athar.components.framesources  # noqa: F401 — registers image_dir
from athar.contracts.config import ConfigLayer, ResolvedConfig
from athar.contracts.manifest import ArtifactRecord, RunManifest, RunRole, RunStatus, VideoInput
from athar.contracts.store import FilesystemRunStore
from athar.core.ids import new_run_id
from athar.core.types import EntityClass
from athar.pipeline.runner import PipelineRunner
from athar.pipeline.stages.package import PackageStage, best_observation
from athar.profiles.base import ClassBranch, ComponentSpec, RunProfile

cv2 = pytest.importorskip("cv2")

W, H = 96, 64


def _profile() -> RunProfile:
    spec = ComponentSpec(name="x")
    return RunProfile(
        name="pkg-test",
        frame_source=ComponentSpec(name="image_dir"),
        detector=spec,
        branches=[
            ClassBranch(
                entity_classes=[EntityClass.PERSON],
                tracker=spec, embedders=[spec], score_terms=[spec], solver=spec,
            )
        ],
    )


def _obs(frame: int, x1: float, y1: float, x2: float, y2: float, conf: float):
    return {
        "frame_index": frame,
        "ts_scene_s": frame / 2.0,
        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "confidence": conf,
        "interpolated": False,
    }


@pytest.fixture()
def store(tmp_path):
    return FilesystemRunStore(tmp_path / "runs")


@pytest.fixture()
def gallery(tmp_path, store):
    """One image-dir camera, 4 frames each filled with its index value;
    two person tracklets + a two-member trajectory payload."""
    frames_dir = tmp_path / "cam01_frames"
    frames_dir.mkdir()
    for i in range(4):
        img = np.full((H, W, 3), i * 40 + 20, dtype=np.uint8)
        cv2.imwrite(str(frames_dir / f"{i:04d}.png"), img)

    manifest = RunManifest(
        run_id=new_run_id(), role=RunRole.GALLERY, profile_name="pkg-test"
    )
    manifest.config = ResolvedConfig.resolve(
        [(ConfigLayer.PROFILE_DEFAULT, {"package": {"thumb_size": 32}})]
    )
    manifest.inputs.append(
        VideoInput(camera_id="cam01", original_path=str(frames_dir), sha256="0" * 64)
    )
    run_dir = store.run_dir(manifest.run_id)
    (run_dir / "tracklets").mkdir(parents=True)

    payload = {
        "schema_version": 1,
        "camera_id": "cam01",
        "tracklets": [
            {
                "key": {"run_id": manifest.run_id, "camera_id": "cam01", "track_id": t},
                "entity_class": "person",
                "start_ts_scene_s": 0.0, "end_ts_scene_s": 1.5,
                "observation_count": 2, "mean_confidence": 0.9,
            }
            for t in (1, 2)
        ],
        "observations": {
            # track 1: best obs (conf 0.9) is on frame 2
            "1": [_obs(0, 4, 4, 40, 30, 0.5), _obs(2, 8, 8, 44, 34, 0.9)],
            # track 2: best obs on frame 3
            "2": [_obs(1, 50, 20, 90, 60, 0.4), _obs(3, 52, 22, 92, 62, 0.8)],
        },
    }
    (run_dir / "tracklets" / "cam01.json").write_text(json.dumps(payload), "utf-8")
    manifest.register_artifact(
        ArtifactRecord(name="tracklets.cam01", relpath="tracklets/cam01.json",
                       schema_version=1, producer="test", row_count=2)
    )

    trajectories = {
        "schema_version": 1,
        "trajectories": [
            {
                "global_id": 0,
                "entity_class": "person",
                "confidence": 0.8,
                "evidence": {"appearance": 0.7},
                "members": [
                    {"run_id": manifest.run_id, "camera_id": "cam01", "track_id": 1},
                    {"run_id": manifest.run_id, "camera_id": "cam01", "track_id": 2},
                ],
            }
        ],
    }
    (run_dir / "trajectories.json").write_text(json.dumps(trajectories), "utf-8")
    manifest.register_artifact(
        ArtifactRecord(name="associate.trajectories", relpath="trajectories.json",
                       schema_version=1, producer="test", row_count=1)
    )
    return manifest


def _run(store, manifest):
    return PipelineRunner(store, [PackageStage()]).run(manifest, _profile())


class TestBestObservation:
    def test_confidence_wins_middle_breaks_ties(self):
        obs = [_obs(0, 0, 0, 1, 1, 0.5), _obs(1, 0, 0, 1, 1, 0.5), _obs(2, 0, 0, 1, 1, 0.5)]
        assert best_observation(obs)["frame_index"] == 1
        obs[2]["confidence"] = 0.9
        assert best_observation(obs)["frame_index"] == 2


class TestPackage:
    def test_thumbnails_and_report(self, store, gallery):
        result = _run(store, gallery)
        assert result.status == RunStatus.COMPLETED

        run_dir = store.run_dir(result.run_id)
        index = json.loads((run_dir / "thumbs" / "index.json").read_text("utf-8"))
        assert set(index["thumbnails"]["cam01"]) == {"1", "2"}

        # thumb 1 comes from frame 2 (fill value 2*40+20=100), scaled to 32 long side
        thumb = cv2.imread(str(run_dir / index["thumbnails"]["cam01"]["1"]))
        assert thumb is not None and max(thumb.shape[:2]) <= 32
        assert abs(int(thumb.mean()) - 100) <= 2

        report = json.loads(
            store.artifact_path(result, "package.report").read_text("utf-8")
        )
        assert report["run"]["config_hash"] == gallery.config.config_hash
        assert report["evidence"][0]["sha256"] == "0" * 64
        (identity,) = report["identities"]
        assert identity["global_id"] == 0 and not identity["cross_camera"]
        by_track = {m["track_id"]: m for m in identity["members"]}
        assert by_track[1]["thumbnail"] == "thumbs/cam01/1.jpg"
        assert by_track[1]["end_ts_scene_s"] == 1.5
        assert by_track[2]["clip"] is None

    def test_clips_all_from_video_evidence(self, store, tmp_path):
        av = pytest.importorskip("av")
        from fractions import Fraction

        from athar.core.timebase import CameraTimeBase

        video = tmp_path / "camv.mp4"
        with av.open(str(video), "w") as container:
            stream = container.add_stream("libx264", rate=Fraction(10, 1))
            stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
            for i in range(40):
                img = np.full((48, 64, 3), (i * 6) % 255, dtype=np.uint8)
                frame = av.VideoFrame.from_ndarray(img, format="rgb24")
                for packet in stream.encode(frame.reformat(format="yuv420p")):
                    container.mux(packet)
            for packet in stream.encode(None):
                container.mux(packet)

        manifest = RunManifest(
            run_id=new_run_id(), role=RunRole.GALLERY, profile_name="pkg-test"
        )
        manifest.config = ResolvedConfig.resolve(
            [(ConfigLayer.PROFILE_DEFAULT,
              {"package": {"clips": "all", "clip_pad_s": 0.5}})]
        )
        manifest.inputs.append(
            VideoInput(camera_id="camv", original_path=str(video), sha256="0" * 64)
        )
        manifest.timebase.cameras["camv"] = CameraTimeBase(camera_id="camv", fps=10.0)
        run_dir = store.run_dir(manifest.run_id)
        (run_dir / "tracklets").mkdir(parents=True)
        payload = {
            "schema_version": 1, "camera_id": "camv",
            "tracklets": [{
                "key": {"run_id": manifest.run_id, "camera_id": "camv", "track_id": 1},
                "entity_class": "person",
                "start_ts_scene_s": 1.0, "end_ts_scene_s": 2.0,
                "observation_count": 1, "mean_confidence": 0.9,
            }],
            "observations": {"1": [_obs(12, 4, 4, 40, 30, 0.9)]},
        }
        (run_dir / "tracklets" / "camv.json").write_text(json.dumps(payload), "utf-8")
        manifest.register_artifact(
            ArtifactRecord(name="tracklets.camv", relpath="tracklets/camv.json",
                           schema_version=1, producer="test", row_count=1)
        )
        trajectories = {
            "schema_version": 1,
            "trajectories": [{
                "global_id": 0, "entity_class": "person", "confidence": 0.9,
                "evidence": {},
                "members": [
                    {"run_id": manifest.run_id, "camera_id": "camv", "track_id": 1}
                ],
            }],
        }
        (run_dir / "trajectories.json").write_text(json.dumps(trajectories), "utf-8")
        manifest.register_artifact(
            ArtifactRecord(name="associate.trajectories", relpath="trajectories.json",
                           schema_version=1, producer="test", row_count=1)
        )

        # thumbnails decode through the profile frame_source — use the real
        # video source so both thumbs and clips read the same mp4
        video_profile = _profile().model_copy(
            update={"frame_source": ComponentSpec(name="video")}
        )
        result = PipelineRunner(store, [PackageStage()]).run(manifest, video_profile)
        assert result.status == RunStatus.COMPLETED
        report = json.loads(
            store.artifact_path(result, "package.report").read_text("utf-8")
        )
        (member,) = report["identities"][0]["members"]
        assert member["clip"] is not None and member["clip"].startswith("clips/")
        clip_path = store.run_dir(result.run_id) / member["clip"]
        assert clip_path.is_file() and clip_path.stat().st_size > 0
        import hashlib

        assert member["clip_sha256"] == hashlib.sha256(
            clip_path.read_bytes()
        ).hexdigest()
        # span [1.0, 2.0] padded 0.5 -> media [0.5, 2.5] at 10 fps: 21 frames
        with av.open(str(clip_path)) as container:
            frames = sum(1 for _ in container.decode(container.streams.video[0]))
        assert frames == 21

    def test_clips_cross_camera_mode_skips_single_camera(self, store, gallery):
        gallery.config = ResolvedConfig.resolve(
            [(ConfigLayer.PROFILE_DEFAULT,
              {"package": {"thumb_size": 32, "clips": "cross_camera"}})]
        )
        result = _run(store, gallery)
        report = json.loads(
            store.artifact_path(result, "package.report").read_text("utf-8")
        )
        assert all(
            m["clip"] is None
            for identity in report["identities"] for m in identity["members"]
        )

    def test_clips_degrade_on_undecodable_evidence(self, store, gallery):
        # image-dir evidence cannot be transcoded — clip stays null, run OK
        gallery.config = ResolvedConfig.resolve(
            [(ConfigLayer.PROFILE_DEFAULT,
              {"package": {"thumb_size": 32, "clips": "all"}})]
        )
        from athar.core.timebase import CameraTimeBase

        gallery.timebase.cameras["cam01"] = CameraTimeBase(camera_id="cam01", fps=2.0)
        result = _run(store, gallery)
        assert result.status == RunStatus.COMPLETED
        report = json.loads(
            store.artifact_path(result, "package.report").read_text("utf-8")
        )
        assert all(
            m["clip"] is None
            for identity in report["identities"] for m in identity["members"]
        )

    def test_is_complete_skips_rerun(self, store, gallery):
        _run(store, gallery)
        reloaded = store.load(gallery.run_id)
        # the runner refuses COMPLETED runs outright; resume semantics are
        # "run again while incomplete", so re-arm the status first
        reloaded.status = RunStatus.RUNNING
        events = []
        runner = PipelineRunner(store, [PackageStage()], extra_sinks=[events.append])
        runner.run(reloaded, _profile())
        assert any(getattr(e, "event", "") == "stage_skipped" for e in events)
