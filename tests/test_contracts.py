"""Contract tests: the invariants every other subsystem builds on."""

from __future__ import annotations

import pytest

from athar.contracts.config import ConfigLayer, ResolvedConfig
from athar.contracts.manifest import (
    ArtifactRecord,
    RunManifest,
    RunRole,
    RunStatus,
)
from athar.contracts.store import FilesystemRunStore, RunNotFound
from athar.core.ids import TrackKey, new_run_id
from athar.core.timebase import CameraTimeBase, SceneClock, TimeBaseSource


class TestResolvedConfig:
    def test_later_layer_wins_and_provenance_records_it(self):
        cfg = ResolvedConfig.resolve(
            [
                (ConfigLayer.PROFILE_DEFAULT, {"detector": {"size": "m", "conf": 0.25}}),
                (ConfigLayer.RUN_OVERRIDE, {"detector": {"size": "x"}}),
            ]
        )
        assert cfg.values["detector.size"] == "x"
        assert cfg.values["detector.conf"] == 0.25
        assert cfg.provenance["detector.size"] is ConfigLayer.RUN_OVERRIDE
        assert cfg.provenance["detector.conf"] is ConfigLayer.PROFILE_DEFAULT

    def test_layer_order_is_canonical_not_call_order(self):
        cfg = ResolvedConfig.resolve(
            [
                (ConfigLayer.RUN_OVERRIDE, {"k": "override"}),
                (ConfigLayer.PROFILE_DEFAULT, {"k": "default"}),
            ]
        )
        assert cfg.values["k"] == "override"

    def test_duplicate_layer_rejected(self):
        with pytest.raises(ValueError, match="duplicate"):
            ResolvedConfig.resolve(
                [
                    (ConfigLayer.CASE, {"a": 1}),
                    (ConfigLayer.CASE, {"b": 2}),
                ]
            )

    def test_hash_is_stable_and_value_sensitive(self):
        layers = [(ConfigLayer.PROFILE_DEFAULT, {"a": 1, "b": {"c": 2}})]
        assert (
            ResolvedConfig.resolve(layers).config_hash
            == ResolvedConfig.resolve(layers).config_hash
        )
        changed = [(ConfigLayer.PROFILE_DEFAULT, {"a": 1, "b": {"c": 3}})]
        assert ResolvedConfig.resolve(layers).config_hash != ResolvedConfig.resolve(changed).config_hash


class TestTimeBase:
    def test_scene_mapping_applies_offset_and_drift(self):
        tb = CameraTimeBase(
            camera_id="c017", fps=25.0, offset_s=10.0, drift_s_per_hour=3.6,
            source=TimeBaseSource.MANUAL, confidence=0.9,
        )
        # frame 25 → 1s local → +10s offset → +3.6*(1/3600)s drift
        assert tb.to_scene(25) == pytest.approx(11.001)

    def test_scene_clock_requires_declared_camera(self):
        clock = SceneClock(cameras={"c1": CameraTimeBase(camera_id="c1", fps=25.0)})
        assert clock.require("c1").fps == 25.0
        with pytest.raises(KeyError, match="c2"):
            clock.require("c2")

    def test_worst_source_reports_weakest_provenance(self):
        clock = SceneClock(
            cameras={
                "a": CameraTimeBase(camera_id="a", fps=25, source=TimeBaseSource.SYNCHRONIZED),
                "b": CameraTimeBase(camera_id="b", fps=25, source=TimeBaseSource.ASSUMED),
            }
        )
        assert clock.worst_source is TimeBaseSource.ASSUMED


class TestRunManifestStore:
    def _manifest(self) -> RunManifest:
        m = RunManifest(
            run_id=new_run_id(), role=RunRole.GALLERY, profile_name="vehicle_person_v1"
        )
        m.config = ResolvedConfig.resolve([(ConfigLayer.PROFILE_DEFAULT, {"x": 1})])
        m.register_artifact(
            ArtifactRecord(
                name="tracks", relpath="tracks/tracks.parquet",
                schema_version=1, producer="detect_track@2.0.0a0",
            )
        )
        return m

    def test_roundtrip_preserves_everything(self, tmp_path):
        store = FilesystemRunStore(tmp_path)
        manifest = self._manifest()
        store.save(manifest)
        loaded = store.load(manifest.run_id)
        assert loaded == manifest

    def test_role_is_a_manifest_attribute_not_a_path(self, tmp_path):
        store = FilesystemRunStore(tmp_path)
        gallery, probe = self._manifest(), self._manifest()
        probe.role = RunRole.PROBE
        store.save(gallery)
        store.save(probe)
        assert [m.run_id for m in store.list(role=RunRole.PROBE)] == [probe.run_id]
        # both live under the SAME root — no prefix namespaces
        assert (tmp_path / gallery.run_id).is_dir()
        assert (tmp_path / probe.run_id).is_dir()

    def test_missing_run_raises_typed_error(self, tmp_path):
        with pytest.raises(RunNotFound):
            FilesystemRunStore(tmp_path).load("run-nope")

    def test_duplicate_artifact_rejected(self):
        m = self._manifest()
        with pytest.raises(ValueError, match="already registered"):
            m.register_artifact(
                ArtifactRecord(
                    name="tracks", relpath="other", schema_version=1, producer="x"
                )
            )

    def test_terminal_status(self):
        assert RunStatus.COMPLETED.is_terminal
        assert not RunStatus.RUNNING.is_terminal


class TestTrackKey:
    def test_frozen_and_usable_as_dict_key(self):
        k1 = TrackKey(run_id="r", camera_id="c1", track_id=5)
        k2 = TrackKey(run_id="r", camera_id="c1", track_id=5)
        assert k1 == k2
        assert len({k1: 1, k2: 2}) == 1
        with pytest.raises(Exception):
            k1.track_id = 6  # type: ignore[misc]
