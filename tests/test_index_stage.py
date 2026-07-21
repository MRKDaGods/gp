"""index stage tests: catalog determinism, per-stream FAISS + row sidecars,
HSV blobs, search round-trip through the catalog join."""

from __future__ import annotations

import json

import numpy as np
import pytest

from athar.contracts.config import ConfigLayer, ResolvedConfig
from athar.contracts.manifest import ArtifactRecord, RunManifest, RunRole, RunStatus, VideoInput
from athar.contracts.store import FilesystemRunStore
from athar.core.ids import new_run_id
from athar.core.types import EntityClass
from athar.pipeline.runner import PipelineRunner
from athar.pipeline.stages.index import IndexStage
from athar.profiles.base import ClassBranch, ComponentSpec, RunProfile

faiss = pytest.importorskip("faiss")

DIM = 8


def _profile() -> RunProfile:
    spec = ComponentSpec(name="x")
    return RunProfile(
        name="index-test",
        detector=spec,
        branches=[
            ClassBranch(
                entity_classes=[EntityClass.CAR],
                tracker=spec, embedders=[spec], score_terms=[spec], solver=spec,
            )
        ],
    )


def _unit(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def _prepare_run(store: FilesystemRunStore) -> RunManifest:
    """Two cameras; cam02 has one tracklet missing from the appearance stream."""
    manifest = RunManifest(
        run_id=new_run_id(), role=RunRole.GALLERY, profile_name="index-test"
    )
    manifest.config = ResolvedConfig.resolve([(ConfigLayer.PROFILE_DEFAULT, {"a": 1})])
    run_dir = store.run_dir(manifest.run_id)
    (run_dir / "tracklets").mkdir(parents=True, exist_ok=True)
    (run_dir / "embeddings").mkdir(parents=True, exist_ok=True)

    def tracklet(cam: str, track_id: int) -> dict:
        return {
            "key": {"run_id": manifest.run_id, "camera_id": cam, "track_id": track_id},
            "entity_class": "car",
            "start_ts_scene_s": float(track_id),
            "end_ts_scene_s": float(track_id) + 2.0,
            "observation_count": 5,
            "mean_confidence": 0.9,
        }

    per_cam = {"cam01": [1, 2], "cam02": [1, 7]}
    for cam, ids in per_cam.items():
        manifest.inputs.append(
            VideoInput(camera_id=cam, original_path=f"unused/{cam}", sha256="0" * 64)
        )
        payload = {
            "schema_version": 1, "camera_id": cam,
            "tracklets": [tracklet(cam, i) for i in ids],
            "observations": {},
        }
        (run_dir / "tracklets" / f"{cam}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        manifest.register_artifact(
            ArtifactRecord(name=f"tracklets.{cam}", relpath=f"tracklets/{cam}.json",
                           schema_version=1, producer="test", row_count=len(ids))
        )

    # appearance stream: cam02 track 7 has NO row (embedder failed on it)
    streams = {"appearance": {"dim": DIM, "artifacts": []}, "hsv": {"dim": 4, "artifacts": []}}
    app_rows = {"cam01": [1, 2], "cam02": [1]}
    for cam, ids in app_rows.items():
        np.savez(
            run_dir / "embeddings" / f"{cam}.appearance.npz",
            embeddings=np.stack([_unit(hash((cam, i)) % 1000) for i in ids]),
            track_ids=np.asarray(ids, dtype=np.int64),
        )
        name = f"embeddings.{cam}.appearance"
        manifest.register_artifact(
            ArtifactRecord(name=name, relpath=f"embeddings/{cam}.appearance.npz",
                           schema_version=1, producer="test", row_count=len(ids))
        )
        streams["appearance"]["artifacts"].append(name)
    for cam, ids in per_cam.items():
        np.savez(
            run_dir / "embeddings" / f"{cam}.hsv.npz",
            embeddings=np.tile(np.arange(4, dtype=np.float32), (len(ids), 1)),
            track_ids=np.asarray(ids, dtype=np.int64),
        )
        name = f"embeddings.{cam}.hsv"
        manifest.register_artifact(
            ArtifactRecord(name=name, relpath=f"embeddings/{cam}.hsv.npz",
                           schema_version=1, producer="test", row_count=len(ids))
        )
        streams["hsv"]["artifacts"].append(name)

    (run_dir / "embed_summary.json").write_text(
        json.dumps({"schema_version": 1, "streams": streams}), encoding="utf-8"
    )
    manifest.register_artifact(
        ArtifactRecord(name="embed.summary", relpath="embed_summary.json",
                       schema_version=1, producer="test", row_count=2)
    )
    return manifest


@pytest.fixture()
def store(tmp_path):
    return FilesystemRunStore(tmp_path / "runs")


class TestIndexStage:
    def test_catalog_faiss_and_join(self, store):
        from athar.components.indexing.metadata_store import MetadataStore

        manifest = _prepare_run(store)
        result = PipelineRunner(store, [IndexStage()]).run(manifest, _profile())
        assert result.status == RunStatus.COMPLETED
        assert {"index.catalog", "index.appearance", "index.appearance.rows"} <= set(
            result.artifacts
        )
        assert "index.hsv" not in result.artifacts  # hsv is catalog-blob, not FAISS

        catalog = MetadataStore(store.artifact_path(result, "index.catalog"))
        try:
            rows = catalog.get_all()
            assert len(rows) == 4
            # deterministic order: manifest camera order then track id
            assert [(r["camera_id"], r["track_id"]) for r in rows] == [
                ("cam01", 1), ("cam01", 2), ("cam02", 1), ("cam02", 7),
            ]
            assert [r["index_id"] for r in rows] == [0, 1, 2, 3]
            assert catalog.get_hsv_histogram(0) is not None
        finally:
            catalog.close()

        # FAISS index excludes the missing appearance row (index_id 3)
        row_map = json.loads(
            store.artifact_path(result, "index.appearance.rows").read_text()
        )
        assert row_map == [0, 1, 2]
        index = faiss.read_index(str(store.artifact_path(result, "index.appearance")))
        assert index.ntotal == 3

        # search round-trip: querying cam01/track2's own vector returns itself
        query = _unit(hash(("cam01", 2)) % 1000).reshape(1, -1)
        scores, positions = index.search(query, 1)
        hit_index_id = row_map[int(positions[0][0])]
        assert hit_index_id == 1  # catalog row for cam01/track2
        assert scores[0][0] == pytest.approx(1.0, abs=1e-5)

    def test_resume_skips(self, store):
        manifest = _prepare_run(store)
        runner = PipelineRunner(store, [IndexStage()])
        result = runner.run(manifest, _profile())
        result.status = RunStatus.CREATED
        rerun = runner.run(result, _profile())
        assert rerun.status == RunStatus.COMPLETED
        events = (store.run_dir(result.run_id) / "events.jsonl").read_text()
        assert "stage_skipped" in events

    def test_no_tracklets_rejected(self, store):
        manifest = _prepare_run(store)
        # drop tracklet artifacts -> stage must refuse
        for name in list(manifest.artifacts):
            if name.startswith("tracklets."):
                del manifest.artifacts[name]
        with pytest.raises(ValueError, match="no tracklets"):
            PipelineRunner(store, [IndexStage()]).run(manifest, _profile())
