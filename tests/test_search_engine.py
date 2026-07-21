"""Gallery search tests: probe→gallery hits, class filtering, compatibility
guards, hypothesis proposal + decide flow."""

from __future__ import annotations

import json

import numpy as np
import pytest

from athar.contracts.config import ConfigLayer, ResolvedConfig
from athar.contracts.manifest import ArtifactRecord, RunManifest, RunRole, VideoInput
from athar.contracts.store import FilesystemRunStore
from athar.core.ids import new_run_id
from athar.core.types import EntityClass
from athar.pipeline.runner import PipelineRunner
from athar.pipeline.stages.index import IndexStage
from athar.profiles.base import ClassBranch, ComponentSpec, RunProfile
from athar.search.case_models import HypothesisStatus, Target
from athar.search.engine import (
    GallerySearcher,
    SearchError,
    propose_appearance_hypotheses,
)

faiss = pytest.importorskip("faiss")

DIM = 12


def _profile() -> RunProfile:
    spec = ComponentSpec(name="x")
    return RunProfile(
        name="search-test",
        detector=spec,
        branches=[
            ClassBranch(entity_classes=[EntityClass.CAR],
                        tracker=spec, embedders=[spec], score_terms=[spec], solver=spec)
        ],
    )


def _unit(seed: int, base: np.ndarray | None = None, jitter: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = (base if base is not None else rng.normal(size=DIM)).astype(np.float32)
    if jitter:
        v = v + jitter * rng.normal(size=DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def _write_run(
    store: FilesystemRunStore,
    role: RunRole,
    tracks: dict[str, list[tuple[int, str, np.ndarray]]],
    build_index: bool,
) -> RunManifest:
    manifest = RunManifest(run_id=new_run_id(), role=role, profile_name="search-test")
    manifest.config = ResolvedConfig.resolve([(ConfigLayer.PROFILE_DEFAULT, {"a": 1})])
    run_dir = store.run_dir(manifest.run_id)
    (run_dir / "tracklets").mkdir(parents=True, exist_ok=True)
    (run_dir / "embeddings").mkdir(parents=True, exist_ok=True)
    streams = {"appearance": {"dim": DIM, "artifacts": []}}
    for cam, items in tracks.items():
        manifest.inputs.append(
            VideoInput(camera_id=cam, original_path=f"unused/{cam}", sha256="0" * 64)
        )
        payload = {
            "schema_version": 1, "camera_id": cam, "observations": {},
            "tracklets": [
                {
                    "key": {"run_id": manifest.run_id, "camera_id": cam, "track_id": tid},
                    "entity_class": cls,
                    "start_ts_scene_s": float(tid), "end_ts_scene_s": float(tid) + 1,
                    "observation_count": 4, "mean_confidence": 0.9,
                }
                for tid, cls, _ in items
            ],
        }
        (run_dir / "tracklets" / f"{cam}.json").write_text(json.dumps(payload), "utf-8")
        manifest.register_artifact(
            ArtifactRecord(name=f"tracklets.{cam}", relpath=f"tracklets/{cam}.json",
                           schema_version=1, producer="test", row_count=len(items))
        )
        np.savez(
            run_dir / "embeddings" / f"{cam}.appearance.npz",
            embeddings=np.stack([v for *_, v in items]),
            track_ids=np.asarray([tid for tid, *_ in items], dtype=np.int64),
        )
        name = f"embeddings.{cam}.appearance"
        manifest.register_artifact(
            ArtifactRecord(name=name, relpath=f"embeddings/{cam}.appearance.npz",
                           schema_version=1, producer="test", row_count=len(items))
        )
        streams["appearance"]["artifacts"].append(name)
    (run_dir / "embed_summary.json").write_text(
        json.dumps({"schema_version": 1, "streams": streams}), "utf-8"
    )
    manifest.register_artifact(
        ArtifactRecord(name="embed.summary", relpath="embed_summary.json",
                       schema_version=1, producer="test", row_count=1)
    )
    if build_index:
        manifest = PipelineRunner(store, [IndexStage()]).run(manifest, _profile())
    return manifest


@pytest.fixture()
def store(tmp_path):
    return FilesystemRunStore(tmp_path / "runs")


@pytest.fixture()
def base_vec():
    return np.random.default_rng(42).normal(size=DIM)


@pytest.fixture()
def gallery(store, base_vec):
    return _write_run(
        store, RunRole.GALLERY,
        {
            "g1": [(1, "car", _unit(1, base_vec, 0.05)), (2, "car", _unit(2))],
            "g2": [(1, "car", _unit(3, base_vec, 0.05)), (9, "car", _unit(4))],
        },
        build_index=True,
    )


@pytest.fixture()
def probe(store, base_vec):
    return _write_run(
        store, RunRole.PROBE,
        {"p1": [(5, "car", _unit(5, base_vec, 0.05))]},
        build_index=False,
    )


class TestSearch:
    def test_probe_finds_lookalikes_ranked(self, store, gallery, probe):
        searcher = GallerySearcher(store, gallery)
        hits = searcher.search_probe(store, probe, "appearance", top_k=4)
        assert hits, "no hits returned"
        top2 = {(h.gallery_key.camera_id, h.gallery_key.track_id) for h in hits[:2]}
        assert top2 == {("g1", 1), ("g2", 1)}  # the two base-like tracklets
        assert hits[0].score > hits[-1].score
        assert hits[0].probe_key.camera_id == "p1" and hits[0].probe_key.track_id == 5
        assert hits[0].gallery_entity_class is EntityClass.CAR

    def test_min_score_filters(self, store, gallery, probe):
        searcher = GallerySearcher(store, gallery)
        hits = searcher.search_probe(store, probe, "appearance", top_k=4, min_score=0.9)
        assert {(h.gallery_key.camera_id, h.gallery_key.track_id) for h in hits} == {
            ("g1", 1), ("g2", 1),
        }

    def test_unindexed_gallery_refused(self, store, base_vec):
        bare = _write_run(
            store, RunRole.GALLERY,
            {"g1": [(1, "car", _unit(1, base_vec))]}, build_index=False,
        )
        with pytest.raises(SearchError, match="not searchable"):
            GallerySearcher(store, bare)

    def test_unknown_stream_refused(self, store, gallery, probe):
        searcher = GallerySearcher(store, gallery)
        with pytest.raises(SearchError, match="no stream"):
            searcher.search_probe(store, probe, "face")

    def test_dim_mismatch_refused(self, store, gallery, probe):
        summary_path = store.run_dir(probe.run_id) / "embed_summary.json"
        summary = json.loads(summary_path.read_text("utf-8"))
        summary["streams"]["appearance"]["dim"] = DIM + 1
        summary_path.write_text(json.dumps(summary), "utf-8")
        with pytest.raises(SearchError, match="dim mismatch"):
            GallerySearcher(store, gallery).search_probe(store, probe, "appearance")

    def test_projection_lineage_mismatch_refused(self, store, gallery, probe):
        summary_path = store.run_dir(probe.run_id) / "embed_summary.json"
        summary = json.loads(summary_path.read_text("utf-8"))
        summary["streams"]["appearance"]["projection_fitted_on"] = "other-run"
        summary_path.write_text(json.dumps(summary), "utf-8")
        with pytest.raises(SearchError, match="lineage mismatch"):
            GallerySearcher(store, gallery).search_probe(store, probe, "appearance")


class TestHypotheses:
    def test_propose_dedupe_and_decide(self, store, gallery, probe):
        searcher = GallerySearcher(store, gallery)
        hits = searcher.search_probe(store, probe, "appearance", top_k=4)
        target = Target(target_id="tgt-1", label="Suspect A")

        added = propose_appearance_hypotheses(target, hits)
        assert added == len(target.hypotheses) > 0
        # re-proposing the same hits adds nothing
        assert propose_appearance_hypotheses(target, hits) == 0

        edge = target.hypotheses[0]
        assert edge.status is HypothesisStatus.PROPOSED
        edge.decide(HypothesisStatus.CONFIRMED, operator="analyst_1")
        assert edge.decided_by == "analyst_1" and edge.decided_at is not None
        with pytest.raises(ValueError, match="already decided"):
            edge.decide(HypothesisStatus.REJECTED, operator="analyst_2")
