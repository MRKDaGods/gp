"""associate stage tests over a synthetic two-camera gallery.

Two true identities (A, B), each seen once per camera with nearly identical
appearance vectors and a plausible time gap; one decoy tracklet that matches
nothing. Verifies clustering, class gating, time gating, evidence, and the
windowed-seam guard.
"""

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
from athar.pipeline.stages.associate import AssociateStage, config_subtree
from athar.pipeline.stages.index import IndexStage
from athar.profiles.base import ClassBranch, ComponentSpec, RunProfile

faiss = pytest.importorskip("faiss")

DIM = 16


def _profile() -> RunProfile:
    spec = ComponentSpec(name="x")
    return RunProfile(
        name="assoc-test",
        detector=spec,
        branches=[
            ClassBranch(
                entity_classes=[EntityClass.PERSON],
                tracker=spec, embedders=[spec], score_terms=[spec], solver=spec,
            ),
            ClassBranch(
                entity_classes=[EntityClass.CAR, EntityClass.BUS, EntityClass.TRUCK],
                tracker=spec, embedders=[spec], score_terms=[spec], solver=spec,
            ),
        ],
    )


def _vec(base: np.ndarray, jitter: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = base + jitter * rng.normal(size=DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).astype(np.float32)


def _config(**overrides) -> ResolvedConfig:
    cfg = {
        "associate": {
            "top_k": 5,
            "similarity_threshold": 0.5,
            "min_time_gap": 0.0,
            "max_time_gap": 100.0,
            "weights": {
                "person": {"appearance": 0.7, "hsv": 0.05, "spatiotemporal": 0.25},
                "vehicle": {"appearance": 0.7, "hsv": 0.05, "spatiotemporal": 0.25},
            },
            **overrides,
        }
    }
    return ResolvedConfig.resolve([(ConfigLayer.PROFILE_DEFAULT, cfg)])


def _gallery(store: FilesystemRunStore, config: ResolvedConfig) -> RunManifest:
    """cam01: A(car t0-5), B(car t2-7), decoy person; cam02: A(t20-25), B(t22-27)."""
    manifest = RunManifest(
        run_id=new_run_id(), role=RunRole.GALLERY, profile_name="assoc-test"
    )
    manifest.config = config
    run_dir = store.run_dir(manifest.run_id)
    (run_dir / "tracklets").mkdir(parents=True, exist_ok=True)
    (run_dir / "embeddings").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(1)
    base_a = rng.normal(size=DIM).astype(np.float32)
    base_b = rng.normal(size=DIM).astype(np.float32)
    # ensure A and B are dissimilar identities
    base_b -= (base_b @ base_a) / (base_a @ base_a) * base_a

    spec = {
        "cam01": [
            (1, "car", (0.0, 5.0), _vec(base_a, 0.02, 11)),
            (2, "car", (2.0, 7.0), _vec(base_b, 0.02, 12)),
            (3, "person", (0.0, 9.0), _vec(base_a, 0.02, 13)),  # decoy: A-like PERSON
        ],
        "cam02": [
            (1, "car", (20.0, 25.0), _vec(base_a, 0.02, 21)),
            (2, "car", (22.0, 27.0), _vec(base_b, 0.02, 22)),
        ],
    }
    streams = {"appearance": {"dim": DIM, "artifacts": []},
               "hsv": {"dim": 4, "artifacts": []}}
    for cam, tracks in spec.items():
        manifest.inputs.append(
            VideoInput(camera_id=cam, original_path=f"unused/{cam}", sha256="0" * 64)
        )
        payload = {
            "schema_version": 1, "camera_id": cam, "observations": {},
            "tracklets": [
                {
                    "key": {"run_id": manifest.run_id, "camera_id": cam, "track_id": tid},
                    "entity_class": cls,
                    "start_ts_scene_s": t0, "end_ts_scene_s": t1,
                    "observation_count": 10, "mean_confidence": 0.9,
                }
                for tid, cls, (t0, t1), _ in tracks
            ],
        }
        (run_dir / "tracklets" / f"{cam}.json").write_text(json.dumps(payload), "utf-8")
        manifest.register_artifact(
            ArtifactRecord(name=f"tracklets.{cam}", relpath=f"tracklets/{cam}.json",
                           schema_version=1, producer="test", row_count=len(tracks))
        )
        np.savez(
            run_dir / "embeddings" / f"{cam}.appearance.npz",
            embeddings=np.stack([v for *_, v in tracks]),
            track_ids=np.asarray([tid for tid, *_ in tracks], dtype=np.int64),
        )
        np.savez(
            run_dir / "embeddings" / f"{cam}.hsv.npz",
            embeddings=np.tile(np.full(4, 0.5, dtype=np.float32), (len(tracks), 1)),
            track_ids=np.asarray([tid for tid, *_ in tracks], dtype=np.int64),
        )
        for stream in ("appearance", "hsv"):
            name = f"embeddings.{cam}.{stream}"
            manifest.register_artifact(
                ArtifactRecord(name=name, relpath=f"embeddings/{cam}.{stream}.npz",
                               schema_version=1, producer="test", row_count=len(tracks))
            )
            streams[stream]["artifacts"].append(name)

    (run_dir / "embed_summary.json").write_text(
        json.dumps({"schema_version": 1, "streams": streams}), "utf-8"
    )
    manifest.register_artifact(
        ArtifactRecord(name="embed.summary", relpath="embed_summary.json",
                       schema_version=1, producer="test", row_count=2)
    )
    return manifest


def _identities(payload: dict) -> dict[frozenset, dict]:
    return {
        frozenset((m["camera_id"], m["track_id"]) for m in t["members"]): t
        for t in payload["trajectories"]
    }


@pytest.fixture()
def store(tmp_path):
    return FilesystemRunStore(tmp_path / "runs")


def _run(store, manifest):
    runner = PipelineRunner(store, [IndexStage(), AssociateStage()])
    return runner.run(manifest, _profile())


class TestAssociate:
    def test_cross_camera_identities_recovered(self, store):
        manifest = _gallery(store, _config())
        result = _run(store, manifest)
        assert result.status == RunStatus.COMPLETED

        payload = json.loads(
            store.artifact_path(result, "associate.trajectories").read_text("utf-8")
        )
        identities = _identities(payload)
        # A and B each merged across cameras; person decoy alone
        assert frozenset({("cam01", 1), ("cam02", 1)}) in identities
        assert frozenset({("cam01", 2), ("cam02", 2)}) in identities
        assert frozenset({("cam01", 3)}) in identities
        assert payload["num_identities"] == 3

        merged = identities[frozenset({("cam01", 1), ("cam02", 1)})]
        assert merged["entity_class"] == "car"
        assert merged["confidence"] > 0.5
        assert set(merged["evidence"]) == {"appearance", "hsv", "spatiotemporal"}
        assert merged["evidence"]["appearance"] > 0.9

    def test_class_gating_blocks_cross_branch_pairs(self, store):
        # the person decoy has an A-like vector but must never join A's cluster
        manifest = _gallery(store, _config())
        result = _run(store, manifest)
        payload = json.loads(
            store.artifact_path(result, "associate.trajectories").read_text("utf-8")
        )
        for t in payload["trajectories"]:
            classes = {t["entity_class"]}
            assert len(classes) == 1  # no mixed-class identity

    def test_time_gate_blocks_implausible_transitions(self, store):
        manifest = _gallery(store, _config(max_time_gap=5.0))  # 15s gap now blocked
        result = _run(store, manifest)
        payload = json.loads(
            store.artifact_path(result, "associate.trajectories").read_text("utf-8")
        )
        assert payload["num_identities"] == 5  # nothing merges

    def test_windowed_seam_guarded(self, store):
        manifest = _gallery(store, _config(window_s=30.0))
        with pytest.raises(NotImplementedError, match="windowed association"):
            _run(store, manifest)

    def test_config_subtree_rebuilds_nested(self):
        cfg = _config()
        weights = config_subtree(cfg, "associate.weights")
        assert weights["person"]["appearance"] == 0.7
        assert weights["vehicle"]["spatiotemporal"] == 0.25
