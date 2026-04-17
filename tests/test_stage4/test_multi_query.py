"""Tests for Stage 4 multi-query similarity handling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from src.core.data_models import Tracklet, TrackletFeatures, TrackletFrame
from src.stage3_indexing.faiss_index import FAISSIndex
from src.stage3_indexing.metadata_store import MetadataStore
from src.stage4_association.pipeline import (
    _build_all_cross_camera_pairs_multi_query,
    run_stage4,
)


def test_build_all_cross_camera_pairs_multi_query_uses_max_pairwise_similarity():
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    mq_embeddings = [
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.float32),
    ]

    pairs = _build_all_cross_camera_pairs_multi_query(
        n=2,
        embeddings=embeddings,
        mq_embeddings=mq_embeddings,
        camera_ids=["cam_a", "cam_b"],
        class_ids=[2, 2],
        min_similarity=0.0,
        mq_weight=0.5,
    )

    assert len(pairs) == 1
    _, _, sim = pairs[0]
    assert sim == 0.5


def test_run_stage4_handles_disabled_multi_query_without_artifact(tmp_path):
    cfg = OmegaConf.create(
        {
            "stage4": {
                "association": {
                    "solver": "cc",
                    "top_k": 2,
                    "secondary_embeddings": {"path": "", "weight": 0.3},
                    "multi_query": {"enabled": False, "weight": 0.5, "dir": ""},
                    "fic": {"enabled": False, "regularisation": 0.1, "min_samples": 5},
                    "fac": {"enabled": False},
                    "query_expansion": {"enabled": False, "k": 5, "alpha": 5.0, "dba": True},
                    "exhaustive_cross_camera": True,
                    "exhaustive_min_similarity": 0.0,
                    "mutual_nn": {"enabled": False, "top_k_per_query": 10},
                    "reranking": {"enabled": False, "k1": 30, "k2": 10, "lambda_value": 0.4},
                    "weights": {
                        "appearance": 1.0,
                        "hsv": 0.0,
                        "spatiotemporal": 0.0,
                        "vehicle": {"appearance": 1.0, "hsv": 0.0, "spatiotemporal": 0.0},
                        "person": {"appearance": 1.0, "hsv": 0.0, "spatiotemporal": 0.0},
                        "length_weight_power": 0.0,
                    },
                    "spatiotemporal": {
                        "max_time_gap": 60,
                        "min_time_gap": 0,
                        "camera_transitions": None,
                    },
                    "graph": {
                        "similarity_threshold": 0.1,
                        "algorithm": "connected_components",
                        "louvain_resolution": 1.0,
                        "louvain_seed": 42,
                        "bridge_prune_margin": 0.0,
                        "max_component_size": 0,
                    },
                    "reciprocal_best_match": {"enabled": False, "min_similarity": 0.2},
                    "csls": {"enabled": False, "k": 10},
                    "cluster_verify": {"enabled": False, "min_connectivity": 0.3},
                    "temporal_split": {"enabled": False, "min_gap": 60.0, "split_threshold": 0.5},
                    "gallery_expansion": {
                        "enabled": False,
                        "threshold": 0.5,
                        "max_rounds": 1,
                        "orphan_match_threshold": 0.4,
                    },
                    "temporal_overlap": {"enabled": False, "bonus": 0.05, "max_mean_time": 5.0},
                    "camera_pair_norm": {"enabled": False, "min_pairs": 10},
                    "camera_bias": {"enabled": False},
                    "zone_model": {"enabled": False, "zone_data_path": ""},
                    "camera_pair_boost": {"enabled": False, "boosts": {}},
                }
            }
        }
    )

    features = [
        TrackletFeatures(
            track_id=1,
            camera_id="cam_a",
            class_id=2,
            embedding=np.array([1.0, 0.0], dtype=np.float32),
            hsv_histogram=np.array([1.0, 0.0], dtype=np.float32),
        ),
        TrackletFeatures(
            track_id=2,
            camera_id="cam_b",
            class_id=2,
            embedding=np.array([0.9, 0.1], dtype=np.float32),
            hsv_histogram=np.array([1.0, 0.0], dtype=np.float32),
        ),
    ]

    tracklets_by_camera = {
        "cam_a": [
            Tracklet(
                track_id=1,
                camera_id="cam_a",
                class_id=2,
                class_name="car",
                frames=[TrackletFrame(frame_id=0, timestamp=0.0, bbox=(0, 0, 10, 10), confidence=0.9)],
            )
        ],
        "cam_b": [
            Tracklet(
                track_id=2,
                camera_id="cam_b",
                class_id=2,
                class_name="car",
                frames=[TrackletFrame(frame_id=5, timestamp=5.0, bbox=(0, 0, 10, 10), confidence=0.9)],
            )
        ],
    }

    faiss_index = FAISSIndex(index_type="flat_ip")
    faiss_index.build(np.stack([feature.embedding for feature in features], axis=0))

    metadata_store = MetadataStore(tmp_path / "metadata.db")
    metadata_store.insert_tracklet(0, 1, "cam_a", 2, 0.0, 0.0, 1)
    metadata_store.insert_tracklet(1, 2, "cam_b", 2, 5.0, 5.0, 1)

    output_dir = tmp_path / "stage4"
    trajectories = run_stage4(
        cfg=cfg,
        faiss_index=faiss_index,
        metadata_store=metadata_store,
        features=features,
        tracklets_by_camera=tracklets_by_camera,
        output_dir=output_dir,
    )

    metadata_store.close()

    assert len(trajectories) >= 1


def test_run_stage4_uses_network_flow_solver_toggle(tmp_path):
    cfg = OmegaConf.create(
        {
            "stage4": {
                "association": {
                    "solver": "network_flow",
                    "top_k": 2,
                    "secondary_embeddings": {"path": "", "weight": 0.3},
                    "multi_query": {"enabled": False, "weight": 0.5, "dir": ""},
                    "fic": {"enabled": False, "regularisation": 0.1, "min_samples": 5},
                    "fac": {"enabled": False},
                    "query_expansion": {"enabled": False, "k": 5, "alpha": 5.0, "dba": True},
                    "exhaustive_cross_camera": True,
                    "exhaustive_min_similarity": 0.0,
                    "mutual_nn": {"enabled": False, "top_k_per_query": 10},
                    "reranking": {"enabled": False, "k1": 30, "k2": 10, "lambda_value": 0.4},
                    "weights": {
                        "appearance": 1.0,
                        "hsv": 0.0,
                        "spatiotemporal": 0.0,
                        "vehicle": {"appearance": 1.0, "hsv": 0.0, "spatiotemporal": 0.0},
                        "person": {"appearance": 1.0, "hsv": 0.0, "spatiotemporal": 0.0},
                        "length_weight_power": 0.0,
                    },
                    "spatiotemporal": {
                        "max_time_gap": 60,
                        "min_time_gap": 0,
                        "camera_transitions": None,
                    },
                    "graph": {
                        "similarity_threshold": 0.1,
                        "algorithm": "connected_components",
                        "merge_verify_threshold": 0.1,
                        "louvain_resolution": 1.0,
                        "louvain_seed": 42,
                        "bridge_prune_margin": 0.0,
                        "max_component_size": 0,
                    },
                    "reciprocal_best_match": {"enabled": False, "min_similarity": 0.2},
                    "csls": {"enabled": False, "k": 10},
                    "cluster_verify": {"enabled": False, "min_connectivity": 0.3},
                    "temporal_split": {"enabled": False, "min_gap": 60.0, "split_threshold": 0.5},
                    "gallery_expansion": {
                        "enabled": False,
                        "threshold": 0.5,
                        "max_rounds": 1,
                        "orphan_match_threshold": 0.4,
                    },
                    "temporal_overlap": {"enabled": False, "bonus": 0.05, "max_mean_time": 5.0},
                    "camera_pair_norm": {"enabled": False, "min_pairs": 10},
                    "camera_bias": {"enabled": False},
                    "zone_model": {"enabled": False, "zone_data_path": ""},
                    "camera_pair_boost": {"enabled": False, "boosts": {}},
                }
            }
        }
    )

    features = [
        TrackletFeatures(
            track_id=1,
            camera_id="cam_a",
            class_id=2,
            embedding=np.array([1.0, 0.0], dtype=np.float32),
            hsv_histogram=np.array([1.0, 0.0], dtype=np.float32),
        ),
        TrackletFeatures(
            track_id=2,
            camera_id="cam_b",
            class_id=2,
            embedding=np.array([0.9, 0.1], dtype=np.float32),
            hsv_histogram=np.array([1.0, 0.0], dtype=np.float32),
        ),
    ]

    tracklets_by_camera = {
        "cam_a": [
            Tracklet(
                track_id=1,
                camera_id="cam_a",
                class_id=2,
                class_name="car",
                frames=[TrackletFrame(frame_id=0, timestamp=0.0, bbox=(0, 0, 10, 10), confidence=0.9)],
            )
        ],
        "cam_b": [
            Tracklet(
                track_id=2,
                camera_id="cam_b",
                class_id=2,
                class_name="car",
                frames=[TrackletFrame(frame_id=5, timestamp=5.0, bbox=(0, 0, 10, 10), confidence=0.9)],
            )
        ],
    }

    faiss_index = FAISSIndex(index_type="flat_ip")
    faiss_index.build(np.stack([feature.embedding for feature in features], axis=0))

    metadata_store = MetadataStore(tmp_path / "metadata.db")
    metadata_store.insert_tracklet(0, 1, "cam_a", 2, 0.0, 0.0, 1)
    metadata_store.insert_tracklet(1, 2, "cam_b", 2, 5.0, 5.0, 1)

    output_dir = tmp_path / "stage4_network_flow"
    trajectories = run_stage4(
        cfg=cfg,
        faiss_index=faiss_index,
        metadata_store=metadata_store,
        features=features,
        tracklets_by_camera=tracklets_by_camera,
        output_dir=output_dir,
    )

    metadata_store.close()

    assert len(trajectories) >= 1
    assert any(len(trajectory.tracklets) == 2 for trajectory in trajectories)
    assert Path(output_dir / "global_trajectories.json").exists()