"""Tests for AFLink motion-based Stage 4 post-association linking."""

from __future__ import annotations

import numpy as np
from omegaconf import OmegaConf

from src.core.data_models import GlobalTrajectory, Tracklet, TrackletFeatures, TrackletFrame
from src.stage3_indexing.faiss_index import FAISSIndex
from src.stage3_indexing.metadata_store import MetadataStore
from src.stage4_association.aflink import aflink_post_association
from src.stage4_association.pipeline import run_stage4


def test_aflink_post_association_merges_motion_consistent_cross_camera_trajectories():
    tracklet_a = _make_tracklet(
        camera_id="cam_a",
        track_id=1,
        frame_ids=[0, 1, 2, 3, 4],
        centres=[(10, 10), (12, 10), (14, 10), (16, 10), (18, 10)],
    )
    tracklet_b = _make_tracklet(
        camera_id="cam_b",
        track_id=2,
        frame_ids=[8, 9, 10, 11, 12],
        centres=[(20, 10), (22, 10), (24, 10), (26, 10), (28, 10)],
    )
    trajectories = [
        GlobalTrajectory(global_id=0, tracklets=[tracklet_a]),
        GlobalTrajectory(global_id=1, tracklets=[tracklet_b]),
    ]

    merged = aflink_post_association(
        trajectories=trajectories,
        feature_to_tracklet_key=[("cam_a", 1), ("cam_b", 2)],
        tracklet_lookup={("cam_a", 1): tracklet_a, ("cam_b", 2): tracklet_b},
        max_time_gap_frames=10,
        max_spatial_gap_px=20.0,
        min_direction_cos=0.7,
        min_velocity_ratio=0.5,
        velocity_window=3,
    )

    assert len(merged) == 1
    assert {(tracklet.camera_id, tracklet.track_id) for tracklet in merged[0].tracklets} == {
        ("cam_a", 1),
        ("cam_b", 2),
    }
    assert any(record.get("merge_stage") == "aflink" for record in merged[0].evidence)


def test_aflink_post_association_blocks_same_camera_conflicts():
    tracklet_a1 = _make_tracklet(
        camera_id="cam_a",
        track_id=1,
        frame_ids=[0, 1, 2, 3],
        centres=[(10, 10), (12, 10), (14, 10), (16, 10)],
    )
    tracklet_b = _make_tracklet(
        camera_id="cam_b",
        track_id=2,
        frame_ids=[6, 7, 8, 9],
        centres=[(18, 10), (20, 10), (22, 10), (24, 10)],
    )
    tracklet_a2 = _make_tracklet(
        camera_id="cam_a",
        track_id=3,
        frame_ids=[12, 13, 14, 15],
        centres=[(26, 10), (28, 10), (30, 10), (32, 10)],
    )
    trajectories = [
        GlobalTrajectory(global_id=0, tracklets=[tracklet_a1]),
        GlobalTrajectory(global_id=1, tracklets=[tracklet_b]),
        GlobalTrajectory(global_id=2, tracklets=[tracklet_a2]),
    ]

    merged = aflink_post_association(
        trajectories=trajectories,
        feature_to_tracklet_key=[("cam_a", 1), ("cam_b", 2), ("cam_a", 3)],
        tracklet_lookup={
            ("cam_a", 1): tracklet_a1,
            ("cam_b", 2): tracklet_b,
            ("cam_a", 3): tracklet_a2,
        },
        max_time_gap_frames=10,
        max_spatial_gap_px=20.0,
        min_direction_cos=0.7,
        min_velocity_ratio=0.5,
        velocity_window=3,
    )

    assert len(merged) == 2
    assert all(len({tracklet.camera_id for tracklet in trajectory.tracklets}) == len(trajectory.tracklets) for trajectory in merged)


def test_run_stage4_applies_aflink_post_association(tmp_path):
    cfg = OmegaConf.create(
        {
            "stage4": {
                "association": {
                    "top_k": 2,
                    "secondary_embeddings": {"path": "", "weight": 0.3},
                    "tertiary_embeddings": {"path": "", "weight": 0.0},
                    "multi_query": {"enabled": False, "weight": 0.5, "dir": ""},
                    "fic": {"enabled": False, "regularisation": 0.1, "min_samples": 5},
                    "fac": {"enabled": False},
                    "query_expansion": {"enabled": False, "k": 5, "alpha": 5.0, "dba": True},
                    "exhaustive_cross_camera": True,
                    "exhaustive_min_similarity": 0.95,
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
                        "similarity_threshold": 0.99,
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
                    "aflink": {
                        "enabled": True,
                        "max_time_gap_frames": 10,
                        "max_spatial_gap_px": 20.0,
                        "min_direction_cos": 0.7,
                        "min_velocity_ratio": 0.5,
                        "velocity_window": 3,
                    },
                    "temporal_overlap": {"enabled": False, "bonus": 0.05, "max_mean_time": 5.0},
                    "camera_pair_norm": {"enabled": False, "min_pairs": 10},
                    "hierarchical": {"enabled": False},
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
            embedding=np.array([0.0, 1.0], dtype=np.float32),
            hsv_histogram=np.array([1.0, 0.0], dtype=np.float32),
        ),
    ]

    tracklet_a = _make_tracklet(
        camera_id="cam_a",
        track_id=1,
        frame_ids=[0, 1, 2, 3, 4],
        centres=[(10, 10), (12, 10), (14, 10), (16, 10), (18, 10)],
    )
    tracklet_b = _make_tracklet(
        camera_id="cam_b",
        track_id=2,
        frame_ids=[8, 9, 10, 11, 12],
        centres=[(20, 10), (22, 10), (24, 10), (26, 10), (28, 10)],
    )
    tracklets_by_camera = {
        "cam_a": [tracklet_a],
        "cam_b": [tracklet_b],
    }

    faiss_index = FAISSIndex(index_type="flat_ip")
    faiss_index.build(np.stack([feature.embedding for feature in features], axis=0))

    metadata_store = MetadataStore(tmp_path / "metadata.db")
    metadata_store.insert_tracklet(0, 1, "cam_a", 2, 0.0, 4.0, 5)
    metadata_store.insert_tracklet(1, 2, "cam_b", 2, 8.0, 12.0, 5)

    trajectories = run_stage4(
        cfg=cfg,
        faiss_index=faiss_index,
        metadata_store=metadata_store,
        features=features,
        tracklets_by_camera=tracklets_by_camera,
        output_dir=tmp_path / "stage4",
    )

    metadata_store.close()

    assert len(trajectories) == 1
    assert len(trajectories[0].tracklets) == 2
    assert any(record.get("merge_stage") == "aflink" for record in trajectories[0].evidence)


def _make_tracklet(
    *,
    camera_id: str,
    track_id: int,
    frame_ids: list[int],
    centres: list[tuple[float, float]],
) -> Tracklet:
    frames = []
    for frame_id, centre in zip(frame_ids, centres):
        x, y = centre
        frames.append(
            TrackletFrame(
                frame_id=frame_id,
                timestamp=float(frame_id),
                bbox=(x - 1.0, y - 1.0, x + 1.0, y + 1.0),
                confidence=0.9,
            )
        )
    return Tracklet(
        track_id=track_id,
        camera_id=camera_id,
        class_id=2,
        class_name="car",
        frames=frames,
    )