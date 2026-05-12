"""Generate synthetic embedding fixtures for timeline service tests.

Run this script once (or after changing fixture design) to write the
binary/JSON files that the pytest fixtures load:

    python tests/fixtures/timeline/generate_fixtures.py

Output layout:
    tests/fixtures/timeline/
        probe/
            embeddings.npy          (10 rows, 384D, L2-normalised float32)
            embedding_index.json    (tracks 1-5, camera c001, class_id 2)
        gallery/
            embeddings.npy          (20 rows, 384D, L2-normalised float32)
            embedding_index.json    (tracks 10-14 cam c002, 20-24 cam c003)
        global_trajectories.json    (3 trajectories spanning c002 tracks)

Design guarantees used in tests
--------------------------------
* Gallery rows 0-9 are **identical** to probe rows 0-9 (same L2-normed
  vectors), so dot-product similarity ≈ 1.0 for those pairs.
* Gallery rows 10-19 are independent random vectors (different trajectories).
* Probe tracks 1-5 map to gallery tracks 10-14 via the trajectory data,
  ensuring test_query_visual_match sees mean_best ≥ 0.82 and p25_best ≥ 0.74.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

FIXTURE_DIR = Path(__file__).resolve().parent
PROBE_DIR   = FIXTURE_DIR / "probe"
GALLERY_DIR = FIXTURE_DIR / "gallery"
DIM         = 384
RNG_SEED    = 42


def _l2(m: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation."""
    return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-8)


def generate() -> None:
    rng = np.random.default_rng(RNG_SEED)

    # ── Probe: 10 rows — 2 rows per track, tracks 1-5, camera c001 ──────
    probe_raw = rng.standard_normal((10, DIM)).astype(np.float32)
    probe_emb = _l2(probe_raw)

    probe_index = [
        {"track_id": (i // 2) + 1, "camera_id": "c001", "class_id": 2}
        for i in range(10)
    ]
    # track 1 → rows 0,1 | track 2 → rows 2,3 | … | track 5 → rows 8,9

    # ── Gallery: 20 rows ─────────────────────────────────────────────────
    # Rows 0-9: identical copies of probe (will give similarity ≈ 1.0)
    # Rows 10-19: independent random vectors
    gallery_extra = _l2(rng.standard_normal((10, DIM)).astype(np.float32))
    gallery_emb   = np.vstack([probe_emb, gallery_extra])  # already L2-normed

    gallery_index = (
        [
            {"track_id": (i // 2) + 10, "camera_id": "c002", "class_id": 2}
            for i in range(10)
        ]
        # track 10 → rows 0,1 | track 11 → rows 2,3 | … | track 14 → rows 8,9
        + [
            {"track_id": (i // 2) + 20, "camera_id": "c003", "class_id": 2}
            for i in range(10)
        ]
    )

    # ── Trajectories: 3, each referencing gallery c002 tracks ───────────
    # Trajectory 0: tracks 10 + 11  → gallery rows 0-3 → matches probe rows 0-3
    # Trajectory 1: tracks 12 + 13  → gallery rows 4-7 → matches probe rows 4-7
    # Trajectory 2: track  14       → gallery rows 8-9 → matches probe rows 8-9
    trajectories = [
        {
            "global_id": 100,
            "tracklets": [
                {"camera_id": "c002", "track_id": 10, "class_id": 2,
                 "start_frame": 0, "end_frame": 30},
                {"camera_id": "c002", "track_id": 11, "class_id": 2,
                 "start_frame": 0, "end_frame": 30},
            ],
            "timeline": [],
        },
        {
            "global_id": 101,
            "tracklets": [
                {"camera_id": "c002", "track_id": 12, "class_id": 2,
                 "start_frame": 30, "end_frame": 60},
                {"camera_id": "c002", "track_id": 13, "class_id": 2,
                 "start_frame": 30, "end_frame": 60},
            ],
            "timeline": [],
        },
        {
            "global_id": 102,
            "tracklets": [
                {"camera_id": "c002", "track_id": 14, "class_id": 2,
                 "start_frame": 60, "end_frame": 90},
            ],
            "timeline": [],
        },
    ]

    # ── Write files ───────────────────────────────────────────────────────
    for d in (PROBE_DIR, GALLERY_DIR):
        d.mkdir(parents=True, exist_ok=True)

    np.save(PROBE_DIR / "embeddings.npy", probe_emb)
    (PROBE_DIR / "embedding_index.json").write_text(
        json.dumps(probe_index, indent=2), encoding="utf-8"
    )

    np.save(GALLERY_DIR / "embeddings.npy", gallery_emb)
    (GALLERY_DIR / "embedding_index.json").write_text(
        json.dumps(gallery_index, indent=2), encoding="utf-8"
    )

    (FIXTURE_DIR / "global_trajectories.json").write_text(
        json.dumps(trajectories, indent=2), encoding="utf-8"
    )

    print(f"Probe    : {probe_emb.shape}  → {PROBE_DIR}")
    print(f"Gallery  : {gallery_emb.shape} → {GALLERY_DIR}")
    print(f"Trajectories: {len(trajectories)} → {FIXTURE_DIR / 'global_trajectories.json'}")
    print("Done.")


if __name__ == "__main__":
    generate()
    sys.exit(0)
