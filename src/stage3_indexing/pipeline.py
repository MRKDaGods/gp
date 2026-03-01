"""Stage 3 — Indexing & Storage pipeline.

Builds a FAISS index for embedding search and populates a SQLite metadata store.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from loguru import logger
from omegaconf import DictConfig

from src.core.data_models import Tracklet, TrackletFeatures
from src.stage3_indexing.faiss_index import FAISSIndex
from src.stage3_indexing.metadata_store import MetadataStore


def run_stage3(
    cfg: DictConfig,
    features: List[TrackletFeatures],
    tracklets_by_camera: Dict[str, List[Tracklet]],
    output_dir: str | Path,
) -> Tuple[FAISSIndex, MetadataStore]:
    """Build FAISS index and metadata store.

    Args:
        cfg: Full pipeline config (uses cfg.stage3).
        features: TrackletFeatures from Stage 2.
        tracklets_by_camera: Tracklets from Stage 1 for metadata.
        output_dir: Directory for stage3 outputs.

    Returns:
        (FAISSIndex, MetadataStore) tuple.
    """
    stage_cfg = cfg.stage3
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build FAISS index
    import numpy as np

    if not features:
        logger.warning("No features to index (0 tracklets). Creating empty index.")
        faiss_index = FAISSIndex(index_type=stage_cfg.faiss.index_type)
        db_path = output_dir / "metadata.db"
        metadata_store = MetadataStore(db_path)
        return faiss_index, metadata_store

    embeddings = np.stack([f.embedding for f in features], axis=0)
    ids = list(range(len(features)))

    faiss_index = FAISSIndex(index_type=stage_cfg.faiss.index_type)
    faiss_index.build(embeddings, ids)

    index_path = output_dir / "faiss_index.bin"
    faiss_index.save(index_path)
    logger.info(f"FAISS index built: {len(features)} vectors, saved to {index_path}")

    # Build metadata store
    db_path = output_dir / "metadata.db"
    metadata_store = MetadataStore(db_path)

    # Build a tracklet lookup
    tracklet_lookup: Dict[Tuple[str, int], Tracklet] = {}
    for cam_id, tracklets in tracklets_by_camera.items():
        for t in tracklets:
            tracklet_lookup[(cam_id, t.track_id)] = t

    # Insert metadata for each feature
    for i, feat in enumerate(features):
        key = (feat.camera_id, feat.track_id)
        tracklet = tracklet_lookup.get(key)

        start_time = tracklet.start_time if tracklet else 0.0
        end_time = tracklet.end_time if tracklet else 0.0
        num_frames = tracklet.num_frames if tracklet else 0

        metadata_store.insert_tracklet(
            index_id=i,
            track_id=feat.track_id,
            camera_id=feat.camera_id,
            class_id=feat.class_id,
            start_time=start_time,
            end_time=end_time,
            num_frames=num_frames,
            hsv_histogram=feat.hsv_histogram,
        )

    logger.info(f"Metadata store populated: {len(features)} entries, saved to {db_path}")

    return faiss_index, metadata_store
