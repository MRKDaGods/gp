"""Stage 2 — Feature Extraction & Refinement pipeline.

Extracts ReID embeddings and HSV histograms from tracklet crops,
applies PCA whitening, and produces refined feature vectors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from loguru import logger
from omegaconf import DictConfig

from src.core.constants import PERSON_CLASSES, VEHICLE_CLASSES
from src.core.data_models import Tracklet, TrackletFeatures
from src.core.io_utils import save_embeddings, save_hsv_features
from src.stage2_features.crop_extractor import CropExtractor
from src.stage2_features.embeddings import l2_normalize
from src.stage2_features.hsv_extractor import HSVExtractor
from src.stage2_features.pca_whitening import PCAWhitener
from src.stage2_features.reid_model import ReIDModel


def run_stage2(
    cfg: DictConfig,
    tracklets_by_camera: Dict[str, List[Tracklet]],
    video_paths: Dict[str, str],
    output_dir: str | Path,
    smoke_test: bool = False,
) -> List[TrackletFeatures]:
    """Run feature extraction on all tracklets.

    Args:
        cfg: Full pipeline config (uses cfg.stage2).
        tracklets_by_camera: Dict[camera_id, List[Tracklet]] from Stage 1.
        video_paths: Dict[camera_id, video_file_path] for crop extraction.
        output_dir: Directory for stage2 outputs.
        smoke_test: If True, process only first 3 tracklets per camera.

    Returns:
        List of TrackletFeatures for all processed tracklets.
    """
    stage_cfg = cfg.stage2
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize extractors
    crop_extractor = CropExtractor(
        min_area=stage_cfg.crop.min_area,
        padding_ratio=stage_cfg.crop.padding_ratio,
        samples_per_tracklet=stage_cfg.crop.samples_per_tracklet,
    )

    hsv_extractor = HSVExtractor(
        h_bins=stage_cfg.hsv.h_bins,
        s_bins=stage_cfg.hsv.s_bins,
        v_bins=stage_cfg.hsv.v_bins,
    )

    # Load ReID models (person and vehicle)
    person_reid = ReIDModel(
        model_name=stage_cfg.reid.person.model_name,
        weights_path=stage_cfg.reid.person.weights_path,
        embedding_dim=stage_cfg.reid.person.embedding_dim,
        input_size=tuple(stage_cfg.reid.person.input_size),
        device=stage_cfg.reid.device,
        half=stage_cfg.reid.half,
    )

    vehicle_reid = ReIDModel(
        model_name=stage_cfg.reid.vehicle.model_name,
        weights_path=stage_cfg.reid.vehicle.weights_path,
        embedding_dim=stage_cfg.reid.vehicle.embedding_dim,
        input_size=tuple(stage_cfg.reid.vehicle.input_size),
        device=stage_cfg.reid.device,
        half=stage_cfg.reid.half,
    )

    # Process all tracklets
    all_features: List[TrackletFeatures] = []
    all_raw_embeddings = []
    index_map = []

    for camera_id, tracklets in tracklets_by_camera.items():
        video_path = video_paths.get(camera_id)
        if video_path is None:
            logger.warning(f"No video path for camera {camera_id}, skipping")
            continue

        if smoke_test:
            tracklets = tracklets[:3]

        logger.info(f"Extracting features for camera {camera_id}: {len(tracklets)} tracklets")

        for tracklet in tracklets:
            # Extract crops
            crops = crop_extractor.extract_crops(tracklet, video_path)
            if not crops:
                logger.debug(f"No valid crops for tracklet {tracklet.track_id}")
                continue

            # Select ReID model based on class
            if tracklet.class_id in PERSON_CLASSES:
                reid = person_reid
            else:
                reid = vehicle_reid

            # Extract embedding
            raw_embedding = reid.get_tracklet_embedding(crops)
            if raw_embedding is None:
                continue

            # Extract HSV histogram
            hsv_hist = hsv_extractor.extract_tracklet_histogram(crops)

            all_raw_embeddings.append(raw_embedding)
            index_map.append({
                "track_id": tracklet.track_id,
                "camera_id": camera_id,
                "class_id": tracklet.class_id,
            })

            all_features.append(
                TrackletFeatures(
                    track_id=tracklet.track_id,
                    camera_id=camera_id,
                    class_id=tracklet.class_id,
                    embedding=raw_embedding,  # will be replaced after PCA
                    hsv_histogram=hsv_hist,
                    raw_embedding=raw_embedding,
                )
            )

    if not all_features:
        logger.warning("No features extracted from any tracklet")
        return []

    # Stack raw embeddings
    raw_matrix = np.stack(all_raw_embeddings, axis=0)  # (N, D)
    logger.info(f"Raw embedding matrix: {raw_matrix.shape}")

    # PCA whitening (only useful with enough samples to estimate covariance)
    if stage_cfg.pca.enabled:
        n_samples, n_features = raw_matrix.shape
        min_samples_for_pca = n_features * 2  # need at least 2x features for reliable PCA

        if n_samples >= min_samples_for_pca:
            whitener = PCAWhitener(n_components=stage_cfg.pca.n_components)
            pca_path = stage_cfg.pca.pca_model_path

            if Path(pca_path).exists():
                whitener.load(pca_path)
                logger.info(f"Loaded PCA model from {pca_path}")
            else:
                whitener.fit(raw_matrix)
                whitener.save(pca_path)
                logger.info(f"Fitted and saved PCA model to {pca_path}")

            embeddings = whitener.transform(raw_matrix)
        else:
            logger.warning(
                f"Skipping PCA: need at least {min_samples_for_pca} samples "
                f"for reliable {n_features}D PCA, but only have {n_samples}. "
                f"Using raw embeddings."
            )
            embeddings = raw_matrix
    else:
        embeddings = raw_matrix

    # L2 normalize
    embeddings = l2_normalize(embeddings)

    # Update features with refined embeddings
    for i, feat in enumerate(all_features):
        feat.embedding = embeddings[i]

    # Save outputs
    hsv_matrix = np.stack([f.hsv_histogram for f in all_features], axis=0)
    save_embeddings(embeddings, index_map, output_dir)
    save_hsv_features(hsv_matrix, output_dir)

    logger.info(
        f"Stage 2 complete: {len(all_features)} tracklet features, "
        f"embedding dim={embeddings.shape[1]}"
    )

    return all_features
