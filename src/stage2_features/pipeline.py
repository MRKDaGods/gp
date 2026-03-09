"""Stage 2 — Feature Extraction & Refinement pipeline (SOTA).

Extracts ReID embeddings and spatial HSV histograms from quality-scored
tracklet crops, applies flip augmentation, quality-weighted temporal
attention pooling, camera-aware batch normalisation, PCA whitening,
and L2 normalisation to produce refined feature vectors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from loguru import logger
from omegaconf import DictConfig

from src.core.constants import PERSON_CLASSES, VEHICLE_CLASSES
from src.core.data_models import Tracklet, TrackletFeatures
from src.core.io_utils import save_embeddings, save_hsv_features
from src.stage2_features.crop_extractor import CropExtractor
from src.stage2_features.embeddings import camera_aware_batch_normalize, l2_normalize
from src.stage2_features.hsv_extractor import HSVExtractor
from src.stage2_features.pca_whitening import PCAWhitener
from src.stage2_features.reid_model import ReIDModel


def _load_frames_for_camera(
    stage0_dir: Path, camera_id: str, needed_frame_ids: set[int],
) -> dict[int, np.ndarray]:
    """Load needed frames from Stage 0 extracted images on disk.

    Falls back gracefully — returns only frames found on disk.

    Args:
        stage0_dir: Stage 0 output directory (contains per-camera subdirectories).
        camera_id: Camera identifier (subdirectory name).
        needed_frame_ids: Set of frame_id values to load.

    Returns:
        Dict[frame_id, BGR image].
    """
    cam_dir = stage0_dir / camera_id
    if not cam_dir.exists():
        return {}

    frames: dict[int, np.ndarray] = {}
    for fid in sorted(needed_frame_ids):
        # Try JPEG first (default), then PNG (lossless mode)
        for ext in (".jpg", ".png"):
            fpath = cam_dir / f"frame_{fid:06d}{ext}"
            if fpath.exists():
                img = cv2.imread(str(fpath))
                if img is not None:
                    frames[fid] = img
                break

    return frames


def run_stage2(
    cfg: DictConfig,
    tracklets_by_camera: Dict[str, List[Tracklet]],
    video_paths: Dict[str, str],
    output_dir: str | Path,
    smoke_test: bool = False,
    stage0_dir: str | Path | None = None,
) -> List[TrackletFeatures]:
    """Run feature extraction on all tracklets.

    Pipeline per tracklet:
    1. Quality-aware crop selection (sharpness, size, confidence)
    2. Flip-augmented ReID embedding extraction
    3. Quality-weighted temporal attention pooling → single embedding
    4. Spatial (3-stripe) HSV histogram with quality weighting

    Global post-processing:
    5. Camera-aware batch normalisation (optional)
    6. PCA whitening
    7. L2 normalisation

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

    # --- Initialise extractors ---
    crop_extractor = CropExtractor(
        min_area=stage_cfg.crop.min_area,
        padding_ratio=stage_cfg.crop.padding_ratio,
        samples_per_tracklet=stage_cfg.crop.samples_per_tracklet,
        min_quality=stage_cfg.crop.get("min_quality", 0.05),
        laplacian_min_var=stage_cfg.crop.get("laplacian_min_var", 0.0),
    )

    hsv_extractor = HSVExtractor(
        h_bins=stage_cfg.hsv.h_bins,
        s_bins=stage_cfg.hsv.s_bins,
        v_bins=stage_cfg.hsv.v_bins,
        n_stripes=stage_cfg.hsv.get("n_stripes", 3),
    )

    flip_augment = stage_cfg.reid.get("flip_augment", True)

    # --- Load ReID models (person and vehicle) ---
    person_reid = ReIDModel(
        model_name=stage_cfg.reid.person.model_name,
        weights_path=stage_cfg.reid.person.weights_path,
        embedding_dim=stage_cfg.reid.person.embedding_dim,
        input_size=tuple(stage_cfg.reid.person.input_size),
        device=stage_cfg.reid.device,
        half=stage_cfg.reid.half,
        flip_augment=flip_augment,
        num_cameras=stage_cfg.reid.person.get("num_cameras", 0),
        vit_model=stage_cfg.reid.person.get("vit_model", "vit_base_patch16_clip_224.openai"),
        clip_normalization=stage_cfg.reid.person.get("clip_normalization", None),
    )

    vehicle_reid = ReIDModel(
        model_name=stage_cfg.reid.vehicle.model_name,
        weights_path=stage_cfg.reid.vehicle.weights_path,
        embedding_dim=stage_cfg.reid.vehicle.embedding_dim,
        input_size=tuple(stage_cfg.reid.vehicle.input_size),
        device=stage_cfg.reid.device,
        half=stage_cfg.reid.half,
        flip_augment=flip_augment,
        num_cameras=stage_cfg.reid.vehicle.get("num_cameras", 0),
        vit_model=stage_cfg.reid.vehicle.get("vit_model", "vit_base_patch16_clip_224.openai"),
        clip_normalization=stage_cfg.reid.vehicle.get("clip_normalization", None),
    )

    # --- Resolve stage0 output directory for fast disk-based frame loading ---
    s0_dir: Optional[Path] = None
    if stage0_dir is not None:
        s0_dir = Path(stage0_dir)
    else:
        # Auto-discover: output_dir is stage2 subdir → sibling stage0
        candidate = output_dir.parent / "stage0"
        if candidate.is_dir():
            s0_dir = candidate

    use_disk_frames = s0_dir is not None and s0_dir.is_dir()
    if use_disk_frames:
        logger.info(f"Using extracted frames from disk: {s0_dir}")
    else:
        logger.info("Reading frames from video (slow — consider keeping stage0 output)")

    # --- Process all tracklets ---
    all_features: List[TrackletFeatures] = []
    all_raw_embeddings: List[np.ndarray] = []
    all_camera_ids: List[str] = []
    index_map: List[dict] = []

    for camera_id, tracklets in tracklets_by_camera.items():
        video_path = video_paths.get(camera_id)
        if video_path is None and not use_disk_frames:
            logger.warning(f"No video path for camera {camera_id}, skipping")
            continue

        if smoke_test:
            tracklets = tracklets[:3]

        logger.info(f"Extracting features for camera {camera_id}: {len(tracklets)} tracklets")

        # Pre-load frames from disk for this camera (much faster than video seeking)
        cam_frame_images: Optional[dict[int, np.ndarray]] = None
        if use_disk_frames:
            # Collect all unique frame_ids needed across all tracklets for this camera
            needed_ids: set[int] = set()
            for t in tracklets:
                for tf in t.frames:
                    needed_ids.add(tf.frame_id)
            cam_frame_images = _load_frames_for_camera(s0_dir, camera_id, needed_ids)
            logger.info(
                f"  Loaded {len(cam_frame_images)}/{len(needed_ids)} frames from disk for {camera_id}"
            )

        for tracklet in tracklets:
            # 1. Quality-aware crop selection
            if cam_frame_images is not None:
                scored_crops = crop_extractor.extract_crops_from_frames(
                    tracklet, cam_frame_images,
                )
            else:
                scored_crops = crop_extractor.extract_crops(tracklet, video_path)
            if not scored_crops:
                logger.debug(f"No valid crops for tracklet {tracklet.track_id}")
                continue

            # Select ReID model based on class
            if tracklet.class_id in PERSON_CLASSES:
                reid = person_reid
            else:
                reid = vehicle_reid

            # 2 & 3. Flip-augmented extraction + quality-weighted attention pooling
            raw_embedding = reid.get_tracklet_embedding_from_scored_crops(scored_crops)
            if raw_embedding is None:
                continue

            # 4. Spatial HSV histogram with quality weighting
            hsv_hist = hsv_extractor.extract_tracklet_histogram_from_scored_crops(
                scored_crops
            )

            all_raw_embeddings.append(raw_embedding)
            all_camera_ids.append(camera_id)
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
                    embedding=raw_embedding,  # will be replaced after post-processing
                    hsv_histogram=hsv_hist,
                    raw_embedding=raw_embedding.copy(),
                )
            )

    if not all_features:
        logger.warning("No features extracted from any tracklet")
        return []

    # --- Global post-processing ---
    raw_matrix = np.stack(all_raw_embeddings, axis=0)  # (N, D)
    logger.info(f"Raw embedding matrix: {raw_matrix.shape}")

    # 5. Camera-aware batch normalisation
    if stage_cfg.get("camera_bn", {}).get("enabled", True):
        logger.info("Applying camera-aware batch normalisation")
        raw_matrix = camera_aware_batch_normalize(raw_matrix, all_camera_ids)

    # 6. PCA whitening
    if stage_cfg.pca.enabled:
        n_samples, n_features = raw_matrix.shape
        n_components = stage_cfg.pca.n_components
        # More lenient minimum: allow PCA with as few as 2x target dims
        min_samples_for_pca = max(50, n_components * 2)

        if n_samples >= min_samples_for_pca:
            whitener = PCAWhitener(n_components=n_components)
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
                f"for reliable {n_components}D PCA, but only have {n_samples}. "
                f"Using camera-BN'd embeddings directly."
            )
            embeddings = raw_matrix
    else:
        embeddings = raw_matrix

    # 7. L2 normalisation
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
        f"embedding dim={embeddings.shape[1]}, "
        f"HSV dim={hsv_matrix.shape[1]}, "
        f"flip_aug={'on' if flip_augment else 'off'}, "
        f"camera_bn={'on' if stage_cfg.get('camera_bn', {}).get('enabled', True) else 'off'}"
    )

    return all_features
