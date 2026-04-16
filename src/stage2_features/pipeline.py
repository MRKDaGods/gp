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
from omegaconf import DictConfig, OmegaConf

from src.core.constants import PERSON_CLASSES, VEHICLE_CLASSES
from src.core.data_models import Tracklet, TrackletFeatures
from src.core.io_utils import save_embeddings, save_hsv_features, save_multi_query_embeddings
from src.stage2_features.crop_extractor import CropExtractor
from src.stage2_features.embeddings import camera_aware_batch_normalize, l2_normalize
from src.stage2_features.foreground_masker import ForegroundMasker
from src.stage2_features.hsv_extractor import HSVExtractor
from src.stage2_features.pca_whitening import PCAWhitener
from src.stage2_features.reid_model import ReIDModel


def _flatten_multi_query_embeddings(
    features: List[TrackletFeatures],
) -> tuple[Optional[np.ndarray], List[int], List[str]]:
    """Flatten per-tracklet multi-query arrays into a single matrix."""
    flattened: List[np.ndarray] = []
    sizes: List[int] = []
    camera_ids: List[str] = []

    for feat in features:
        mq = feat.multi_query_embeddings
        if mq is None:
            continue
        flattened.append(mq)
        sizes.append(mq.shape[0])
        camera_ids.extend([feat.camera_id] * mq.shape[0])

    if not flattened:
        return None, [], []

    return np.concatenate(flattened, axis=0), sizes, camera_ids


def _restore_multi_query_embeddings(
    features: List[TrackletFeatures],
    mq_flat: np.ndarray,
    sizes: List[int],
) -> None:
    """Restore flattened multi-query rows back onto each tracklet feature."""
    offset = 0
    size_idx = 0
    for feat in features:
        if feat.multi_query_embeddings is None:
            continue
        size = sizes[size_idx]
        feat.multi_query_embeddings = mq_flat[offset:offset + size]
        offset += size
        size_idx += 1


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

    foreground_masker: Optional[ForegroundMasker] = None
    foreground_mask_cfg = stage_cfg.crop.get("foreground_masking", {})
    if foreground_mask_cfg.get("enabled", False):
        foreground_masker = ForegroundMasker(
            model_name=foreground_mask_cfg.get("model_name", "facebook/sam2.1-hiera-tiny"),
            min_crop_size=foreground_mask_cfg.get("min_crop_size", 48),
            fill_value=foreground_mask_cfg.get("fill_value", "mean"),
            device=stage_cfg.reid.device,
        )

    hsv_extractor = HSVExtractor(
        h_bins=stage_cfg.hsv.h_bins,
        s_bins=stage_cfg.hsv.s_bins,
        v_bins=stage_cfg.hsv.v_bins,
        n_stripes=stage_cfg.hsv.get("n_stripes", 3),
    )

    flip_augment = stage_cfg.reid.get("flip_augment", True)
    color_augment = stage_cfg.reid.get("color_augment", False)
    multiscale_raw = stage_cfg.reid.get("multiscale_sizes", [])
    multiscale_sizes = [tuple(s) for s in multiscale_raw] if multiscale_raw else []
    quality_temperature = float(stage_cfg.reid.get("quality_temperature", 3.0))
    multi_query_k = int(stage_cfg.get("multi_query", {}).get("k", 0))
    target_classes = OmegaConf.select(cfg, "dataset.target_classes", default=None)
    if target_classes is None:
        target_classes = OmegaConf.select(cfg, "stage0.target_classes", default=None)
    has_person_classes = target_classes is None or any(
        class_id in PERSON_CLASSES for class_id in target_classes
    )

    # --- Load ReID models (person and vehicle) ---
    person_reid: Optional[ReIDModel] = None
    if has_person_classes:
        person_reid = ReIDModel(
            model_name=stage_cfg.reid.person.model_name,
            weights_path=stage_cfg.reid.person.weights_path,
            embedding_dim=stage_cfg.reid.person.embedding_dim,
            input_size=tuple(stage_cfg.reid.person.input_size),
            device=stage_cfg.reid.device,
            half=stage_cfg.reid.half,
            flip_augment=flip_augment,
            color_augment=color_augment,
            multiscale_sizes=multiscale_sizes,
            num_cameras=stage_cfg.reid.person.get("num_cameras", 0),
            vit_model=stage_cfg.reid.person.get("vit_model", "vit_base_patch16_clip_224.openai"),
            clip_normalization=stage_cfg.reid.person.get("clip_normalization", None),
        )

    _vehicle_weights = stage_cfg.reid.vehicle.weights_path
    _vehicle_fallback = stage_cfg.reid.vehicle.get("weights_fallback")
    if _vehicle_fallback and not Path(_vehicle_weights).exists() and Path(_vehicle_fallback).exists():
        logger.warning(
            f"Primary vehicle weights not found: {_vehicle_weights}. "
            f"Using fallback: {_vehicle_fallback}"
        )
        _vehicle_weights = _vehicle_fallback
    vehicle_reid = ReIDModel(
        model_name=stage_cfg.reid.vehicle.model_name,
        weights_path=_vehicle_weights,
        embedding_dim=stage_cfg.reid.vehicle.embedding_dim,
        input_size=tuple(stage_cfg.reid.vehicle.input_size),
        device=stage_cfg.reid.device,
        half=stage_cfg.reid.half,
        flip_augment=flip_augment,
        color_augment=color_augment,
        multiscale_sizes=multiscale_sizes,
        num_cameras=stage_cfg.reid.vehicle.get("num_cameras", 0),
        vit_model=stage_cfg.reid.vehicle.get("vit_model", "vit_base_patch16_clip_224.openai"),
        clip_normalization=stage_cfg.reid.vehicle.get("clip_normalization", None),
        concat_patch=stage_cfg.reid.vehicle.get("concat_patch", False),
    )

    # --- Optional second vehicle ReID model for ensemble (concatenated features) ---
    # SOTA: ensemble of TransReID (domain-specific fine-tuned) + OSNet (general, fast)
    # produces complementary features → improved recall on hard cases (occlusion, viewpoint).
    vehicle_reid2: Optional[ReIDModel] = None
    vehicle2_cfg = stage_cfg.reid.get("vehicle2", {})
    if vehicle2_cfg.get("enabled", False):
        weights_path2 = vehicle2_cfg.get("weights_path")
        if weights_path2 and Path(weights_path2).exists():
            vehicle_reid2 = ReIDModel(
                model_name=vehicle2_cfg.get("model_name", "osnet_x1_0"),
                weights_path=weights_path2,
                embedding_dim=vehicle2_cfg.get("embedding_dim", 512),
                input_size=tuple(vehicle2_cfg.get("input_size", [256, 128])),
                device=stage_cfg.reid.device,
                half=stage_cfg.reid.half,
                flip_augment=flip_augment,
                color_augment=color_augment,
                num_cameras=vehicle2_cfg.get("num_cameras", 0),
                clip_normalization=vehicle2_cfg.get("clip_normalization", False),
            )
            logger.info(
                f"Ensemble ReID enabled: primary={stage_cfg.reid.vehicle.model_name} "
                f"+ secondary={vehicle2_cfg.get('model_name', 'osnet_x1_0')}"
            )
        else:
            logger.warning(
                f"Ensemble vehicle2 weights not found: {weights_path2}. "
                "Falling back to single-model."
            )

    vehicle_reid3: Optional[ReIDModel] = None
    vehicle3_cfg = stage_cfg.reid.get("vehicle3", {})
    if vehicle3_cfg.get("enabled", False):
        weights_path3 = vehicle3_cfg.get("weights_path")
        if weights_path3 and Path(weights_path3).exists():
            vehicle_reid3 = ReIDModel(
                model_name=vehicle3_cfg.get("model_name", "resnext101_ibn_a"),
                weights_path=weights_path3,
                embedding_dim=vehicle3_cfg.get("embedding_dim", 2048),
                input_size=tuple(vehicle3_cfg.get("input_size", [384, 384])),
                device=stage_cfg.reid.device,
                half=stage_cfg.reid.half,
                flip_augment=flip_augment,
                color_augment=color_augment,
                num_cameras=vehicle3_cfg.get("num_cameras", 0),
                clip_normalization=vehicle3_cfg.get("clip_normalization", False),
            )
            logger.info(
                f"Tertiary ensemble ReID enabled: {vehicle3_cfg.get('model_name', 'resnext101_ibn_a')}"
            )
        else:
            logger.warning(
                f"Ensemble vehicle3 weights not found: {weights_path3}. "
                "Falling back to up-to-2-model extraction."
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
    all_secondary_embeddings: List[Optional[np.ndarray]] = []
    all_tertiary_embeddings: List[Optional[np.ndarray]] = []
    all_camera_ids: List[str] = []
    index_map: List[dict] = []
    vehicle2_separate = vehicle2_cfg.get("save_separate", False) and vehicle_reid2 is not None
    vehicle3_separate = vehicle3_cfg.get("save_separate", True) and vehicle_reid3 is not None

    # --- SIE camera ID mapping for TransReID models ---
    sie_camera_map: Dict[str, int] = stage_cfg.reid.vehicle.get("sie_camera_map", {}) or {}
    if sie_camera_map:
        logger.info(f"SIE camera map: {dict(sie_camera_map)}")

    # --- Camera-specific Test-Time Adaptation (CamTTA) ---
    # Adapt the BNNeck running statistics to each camera before feature extraction.
    # Requires disk frames (use_disk_frames) so we can pre-collect all crops.
    tta_cfg = stage_cfg.get("camera_tta", {})
    camera_tta_enabled = tta_cfg.get("enabled", False) and use_disk_frames

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

        # --- CamTTA: pre-collect crops and warm up BNNeck for this camera ---
        # When enabled, extract ALL crops for this camera first to build a
        # camera-representative sample, then warm up the BNNeck, and then
        # extract the final embeddings using the adapted statistics.
        scored_crops_cache: Optional[List] = None
        if camera_tta_enabled and cam_frame_images is not None:
            all_cam_crops_for_tta = []
            scored_crops_per_tracklet = []
            for tracklet in tracklets:
                sc = crop_extractor.extract_crops_from_frames(tracklet, cam_frame_images)
                if foreground_masker is not None and tracklet.class_id in VEHICLE_CLASSES:
                    foreground_masker.mask_crops(sc)
                scored_crops_per_tracklet.append(sc)
                all_cam_crops_for_tta.extend([c.image for c in sc])

            if all_cam_crops_for_tta:
                orig_state = vehicle_reid.save_bn_state()
                vehicle_reid.warmup_camera_bn(all_cam_crops_for_tta)
                logger.info(
                    f"  CamTTA: warmed up BNNeck with {len(all_cam_crops_for_tta)} crops "
                    f"for camera {camera_id}"
                )
            scored_crops_cache = scored_crops_per_tracklet

        dropped_no_crops = 0
        dropped_no_embedding = 0

        tracklet_iter = (
            zip(tracklets, scored_crops_cache)
            if scored_crops_cache is not None
            else ((t, None) for t in tracklets)
        )

        for tracklet, preloaded_scored_crops in tracklet_iter:
            # 1. Quality-aware crop selection
            if preloaded_scored_crops is not None:
                scored_crops = preloaded_scored_crops
            elif cam_frame_images is not None:
                scored_crops = crop_extractor.extract_crops_from_frames(
                    tracklet, cam_frame_images,
                )
            else:
                scored_crops = crop_extractor.extract_crops(tracklet, video_path)
            if not scored_crops:
                dropped_no_crops += 1
                continue

            if (
                foreground_masker is not None
                and preloaded_scored_crops is None
                and tracklet.class_id in VEHICLE_CLASSES
            ):
                foreground_masker.mask_crops(scored_crops)

            # Select ReID model based on class
            if tracklet.class_id in PERSON_CLASSES:
                reid = person_reid
                reid2 = None
                reid3 = None
                sie_cam_id = None  # Person model doesn't use SIE camera map
            else:
                reid = vehicle_reid
                reid2 = vehicle_reid2
                reid3 = vehicle_reid3
                sie_cam_id = sie_camera_map.get(camera_id)

            # 2 & 3. Flip-augmented extraction + quality-weighted attention pooling
            raw_embedding = reid.get_tracklet_embedding_from_scored_crops(scored_crops, cam_id=sie_cam_id, quality_temperature=quality_temperature)
            if raw_embedding is None:
                dropped_no_embedding += 1
                continue

            mq_embeddings = None
            if multi_query_k > 0:
                mq_embeddings = reid.get_tracklet_multi_query_embeddings(
                    scored_crops,
                    k=multi_query_k,
                    cam_id=sie_cam_id,
                    quality_temperature=quality_temperature,
                )

            # Ensemble: extract from second model
            raw_embedding2 = None
            if reid2 is not None:
                raw_embedding2 = reid2.get_tracklet_embedding_from_scored_crops(scored_crops, cam_id=sie_cam_id, quality_temperature=quality_temperature)
                if raw_embedding2 is not None:
                    if vehicle2_separate:
                        # Save separately for stage4 fusion
                        pass
                    else:
                        # Legacy: concatenate into single vector
                        norm1 = np.linalg.norm(raw_embedding)
                        norm2 = np.linalg.norm(raw_embedding2)
                        e1 = raw_embedding / max(norm1, 1e-8)
                        e2 = raw_embedding2 / max(norm2, 1e-8)
                        raw_embedding = np.concatenate([e1, e2], axis=0)

            raw_embedding3 = None
            if reid3 is not None:
                raw_embedding3 = reid3.get_tracklet_embedding_from_scored_crops(scored_crops, cam_id=sie_cam_id, quality_temperature=quality_temperature)
                if raw_embedding3 is not None and not vehicle3_separate:
                    norm_current = np.linalg.norm(raw_embedding)
                    norm3 = np.linalg.norm(raw_embedding3)
                    current_embedding = raw_embedding / max(norm_current, 1e-8)
                    e3 = raw_embedding3 / max(norm3, 1e-8)
                    raw_embedding = np.concatenate([current_embedding, e3], axis=0)

            # 4. Spatial HSV histogram with quality weighting
            hsv_hist = hsv_extractor.extract_tracklet_histogram_from_scored_crops(
                scored_crops
            )

            all_raw_embeddings.append(raw_embedding)
            all_camera_ids.append(camera_id)
            if vehicle2_separate:
                all_secondary_embeddings.append(raw_embedding2)
            if vehicle3_separate:
                all_tertiary_embeddings.append(raw_embedding3)
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
                    multi_query_embeddings=mq_embeddings,
                )
            )

        # Per-camera drop summary for forensic audit trail
        total_cam = len(tracklets)
        extracted = total_cam - dropped_no_crops - dropped_no_embedding
        if dropped_no_crops or dropped_no_embedding:
            logger.warning(
                f"  {camera_id}: {extracted}/{total_cam} tracklets extracted, "
                f"{dropped_no_crops} dropped (no crops), "
                f"{dropped_no_embedding} dropped (no embedding)"
            )

        # Restore original BN state after processing this camera so the next
        # camera starts from the same baseline (not from this camera's stats).
        if camera_tta_enabled and all_cam_crops_for_tta:
            vehicle_reid.restore_bn_state(orig_state)

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

        mq_flat, mq_sizes, mq_camera_ids = _flatten_multi_query_embeddings(all_features)
        if mq_flat is not None:
            mq_flat = camera_aware_batch_normalize(mq_flat, mq_camera_ids)
            _restore_multi_query_embeddings(all_features, mq_flat, mq_sizes)

    # 5b. Power normalization (before PCA to compress outlier magnitudes)
    pn_alpha = stage_cfg.get("power_norm", {}).get("alpha", 0.0)
    if pn_alpha > 0:
        logger.info(f"Applying power normalization (alpha={pn_alpha})")
        raw_matrix = np.sign(raw_matrix) * np.abs(raw_matrix) ** pn_alpha
        norms = np.linalg.norm(raw_matrix, axis=1, keepdims=True)
        raw_matrix = raw_matrix / np.maximum(norms, 1e-8)

        mq_flat, mq_sizes, _ = _flatten_multi_query_embeddings(all_features)
        if mq_flat is not None:
            mq_flat = np.sign(mq_flat) * np.abs(mq_flat) ** pn_alpha
            mq_norms = np.linalg.norm(mq_flat, axis=1, keepdims=True)
            mq_flat = mq_flat / np.maximum(mq_norms, 1e-8)
            _restore_multi_query_embeddings(all_features, mq_flat, mq_sizes)

    # 6. PCA whitening
    if stage_cfg.pca.enabled:
        n_samples, n_features = raw_matrix.shape
        n_components = stage_cfg.pca.n_components
        # Hard minimum: n_samples must exceed n_components to form a valid covariance.
        # Use n_components itself (the mathematical minimum), no extra factor.
        min_samples_for_pca = max(50, n_components)

        if n_samples >= min_samples_for_pca:
            whitener = PCAWhitener(n_components=n_components)
            pca_path = stage_cfg.pca.pca_model_path

            if Path(pca_path).exists():
                whitener.load(pca_path)
                # Validate loaded PCA matches requested n_components
                if whitener.n_components != n_components:
                    logger.warning(
                        f"PCA model has {whitener.n_components} components "
                        f"but config requests {n_components}. Refitting."
                    )
                    whitener = PCAWhitener(n_components=n_components)
                    whitener.fit(raw_matrix)
                    whitener.save(pca_path)
                    logger.info(f"Refitted and saved PCA model to {pca_path}")
                else:
                    logger.info(f"Loaded PCA model from {pca_path}")
            else:
                whitener.fit(raw_matrix)
                whitener.save(pca_path)
                logger.info(f"Fitted and saved PCA model to {pca_path}")

            embeddings = whitener.transform(raw_matrix)

            mq_flat, mq_sizes, _ = _flatten_multi_query_embeddings(all_features)
            if mq_flat is not None:
                mq_flat = whitener.transform(mq_flat)
                _restore_multi_query_embeddings(all_features, mq_flat, mq_sizes)
        else:
            logger.warning(
                f"Skipping PCA: need at least {min_samples_for_pca} samples "
                f"for reliable {n_components}D PCA, but only have {n_samples}. "
                f"Using camera-BN'd embeddings directly."
            )
            embeddings = raw_matrix
    else:
        embeddings = raw_matrix

    # 6b. Separate PCA whitening for secondary embeddings (score-level fusion).
    sec_matrix: Optional[np.ndarray] = None
    if vehicle2_separate and all_secondary_embeddings:
        sec_valid = [embedding for embedding in all_secondary_embeddings if embedding is not None]
        if sec_valid:
            sec_matrix = np.stack(sec_valid, axis=0)
            logger.info(f"Secondary embedding matrix: {sec_matrix.shape}")

            if stage_cfg.get("camera_bn", {}).get("enabled", True):
                sec_camera_ids = [
                    all_camera_ids[i]
                    for i, embedding in enumerate(all_secondary_embeddings)
                    if embedding is not None
                ]
                logger.info("Applying camera-aware batch normalisation to secondary embeddings")
                sec_matrix = camera_aware_batch_normalize(sec_matrix, sec_camera_ids)

            if stage_cfg.pca.enabled:
                n_sec, d_sec = sec_matrix.shape
                sec_components = min(int(stage_cfg.pca.n_components), d_sec)
                sec_min_samples = max(50, sec_components)
                sec_pca_path = stage_cfg.pca.get(
                    "secondary_pca_model_path",
                    "models/reid/pca_transform_secondary.pkl",
                )

                if n_sec >= sec_min_samples:
                    sec_whitener = PCAWhitener(n_components=sec_components)
                    if Path(sec_pca_path).exists():
                        sec_whitener.load(sec_pca_path)
                        if sec_whitener.n_components != sec_components:
                            logger.warning(
                                f"Secondary PCA model has {sec_whitener.n_components} components "
                                f"but config requests {sec_components}. Refitting."
                            )
                            sec_whitener = PCAWhitener(n_components=sec_components)
                            sec_whitener.fit(sec_matrix)
                            sec_whitener.save(sec_pca_path)
                            logger.info(f"Refitted and saved secondary PCA model to {sec_pca_path}")
                        else:
                            logger.info(f"Loaded secondary PCA model from {sec_pca_path}")
                    else:
                        sec_whitener.fit(sec_matrix)
                        sec_whitener.save(sec_pca_path)
                        logger.info(f"Fitted and saved secondary PCA model to {sec_pca_path}")

                    sec_matrix = sec_whitener.transform(sec_matrix)
                    logger.info(f"Secondary PCA: {d_sec}D -> {sec_components}D")
                else:
                    logger.warning(
                        f"Skipping secondary PCA: need at least {sec_min_samples} samples "
                        f"for reliable {sec_components}D PCA, but only have {n_sec}."
                    )

    tert_matrix: Optional[np.ndarray] = None
    if vehicle3_separate and all_tertiary_embeddings:
        tert_valid = [embedding for embedding in all_tertiary_embeddings if embedding is not None]
        if tert_valid:
            tert_matrix = np.stack(tert_valid, axis=0)
            logger.info(f"Tertiary embedding matrix: {tert_matrix.shape}")

            if stage_cfg.get("camera_bn", {}).get("enabled", True):
                tert_camera_ids = [
                    all_camera_ids[i]
                    for i, embedding in enumerate(all_tertiary_embeddings)
                    if embedding is not None
                ]
                logger.info("Applying camera-aware batch normalisation to tertiary embeddings")
                tert_matrix = camera_aware_batch_normalize(tert_matrix, tert_camera_ids)

            if stage_cfg.pca.enabled:
                n_tert, d_tert = tert_matrix.shape
                tert_components = min(int(stage_cfg.pca.n_components), d_tert)
                tert_min_samples = max(50, tert_components)
                tert_pca_path = stage_cfg.pca.get(
                    "tertiary_pca_model_path",
                    "models/reid/pca_transform_tertiary.pkl",
                )

                if n_tert >= tert_min_samples:
                    tert_whitener = PCAWhitener(n_components=tert_components)
                    if Path(tert_pca_path).exists():
                        tert_whitener.load(tert_pca_path)
                        if tert_whitener.n_components != tert_components:
                            logger.warning(
                                f"Tertiary PCA model has {tert_whitener.n_components} components "
                                f"but config requests {tert_components}. Refitting."
                            )
                            tert_whitener = PCAWhitener(n_components=tert_components)
                            tert_whitener.fit(tert_matrix)
                            tert_whitener.save(tert_pca_path)
                            logger.info(f"Refitted and saved tertiary PCA model to {tert_pca_path}")
                        else:
                            logger.info(f"Loaded tertiary PCA model from {tert_pca_path}")
                    else:
                        tert_whitener.fit(tert_matrix)
                        tert_whitener.save(tert_pca_path)
                        logger.info(f"Fitted and saved tertiary PCA model to {tert_pca_path}")

                    tert_matrix = tert_whitener.transform(tert_matrix)
                    logger.info(f"Tertiary PCA: {d_tert}D -> {tert_components}D")
                else:
                    logger.warning(
                        f"Skipping tertiary PCA: need at least {tert_min_samples} samples "
                        f"for reliable {tert_components}D PCA, but only have {n_tert}."
                    )

    # 7. L2 normalisation
    embeddings = l2_normalize(embeddings)

    mq_flat, mq_sizes, _ = _flatten_multi_query_embeddings(all_features)
    if mq_flat is not None:
        mq_flat = l2_normalize(mq_flat)
        _restore_multi_query_embeddings(all_features, mq_flat, mq_sizes)

    # Update features with refined embeddings
    for i, feat in enumerate(all_features):
        feat.embedding = embeddings[i]
        if multi_query_k > 0 and feat.multi_query_embeddings is None:
            feat.multi_query_embeddings = np.repeat(
                embeddings[i][np.newaxis, :],
                multi_query_k,
                axis=0,
            ).astype(np.float32)

    # Save outputs
    hsv_matrix = np.stack([f.hsv_histogram for f in all_features], axis=0)
    save_embeddings(embeddings, index_map, output_dir)
    save_hsv_features(hsv_matrix, output_dir)
    if multi_query_k > 0:
        save_multi_query_embeddings(
            [f.multi_query_embeddings for f in all_features if f.multi_query_embeddings is not None],
            output_dir,
        )

    # Save secondary model embeddings separately (for stage4 fusion)
    if vehicle2_separate and sec_matrix is not None:
        sec_matrix = l2_normalize(sec_matrix)
        sec_path = output_dir / "embeddings_secondary.npy"
        np.save(sec_path, sec_matrix.astype(np.float32))
        logger.info(f"Secondary embeddings saved: {sec_matrix.shape} -> {sec_path}")
    elif vehicle2_separate and all_secondary_embeddings:
        valid_secondary = sum(embedding is not None for embedding in all_secondary_embeddings)
        logger.warning(
            f"Secondary embeddings incomplete: {valid_secondary}/{len(all_secondary_embeddings)}"
        )

    if vehicle3_separate and tert_matrix is not None:
        tert_matrix = l2_normalize(tert_matrix)
        tert_path = output_dir / "embeddings_tertiary.npy"
        np.save(tert_path, tert_matrix.astype(np.float32))
        logger.info(f"Tertiary embeddings saved: {tert_matrix.shape} -> {tert_path}")
    elif vehicle3_separate and all_tertiary_embeddings:
        valid_tertiary = sum(embedding is not None for embedding in all_tertiary_embeddings)
        logger.warning(
            f"Tertiary embeddings incomplete: {valid_tertiary}/{len(all_tertiary_embeddings)}"
        )

    mq_log = f"multi_query_k={multi_query_k}, " if multi_query_k > 0 else ""
    logger.info(
        f"Stage 2 complete: {len(all_features)} tracklet features, "
        f"embedding dim={embeddings.shape[1]}, "
        f"HSV dim={hsv_matrix.shape[1]}, "
        f"flip_aug={'on' if flip_augment else 'off'}, "
        f"{mq_log}"
        f"camera_bn={'on' if stage_cfg.get('camera_bn', {}).get('enabled', True) else 'off'}, "
        f"foreground_masking={'on' if foreground_masker is not None else 'off'}"
    )

    return all_features
