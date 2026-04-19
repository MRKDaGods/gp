"""ReID model wrapper with flip augmentation and quality-weighted pooling.

Features:
- Flip-augmented feature extraction (original + horizontal flip, averaged)
- Quality-weighted temporal attention pooling for tracklet embedding
- Supports OSNet, ResNet50-IBN, TransReID (ViT/CLIP), and other torchreid architectures
- CLIP normalization for CLIP-pretrained ViT backbones
"""

from __future__ import annotations

from typing import List, Optional, Tuple, TYPE_CHECKING

import cv2
import numpy as np
import torch
from loguru import logger

if TYPE_CHECKING:
    from src.stage2_features.crop_extractor import QualityScoredCrop

# CLIP normalization stats (from OpenAI CLIP training)
_CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

# ImageNet normalization stats
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ReIDModel:
    """Wraps a ReID model for feature extraction.

    Supports OSNet, ResNet50-IBN, TransReID (ViT/CLIP), and other torchreid
    architectures.  Includes flip augmentation and quality-weighted temporal
    attention pooling.
    """

    # Model names routed to TransReID instead of torchreid
    _TRANSREID_NAMES = {"transreid", "vit_small", "vit_base", "transreid_vit", "eva02_vit"}

    def __init__(
        self,
        model_name: str = "osnet_x1_0",
        weights_path: Optional[str] = None,
        embedding_dim: int = 512,
        input_size: Tuple[int, int] = (256, 128),  # (H, W)
        device: str = "cuda:0",
        half: bool = True,
        flip_augment: bool = True,
        color_augment: bool = False,
        multiscale_sizes: Optional[List[Tuple[int, int]]] = None,
        num_cameras: int = 0,
        vit_model: str = "vit_base_patch16_clip_224.openai",
        clip_normalization: Optional[bool] = None,
        concat_patch: bool = False,
    ):
        self.model_name = model_name
        self.is_transreid = model_name.lower() in self._TRANSREID_NAMES
        self.input_size = input_size  # (H, W)
        self.embedding_dim = embedding_dim
        self.device = device
        self.half = half and "cuda" in device
        self.flip_augment = flip_augment
        self.color_augment = color_augment
        self.multiscale_sizes = multiscale_sizes or []  # additional (H,W) sizes for TTA
        self.num_cameras = num_cameras
        self.vit_model = vit_model

        # Auto-detect CLIP normalization from vit_model name if not explicit
        if clip_normalization is None:
            self.clip_normalization = self.is_transreid and "clip" in vit_model.lower()
        else:
            self.clip_normalization = clip_normalization

        # Select normalization stats
        if self.clip_normalization:
            self._norm_mean = _CLIP_MEAN
            self._norm_std = _CLIP_STD
        else:
            self._norm_mean = _IMAGENET_MEAN
            self._norm_std = _IMAGENET_STD

        # Select interpolation: BICUBIC for ViT (matches training), LINEAR for CNNs
        self._interp = cv2.INTER_CUBIC if self.is_transreid else cv2.INTER_LINEAR

        self.model = self._build_model(model_name, weights_path)
        self.model.eval()
        self.model.to(device)
        if self.half:
            self.model.half()

        # Enable CLS+GeM(patches) concatenation for TransReID
        if concat_patch and self.is_transreid:
            self.model._concat_patch = True
            self.embedding_dim = embedding_dim * 2  # 768 → 1536

        norm_tag = "CLIP" if self.clip_normalization else "ImageNet"
        interp_tag = "BICUBIC" if self._interp == cv2.INTER_CUBIC else "BILINEAR"
        logger.info(
            f"ReID model loaded: {model_name}, dim={embedding_dim}, "
            f"input={self.input_size}, norm={norm_tag}, interp={interp_tag}, "
            f"device={device}"
            + (f", multiscale={self.multiscale_sizes}" if self.multiscale_sizes else "")
        )

    def _build_model(self, model_name: str, weights_path: Optional[str]):
        """Build model and load weights.

        Routes to TransReID (ViT) or torchreid depending on ``model_name``.
        """
        if self.is_transreid:
            return self._build_transreid(weights_path)
        if model_name.lower() in {"fastreid_sbs_r50_ibn", "fastreid_r50_ibn"}:
            return self._build_fastreid_sbs_r50_ibn(weights_path)
        if model_name.lower() == "resnet101_ibn_a":
            return self._build_resnet101_ibn(weights_path)
        if model_name.lower() == "resnext101_ibn_a":
            return self._build_resnext101_ibn(weights_path)
        return self._build_torchreid(model_name, weights_path)

    @staticmethod
    def _unwrap_checkpoint_state_dict(checkpoint: object) -> dict:
        """Unwrap common checkpoint containers to a raw state dict."""
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                return checkpoint["state_dict"]
            if "model" in checkpoint:
                return checkpoint["model"]
            return checkpoint
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")

    @staticmethod
    def _remap_fastreid_sbs_r50_ibn_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Remap fast-reid SBS(R50-IBN) keys to our inference model."""
        remapped_state_dict: dict[str, torch.Tensor] = {}
        skip_prefixes = (
            "pixel_mean",
            "pixel_std",
            "heads.classifier",
        )

        for key, value in state_dict.items():
            key = key.replace("module.", "", 1)

            if key.startswith(skip_prefixes):
                continue
            if key.startswith("backbone.NL_"):
                continue

            if key == "heads.pool_layer.p":
                remapped_state_dict["pool.p"] = value
                continue

            if key.startswith("heads.bottleneck."):
                suffix = key[len("heads.bottleneck.") :]
                if suffix.startswith("0."):
                    suffix = suffix[2:]
                remapped_state_dict[f"bottleneck.{suffix}"] = value
                continue

            if key == "heads.bnneck.num_batches_tracked":
                remapped_state_dict["bottleneck.num_batches_tracked"] = value
                continue

            if key.startswith("backbone."):
                remapped_state_dict[key] = value

        return remapped_state_dict

    @staticmethod
    def _prepare_native_fastreid_sbs_r50_ibn_state_dict(
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Normalize native R50-IBN checkpoints before loading for inference."""
        prepared_state_dict: dict[str, torch.Tensor] = {}

        for key, value in state_dict.items():
            key = key.replace("module.", "", 1)

            if key.startswith("classifier"):
                continue

            if key.startswith("bn_neck."):
                key = f"bottleneck.{key[len('bn_neck.'):]}"

            prepared_state_dict[key] = value

        return prepared_state_dict

    def _build_fastreid_sbs_r50_ibn(self, weights_path: Optional[str]):
        """Build fast-reid SBS(R50-IBN-a) and return 2048D pre-BNNeck features."""
        from src.training.model import ReIDModelResNet50IBN

        model = ReIDModelResNet50IBN(
            num_classes=1,
            last_stride=1,
            pretrained=False,
            gem_p=3.0,
            eval_feature="global",
        )

        if weights_path is not None:
            try:
                checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
                state_dict = self._unwrap_checkpoint_state_dict(checkpoint)
                state_dict_keys = {key.replace("module.", "", 1) for key in state_dict}
                has_native_keys = any(
                    key in state_dict_keys
                    for key in ("bn_neck.weight", "bottleneck.weight", "pool.p")
                )
                needs_remap = not has_native_keys and (
                    "heads.pool_layer.p" in state_dict_keys
                    or any(key.startswith("heads.bottleneck.") for key in state_dict_keys)
                )

                if needs_remap:
                    state_dict = self._remap_fastreid_sbs_r50_ibn_state_dict(state_dict)
                else:
                    state_dict = self._prepare_native_fastreid_sbs_r50_ibn_state_dict(state_dict)

                missing, unexpected = model.load_state_dict(state_dict, strict=False)

                critical_missing = [
                    key for key in missing
                    if not key.startswith("classifier")
                ]
                if critical_missing:
                    logger.warning(
                        f"fast-reid SBS R50-IBN missing critical keys: {critical_missing}"
                    )
                if unexpected:
                    logger.debug(f"fast-reid SBS R50-IBN unexpected keys: {unexpected}")

                logger.info(f"Loaded fast-reid SBS R50-IBN weights from {weights_path}")
            except Exception as e:
                logger.warning(f"Could not load fast-reid SBS R50-IBN weights from {weights_path}: {e}")
                logger.warning("Using randomly initialized fast-reid SBS R50-IBN weights")

        return model

    def _build_resnet101_ibn(self, weights_path: Optional[str]):
        """Build ResNet101-IBN-a model for inference."""
        from src.training.model import ReIDModelResNet101IBN

        model = ReIDModelResNet101IBN(
            num_classes=1,
            last_stride=1,
            pretrained=False,
            gem_p=3.0,
        )

        if weights_path is not None:
            try:
                state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
                if "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                elif "model" in state_dict:
                    state_dict = state_dict["model"]

                remapped_state_dict = {}
                backbone_roots = (
                    "conv1",
                    "bn1",
                    "layer1",
                    "layer2",
                    "layer3",
                    "layer4",
                )
                for key, value in state_dict.items():
                    key = key.replace("module.", "", 1)
                    key = key.replace(".in_norm.", ".IN.").replace(".bn_norm.", ".BN.")

                    if key.startswith("classifier"):
                        continue

                    if key.startswith(backbone_roots):
                        key = f"backbone.{key}"

                    remapped_state_dict[key] = value

                state_dict = remapped_state_dict
                missing, unexpected = model.load_state_dict(state_dict, strict=False)

                critical_missing = [k for k in missing if not k.startswith("classifier")]
                if critical_missing:
                    logger.warning(
                        f"ResNet101-IBN missing critical keys: {critical_missing}"
                    )
                if unexpected:
                    logger.debug(f"ResNet101-IBN unexpected keys: {unexpected}")

                logger.info(f"Loaded ResNet101-IBN-a weights from {weights_path}")
            except Exception as e:
                logger.warning(f"Could not load ResNet101-IBN-a weights from {weights_path}: {e}")
                logger.warning("Using randomly initialized ResNet101-IBN-a weights")

        return model

    def _build_resnext101_ibn(self, weights_path: Optional[str]):
        """Build ResNeXt101-IBN-a model for inference."""
        from src.training.model import ReIDModelResNeXt101IBN

        model = ReIDModelResNeXt101IBN(
            num_classes=1,
            last_stride=1,
            pretrained=False,
            gem_p=3.0,
        )

        if weights_path is not None:
            try:
                state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
                if "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                elif "model" in state_dict:
                    state_dict = state_dict["model"]

                remapped_state_dict = {}
                backbone_roots = (
                    "conv1",
                    "bn1",
                    "layer1",
                    "layer2",
                    "layer3",
                    "layer4",
                )
                for key, value in state_dict.items():
                    key = key.replace("module.", "", 1)
                    key = key.replace(".in_norm.", ".IN.").replace(".bn_norm.", ".BN.")

                    if key.startswith("classifier"):
                        continue

                    if key.startswith(backbone_roots):
                        key = f"backbone.{key}"

                    remapped_state_dict[key] = value

                state_dict = remapped_state_dict
                missing, unexpected = model.load_state_dict(state_dict, strict=False)

                critical_missing = [k for k in missing if not k.startswith("classifier")]
                if critical_missing:
                    logger.warning(
                        f"ResNeXt101-IBN missing critical keys: {critical_missing}"
                    )
                if unexpected:
                    logger.debug(f"ResNeXt101-IBN unexpected keys: {unexpected}")

                logger.info(f"Loaded ResNeXt101-IBN-a weights from {weights_path}")
            except Exception as e:
                logger.warning(f"Could not load ResNeXt101-IBN-a weights from {weights_path}: {e}")
                logger.warning("Using randomly initialized ResNeXt101-IBN-a weights")

        return model

    def _build_transreid(self, weights_path: Optional[str]):
        """Build TransReID ViT model."""
        from src.stage2_features.transreid_model import build_transreid

        model = build_transreid(
            num_classes=1,
            num_cameras=self.num_cameras,
            embed_dim=self.embedding_dim,
            vit_model=self.vit_model,
            pretrained=weights_path is None,
            weights_path=weights_path,
            img_size=self.input_size,  # (H, W) — sets correct patch grid
        )
        return model

    def _build_torchreid(self, model_name: str, weights_path: Optional[str]):
        """Build torchreid model (OSNet, ResNet50-IBN, etc.)."""
        import torchreid

        model = torchreid.models.build_model(
            name=model_name,
            num_classes=1,  # not used for feature extraction
            loss="softmax",
            pretrained=weights_path is None,
        )

        if weights_path is not None:
            try:
                state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
                # Handle state_dict wrapped in a checkpoint
                if "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                elif "model" in state_dict:
                    state_dict = state_dict["model"]

                # Remove classifier keys for feature extraction
                # Also strip 'module.' prefix from DataParallel checkpoints
                state_dict = {
                    k.replace("module.", "", 1): v for k, v in state_dict.items()
                    if not k.replace("module.", "", 1).startswith("classifier")
                }
                model.load_state_dict(state_dict, strict=False)
                logger.info(f"Loaded ReID weights from {weights_path}")
            except Exception as e:
                logger.warning(f"Could not load weights from {weights_path}: {e}")
                logger.warning("Using pretrained ImageNet weights instead")

        return model

    def _preprocess(self, crops: List[np.ndarray]) -> torch.Tensor:
        """Preprocess crops for the ReID model.

        Args:
            crops: List of BGR uint8 numpy arrays of varying sizes.

        Returns:
            Batched tensor of shape (N, 3, H, W), normalized.
        """
        h, w = self.input_size
        processed = []

        for crop in crops:
            # BGR -> RGB
            img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            # Resize (BICUBIC for ViT to match training, BILINEAR for CNNs)
            img = cv2.resize(img, (w, h), interpolation=self._interp)
            # Normalize to [0, 1] then with model-specific stats
            img = img.astype(np.float32) / 255.0
            img = (img - self._norm_mean) / self._norm_std
            # HWC -> CHW
            img = np.transpose(img, (2, 0, 1))
            processed.append(img)

        tensor = torch.from_numpy(np.stack(processed, axis=0))
        return tensor

    def _make_cam_tensor(self, batch_size: int, cam_id: Optional[int]) -> Optional[torch.Tensor]:
        """Create a camera ID tensor for SIE if cam_id is provided and model supports it."""
        if cam_id is not None and self.is_transreid:
            return torch.full((batch_size,), cam_id, dtype=torch.long, device=self.device)
        return None

    @torch.no_grad()
    def _extract_batch(self, batch_crops: List[np.ndarray], cam_id: Optional[int] = None) -> np.ndarray:
        """Extract embeddings for a single batch with optional augmentation.

        Args:
            batch_crops: List of BGR uint8 crops.
            cam_id: Optional integer camera ID for SIE (TransReID).

        Returns:
            (N, D) float32 numpy array.
        """
        batch_tensor = self._preprocess(batch_crops).to(self.device)
        if self.half:
            batch_tensor = batch_tensor.half()
        cam_tensor = self._make_cam_tensor(len(batch_crops), cam_id)
        if cam_tensor is not None:
            features = self.model(batch_tensor, cam_ids=cam_tensor)
        else:
            features = self.model(batch_tensor)
        if isinstance(features, (tuple, list)):
            features = features[0]
        features = features.float().cpu().numpy()

        n_views = 1

        if self.flip_augment:
            flipped_crops = [cv2.flip(c, 1) for c in batch_crops]
            flip_tensor = self._preprocess(flipped_crops).to(self.device)
            if self.half:
                flip_tensor = flip_tensor.half()
            if cam_tensor is not None:
                flip_features = self.model(flip_tensor, cam_ids=cam_tensor)
            else:
                flip_features = self.model(flip_tensor)
            if isinstance(flip_features, (tuple, list)):
                flip_features = flip_features[0]
            features = features + flip_features.float().cpu().numpy()
            n_views += 1

        if self.color_augment:
            for alpha, beta in [(1.2, 15), (0.8, -10)]:
                aug_crops = [
                    cv2.convertScaleAbs(c, alpha=alpha, beta=beta)
                    for c in batch_crops
                ]
                aug_tensor = self._preprocess(aug_crops).to(self.device)
                if self.half:
                    aug_tensor = aug_tensor.half()
                if cam_tensor is not None:
                    aug_features = self.model(aug_tensor, cam_ids=cam_tensor)
                else:
                    aug_features = self.model(aug_tensor)
                if isinstance(aug_features, (tuple, list)):
                    aug_features = aug_features[0]
                features = features + aug_features.float().cpu().numpy()
                n_views += 1

        # Multi-scale TTA: resize crops to intermediate size, then back to
        # model input size.  This lets the model see "zoomed" content without
        # changing the input tensor dimensions (critical for ViT positional
        # embeddings which are fixed-size).
        if self.multiscale_sizes:
            h, w = self.input_size
            for ms_size in self.multiscale_sizes:
                ms_h, ms_w = ms_size
                # Step 1: resize to intermediate scale
                # Step 2: resize back to model input size
                ms_crops = [
                    cv2.resize(
                        cv2.resize(c, (ms_w, ms_h), interpolation=self._interp),
                        (w, h), interpolation=self._interp,
                    )
                    for c in batch_crops
                ]
                ms_tensor = self._preprocess(ms_crops).to(self.device)
                if self.half:
                    ms_tensor = ms_tensor.half()
                if cam_tensor is not None:
                    ms_features = self.model(ms_tensor, cam_ids=cam_tensor)
                else:
                    ms_features = self.model(ms_tensor)
                if isinstance(ms_features, (tuple, list)):
                    ms_features = ms_features[0]
                features = features + ms_features.float().cpu().numpy()
                n_views += 1

                # Flip TTA on multi-scale views for additional viewpoint diversity
                if self.flip_augment:
                    ms_flipped = [cv2.flip(c, 1) for c in ms_crops]
                    ms_flip_tensor = self._preprocess(ms_flipped).to(self.device)
                    if self.half:
                        ms_flip_tensor = ms_flip_tensor.half()
                    if cam_tensor is not None:
                        ms_flip_feat = self.model(ms_flip_tensor, cam_ids=cam_tensor)
                    else:
                        ms_flip_feat = self.model(ms_flip_tensor)
                    if isinstance(ms_flip_feat, (tuple, list)):
                        ms_flip_feat = ms_flip_feat[0]
                    features = features + ms_flip_feat.float().cpu().numpy()
                    n_views += 1

        return features / n_views

    @torch.no_grad()
    def extract_features(self, crops: List[np.ndarray], batch_size: int = 64, cam_id: Optional[int] = None) -> np.ndarray:
        """Extract embeddings from a list of crops with optional flip augmentation.

        When ``flip_augment`` is True, each crop is processed twice (original +
        horizontally flipped) and the two embeddings are averaged.  This is a
        standard ReID trick that improves robustness to left-right orientation.

        Args:
            crops: List of BGR uint8 crops.
            batch_size: Batch size for inference.
            cam_id: Optional integer camera ID for SIE (TransReID).

        Returns:
            (N, D) float32 numpy array of embeddings.
        """
        if not crops:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        all_embeddings = []

        for i in range(0, len(crops), batch_size):
            batch_crops = crops[i : i + batch_size]

            try:
                features = self._extract_batch(batch_crops, cam_id=cam_id)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    # CUDA OOM — retry with halved batch size
                    import torch
                    torch.cuda.empty_cache()
                    half_bs = max(1, len(batch_crops) // 2)
                    logger.warning(
                        f"CUDA OOM with batch_size={len(batch_crops)}, "
                        f"retrying with batch_size={half_bs}"
                    )
                    sub_embeddings = []
                    for j in range(0, len(batch_crops), half_bs):
                        sub_batch = batch_crops[j : j + half_bs]
                        sub_embeddings.append(self._extract_batch(sub_batch, cam_id=cam_id))
                    features = np.concatenate(sub_embeddings, axis=0)
                else:
                    raise

            all_embeddings.append(features)

        embeddings = np.concatenate(all_embeddings, axis=0)
        return embeddings

    def get_tracklet_embedding(
        self,
        crops: List[np.ndarray],
        quality_scores: Optional[List[float]] = None,
        cam_id: Optional[int] = None,
        quality_temperature: float = 3.0,
    ) -> Optional[np.ndarray]:
        """Extract a single embedding for a tracklet using quality-weighted attention.

        Instead of naive mean pooling, weights each crop's embedding by its
        quality score (from :class:`QualityScoredCrop`), producing an embedding
        biased toward sharper, larger, higher-confidence crops.

        Args:
            crops: List of BGR uint8 crops from the tracklet.
            quality_scores: Per-crop quality scores in [0, 1]. If None,
                falls back to uniform weighting (simple average).
            cam_id: Optional integer camera ID for SIE (TransReID).
            quality_temperature: Exponent for softmax quality weighting.
                Higher = sharper (more weight on best crops). 0 = uniform.

        Returns:
            (D,) embedding vector, or None if no valid crops.
        """
        if not crops:
            return None

        embeddings = self.extract_features(crops, cam_id=cam_id)
        if embeddings.shape[0] == 0:
            return None

        if quality_scores is not None and len(quality_scores) == embeddings.shape[0]:
            weights = np.array(quality_scores, dtype=np.float32)
            # Softmax-style temperature scaling to sharpen attention
            weights = np.exp(weights * quality_temperature)
            weights = weights / weights.sum()
            weighted_embedding = (embeddings * weights[:, np.newaxis]).sum(axis=0)
            return weighted_embedding
        else:
            return embeddings.mean(axis=0)

    def get_tracklet_embedding_from_scored_crops(
        self,
        scored_crops: List["QualityScoredCrop"],
        cam_id: Optional[int] = None,
        quality_temperature: float = 3.0,
    ) -> Optional[np.ndarray]:
        """Convenience wrapper that accepts QualityScoredCrop objects directly.

        Args:
            scored_crops: List of QualityScoredCrop from CropExtractor.
            cam_id: Optional integer camera ID for SIE (TransReID).
            quality_temperature: Exponent for softmax quality weighting.

        Returns:
            (D,) quality-weighted embedding, or None.
        """
        if not scored_crops:
            return None
        crops = [sc.image for sc in scored_crops]
        qualities = [sc.quality for sc in scored_crops]
        return self.get_tracklet_embedding(crops, quality_scores=qualities, cam_id=cam_id, quality_temperature=quality_temperature)

    def get_tracklet_multi_query_embeddings(
        self,
        scored_crops: List["QualityScoredCrop"],
        k: int = 5,
        cam_id: Optional[int] = None,
        quality_temperature: float = 3.0,
    ) -> Optional[np.ndarray]:
        """Extract top-K representative crop embeddings for a tracklet.

        Selection is quality-based: the highest-quality crops are retained.
        If the tracklet has fewer than K crops, the remaining rows are padded
        with the tracklet's pooled embedding so Stage 4 can load a dense
        ``(N, K, D)`` tensor without special-casing variable-length tracks.
        """
        if not scored_crops or k <= 0:
            return None

        crops = [sc.image for sc in scored_crops]
        qualities = np.array([sc.quality for sc in scored_crops], dtype=np.float32)
        embeddings = self.extract_features(crops, cam_id=cam_id)
        if embeddings.shape[0] == 0:
            return None

        top_count = min(k, embeddings.shape[0])
        top_indices = np.argsort(-qualities)[:top_count]
        selected = embeddings[top_indices].astype(np.float32)

        if selected.shape[0] < k:
            if qualities.shape[0] == embeddings.shape[0]:
                weights = np.exp(qualities * quality_temperature)
                weights = weights / max(weights.sum(), 1e-8)
                pooled = (embeddings * weights[:, np.newaxis]).sum(axis=0, keepdims=True)
            else:
                pooled = embeddings.mean(axis=0, keepdims=True)
            pad = np.repeat(pooled.astype(np.float32), k - selected.shape[0], axis=0)
            selected = np.concatenate([selected, pad], axis=0)

        return selected.astype(np.float32)

    # ------------------------------------------------------------------
    # Camera-specific Test-Time Adaptation (CamTTA)
    # ------------------------------------------------------------------

    def save_bn_state(self) -> dict:
        """Save BNNeck running statistics for later restoration.

        Returns an opaque dict that can be passed to :meth:`restore_bn_state`.
        Returns an empty dict for non-TransReID models.
        """
        if not self.is_transreid or not hasattr(self.model, "bn"):
            return {}
        bn = self.model.bn
        return {
            "running_mean": bn.running_mean.clone(),
            "running_var": bn.running_var.clone(),
            "num_batches_tracked": bn.num_batches_tracked.clone(),
            "momentum": bn.momentum,
        }

    def restore_bn_state(self, state: dict) -> None:
        """Restore BNNeck running statistics from a previously saved state.

        Args:
            state: Dict returned by :meth:`save_bn_state`.
        """
        if not state or not self.is_transreid or not hasattr(self.model, "bn"):
            return
        bn = self.model.bn
        bn.running_mean.copy_(state["running_mean"])
        bn.running_var.copy_(state["running_var"])
        bn.num_batches_tracked.copy_(state["num_batches_tracked"])
        bn.momentum = state.get("momentum", 0.1)

    def warmup_camera_bn(self, crops: List[np.ndarray], batch_size: int = 64) -> None:
        """Adapt BNNeck statistics to a specific camera's appearance distribution.

        Runs all provided crops through the BNNeck in training mode (updating
        ``running_mean`` / ``running_var``), then restores eval mode.  After
        calling this, subsequent ``extract_features`` calls will use the
        camera-adapted statistics for BN normalisation.

        Uses cumulative moving average (``momentum=None``) so the running
        statistics converge to the exact per-camera batch statistics after
        all warmup crops are processed, producing a stable estimate regardless
        of the number of crops.

        Only applies to TransReID models (BNNeck is ``BatchNorm1d``).  Returns
        immediately for other architectures.

        Args:
            crops: All BGR uint8 crops available for this camera.  More crops
                produce a more accurate per-camera BN estimate.
            batch_size: Forward-pass batch size for the warmup loop.
        """
        if not self.is_transreid or not hasattr(self.model, "bn") or len(crops) < 2:
            return

        bn = self.model.bn
        orig_momentum = bn.momentum

        # Use CMA (momentum=None): running stats will exactly equal the
        # cumulative mean/var of all warmup samples, regardless of batch count.
        bn.momentum = None
        bn.reset_running_stats()
        bn.training = True  # enable running stat updates

        with torch.no_grad():
            for start in range(0, len(crops), batch_size):
                batch = crops[start : start + batch_size]
                if len(batch) < 2:
                    continue  # BatchNorm1d requires batch_size >= 2 in training mode
                t = self._preprocess(batch).to(self.device)
                if self.half:
                    t = t.half()
                # Full forward pass: ViT backbone → BNNeck (updates running stats)
                _ = self.model(t)

        # Restore momentum and switch back to eval (locks in the adapted stats)
        bn.momentum = orig_momentum
        bn.training = False
