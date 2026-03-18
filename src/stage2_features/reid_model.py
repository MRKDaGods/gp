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
    _TRANSREID_NAMES = {"transreid", "vit_small", "vit_base", "transreid_vit"}

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
        return self._build_torchreid(model_name, weights_path)

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
            weights = np.exp(weights * 3.0)  # temperature = 1/3
            weights = weights / weights.sum()
            weighted_embedding = (embeddings * weights[:, np.newaxis]).sum(axis=0)
            return weighted_embedding
        else:
            return embeddings.mean(axis=0)

    def get_tracklet_embedding_from_scored_crops(
        self,
        scored_crops: List["QualityScoredCrop"],
        cam_id: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        """Convenience wrapper that accepts QualityScoredCrop objects directly.

        Args:
            scored_crops: List of QualityScoredCrop from CropExtractor.
            cam_id: Optional integer camera ID for SIE (TransReID).

        Returns:
            (D,) quality-weighted embedding, or None.
        """
        if not scored_crops:
            return None
        crops = [sc.image for sc in scored_crops]
        qualities = [sc.quality for sc in scored_crops]
        return self.get_tracklet_embedding(crops, quality_scores=qualities, cam_id=cam_id)
