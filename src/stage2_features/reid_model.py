"""ReID model wrapper for feature extraction using torchreid."""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from loguru import logger


class ReIDModel:
    """Wraps a torchreid ReID model for feature extraction.

    Supports OSNet, ResNet50-IBN, and other torchreid architectures.
    """

    def __init__(
        self,
        model_name: str = "osnet_x1_0",
        weights_path: Optional[str] = None,
        embedding_dim: int = 512,
        input_size: Tuple[int, int] = (256, 128),  # (H, W)
        device: str = "cuda:0",
        half: bool = True,
    ):
        self.model_name = model_name
        self.input_size = input_size  # (H, W)
        self.embedding_dim = embedding_dim
        self.device = device
        self.half = half and "cuda" in device

        self.model = self._build_model(model_name, weights_path)
        self.model.eval()
        self.model.to(device)
        if self.half:
            self.model.half()

        logger.info(
            f"ReID model loaded: {model_name}, dim={embedding_dim}, "
            f"input={input_size}, device={device}"
        )

    def _build_model(self, model_name: str, weights_path: Optional[str]):
        """Build model using torchreid and load weights."""
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
            # Resize
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
            # Normalize to [0, 1] then with ImageNet stats
            img = img.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img = (img - mean) / std
            # HWC -> CHW
            img = np.transpose(img, (2, 0, 1))
            processed.append(img)

        tensor = torch.from_numpy(np.stack(processed, axis=0))
        return tensor

    @torch.no_grad()
    def extract_features(self, crops: List[np.ndarray], batch_size: int = 64) -> np.ndarray:
        """Extract embeddings from a list of crops.

        Args:
            crops: List of BGR uint8 crops.
            batch_size: Batch size for inference.

        Returns:
            (N, D) float32 numpy array of embeddings.
        """
        if not crops:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        all_embeddings = []

        for i in range(0, len(crops), batch_size):
            batch_crops = crops[i : i + batch_size]
            batch_tensor = self._preprocess(batch_crops).to(self.device)
            if self.half:
                batch_tensor = batch_tensor.half()

            features = self.model(batch_tensor)

            # Handle models that return tuple (features, classifier_output)
            if isinstance(features, (tuple, list)):
                features = features[0]

            features = features.float().cpu().numpy()
            all_embeddings.append(features)

        embeddings = np.concatenate(all_embeddings, axis=0)
        return embeddings

    def get_tracklet_embedding(self, crops: List[np.ndarray]) -> Optional[np.ndarray]:
        """Extract a single embedding for a tracklet by averaging crop embeddings.

        Args:
            crops: List of crops from the tracklet.

        Returns:
            (D,) embedding vector, or None if no valid crops.
        """
        if not crops:
            return None

        embeddings = self.extract_features(crops)
        if embeddings.shape[0] == 0:
            return None

        # Average pool across crops
        mean_embedding = embeddings.mean(axis=0)
        return mean_embedding
