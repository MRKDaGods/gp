"""Spatial HSV color histogram extraction for tracklet crops.

Implements horizontal-stripe spatial histograms: the crop is divided into
*n_stripes* horizontal bands (e.g. 3 for head / torso / legs) and a
separate HSV histogram is computed for each band.  The concatenated,
L2-normalised vector preserves spatial colour layout — a red-top / blue-jeans
person gets a different descriptor from blue-top / red-jeans.
"""

from __future__ import annotations

from typing import List, TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from src.stage2_features.crop_extractor import QualityScoredCrop


class HSVExtractor:
    """Extracts spatial HSV colour histograms from object crops.

    Produces a concatenated histogram of H, S, V channels per horizontal stripe,
    L2-normalised for use as a similarity feature.
    """

    def __init__(
        self,
        h_bins: int = 32,
        s_bins: int = 16,
        v_bins: int = 16,
        n_stripes: int = 3,
    ):
        self.h_bins = h_bins
        self.s_bins = s_bins
        self.v_bins = v_bins
        self.n_stripes = n_stripes
        self.bins_per_stripe = h_bins + s_bins + v_bins
        self.total_bins = self.bins_per_stripe * n_stripes

    def extract_histogram(self, crop: np.ndarray) -> np.ndarray:
        """Extract spatial HSV histogram from a single BGR crop.

        The crop is split into *n_stripes* equal horizontal bands and each
        band gets its own histogram.  The results are concatenated.

        Args:
            crop: BGR uint8 image.

        Returns:
            Concatenated normalised histogram, shape (total_bins,).
        """
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h_crop = hsv.shape[0]
        stripe_h = max(h_crop // self.n_stripes, 1)

        parts: List[np.ndarray] = []
        for s in range(self.n_stripes):
            y_start = s * stripe_h
            y_end = h_crop if s == self.n_stripes - 1 else (s + 1) * stripe_h
            stripe = hsv[y_start:y_end, :, :]

            h_hist = cv2.calcHist([stripe], [0], None, [self.h_bins], [0, 180])
            s_hist = cv2.calcHist([stripe], [1], None, [self.s_bins], [0, 256])
            v_hist = cv2.calcHist([stripe], [2], None, [self.v_bins], [0, 256])

            stripe_hist = np.concatenate([
                h_hist.flatten(), s_hist.flatten(), v_hist.flatten()
            ])
            # Per-stripe L1 normalisation (sum to 1)
            stripe_sum = stripe_hist.sum()
            if stripe_sum > 1e-6:
                stripe_hist = stripe_hist / stripe_sum
            parts.append(stripe_hist)

        hist = np.concatenate(parts)

        # Global L2 normalisation
        norm = np.linalg.norm(hist)
        if norm > 1e-6:
            hist = hist / norm

        return hist.astype(np.float32)

    def extract_tracklet_histogram(
        self,
        crops: List[np.ndarray],
        quality_scores: List[float] | None = None,
    ) -> np.ndarray:
        """Extract quality-weighted average spatial HSV histogram.

        Args:
            crops: List of BGR uint8 crops.
            quality_scores: Optional per-crop quality weights.

        Returns:
            Averaged, L2-normalised histogram, shape (total_bins,).
        """
        if not crops:
            return np.zeros(self.total_bins, dtype=np.float32)

        histograms = np.stack([self.extract_histogram(c) for c in crops], axis=0)

        if quality_scores is not None and len(quality_scores) == len(crops):
            weights = np.array(quality_scores, dtype=np.float32)
            weights = weights / max(weights.sum(), 1e-8)
            mean_hist = (histograms * weights[:, np.newaxis]).sum(axis=0)
        else:
            mean_hist = histograms.mean(axis=0)

        # Re-normalise
        norm = np.linalg.norm(mean_hist)
        if norm > 1e-6:
            mean_hist = mean_hist / norm

        return mean_hist.astype(np.float32)

    def extract_tracklet_histogram_from_scored_crops(
        self,
        scored_crops: List["QualityScoredCrop"],
    ) -> np.ndarray:
        """Convenience wrapper accepting QualityScoredCrop objects.

        Args:
            scored_crops: List of QualityScoredCrop from CropExtractor.

        Returns:
            Quality-weighted spatial HSV histogram.
        """
        if not scored_crops:
            return np.zeros(self.total_bins, dtype=np.float32)
        crops = [sc.image for sc in scored_crops]
        qualities = [sc.quality for sc in scored_crops]
        return self.extract_tracklet_histogram(crops, quality_scores=qualities)
