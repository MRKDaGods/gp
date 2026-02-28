"""HSV color histogram extraction for tracklet crops."""

from __future__ import annotations

from typing import List

import cv2
import numpy as np


class HSVExtractor:
    """Extracts HSV color histograms from object crops.

    Produces a concatenated histogram of H, S, V channels,
    L2-normalized for use as a similarity feature.
    """

    def __init__(self, h_bins: int = 16, s_bins: int = 8, v_bins: int = 8):
        self.h_bins = h_bins
        self.s_bins = s_bins
        self.v_bins = v_bins
        self.total_bins = h_bins + s_bins + v_bins

    def extract_histogram(self, crop: np.ndarray) -> np.ndarray:
        """Extract HSV histogram from a single BGR crop.

        Args:
            crop: BGR uint8 image.

        Returns:
            Concatenated normalized histogram, shape (total_bins,).
        """
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        h_hist = cv2.calcHist([hsv], [0], None, [self.h_bins], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [self.s_bins], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], None, [self.v_bins], [0, 256])

        hist = np.concatenate([h_hist.flatten(), s_hist.flatten(), v_hist.flatten()])

        # L2 normalize
        norm = np.linalg.norm(hist)
        if norm > 1e-6:
            hist = hist / norm

        return hist.astype(np.float32)

    def extract_tracklet_histogram(self, crops: List[np.ndarray]) -> np.ndarray:
        """Extract average HSV histogram across multiple crops.

        Args:
            crops: List of BGR uint8 crops.

        Returns:
            Averaged, L2-normalized histogram, shape (total_bins,).
        """
        if not crops:
            return np.zeros(self.total_bins, dtype=np.float32)

        histograms = [self.extract_histogram(crop) for crop in crops]
        mean_hist = np.mean(histograms, axis=0)

        # Re-normalize
        norm = np.linalg.norm(mean_hist)
        if norm > 1e-6:
            mean_hist = mean_hist / norm

        return mean_hist.astype(np.float32)
