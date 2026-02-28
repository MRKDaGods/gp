"""Crop extraction from tracklets using video frames."""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger

from src.core.data_models import Tracklet
from src.core.video_utils import read_single_frame


class CropExtractor:
    """Extracts object crops from video frames based on tracklet bounding boxes."""

    def __init__(
        self,
        min_area: float = 500,
        padding_ratio: float = 0.1,
        samples_per_tracklet: int = 10,
    ):
        self.min_area = min_area
        self.padding_ratio = padding_ratio
        self.samples_per_tracklet = samples_per_tracklet

    def extract_crops(
        self,
        tracklet: Tracklet,
        video_path: str,
    ) -> List[np.ndarray]:
        """Extract evenly-spaced crops from a tracklet.

        Args:
            tracklet: Tracklet with frame-level bounding boxes.
            video_path: Path to the source video.

        Returns:
            List of BGR uint8 crops (resized consistently).
        """
        n_frames = len(tracklet.frames)
        if n_frames == 0:
            return []

        # Select evenly-spaced frame indices
        if n_frames <= self.samples_per_tracklet:
            selected_indices = list(range(n_frames))
        else:
            step = n_frames / self.samples_per_tracklet
            selected_indices = [int(i * step) for i in range(self.samples_per_tracklet)]

        crops = []
        for idx in selected_indices:
            tf = tracklet.frames[idx]
            x1, y1, x2, y2 = tf.bbox
            area = (x2 - x1) * (y2 - y1)

            if area < self.min_area:
                continue

            try:
                frame = read_single_frame(video_path, tf.frame_id)
            except IOError:
                continue

            crop = self._extract_padded_crop(frame, (x1, y1, x2, y2))
            if crop is not None:
                crops.append(crop)

        return crops

    def extract_crops_from_frames(
        self,
        tracklet: Tracklet,
        frame_images: dict[int, np.ndarray],
    ) -> List[np.ndarray]:
        """Extract crops when frames are already loaded in memory.

        Args:
            tracklet: Tracklet with bounding boxes.
            frame_images: Dict[frame_id, BGR image].

        Returns:
            List of BGR uint8 crops.
        """
        n_frames = len(tracklet.frames)
        if n_frames == 0:
            return []

        if n_frames <= self.samples_per_tracklet:
            selected_indices = list(range(n_frames))
        else:
            step = n_frames / self.samples_per_tracklet
            selected_indices = [int(i * step) for i in range(self.samples_per_tracklet)]

        crops = []
        for idx in selected_indices:
            tf = tracklet.frames[idx]
            if tf.frame_id not in frame_images:
                continue

            area = (tf.bbox[2] - tf.bbox[0]) * (tf.bbox[3] - tf.bbox[1])
            if area < self.min_area:
                continue

            crop = self._extract_padded_crop(frame_images[tf.frame_id], tf.bbox)
            if crop is not None:
                crops.append(crop)

        return crops

    def _extract_padded_crop(
        self,
        frame: np.ndarray,
        bbox: Tuple[float, float, float, float],
    ) -> Optional[np.ndarray]:
        """Extract a crop with padding, clipped to frame boundaries."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox

        # Add padding
        bw, bh = x2 - x1, y2 - y1
        pad_x = bw * self.padding_ratio
        pad_y = bh * self.padding_ratio

        x1 = max(0, int(x1 - pad_x))
        y1 = max(0, int(y1 - pad_y))
        x2 = min(w, int(x2 + pad_x))
        y2 = min(h, int(y2 + pad_y))

        if x2 <= x1 or y2 <= y1:
            return None

        return frame[y1:y2, x1:x2].copy()
