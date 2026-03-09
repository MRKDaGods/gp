"""Quality-aware crop extraction from tracklets using video frames.

Scores each candidate crop by sharpness (Laplacian variance), bounding-box
area, aspect ratio, and detection confidence.  Returns top-quality crops
together with a per-crop quality score used for downstream attention pooling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger

from src.core.data_models import Tracklet
from src.core.video_utils import read_single_frame


# ---------------------------------------------------------------------------
# Quality scoring
# ---------------------------------------------------------------------------

@dataclass
class QualityScoredCrop:
    """A crop together with its quality score."""
    image: np.ndarray          # BGR uint8
    quality: float             # [0, 1] composite quality score
    frame_id: int
    confidence: float


def compute_crop_quality(
    crop: np.ndarray,
    confidence: float = 1.0,
    target_area: float = 20_000,
) -> float:
    """Compute a composite quality score for a single crop.

    Components (each normalised to [0, 1]):
    - **Sharpness**: Laplacian variance (higher = sharper).
    - **Size**: Area relative to *target_area*.
    - **Aspect ratio**: Penalty for extreme ratios (< 0.25 or > 4.0).
    - **Confidence**: Detection confidence from the tracker.

    Returns:
        A float in [0, 1].
    """
    h, w = crop.shape[:2]
    area = h * w

    # Sharpness via Laplacian variance
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness = min(laplacian_var / 500.0, 1.0)  # saturate at 500

    # Size score
    size_score = min(area / target_area, 1.0)

    # Aspect ratio penalty
    aspect = w / max(h, 1)
    if 0.25 <= aspect <= 4.0:
        aspect_score = 1.0
    else:
        aspect_score = 0.3

    # Compose
    quality = (
        0.35 * sharpness
        + 0.25 * size_score
        + 0.15 * aspect_score
        + 0.25 * confidence
    )
    return float(np.clip(quality, 0.0, 1.0))


class CropExtractor:
    """Extracts quality-scored object crops from video frames."""

    def __init__(
        self,
        min_area: float = 500,
        padding_ratio: float = 0.1,
        samples_per_tracklet: int = 16,
        min_quality: float = 0.05,
        laplacian_min_var: float = 0.0,
    ):
        self.min_area = min_area
        self.padding_ratio = padding_ratio
        self.samples_per_tracklet = samples_per_tracklet
        self.min_quality = min_quality
        self.laplacian_min_var = laplacian_min_var

    def extract_crops(
        self,
        tracklet: Tracklet,
        video_path: str,
    ) -> List[QualityScoredCrop]:
        """Extract quality-scored crops from a tracklet.

        Selects evenly-spaced frames, scores each crop, discards low-quality
        ones, and returns the best *samples_per_tracklet* crops sorted by
        quality (descending).

        Args:
            tracklet: Tracklet with frame-level bounding boxes.
            video_path: Path to the source video.

        Returns:
            List of QualityScoredCrop, sorted by quality descending.
        """
        n_frames = len(tracklet.frames)
        if n_frames == 0:
            return []

        # Over-sample: pick ~2x candidates then keep the best
        n_candidates = min(n_frames, self.samples_per_tracklet * 2)
        if n_frames <= n_candidates:
            selected_indices = list(range(n_frames))
        else:
            step = n_frames / n_candidates
            selected_indices = [int(i * step) for i in range(n_candidates)]

        candidates: List[QualityScoredCrop] = []
        skipped_area = skipped_conf = skipped_read = skipped_crop = 0
        skipped_blur = skipped_quality = 0
        for idx in selected_indices:
            tf = tracklet.frames[idx]
            x1, y1, x2, y2 = tf.bbox
            area = (x2 - x1) * (y2 - y1)
            if area < self.min_area:
                skipped_area += 1
                continue

            # Skip interpolated frames (confidence == 0)
            if tf.confidence <= 0:
                skipped_conf += 1
                continue

            try:
                frame = read_single_frame(video_path, tf.frame_id)
            except IOError:
                skipped_read += 1
                continue

            crop = self._extract_padded_crop(frame, (x1, y1, x2, y2))
            if crop is None:
                skipped_crop += 1
                continue

            # Optional Laplacian-variance hard sharpness filter
            if self.laplacian_min_var > 0:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                if lap_var < self.laplacian_min_var:
                    skipped_blur += 1
                    continue

            quality = compute_crop_quality(crop, confidence=tf.confidence)
            if quality < self.min_quality:
                skipped_quality += 1
                continue

            candidates.append(QualityScoredCrop(
                image=crop,
                quality=quality,
                frame_id=tf.frame_id,
                confidence=tf.confidence,
            ))

        # Sort by quality and keep top-N
        if not candidates and len(selected_indices) > 0:
            logger.debug(
                f"Tracklet {tracklet.track_id}: 0/{len(selected_indices)} crops survived filtering "
                f"(area={skipped_area}, conf={skipped_conf}, read={skipped_read}, "
                f"crop={skipped_crop}, blur={skipped_blur}, quality={skipped_quality})"
            )
        candidates.sort(key=lambda c: c.quality, reverse=True)
        return candidates[: self.samples_per_tracklet]

    def extract_crops_from_frames(
        self,
        tracklet: Tracklet,
        frame_images: dict[int, np.ndarray],
    ) -> List[QualityScoredCrop]:
        """Extract quality-scored crops when frames are already in memory.

        Args:
            tracklet: Tracklet with bounding boxes.
            frame_images: Dict[frame_id, BGR image].

        Returns:
            List of QualityScoredCrop, sorted by quality descending.
        """
        n_frames = len(tracklet.frames)
        if n_frames == 0:
            return []

        n_candidates = min(n_frames, self.samples_per_tracklet * 2)
        if n_frames <= n_candidates:
            selected_indices = list(range(n_frames))
        else:
            step = n_frames / n_candidates
            selected_indices = [int(i * step) for i in range(n_candidates)]

        candidates: List[QualityScoredCrop] = []
        for idx in selected_indices:
            tf = tracklet.frames[idx]
            if tf.frame_id not in frame_images:
                continue
            if tf.confidence <= 0:
                continue

            area = (tf.bbox[2] - tf.bbox[0]) * (tf.bbox[3] - tf.bbox[1])
            if area < self.min_area:
                continue

            crop = self._extract_padded_crop(frame_images[tf.frame_id], tf.bbox)
            if crop is None:
                continue

            # Optional Laplacian-variance hard sharpness filter
            if self.laplacian_min_var > 0:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                if lap_var < self.laplacian_min_var:
                    continue

            quality = compute_crop_quality(crop, confidence=tf.confidence)
            if quality < self.min_quality:
                continue

            candidates.append(QualityScoredCrop(
                image=crop,
                quality=quality,
                frame_id=tf.frame_id,
                confidence=tf.confidence,
            ))

        candidates.sort(key=lambda c: c.quality, reverse=True)
        return candidates[: self.samples_per_tracklet]

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
