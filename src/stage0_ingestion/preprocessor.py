"""Frame preprocessing: resize, normalize, denoise, CLAHE enhancement."""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


def apply_clahe(
    frame: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Converts to LAB colour space, applies CLAHE to the L (lightness) channel,
    then converts back to BGR.  This improves visibility in low-light CCTV
    footage without blowing out well-lit regions.

    Args:
        frame: BGR uint8 image.
        clip_limit: CLAHE contrast clip limit.
        tile_grid_size: Grid size for local histogram equalization.

    Returns:
        Enhanced BGR uint8 image.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l_channel)
    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def preprocess_frame(
    frame: np.ndarray,
    target_size: Optional[Tuple[int, int]] = None,
    normalize: bool = False,
    denoise: bool = False,
    denoise_strength: int = 3,
    clahe: bool = False,
    clahe_clip_limit: float = 2.0,
) -> np.ndarray:
    """Apply preprocessing to a single BGR frame.

    Args:
        frame: Input BGR uint8 numpy array (H, W, 3).
        target_size: (width, height) to resize to. None keeps original size.
        normalize: If True, scale pixel values to [0, 1] float32.
        denoise: If True, apply bilateral filtering.
        denoise_strength: Bilateral filter d parameter.
        clahe: If True, apply CLAHE enhancement for low-light CCTV.
        clahe_clip_limit: CLAHE contrast clip limit.

    Returns:
        Preprocessed frame (BGR uint8 or float32 if normalized).
    """
    # CLAHE enhancement first — improves all downstream processing
    if clahe:
        frame = apply_clahe(frame, clip_limit=clahe_clip_limit)

    if denoise:
        frame = cv2.bilateralFilter(frame, denoise_strength, 75, 75)

    if target_size is not None:
        w, h = target_size
        if (frame.shape[1], frame.shape[0]) != (w, h):
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)

    if normalize:
        frame = frame.astype(np.float32) / 255.0

    return frame
