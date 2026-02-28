"""Frame preprocessing: resize, normalize, denoise."""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


def preprocess_frame(
    frame: np.ndarray,
    target_size: Optional[Tuple[int, int]] = None,
    normalize: bool = False,
    denoise: bool = False,
    denoise_strength: int = 3,
) -> np.ndarray:
    """Apply preprocessing to a single BGR frame.

    Args:
        frame: Input BGR uint8 numpy array (H, W, 3).
        target_size: (width, height) to resize to. None keeps original size.
        normalize: If True, scale pixel values to [0, 1] float32.
        denoise: If True, apply bilateral filtering.
        denoise_strength: Bilateral filter d parameter.

    Returns:
        Preprocessed frame (BGR uint8 or float32 if normalized).
    """
    if denoise:
        frame = cv2.bilateralFilter(frame, denoise_strength, 75, 75)

    if target_size is not None:
        w, h = target_size
        if (frame.shape[1], frame.shape[0]) != (w, h):
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)

    if normalize:
        frame = frame.astype(np.float32) / 255.0

    return frame
