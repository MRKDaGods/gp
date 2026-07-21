"""FrameSource foundations: the decoded-batch container and shared iteration.

Contract (all implementations):

- ``frame_indices`` are ORIGINAL container frame indices — sampling with
  ``step`` must preserve them, because ``CameraTimeBase.to_scene`` maps
  original indices to scene seconds. A source never renumbers frames.
- ``images()`` returns ``(N, H, W, 3) uint8 BGR`` — the layout every ported
  v1 kernel (detector crops, HSV extractor) expects.
- ``pts_s`` carries the decoder's true presentation timestamps when known
  (torchcodec / PyAV). On VFR footage these are the ground truth; frame-index
  arithmetic is only exact for CFR. Downstream prefers pts when present.
- Decode is on demand, batch by batch. Nothing here writes JPEG trees.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Sequence

import numpy as np


class FrameSourceError(RuntimeError):
    pass


@dataclass(frozen=True)
class DecodedFrameBatch:
    """Concrete FrameBatch: one contiguous chunk of decoded frames."""

    camera_id: str
    frame_indices: tuple[int, ...]
    _images: np.ndarray  # (N, H, W, 3) uint8 BGR
    pts_s: Optional[tuple[float, ...]] = None

    def __post_init__(self) -> None:
        if self._images.ndim != 4 or self._images.shape[-1] != 3:
            raise FrameSourceError(
                f"batch images must be (N, H, W, 3); got {self._images.shape}"
            )
        if self._images.shape[0] != len(self.frame_indices):
            raise FrameSourceError(
                f"{self._images.shape[0]} images != {len(self.frame_indices)} indices"
            )
        if self.pts_s is not None and len(self.pts_s) != len(self.frame_indices):
            raise FrameSourceError(
                f"{len(self.pts_s)} pts != {len(self.frame_indices)} indices"
            )

    def images(self) -> np.ndarray:
        return self._images

    def __len__(self) -> int:
        return len(self.frame_indices)


def plan_indices(
    frame_count: Optional[int],
    start: int = 0,
    stop: Optional[int] = None,
    step: int = 1,
) -> Optional[range]:
    """The original-index sampling plan shared by all sources.

    Returns None when frame_count is unknown (purely sequential sources
    then iterate until EOF applying the same start/stop/step filter).
    """
    if start < 0 or step < 1:
        raise FrameSourceError(f"invalid sampling: start={start} step={step}")
    if stop is not None and stop < start:
        raise FrameSourceError(f"invalid sampling: stop={stop} < start={start}")
    if frame_count is None:
        return None
    end = frame_count if stop is None else min(stop, frame_count)
    return range(start, end, step)


def chunked(seq: Sequence[int], size: int) -> Iterator[Sequence[int]]:
    if size < 1:
        raise FrameSourceError(f"batch_size must be >= 1, got {size}")
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def pts_deviation_s(batch: DecodedFrameBatch, fps: float) -> Optional[float]:
    """Max |actual pts gap − index-implied gap| inside a batch.

    ≈0 for CFR footage; grows on VFR/dropped-frame CCTV, signalling that
    frame-index time arithmetic (CameraTimeBase) is unsafe and pts should
    drive timing instead.
    """
    if batch.pts_s is None or len(batch) < 2:
        return None
    worst = 0.0
    for a, b, ia, ib in zip(
        batch.pts_s, batch.pts_s[1:], batch.frame_indices, batch.frame_indices[1:]
    ):
        worst = max(worst, abs((b - a) - (ib - ia) / fps))
    return worst
