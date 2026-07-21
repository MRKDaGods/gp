"""FrameSource over a directory of extracted frames.

Benchmark datasets (WILDTRACK, some CityFlow exports) ship frames, not
containers; lexicographic filename order defines frame indices. Also the
lossless path for pipeline tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from athar.components.framesources.base import (
    DecodedFrameBatch,
    FrameSourceError,
    chunked,
    plan_indices,
)

_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ImageDirFrameSource:
    def __init__(
        self,
        camera_id: str,
        path: Path | str,
        start: int = 0,
        stop: Optional[int] = None,
        step: int = 1,
        fps: Optional[float] = None,
    ) -> None:
        path = Path(path)
        if not path.is_dir():
            raise FrameSourceError(f"image directory not found: {path}")
        self.camera_id = camera_id
        self.path = path
        self.nominal_fps = fps
        self._files = sorted(
            p for p in path.iterdir() if p.suffix.lower() in _EXTENSIONS
        )
        if not self._files:
            raise FrameSourceError(f"no image files in {path}")
        self.frame_count = len(self._files)
        plan = plan_indices(self.frame_count, start, stop, step)
        assert plan is not None
        self._plan = plan

    def batches(self, batch_size: int) -> Iterator[DecodedFrameBatch]:
        import cv2  # noqa: PLC0415 — lazy so importing the module never needs it

        for indices in chunked(list(self._plan), batch_size):
            images = []
            for i in indices:
                img = cv2.imread(str(self._files[i]), cv2.IMREAD_COLOR)  # BGR
                if img is None:
                    raise FrameSourceError(f"cannot read image: {self._files[i]}")
                images.append(img)
            yield DecodedFrameBatch(
                camera_id=self.camera_id,
                frame_indices=tuple(indices),
                _images=np.stack(images),
            )
