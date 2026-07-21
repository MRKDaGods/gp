"""Video-file FrameSources.

Three decoders, one contract (see base.py):

- **torchcodec** (primary, docs/STACK.md): random-access frame-EXACT decode
  (``seek_mode="exact"`` scans the container index up front), true pts out,
  native batch decode. The only implementation that can decode a sparse
  sampling plan without reading every frame.
- **PyAV** (fallback): strictly sequential decode with frame counting —
  frame-exact by construction, pts out; skipped frames are still decoded.
- **OpenCV** (last resort): strictly sequential ``grab()``/``retrieve()``.
  NEVER ``set(CAP_PROP_POS_FRAMES)`` — cv2 seeking is not frame-accurate on
  long-GOP CCTV (the v1 bug class this module retires). No pts.

``create_video_source`` picks the best available decoder at runtime.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from athar.components.framesources.base import (
    DecodedFrameBatch,
    FrameSourceError,
    chunked,
    plan_indices,
)

logger = logging.getLogger(__name__)


def _require_file(path: Path | str) -> Path:
    path = Path(path)
    if not path.is_file():
        raise FrameSourceError(f"video not found: {path}")
    return path


class TorchcodecFrameSource:
    """Random-access, frame-exact decode via torchcodec (RGB → BGR copy)."""

    def __init__(
        self,
        camera_id: str,
        path: Path | str,
        start: int = 0,
        stop: Optional[int] = None,
        step: int = 1,
        device: str = "cpu",
    ) -> None:
        from torchcodec.decoders import VideoDecoder  # noqa: PLC0415 — optional dep

        self.camera_id = camera_id
        self.path = _require_file(path)
        self._decoder = VideoDecoder(
            str(self.path),
            dimension_order="NHWC",
            seek_mode="exact",
            device=device,
        )
        meta = self._decoder.metadata
        self.nominal_fps: Optional[float] = meta.average_fps
        self.frame_count: int = len(self._decoder)
        plan = plan_indices(self.frame_count, start, stop, step)
        assert plan is not None  # frame_count is always known here
        self._plan = plan

    def batches(self, batch_size: int) -> Iterator[DecodedFrameBatch]:
        for indices in chunked(list(self._plan), batch_size):
            fb = self._decoder.get_frames_at(indices=list(indices))
            rgb = fb.data.cpu().numpy()  # (N, H, W, 3) uint8 RGB
            yield DecodedFrameBatch(
                camera_id=self.camera_id,
                frame_indices=tuple(indices),
                _images=np.ascontiguousarray(rgb[..., ::-1]),  # → BGR
                pts_s=tuple(float(p) for p in fb.pts_seconds),
            )


class PyAVFrameSource:
    """Sequential PyAV decode; frame-exact by counting, pts from the demuxer."""

    def __init__(
        self,
        camera_id: str,
        path: Path | str,
        start: int = 0,
        stop: Optional[int] = None,
        step: int = 1,
    ) -> None:
        import av  # noqa: PLC0415 — optional dep

        self.camera_id = camera_id
        self.path = _require_file(path)
        self._av = av
        with av.open(str(self.path)) as container:
            stream = next((s for s in container.streams if s.type == "video"), None)
            if stream is None:
                raise FrameSourceError(f"no video stream in {self.path}")
            self.nominal_fps = float(stream.average_rate) if stream.average_rate else None
            self.frame_count: Optional[int] = stream.frames or None
        self._start, self._stop, self._step = start, stop, step
        plan_indices(self.frame_count, start, stop, step)  # validate early

    def _wanted(self, idx: int) -> bool:
        if idx < self._start or (self._stop is not None and idx >= self._stop):
            return False
        return (idx - self._start) % self._step == 0

    def batches(self, batch_size: int) -> Iterator[DecodedFrameBatch]:
        if batch_size < 1:
            raise FrameSourceError(f"batch_size must be >= 1, got {batch_size}")
        images: list[np.ndarray] = []
        indices: list[int] = []
        pts: list[float] = []
        with self._av.open(str(self.path)) as container:
            stream = next(s for s in container.streams if s.type == "video")
            stream.thread_type = "AUTO"
            for idx, frame in enumerate(container.decode(stream)):
                if self._stop is not None and idx >= self._stop:
                    break
                if not self._wanted(idx):
                    continue
                images.append(frame.to_ndarray(format="bgr24"))
                indices.append(idx)
                pts.append(
                    float(frame.pts * stream.time_base) if frame.pts is not None else -1.0
                )
                if len(images) == batch_size:
                    yield self._flush(images, indices, pts)
                    images, indices, pts = [], [], []
        if images:
            yield self._flush(images, indices, pts)

    def _flush(
        self, images: list[np.ndarray], indices: list[int], pts: list[float]
    ) -> DecodedFrameBatch:
        return DecodedFrameBatch(
            camera_id=self.camera_id,
            frame_indices=tuple(indices),
            _images=np.stack(images),
            pts_s=None if any(p < 0 for p in pts) else tuple(pts),
        )


class OpenCVFrameSource:
    """Sequential-only cv2 decode (grab/retrieve); the no-extra-deps fallback."""

    def __init__(
        self,
        camera_id: str,
        path: Path | str,
        start: int = 0,
        stop: Optional[int] = None,
        step: int = 1,
    ) -> None:
        import cv2  # noqa: PLC0415

        self.camera_id = camera_id
        self.path = _require_file(path)
        self._cv2 = cv2
        cap = cv2.VideoCapture(str(self.path))
        try:
            if not cap.isOpened():
                raise FrameSourceError(f"cannot open video: {self.path}")
            self.nominal_fps = cap.get(cv2.CAP_PROP_FPS) or None
            self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        finally:
            cap.release()
        self._start, self._stop, self._step = start, stop, step
        plan_indices(self.frame_count, start, stop, step)  # validate early

    def batches(self, batch_size: int) -> Iterator[DecodedFrameBatch]:
        if batch_size < 1:
            raise FrameSourceError(f"batch_size must be >= 1, got {batch_size}")
        cap = self._cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise FrameSourceError(f"cannot open video: {self.path}")
        images: list[np.ndarray] = []
        indices: list[int] = []
        try:
            idx = 0
            while cap.grab():  # sequential — never CAP_PROP_POS_FRAMES
                if self._stop is not None and idx >= self._stop:
                    break
                wanted = idx >= self._start and (idx - self._start) % self._step == 0
                if wanted:
                    ok, img = cap.retrieve()
                    if not ok:
                        raise FrameSourceError(f"decode failed at frame {idx}: {self.path}")
                    images.append(img)
                    indices.append(idx)
                    if len(images) == batch_size:
                        yield DecodedFrameBatch(
                            camera_id=self.camera_id,
                            frame_indices=tuple(indices),
                            _images=np.stack(images),
                        )
                        images, indices = [], []
                idx += 1
        finally:
            cap.release()
        if images:
            yield DecodedFrameBatch(
                camera_id=self.camera_id,
                frame_indices=tuple(indices),
                _images=np.stack(images),
            )


_PREFERENCE = (
    ("torchcodec", TorchcodecFrameSource),
    ("av", PyAVFrameSource),
    ("cv2", OpenCVFrameSource),
)
_selected: Optional[type] = None


def create_video_source(camera_id: str, path: Path | str, **kwargs: object):
    """Best-available video source: torchcodec → PyAV → OpenCV."""
    global _selected
    if _selected is None:
        for module_name, cls in _PREFERENCE:
            try:
                __import__(module_name)
            except ImportError:
                continue
            _selected = cls
            logger.info("video FrameSource decoder: %s", cls.__name__)
            break
        else:  # pragma: no cover — cv2 is a hard dep of the ml extra
            raise FrameSourceError("no video decoder available (torchcodec/av/cv2)")
    return _selected(camera_id, path, **kwargs)  # type: ignore[arg-type]
