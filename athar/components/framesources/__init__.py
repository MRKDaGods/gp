"""FrameSource implementations + registry wiring.

Registered names (kind = ``frame_source``):
- ``video``            best available decoder (torchcodec → PyAV → OpenCV)
- ``video_torchcodec`` explicit torchcodec (frame-exact random access)
- ``video_pyav``       explicit PyAV (sequential, pts-accurate)
- ``video_opencv``     explicit OpenCV (sequential-only; no pts)
- ``image_dir``        directory of extracted frames
"""

from __future__ import annotations

from athar.components.framesources.base import (
    DecodedFrameBatch,
    FrameSourceError,
    pts_deviation_s,
)
from athar.components.framesources.image_dir import ImageDirFrameSource
from athar.components.framesources.video import (
    OpenCVFrameSource,
    PyAVFrameSource,
    TorchcodecFrameSource,
    create_video_source,
)
from athar.components.protocols import ComponentKindName
from athar.components.registry import registry

registry.register(ComponentKindName.FRAME_SOURCE, "video")(create_video_source)
registry.register(ComponentKindName.FRAME_SOURCE, "video_torchcodec")(TorchcodecFrameSource)
registry.register(ComponentKindName.FRAME_SOURCE, "video_pyav")(PyAVFrameSource)
registry.register(ComponentKindName.FRAME_SOURCE, "video_opencv")(OpenCVFrameSource)
registry.register(ComponentKindName.FRAME_SOURCE, "image_dir")(ImageDirFrameSource)

__all__ = [
    "DecodedFrameBatch",
    "FrameSourceError",
    "ImageDirFrameSource",
    "OpenCVFrameSource",
    "PyAVFrameSource",
    "TorchcodecFrameSource",
    "create_video_source",
    "pts_deviation_s",
]
