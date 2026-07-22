"""Protocol adapters: ported v1 kernels behind v2 component protocols.

Adapters are where the two type worlds meet — v1 kernels speak per-frame
numpy + v1 dataclasses; v2 protocols speak FrameBatch + core.types models.
Nothing outside this package imports v1 kernel types directly.

Registered names:
- detector /
    ``yolo_v1``          ported Ultralytics wrapper (parity component, D18)
- tracker /
    ``boxmot_v1``        ported BoxMOT wrapper (stateful, per camera)
- embedder /
    ``transreid_v1``     ported TransReID ReIDModel (flip TTA + quality pooling)
    ``clipsenet_v1``     CLIP-SENet v6 (offline construction + strict ckpt load)
    ``dinov2_v1``        DINOv2-L TransReID tertiary (09s kernel port, 14e recipe)
    ``hsv_v1``           striped HSV histograms (pure numpy/cv2)
"""

from __future__ import annotations

from athar.components.adapters.detection import YoloDetectorAdapter
from athar.components.adapters.embedding import (
    ClipSenetEmbedderAdapter,
    Dinov2EmbedderAdapter,
    HsvEmbedderAdapter,
    TransReidEmbedderAdapter,
)
from athar.components.adapters.tracking import BoxmotTrackerAdapter
from athar.components.protocols import ComponentKindName
from athar.components.registry import registry

registry.register(ComponentKindName.DETECTOR, "yolo_v1")(YoloDetectorAdapter)
registry.register(ComponentKindName.TRACKER, "boxmot_v1")(BoxmotTrackerAdapter)
registry.register(ComponentKindName.EMBEDDER, "transreid_v1")(TransReidEmbedderAdapter)
registry.register(ComponentKindName.EMBEDDER, "clipsenet_v1")(ClipSenetEmbedderAdapter)
registry.register(ComponentKindName.EMBEDDER, "dinov2_v1")(Dinov2EmbedderAdapter)
registry.register(ComponentKindName.EMBEDDER, "hsv_v1")(HsvEmbedderAdapter)

__all__ = [
    "YoloDetectorAdapter",
    "BoxmotTrackerAdapter",
    "TransReidEmbedderAdapter",
    "ClipSenetEmbedderAdapter",
    "Dinov2EmbedderAdapter",
    "HsvEmbedderAdapter",
]
