"""Protocol adapters: ported v1 kernels behind v2 component protocols.

Adapters are where the two type worlds meet — v1 kernels speak per-frame
numpy + v1 dataclasses; v2 protocols speak FrameBatch + core.types models.
Nothing outside this package imports v1 kernel types directly.

Registered names:
- detector /
    ``yolo_v1``      ported Ultralytics wrapper (parity component, D18)
- tracker /
    ``boxmot_v1``    ported BoxMOT wrapper (stateful, per camera)
"""

from __future__ import annotations

from athar.components.adapters.detection import YoloDetectorAdapter
from athar.components.adapters.tracking import BoxmotTrackerAdapter
from athar.components.protocols import ComponentKindName
from athar.components.registry import registry

registry.register(ComponentKindName.DETECTOR, "yolo_v1")(YoloDetectorAdapter)
registry.register(ComponentKindName.TRACKER, "boxmot_v1")(BoxmotTrackerAdapter)

__all__ = ["YoloDetectorAdapter", "BoxmotTrackerAdapter"]
