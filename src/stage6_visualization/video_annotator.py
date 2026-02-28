"""Video annotation: draw bounding boxes, IDs, and motion trails on video frames."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger

from src.core.data_models import Tracklet
from src.core.video_utils import get_video_info


# Color palette for global IDs (distinct colors)
_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
    (128, 0, 128), (0, 128, 128), (255, 128, 0), (255, 0, 128), (128, 255, 0),
    (0, 255, 128), (0, 128, 255), (128, 0, 255), (255, 128, 128), (128, 255, 128),
]


def _get_color(identity_id: int) -> Tuple[int, int, int]:
    return _COLORS[identity_id % len(_COLORS)]


class VideoAnnotator:
    """Draws tracking annotations on video frames."""

    def __init__(
        self,
        draw_bboxes: bool = True,
        draw_ids: bool = True,
        draw_trails: bool = True,
        trail_length: int = 30,
        output_fps: float = 10.0,
        codec: str = "mp4v",
    ):
        self.draw_bboxes = draw_bboxes
        self.draw_ids = draw_ids
        self.draw_trails = draw_trails
        self.trail_length = trail_length
        self.output_fps = output_fps
        self.codec = codec

    def annotate_video(
        self,
        video_path: str,
        tracklets: List[Tracklet],
        global_id_map: Dict[tuple, int],
        output_path: str,
    ) -> None:
        """Generate annotated video with tracking overlays.

        Args:
            video_path: Source video path.
            tracklets: Per-camera tracklets.
            global_id_map: (camera_id, track_id) -> global_id mapping.
            output_path: Output video path.
        """
        info = get_video_info(video_path)

        # Build frame-level lookup: frame_id -> list of (bbox, track_id, global_id, class_name)
        frame_data = defaultdict(list)
        for t in tracklets:
            global_id = global_id_map.get((t.camera_id, t.track_id), t.track_id)
            for f in t.frames:
                frame_data[f.frame_id].append({
                    "bbox": f.bbox,
                    "track_id": t.track_id,
                    "global_id": global_id,
                    "class_name": t.class_name,
                })

        # Trail history: global_id -> list of (cx, cy) centers
        trails: Dict[int, list] = defaultdict(list)

        # Process video
        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        writer = cv2.VideoWriter(
            output_path, fourcc, self.output_fps, (info.width, info.height)
        )

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx in frame_data:
                for det in frame_data[frame_idx]:
                    x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
                    gid = det["global_id"]
                    color = _get_color(gid)

                    # Bounding box
                    if self.draw_bboxes:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # ID label
                    if self.draw_ids:
                        label = f"ID:{gid} {det['class_name']}"
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
                        cv2.putText(
                            frame, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                        )

                    # Trail
                    if self.draw_trails:
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2
                        trails[gid].append((cx, cy))
                        if len(trails[gid]) > self.trail_length:
                            trails[gid] = trails[gid][-self.trail_length:]

                        pts = trails[gid]
                        for k in range(1, len(pts)):
                            alpha = k / len(pts)
                            thick = max(1, int(2 * alpha))
                            cv2.line(frame, pts[k - 1], pts[k], color, thick)

            writer.write(frame)
            frame_idx += 1

        cap.release()
        writer.release()
        logger.debug(f"Annotated video: {frame_idx} frames -> {output_path}")
