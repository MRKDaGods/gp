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


def _lerp_bbox(bbox_a, bbox_b, t: float):
    """Linearly interpolate between two bounding boxes."""
    return tuple(a + (b - a) * t for a, b in zip(bbox_a, bbox_b))


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
        max_interp_seconds: float = 0.8,
    ):
        self.draw_bboxes = draw_bboxes
        self.draw_ids = draw_ids
        self.draw_trails = draw_trails
        self.trail_length = trail_length
        self.output_fps = output_fps
        self.codec = codec
        self.max_interp_seconds = max_interp_seconds

    def _build_interpolated_frame_data(
        self,
        tracklets: List[Tracklet],
        global_id_map: Dict[tuple, int],
        total_frames: int,
        video_fps: float,
    ) -> Dict[int, list]:
        """Build per-frame annotation data with interpolation between keyframes.

        For each tracklet, we have detections at sparse keyframes.
        This interpolates bounding boxes linearly for frames in between,
        but only when the gap is small enough that interpolation is reliable.
        Large gaps (where the person was lost) are skipped to avoid drawing
        stale boxes at wrong positions.
        """
        # Max gap in original video frames for interpolation
        max_interp_gap = int(video_fps * self.max_interp_seconds)
        logger.debug(
            f"Interpolation: max gap = {max_interp_gap} frames "
            f"({self.max_interp_seconds}s at {video_fps:.0f}fps)"
        )

        frame_data: Dict[int, list] = defaultdict(list)

        for t in tracklets:
            global_id = global_id_map.get((t.camera_id, t.track_id), t.track_id)
            if not t.frames:
                continue

            # Sort frames by frame_id
            sorted_frames = sorted(t.frames, key=lambda f: f.frame_id)

            # Add keyframe data directly
            for f in sorted_frames:
                frame_data[f.frame_id].append({
                    "bbox": f.bbox,
                    "track_id": t.track_id,
                    "global_id": global_id,
                    "class_name": t.class_name,
                })

            # Interpolate between consecutive keyframes (only if gap is small)
            for i in range(len(sorted_frames) - 1):
                f_a = sorted_frames[i]
                f_b = sorted_frames[i + 1]
                gap = f_b.frame_id - f_a.frame_id

                if gap <= 1:
                    continue  # Adjacent frames, no interpolation needed

                if gap > max_interp_gap:
                    continue  # Gap too large, person was likely lost — don't interpolate

                for mid in range(f_a.frame_id + 1, f_b.frame_id):
                    t_ratio = (mid - f_a.frame_id) / gap
                    interp_bbox = _lerp_bbox(f_a.bbox, f_b.bbox, t_ratio)
                    frame_data[mid].append({
                        "bbox": interp_bbox,
                        "track_id": t.track_id,
                        "global_id": global_id,
                        "class_name": t.class_name,
                    })

        return frame_data

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

        # Max gap in frames for trail continuity (same as interpolation)
        max_trail_gap = int(info.fps * self.max_interp_seconds)

        # Build interpolated frame data so annotations are smooth
        frame_data = self._build_interpolated_frame_data(
            tracklets, global_id_map, info.total_frames, info.fps,
        )

        # Trail history: global_id -> list of (cx, cy) centers
        trails: Dict[int, list] = defaultdict(list)
        # Track last frame each gid was seen (for trail gap detection)
        trail_last_frame: Dict[int, int] = {}

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

                        # Reset trail if there's a large temporal gap
                        if gid in trail_last_frame:
                            if (frame_idx - trail_last_frame[gid]) > max_trail_gap:
                                trails[gid] = []

                        trail_last_frame[gid] = frame_idx
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
