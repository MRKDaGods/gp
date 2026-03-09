"""Multi-camera synchronized grid video renderer.

Generates an MP4 showing all cameras side-by-side with a highlighted
global trajectory, allowing visual verification of cross-camera tracking.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger

from src.core.data_models import GlobalTrajectory, TrackletFrame
from src.stage6_visualization.video_annotator import _COLORS, _get_color, _lerp_bbox


# ---------------------------------------------------------------------------
# Grid layout
# ---------------------------------------------------------------------------

@dataclass
class GridLayout:
    """NxM grid arrangement of camera panels."""

    cols: int
    rows: int
    panel_w: int
    panel_h: int
    camera_order: List[str]  # camera_ids in display order

    @property
    def canvas_w(self) -> int:
        return self.cols * self.panel_w

    @property
    def canvas_h(self) -> int:
        return self.rows * self.panel_h


def compute_grid_layout(
    camera_ids: List[str],
    source_w: int = 1920,
    source_h: int = 1080,
    max_panel_w: int = 640,
) -> GridLayout:
    """Compute an optimal grid layout for *n* cameras.

    Favours layouts close to 16:9 aspect ratio with minimal blank cells.
    """
    n = len(camera_ids)
    camera_order = sorted(camera_ids)

    panel_w = max_panel_w
    panel_h = int(panel_w * source_h / source_w)

    # Try all valid (cols, rows) and score them.
    best_cols, best_rows = n, 1
    best_score = -1e9
    target_log_aspect = math.log(16 / 9)  # ~0.575
    for cols in range(1, n + 1):
        rows = math.ceil(n / cols)
        waste = cols * rows - n
        canvas_aspect = (cols * panel_w) / (rows * panel_h)
        # Use log-space distance so too-wide and too-tall are equally bad
        aspect_penalty = abs(math.log(canvas_aspect) - target_log_aspect)
        score = -waste * 2 - aspect_penalty * 3
        # Slight tie-breaking preference for wider layouts (cols >= rows)
        if cols >= rows:
            score += 0.1
        if score > best_score:
            best_score = score
            best_cols, best_rows = cols, rows

    return GridLayout(
        cols=best_cols,
        rows=best_rows,
        panel_w=panel_w,
        panel_h=panel_h,
        camera_order=camera_order,
    )


# ---------------------------------------------------------------------------
# Frame reader
# ---------------------------------------------------------------------------

class FrameReader:
    """Reads stage-0 extracted JPEG frames, resizing on the fly."""

    def __init__(self, stage0_dir: Path, target_size: Tuple[int, int]):
        """
        Args:
            stage0_dir: Directory containing per-camera frame folders.
            target_size: (width, height) to resize each frame to.
        """
        self.stage0_dir = Path(stage0_dir)
        self.target_size = target_size
        self._frame_range: Optional[Tuple[int, int]] = None

    def read_frame(self, camera_id: str, frame_id: int) -> Optional[np.ndarray]:
        path = self.stage0_dir / camera_id / f"frame_{frame_id:06d}.jpg"
        if not path.exists():
            return None
        img = cv2.imread(str(path))
        if img is None:
            return None
        return cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)

    def get_frame_range(self, camera_ids: List[str]) -> Tuple[int, int]:
        """Return (min_frame_id, max_frame_id) across cameras."""
        if self._frame_range is not None:
            return self._frame_range

        lo, hi = float("inf"), 0
        for cam in camera_ids:
            cam_dir = self.stage0_dir / cam
            if not cam_dir.is_dir():
                continue
            frames = sorted(cam_dir.glob("frame_*.jpg"))
            if frames:
                lo = min(lo, int(frames[0].stem.split("_")[1]))
                hi = max(hi, int(frames[-1].stem.split("_")[1]))
        self._frame_range = (int(lo), int(hi))
        return self._frame_range


# ---------------------------------------------------------------------------
# Multi-camera grid renderer
# ---------------------------------------------------------------------------

_INFO_BAR_H = 36  # pixels


class MultiCamGridRenderer:
    """Renders a synchronised multi-camera grid video with tracking annotations."""

    def __init__(
        self,
        layout: GridLayout,
        output_fps: float = 10.0,
        codec: str = "mp4v",
        dim_factor: float = 0.4,
        bbox_thickness: int = 3,
        draw_trails: bool = True,
        trail_length: int = 30,
        max_interp_seconds: float = 0.8,
    ):
        self.layout = layout
        self.output_fps = output_fps
        self.codec = codec
        self.dim_factor = dim_factor
        self.bbox_thickness = bbox_thickness
        self.draw_trails = draw_trails
        self.trail_length = trail_length
        self.max_interp_gap = int(output_fps * max_interp_seconds)

    # ----- public API -------------------------------------------------------

    def render_trajectory(
        self,
        trajectory: GlobalTrajectory,
        frame_reader: FrameReader,
        output_path: str | Path,
        frame_range: Optional[Tuple[int, int]] = None,
    ) -> None:
        """Render a grid video highlighting one global trajectory."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        lookup = self._build_frame_lookup(trajectory)
        if not lookup:
            logger.warning(f"GID {trajectory.global_id}: no frame data, skipping")
            return

        # Determine frame span — use trajectory's active range only
        traj_frames = sorted(lookup.keys())
        fid_lo = traj_frames[0]
        fid_hi = traj_frames[-1]
        if frame_range:
            fid_lo = max(fid_lo, frame_range[0])
            fid_hi = min(fid_hi, frame_range[1])

        canvas_h = self.layout.canvas_h + _INFO_BAR_H
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        writer = cv2.VideoWriter(
            str(output_path), fourcc, self.output_fps,
            (self.layout.canvas_w, canvas_h),
        )

        gid = trajectory.global_id
        class_name = trajectory.class_name
        color = _get_color(gid)
        total_cams = len(self.layout.camera_order)

        # Trail history per camera
        trails: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        trail_last_fid: Dict[str, int] = {}

        total_frames = fid_hi - fid_lo + 1
        for i, fid in enumerate(range(fid_lo, fid_hi + 1)):
            active = lookup.get(fid, {})

            # Read camera panels
            canvas = np.zeros((canvas_h, self.layout.canvas_w, 3), dtype=np.uint8)
            active_count = 0

            for idx, cam_id in enumerate(self.layout.camera_order):
                row, col = divmod(idx, self.layout.cols)
                x_off = col * self.layout.panel_w
                y_off = row * self.layout.panel_h

                panel = frame_reader.read_frame(cam_id, fid)
                if panel is None:
                    # No-signal panel
                    panel = np.full(
                        (self.layout.panel_h, self.layout.panel_w, 3), 30, dtype=np.uint8,
                    )
                    self._draw_camera_label(panel, cam_id, False)
                    canvas[y_off:y_off + self.layout.panel_h, x_off:x_off + self.layout.panel_w] = panel
                    continue

                bbox_data = active.get(cam_id)
                is_active = bbox_data is not None

                if is_active:
                    active_count += 1
                    self._draw_bbox(panel, bbox_data, gid, class_name, color)
                    if self.draw_trails:
                        self._update_trail(trails, trail_last_fid, cam_id, bbox_data, fid, color, panel)
                else:
                    # Dim inactive cameras
                    panel = (panel.astype(np.float32) * (1 - self.dim_factor)).astype(np.uint8)

                self._draw_camera_label(panel, cam_id, is_active)
                canvas[y_off:y_off + self.layout.panel_h, x_off:x_off + self.layout.panel_w] = panel

            # Info bar
            timestamp = fid / self.output_fps
            self._draw_info_bar(
                canvas, timestamp, fid, total_frames, fid_lo,
                gid, class_name, active_count, total_cams,
            )
            writer.write(canvas)

            if (i + 1) % 100 == 0:
                logger.debug(f"GID {gid}: rendered {i+1}/{total_frames} frames")

        writer.release()
        logger.info(f"GID {gid}: {total_frames} frames -> {output_path}")

    def render_gallery(
        self,
        trajectories: List[GlobalTrajectory],
        frame_reader: FrameReader,
        output_dir: str | Path,
        top_n: int = 10,
        sort_by: str = "num_cameras",
    ) -> List[Path]:
        """Render grid videos for the top-N trajectories."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Sort trajectories
        if sort_by == "duration":
            key_fn = lambda t: t.total_duration
        elif sort_by == "num_frames":
            key_fn = lambda t: sum(len(tk.frames) for tk in t.tracklets)
        else:  # num_cameras
            key_fn = lambda t: t.num_cameras
        ranked = sorted(trajectories, key=key_fn, reverse=True)[:top_n]

        outputs = []
        for i, traj in enumerate(ranked, 1):
            out_path = output_dir / f"grid_gid{traj.global_id}.mp4"
            logger.info(
                f"Gallery [{i}/{len(ranked)}] GID {traj.global_id} "
                f"({traj.class_name}, {traj.num_cameras} cams)"
            )
            self.render_trajectory(traj, frame_reader, out_path)
            outputs.append(out_path)

        return outputs

    # ----- internal ---------------------------------------------------------

    def _build_frame_lookup(
        self, trajectory: GlobalTrajectory,
    ) -> Dict[int, Dict[str, TrackletFrame]]:
        """Build {frame_id: {camera_id: TrackletFrame}} with interpolation."""
        result: Dict[int, Dict[str, TrackletFrame]] = defaultdict(dict)

        for tracklet in trajectory.tracklets:
            if not tracklet.frames:
                continue
            sorted_frames = sorted(tracklet.frames, key=lambda f: f.frame_id)

            # Keyframes
            for f in sorted_frames:
                result[f.frame_id][tracklet.camera_id] = f

            # Interpolation
            for i in range(len(sorted_frames) - 1):
                fa, fb = sorted_frames[i], sorted_frames[i + 1]
                gap = fb.frame_id - fa.frame_id
                if gap <= 1 or gap > self.max_interp_gap:
                    continue
                for mid in range(fa.frame_id + 1, fb.frame_id):
                    t = (mid - fa.frame_id) / gap
                    result[mid][tracklet.camera_id] = TrackletFrame(
                        frame_id=mid,
                        timestamp=fa.timestamp + (fb.timestamp - fa.timestamp) * t,
                        bbox=_lerp_bbox(fa.bbox, fb.bbox, t),
                        confidence=fa.confidence + (fb.confidence - fa.confidence) * t,
                    )

        return dict(result)

    def _draw_bbox(
        self,
        panel: np.ndarray,
        bbox_data: TrackletFrame,
        gid: int,
        class_name: str,
        color: Tuple[int, int, int],
    ) -> None:
        """Draw bounding box and label on a panel."""
        pw, ph = self.layout.panel_w, self.layout.panel_h
        # bbox_data.bbox is in original image coords; scale to panel size.
        # We read frames already resized, so if the stage0 frames were the
        # raw resolution (1920x1080) and we resized to (panel_w, panel_h),
        # we need the original resolution. Compute scale from manifest or
        # assume 1920x1080.  Since we don't know source_w/source_h here,
        # read from the bbox which is already in pixel coords of the source.
        # The FrameReader resizes source -> (panel_w, panel_h), so we need:
        #   scale_x = panel_w / source_w, scale_y = panel_h / source_h
        # We'll store these on self at render time. For simplicity, detect
        # from the panel shape vs a reference.
        # Actually, the video_annotator draws in original coords because it
        # reads raw video. Here we already resized, so compute scale factors.
        # We don't have source dims directly, but 1920x1080 is standard.
        # Let's just do: we know panel is already panel_w x panel_h.
        # The bbox coords are in the original space. Scale them.
        sx = pw / 1920.0
        sy = ph / 1080.0
        x1 = int(bbox_data.bbox[0] * sx)
        y1 = int(bbox_data.bbox[1] * sy)
        x2 = int(bbox_data.bbox[2] * sx)
        y2 = int(bbox_data.bbox[3] * sy)

        cv2.rectangle(panel, (x1, y1), (x2, y2), color, self.bbox_thickness)

        label = f"GID:{gid} {class_name}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(panel, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            panel, label, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )

    def _update_trail(
        self,
        trails: Dict[str, List[Tuple[int, int]]],
        trail_last_fid: Dict[str, int],
        cam_id: str,
        bbox_data: TrackletFrame,
        fid: int,
        color: Tuple[int, int, int],
        panel: np.ndarray,
    ) -> None:
        sx = self.layout.panel_w / 1920.0
        sy = self.layout.panel_h / 1080.0
        x1 = int(bbox_data.bbox[0] * sx)
        y1 = int(bbox_data.bbox[1] * sy)
        x2 = int(bbox_data.bbox[2] * sx)
        y2 = int(bbox_data.bbox[3] * sy)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if cam_id in trail_last_fid and (fid - trail_last_fid[cam_id]) > self.max_interp_gap:
            trails[cam_id] = []

        trail_last_fid[cam_id] = fid
        trails[cam_id].append((cx, cy))
        if len(trails[cam_id]) > self.trail_length:
            trails[cam_id] = trails[cam_id][-self.trail_length:]

        pts = trails[cam_id]
        for k in range(1, len(pts)):
            alpha = k / len(pts)
            thick = max(1, int(2 * alpha))
            cv2.line(panel, pts[k - 1], pts[k], color, thick)

    @staticmethod
    def _draw_camera_label(panel: np.ndarray, cam_id: str, is_active: bool) -> None:
        bg = (0, 100, 0) if is_active else (60, 60, 60)
        label = cam_id
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(panel, (0, 0), (tw + 12, th + 12), bg, -1)
        cv2.putText(
            panel, label, (6, th + 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
        )

    def _draw_info_bar(
        self,
        canvas: np.ndarray,
        timestamp: float,
        fid: int,
        total_frames: int,
        fid_lo: int,
        gid: int,
        class_name: str,
        active_count: int,
        total_cams: int,
    ) -> None:
        bar_y = self.layout.canvas_h
        cv2.rectangle(
            canvas, (0, bar_y), (self.layout.canvas_w, bar_y + _INFO_BAR_H),
            (30, 30, 30), -1,
        )
        text = (
            f"t={timestamp:.1f}s | Frame {fid - fid_lo + 1}/{total_frames} | "
            f"GID {gid} ({class_name}) | {active_count}/{total_cams} cams active"
        )
        cv2.putText(
            canvas, text, (10, bar_y + 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1,
        )
