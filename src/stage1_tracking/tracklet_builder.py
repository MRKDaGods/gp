"""Tracklet builder: converts raw per-frame tracker outputs into Tracklet objects."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from loguru import logger

from src.core.constants import CLASS_NAMES
from src.core.data_models import Tracklet, TrackletFrame


class TrackletBuilder:
    """Accumulates per-frame tracking outputs and builds Tracklet objects.

    Usage:
        builder = TrackletBuilder(camera_id="cam01")
        for frame_info in frames:
            tracks = tracker.update(dets, img)
            builder.add_frame(tracks, frame_info.frame_id, frame_info.timestamp)
        tracklets = builder.finalize()
    """

    def __init__(
        self,
        camera_id: str,
        min_length: int = 5,
        min_area: float = 500,
    ):
        """
        Args:
            camera_id: Camera identifier for all tracklets.
            min_length: Minimum number of frames for a valid tracklet.
            min_area: Minimum average bounding box area to keep a tracklet.
        """
        self.camera_id = camera_id
        self.min_length = min_length
        self.min_area = min_area

        # Internal: track_id -> list of TrackletFrame
        self._tracks: Dict[int, List[TrackletFrame]] = {}
        # Internal: track_id -> class_id (majority vote)
        self._track_classes: Dict[int, List[int]] = {}

    def add_frame(
        self,
        tracks: np.ndarray,
        frame_id: int,
        timestamp: float,
    ) -> None:
        """Add one frame's worth of tracker output.

        Args:
            tracks: (M, 8) array: [x1, y1, x2, y2, track_id, conf, class_id, det_idx]
                    or (M, 7) array: [x1, y1, x2, y2, track_id, conf, class_id]
            frame_id: Frame index.
            timestamp: Frame timestamp in seconds.
        """
        if tracks is None or len(tracks) == 0:
            return

        for row in tracks:
            x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            track_id = int(row[4])
            conf = float(row[5])
            class_id = int(row[6]) if len(row) > 6 else 0

            tf = TrackletFrame(
                frame_id=frame_id,
                timestamp=timestamp,
                bbox=(x1, y1, x2, y2),
                confidence=conf,
            )

            if track_id not in self._tracks:
                self._tracks[track_id] = []
                self._track_classes[track_id] = []

            self._tracks[track_id].append(tf)
            self._track_classes[track_id].append(class_id)

    def finalize(self) -> List[Tracklet]:
        """Build and filter Tracklet objects from accumulated data.

        Filters out tracklets that are too short or have too-small bounding boxes.

        Returns:
            List of valid Tracklets, sorted by start time.
        """
        tracklets = []

        for track_id, frames in self._tracks.items():
            # Filter by length
            if len(frames) < self.min_length:
                continue

            # Compute average area
            areas = []
            for f in frames:
                x1, y1, x2, y2 = f.bbox
                areas.append((x2 - x1) * (y2 - y1))
            mean_area = sum(areas) / len(areas)

            if mean_area < self.min_area:
                continue

            # Majority vote for class
            class_ids = self._track_classes[track_id]
            class_id = max(set(class_ids), key=class_ids.count)

            # Sort frames by frame_id
            frames.sort(key=lambda f: f.frame_id)

            tracklets.append(
                Tracklet(
                    track_id=track_id,
                    camera_id=self.camera_id,
                    class_id=class_id,
                    class_name=CLASS_NAMES.get(class_id, f"class_{class_id}"),
                    frames=frames,
                )
            )

        # Sort by start time
        tracklets.sort(key=lambda t: t.start_time)

        logger.debug(
            f"TrackletBuilder({self.camera_id}): {len(self._tracks)} raw tracks "
            f"-> {len(tracklets)} valid tracklets"
        )

        return tracklets
