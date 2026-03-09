"""Tracklet builder: converts raw per-frame tracker outputs into Tracklet objects.

Includes linear bounding-box interpolation to fill detection gaps and
intra-camera tracklet merging to reduce fragmentation.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from src.core.constants import CLASS_NAMES
from src.core.data_models import Tracklet, TrackletFrame


# ---------------------------------------------------------------------------
# Bbox interpolation
# ---------------------------------------------------------------------------

def interpolate_tracklet_frames(
    frames: List[TrackletFrame],
    max_gap: int = 30,
) -> List[TrackletFrame]:
    """Fill temporal gaps in a tracklet with linearly-interpolated bboxes.

    If two consecutive observations are separated by ≤ *max_gap* frames,
    intermediate ``TrackletFrame`` entries are synthesised with linearly
    interpolated bounding boxes and confidence = 0 (so downstream stages
    can distinguish real vs. interpolated detections).

    Args:
        frames: Sorted list of TrackletFrame (by frame_id).
        max_gap: Maximum frame gap to interpolate across.

    Returns:
        New list with original + interpolated frames, sorted by frame_id.
    """
    if len(frames) <= 1:
        return list(frames)

    result: List[TrackletFrame] = [frames[0]]

    for i in range(1, len(frames)):
        prev = frames[i - 1]
        curr = frames[i]
        gap = curr.frame_id - prev.frame_id

        if 1 < gap <= max_gap:
            # Linearly interpolate bbox and timestamp for each missing frame
            for k in range(1, gap):
                alpha = k / gap
                x1 = prev.bbox[0] + alpha * (curr.bbox[0] - prev.bbox[0])
                y1 = prev.bbox[1] + alpha * (curr.bbox[1] - prev.bbox[1])
                x2 = prev.bbox[2] + alpha * (curr.bbox[2] - prev.bbox[2])
                y2 = prev.bbox[3] + alpha * (curr.bbox[3] - prev.bbox[3])
                ts = prev.timestamp + alpha * (curr.timestamp - prev.timestamp)
                result.append(
                    TrackletFrame(
                        frame_id=prev.frame_id + k,
                        timestamp=ts,
                        bbox=(x1, y1, x2, y2),
                        confidence=0.0,  # marks interpolated
                    )
                )

        result.append(curr)

    return result


# ---------------------------------------------------------------------------
# Intra-camera track merging
# ---------------------------------------------------------------------------

def merge_intra_camera_tracklets(
    tracklets: List[Tracklet],
    max_time_gap: float = 5.0,
    max_iou_distance: float = 0.7,
) -> List[Tracklet]:
    """Merge same-camera tracklets that are temporally close and spatially overlapping.

    When the tracker loses an object and re-acquires it shortly after, two
    separate tracklets are created for the same identity.  This function
    greedily merges pairs whose temporal gap is ≤ *max_time_gap* seconds and
    whose last/first bounding boxes have IoU ≥ (1 - *max_iou_distance*).

    Args:
        tracklets: List of tracklets from the same camera, sorted by start_time.
        max_time_gap: Maximum seconds between end of A and start of B.
        max_iou_distance: Maximum (1 - IoU) to consider a merge.

    Returns:
        Merged list of tracklets.
    """
    if len(tracklets) <= 1:
        return tracklets

    # Sort by start time
    tracklets = sorted(tracklets, key=lambda t: t.start_time)
    merged_flags = [False] * len(tracklets)
    result: List[Tracklet] = []

    for i in range(len(tracklets)):
        if merged_flags[i]:
            continue

        current = tracklets[i]
        # Greedily try to merge subsequent tracklets into current
        for j in range(i + 1, len(tracklets)):
            if merged_flags[j]:
                continue
            candidate = tracklets[j]

            # Must be same class
            if current.class_id != candidate.class_id:
                continue

            time_gap = candidate.start_time - current.end_time
            if time_gap < 0 or time_gap > max_time_gap:
                if time_gap > max_time_gap:
                    break  # sorted, no point looking further
                continue

            # Check spatial overlap of last bbox of current and first bbox of candidate
            iou = _compute_iou(current.frames[-1].bbox, candidate.frames[0].bbox)
            if iou < (1.0 - max_iou_distance):
                continue

            # Merge: append candidate's frames into current
            all_frames = current.frames + candidate.frames
            all_frames.sort(key=lambda f: f.frame_id)
            current = Tracklet(
                track_id=current.track_id,
                camera_id=current.camera_id,
                class_id=current.class_id,
                class_name=current.class_name,
                frames=all_frames,
            )
            merged_flags[j] = True

        result.append(current)

    n_merged = sum(merged_flags)
    if n_merged > 0:
        logger.info(
            f"Intra-camera merge ({current.camera_id}): "
            f"merged {n_merged} fragments, {len(tracklets)} -> {len(result)} tracklets"
        )

    return result


def _compute_iou(
    box_a: Tuple[float, float, float, float],
    box_b: Tuple[float, float, float, float],
) -> float:
    """Compute IoU between two (x1, y1, x2, y2) bounding boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / max(union, 1e-6)


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
        interpolate: bool = True,
        interpolation_max_gap: int = 30,
        intra_merge: bool = True,
        merge_max_time_gap: float = 5.0,
        merge_max_iou_distance: float = 0.7,
    ):
        """
        Args:
            camera_id: Camera identifier for all tracklets.
            min_length: Minimum number of frames for a valid tracklet.
            min_area: Minimum average bounding box area to keep a tracklet.
            interpolate: Whether to interpolate missing frames in tracklets.
            interpolation_max_gap: Maximum frame gap to interpolate across.
            intra_merge: Whether to merge same-camera fragmented tracklets.
            merge_max_time_gap: Max seconds gap for intra-camera merge.
            merge_max_iou_distance: Max (1-IoU) for intra-camera merge.
        """
        self.camera_id = camera_id
        self.min_length = min_length
        self.min_area = min_area
        self.interpolate = interpolate
        self.interpolation_max_gap = interpolation_max_gap
        self.intra_merge = intra_merge
        self.merge_max_time_gap = merge_max_time_gap
        self.merge_max_iou_distance = merge_max_iou_distance

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

            # Linear bbox interpolation to fill detection gaps
            if self.interpolate:
                frames = interpolate_tracklet_frames(
                    frames, max_gap=self.interpolation_max_gap
                )

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

        # Intra-camera track merge to reduce fragmentation
        if self.intra_merge and len(tracklets) > 1:
            tracklets = merge_intra_camera_tracklets(
                tracklets,
                max_time_gap=self.merge_max_time_gap,
                max_iou_distance=self.merge_max_iou_distance,
            )

        logger.debug(
            f"TrackletBuilder({self.camera_id}): {len(self._tracks)} raw tracks "
            f"-> {len(tracklets)} valid tracklets "
            f"(interpolation={'on' if self.interpolate else 'off'}, "
            f"intra_merge={'on' if self.intra_merge else 'off'})"
        )

        return tracklets
