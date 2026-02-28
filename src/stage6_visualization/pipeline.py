"""Stage 6 — Visualization, Enhancement & Outputs pipeline.

Generates annotated videos, BEV maps, timeline views, and data exports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger
from omegaconf import DictConfig

from src.core.data_models import GlobalTrajectory, Tracklet
from src.stage6_visualization.export import export_csv, export_json
from src.stage6_visualization.timeline_view import create_timeline
from src.stage6_visualization.video_annotator import VideoAnnotator


def run_stage6(
    cfg: DictConfig,
    trajectories: List[GlobalTrajectory],
    tracklets_by_camera: Dict[str, List[Tracklet]],
    video_paths: Dict[str, str],
    output_dir: str | Path,
) -> None:
    """Run visualization and export pipeline.

    Args:
        cfg: Full pipeline config (uses cfg.stage6).
        trajectories: Global trajectories from Stage 4.
        tracklets_by_camera: Per-camera tracklets from Stage 1.
        video_paths: Dict[camera_id, video_file_path].
        output_dir: Directory for stage6 outputs.
    """
    stage_cfg = cfg.stage6
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build global_id mapping: (camera_id, track_id) -> global_id
    global_id_map: Dict[tuple, int] = {}
    for traj in trajectories:
        for tracklet in traj.tracklets:
            global_id_map[(tracklet.camera_id, tracklet.track_id)] = traj.global_id

    # 1. Annotated videos
    if stage_cfg.video.enabled:
        logger.info("Generating annotated videos...")
        annotator = VideoAnnotator(
            draw_bboxes=stage_cfg.video.draw_bboxes,
            draw_ids=stage_cfg.video.draw_ids,
            draw_trails=stage_cfg.video.draw_trails,
            trail_length=stage_cfg.video.trail_length,
            output_fps=stage_cfg.video.output_fps,
            codec=stage_cfg.video.codec,
        )

        videos_dir = output_dir / "videos"
        videos_dir.mkdir(exist_ok=True)

        for camera_id, video_path in video_paths.items():
            tracklets = tracklets_by_camera.get(camera_id, [])
            if not tracklets:
                continue

            output_video = videos_dir / f"annotated_{camera_id}.mp4"
            annotator.annotate_video(
                video_path=video_path,
                tracklets=tracklets,
                global_id_map=global_id_map,
                output_path=str(output_video),
            )
            logger.info(f"  Annotated video saved: {output_video}")

    # 2. Timeline view
    if stage_cfg.timeline.enabled:
        logger.info("Generating timeline view...")
        timeline_path = output_dir / "timeline.html"
        fig = create_timeline(trajectories)
        fig.write_html(str(timeline_path))
        logger.info(f"Timeline saved: {timeline_path}")

    # 3. Export data
    export_dir = output_dir / "exports"
    export_dir.mkdir(exist_ok=True)

    for fmt in stage_cfg.export.formats:
        if fmt == "json":
            export_json(trajectories, export_dir / "trajectories.json")
        elif fmt == "csv":
            export_csv(trajectories, export_dir / "trajectories.csv")

    logger.info(f"Stage 6 complete: outputs in {output_dir}")
