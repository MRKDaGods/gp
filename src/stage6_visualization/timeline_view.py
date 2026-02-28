"""Gantt-chart style timeline view of global trajectories using Plotly."""

from __future__ import annotations

from typing import List

import plotly.graph_objects as go
from loguru import logger

from src.core.data_models import GlobalTrajectory


def create_timeline(
    trajectories: List[GlobalTrajectory],
    max_trajectories: int = 50,
    title: str = "Trajectory Timeline",
) -> go.Figure:
    """Create an interactive Gantt-chart timeline of trajectories.

    Each row is a global identity. Each bar segment shows when that identity
    was visible on a particular camera.

    Args:
        trajectories: Global trajectories.
        max_trajectories: Maximum number of trajectories to display.
        title: Chart title.

    Returns:
        Plotly Figure.
    """
    # Sort by duration (longest first) and limit
    sorted_traj = sorted(trajectories, key=lambda t: t.total_duration, reverse=True)
    sorted_traj = sorted_traj[:max_trajectories]

    # Build camera color map
    all_cameras = set()
    for traj in sorted_traj:
        for t in traj.tracklets:
            all_cameras.add(t.camera_id)

    camera_colors = {}
    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    ]
    for i, cam_id in enumerate(sorted(all_cameras)):
        camera_colors[cam_id] = colors[i % len(colors)]

    fig = go.Figure()

    for traj in sorted_traj:
        y_label = f"ID {traj.global_id} ({traj.class_name})"

        for tracklet in traj.tracklets:
            fig.add_trace(go.Bar(
                y=[y_label],
                x=[tracklet.duration],
                base=[tracklet.start_time],
                orientation="h",
                marker_color=camera_colors.get(tracklet.camera_id, "#333"),
                name=tracklet.camera_id,
                showlegend=False,
                hovertemplate=(
                    f"<b>Global ID:</b> {traj.global_id}<br>"
                    f"<b>Camera:</b> {tracklet.camera_id}<br>"
                    f"<b>Track ID:</b> {tracklet.track_id}<br>"
                    f"<b>Start:</b> {tracklet.start_time:.1f}s<br>"
                    f"<b>End:</b> {tracklet.end_time:.1f}s<br>"
                    f"<b>Duration:</b> {tracklet.duration:.1f}s<br>"
                    f"<b>Frames:</b> {tracklet.num_frames}"
                    "<extra></extra>"
                ),
            ))

    # Add legend entries for cameras
    for cam_id, color in camera_colors.items():
        fig.add_trace(go.Bar(
            y=[None], x=[None],
            marker_color=color,
            name=cam_id,
            showlegend=True,
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title="Time (seconds)",
        yaxis_title="Global Identity",
        barmode="overlay",
        height=max(400, len(sorted_traj) * 30 + 100),
        template="plotly_white",
        legend=dict(title="Camera"),
    )

    return fig
