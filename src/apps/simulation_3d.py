"""3D trajectory simulation and visualization using Plotly."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
from loguru import logger

from src.core.data_models import GlobalTrajectory


class Simulator3D:
    """3D visualization of global trajectories.

    Creates an interactive 3D scene where:
    - X-Y plane represents spatial position (approximated from bbox centers)
    - Z axis represents time
    - Each trajectory is a colored 3D line
    - Camera positions are marked as reference points
    """

    def __init__(
        self,
        camera_layout: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        """
        Args:
            camera_layout: Optional dict[camera_id, (x, y)] positions.
                          If None, cameras are arranged in a circle.
        """
        self.camera_layout = camera_layout or {}

    def render(
        self,
        trajectories: List[GlobalTrajectory],
        max_trajectories: int = 100,
        title: str = "3D Trajectory Simulation",
    ) -> go.Figure:
        """Render 3D trajectory visualization.

        Args:
            trajectories: Global trajectories to visualize.
            max_trajectories: Maximum trajectories to render.
            title: Plot title.

        Returns:
            Plotly 3D Figure.
        """
        # Sort by number of cameras (most interesting first)
        sorted_traj = sorted(
            trajectories,
            key=lambda t: (t.num_cameras, t.total_duration),
            reverse=True,
        )[:max_trajectories]

        # Auto-layout cameras if not provided
        camera_positions = self._get_camera_positions(sorted_traj)

        fig = go.Figure()

        # Color palette
        colors = [
            f"hsl({h}, 70%, 50%)"
            for h in np.linspace(0, 360, max_trajectories, endpoint=False)
        ]

        # Plot trajectories
        for i, traj in enumerate(sorted_traj):
            color = colors[i % len(colors)]
            xs, ys, zs = [], [], []
            hover_texts = []

            for tracklet in sorted(traj.tracklets, key=lambda t: t.start_time):
                cam_pos = camera_positions.get(tracklet.camera_id, (0, 0))

                for frame in tracklet.frames:
                    x1, y1, x2, y2 = frame.bbox
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2

                    # Map to world coordinates using camera position
                    wx = cam_pos[0] + (cx - 320) * 0.5  # rough scaling
                    wy = cam_pos[1] + (cy - 240) * 0.5
                    wz = frame.timestamp

                    xs.append(wx)
                    ys.append(wy)
                    zs.append(wz)
                    hover_texts.append(
                        f"ID: {traj.global_id}<br>"
                        f"Camera: {tracklet.camera_id}<br>"
                        f"Time: {frame.timestamp:.1f}s<br>"
                        f"Class: {tracklet.class_name}"
                    )

                # Add gap between tracklets (for visual separation)
                xs.append(None)
                ys.append(None)
                zs.append(None)
                hover_texts.append("")

            fig.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode="lines",
                line=dict(color=color, width=3),
                name=f"ID {traj.global_id} ({traj.class_name})",
                hovertemplate="%{text}<extra></extra>",
                text=hover_texts,
            ))

        # Plot camera positions
        cam_xs, cam_ys, cam_labels = [], [], []
        for cam_id, (cx, cy) in camera_positions.items():
            cam_xs.append(cx)
            cam_ys.append(cy)
            cam_labels.append(cam_id)

        if cam_xs:
            fig.add_trace(go.Scatter3d(
                x=cam_xs, y=cam_ys, z=[0] * len(cam_xs),
                mode="markers+text",
                marker=dict(size=8, color="black", symbol="diamond"),
                text=cam_labels,
                textposition="top center",
                name="Cameras",
                showlegend=True,
            ))

        fig.update_layout(
            title=dict(text=title, font=dict(size=16)),
            scene=dict(
                xaxis_title="X Position",
                yaxis_title="Y Position",
                zaxis_title="Time (seconds)",
                aspectmode="manual",
                aspectratio=dict(x=1, y=1, z=2),
            ),
            height=700,
            template="plotly_white",
        )

        return fig

    def _get_camera_positions(
        self, trajectories: List[GlobalTrajectory]
    ) -> Dict[str, Tuple[float, float]]:
        """Get or compute camera positions."""
        if self.camera_layout:
            return self.camera_layout

        # Auto-layout: arrange cameras in a circle
        all_cameras = set()
        for traj in trajectories:
            for t in traj.tracklets:
                all_cameras.add(t.camera_id)

        positions = {}
        n = max(len(all_cameras), 1)
        for i, cam_id in enumerate(sorted(all_cameras)):
            angle = 2 * np.pi * i / n
            positions[cam_id] = (500 * np.cos(angle), 500 * np.sin(angle))

        return positions
