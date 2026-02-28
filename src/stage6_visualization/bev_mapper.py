"""Bird's-eye-view (BEV) trajectory map generation."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from src.core.data_models import GlobalTrajectory


class BEVMapper:
    """Generates bird's-eye-view trajectory maps.

    Projects tracklet bounding box centers onto a 2D plane and draws
    trajectories color-coded by global identity.
    """

    def __init__(
        self,
        map_image: Optional[str] = None,
        scale: float = 1.0,
        figsize: Tuple[int, int] = (14, 10),
    ):
        self.map_image = map_image
        self.scale = scale
        self.figsize = figsize

    def plot_trajectories(
        self,
        trajectories: List[GlobalTrajectory],
        camera_positions: Optional[Dict[str, Tuple[float, float]]] = None,
        title: str = "Bird's-Eye View — Global Trajectories",
    ) -> plt.Figure:
        """Plot BEV trajectory map.

        Without geometric calibration, uses bbox bottom-center as a proxy
        for ground-plane position within each camera's local coordinate system.

        Args:
            trajectories: Global trajectories to visualize.
            camera_positions: Optional dict[camera_id, (x, y)] for camera layout.
            title: Plot title.

        Returns:
            matplotlib Figure.
        """
        fig, ax = plt.subplots(1, 1, figsize=self.figsize)

        if self.map_image:
            try:
                img = plt.imread(self.map_image)
                ax.imshow(img, extent=[0, img.shape[1], 0, img.shape[0]], alpha=0.3)
            except Exception as e:
                logger.warning(f"Could not load map image: {e}")

        # Assign each camera an offset on the BEV plane
        if camera_positions is None:
            all_cameras = set()
            for traj in trajectories:
                for t in traj.tracklets:
                    all_cameras.add(t.camera_id)
            camera_positions = {}
            for i, cam_id in enumerate(sorted(all_cameras)):
                angle = 2 * np.pi * i / max(len(all_cameras), 1)
                camera_positions[cam_id] = (500 + 400 * np.cos(angle), 500 + 400 * np.sin(angle))

        # Plot camera positions
        for cam_id, (cx, cy) in camera_positions.items():
            ax.plot(cx, cy, "ks", markersize=10)
            ax.annotate(cam_id, (cx, cy), textcoords="offset points",
                        xytext=(5, 5), fontsize=8, fontweight="bold")

        # Color palette
        cmap = plt.cm.get_cmap("tab20")

        for traj in trajectories:
            color = cmap(traj.global_id % 20)

            for tracklet in traj.tracklets:
                cam_offset = camera_positions.get(tracklet.camera_id, (0, 0))

                # Use bbox bottom-center as position proxy
                points = []
                for f in tracklet.frames:
                    x1, y1, x2, y2 = f.bbox
                    bx = (x1 + x2) / 2 * self.scale + cam_offset[0] - 250
                    by = y2 * self.scale + cam_offset[1] - 250
                    points.append((bx, by))

                if points:
                    xs, ys = zip(*points)
                    ax.plot(xs, ys, "-", color=color, alpha=0.6, linewidth=1.5)
                    ax.plot(xs[0], ys[0], "o", color=color, markersize=4)
                    ax.plot(xs[-1], ys[-1], "s", color=color, markersize=4)

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect("equal")
        fig.tight_layout()

        return fig
