"""Multi-camera synchronized grid viewer for cross-camera tracking verification.

Usage:
    python scripts/view_multicam.py --run-dir data/outputs/run_XXX --global-id 5
    python scripts/view_multicam.py --run-dir data/outputs/run_XXX --gallery --top-n 10
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.io_utils import load_global_trajectories, load_frame_manifest
from src.stage6_visualization.multicam_grid import (
    MultiCamGridRenderer,
    FrameReader,
    compute_grid_layout,
)

console = Console()


@click.command()
@click.option("--run-dir", "-r", required=True, type=click.Path(exists=True),
              help="Pipeline run output directory")
@click.option("--global-id", "-g", type=int, default=None,
              help="Global trajectory ID to highlight")
@click.option("--gallery", is_flag=True, default=False,
              help="Generate gallery of top trajectories")
@click.option("--top-n", type=int, default=10,
              help="Number of trajectories for gallery mode")
@click.option("--sort-by", type=click.Choice(["num_cameras", "duration", "num_frames"]),
              default="num_cameras", help="Sort criterion for gallery")
@click.option("--panel-width", type=int, default=640,
              help="Width of each camera panel in pixels")
@click.option("--dim-factor", type=float, default=0.4,
              help="Dimming for cameras where target is absent (0=none, 1=black)")
@click.option("--no-trails", is_flag=True, default=False, help="Disable motion trails")
@click.option("--output", "-o", type=str, default=None, help="Output path override")
@click.option("--fps", type=float, default=None,
              help="Output FPS (default: auto-detect from config)")
def main(run_dir, global_id, gallery, top_n, sort_by, panel_width,
         dim_factor, no_trails, output, fps):
    """View multi-camera grid for cross-camera tracking verification."""
    run_dir = Path(run_dir)

    # --- Load data ---
    traj_path = run_dir / "stage4" / "global_trajectories.json"
    if not traj_path.exists():
        console.print(f"[red]No trajectories found at {traj_path}[/red]")
        raise SystemExit(1)

    console.print("[cyan]Loading trajectories...[/cyan]")
    trajectories = load_global_trajectories(traj_path)
    console.print(f"  {len(trajectories)} global trajectories loaded")

    # --- Discover cameras from stage0 ---
    stage0_dir = run_dir / "stage0"
    if not stage0_dir.exists():
        console.print(f"[red]No stage0 directory at {stage0_dir}[/red]")
        raise SystemExit(1)

    manifest = load_frame_manifest(stage0_dir / "frames_manifest.json")
    camera_ids = sorted({f.camera_id for f in manifest})
    console.print(f"  {len(camera_ids)} cameras: {', '.join(camera_ids)}")

    # Detect source resolution from manifest
    source_w = manifest[0].width
    source_h = manifest[0].height

    # Auto-detect FPS from saved config
    if fps is None:
        cfg_path = run_dir / "config.yaml"
        if cfg_path.exists():
            from omegaconf import OmegaConf
            cfg = OmegaConf.load(str(cfg_path))
            fps = cfg.get("stage0", {}).get("output_fps", 10.0)
        else:
            fps = 10.0
    console.print(f"  Output FPS: {fps}")

    # --- Build layout and renderer ---
    layout = compute_grid_layout(camera_ids, source_w, source_h, panel_width)
    console.print(
        f"  Grid: {layout.cols}x{layout.rows} "
        f"({layout.panel_w}x{layout.panel_h} per panel, "
        f"{layout.canvas_w}x{layout.canvas_h} canvas)"
    )

    reader = FrameReader(stage0_dir, (layout.panel_w, layout.panel_h))
    renderer = MultiCamGridRenderer(
        layout=layout,
        output_fps=fps,
        dim_factor=dim_factor,
        draw_trails=not no_trails,
    )

    out_dir = run_dir / "multicam"

    # --- Single trajectory mode ---
    if global_id is not None:
        traj = next((t for t in trajectories if t.global_id == global_id), None)
        if traj is None:
            console.print(f"[red]Global ID {global_id} not found[/red]")
            valid = sorted(t.global_id for t in trajectories)[:20]
            console.print(f"  Available IDs (first 20): {valid}")
            raise SystemExit(1)

        out_path = Path(output) if output else out_dir / f"grid_gid{global_id}.mp4"
        console.print(
            f"\n[bold]Rendering GID {global_id} ({traj.class_name}, "
            f"{traj.num_cameras} cameras, {traj.total_duration:.1f}s)[/bold]"
        )
        renderer.render_trajectory(traj, reader, out_path)
        console.print(f"\n[green]Output: {out_path}[/green]")
        return

    # --- Gallery mode ---
    if gallery:
        console.print(f"\n[bold]Rendering gallery: top {top_n} by {sort_by}[/bold]")
        out_path = Path(output) if output else out_dir
        outputs = renderer.render_gallery(
            trajectories, reader, out_path, top_n=top_n, sort_by=sort_by,
        )
        console.print(f"\n[green]{len(outputs)} videos saved to {out_path}[/green]")
        return

    # --- Default: list available trajectories ---
    console.print("\n[yellow]No --global-id or --gallery specified. Listing trajectories:[/yellow]")
    ranked = sorted(trajectories, key=lambda t: t.num_cameras, reverse=True)[:20]
    for t in ranked:
        n_frames = sum(len(tk.frames) for tk in t.tracklets)
        console.print(
            f"  GID {t.global_id:>4d}  {t.class_name:<8s}  "
            f"{t.num_cameras} cams  {t.total_duration:>6.1f}s  {n_frames:>5d} frames"
        )
    console.print("\nUse --global-id N or --gallery to render.")


if __name__ == "__main__":
    main()
