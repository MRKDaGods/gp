"""Entry point: run the full MTMC tracking pipeline or selected stages.

Usage:
    python scripts/run_pipeline.py --config configs/default.yaml
    python scripts/run_pipeline.py --config configs/default.yaml --stages 0,1,2
    python scripts/run_pipeline.py --config configs/default.yaml --smoke-test
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel

# Ensure project root on path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.config import load_config, save_config
from src.core.logging_utils import setup_logging

console = Console()


@click.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Config YAML path")
@click.option("--dataset-config", "-d", type=click.Path(exists=True), default=None, help="Dataset config")
@click.option("--stages", "-s", default="0,1,2,3,4,5,6", help="Comma-separated stage numbers to run")
@click.option("--smoke-test", is_flag=True, default=False, help="Run on tiny data subset")
@click.option("--override", "-o", multiple=True, help="Config overrides in dotlist format")
def main(config: str, dataset_config: str, stages: str, smoke_test: bool, override: tuple):
    """Run the MTMC tracking pipeline."""
    # Parse stages
    stage_nums = [int(s.strip()) for s in stages.split(",")]

    # Load config
    cfg = load_config(config, overrides=list(override), dataset_config=dataset_config)

    # Setup run directory
    run_name = cfg.project.get("run_name") or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_base = Path(cfg.project.output_dir) / run_name

    setup_logging(level=cfg.project.get("log_level", "INFO"), log_file=output_base / "pipeline.log")
    save_config(cfg, output_base / "config.yaml")

    console.print(Panel(
        f"[bold]MTMC Tracking Pipeline[/bold]\n"
        f"Config: {config}\n"
        f"Stages: {stage_nums}\n"
        f"Output: {output_base}\n"
        f"Smoke test: {smoke_test}",
        title="Pipeline Start",
    ))

    # Stage data passed between stages
    frames = None
    tracklets_by_camera = None
    features = None
    faiss_index = None
    metadata_store = None
    trajectories = None

    # Discover video paths (needed by multiple stages)
    video_paths = _discover_video_paths(cfg)

    # --- Stage 0: Ingestion ---
    if 0 in stage_nums:
        console.print("\n[bold cyan]Stage 0: Ingestion & Pre-Processing[/bold cyan]")
        from src.stage0_ingestion import run_stage0

        frames = run_stage0(cfg, output_dir=output_base / "stage0", smoke_test=smoke_test)

    # --- Stage 1: Detection & Tracking ---
    if 1 in stage_nums:
        console.print("\n[bold cyan]Stage 1: Per-Camera Detection & Tracking[/bold cyan]")
        from src.stage1_tracking import run_stage1

        if frames is None:
            from src.core.io_utils import load_frame_manifest
            frames = load_frame_manifest(output_base / "stage0" / "frames_manifest.json")

        tracklets_by_camera = run_stage1(
            cfg, frames, output_dir=output_base / "stage1", smoke_test=smoke_test
        )

    # --- Stage 2: Feature Extraction ---
    if 2 in stage_nums:
        console.print("\n[bold cyan]Stage 2: Feature Extraction & Refinement[/bold cyan]")
        from src.stage2_features import run_stage2

        if tracklets_by_camera is None:
            from src.core.io_utils import load_tracklets_by_camera
            tracklets_by_camera = load_tracklets_by_camera(output_base / "stage1")

        features = run_stage2(
            cfg, tracklets_by_camera, video_paths,
            output_dir=output_base / "stage2", smoke_test=smoke_test,
            stage0_dir=output_base / "stage0",
        )

    # --- Stage 3: Indexing ---
    if 3 in stage_nums:
        console.print("\n[bold cyan]Stage 3: Indexing & Storage[/bold cyan]")
        from src.stage3_indexing import run_stage3

        # Load stage 2 features from disk if not already in memory
        if features is None:
            stage2_dir = output_base / "stage2"
            if (stage2_dir / "embeddings.npy").exists():
                from src.core.io_utils import load_embeddings, load_hsv_features
                from src.core.data_models import TrackletFeatures
                embeddings, index_map = load_embeddings(stage2_dir)
                hsv_matrix = load_hsv_features(stage2_dir)
                features = [
                    TrackletFeatures(
                        track_id=m["track_id"],
                        camera_id=m["camera_id"],
                        class_id=m["class_id"],
                        embedding=embeddings[i],
                        hsv_histogram=hsv_matrix[i],
                    )
                    for i, m in enumerate(index_map)
                ]
                console.print(f"  Loaded {len(features)} features from disk")

        if features is None:
            console.print("[yellow]Stage 3 requires features from Stage 2. Skipping.[/yellow]")
        else:
            if tracklets_by_camera is None:
                from src.core.io_utils import load_tracklets_by_camera
                tracklets_by_camera = load_tracklets_by_camera(output_base / "stage1")
            faiss_index, metadata_store = run_stage3(
                cfg, features, tracklets_by_camera, output_dir=output_base / "stage3"
            )

    # --- Stage 4: Cross-Camera Association ---
    if 4 in stage_nums:
        console.print("\n[bold cyan]Stage 4: Multi-Camera Association[/bold cyan]")
        from src.stage4_association import run_stage4

        # Load Stage 2-3 artifacts from disk if not in memory
        if features is None:
            stage2_dir = output_base / "stage2"
            if (stage2_dir / "embeddings.npy").exists():
                from src.core.io_utils import load_embeddings, load_hsv_features
                from src.core.data_models import TrackletFeatures
                embeddings, index_map = load_embeddings(stage2_dir)
                hsv_matrix = load_hsv_features(stage2_dir)
                features = [
                    TrackletFeatures(
                        track_id=m["track_id"],
                        camera_id=m["camera_id"],
                        class_id=m["class_id"],
                        embedding=embeddings[i],
                        hsv_histogram=hsv_matrix[i],
                    )
                    for i, m in enumerate(index_map)
                ]
                console.print(f"  Loaded {len(features)} features from disk")

        if faiss_index is None:
            index_path = output_base / "stage3" / "faiss_index.bin"
            if index_path.exists():
                from src.stage3_indexing.faiss_index import FAISSIndex
                faiss_index = FAISSIndex(cfg.stage3.faiss.get("index_type", "flat_ip"))
                faiss_index.load(index_path)
                console.print("  Loaded FAISS index from disk")

        if metadata_store is None:
            db_path = output_base / "stage3" / "metadata.db"
            if db_path.exists():
                from src.stage3_indexing.metadata_store import MetadataStore
                metadata_store = MetadataStore(db_path)
                console.print("  Loaded metadata store from disk")

        if faiss_index is None or metadata_store is None or features is None:
            console.print("[yellow]Stage 4 requires Stages 2-3. Skipping.[/yellow]")
        else:
            if tracklets_by_camera is None:
                from src.core.io_utils import load_tracklets_by_camera
                tracklets_by_camera = load_tracklets_by_camera(output_base / "stage1")

            query_cameras = None

            if cfg.stage4.get("global_gallery", {}).get("enabled", False):
                gallery_run_id = cfg.stage4.global_gallery.get("run_id", "")
                if gallery_run_id:
                    gallery_dir = Path(cfg.project.output_dir) / gallery_run_id
                    console.print(f"  [cyan]Loading global gallery from {gallery_run_id}[/cyan]")
                    try:
                        from src.core.io_utils import load_embeddings, load_hsv_features, load_tracklets_by_camera
                        from src.core.data_models import TrackletFeatures
                        from src.stage3_indexing import run_stage3

                        # Load gallery stage 2
                        gal_embeddings, gal_index_map = load_embeddings(gallery_dir / "stage2")
                        gal_hsv_matrix = load_hsv_features(gallery_dir / "stage2")

                        # Dimension compatibility check — probe and gallery must share the
                        # same embedding dimension (same ReID model).  If they differ, skip
                        # the gallery merge rather than crash with a cryptic numpy error.
                        if features and gal_embeddings.shape[1] != features[0].embedding.shape[0]:
                            probe_dim = features[0].embedding.shape[0]
                            gal_dim = gal_embeddings.shape[1]
                            console.print(
                                f"[yellow]  ⚠ Gallery embedding dim ({gal_dim}) ≠ probe dim "
                                f"({probe_dim}) — skipping gallery merge. "
                                f"Re-compute the gallery with the same ReID model.[/yellow]"
                            )
                            raise ValueError(
                                f"Embedding dimension mismatch: gallery={gal_dim}, probe={probe_dim}"
                            )

                        gal_features = [
                            TrackletFeatures(
                                track_id=m["track_id"],
                                camera_id=m["camera_id"],
                                class_id=m["class_id"],
                                embedding=gal_embeddings[i],
                                hsv_histogram=gal_hsv_matrix[i],
                            )
                            for i, m in enumerate(gal_index_map)
                        ]

                        # Load gallery tracklets
                        gal_tracklets_by_camera = load_tracklets_by_camera(gallery_dir / "stage1")

                        # Prefix current run cameras to avoid collision (current run = query)
                        query_cameras = set()
                        new_features = []
                        for f in features:
                            new_cam_id = f"query_{f.camera_id}"
                            f.camera_id = new_cam_id
                            query_cameras.add(new_cam_id)
                            new_features.append(f)

                        new_tracklets_by_camera = {}
                        for cam_id, tracklets in tracklets_by_camera.items():
                            new_cam_id = f"query_{cam_id}"
                            for t in tracklets:
                                t.camera_id = new_cam_id
                            new_tracklets_by_camera[new_cam_id] = tracklets

                        # Merge — do this last so features is never half-mutated on error
                        merged_features = gal_features + new_features
                        merged_tracklets = dict(new_tracklets_by_camera)
                        merged_tracklets.update(gal_tracklets_by_camera)

                        # Re-run Stage 3 in memory for the combined set
                        combined_out = output_base / "stage3_combined"
                        combined_faiss, combined_meta = run_stage3(
                            cfg, merged_features, merged_tracklets, output_dir=combined_out
                        )

                        # Only commit to the merged state after everything succeeded
                        features = merged_features
                        tracklets_by_camera = merged_tracklets
                        faiss_index = combined_faiss
                        metadata_store = combined_meta

                        console.print(f"  [cyan]Merged query features with gallery: total {len(features)} features[/cyan]")
                    except Exception as e:
                        console.print(f"[red]Failed to load or merge global gallery: {e}[/red]")

            trajectories = run_stage4(
                cfg, faiss_index, metadata_store, features,
                tracklets_by_camera, output_dir=output_base / "stage4",
                query_cameras=query_cameras
            )

    # --- Stage 5: Evaluation ---
    if 5 in stage_nums:
        console.print("\n[bold cyan]Stage 5: Evaluation[/bold cyan]")
        from src.stage5_evaluation import run_stage5

        if trajectories is None:
            from src.core.io_utils import load_global_trajectories
            traj_path = output_base / "stage4" / "global_trajectories.json"
            if traj_path.exists():
                trajectories = load_global_trajectories(traj_path)
            else:
                console.print("[yellow]No trajectories found. Skipping evaluation.[/yellow]")

        if trajectories is not None:
            run_stage5(cfg, trajectories, output_dir=output_base / "stage5")

    # --- Stage 6: Visualization ---
    if 6 in stage_nums:
        console.print("\n[bold cyan]Stage 6: Visualization & Outputs[/bold cyan]")
        from src.stage6_visualization import run_stage6

        if trajectories is None:
            from src.core.io_utils import load_global_trajectories
            traj_path = output_base / "stage4" / "global_trajectories.json"
            if traj_path.exists():
                trajectories = load_global_trajectories(traj_path)

        if tracklets_by_camera is None:
            from src.core.io_utils import load_tracklets_by_camera
            stage1_dir = output_base / "stage1"
            if stage1_dir.exists():
                tracklets_by_camera = load_tracklets_by_camera(stage1_dir)

        if trajectories is not None and tracklets_by_camera is not None:
            run_stage6(
                cfg, trajectories, tracklets_by_camera, video_paths,
                output_dir=output_base / "stage6"
            )
        else:
            console.print("[yellow]Missing data for visualization. Skipping.[/yellow]")

    console.print(Panel(
        f"[bold green]Pipeline complete![/bold green]\nOutputs: {output_base}",
        title="Done",
    ))


def _discover_video_paths(cfg) -> dict:
    """Build a camera_id -> video_path mapping from the input directory."""
    input_dir = Path(cfg.stage0.input_dir)
    extensions = cfg.stage0.get("video_extensions", [".mp4", ".avi", ".mkv", ".mov"])
    video_paths = {}

    for ext in extensions:
        for vpath in input_dir.rglob(f"*{ext}"):
            relative = vpath.relative_to(input_dir)
            if len(relative.parts) > 1:
                cam_id = relative.parts[-2]
            else:
                cam_id = vpath.stem
            video_paths[cam_id] = str(vpath)

    return video_paths


if __name__ == "__main__":
    main()
