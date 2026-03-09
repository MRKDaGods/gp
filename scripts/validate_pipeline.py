"""Validate pipeline outputs stage-by-stage across multiple datasets.

Runs the full pipeline on each configured dataset then prints a per-stage
diagnostic report so you can spot exactly where quality drops off.

Usage:
    python scripts/validate_pipeline.py --datasets wildtrack cityflowv2
    python scripts/validate_pipeline.py --datasets wildtrack --stages 0,1,2,3,4,5
    python scripts/validate_pipeline.py --datasets wildtrack cityflowv2 --skip-run
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import click
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

console = Console()

# ─── Config datasets ────────────────────────────────────────────────────

DATASET_CONFIGS: Dict[str, str] = {
    "wildtrack": "configs/datasets/wildtrack.yaml",
    "cityflowv2": "configs/datasets/cityflowv2.yaml",
    "epfl_lab": "configs/datasets/epfl_lab.yaml",
}


# ─── Stage validators ───────────────────────────────────────────────────

def validate_stage0(run_dir: Path) -> Dict:
    """Check frame extraction outputs."""
    stage_dir = run_dir / "stage0"
    manifest = stage_dir / "frames_manifest.json"
    if not manifest.exists():
        return {"status": "SKIP", "reason": "no manifest"}

    frames = json.loads(manifest.read_text())
    n_frames = len(frames)
    cameras = set(f["camera_id"] for f in frames)
    per_cam = {c: sum(1 for f in frames if f["camera_id"] == c) for c in cameras}

    # Check frame files exist
    missing = sum(1 for f in frames if not Path(f["frame_path"]).exists())

    return {
        "status": "OK" if missing == 0 else "WARN",
        "total_frames": n_frames,
        "cameras": len(cameras),
        "per_camera": per_cam,
        "missing_files": missing,
    }


def validate_stage1(run_dir: Path) -> Dict:
    """Check tracking outputs."""
    stage_dir = run_dir / "stage1"
    tracklet_files = sorted(stage_dir.glob("tracklets_*.json"))
    if not tracklet_files:
        return {"status": "SKIP", "reason": "no tracklet files"}

    total_tracklets = 0
    total_detections = 0
    per_cam = {}
    short_tracklets = 0

    for f in tracklet_files:
        cam_id = f.stem.replace("tracklets_", "")
        data = json.loads(f.read_text())
        n = len(data)
        total_tracklets += n
        det = sum(len(t["frames"]) for t in data)
        total_detections += det
        short = sum(1 for t in data if len(t["frames"]) < 5)
        short_tracklets += short
        per_cam[cam_id] = {"tracklets": n, "detections": det, "short": short}

    avg_len = total_detections / max(total_tracklets, 1)

    return {
        "status": "OK",
        "total_tracklets": total_tracklets,
        "total_detections": total_detections,
        "avg_tracklet_length": round(avg_len, 1),
        "short_tracklets_lt5": short_tracklets,
        "cameras": len(per_cam),
        "per_camera": per_cam,
    }


def validate_stage2(run_dir: Path) -> Dict:
    """Check feature extraction outputs."""
    stage_dir = run_dir / "stage2"
    emb_path = stage_dir / "embeddings.npy"
    hsv_path = stage_dir / "hsv_features.npy"
    idx_path = stage_dir / "embedding_index.json"

    if not emb_path.exists():
        return {"status": "SKIP", "reason": "no embeddings"}

    embeddings = np.load(emb_path)
    index_map = json.loads(idx_path.read_text()) if idx_path.exists() else []

    result = {
        "status": "OK",
        "n_embeddings": embeddings.shape[0],
        "embedding_dim": embeddings.shape[1],
    }

    # Check L2 norms (should be ~1.0 if normalised)
    norms = np.linalg.norm(embeddings, axis=1)
    result["norm_mean"] = round(float(norms.mean()), 4)
    result["norm_std"] = round(float(norms.std()), 4)
    result["norm_ok"] = bool(abs(norms.mean() - 1.0) < 0.05)

    # Check for duplicate/collapsed embeddings
    if embeddings.shape[0] > 1:
        # Sample pairwise cosine sims
        n = min(embeddings.shape[0], 500)
        sample = embeddings[:n]
        sim_matrix = sample @ sample.T
        np.fill_diagonal(sim_matrix, 0)
        result["mean_pairwise_sim"] = round(float(sim_matrix.mean()), 4)
        result["max_pairwise_sim"] = round(float(sim_matrix.max()), 4)
        # If mean pairwise sim is too high, embeddings are not discriminative
        result["discriminative"] = bool(sim_matrix.mean() < 0.5)

    if hsv_path.exists():
        hsv = np.load(hsv_path)
        result["hsv_dim"] = hsv.shape[1] if len(hsv.shape) > 1 else 0
        hsv_norms = np.linalg.norm(hsv, axis=1)
        result["hsv_norm_ok"] = bool(abs(hsv_norms.mean() - 1.0) < 0.05)

    # Per-camera counts
    if index_map:
        cameras = set(m["camera_id"] for m in index_map)
        result["cameras"] = len(cameras)
        result["per_camera"] = {
            c: sum(1 for m in index_map if m["camera_id"] == c) for c in cameras
        }

    return result


def validate_stage3(run_dir: Path) -> Dict:
    """Check FAISS index."""
    stage_dir = run_dir / "stage3"
    index_path = stage_dir / "faiss_index.bin"
    db_path = stage_dir / "metadata.db"

    if not index_path.exists():
        return {"status": "SKIP", "reason": "no FAISS index"}

    import faiss
    index = faiss.read_index(str(index_path))

    return {
        "status": "OK",
        "index_size": index.ntotal,
        "dimension": index.d,
        "metadata_db": db_path.exists(),
    }


def validate_stage4(run_dir: Path) -> Dict:
    """Check association outputs."""
    traj_path = run_dir / "stage4" / "global_trajectories.json"
    if not traj_path.exists():
        return {"status": "SKIP", "reason": "no trajectories"}

    data = json.loads(traj_path.read_text())
    n_traj = len(data)
    total_tracklets = sum(len(t["tracklets"]) for t in data)
    multi_cam = sum(1 for t in data if len(set(tk["camera_id"] for tk in t["tracklets"])) > 1)
    singletons = sum(1 for t in data if len(t["tracklets"]) == 1)

    # Camera coverage
    all_cameras = set()
    for t in data:
        for tk in t["tracklets"]:
            all_cameras.add(tk["camera_id"])

    sizes = [len(t["tracklets"]) for t in data]

    return {
        "status": "OK",
        "global_identities": n_traj,
        "total_tracklets_linked": total_tracklets,
        "multi_camera_trajectories": multi_cam,
        "singletons": singletons,
        "cameras_covered": len(all_cameras),
        "avg_tracklets_per_id": round(total_tracklets / max(n_traj, 1), 2),
        "max_tracklets_per_id": max(sizes) if sizes else 0,
    }


def validate_stage5(run_dir: Path) -> Dict:
    """Check evaluation results."""
    eval_path = run_dir / "stage5" / "evaluation_report.json"
    if not eval_path.exists():
        return {"status": "SKIP", "reason": "no evaluation report"}

    result = json.loads(eval_path.read_text())

    out = {
        "status": "OK",
        "MOTA": round(result.get("mota", 0), 4),
        "IDF1": round(result.get("idf1", 0), 4),
        "HOTA": round(result.get("hota", 0), 4),
        "ID_Switches": result.get("id_switches", 0),
    }

    # Per-camera if available
    details = result.get("details", {})
    per_cam = details.get("per_camera", {})
    if per_cam:
        out["per_camera"] = per_cam

    return out


STAGE_VALIDATORS = {
    0: ("Ingestion", validate_stage0),
    1: ("Tracking", validate_stage1),
    2: ("Features", validate_stage2),
    3: ("Indexing", validate_stage3),
    4: ("Association", validate_stage4),
    5: ("Evaluation", validate_stage5),
}


# ─── Pipeline runner ────────────────────────────────────────────────────

def run_pipeline_for_dataset(dataset: str, stages: List[int]) -> Path:
    """Run the pipeline for a dataset and return the output directory."""
    from src.core.config import load_config, save_config
    from src.core.logging_utils import setup_logging

    config_path = "configs/default.yaml"
    dataset_config = DATASET_CONFIGS.get(dataset)
    if not dataset_config:
        raise ValueError(f"Unknown dataset: {dataset}")

    cfg = load_config(config_path, dataset_config=dataset_config)

    run_name = f"{dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_base = Path(cfg.project.output_dir) / run_name
    output_base.mkdir(parents=True, exist_ok=True)

    setup_logging(level="INFO", log_file=output_base / "pipeline.log")
    save_config(cfg, output_base / "config.yaml")

    # Save dataset name for the dashboard
    (output_base / "dataset_name.txt").write_text(dataset)

    console.print(Panel(
        f"[bold]{dataset.upper()}[/bold]\n"
        f"Stages: {stages}\nOutput: {output_base}",
        title="Pipeline Run",
    ))

    # Import stages lazily
    video_paths = _discover_video_paths(cfg)

    frames = None
    tracklets_by_camera = None
    features = None
    faiss_index = None
    metadata_store = None
    trajectories = None

    if 0 in stages:
        console.print("[cyan]Stage 0: Ingestion[/cyan]")
        from src.stage0_ingestion import run_stage0
        frames = run_stage0(cfg, output_dir=output_base / "stage0")

    if 1 in stages:
        console.print("[cyan]Stage 1: Tracking[/cyan]")
        from src.stage1_tracking import run_stage1
        if frames is None:
            from src.core.io_utils import load_frame_manifest
            frames = load_frame_manifest(output_base / "stage0" / "frames_manifest.json")
        tracklets_by_camera = run_stage1(cfg, frames, output_dir=output_base / "stage1")

    if 2 in stages:
        console.print("[cyan]Stage 2: Features[/cyan]")
        from src.stage2_features import run_stage2
        if tracklets_by_camera is None:
            from src.core.io_utils import load_tracklets_by_camera
            tracklets_by_camera = load_tracklets_by_camera(output_base / "stage1")
        features = run_stage2(cfg, tracklets_by_camera, video_paths, output_dir=output_base / "stage2")

    if 3 in stages:
        console.print("[cyan]Stage 3: Indexing[/cyan]")
        from src.stage3_indexing import run_stage3
        if features is not None and tracklets_by_camera is not None:
            faiss_index, metadata_store = run_stage3(cfg, features, tracklets_by_camera, output_dir=output_base / "stage3")

    if 4 in stages:
        console.print("[cyan]Stage 4: Association[/cyan]")
        from src.stage4_association import run_stage4
        if features is None:
            stage2_dir = output_base / "stage2"
            if (stage2_dir / "embeddings.npy").exists():
                from src.core.io_utils import load_embeddings, load_hsv_features
                from src.core.data_models import TrackletFeatures
                embeddings, index_map = load_embeddings(stage2_dir)
                hsv_matrix = load_hsv_features(stage2_dir)
                features = [
                    TrackletFeatures(
                        track_id=m["track_id"], camera_id=m["camera_id"],
                        class_id=m["class_id"], embedding=embeddings[i],
                        hsv_histogram=hsv_matrix[i],
                    ) for i, m in enumerate(index_map)
                ]
        if faiss_index is None:
            index_path = output_base / "stage3" / "faiss_index.bin"
            if index_path.exists():
                from src.stage3_indexing.faiss_index import FAISSIndex
                faiss_index = FAISSIndex()
                faiss_index.load(index_path)
        if metadata_store is None:
            db_path = output_base / "stage3" / "metadata.db"
            if db_path.exists():
                from src.stage3_indexing.metadata_store import MetadataStore
                metadata_store = MetadataStore(db_path)
        if faiss_index and metadata_store and features:
            if tracklets_by_camera is None:
                from src.core.io_utils import load_tracklets_by_camera
                tracklets_by_camera = load_tracklets_by_camera(output_base / "stage1")
            trajectories = run_stage4(cfg, faiss_index, metadata_store, features, tracklets_by_camera, output_dir=output_base / "stage4")

    if 5 in stages:
        console.print("[cyan]Stage 5: Evaluation[/cyan]")
        from src.stage5_evaluation import run_stage5
        if trajectories is None:
            traj_path = output_base / "stage4" / "global_trajectories.json"
            if traj_path.exists():
                from src.core.io_utils import load_global_trajectories
                trajectories = load_global_trajectories(traj_path)
        if trajectories is not None:
            run_stage5(cfg, trajectories, output_dir=output_base / "stage5")

    return output_base


def _discover_video_paths(cfg) -> dict:
    input_dir = Path(cfg.stage0.input_dir)
    extensions = cfg.stage0.get("video_extensions", [".mp4", ".avi", ".mkv", ".mov"])
    video_paths = {}
    for ext in extensions:
        for vpath in input_dir.rglob(f"*{ext}"):
            relative = vpath.relative_to(input_dir)
            cam_id = relative.parts[-2] if len(relative.parts) > 1 else vpath.stem
            video_paths[cam_id] = str(vpath)
    return video_paths


# ─── Report ─────────────────────────────────────────────────────────────

def print_validation_report(dataset: str, run_dir: Path, stages: List[int]):
    """Print a per-stage diagnostic table."""
    console.print(f"\n[bold yellow]{'='*60}[/bold yellow]")
    console.print(f"[bold yellow]  Validation Report: {dataset.upper()}[/bold yellow]")
    console.print(f"[bold yellow]  Run: {run_dir}[/bold yellow]")
    console.print(f"[bold yellow]{'='*60}[/bold yellow]\n")

    for stage_num in sorted(stages):
        if stage_num not in STAGE_VALIDATORS:
            continue
        name, validator = STAGE_VALIDATORS[stage_num]
        result = validator(run_dir)

        status = result.pop("status", "?")
        color = "green" if status == "OK" else ("yellow" if status == "WARN" else "red")

        table = Table(title=f"Stage {stage_num}: {name} [{status}]", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")

        for key, value in result.items():
            if key == "per_camera":
                # Show per-camera as sub-rows
                for cam_id, cam_val in sorted(value.items()):
                    if isinstance(cam_val, dict):
                        cam_str = ", ".join(f"{k}={v}" for k, v in cam_val.items())
                    else:
                        cam_str = str(cam_val)
                    table.add_row(f"  {cam_id}", cam_str)
            else:
                val_str = str(value)
                style = ""
                if key == "discriminative" and value is False:
                    style = "bold red"
                elif key == "norm_ok" and value is False:
                    style = "bold red"
                table.add_row(key, f"[{style}]{val_str}[/{style}]" if style else val_str)

        console.print(table)
        console.print()


# ─── CLI ────────────────────────────────────────────────────────────────

@click.command()
@click.option("--datasets", "-d", multiple=True, required=True,
              help="Datasets to run: wildtrack, cityflowv2, epfl_lab")
@click.option("--stages", "-s", default="0,1,2,3,4,5",
              help="Comma-separated stage numbers")
@click.option("--skip-run", is_flag=True, default=False,
              help="Skip pipeline run, only validate existing outputs")
@click.option("--run-dir", type=str, default=None,
              help="Existing run dir to validate (with --skip-run)")
def main(datasets: tuple, stages: str, skip_run: bool, run_dir: Optional[str]):
    stage_nums = [int(s.strip()) for s in stages.split(",")]

    for dataset in datasets:
        if dataset not in DATASET_CONFIGS:
            console.print(f"[red]Unknown dataset: {dataset}[/red]")
            console.print(f"Available: {list(DATASET_CONFIGS.keys())}")
            sys.exit(1)

        if skip_run:
            if run_dir:
                output = Path(run_dir)
            else:
                # Find latest run for this dataset
                output_base = Path("data/outputs")
                candidates = sorted(output_base.glob(f"{dataset}_*"), reverse=True)
                if not candidates:
                    console.print(f"[red]No existing runs for {dataset}[/red]")
                    continue
                output = candidates[0]
                console.print(f"Using latest run: {output}")
        else:
            output = run_pipeline_for_dataset(dataset, stage_nums)

        print_validation_report(dataset, output, stage_nums)


if __name__ == "__main__":
    main()
