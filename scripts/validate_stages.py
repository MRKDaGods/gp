"""Stage-by-stage diagnostic validator.

Reads outputs from a completed (or partial) pipeline run and prints a
detailed quality report for each stage.  Useful for debugging metric
drops and verifying that each stage produces sensible outputs.

Usage:
    python scripts/validate_stages.py --run-dir data/outputs/epfl_lab_v3
    python scripts/validate_stages.py --run-dir data/outputs/epfl_lab_v3 --stage 2
"""

from __future__ import annotations

import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

import click
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

console = Console()

# ── Colour helpers ────────────────────────────────────────────────────────

OK = "[bold green]OK[/]"
WARN = "[bold yellow]WARN[/]"
FAIL = "[bold red]FAIL[/]"


def grade(val: float, ok_lo: float, ok_hi: float, label: str = "") -> str:
    """Return a coloured grade for *val*."""
    if ok_lo <= val <= ok_hi:
        return f"{OK}  {label}{val:.4f}" if isinstance(val, float) else f"{OK}  {label}{val}"
    return f"{WARN}  {label}{val:.4f}" if isinstance(val, float) else f"{WARN}  {label}{val}"


# ── Stage 0 ───────────────────────────────────────────────────────────────

def validate_stage0(run_dir: Path):
    console.print(Panel("[bold cyan]Stage 0 — Ingestion & Pre-Processing[/]"))
    stage_dir = run_dir / "stage0"
    manifest = stage_dir / "frames_manifest.json"

    if not manifest.exists():
        console.print(f"  {FAIL}  frames_manifest.json not found")
        return

    with open(manifest) as f:
        frames = json.load(f)

    n_frames = len(frames)
    cameras = defaultdict(int)
    for fr in frames:
        cameras[fr.get("camera_id", "?")] += 1

    tbl = Table(title="Frame Extraction Summary")
    tbl.add_column("Camera")
    tbl.add_column("Frames", justify="right")
    tbl.add_column("Status")
    for cam, cnt in sorted(cameras.items()):
        status = OK if cnt > 10 else WARN
        tbl.add_row(cam, str(cnt), status)
    tbl.add_row("[bold]TOTAL[/]", f"[bold]{n_frames}[/]", "")
    console.print(tbl)

    # Check frame files on disk
    for cam in sorted(cameras):
        cam_dir = stage_dir / cam
        if cam_dir.is_dir():
            jpgs = list(cam_dir.glob("frame_*.jpg"))
            pngs = list(cam_dir.glob("frame_*.png"))
            disk_cnt = len(jpgs) + len(pngs)
            match = OK if disk_cnt == cameras[cam] else WARN
            fmt = "PNG" if pngs else "JPEG"
            console.print(f"  {cam}: {disk_cnt} files on disk ({fmt}) {match}")

            # Sample a frame to check resolution
            sample = jpgs[0] if jpgs else (pngs[0] if pngs else None)
            if sample:
                import cv2
                img = cv2.imread(str(sample))
                if img is not None:
                    h, w = img.shape[:2]
                    console.print(f"    Resolution: {w}x{h}")
                    # Check CLAHE (higher contrast = higher std in luma)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    console.print(f"    Luma mean={gray.mean():.1f}  std={gray.std():.1f}")


# ── Stage 1 ───────────────────────────────────────────────────────────────

def validate_stage1(run_dir: Path):
    console.print(Panel("[bold cyan]Stage 1 — Detection & Tracking[/]"))
    stage_dir = run_dir / "stage1"

    if not stage_dir.exists():
        console.print(f"  {FAIL}  stage1/ not found")
        return

    tracklet_files = sorted(stage_dir.glob("tracklets_*.json"))
    if not tracklet_files:
        console.print(f"  {FAIL}  No tracklet files found")
        return

    total_tracklets = 0
    total_frames = 0

    tbl = Table(title="Per-Camera Tracking Summary")
    tbl.add_column("Camera")
    tbl.add_column("Tracklets", justify="right")
    tbl.add_column("Avg Frames", justify="right")
    tbl.add_column("Min Frames", justify="right")
    tbl.add_column("Max Frames", justify="right")
    tbl.add_column("Avg Conf", justify="right")

    for tf in tracklet_files:
        with open(tf) as f:
            tracklets = json.load(f)

        cam_id = tf.stem.replace("tracklets_", "")
        n = len(tracklets)
        total_tracklets += n

        if n == 0:
            tbl.add_row(cam_id, "0", "-", "-", "-", "-")
            continue

        frame_counts = [len(t.get("frames", [])) for t in tracklets]
        confs = []
        for t in tracklets:
            for fr in t.get("frames", []):
                confs.append(fr.get("confidence", 0))

        total_frames += sum(frame_counts)
        avg_f = np.mean(frame_counts)
        avg_c = np.mean(confs) if confs else 0

        tbl.add_row(
            cam_id, str(n),
            f"{avg_f:.1f}", str(min(frame_counts)), str(max(frame_counts)),
            f"{avg_c:.2f}",
        )

    console.print(tbl)
    console.print(f"  Total tracklets: [bold]{total_tracklets}[/]")
    console.print(f"  Total detection frames: [bold]{total_frames}[/]")

    # Fragmentation analysis
    if total_tracklets > 0:
        all_lengths = []
        for tf in tracklet_files:
            with open(tf) as f:
                tracklets = json.load(f)
            for t in tracklets:
                all_lengths.append(len(t.get("frames", [])))
        short = sum(1 for l in all_lengths if l < 10)
        console.print(
            f"  Short tracklets (<10 frames): {short}/{total_tracklets} "
            f"({100 * short / total_tracklets:.0f}%) "
            f"{'— high fragmentation!' if short > total_tracklets * 0.5 else ''}"
        )


# ── Stage 2 ───────────────────────────────────────────────────────────────

def validate_stage2(run_dir: Path):
    console.print(Panel("[bold cyan]Stage 2 — Feature Extraction[/]"))
    stage_dir = run_dir / "stage2"

    emb_path = stage_dir / "embeddings.npy"
    hsv_path = stage_dir / "hsv_features.npy"
    idx_path = stage_dir / "embedding_index.json"

    if not emb_path.exists():
        console.print(f"  {FAIL}  embeddings.npy not found")
        return

    embeddings = np.load(emb_path)
    console.print(f"  Embedding matrix: shape={embeddings.shape}, dtype={embeddings.dtype}")

    # Norm distribution — should be ~1.0 if L2-normalized
    norms = np.linalg.norm(embeddings, axis=1)
    console.print(
        f"  L2 norms: mean={norms.mean():.4f}  std={norms.std():.4f}  "
        f"min={norms.min():.4f}  max={norms.max():.4f}"
    )
    if 0.99 < norms.mean() < 1.01 and norms.std() < 0.01:
        console.print(f"  {OK}  Embeddings are properly L2-normalized")
    else:
        console.print(f"  {WARN}  Embeddings may not be L2-normalized!")

    # Intra-class similarity (self-similarity distribution)
    # Compute pairwise similarities for a random sample
    n = embeddings.shape[0]
    if n > 1:
        sim_matrix = embeddings @ embeddings.T
        np.fill_diagonal(sim_matrix, 0)
        upper = sim_matrix[np.triu_indices(n, k=1)]
        console.print(
            f"  Pairwise cosine sim: mean={upper.mean():.4f}  "
            f"std={upper.std():.4f}  max={upper.max():.4f}  "
            f"min={upper.min():.4f}"
        )
        # Ideal: mean ~0.2-0.4, good discrimination means wide spread
        if upper.std() < 0.05:
            console.print(f"  {WARN}  Very low variance — embeddings may lack discriminability")
        elif upper.std() > 0.15:
            console.print(f"  {OK}  Good embedding spread (std={upper.std():.3f})")
        else:
            console.print(f"  {OK}  Moderate embedding spread")

        # Top-1 retrieval analysis: for each embedding, what's the similarity
        # to its nearest neighbor?
        np.fill_diagonal(sim_matrix, -1)
        top1_sims = sim_matrix.max(axis=1)
        console.print(
            f"  Top-1 NN similarity: mean={top1_sims.mean():.4f}  "
            f"std={top1_sims.std():.4f}"
        )

    # HSV features
    if hsv_path.exists():
        hsv = np.load(hsv_path)
        console.print(f"  HSV features: shape={hsv.shape}, dtype={hsv.dtype}")
        hsv_norms = np.linalg.norm(hsv, axis=1)
        zero_hsv = (hsv_norms < 1e-6).sum()
        console.print(
            f"  HSV norms: mean={hsv_norms.mean():.4f}  "
            f"zero-vectors={zero_hsv}/{n}"
        )

    # Index metadata
    if idx_path.exists():
        with open(idx_path) as f:
            index_map = json.load(f)
        cams = Counter(m.get("camera_id", "?") for m in index_map)
        classes = Counter(m.get("class_id", -1) for m in index_map)
        console.print(f"  Per-camera counts: {dict(sorted(cams.items()))}")
        console.print(f"  Per-class counts: {dict(sorted(classes.items()))}")


# ── Stage 3 ───────────────────────────────────────────────────────────────

def validate_stage3(run_dir: Path):
    console.print(Panel("[bold cyan]Stage 3 — Indexing[/]"))
    stage_dir = run_dir / "stage3"

    faiss_path = stage_dir / "faiss_index.bin"
    db_path = stage_dir / "metadata.db"

    if not faiss_path.exists():
        console.print(f"  {FAIL}  faiss_index.bin not found")
        return

    import faiss
    index = faiss.read_index(str(faiss_path))
    console.print(f"  FAISS index: {index.ntotal} vectors, dim={index.d}")
    console.print(f"  Index type: {type(index).__name__}")

    if db_path.exists():
        conn = sqlite3.connect(str(db_path))
        row = conn.execute("SELECT COUNT(*) FROM tracklet_metadata").fetchone()
        console.print(f"  Metadata store: {row[0]} entries")
        conn.close()

    # Self-retrieval sanity check
    emb_path = run_dir / "stage2" / "embeddings.npy"
    if emb_path.exists() and index.ntotal > 0:
        embeddings = np.load(emb_path).astype(np.float32)
        # Query first 10 embeddings — top-1 should be themselves
        k = min(5, index.ntotal)
        q = embeddings[:min(10, len(embeddings))]
        D, I = index.search(q, k)
        self_hits = sum(1 for i, row in enumerate(I) if row[0] == i)
        console.print(
            f"  Self-retrieval: {self_hits}/{len(q)} "
            f"{'— ' + OK if self_hits == len(q) else '— ' + WARN + ' index may be corrupt'}"
        )


# ── Stage 4 ───────────────────────────────────────────────────────────────

def validate_stage4(run_dir: Path):
    console.print(Panel("[bold cyan]Stage 4 — Cross-Camera Association[/]"))
    stage_dir = run_dir / "stage4"

    traj_path = stage_dir / "global_trajectories.json"
    if not traj_path.exists():
        console.print(f"  {FAIL}  global_trajectories.json not found")
        return

    with open(traj_path) as f:
        trajectories = json.load(f)

    n_traj = len(trajectories)
    console.print(f"  Global trajectories: [bold]{n_traj}[/]")

    if n_traj == 0:
        return

    # Analyse cluster composition
    multi_cam = 0
    single_cam = 0
    cam_counts = []
    tracklet_counts = []
    durations = []

    for t in trajectories:
        tracklets = t.get("tracklets", [])
        tracklet_counts.append(len(tracklets))
        cams = set(tk.get("camera_id", "") for tk in tracklets)
        cam_counts.append(len(cams))
        if len(cams) > 1:
            multi_cam += 1
        else:
            single_cam += 1
        # Duration
        starts = [tk.get("start_time", 0) for tk in tracklets]
        ends = [tk.get("end_time", 0) for tk in tracklets]
        if starts and ends:
            durations.append(max(ends) - min(starts))

    console.print(f"  Multi-camera trajectories: {multi_cam}")
    console.print(f"  Single-camera trajectories: {single_cam}")
    console.print(
        f"  Cameras per trajectory: "
        f"mean={np.mean(cam_counts):.1f}  max={max(cam_counts)}"
    )
    console.print(
        f"  Tracklets per trajectory: "
        f"mean={np.mean(tracklet_counts):.1f}  max={max(tracklet_counts)}"
    )
    if durations:
        console.print(
            f"  Duration: mean={np.mean(durations):.1f}s  "
            f"max={max(durations):.1f}s"
        )

    # Class distribution
    class_dist = Counter()
    for t in trajectories:
        cls = t.get("class_name", "unknown")
        class_dist[cls] += 1
    console.print(f"  Class distribution: {dict(class_dist)}")

    # Count total tracklets vs trajectories — fragmentation ratio
    total_tk = sum(tracklet_counts)
    # Load stage 1 tracklet count for comparison
    stage1_dir = run_dir / "stage1"
    stage1_count = 0
    for tf in stage1_dir.glob("tracklets_*.json"):
        with open(tf) as f:
            stage1_count += len(json.load(f))

    if stage1_count > 0:
        merge_ratio = stage1_count / max(n_traj, 1)
        console.print(
            f"  Merge ratio: {stage1_count} tracklets → {n_traj} identities "
            f"({merge_ratio:.1f}x compression)"
        )
        if merge_ratio < 2:
            console.print(f"  {WARN}  Low merge ratio — association may be too conservative")


# ── Stage 5 ───────────────────────────────────────────────────────────────

def validate_stage5(run_dir: Path):
    console.print(Panel("[bold cyan]Stage 5 — Evaluation[/]"))
    stage_dir = run_dir / "stage5"

    report_path = stage_dir / "evaluation_report.json"
    if not report_path.exists():
        console.print(f"  {FAIL}  evaluation_report.json not found")
        return

    with open(report_path) as f:
        result = json.load(f)

    mota = result.get("mota", 0)
    idf1 = result.get("idf1", 0)
    hota = result.get("hota", 0)
    idsw = result.get("id_switches", 0)
    n_gt = result.get("num_gt_ids", 0)
    n_pred = result.get("num_pred_ids", 0)

    if n_gt == 0 and mota == 0 and idf1 == 0:
        console.print(f"  {WARN}  No ground truth was evaluated (all metrics = 0)")
        console.print(f"         Set stage5.ground_truth_dir in your config!")
        console.print(f"  Predicted identities: {n_pred}")
        details = result.get("details", {})
        if details:
            for k, v in details.items():
                console.print(f"    {k}: {v}")
        return

    tbl = Table(title="Evaluation Metrics")
    tbl.add_column("Metric")
    tbl.add_column("Value", justify="right")
    tbl.add_column("Status")

    tbl.add_row("MOTA", f"{mota:.4f}", OK if mota > 0.5 else WARN)
    tbl.add_row("IDF1", f"{idf1:.4f}", OK if idf1 > 0.5 else WARN)
    tbl.add_row("HOTA", f"{hota:.4f}", OK if hota > 0.4 else WARN)
    tbl.add_row("ID Switches", str(idsw), OK if idsw < 20 else WARN)
    tbl.add_row("GT IDs", str(n_gt), "")
    tbl.add_row("Pred IDs", str(n_pred), "")
    if n_gt > 0:
        tbl.add_row("ID Ratio", f"{n_pred / n_gt:.2f}x", OK if 0.8 < n_pred / n_gt < 1.5 else WARN)

    console.print(tbl)

    # Per-camera breakdown
    per_cam = result.get("details", {}).get("per_camera", {})
    if per_cam:
        cam_tbl = Table(title="Per-Camera Breakdown")
        cam_tbl.add_column("Camera")
        cam_tbl.add_column("MOTA", justify="right")
        cam_tbl.add_column("IDF1", justify="right")
        cam_tbl.add_column("IDSW", justify="right")
        for cam, m in sorted(per_cam.items()):
            cam_tbl.add_row(
                cam,
                f"{m.get('mota', 0):.4f}",
                f"{m.get('idf1', 0):.4f}",
                str(m.get("id_switches", 0)),
            )
        console.print(cam_tbl)


# ── Main ──────────────────────────────────────────────────────────────────

@click.command()
@click.option("--run-dir", "-r", required=True, type=click.Path(exists=True))
@click.option("--stage", "-s", type=int, default=None, help="Validate specific stage only")
def main(run_dir: str, stage: int | None):
    """Run stage-by-stage diagnostic validation on a pipeline run."""
    run = Path(run_dir)
    console.print(Panel(f"[bold]Validating run: {run}[/]", title="Stage Validator"))

    stages = {
        0: validate_stage0,
        1: validate_stage1,
        2: validate_stage2,
        3: validate_stage3,
        4: validate_stage4,
        5: validate_stage5,
    }

    if stage is not None:
        if stage in stages:
            stages[stage](run)
        else:
            console.print(f"[red]Unknown stage: {stage}[/]")
    else:
        for s in sorted(stages):
            stages[s](run)
            console.print()


if __name__ == "__main__":
    main()
