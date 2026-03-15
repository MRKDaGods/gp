"""Quick hyperparameter scan for Stage 4 (association) using pre-computed embeddings.

Re-runs Stages 4+5 with different parameter combinations to find the optimal
configuration for a given Stage 1-3 run.

Usage:
    python scripts/scan_stage4_params.py --run run_20260315_v4

The script re-uses the existing Stage 1-3 outputs and only re-runs the fast
CPU-bound association and evaluation stages.

Examples:
    # Scan AQE k values
    python scripts/scan_stage4_params.py --run run_20260315_v4 --scan aqe_k

    # Scan similarity threshold
    python scripts/scan_stage4_params.py --run run_20260315_v4 --scan sim_thresh

    # Scan louvain resolution
    python scripts/scan_stage4_params.py --run run_20260315_v4 --scan louvain_res

    # Custom full grid scan
    python scripts/scan_stage4_params.py --run run_20260315_v4 --scan full
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

console = Console()


# Parameter grids for each scan type
PARAM_GRIDS = {
    "aqe_k": [
        {"stage4.association.query_expansion.k": k}
        for k in [1, 3, 5, 7, 10]
    ],
    "sim_thresh": [
        {"stage4.association.graph.similarity_threshold": t}
        for t in [0.35, 0.40, 0.42, 0.45, 0.48, 0.50]
    ],
    "louvain_res": [
        {"stage4.association.graph.louvain_resolution": r}
        for r in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
    ],
    "appearance_weight": [
        {
            "stage4.association.weights.vehicle.appearance": a,
            "stage4.association.weights.vehicle.spatiotemporal": round(1.0 - a - 0.05, 2),
        }
        for a in [0.60, 0.65, 0.70, 0.75, 0.80]
    ],
    "full": [
        {"stage4.association.query_expansion.k": k, "stage4.association.graph.similarity_threshold": t}
        for k in [3, 5, 7]
        for t in [0.40, 0.45, 0.50]
    ],
}


def run_stage45(run_name: str, overrides: dict, config: str, dataset_config: str) -> dict | None:
    """Run Stages 4+5 with given overrides and return metrics."""
    cmd = [
        sys.executable, "scripts/run_pipeline.py",
        "--config", config,
        "--dataset-config", dataset_config,
        "--stages", "4,5",
        "-o", f"project.run_name={run_name}",
    ]
    for key, val in overrides.items():
        cmd += ["-o", f"{key}={val}"]

    console.print(f"  Overrides: {overrides}", style="dim")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(project_root))

    if result.returncode != 0:
        console.print(f"  [red]FAILED[/red]: {result.stderr[-500:]}")
        return None

    # Load evaluation results
    report_path = project_root / "data" / "outputs" / run_name / "stage5" / "evaluation_report.json"
    if not report_path.exists():
        console.print(f"  [red]No evaluation report found[/red]")
        return None

    with open(report_path) as f:
        return json.load(f)


@click.command()
@click.option("--run", "-r", required=True, help="Run name (e.g. run_20260315_v4)")
@click.option("--scan", "-s", default="aqe_k",
              type=click.Choice(list(PARAM_GRIDS.keys())),
              help="Which parameter to scan")
@click.option("--config", "-c", default="configs/default.yaml", show_default=True)
@click.option("--dataset-config", "-d", default="configs/datasets/cityflowv2.yaml", show_default=True)
def main(run: str, scan: str, config: str, dataset_config: str):
    """Scan Stage 4 hyperparameters and report results."""
    param_grid = PARAM_GRIDS[scan]
    console.print(f"\n[bold]Stage 4 parameter scan: {scan.upper()}[/bold]")
    console.print(f"Run: [cyan]{run}[/cyan] | {len(param_grid)} configurations\n")

    results = []
    for i, overrides in enumerate(param_grid, 1):
        console.print(f"[{i}/{len(param_grid)}] {overrides}")
        metrics = run_stage45(run, overrides, config, dataset_config)
        if metrics:
            results.append({
                "overrides": overrides,
                "idf1": metrics.get("idf1", 0),
                "mota": metrics.get("mota", 0),
                "hota": metrics.get("hota", 0),
                "id_switches": metrics.get("id_switches", 0),
            })
            console.print(
                f"  → IDF1={results[-1]['idf1']:.4f} "
                f"MOTA={results[-1]['mota']:.4f} "
                f"HOTA={results[-1]['hota']:.4f} "
                f"IDSW={results[-1]['id_switches']}"
            )

    if not results:
        console.print("[red]No results collected.[/red]")
        return

    # Sort by IDF1
    results.sort(key=lambda r: r["idf1"], reverse=True)

    console.print(f"\n[bold green]Results (sorted by IDF1):[/bold green]")
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("IDF1", justify="right")
    table.add_column("MOTA", justify="right")
    table.add_column("HOTA", justify="right")
    table.add_column("IDSW", justify="right")
    table.add_column("Overrides")

    for r in results:
        table.add_row(
            f"{r['idf1']*100:.2f}%",
            f"{r['mota']*100:.2f}%",
            f"{r['hota']*100:.2f}%",
            str(r["id_switches"]),
            str(r["overrides"]),
        )
    console.print(table)

    # Save results to file
    out_path = project_root / "data" / "outputs" / f"scan_{scan}_{run}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"\n[dim]Results saved to {out_path}[/dim]")


if __name__ == "__main__":
    main()
