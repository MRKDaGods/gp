"""Entry point: run a single pipeline stage.

Usage:
    python scripts/run_stage.py --config configs/default.yaml --stage 0
    python scripts/run_stage.py --config configs/default.yaml --stage 1 --smoke-test
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import click

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.config import load_config
from src.core.logging_utils import setup_logging


@click.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True))
@click.option("--stage", "-s", required=True, type=int, help="Stage number (0-6)")
@click.option("--run-dir", type=str, default=None, help="Existing run directory to use")
@click.option("--smoke-test", is_flag=True, default=False)
@click.option("--override", "-o", multiple=True)
def main(config: str, stage: int, run_dir: str, smoke_test: bool, override: tuple):
    """Run a single stage of the MTMC tracking pipeline."""
    cfg = load_config(config, overrides=list(override))

    if run_dir:
        output_base = Path(run_dir)
    else:
        run_name = cfg.project.get("run_name") or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_base = Path(cfg.project.output_dir) / run_name

    setup_logging(level=cfg.project.get("log_level", "INFO"))

    print(f"Running Stage {stage} | Output: {output_base}")

    if stage == 0:
        from src.stage0_ingestion import run_stage0
        run_stage0(cfg, output_dir=output_base / "stage0", smoke_test=smoke_test)

    elif stage == 1:
        from src.core.io_utils import load_frame_manifest
        from src.stage1_tracking import run_stage1
        frames = load_frame_manifest(output_base / "stage0" / "frames_manifest.json")
        run_stage1(cfg, frames, output_dir=output_base / "stage1", smoke_test=smoke_test)

    elif stage == 5:
        from src.core.io_utils import load_global_trajectories
        from src.stage5_evaluation import run_stage5
        trajectories = load_global_trajectories(
            output_base / "stage4" / "global_trajectories.json"
        )
        run_stage5(cfg, trajectories, output_dir=output_base / "stage5")

    else:
        print(f"Stage {stage}: Use run_pipeline.py with --stages {stage} for full dependency handling.")

    print("Done.")


if __name__ == "__main__":
    main()
