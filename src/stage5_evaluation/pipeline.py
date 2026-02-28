"""Stage 5 — System Evaluation & Quality Assessment pipeline.

Evaluates tracking results against ground truth using standard metrics
(HOTA, IDF1, MOTA) and generates evaluation reports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger
from omegaconf import DictConfig

from src.core.data_models import EvaluationResult, GlobalTrajectory
from src.core.io_utils import save_evaluation_result
from src.stage5_evaluation.format_converter import trajectories_to_mot_submission
from src.stage5_evaluation.metrics import evaluate_mot
from src.stage5_evaluation.report_generator import generate_report


def run_stage5(
    cfg: DictConfig,
    trajectories: List[GlobalTrajectory],
    output_dir: str | Path,
    ground_truth_dir: Optional[str | Path] = None,
) -> EvaluationResult:
    """Run evaluation on tracking results.

    Args:
        cfg: Full pipeline config (uses cfg.stage5).
        trajectories: Global trajectories from Stage 4.
        output_dir: Directory for stage5 outputs.
        ground_truth_dir: Path to ground truth annotations. Overrides config.

    Returns:
        EvaluationResult with computed metrics.
    """
    stage_cfg = cfg.stage5
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_dir = ground_truth_dir or stage_cfg.get("ground_truth_dir")

    # Convert predictions to MOTChallenge format
    pred_dir = output_dir / "predictions_mot"
    trajectories_to_mot_submission(trajectories, pred_dir)
    logger.info(f"Predictions converted to MOT format in {pred_dir}")

    # Run evaluation
    if gt_dir is not None and Path(gt_dir).exists():
        result = evaluate_mot(
            gt_dir=str(gt_dir),
            pred_dir=str(pred_dir),
            metrics=list(stage_cfg.get("metrics", ["HOTA", "MOTA", "IDF1"])),
        )
        logger.info(
            f"Evaluation results: MOTA={result.mota:.3f}, "
            f"IDF1={result.idf1:.3f}, HOTA={result.hota:.3f}, "
            f"ID Switches={result.id_switches}"
        )
    else:
        logger.warning(
            f"Ground truth not found at {gt_dir}. "
            "Generating summary statistics only."
        )
        result = _compute_summary_stats(trajectories)

    # Save results
    save_evaluation_result(result, output_dir / "evaluation_report.json")

    # Generate human-readable report
    if stage_cfg.get("generate_report", True):
        report_format = stage_cfg.get("report_format", "html")
        report_path = output_dir / f"evaluation_report.{report_format}"
        generate_report(result, trajectories, report_path, fmt=report_format)
        logger.info(f"Evaluation report saved to {report_path}")

    return result


def _compute_summary_stats(trajectories: List[GlobalTrajectory]) -> EvaluationResult:
    """Compute basic summary statistics when no ground truth is available."""
    num_trajectories = len(trajectories)
    total_tracklets = sum(len(t.tracklets) for t in trajectories)
    multi_cam = sum(1 for t in trajectories if t.num_cameras > 1)

    return EvaluationResult(
        num_pred_ids=num_trajectories,
        details={
            "total_tracklets": total_tracklets,
            "multi_camera_trajectories": multi_cam,
            "single_camera_trajectories": num_trajectories - multi_cam,
            "avg_cameras_per_trajectory": (
                sum(t.num_cameras for t in trajectories) / num_trajectories
                if num_trajectories > 0
                else 0
            ),
            "avg_duration": (
                sum(t.total_duration for t in trajectories) / num_trajectories
                if num_trajectories > 0
                else 0
            ),
        },
    )
