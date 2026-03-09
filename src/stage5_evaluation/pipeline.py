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

    # Build ROI config for WILDTRACK ground-plane filtering
    dataset_cfg = cfg.get("dataset", {})
    gp_eval_cfg = stage_cfg.get("ground_plane_eval", {})
    roi_config = None

    if gp_eval_cfg.get("enabled", False) or dataset_cfg.get("name") == "wildtrack":
        annotations_dir = gp_eval_cfg.get(
            "annotations_dir",
            dataset_cfg.get("root_dir", "") + "/annotations_positions",
        )
        calibrations_dir = gp_eval_cfg.get(
            "calibrations_dir",
            dataset_cfg.get("root_dir", "") + "/calibrations",
        )
        if Path(annotations_dir).exists() and Path(calibrations_dir).exists():
            roi_config = {
                "annotations_dir": annotations_dir,
                "calibrations_dir": calibrations_dir,
                "margin_cm": 100.0,
            }

    trajectories_to_mot_submission(trajectories, pred_dir, roi_config=roi_config)
    logger.info(f"Predictions converted to MOT format in {pred_dir}")

    # ── Ground-plane evaluation for multi-view overlapping datasets ──
    # WILDTRACK and similar datasets use ground-plane MODA as primary metric.
    # This is a fundamentally different protocol from per-camera 2D MOT eval.
    gp_result = None

    if gp_eval_cfg.get("enabled", False) or dataset_cfg.get("name") == "wildtrack":
        annotations_dir = gp_eval_cfg.get(
            "annotations_dir",
            dataset_cfg.get("root_dir", "") + "/annotations_positions",
        )
        calibrations_dir = gp_eval_cfg.get(
            "calibrations_dir",
            dataset_cfg.get("root_dir", "") + "/calibrations",
        )
        if Path(annotations_dir).exists() and Path(calibrations_dir).exists():
            from src.stage5_evaluation.ground_plane_eval import (
                evaluate_wildtrack_ground_plane,
            )
            logger.info("Running WILDTRACK ground-plane evaluation (published protocol)")
            gp_result = evaluate_wildtrack_ground_plane(
                trajectories=trajectories,
                annotations_dir=annotations_dir,
                calibrations_dir=calibrations_dir,
                conf_threshold=float(gp_eval_cfg.get("conf_threshold", 0.25)),
                match_threshold_cm=float(gp_eval_cfg.get("match_threshold_cm", 50.0)),
                nms_radius_cm=float(gp_eval_cfg.get("nms_radius_cm", 50.0)),
            )
        else:
            logger.warning(
                f"Ground-plane eval requested but paths not found: "
                f"annotations={annotations_dir}, calibrations={calibrations_dir}"
            )

    # ── Standard per-camera 2D MOT evaluation ──
    if gt_dir is not None and Path(gt_dir).exists():
        iou_threshold = float(stage_cfg.get("iou_threshold", 0.5))
        result = evaluate_mot(
            gt_dir=str(gt_dir),
            pred_dir=str(pred_dir),
            metrics=list(stage_cfg.get("metrics", ["HOTA", "MOTA", "IDF1"])),
            iou_threshold=iou_threshold,
        )
        logger.info(
            f"Per-camera 2D eval (IoU>={iou_threshold}): MOTA={result.mota:.3f}, "
            f"IDF1={result.idf1:.3f}, HOTA={result.hota:.3f}, "
            f"ID Switches={result.id_switches}"
        )

        # Log per-camera breakdown if available
        per_cam = (result.details or {}).get("per_camera", {})
        if per_cam:
            logger.info("Per-camera breakdown:")
            for cam_id, cam_metrics in sorted(per_cam.items()):
                parts = [f"  {cam_id}:"]
                for k, v in sorted(cam_metrics.items()):
                    if isinstance(v, float):
                        parts.append(f"{k}={v:.3f}")
                    else:
                        parts.append(f"{k}={v}")
                logger.info(" ".join(parts))

        # If ground-plane eval ran, merge its results as primary
        if gp_result is not None:
            result.details = result.details or {}
            result.details["ground_plane"] = gp_result.details
            result.details["ground_plane"]["moda"] = gp_result.mota
            result.details["ground_plane"]["idf1"] = gp_result.idf1
            # For WILDTRACK, ground-plane MODA is the primary metric
            logger.info(
                f"[PRIMARY] Ground-plane MODA={gp_result.mota*100:.1f}%, "
                f"IDF1={gp_result.idf1*100:.1f}%"
            )
    elif gp_result is not None:
        # Only ground-plane eval available
        result = gp_result
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
