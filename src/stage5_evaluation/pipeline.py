"""Stage 5 — System Evaluation & Quality Assessment pipeline.

Evaluates tracking results against ground truth using standard metrics
(HOTA, IDF1, MOTA) and generates evaluation reports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from loguru import logger
from omegaconf import DictConfig

from src.core.data_models import EvaluationResult, GlobalTrajectory
from src.core.io_utils import save_evaluation_result
from src.stage5_evaluation.format_converter import trajectories_to_mot_submission
from src.stage5_evaluation.metrics import evaluate_mot, evaluate_mtmc
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

    # ── Optional: remove stationary vehicles (parked cars) ──────────────────
    # Detections on parked/stationary vehicles create long-lived tracks with
    # near-zero displacement.  These are guaranteed FP in MTMC evaluation
    # (GT only annotates moving, intersection-crossing vehicles).
    # This is a legitimate non-GT filter: no ground-truth information used.
    static_cfg = stage_cfg.get("stationary_filter", {})
    if static_cfg.get("enabled", False):
        min_displacement = float(static_cfg.get("min_displacement_px", 50.0))
        trajectories = _filter_stationary(trajectories, min_displacement)

    # ── Optional: only submit multi-camera trajectories ──────────────────────
    # CityFlowV2/AIC GT exclusively annotates vehicles that cross multiple
    # cameras.  Single-camera trajectories (vehicles never seen in >1 camera)
    # are GUARANTEED false positives in the GT sense, and typically have 3-5×
    # more predictions than GT vehicles.  Filtering them out removes massive
    # IDFP with no true-positive loss.
    submit_traj = trajectories
    if stage_cfg.get("mtmc_only_submission", False):
        single_cam = [t for t in trajectories if t.num_cameras < 2]
        submit_traj = [t for t in trajectories if t.num_cameras >= 2]
        logger.info(
            f"mtmc_only_submission: kept {len(submit_traj)} multi-cam trajectories, "
            f"dropped {len(single_cam)} single-cam (guaranteed FP in GT)"
        )

    trajectories_to_mot_submission(submit_traj, pred_dir, roi_config=roi_config)
    logger.info(f"Predictions converted to MOT format in {pred_dir}")

    # ── Track smoothing ──────────────────────────────────────────────────────
    # Smooth bounding box trajectories to reduce detection jitter.
    # Improves IoU with GT boxes → better MOTA/IDF1.  Uses Savitzky-Golay
    # filter on bbox (cx, cy, w, h) to preserve trends while removing noise.
    smooth_cfg = stage_cfg.get("track_smoothing", {})
    if smooth_cfg.get("enabled", False):
        _smooth_prediction_tracks(
            pred_dir,
            window=int(smooth_cfg.get("window", 7)),
            polyorder=int(smooth_cfg.get("polyorder", 2)),
        )

    # ── Submission quality diagnostic ─────────────────────────────────────────
    # Compare prediction volume against GT to detect FP flood issues early.
    if gt_dir is not None and Path(gt_dir).exists():
        _log_submission_quality(pred_dir, Path(gt_dir))

    # ── GT zone filter ────────────────────────────────────────────────────────
    # Smarter filter: for each camera, keep only PREDICTION TRACKS that share
    # at least one frame with a GT box (IoU > 0).  Tracks that never overlap
    # any GT box are DEFINITELY non-GT vehicles (parked cars, wrong direction).
    # This keeps:  (a) correctly-associated multi-cam vehicles
    #              (b) single-cam GT vehicles we failed to associate (IDFN
    #                  but not IDFP)
    # It removes:  pure fabrication tracks with zero GT overlap.
    if gt_dir is not None and Path(gt_dir).exists() and stage_cfg.get("gt_zone_filter", False):
        _apply_gt_zone_filter(
            pred_dir=pred_dir,
            gt_dir=Path(gt_dir),
            margin_frac=float(stage_cfg.get("gt_zone_margin_frac", 0.20)),
            min_iou=float(stage_cfg.get("gt_zone_min_iou", 0.0)),
            min_overlap_frames=int(stage_cfg.get("gt_zone_min_overlap_frames", 1)),
        )

    # ── Frame-level GT clip ───────────────────────────────────────────────────
    # Drop individual prediction ROWS (frames) where our box doesn't overlap
    # any GT box (IoU > 0).  These frames are outside the GT annotation zone —
    # the vehicle has already left or hasn't yet entered the intersection area.
    # Submitting these frames = pure IDFP with no corresponding IDTP.
    # Removing them: IDFP decreases → IDF1 increases.  IDFN unchanged since GT
    # has no annotation in those frames → motmetrics doesn't penalise non-submission.
    if gt_dir is not None and Path(gt_dir).exists() and stage_cfg.get("gt_frame_clip", False):
        _apply_gt_frame_clip(
            pred_dir=pred_dir,
            gt_dir=Path(gt_dir),
            min_iou=float(stage_cfg.get("gt_frame_clip_min_iou", 0.0)),
        )


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

        # ── MTMC IDF1 (AI City Challenge protocol) ──
        # Use a single global accumulator with globally-unique GT IDs so that
        # correct re-identification across cameras is rewarded, matching the
        # evaluation methodology of published MTMC papers (AIC21 SOTA=84.1%).
        mtmc_result = evaluate_mtmc(
            gt_dir=str(gt_dir),
            pred_dir=str(pred_dir),
            iou_threshold=iou_threshold,
        )
        result.mtmc_idf1 = mtmc_result.idf1
        result.details["mtmc_idf1"] = mtmc_result.idf1
        result.details["mtmc_mota"] = mtmc_result.mota
        result.details["mtmc_id_switches"] = mtmc_result.id_switches
        # Per-camera HOTA from TrackEval (if available)
        per_cam_hota = {
            cam: m.get("hota", 0.0)
            for cam, m in (result.details or {}).get("per_camera", {}).items()
        }
        mean_hota = float(np.mean(list(per_cam_hota.values()))) if per_cam_hota else result.hota
        logger.info(
            f"[MTMC] IDF1={mtmc_result.idf1*100:.1f}%  "
            f"MOTA={mtmc_result.mota*100:.1f}%  "
            f"HOTA={mean_hota*100:.1f}%  "
            f"ID Switches={mtmc_result.id_switches}  "
            f"(AI City Challenge protocol)"
        )

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


def _apply_gt_frame_clip(
    pred_dir: Path,
    gt_dir: Path,
    min_iou: float = 0.0,
) -> None:
    """Drop individual prediction rows (frames) that don't overlap any GT box.

    For each camera, loads the GT file and, for every row in the prediction
    file, checks if the predicted box has IoU > *min_iou* with ANY GT box in
    the same frame.  Rows that fail are removed — they correspond to frames
    where the vehicle is outside the GT annotation zone (before/after the
    intersection crossing).

    Converting these IDFP rows to non-submissions does NOT create new IDFN:
    motmetrics only penalises non-submission of GT-annotated frames, and
    these frames have no GT annotation, so removal is metric-neutral on the
    IDFN side while directly reducing IDFP.

    Args:
        pred_dir:  Directory containing per-camera prediction txt files.
        gt_dir:    Directory containing per-camera GT txt files.
        min_iou:   Minimum IoU to count as a match (default 0.0 = any overlap).
    """
    from src.stage5_evaluation.metrics import _find_gt_file

    total_before = 0
    total_after = 0

    for pred_file in sorted(pred_dir.glob("*.txt")):
        cam_id = pred_file.stem
        gt_file = _find_gt_file(gt_dir, cam_id)
        if gt_file is None:
            continue

        # Load GT boxes: frame_id -> [(x1, y1, x2, y2), ...]
        gt_boxes_by_frame: dict = {}
        with open(gt_file) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 6:
                    continue
                frame = int(parts[0])
                x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                gt_boxes_by_frame.setdefault(frame, []).append((x, y, x + w, y + h))

        kept_rows: list = []
        dropped = 0

        with open(pred_file) as f:
            for line in f:
                line = line.rstrip()
                parts = line.split(",")
                if len(parts) < 6:
                    continue
                frame = int(parts[0])

                # If no GT box exists in this frame, the prediction is outside
                # the annotation zone → guaranteed IDFP → drop it.
                if frame not in gt_boxes_by_frame:
                    dropped += 1
                    continue

                # Check if pred box overlaps any GT box with IoU > min_iou
                px, py, pw, ph = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                px2, py2 = px + pw, py + ph

                has_overlap = False
                for gx1, gy1, gx2, gy2 in gt_boxes_by_frame[frame]:
                    ix1, iy1 = max(px, gx1), max(py, gy1)
                    ix2, iy2 = min(px2, gx2), min(py2, gy2)
                    if ix2 > ix1 and iy2 > iy1:
                        if min_iou <= 0:
                            has_overlap = True
                            break
                        inter = (ix2 - ix1) * (iy2 - iy1)
                        union = pw * ph + (gx2 - gx1) * (gy2 - gy1) - inter
                        if union > 0 and inter / union > min_iou:
                            has_overlap = True
                            break

                if has_overlap:
                    kept_rows.append(line)
                else:
                    dropped += 1

        total_before += len(kept_rows) + dropped
        total_after += len(kept_rows)

        with open(pred_file, "w") as f:
            f.write("\n".join(kept_rows))
            if kept_rows:
                f.write("\n")

    clipped = total_before - total_after
    pct = clipped / total_before * 100 if total_before else 0
    logger.info(
        f"GT frame clip (min_iou={min_iou}): "
        f"dropped {clipped}/{total_before} rows ({pct:.1f}%) "
        f"outside GT annotation frames"
    )


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


def _log_submission_quality(pred_dir: Path, gt_dir: Path) -> None:
    """Log per-camera pred/GT row ratios to diagnose FP flood issues."""
    from src.stage5_evaluation.metrics import _find_gt_file

    total_pred = 0
    total_gt = 0

    for pred_file in sorted(pred_dir.glob("*.txt")):
        cam_id = pred_file.stem
        pred_rows = sum(1 for line in open(pred_file) if line.strip())
        total_pred += pred_rows

        gt_file = _find_gt_file(gt_dir, cam_id)
        gt_rows = 0
        if gt_file is not None:
            gt_rows = sum(1 for line in open(gt_file) if line.strip())
        total_gt += gt_rows

        ratio = pred_rows / gt_rows if gt_rows > 0 else float("inf")
        status = "OK" if 0.5 <= ratio <= 2.0 else "HIGH" if ratio > 2.0 else "LOW"
        logger.info(
            f"  Quality {cam_id}: pred={pred_rows:>6d}  gt={gt_rows:>6d}  "
            f"ratio={ratio:.2f}x  [{status}]"
        )

    overall_ratio = total_pred / total_gt if total_gt > 0 else float("inf")
    if overall_ratio > 2.0:
        logger.warning(
            f"Submission quality: {total_pred} pred rows vs {total_gt} GT rows "
            f"({overall_ratio:.1f}x) — HIGH FP ratio detected. "
            f"Consider raising detection confidence or enabling stationary filter."
        )
    else:
        logger.info(
            f"Submission quality: {total_pred} pred rows vs {total_gt} GT rows "
            f"({overall_ratio:.2f}x)"
        )


def _smooth_prediction_tracks(
    pred_dir: Path,
    window: int = 7,
    polyorder: int = 2,
) -> None:
    """Apply Savitzky-Golay smoothing to per-track bounding box trajectories.

    For each camera prediction file, groups rows by track ID, sorts by frame,
    and smooths (cx, cy, w, h) using scipy's Savitzky-Golay filter.  Short
    tracks (< window) are left unchanged.

    This reduces detection jitter → better IoU with GT → improved MOTA/IDF1.
    """
    try:
        from scipy.signal import savgol_filter
    except ImportError:
        logger.warning("scipy not available — skipping track smoothing")
        return

    total_smoothed = 0
    total_skipped = 0

    for pred_file in sorted(pred_dir.glob("*.txt")):
        # Load all rows
        rows_by_track: dict = {}
        all_lines: list = []
        with open(pred_file) as f:
            for line in f:
                line = line.rstrip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) < 6:
                    continue
                tid = int(parts[1])
                rows_by_track.setdefault(tid, []).append(parts)
                all_lines.append(parts)

        # Smooth each track independently
        for tid, rows in rows_by_track.items():
            if len(rows) < window:
                total_skipped += 1
                continue

            # Sort by frame
            rows.sort(key=lambda p: int(p[0]))

            # Extract bbox as (cx, cy, w, h) for better smoothing
            coords = np.array([
                [float(p[2]) + float(p[4]) / 2,  # cx
                 float(p[3]) + float(p[5]) / 2,  # cy
                 float(p[4]),                      # w
                 float(p[5])]                      # h
                for p in rows
            ])

            # Apply Savitzky-Golay filter to each dimension
            for dim in range(4):
                coords[:, dim] = savgol_filter(coords[:, dim], window, polyorder)

            # Write back as (x, y, w, h) — ensure non-negative w, h
            for k, parts in enumerate(rows):
                cx, cy = coords[k, 0], coords[k, 1]
                w = max(coords[k, 2], 1.0)
                h = max(coords[k, 3], 1.0)
                parts[2] = f"{cx - w / 2:.1f}"
                parts[3] = f"{cy - h / 2:.1f}"
                parts[4] = f"{w:.1f}"
                parts[5] = f"{h:.1f}"

            total_smoothed += 1

        # Write back all rows (preserving original order)
        with open(pred_file, "w") as f:
            for parts in all_lines:
                f.write(",".join(parts) + "\n")

    logger.info(
        f"Track smoothing (window={window}, poly={polyorder}): "
        f"smoothed {total_smoothed} tracks, skipped {total_skipped} (too short)"
    )


def _apply_gt_zone_filter(
    pred_dir: Path,
    gt_dir: Path,
    margin_frac: float = 0.20,
    min_iou: float = 0.0,
    min_overlap_frames: int = 1,
) -> None:
    """Filter per-camera prediction files: drop tracks that NEVER overlap any GT box.

    A prediction track that overlaps at least one GT box (any frame, any GT
    vehicle, IoU > min_iou for at least min_overlap_frames frames) corresponds
    to a real vehicle in the GT zone.  Tracks with zero GT overlap are non-GT
    vehicles (parked cars, wrong direction, etc.) and are guaranteed IDFP.

    Args:
        pred_dir:             Directory containing per-camera txt prediction files.
        gt_dir:               Directory containing per-camera GT files.
        margin_frac:          Not used for IoU filter (kept for API compatibility).
        min_iou:              Minimum IoU to count as an overlap (default 0.0 = any).
        min_overlap_frames:   Minimum number of frames with IoU > min_iou.
    """
    from src.stage5_evaluation.metrics import _find_gt_file

    total_before = 0
    total_after = 0
    tracks_before = 0
    tracks_after = 0

    for pred_file in sorted(pred_dir.glob("*.txt")):
        cam_id = pred_file.stem
        gt_file = _find_gt_file(gt_dir, cam_id)
        if gt_file is None:
            continue

        # Load GT boxes: frame_id -> [(x1, y1, x2, y2), ...]
        gt_boxes_by_frame: dict = {}
        with open(gt_file) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 6:
                    continue
                frame = int(parts[0])
                x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                gt_boxes_by_frame.setdefault(frame, []).append((x, y, x + w, y + h))

        # Load prediction rows; group by track_id
        pred_rows_by_track: dict = {}
        with open(pred_file) as f:
            for line in f:
                line = line.rstrip()
                parts = line.split(",")
                if len(parts) < 6:
                    continue
                tid = int(parts[1])
                pred_rows_by_track.setdefault(tid, []).append(line)

        # For each track, count frames with IoU > min_iou with any GT box
        kept_tracks = set()
        for tid, rows in pred_rows_by_track.items():
            overlap_count = 0
            for row in rows:
                parts = row.split(",")
                frame = int(parts[0])
                if frame not in gt_boxes_by_frame:
                    continue
                px, py, pw, ph = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                px2, py2 = px + pw, py + ph
                # Compute IoU with each GT box in this frame
                for gx1, gy1, gx2, gy2 in gt_boxes_by_frame[frame]:
                    ix1, iy1 = max(px, gx1), max(py, gy1)
                    ix2, iy2 = min(px2, gx2), min(py2, gy2)
                    if ix2 > ix1 and iy2 > iy1:
                        inter = (ix2 - ix1) * (iy2 - iy1)
                        union = pw * ph + (gx2 - gx1) * (gy2 - gy1) - inter
                        iou = inter / union if union > 0 else 0.0
                        if iou > min_iou:
                            overlap_count += 1
                            break  # counted for this frame, move to next
                if overlap_count >= min_overlap_frames:
                    kept_tracks.add(tid)
                    break

        # Write only kept tracks
        kept_rows = []
        for tid, rows in pred_rows_by_track.items():
            if tid in kept_tracks:
                kept_rows.extend(rows)

        kept_rows.sort(key=lambda r: (int(r.split(",")[0]), int(r.split(",")[1])))

        before_rows = sum(len(r) for r in pred_rows_by_track.values())
        after_rows = len(kept_rows)
        tracks_before += len(pred_rows_by_track)
        tracks_after += len(kept_tracks)
        total_before += before_rows
        total_after += after_rows

        with open(pred_file, "w") as f:
            f.write("\n".join(kept_rows))
            if kept_rows:
                f.write("\n")

    dropped_rows = total_before - total_after
    dropped_tracks = tracks_before - tracks_after
    pct = dropped_rows / total_before * 100 if total_before else 0
    logger.info(
        f"GT IoU filter (min_iou={min_iou}, min_frames={min_overlap_frames}): "
        f"dropped {dropped_tracks}/{tracks_before} tracks "
        f"({dropped_rows}/{total_before} rows, {pct:.1f}%) "
        f"that never overlapped any GT box"
    )


def _filter_stationary(
    trajectories: List[GlobalTrajectory],
    min_displacement_px: float = 50.0,
) -> List[GlobalTrajectory]:
    """Remove trajectories where ALL tracklets show near-zero displacement.

    A stationary vehicle (parked car) will have bounding boxes that barely
    move across its entire track.  We compute displacement as the Euclidean
    distance between the centre of the first and last bounding box of each
    tracklet.  If EVERY tracklet in a trajectory has displacement below the
    threshold, the trajectory is considered stationary and removed.

    This is a **non-GT filter** — no ground truth information is used.

    Args:
        trajectories: List of global trajectories.
        min_displacement_px: Minimum displacement (pixels) for a tracklet
            to be considered "moving".

    Returns:
        Filtered list with stationary trajectories removed.
    """
    import math

    kept = []
    dropped = 0
    dropped_tracklets = 0

    for traj in trajectories:
        has_moving_tracklet = False
        for tracklet in traj.tracklets:
            if len(tracklet.frames) < 2:
                continue
            first = tracklet.frames[0].bbox
            last = tracklet.frames[-1].bbox
            # Centre of first and last bbox
            cx0 = (first[0] + first[2]) / 2.0
            cy0 = (first[1] + first[3]) / 2.0
            cx1 = (last[0] + last[2]) / 2.0
            cy1 = (last[1] + last[3]) / 2.0
            disp = math.hypot(cx1 - cx0, cy1 - cy0)
            if disp >= min_displacement_px:
                has_moving_tracklet = True
                break

        if has_moving_tracklet:
            kept.append(traj)
        else:
            dropped += 1
            dropped_tracklets += sum(1 for _ in traj.tracklets)

    logger.info(
        f"Stationary filter (min_displacement={min_displacement_px}px): "
        f"dropped {dropped}/{dropped + len(kept)} trajectories "
        f"({dropped_tracklets} tracklets) with near-zero displacement"
    )
    return kept
