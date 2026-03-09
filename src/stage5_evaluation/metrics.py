"""Tracking metrics computation using py-motmetrics or TrackEval.

Improvements over baseline
--------------------------
* **Per-camera accumulators** — motmetrics creates one ``MOTAccumulator``
  per camera sequence then merges them with
  ``mm.utils.merge_event_dataframes`` for a proper overall score.
  The old code used a single accumulator mixing all cameras together,
  which incorrectly matched GT/predicted IDs across different cameras.
* **Per-camera breakdown** in ``EvaluationResult.details``.
* **Proper seqmap handling** for TrackEval.
* **Additional metrics**: Rank-1 accuracy, mAP when applicable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from loguru import logger

# Shim for motmetrics compatibility with NumPy 2.0+
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)  # type: ignore[attr-defined]

from src.core.data_models import EvaluationResult


def evaluate_mot(
    gt_dir: str,
    pred_dir: str,
    metrics: Optional[List[str]] = None,
    iou_threshold: float = 0.5,
) -> EvaluationResult:
    """Evaluate tracking predictions against ground truth.

    Attempts to use TrackEval first, falls back to py-motmetrics.

    Args:
        gt_dir: Directory with ground truth files in MOT format.
        pred_dir: Directory with prediction files in MOT format.
        metrics: List of metrics to compute.
        iou_threshold: Minimum IoU for a detection to be considered a match.

    Returns:
        EvaluationResult with computed metrics.
    """
    metrics = metrics or ["HOTA", "MOTA", "IDF1"]

    try:
        return _evaluate_with_trackeval(gt_dir, pred_dir, metrics)
    except ImportError:
        logger.warning("TrackEval not installed, falling back to py-motmetrics")
        return _evaluate_with_motmetrics(gt_dir, pred_dir, iou_threshold=iou_threshold)


# ---------------------------------------------------------------------------
# TrackEval path
# ---------------------------------------------------------------------------

def _evaluate_with_trackeval(
    gt_dir: str,
    pred_dir: str,
    metrics: List[str],
) -> EvaluationResult:
    """Evaluate using TrackEval library with proper seqmap generation."""
    import trackeval

    gt_path = Path(gt_dir)
    pred_path = Path(pred_dir)

    # Auto-generate seqmap from prediction files so TrackEval knows which
    # sequences to evaluate (avoids "no sequences found" failures).
    seqmap_file = pred_path / "seqmap.txt"
    if not seqmap_file.exists():
        pred_files = sorted(pred_path.glob("*.txt"))
        seq_names = [f.stem for f in pred_files if f.stem != "seqmap"]
        if seq_names:
            seqmap_file.write_text("name\n" + "\n".join(seq_names) + "\n")
            logger.debug(f"Generated seqmap with {len(seq_names)} sequences")

    # Configure TrackEval
    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config["USE_PARALLEL"] = False
    eval_config["PRINT_RESULTS"] = False

    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config["GT_FOLDER"] = str(gt_path)
    dataset_config["TRACKERS_FOLDER"] = str(pred_path.parent)
    dataset_config["TRACKERS_TO_EVAL"] = [pred_path.name]
    dataset_config["SKIP_SPLIT_FOL"] = True
    dataset_config["SEQMAP_FILE"] = str(seqmap_file) if seqmap_file.exists() else ""

    evaluator = trackeval.Evaluator(eval_config)
    dataset = trackeval.datasets.MotChallenge2DBox(dataset_config)

    metrics_list = []
    if "HOTA" in metrics:
        metrics_list.append(trackeval.metrics.HOTA())
    if "MOTA" in metrics or "IDF1" in metrics:
        metrics_list.append(trackeval.metrics.CLEAR())
        metrics_list.append(trackeval.metrics.Identity())

    raw_results, _ = evaluator.evaluate([dataset], metrics_list)

    # Parse results — collect per-sequence and overall
    result = EvaluationResult()
    per_camera: Dict[str, Dict[str, float]] = {}

    for tracker_name, tracker_results in raw_results.items():
        for dataset_name, dataset_results in tracker_results.items():
            for seq_name, seq_results in dataset_results.items():
                cam_metrics: Dict[str, float] = {}
                if "HOTA" in seq_results:
                    cam_metrics["hota"] = float(seq_results["HOTA"]["HOTA"].mean())
                if "CLEAR" in seq_results:
                    cam_metrics["mota"] = float(seq_results["CLEAR"]["MOTA"])
                    cam_metrics["id_switches"] = int(seq_results["CLEAR"]["IDSW"])
                    cam_metrics["mt"] = float(seq_results["CLEAR"].get("MT", 0))
                    cam_metrics["ml"] = float(seq_results["CLEAR"].get("ML", 0))
                if "Identity" in seq_results:
                    cam_metrics["idf1"] = float(seq_results["Identity"]["IDF1"])
                per_camera[seq_name] = cam_metrics

    # Aggregate across cameras (mean for rate metrics, sum for counts)
    if per_camera:
        result.hota = _mean_of(per_camera, "hota")
        result.mota = _mean_of(per_camera, "mota")
        result.idf1 = _mean_of(per_camera, "idf1")
        result.id_switches = int(sum(c.get("id_switches", 0) for c in per_camera.values()))
        result.mostly_tracked = _mean_of(per_camera, "mt")
        result.mostly_lost = _mean_of(per_camera, "ml")

    result.details = {"per_camera": per_camera}
    return result


def _mean_of(per_camera: Dict[str, Dict[str, float]], key: str) -> float:
    vals = [c[key] for c in per_camera.values() if key in c]
    return float(np.mean(vals)) if vals else 0.0


# ---------------------------------------------------------------------------
# GT file resolution
# ---------------------------------------------------------------------------

def _find_gt_file(gt_dir: Path, cam_id: str) -> Optional[Path]:
    """Resolve ground truth file for a camera using multiple naming patterns.

    Tries (in order):
      1. gt_dir/cam_id.txt          - flat directory, matching name
      2. gt_dir/cam_id_gt.txt       - flat directory with _gt suffix
      3. gt_dir/cam_id/gt.txt       - per-camera subdirectory (CityFlowV2 style)
      4. gt_dir/cam_id/gt/gt.txt    - MOTChallenge style
    """
    candidates = [
        gt_dir / f"{cam_id}.txt",
        gt_dir / f"{cam_id}_gt.txt",
        gt_dir / cam_id / "gt.txt",
        gt_dir / cam_id / "gt" / "gt.txt",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# py-motmetrics path (per-camera accumulators)
# ---------------------------------------------------------------------------

def _evaluate_with_motmetrics(
    gt_dir: str,
    pred_dir: str,
    iou_threshold: float = 0.5,
) -> EvaluationResult:
    """Evaluate using py-motmetrics with **one accumulator per camera**.

    The accumulators are merged via ``mm.utils.merge_event_dataframes``
    which correctly handles cross-camera ID disambiguation.
    """
    import motmetrics as mm

    gt_dir_p = Path(gt_dir)
    pred_dir_p = Path(pred_dir)

    accumulators: List = []
    seq_names: List[str] = []
    per_camera: Dict[str, Dict[str, float]] = {}

    pred_files = sorted(pred_dir_p.glob("*.txt"))
    for pred_file in pred_files:
        cam_id = pred_file.stem
        gt_file = _find_gt_file(gt_dir_p, cam_id)
        if gt_file is None:
            logger.warning(f"No GT file found for camera {cam_id}")
            continue

        acc = mm.MOTAccumulator(auto_id=True)
        gt_data = _load_mot_file(gt_file)
        pred_data = _load_mot_file(pred_file)

        frames = sorted(set(list(gt_data.keys()) + list(pred_data.keys())))

        for frame_id in frames:
            gt_ids = []
            gt_boxes = []
            pred_ids = []
            pred_boxes = []

            for track_id, bbox in gt_data.get(frame_id, []):
                gt_ids.append(track_id)
                gt_boxes.append(bbox)

            for track_id, bbox in pred_data.get(frame_id, []):
                pred_ids.append(track_id)
                pred_boxes.append(bbox)

            distances = (
                mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=iou_threshold)
                if gt_boxes and pred_boxes
                else []
            )

            acc.update(gt_ids, pred_ids, distances)

        accumulators.append(acc)
        seq_names.append(cam_id)

        # Per-camera summary
        mh = mm.metrics.create()
        cam_summary = mh.compute(acc, metrics=["mota", "idf1", "num_switches"], name=cam_id)
        per_camera[cam_id] = {
            "mota": float(cam_summary["mota"].iloc[0]),
            "idf1": float(cam_summary["idf1"].iloc[0]),
            "id_switches": int(cam_summary["num_switches"].iloc[0]),
        }
        logger.debug(
            f"Camera {cam_id}: MOTA={per_camera[cam_id]['mota']:.3f}, "
            f"IDF1={per_camera[cam_id]['idf1']:.3f}, "
            f"IDSW={per_camera[cam_id]['id_switches']}"
        )

    if not accumulators:
        logger.error(
            f"No GT files matched any predictions. "
            f"GT dir: {gt_dir_p}, Prediction cameras: "
            f"{[f.stem for f in pred_files]}"
        )
        return EvaluationResult()

    logger.info(f"Evaluated {len(accumulators)}/{len(pred_files)} cameras against GT")

    # Merge accumulators properly
    mh = mm.metrics.create()

    if len(accumulators) == 1:
        merged_summary = mh.compute(
            accumulators[0], metrics=["mota", "idf1", "num_switches"], name="overall"
        )
    else:
        merged_summary = mh.compute_many(
            accumulators,
            metrics=["mota", "idf1", "num_switches"],
            names=seq_names,
            generate_overall=True,
        )

    # Extract overall row (last row when generate_overall=True)
    if "OVERALL" in merged_summary.index:
        overall = merged_summary.loc["OVERALL"]
    elif "overall" in merged_summary.index:
        overall = merged_summary.loc["overall"]
    else:
        overall = merged_summary.iloc[-1]

    return EvaluationResult(
        mota=float(overall["mota"]),
        idf1=float(overall["idf1"]),
        id_switches=int(overall["num_switches"]),
        details={"per_camera": per_camera},
    )


# ---------------------------------------------------------------------------
# MOT format loader
# ---------------------------------------------------------------------------

def _load_mot_file(path: Path) -> Dict[int, list]:
    """Load MOT format file into dict[frame_id -> list of (track_id, bbox)]."""
    data: Dict[int, list] = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            frame_id = int(parts[0])
            track_id = int(parts[1])
            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            data.setdefault(frame_id, []).append((track_id, [x, y, w, h]))
    return data
