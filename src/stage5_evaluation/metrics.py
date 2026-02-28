"""Tracking metrics computation using py-motmetrics or TrackEval."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

from src.core.data_models import EvaluationResult


def evaluate_mot(
    gt_dir: str,
    pred_dir: str,
    metrics: Optional[List[str]] = None,
) -> EvaluationResult:
    """Evaluate tracking predictions against ground truth.

    Attempts to use TrackEval first, falls back to py-motmetrics.

    Args:
        gt_dir: Directory with ground truth files in MOT format.
        pred_dir: Directory with prediction files in MOT format.
        metrics: List of metrics to compute.

    Returns:
        EvaluationResult with computed metrics.
    """
    metrics = metrics or ["HOTA", "MOTA", "IDF1"]

    try:
        return _evaluate_with_trackeval(gt_dir, pred_dir, metrics)
    except ImportError:
        logger.warning("TrackEval not installed, falling back to py-motmetrics")
        return _evaluate_with_motmetrics(gt_dir, pred_dir)


def _evaluate_with_trackeval(
    gt_dir: str,
    pred_dir: str,
    metrics: List[str],
) -> EvaluationResult:
    """Evaluate using TrackEval library."""
    import trackeval

    # Configure TrackEval
    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config["USE_PARALLEL"] = False
    eval_config["PRINT_RESULTS"] = False

    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config["GT_FOLDER"] = gt_dir
    dataset_config["TRACKERS_FOLDER"] = str(Path(pred_dir).parent)
    dataset_config["TRACKERS_TO_EVAL"] = [Path(pred_dir).name]
    dataset_config["SKIP_SPLIT_FOL"] = True

    evaluator = trackeval.Evaluator(eval_config)
    dataset = trackeval.datasets.MotChallenge2DBox(dataset_config)

    metrics_list = []
    if "HOTA" in metrics:
        metrics_list.append(trackeval.metrics.HOTA())
    if "MOTA" in metrics or "IDF1" in metrics:
        metrics_list.append(trackeval.metrics.CLEAR())
        metrics_list.append(trackeval.metrics.Identity())

    raw_results, _ = evaluator.evaluate([dataset], metrics_list)

    # Parse results
    result = EvaluationResult()
    details = {}

    # TrackEval returns nested dict structure
    for tracker_name, tracker_results in raw_results.items():
        for dataset_name, dataset_results in tracker_results.items():
            for seq_name, seq_results in dataset_results.items():
                if "HOTA" in seq_results:
                    result.hota = float(seq_results["HOTA"]["HOTA"].mean())
                if "CLEAR" in seq_results:
                    result.mota = float(seq_results["CLEAR"]["MOTA"])
                    result.id_switches = int(seq_results["CLEAR"]["IDSW"])
                    result.mostly_tracked = float(seq_results["CLEAR"].get("MT", 0))
                    result.mostly_lost = float(seq_results["CLEAR"].get("ML", 0))
                if "Identity" in seq_results:
                    result.idf1 = float(seq_results["Identity"]["IDF1"])

    result.details = details
    return result


def _evaluate_with_motmetrics(gt_dir: str, pred_dir: str) -> EvaluationResult:
    """Evaluate using py-motmetrics library."""
    import motmetrics as mm

    gt_dir = Path(gt_dir)
    pred_dir = Path(pred_dir)

    accumulator = mm.MOTAccumulator(auto_id=True)

    # Find matching GT and prediction files
    gt_files = sorted(gt_dir.glob("*.txt"))

    for gt_file in gt_files:
        pred_file = pred_dir / gt_file.name
        if not pred_file.exists():
            logger.warning(f"No prediction file for {gt_file.name}")
            continue

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

            distances = mm.distances.iou_matrix(
                gt_boxes, pred_boxes, max_iou=0.5
            ) if gt_boxes and pred_boxes else []

            accumulator.update(gt_ids, pred_ids, distances)

    # Compute metrics
    mh = mm.metrics.create()
    summary = mh.compute(accumulator, metrics=["mota", "idf1", "num_switches"], name="overall")

    return EvaluationResult(
        mota=float(summary["mota"].iloc[0]),
        idf1=float(summary["idf1"].iloc[0]),
        id_switches=int(summary["num_switches"].iloc[0]),
    )


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
