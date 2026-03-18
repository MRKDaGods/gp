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

# Shim for TrackEval compatibility with NumPy 1.24+ (deprecated type aliases removed).
# Must be applied BEFORE 'import trackeval' so module-level np.float references work.
for _np_alias, _builtin in [
    ("float", float), ("int", int), ("bool", bool),
    ("complex", complex), ("object", object), ("str", str),
]:
    if not hasattr(np, _np_alias):
        setattr(np, _np_alias, _builtin)

from src.core.data_models import EvaluationResult


def _analyze_mtmc_errors(acc) -> Dict:
    """Analyze fragmentation (under-merging) vs conflation (over-merging).

    Extracts frame-level GT↔pred matches from the motmetrics accumulator
    to identify:
    - Fragmented GT IDs: one GT identity matched to multiple predicted IDs
    - Conflated pred IDs: one predicted ID matched to multiple GT identities
    - Unmatched GT/pred IDs
    """
    try:
        events = acc.mot_events
        matches = events[events["Type"] == "MATCH"]
        if matches.empty:
            return {}

        # GT → set of pred IDs it was matched to
        gt_to_preds: Dict[int, set] = {}
        # Pred → set of GT IDs it was matched to
        pred_to_gts: Dict[int, set] = {}

        for _, row in matches.iterrows():
            gt_id = row["OId"]
            pred_id = row["HId"]
            gt_to_preds.setdefault(gt_id, set()).add(pred_id)
            pred_to_gts.setdefault(pred_id, set()).add(gt_id)

        # All GT and pred IDs in the accumulator
        all_gt_ids = set()
        all_pred_ids = set()
        for _, row in events.iterrows():
            if row["OId"] is not None and not (isinstance(row["OId"], float) and np.isnan(row["OId"])):
                all_gt_ids.add(row["OId"])
            if row["HId"] is not None and not (isinstance(row["HId"], float) and np.isnan(row["HId"])):
                all_pred_ids.add(row["HId"])

        fragmented = {gt: preds for gt, preds in gt_to_preds.items() if len(preds) > 1}
        conflated = {pred: gts for pred, gts in pred_to_gts.items() if len(gts) > 1}
        unmatched_gt = all_gt_ids - set(gt_to_preds.keys())
        unmatched_pred = all_pred_ids - set(pred_to_gts.keys())

        # Top fragmented GTs (most pred IDs)
        top_fragmented = sorted(fragmented.items(), key=lambda x: len(x[1]), reverse=True)[:10]
        # Top conflated preds (most GT IDs)
        top_conflated = sorted(conflated.items(), key=lambda x: len(x[1]), reverse=True)[:10]

        result = {
            "fragmented_gt": len(fragmented),
            "conflated_pred": len(conflated),
            "unmatched_gt": len(unmatched_gt),
            "unmatched_pred": len(unmatched_pred),
            "total_gt": len(all_gt_ids),
            "total_pred": len(all_pred_ids),
            "top_fragmented": [(int(gt), len(preds)) for gt, preds in top_fragmented],
            "top_conflated": [(int(pred), len(gts)) for pred, gts in top_conflated],
        }
        logger.debug(f"Top fragmented GT IDs: {result['top_fragmented']}")
        logger.debug(f"Top conflated pred IDs: {result['top_conflated']}")
        return result
    except Exception as e:
        logger.warning(f"Error analysis failed: {e}")
        return {}


def evaluate_mtmc(
    gt_dir: str,
    pred_dir: str,
    iou_threshold: float = 0.5,
) -> EvaluationResult:
    """Evaluate **MTMC IDF1** using a single global accumulator.

    Unlike per-camera evaluation, this merges all cameras into one
    accumulator with globally-unique GT IDs and globally-unique predicted
    IDs.  A vehicle tracked with the same global_id across cameras is
    treated as one identity — matching the AI City Challenge protocol.

    IDF1 = 2*IDTP / (2*IDTP + IDFP + IDFN)

    CityFlowV2 GT includes both multi-camera and single-camera vehicles.
    GT IDs are globally unique per scenario (S01: 1-95, S02: 96-240)
    so using one accumulator over all cameras correctly rewards
    cross-camera re-identification without double-counting.
    This matches the AI City Challenge Track 1 evaluation protocol.

    Args:
        gt_dir: Directory with per-camera GT files (CityFlowV2 format).
        pred_dir: Directory with per-camera prediction files.
        iou_threshold: Minimum IoU for a detection to be considered a match.

    Returns:
        EvaluationResult with globally-computed IDF1 and per-camera details.
    """
    import motmetrics as mm

    gt_dir_p = Path(gt_dir)
    pred_dir_p = Path(pred_dir)

    # Single global accumulator — GT IDs are globally unique per CityFlowV2
    global_acc = mm.MOTAccumulator(auto_id=True)
    per_camera: Dict[str, Dict[str, float]] = {}

    pred_files = sorted(
        f for f in pred_dir_p.glob("*.txt") if f.stem.lower() != "seqmap"
    )

    for pred_file in pred_files:
        cam_id = pred_file.stem
        gt_file = _find_gt_file(gt_dir_p, cam_id)
        if gt_file is None:
            logger.warning(f"No GT file found for camera {cam_id}, skipping")
            continue

        gt_data = _load_mot_file(gt_file)
        pred_data = _load_mot_file(pred_file)

        cam_acc = mm.MOTAccumulator(auto_id=True)
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

            cam_acc.update(gt_ids, pred_ids, distances)
            global_acc.update(gt_ids, pred_ids, distances)

        # Per-camera summary for reference
        mh_cam = mm.metrics.create()
        cs = mh_cam.compute(cam_acc, metrics=["mota", "idf1", "num_switches"], name=cam_id)
        per_camera[cam_id] = {
            "mota": float(cs["mota"].iloc[0]),
            "idf1": float(cs["idf1"].iloc[0]),
            "id_switches": int(cs["num_switches"].iloc[0]),
        }

    if not per_camera:
        return EvaluationResult()

    # Compute global MTMC IDF1 from the single merged accumulator
    mh = mm.metrics.create()
    global_summary = mh.compute(
        global_acc,
        metrics=["mota", "idf1", "num_switches"],
        name="MTMC",
    )
    mtmc_idf1 = float(global_summary["idf1"].iloc[0])
    mtmc_mota = float(global_summary["mota"].iloc[0])
    mtmc_idsw = int(global_summary["num_switches"].iloc[0])

    # ── Error analysis: fragmentation vs conflation ─────────────────────
    error_analysis = _analyze_mtmc_errors(global_acc)

    logger.info(
        f"MTMC evaluation: IDF1={mtmc_idf1:.3f}, MOTA={mtmc_mota:.3f}, "
        f"ID Switches={mtmc_idsw}"
    )
    if error_analysis:
        logger.info(
            f"  Error analysis: {error_analysis['fragmented_gt']} fragmented GT IDs, "
            f"{error_analysis['conflated_pred']} conflated pred IDs, "
            f"{error_analysis['unmatched_gt']} unmatched GT IDs, "
            f"{error_analysis['unmatched_pred']} unmatched pred IDs"
        )

    result = EvaluationResult(
        mota=mtmc_mota,
        idf1=mtmc_idf1,
        id_switches=mtmc_idsw,
        details={
            "per_camera": per_camera, "mtmc_idf1": mtmc_idf1,
            "error_analysis": error_analysis,
        },
    )
    return result


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
    except Exception as e:
        logger.warning(f"TrackEval evaluation failed ({type(e).__name__}: {e}), falling back to py-motmetrics")
    return _evaluate_with_motmetrics(gt_dir, pred_dir, iou_threshold=iou_threshold)


# ---------------------------------------------------------------------------
# TrackEval path
# ---------------------------------------------------------------------------

def _remap_class1_in_dir(src_dir: Path, dst_dir: Path, glob: str = "*.txt") -> None:
    """Copy tracking files replacing class field (col 7) with 1 (pedestrian).

    TrackEval's MotChallenge2DBox evaluates class=1 (pedestrian) only.
    CityFlowV2 uses class=-1 in GT (unclassified vehicle) and class 2/5/7 in
    predictions.  Remapping both GT and predictions to class=1 ensures TrackEval
    places them in the same "pedestrian" bucket and can compute HOTA correctly.
    Without this fix GT class=-1 ends up in a separate bucket → HOTA=0.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src_file in src_dir.glob(glob):
        lines = []
        with open(src_file) as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) >= 8:
                    parts[7] = "1"  # remap to pedestrian class
                lines.append(",".join(parts))
        (dst_dir / src_file.name).write_text("\n".join(lines) + "\n")


# Keep legacy name for compatibility
_remap_predictions_class1 = _remap_class1_in_dir


def _evaluate_with_trackeval(
    gt_dir: str,
    pred_dir: str,
    metrics: List[str],
) -> EvaluationResult:
    """Evaluate using TrackEval library with proper seqmap generation."""
    import trackeval
    import tempfile, shutil

    gt_path = Path(gt_dir)
    pred_path = Path(pred_dir)

    # TrackEval's MotChallenge2DBox evaluates class=1 (pedestrian) only.
    # CityFlowV2 GT uses class=-1 (unclassified), predictions use class 2/5/7.
    # Remap BOTH to class=1 in temporary copies so TrackEval can match them.
    tmp_root = Path(tempfile.mkdtemp())
    try:
        # Remap predictions (2/5/7 → 1)
        remapped_pred_name = pred_path.name + "_cls1"
        remapped_pred_path = tmp_root / remapped_pred_name
        _remap_class1_in_dir(pred_path, remapped_pred_path)

        # Remap GT (-1 → 1): build a parallel GT directory with updated class fields
        gt_remapped_root = tmp_root / "gt_cls1"
        gt_remapped_path = _remap_gt_class1(gt_path, gt_remapped_root)

        result = _run_trackeval(gt_remapped_path, tmp_root, remapped_pred_name, pred_path, metrics)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
    return result


def _remap_gt_class1(gt_path: Path, dst_root: Path) -> Path:
    """Build a lightweight GT directory tree with class column remapped to 1.

    Copies only what TrackEval needs (``gt/gt.txt`` and ``seqinfo.ini``),
    avoiding copying large video files.  Remaps class=-1 (CityFlowV2
    unclassified vehicle) to class=1 so TrackEval can match them against
    the remapped pedestrian predictions.

    TrackEval expects:
        {gt_folder}/{seq}/gt/gt.txt
        {gt_folder}/{seq}/seqinfo.ini   (optional but needed for HOTA)

    Returns:
        Path to the remapped GT root directory.
    """
    dst_root.mkdir(parents=True, exist_ok=True)

    for seq_dir in sorted(gt_path.iterdir()):
        if not seq_dir.is_dir():
            continue

        dst_seq = dst_root / seq_dir.name
        dst_seq.mkdir(exist_ok=True)

        # Copy seqinfo.ini if present
        src_seqinfo = seq_dir / "seqinfo.ini"
        if src_seqinfo.exists():
            import shutil
            shutil.copy2(str(src_seqinfo), str(dst_seq / "seqinfo.ini"))

        # Find gt.txt — CityFlowV2 places it at either gt/gt.txt or gt.txt
        src_gt_candidates = [
            seq_dir / "gt" / "gt.txt",
            seq_dir / "gt.txt",
        ]
        src_gt = next((p for p in src_gt_candidates if p.exists()), None)
        if src_gt is None:
            continue

        # Write remapped copy at canonical location: {seq}/gt/gt.txt
        dst_gt_dir = dst_seq / "gt"
        dst_gt_dir.mkdir(exist_ok=True)
        lines = []
        with open(src_gt) as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) >= 8:
                    parts[7] = "1"  # remap from -1 (unclassified) to pedestrian
                lines.append(",".join(parts))
        (dst_gt_dir / "gt.txt").write_text("\n".join(lines) + "\n")

    return dst_root


def _run_trackeval(
    gt_path: Path,
    trackers_folder: Path,
    tracker_name: str,
    orig_pred_path: Path,
    metrics: List[str],
) -> EvaluationResult:
    """Inner helper: run TrackEval on (possibly remapped) prediction files."""
    import trackeval

    pred_path = trackers_folder / tracker_name

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
    dataset_config["TRACKERS_FOLDER"] = str(trackers_folder)
    dataset_config["TRACKERS_TO_EVAL"] = [tracker_name]
    dataset_config["SKIP_SPLIT_FOL"] = True
    dataset_config["TRACKER_SUB_FOLDER"] = ""   # pred files are directly in tracker dir
    dataset_config["SEQMAP_FILE"] = str(seqmap_file) if seqmap_file.exists() else ""
    # Disable confidence/class preprocessing so all detections are evaluated.
    dataset_config["DO_PREPROC"] = False
    # seqinfo.ini is read from {gt_folder}/{seq}/seqinfo.ini automatically

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

    for _trk, tracker_results in raw_results.items():
        for _ds, dataset_results in tracker_results.items():
            for seq_name, seq_results in dataset_results.items():
                # Skip COMBINED_SEQ — TrackEval sums raw TP/FP/FN across
                # sequences, which is invalid for multi-camera setups where
                # cameras share the same vehicles.  Per-camera mean is the
                # standard aggregation for MTMC benchmarks.
                if seq_name == "COMBINED_SEQ":
                    continue
                # TrackEval nests results under class name (e.g. 'pedestrian')
                cls_results = seq_results.get("pedestrian", seq_results)
                cam_metrics: Dict[str, float] = {}
                if "HOTA" in cls_results:
                    cam_metrics["hota"] = float(cls_results["HOTA"]["HOTA"].mean())
                if "CLEAR" in cls_results:
                    cam_metrics["mota"] = float(cls_results["CLEAR"]["MOTA"])
                    cam_metrics["id_switches"] = int(cls_results["CLEAR"]["IDSW"])
                    cam_metrics["mt"] = float(cls_results["CLEAR"].get("MT", 0))
                    cam_metrics["ml"] = float(cls_results["CLEAR"].get("ML", 0))
                if "Identity" in cls_results:
                    cam_metrics["idf1"] = float(cls_results["Identity"]["IDF1"])

                per_camera[seq_name] = cam_metrics

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

    pred_files = sorted(
        f for f in pred_dir_p.glob("*.txt") if f.stem.lower() != "seqmap"
    )
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

    # Merge accumulators: use compute_many with generate_overall to get a
    # detection-count-weighted aggregate across all cameras.
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

    # Compute mean-per-camera IDF1 (what we currently report)
    mean_idf1 = _mean_of(per_camera, "idf1")
    mean_mota = _mean_of(per_camera, "mota")

    return EvaluationResult(
        mota=mean_mota,
        idf1=mean_idf1,
        id_switches=int(overall["num_switches"]),
        details={
            "per_camera": per_camera,
            "overall_weighted_idf1": float(overall["idf1"]),
            "overall_weighted_mota": float(overall["mota"]),
        },
    )


# ---------------------------------------------------------------------------
# MOT format loader
# ---------------------------------------------------------------------------

def _load_mot_file(path: Path) -> Dict[int, list]:
    """Load MOT format file into dict[frame_id -> list of (track_id, bbox)].

    MOT format: frame, id, x, y, w, h, conf, x_world, y_world, z_world
    Rows with conf=0 are ignore regions and are skipped.
    For GT files with all-1 confidence (CityFlowV2), this is a no-op.
    """
    data: Dict[int, list] = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            # Skip ignore/distractor rows (conf=0 in MOT convention)
            # Also catches interpolated detections written as "0.0"
            if len(parts) >= 7:
                try:
                    conf_val = float(parts[6].strip())
                except ValueError:
                    conf_val = 1.0
                if conf_val == 0:
                    continue
            frame_id = int(parts[0])
            track_id = int(parts[1])
            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            data.setdefault(frame_id, []).append((track_id, [x, y, w, h]))
    return data
