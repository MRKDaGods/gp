"""Evaluate ground-plane detection from Stage 1 tracklets directly.

This bypasses Stage 4 association and gives the "detection ceiling" —
the best possible MODA from our per-camera detections.

Usage:
    python scripts/eval_gp_from_tracklets.py --run data/outputs/run_tuned_v1
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.io_utils import load_tracklets_by_camera
from src.stage5_evaluation.ground_plane_eval import (
    _load_calibration,
    _pixel_to_ground,
    load_gt_ground_positions,
)

# Shim
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="Run output directory")
    parser.add_argument("--ann", default="data/raw/wildtrack/annotations_positions")
    parser.add_argument("--cal", default="data/raw/wildtrack/calibrations")
    parser.add_argument("--conf", type=float, default=0.30)
    parser.add_argument("--nms-radius", type=float, default=50.0)
    parser.add_argument("--threshold", type=float, default=50.0)
    args = parser.parse_args()

    run_dir = Path(args.run)
    stage1_dir = run_dir / "stage1"

    # Load tracklets
    tracklets_by_camera = load_tracklets_by_camera(stage1_dir)
    total = sum(len(v) for v in tracklets_by_camera.values())
    print(f"Loaded {total} tracklets from {len(tracklets_by_camera)} cameras")

    # Load calibrations
    cals = _load_calibration(Path(args.cal))
    print(f"Loaded calibrations for {len(cals)} cameras: {sorted(cals.keys())}")

    # Load GT
    gt = load_gt_ground_positions(Path(args.ann))
    avg_gt = np.mean([len(v) for v in gt.values()])
    print(f"GT: {len(gt)} frames, avg {avg_gt:.1f} people/frame")

    # Compute GT ground-plane ROI (bounding box of all GT positions + margin)
    all_gx, all_gy = [], []
    for positions in gt.values():
        for _, gx, gy in positions:
            all_gx.append(gx)
            all_gy.append(gy)
    roi_margin = 100.0  # cm
    roi_xmin, roi_xmax = min(all_gx) - roi_margin, max(all_gx) + roi_margin
    roi_ymin, roi_ymax = min(all_gy) - roi_margin, max(all_gy) + roi_margin
    print(f"GT ROI: ({roi_xmin:.0f},{roi_ymin:.0f}) - ({roi_xmax:.0f},{roi_ymax:.0f}) cm")

    # Build per-frame ground-plane detections from all cameras
    frame_detections: dict[int, list[tuple[int, float, float, float]]] = defaultdict(list)
    # (track_id, gx, gy, conf)

    det_count = 0
    for cam_id, tracklets in tracklets_by_camera.items():
        if cam_id not in cals:
            print(f"  Skip {cam_id}: no calibration")
            continue
        K = cals[cam_id]["K"]
        R = cals[cam_id]["R"]
        tvec = cals[cam_id]["tvec"]

        for tracklet in tracklets:
            for frame in tracklet.frames:
                if frame.confidence < args.conf:
                    continue
                x1, y1, x2, y2 = frame.bbox
                foot_x = (x1 + x2) / 2
                foot_y = y2
                gp = _pixel_to_ground(foot_x, foot_y, K, R, tvec)
                if gp is not None:
                    gx, gy = gp
                    # ROI filter: skip detections outside GT area
                    if gx < roi_xmin or gx > roi_xmax or gy < roi_ymin or gy > roi_ymax:
                        continue
                    frame_detections[frame.frame_id].append(
                        (tracklet.track_id, gx, gy, frame.confidence, cam_id)
                    )
                    det_count += 1

    print(f"Generated {det_count} ground-plane detections across {len(frame_detections)} frames")

    # Cluster per-frame detections using DBSCAN with multi-view consensus
    from sklearn.cluster import DBSCAN

    min_cameras = 2  # Require detection from at least N cameras

    frame_predictions: dict[int, list[tuple[int, float, float]]] = {}
    for frame_id, dets in sorted(frame_detections.items()):
        if not dets:
            frame_predictions[frame_id] = []
            continue

        positions = np.array([[d[1], d[2]] for d in dets])
        cam_ids = [d[4] for d in dets]
        clustering = DBSCAN(eps=args.nms_radius, min_samples=1).fit(positions)

        preds = []
        for label in set(clustering.labels_):
            mask = clustering.labels_ == label
            cluster_pos = positions[mask]
            cluster_cams = set(cam_ids[i] for i in range(len(cam_ids)) if mask[i])
            # Multi-view consensus: require at least min_cameras
            if len(cluster_cams) >= min_cameras:
                centroid = cluster_pos.mean(axis=0)
                preds.append((label, float(centroid[0]), float(centroid[1])))
        frame_predictions[frame_id] = preds

    avg_pred = np.mean([len(v) for v in frame_predictions.values()])
    print(f"After NMS (r={args.nms_radius}cm): avg {avg_pred:.1f} pred/frame vs {avg_gt:.1f} GT/frame")

    # Evaluate with motmetrics (detection-level, since IDs are pseudo)
    import motmetrics as mm
    acc = mm.MOTAccumulator(auto_id=True)

    all_frames = sorted(set(gt.keys()) | set(frame_predictions.keys()))
    for frame_id in all_frames:
        gt_list = gt.get(frame_id, [])
        pred_list = frame_predictions.get(frame_id, [])

        gt_ids = [g[0] for g in gt_list]
        pred_ids = [p[0] for p in pred_list]

        if gt_list and pred_list:
            gt_pos = np.array([[g[1], g[2]] for g in gt_list])
            pred_pos = np.array([[p[1], p[2]] for p in pred_list])
            dist = np.full((len(gt_pos), len(pred_pos)), np.nan)
            for i in range(len(gt_pos)):
                for j in range(len(pred_pos)):
                    d = np.sqrt((gt_pos[i, 0] - pred_pos[j, 0]) ** 2 +
                                (gt_pos[i, 1] - pred_pos[j, 1]) ** 2)
                    if d <= args.threshold:
                        dist[i, j] = d
        else:
            dist = np.empty((len(gt_list), len(pred_list)))

        acc.update(gt_ids, pred_ids, dist)

    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=["mota", "motp", "precision", "recall",
                 "num_false_positives", "num_misses",
                 "num_switches", "num_objects", "num_predictions"],
        name="gp_tracklets",
    )

    print("\n" + "=" * 60)
    print("GROUND-PLANE EVALUATION (from Stage 1 tracklets directly)")
    print("=" * 60)
    moda = summary["mota"].iloc[0]
    prec = summary["precision"].iloc[0]
    rec = summary["recall"].iloc[0]
    fp = summary["num_false_positives"].iloc[0]
    fn = summary["num_misses"].iloc[0]
    objects = summary["num_objects"].iloc[0]
    print(f"  MODA:      {moda * 100:6.1f}%")
    print(f"  Precision: {prec * 100:6.1f}%")
    print(f"  Recall:    {rec * 100:6.1f}%")
    print(f"  FP:        {fp}")
    print(f"  FN (Miss): {fn}")
    print(f"  Objects:   {objects}")
    print(f"  MODP (cm): {summary['motp'].iloc[0]:.1f}")

    # Try different confidence thresholds
    print("\n--- Confidence sweep ---")
    for conf in [0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70]:
        frame_dets_c: dict[int, list] = defaultdict(list)
        for cam_id, tracklets in tracklets_by_camera.items():
            if cam_id not in cals:
                continue
            K = cals[cam_id]["K"]
            R = cals[cam_id]["R"]
            tvec = cals[cam_id]["tvec"]
            for tracklet in tracklets:
                for frame in tracklet.frames:
                    if frame.confidence < conf:
                        continue
                    x1, y1, x2, y2 = frame.bbox
                    gp = _pixel_to_ground((x1+x2)/2, y2, K, R, tvec)
                    if gp:
                        gx, gy = gp
                        if gx < roi_xmin or gx > roi_xmax or gy < roi_ymin or gy > roi_ymax:
                            continue
                        frame_dets_c[frame.frame_id].append(gp)

        acc_c = mm.MOTAccumulator(auto_id=True)
        for frame_id in all_frames:
            gt_list = gt.get(frame_id, [])
            dets = frame_dets_c.get(frame_id, [])

            # Cluster
            if dets:
                positions = np.array(dets)
                clust = DBSCAN(eps=args.nms_radius, min_samples=1).fit(positions)
                preds = []
                for label in set(clust.labels_):
                    mask = clust.labels_ == label
                    cent = positions[mask].mean(axis=0)
                    preds.append((label, float(cent[0]), float(cent[1])))
            else:
                preds = []

            gt_ids = [g[0] for g in gt_list]
            pred_ids = [p[0] for p in preds]

            if gt_list and preds:
                gt_pos = np.array([[g[1], g[2]] for g in gt_list])
                pred_pos = np.array([[p[1], p[2]] for p in preds])
                dist = np.full((len(gt_pos), len(pred_pos)), np.nan)
                for i in range(len(gt_pos)):
                    for j in range(len(pred_pos)):
                        d = np.sqrt((gt_pos[i,0]-pred_pos[j,0])**2 +
                                    (gt_pos[i,1]-pred_pos[j,1])**2)
                        if d <= args.threshold:
                            dist[i, j] = d
            else:
                dist = np.empty((len(gt_list), len(preds)))

            acc_c.update(gt_ids, pred_ids, dist)

        s = mh.compute(acc_c, metrics=["mota", "precision", "recall",
                                        "num_false_positives", "num_misses"],
                       name=f"conf{conf}")
        m, p, r = s["mota"].iloc[0], s["precision"].iloc[0], s["recall"].iloc[0]
        fp_c, fn_c = s["num_false_positives"].iloc[0], s["num_misses"].iloc[0]
        avg_p = np.mean([len(frame_dets_c.get(f, [])) for f in all_frames]) if frame_dets_c else 0
        print(f"  conf>={conf:.2f}: MODA={m*100:6.1f}%, P={p*100:5.1f}%, R={r*100:5.1f}%, "
              f"FP={fp_c}, FN={fn_c}, avg_raw_det/f={avg_p:.0f}")


if __name__ == "__main__":
    main()
