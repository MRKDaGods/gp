"""Deep error analysis: understand exactly WHERE we lose IDF1 points.
Runs best config, saves predictions, compares with GT at per-camera and per-trajectory level."""
import sys, json, time, tempfile, shutil, collections
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.core.config import load_config
from src.core.io_utils import load_embeddings, load_hsv_features, load_tracklets_by_camera
from src.core.data_models import TrackletFeatures
from src.stage3_indexing import run_stage3
from src.stage4_association import run_stage4
from src.stage5_evaluation import run_stage5
from src.core.logging_utils import setup_logging

RUN_DIR = Path("data/outputs/kaggle_10a_v11/extracted/run_kaggle_20260318_114803")
CONFIG_PATH = str(RUN_DIR / "config.yaml")
SEC_EMB_PATH = str(RUN_DIR / "stage2" / "embeddings_secondary.npy")
GT_DIR = Path("data/raw/cityflowv2")

BEST_OVERRIDES = {
    "stage4.association.graph.similarity_threshold": "0.53",
    "stage4.association.fic.regularisation": "0.15",
    "stage4.association.fac.enabled": "true",
    "stage4.association.fac.knn": "20",
    "stage4.association.fac.learning_rate": "0.5",
    "stage4.association.fac.beta": "0.08",
    "stage4.association.query_expansion.k": "3",
    "stage4.association.intra_camera_merge.enabled": "true",
    "stage4.association.intra_camera_merge.threshold": "0.75",
    "stage4.association.intra_camera_merge.max_time_gap": "60",
    f"stage4.association.secondary_embeddings.path": SEC_EMB_PATH,
    "stage4.association.secondary_embeddings.weight": "0.25",
    "stage5.stationary_filter.enabled": "true",
    "stage5.stationary_filter.min_displacement_px": "150",
}


def load_gt(cam_dir):
    """Load GT file: frame,id,x,y,w,h,conf,class,vis"""
    gt_path = cam_dir / "gt" / "gt.txt"
    if not gt_path.exists():
        return {}
    gt_by_id = collections.defaultdict(list)
    for line in open(gt_path):
        parts = line.strip().split(",")
        if len(parts) < 6:
            continue
        fid, tid = int(parts[0]), int(parts[1])
        gt_by_id[tid].append(fid)
    return gt_by_id


def analyze_trajectories(trajectories):
    """Analyze trajectory properties."""
    stats = {
        "total_trajectories": len(trajectories),
        "cross_camera": 0,
        "single_camera": 0,
        "by_num_cameras": collections.Counter(),
        "by_camera_set": collections.Counter(),
        "total_tracklets": 0,
        "total_frames": 0,
        "confidence_dist": [],
        "duration_dist": [],
    }
    
    for traj in trajectories:
        n_cams = len(set(t.camera_id for t in traj.tracklets))
        stats["by_num_cameras"][n_cams] += 1
        if n_cams > 1:
            stats["cross_camera"] += 1
        else:
            stats["single_camera"] += 1
        
        cam_set = tuple(sorted(set(t.camera_id for t in traj.tracklets)))
        stats["by_camera_set"][str(cam_set)] += 1
        
        stats["total_tracklets"] += len(traj.tracklets)
        stats["total_frames"] += sum(len(t.frames) for t in traj.tracklets)
        stats["confidence_dist"].append(traj.confidence)
        stats["duration_dist"].append(traj.time_span)
    
    return stats


def analyze_gt():
    """Analyze ground truth structure."""
    cameras = ["S01_c001", "S01_c002", "S01_c003", "S02_c006", "S02_c007", "S02_c008"]
    all_gt_ids = {}
    gt_stats = {}
    
    for cam in cameras:
        cam_dir = None
        for scene in ["S01", "S02"]:
            p = GT_DIR / scene / cam
            if p.exists():
                cam_dir = p
                break
        if not cam_dir:
            continue
        
        gt_by_id = load_gt(cam_dir)
        gt_stats[cam] = {
            "num_ids": len(gt_by_id),
            "total_frames": sum(len(frames) for frames in gt_by_id.values()),
            "avg_track_len": np.mean([len(f) for f in gt_by_id.values()]) if gt_by_id else 0,
        }
        
        for tid, frames in gt_by_id.items():
            if tid not in all_gt_ids:
                all_gt_ids[tid] = set()
            all_gt_ids[tid].add(cam)
    
    # How many GT IDs appear in multiple cameras?
    cross_cam_ids = {tid: cams for tid, cams in all_gt_ids.items() if len(cams) > 1}
    single_cam_ids = {tid: cams for tid, cams in all_gt_ids.items() if len(cams) == 1}
    
    return {
        "per_camera": gt_stats,
        "total_unique_ids": len(all_gt_ids),
        "cross_camera_ids": len(cross_cam_ids),
        "single_camera_ids": len(single_cam_ids),
        "cross_cam_cameras": collections.Counter(
            str(tuple(sorted(cams))) for cams in cross_cam_ids.values()
        ),
    }


def main():
    setup_logging(level="WARNING")
    
    print("=" * 70)
    print("ERROR ANALYSIS")
    print("=" * 70)
    
    # 1. Analyze GT
    print("\n--- GROUND TRUTH ANALYSIS ---")
    gt_info = analyze_gt()
    print(f"Total unique GT IDs: {gt_info['total_unique_ids']}")
    print(f"Cross-camera IDs:    {gt_info['cross_camera_ids']}")
    print(f"Single-camera IDs:   {gt_info['single_camera_ids']}")
    print(f"\nPer-camera GT:")
    for cam, stats in sorted(gt_info["per_camera"].items()):
        print(f"  {cam}: {stats['num_ids']} IDs, {stats['total_frames']} frames, avg len={stats['avg_track_len']:.0f}")
    print(f"\nCross-camera GT ID distributions:")
    for cam_set, count in gt_info["cross_cam_cameras"].most_common():
        print(f"  {cam_set}: {count} IDs")
    
    # 2. Run best config  
    print("\n--- RUNNING BEST CONFIG ---")
    overrides = dict(BEST_OVERRIDES)
    override_list = [f"{k}={v}" for k, v in overrides.items()]
    cfg = load_config(CONFIG_PATH, overrides=override_list)
    
    embeddings, index_map = load_embeddings(str(RUN_DIR / "stage2"))
    hsv_matrix = load_hsv_features(str(RUN_DIR / "stage2"))
    features = [
        TrackletFeatures(
            track_id=m["track_id"], camera_id=m["camera_id"],
            class_id=m["class_id"], embedding=embeddings[i],
            hsv_histogram=hsv_matrix[i],
        )
        for i, m in enumerate(index_map)
    ]
    tracklets_by_camera = load_tracklets_by_camera(str(RUN_DIR / "stage1"))
    
    out_dir = Path("data/outputs/error_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    faiss_index, metadata_store = run_stage3(
        cfg, features, tracklets_by_camera, output_dir=out_dir / "stage3"
    )
    
    trajectories = run_stage4(
        cfg, faiss_index, metadata_store, features,
        tracklets_by_camera, output_dir=out_dir / "stage4"
    )
    result = run_stage5(cfg, trajectories, output_dir=out_dir / "stage5")
    
    idf1 = getattr(result, "mtmc_idf1", 0.0) or 0.0
    mota = result.details.get("mtmc_mota", 0.0) if hasattr(result, "details") else 0.0
    print(f"\nResult: IDF1={idf1:.4f}  MOTA={mota:.4f}")
    
    # 3. Analyze our trajectories
    print("\n--- TRAJECTORY ANALYSIS ---")
    traj_stats = analyze_trajectories(trajectories)
    print(f"Total trajectories:  {traj_stats['total_trajectories']}")
    print(f"Cross-camera:        {traj_stats['cross_camera']}")
    print(f"Single-camera:       {traj_stats['single_camera']}")
    print(f"Total tracklets:     {traj_stats['total_tracklets']}")
    print(f"Total frames:        {traj_stats['total_frames']}")
    
    print(f"\nBy number of cameras:")
    for n, cnt in sorted(traj_stats["by_num_cameras"].items()):
        print(f"  {n} cam(s): {cnt} trajectories")
    
    print(f"\nBy camera set (top 20):")
    for cam_set, cnt in collections.Counter(traj_stats["by_camera_set"]).most_common(20):
        print(f"  {cam_set}: {cnt}")
    
    if traj_stats["confidence_dist"]:
        confs = np.array(traj_stats["confidence_dist"])
        print(f"\nTrajectory confidence: mean={confs.mean():.3f}, "
              f"std={confs.std():.3f}, min={confs.min():.3f}, "
              f"p25={np.percentile(confs, 25):.3f}, "
              f"median={np.median(confs):.3f}, "
              f"p75={np.percentile(confs, 75):.3f}, "
              f"max={confs.max():.3f}")
    
    # 4. Per-camera tracking stats
    print("\n--- PER-CAMERA TRACKING STATS ---")
    for cam, tracklets in sorted(tracklets_by_camera.items()):
        n_tracks = len(tracklets)
        total_frames = sum(len(t.frames) for t in tracklets)
        avg_len = total_frames / n_tracks if n_tracks > 0 else 0
        print(f"  {cam}: {n_tracks} tracklets, {total_frames} frames, avg len={avg_len:.0f}")
    
    # 5. Analyze per-camera eval details
    print("\n--- PER-CAMERA EVALUATION DETAILS ---")
    if hasattr(result, "details"):
        for key, val in sorted(result.details.items()):
            if isinstance(val, float):
                print(f"  {key}: {val:.4f}")
            else:
                print(f"  {key}: {val}")
    
    # 6. Compare pred vs GT track counts
    print("\n--- PRED vs GT TRACK COUNTS PER CAMERA ---")
    pred_tracks_per_cam = collections.defaultdict(set)
    for traj in trajectories:
        for t in traj.tracklets:
            pred_tracks_per_cam[t.camera_id].add(traj.global_id)
    
    for cam in sorted(set(list(gt_info["per_camera"].keys()) + list(pred_tracks_per_cam.keys()))):
        gt_count = gt_info["per_camera"].get(cam, {}).get("num_ids", 0)
        pred_count = len(pred_tracks_per_cam.get(cam, set()))
        ratio = pred_count / gt_count if gt_count > 0 else float("inf")
        print(f"  {cam}: GT={gt_count}, Pred={pred_count}, ratio={ratio:.2f}x")
    
    # 7. Analyze tracklet embedding quality
    print("\n--- EMBEDDING ANALYSIS ---")
    # Per-camera similarity stats  
    from scipy.spatial.distance import cdist
    for cam in sorted(tracklets_by_camera.keys()):
        cam_indices = [i for i, m in enumerate(index_map) if m["camera_id"] == cam]
        if len(cam_indices) < 2:
            continue
        cam_embs = embeddings[cam_indices]
        sims = 1 - cdist(cam_embs, cam_embs, metric="cosine")
        np.fill_diagonal(sims, 0)
        # get max off-diagonal per row
        max_sims = sims.max(axis=1)
        print(f"  {cam}: {len(cam_indices)} embs, intra-cam max_sim mean={max_sims.mean():.3f}, "
              f"std={max_sims.std():.3f}")
    
    # Cross-camera similarity
    scene1_cams = ["S01_c001", "S01_c002", "S01_c003"]
    scene2_cams = ["S02_c006", "S02_c007", "S02_c008"]
    
    for scene_cams, scene_name in [(scene1_cams, "S01"), (scene2_cams, "S02")]:
        for i, cam_a in enumerate(scene_cams):
            for cam_b in scene_cams[i+1:]:
                idx_a = [j for j, m in enumerate(index_map) if m["camera_id"] == cam_a]
                idx_b = [j for j, m in enumerate(index_map) if m["camera_id"] == cam_b]
                if not idx_a or not idx_b:
                    continue
                cross_sims = 1 - cdist(embeddings[idx_a], embeddings[idx_b], metric="cosine")
                top1 = cross_sims.max(axis=1)
                print(f"  {cam_a} → {cam_b}: {len(idx_a)}x{len(idx_b)}, "
                      f"top1 mean={top1.mean():.3f}, std={top1.std():.3f}, "
                      f"max={top1.max():.3f}")
    
    print("\n--- ANALYSIS COMPLETE ---")
    
    # Save results
    summary = {
        "gt_info": {
            "total_unique_ids": gt_info["total_unique_ids"],
            "cross_camera_ids": gt_info["cross_camera_ids"],
            "single_camera_ids": gt_info["single_camera_ids"],
        },
        "pred_info": {
            "total_trajectories": traj_stats["total_trajectories"],
            "cross_camera": traj_stats["cross_camera"],
            "single_camera": traj_stats["single_camera"],
        },
        "idf1": idf1,
        "mota": mota,
    }
    with open(out_dir / "analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
