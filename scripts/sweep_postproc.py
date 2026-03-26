"""Quick test: effect of stage5 post-processing (stationary filter, track smoothing) 
on the baseline config."""
import sys, json, time, tempfile, shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.core.config import load_config
from src.core.io_utils import load_embeddings, load_hsv_features, load_tracklets_by_camera
from src.core.data_models import TrackletFeatures
from src.stage3_indexing.faiss_index import FAISSIndex
from src.stage3_indexing import run_stage3
from src.stage4_association import run_stage4
from src.stage5_evaluation import run_stage5
from src.core.logging_utils import setup_logging

RUN_DIR = Path("data/outputs/kaggle_10a_v11/extracted/run_kaggle_20260318_114803")
CONFIG_PATH = str(RUN_DIR / "config.yaml")

BASE_OVERRIDES = {
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
    "stage4.association.secondary_embeddings.weight": "0.25",
}

EXPERIMENTS = [
    # Baseline (with GT-assist)
    {"name": "base_gt_assist", "overrides": {}},
    
    # Add stationary filter
    {"name": "stat_filter", "overrides": {
        "stage5.stationary_filter.enabled": "true",
        "stage5.stationary_filter.min_displacement_px": "50",
    }},
    
    # Add track smoothing
    {"name": "track_smooth", "overrides": {
        "stage5.track_smoothing.enabled": "true",
        "stage5.track_smoothing.window": "7",
        "stage5.track_smoothing.polyorder": "2",
    }},
    
    # Both
    {"name": "stat_smooth", "overrides": {
        "stage5.stationary_filter.enabled": "true",
        "stage5.stationary_filter.min_displacement_px": "50",
        "stage5.track_smoothing.enabled": "true",
        "stage5.track_smoothing.window": "7",
        "stage5.track_smoothing.polyorder": "2",
    }},
    
    # Without GT-assist but with stationary + smooth
    {"name": "no_gt_stat_smooth", "overrides": {
        "stage5.gt_frame_clip": "false",
        "stage5.gt_zone_filter": "false",
        "stage5.stationary_filter.enabled": "true",
        "stage5.stationary_filter.min_displacement_px": "50",
        "stage5.track_smoothing.enabled": "true",
        "stage5.track_smoothing.window": "7",
        "stage5.track_smoothing.polyorder": "2",
    }},

    # Without GT-assist, no extras 
    {"name": "no_gt_plain", "overrides": {
        "stage5.gt_frame_clip": "false",
        "stage5.gt_zone_filter": "false",
    }},

    # MTMC only (no single-cam) + stat + smooth (fair SOTA comparison)
    {"name": "mtmc_only_stat_smooth", "overrides": {
        "stage5.gt_frame_clip": "false",
        "stage5.gt_zone_filter": "false",
        "stage5.mtmc_only_submission": "true",
        "stage5.stationary_filter.enabled": "true",
        "stage5.stationary_filter.min_displacement_px": "50",
        "stage5.track_smoothing.enabled": "true",
        "stage5.track_smoothing.window": "7",
        "stage5.track_smoothing.polyorder": "2",
    }},
]


def run_experiment(name, extra_overrides):
    overrides = dict(BASE_OVERRIDES)
    overrides.update(extra_overrides)
    override_list = [f"{k}={v}" for k, v in overrides.items()]
    sec_path = str(RUN_DIR / "stage2" / "embeddings_secondary.npy")
    override_list.append(f"stage4.association.secondary_embeddings.path={sec_path}")
    
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
    
    tmp_dir = Path(tempfile.mkdtemp())
    faiss_index, metadata_store = run_stage3(
        cfg, features, tracklets_by_camera, output_dir=tmp_dir / "stage3"
    )
    
    t0 = time.time()
    trajectories = run_stage4(
        cfg, faiss_index, metadata_store, features,
        tracklets_by_camera, output_dir=tmp_dir / "stage4"
    )
    result = run_stage5(cfg, trajectories, output_dir=tmp_dir / "stage5")
    elapsed = time.time() - t0
    
    idf1 = getattr(result, "mtmc_idf1", 0.0) or 0.0
    mota = result.details.get("mtmc_mota", 0.0) if hasattr(result, "details") else 0.0
    
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return {"name": name, "idf1": round(idf1, 4), "mota": round(mota, 4), "time_s": round(elapsed, 1)}


def main():
    setup_logging(level="WARNING")
    results = []
    
    for i, exp in enumerate(EXPERIMENTS):
        name = exp["name"]
        print(f"\n[{i+1}/{len(EXPERIMENTS)}] {name}")
        try:
            result = run_experiment(name, exp["overrides"])
            results.append(result)
            print(f"  IDF1={result['idf1']:.4f}  MOTA={result['mota']:.4f}  time={result['time_s']:.1f}s")
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback; traceback.print_exc()
            results.append({"name": name, "idf1": 0.0, "error": str(e)})
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    results.sort(key=lambda r: r.get("idf1", 0), reverse=True)
    for r in results:
        print(f"  {r['name']:30s} IDF1={r.get('idf1',0):.4f}  MOTA={r.get('mota',0):.4f}")
    
    with open("data/outputs/sweep_stage5_postproc.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to data/outputs/sweep_stage5_postproc.json")


if __name__ == "__main__":
    main()
