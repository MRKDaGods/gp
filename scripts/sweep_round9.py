"""Round 9: Targeted improvements based on error analysis.
Focus: reduce FP via submission confidence filtering, optimize per-camera."""
import sys, json, time, tempfile, shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

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
    f"stage4.association.secondary_embeddings.path": SEC_EMB_PATH,
    "stage4.association.secondary_embeddings.weight": "0.25",
    "stage5.stationary_filter.enabled": "true",
    "stage5.stationary_filter.min_displacement_px": "150",
}

EXPERIMENTS = [
    # Baseline
    {"name": "base", "overrides": {}},
    
    # Min submission confidence (removes interpolated + low-det frames)
    {"name": "min_conf_010", "overrides": {
        "stage5.min_submission_confidence": "0.10",
    }},
    {"name": "min_conf_020", "overrides": {
        "stage5.min_submission_confidence": "0.20",
    }},
    {"name": "min_conf_030", "overrides": {
        "stage5.min_submission_confidence": "0.30",
    }},
    {"name": "min_conf_040", "overrides": {
        "stage5.min_submission_confidence": "0.40",
    }},
    
    # Min trajectory confidence (removes low-confidence cross-cam matches)
    {"name": "traj_conf_010", "overrides": {
        "stage5.min_trajectory_confidence": "0.10",
    }},
    {"name": "traj_conf_030", "overrides": {
        "stage5.min_trajectory_confidence": "0.30",
    }},
    {"name": "traj_conf_050", "overrides": {
        "stage5.min_trajectory_confidence": "0.50",
    }},
    
    # Combine best so far: d150 + min_traj_frames + min_sub_conf
    {"name": "mf10_conf020", "overrides": {
        "stage5.min_trajectory_frames": "10",
        "stage5.min_submission_confidence": "0.20",
    }},
    
    # Higher cross_id_nms_iou (more aggressive NMS between IDs)
    {"name": "nms03", "overrides": {
        "stage5.cross_id_nms_iou": "0.3",
    }},
    {"name": "nms04", "overrides": {
        "stage5.cross_id_nms_iou": "0.4",
    }},
    
    # MTMC only (drop single-cam trajectories from submission)
    {"name": "mtmc_only", "overrides": {
        "stage5.mtmc_only_submission": "true",
    }},
    
    # Combine: d150 + mtmc_only + min_conf
    {"name": "mtmc_only_conf020", "overrides": {
        "stage5.mtmc_only_submission": "true",
        "stage5.min_submission_confidence": "0.20",
    }},
    
    # Combine: d150 + mf10 + mtmc_only
    {"name": "mf10_mtmc_only", "overrides": {
        "stage5.min_trajectory_frames": "10",
        "stage5.mtmc_only_submission": "true",
    }},
    
    # Gallery expansion variations (maybe more rounds helps)
    {"name": "gallery_3rounds", "overrides": {
        "stage4.association.gallery_expansion.max_rounds": "3",
    }},
    {"name": "gallery_5rounds", "overrides": {
        "stage4.association.gallery_expansion.max_rounds": "5",
    }},
    {"name": "gallery_thresh045", "overrides": {
        "stage4.association.gallery_expansion.threshold": "0.45",
    }},
    {"name": "gallery_orphan035", "overrides": {
        "stage4.association.gallery_expansion.orphan_match_threshold": "0.35",
    }},
    
    # Bigger max component size (allow larger clusters)
    {"name": "max_comp_15", "overrides": {
        "stage4.association.graph.max_component_size": "15",
    }},
    {"name": "max_comp_20", "overrides": {
        "stage4.association.graph.max_component_size": "20",
    }},
    
    # Spatio-temporal adjustments
    {"name": "st_max_time_200", "overrides": {
        "stage4.association.spatiotemporal.max_time_gap": "200",
    }},
    {"name": "st_max_time_600", "overrides": {
        "stage4.association.spatiotemporal.max_time_gap": "600",
    }},
    
    # Disable FAC (it was re-enabled in v43, maybe hurts with d150)
    {"name": "no_fac", "overrides": {
        "stage4.association.fac.enabled": "false",
    }},
    
    # Different FIC regularisation with d150
    {"name": "fic_010", "overrides": {
        "stage4.association.fic.regularisation": "0.10",
    }},
    {"name": "fic_020", "overrides": {
        "stage4.association.fic.regularisation": "0.20",
    }},
]


def run_experiment(name, extra_overrides):
    overrides = dict(BASE_OVERRIDES)
    overrides.update(extra_overrides)
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
    
    tmp_dir = Path(tempfile.mkdtemp())
    try:
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
        idsw = result.details.get("mtmc_id_switches", 0) if hasattr(result, "details") else 0
        
        return {"name": name, "idf1": round(idf1, 4), "mota": round(mota, 4), 
                "idsw": idsw, "time_s": round(elapsed, 1)}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    setup_logging(level="WARNING")
    results = []
    
    for i, exp in enumerate(EXPERIMENTS):
        name = exp["name"]
        print(f"\n[{i+1}/{len(EXPERIMENTS)}] {name}")
        try:
            result = run_experiment(name, exp["overrides"])
            results.append(result)
            print(f"  IDF1={result['idf1']:.4f}  MOTA={result['mota']:.4f}  "
                  f"IDsw={result['idsw']}  time={result['time_s']:.1f}s")
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback; traceback.print_exc()
            results.append({"name": name, "idf1": 0.0, "error": str(e)})
    
    print(f"\n{'='*70}")
    print("ROUND 9 SUMMARY (sorted by IDF1)")
    print(f"{'='*70}")
    results.sort(key=lambda r: r.get("idf1", 0), reverse=True)
    for r in results:
        err = f"  ERR: {r['error']}" if "error" in r else ""
        n = r["name"]
        i = r.get("idf1", 0)
        m = r.get("mota", 0)
        s = r.get("idsw", "?")
        print(f"  {n:30s} IDF1={i:.4f}  MOTA={m:.4f}  IDsw={s}{err}")
    
    out_path = "data/outputs/sweep_round9.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
