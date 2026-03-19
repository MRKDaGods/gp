"""Round 8 sweep: Test ALL disabled stage4 features with stationary filter d150.
Each tested individually to measure marginal contribution."""
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

# Best known config (v43 + stationary filter d150)
BASE_OVERRIDES = {
    # Stage 4 - v43 optimized
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
    # Stage 5 - stationary filter d150
    "stage5.stationary_filter.enabled": "true",
    "stage5.stationary_filter.min_displacement_px": "150",
}

EXPERIMENTS = [
    # Baseline with d150 stationary filter
    {"name": "baseline_d150", "overrides": {}},
    
    # K-reciprocal re-ranking (was disabled for 512D, retesting with 280D)
    {"name": "rerank_k30k10_l04", "overrides": {
        "stage4.association.reranking.enabled": "true",
        "stage4.association.reranking.k1": "30",
        "stage4.association.reranking.k2": "10",
        "stage4.association.reranking.lambda_value": "0.4",
    }},
    {"name": "rerank_k20k6_l05", "overrides": {
        "stage4.association.reranking.enabled": "true",
        "stage4.association.reranking.k1": "20",
        "stage4.association.reranking.k2": "6",
        "stage4.association.reranking.lambda_value": "0.5",
    }},
    {"name": "rerank_k15k5_l06", "overrides": {
        "stage4.association.reranking.enabled": "true",
        "stage4.association.reranking.k1": "15",
        "stage4.association.reranking.k2": "5",
        "stage4.association.reranking.lambda_value": "0.6",
    }},
    
    # Hierarchical centroid expansion
    {"name": "hierarch_default", "overrides": {
        "stage4.association.hierarchical.enabled": "true",
    }},
    {"name": "hierarch_low", "overrides": {
        "stage4.association.hierarchical.enabled": "true",
        "stage4.association.hierarchical.centroid_threshold": "0.30",
        "stage4.association.hierarchical.merge_threshold": "0.30",
        "stage4.association.hierarchical.orphan_threshold": "0.25",
    }},
    {"name": "hierarch_high", "overrides": {
        "stage4.association.hierarchical.enabled": "true",
        "stage4.association.hierarchical.centroid_threshold": "0.40",
        "stage4.association.hierarchical.merge_threshold": "0.40",
        "stage4.association.hierarchical.orphan_threshold": "0.35",
    }},
    
    # Camera distance bias
    {"name": "cam_bias", "overrides": {
        "stage4.association.camera_bias.enabled": "true",
        "stage4.association.camera_bias.iterations": "3",
    }},
    
    # Zone model
    {"name": "zone_model", "overrides": {
        "stage4.association.zone_model.enabled": "true",
        "stage4.association.zone_model.bonus": "0.03",
        "stage4.association.zone_model.penalty": "0.03",
    }},
    {"name": "zone_model_strong", "overrides": {
        "stage4.association.zone_model.enabled": "true",
        "stage4.association.zone_model.bonus": "0.05",
        "stage4.association.zone_model.penalty": "0.05",
    }},
    
    # Camera pair boost
    {"name": "cam_pair_boost", "overrides": {
        "stage4.association.camera_pair_boost.enabled": "true",
    }},
    
    # CSLS hubness reduction
    {"name": "csls_k10", "overrides": {
        "stage4.association.csls.enabled": "true",
        "stage4.association.csls.k": "10",
    }},
    {"name": "csls_k5", "overrides": {
        "stage4.association.csls.enabled": "true",
        "stage4.association.csls.k": "5",
    }},
    
    # Cluster verification
    {"name": "cluster_verify_030", "overrides": {
        "stage4.association.cluster_verify.enabled": "true",
        "stage4.association.cluster_verify.min_connectivity": "0.30",
    }},
    {"name": "cluster_verify_020", "overrides": {
        "stage4.association.cluster_verify.enabled": "true",
        "stage4.association.cluster_verify.min_connectivity": "0.20",
    }},
    
    # Disable gallery expansion (maybe it hurts?)
    {"name": "no_gallery_exp", "overrides": {
        "stage4.association.gallery_expansion.enabled": "false",
    }},
    
    # Higher graph threshold (more conservative)
    {"name": "thresh_055", "overrides": {
        "stage4.association.graph.similarity_threshold": "0.55",
    }},
    
    # Lower graph threshold (more aggressive)
    {"name": "thresh_050", "overrides": {
        "stage4.association.graph.similarity_threshold": "0.50",
    }},
    
    # Temporal overlap bonus increase
    {"name": "temp_bonus_010", "overrides": {
        "stage4.association.temporal_overlap.bonus": "0.10",
    }},
    
    # Length weight variations
    {"name": "len_weight_05", "overrides": {
        "stage4.association.length_weight_power": "0.5",
    }},
    {"name": "len_weight_00", "overrides": {
        "stage4.association.length_weight_power": "0.0",
    }},
    
    # Bridge pruning
    {"name": "bridge_prune_005", "overrides": {
        "stage4.association.graph.bridge_prune_margin": "0.05",
    }},
    
    # Secondary embedding weight variations
    {"name": "sec_weight_030", "overrides": {
        "stage4.association.secondary_embeddings.weight": "0.30",
    }},
    {"name": "sec_weight_015", "overrides": {
        "stage4.association.secondary_embeddings.weight": "0.15",
    }},
    {"name": "sec_weight_040", "overrides": {
        "stage4.association.secondary_embeddings.weight": "0.40",
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
        
        return {"name": name, "idf1": round(idf1, 4), "mota": round(mota, 4), "time_s": round(elapsed, 1)}
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
            print(f"  IDF1={result['idf1']:.4f}  MOTA={result['mota']:.4f}  time={result['time_s']:.1f}s")
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback; traceback.print_exc()
            results.append({"name": name, "idf1": 0.0, "error": str(e)})
    
    print(f"\n{'='*60}")
    print("ROUND 8 SUMMARY (sorted by IDF1)")
    print(f"{'='*60}")
    results.sort(key=lambda r: r.get("idf1", 0), reverse=True)
    for r in results:
        err = f"  ERROR: {r['error']}" if "error" in r else ""
        print(f"  {r['name']:30s} IDF1={r.get('idf1',0):.4f}  MOTA={r.get('mota',0):.4f}{err}")
    
    out_path = "data/outputs/sweep_round8.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
