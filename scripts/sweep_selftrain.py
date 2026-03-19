"""Self-training embedding refinement: cluster → centroid → refine → recluster.

After initial clustering, adjusts embeddings toward their cluster centroids,
then re-runs the full pipeline. This is a form of unsupervised domain adaptation
that sharpens cluster boundaries without external labels.
"""
import sys, json, time, tempfile, shutil, copy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.core.config import load_config
from src.core.io_utils import load_embeddings, load_hsv_features, load_tracklets_by_camera
from src.core.data_models import TrackletFeatures
from src.stage3_indexing import run_stage3
from src.stage3_indexing.faiss_index import FAISSIndex
from src.stage3_indexing.metadata_store import MetadataStore
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


def refine_embeddings(embeddings, trajectories, features, alpha=0.2, cross_cam_only=True):
    """Move each embedding toward its cluster centroid.
    
    Args:
        embeddings: (N, D) embedding matrix
        trajectories: List of GlobalTrajectory from pipeline
        features: List[TrackletFeatures] 
        alpha: interpolation weight (0=no change, 1=replace with centroid)
        cross_cam_only: only refine members of cross-camera clusters
    
    Returns:
        refined embeddings (N, D), L2-normalized
    """
    # Build track_id → index mapping
    tid_to_idx = {}
    for i, f in enumerate(features):
        tid_to_idx[(f.camera_id, f.track_id)] = i
    
    refined = embeddings.copy()
    n_refined = 0
    
    for traj in trajectories:
        # Get member indices
        member_indices = []
        for t in traj.tracklets:
            idx = tid_to_idx.get((t.camera_id, t.track_id))
            if idx is not None:
                member_indices.append(idx)
        
        if len(member_indices) < 2:
            continue
        
        if cross_cam_only:
            cams = set(features[i].camera_id for i in member_indices)
            if len(cams) < 2:
                continue
        
        # Compute centroid
        centroid = embeddings[member_indices].mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
        
        # Move each member toward centroid
        for idx in member_indices:
            refined[idx] = (1 - alpha) * embeddings[idx] + alpha * centroid
            n_refined += 1
    
    # L2-normalize
    norms = np.linalg.norm(refined, axis=1, keepdims=True)
    refined = refined / np.maximum(norms, 1e-8)
    
    return refined, n_refined


def run_with_embeddings(cfg, embeddings, hsv_features, features, tracklets_by_camera, 
                        tmp_dir, index_map):
    """Run stages 3-5 with given embeddings."""
    # Reconstruct features with potentially refined embeddings
    refined_features = [
        TrackletFeatures(
            track_id=m["track_id"], camera_id=m["camera_id"],
            class_id=m["class_id"], embedding=embeddings[i],
            hsv_histogram=hsv_features[i],
        )
        for i, m in enumerate(index_map)
    ]
    
    faiss_index, metadata_store = run_stage3(
        cfg, refined_features, tracklets_by_camera, output_dir=tmp_dir / "stage3"
    )
    
    trajectories = run_stage4(
        cfg, faiss_index, metadata_store, refined_features,
        tracklets_by_camera, output_dir=tmp_dir / "stage4"
    )
    result = run_stage5(cfg, trajectories, output_dir=tmp_dir / "stage5")
    
    idf1 = getattr(result, "mtmc_idf1", 0.0) or 0.0
    mota = result.details.get("mtmc_mota", 0.0) if hasattr(result, "details") else 0.0
    idsw = result.details.get("mtmc_id_switches", 0) if hasattr(result, "details") else 0
    
    return trajectories, idf1, mota, idsw, refined_features


def main():
    setup_logging(level="WARNING")
    
    override_list = [f"{k}={v}" for k, v in BASE_OVERRIDES.items()]
    cfg = load_config(CONFIG_PATH, overrides=override_list)
    
    # Load data
    embeddings_orig, index_map = load_embeddings(str(RUN_DIR / "stage2"))
    hsv_matrix = load_hsv_features(str(RUN_DIR / "stage2"))
    features = [
        TrackletFeatures(
            track_id=m["track_id"], camera_id=m["camera_id"],
            class_id=m["class_id"], embedding=embeddings_orig[i],
            hsv_histogram=hsv_matrix[i],
        )
        for i, m in enumerate(index_map)
    ]
    tracklets_by_camera = load_tracklets_by_camera(str(RUN_DIR / "stage1"))
    
    # Test different alpha values and iteration counts
    configs = [
        # (alpha, iterations, cross_cam_only)
        (0.1, 1, True),
        (0.2, 1, True),
        (0.3, 1, True),
        (0.1, 2, True),
        (0.2, 2, True),
        (0.3, 2, True),
        (0.1, 3, True),
        (0.2, 3, True),
        (0.1, 1, False),
        (0.2, 1, False),
    ]
    
    results = []
    
    for ci, (alpha, n_iters, cross_only) in enumerate(configs):
        name = f"a{int(alpha*100):02d}_i{n_iters}_{'cc' if cross_only else 'all'}"
        print(f"\n[{ci+1}/{len(configs)}] {name}")
        
        try:
            embeddings = embeddings_orig.copy()
            
            for iteration in range(n_iters):
                tmp_dir = Path(tempfile.mkdtemp())
                try:
                    trajectories, idf1, mota, idsw, feats = run_with_embeddings(
                        cfg, embeddings, hsv_matrix, features, 
                        tracklets_by_camera, tmp_dir, index_map
                    )
                    
                    if iteration < n_iters - 1:
                        # Refine embeddings for next iteration
                        embeddings, n_ref = refine_embeddings(
                            embeddings, trajectories, features, 
                            alpha=alpha, cross_cam_only=cross_only
                        )
                        print(f"  iter {iteration+1}: IDF1={idf1:.4f}  refined={n_ref} embs")
                    else:
                        print(f"  final:  IDF1={idf1:.4f}  MOTA={mota:.4f}  IDsw={idsw}")
                finally:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
            
            results.append({
                "name": name, "alpha": alpha, "iterations": n_iters,
                "cross_cam_only": cross_only,
                "idf1": round(idf1, 4), "mota": round(mota, 4), "idsw": idsw,
            })
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback; traceback.print_exc()
            results.append({"name": name, "idf1": 0.0, "error": str(e)})
    
    print(f"\n{'='*70}")
    print("SELF-TRAINING RESULTS (sorted by IDF1)")
    print(f"{'='*70}")
    results.sort(key=lambda r: r.get("idf1", 0), reverse=True)
    for r in results:
        n = r["name"]
        i = r.get("idf1", 0)
        m = r.get("mota", 0)
        s = r.get("idsw", "?")
        print(f"  {n:30s} IDF1={i:.4f}  MOTA={m:.4f}  IDsw={s}")
    
    with open("data/outputs/sweep_selftrain.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to data/outputs/sweep_selftrain.json")


if __name__ == "__main__":
    main()
