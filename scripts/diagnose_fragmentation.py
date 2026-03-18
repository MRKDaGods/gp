#!/usr/bin/env python
"""Diagnose fragmentation: trace where true cross-camera matches are lost.

Loads GT, maps tracklets to GT IDs, then checks:
1. Raw cosine similarity between true match pairs
2. Whether they survive combined_sim computation
3. Distribution of combined_sim for true vs false pairs
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from collections import defaultdict
from omegaconf import OmegaConf

def load_gt(gt_dir: Path, cameras: list):
    """Load GT annotations, return dict: (camera_id, frame_id) -> list of (gt_id, bbox)."""
    gt_data = {}  # cam_id -> dict[frame_id -> list of (gt_id, bbox)]
    for cam in cameras:
        gt_file = gt_dir / cam / "gt.txt"
        if not gt_file.exists():
            continue
        cam_gt = defaultdict(list)
        with open(gt_file) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 6:
                    continue
                if len(parts) >= 7 and float(parts[6]) == 0:
                    continue
                frame_id = int(parts[0])
                gt_id = int(parts[1])
                x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                cam_gt[frame_id].append((gt_id, [x, y, w, h]))
        gt_data[cam] = cam_gt
    return gt_data

def map_tracklets_to_gt(features, gt_data, tracklets_by_camera):
    """Map each tracklet to its most-overlapping GT ID using IoU matching."""
    from src.core.data_models import Tracklet
    
    tracklet_lookup = {}
    for cam_id, tracklets in tracklets_by_camera.items():
        for t in tracklets:
            tracklet_lookup[(t.camera_id, t.track_id)] = t
    
    tracklet_gt_map = {}  # tracklet_index -> gt_id
    
    for idx, feat in enumerate(features):
        t = tracklet_lookup.get((feat.camera_id, feat.track_id))
        if not t or not t.frames:
            continue
        cam = feat.camera_id
        if cam not in gt_data:
            continue
        
        # Vote across frames
        gt_votes = defaultdict(float)
        for frame_det in t.frames:
            frame_id = frame_det.frame_id
            if frame_id not in gt_data[cam]:
                continue
            # Find best IoU match
            det_bbox = frame_det.bbox  # (x1, y1, x2, y2)
            det_x, det_y = det_bbox[0], det_bbox[1]
            det_w, det_h = det_bbox[2] - det_bbox[0], det_bbox[3] - det_bbox[1]
            
            best_iou = 0
            best_gt = None
            for gt_id, gt_bbox in gt_data[cam][frame_id]:
                gx, gy, gw, gh = gt_bbox
                # IoU
                x1 = max(det_x, gx)
                y1 = max(det_y, gy)
                x2 = min(det_x + det_w, gx + gw)
                y2 = min(det_y + det_h, gy + gh)
                inter = max(0, x2 - x1) * max(0, y2 - y1)
                union = det_w * det_h + gw * gh - inter
                iou = inter / max(union, 1e-6)
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt_id
            if best_gt is not None and best_iou >= 0.3:
                gt_votes[best_gt] += best_iou
        
        if gt_votes:
            tracklet_gt_map[idx] = max(gt_votes, key=gt_votes.get)
    
    return tracklet_gt_map


def main():
    import pickle
    from src.core.io_utils import load_embeddings, load_hsv_features
    
    run_dir = Path("data/outputs/run_20260315_v2")
    gt_dir = Path("data/raw/cityflowv2")
    
    # Load config with PCA 280D overrides
    cfg = OmegaConf.load("configs/default.yaml")
    # Override for PCA 280D best config
    cfg.stage4.association.fic.regularisation = 1.0
    cfg.stage4.association.graph.similarity_threshold = 0.60
    cfg.stage4.association.fac.enabled = False
    cfg.stage4.association.camera_bias = {"enabled": False}
    cfg.stage4.association.camera_pair_boost = {"enabled": False}
    cfg.stage4.association.zone_model = {"enabled": False}
    cfg.stage4.association.gallery_expansion.threshold = 0.50
    cfg.stage4.association.gallery_expansion.orphan_match_threshold = 0.40
    
    # Load stage 1 tracklets
    from src.core.io_utils import load_tracklets_by_camera
    tracklets_by_camera = load_tracklets_by_camera(run_dir / "stage1")
    cameras = sorted(tracklets_by_camera.keys())
    print(f"Cameras: {cameras}")
    
    # Load stage 2 features
    embeddings, index_map = load_embeddings(run_dir / "stage2")
    hsv_features = load_hsv_features(run_dir / "stage2")
    
    # Build features list
    from src.core.data_models import TrackletFeatures
    features = []
    for i, meta in enumerate(index_map):
        feat = TrackletFeatures(
            camera_id=meta["camera_id"],
            track_id=meta["track_id"],
            class_id=meta.get("class_id", 0),
            embedding=embeddings[i],
            hsv_histogram=hsv_features[i],
        )
        features.append(feat)
    
    n = len(features)
    camera_ids = [f.camera_id for f in features]
    class_ids = [f.class_id for f in features]
    print(f"Total tracklets: {n}")
    
    # Load GT
    gt_data = load_gt(gt_dir, cameras)
    
    # Map tracklets to GT IDs
    tracklet_gt_map = map_tracklets_to_gt(features, gt_data, tracklets_by_camera)
    print(f"Tracklets mapped to GT: {len(tracklet_gt_map)}/{n}")
    
    # Find GT IDs that appear in multiple cameras
    gt_to_tracklets = defaultdict(list)  # gt_id -> list of (tracklet_idx, camera_id)
    for idx, gt_id in tracklet_gt_map.items():
        gt_to_tracklets[gt_id].append((idx, camera_ids[idx]))
    
    multi_cam_gt = {
        gt_id: tracklets
        for gt_id, tracklets in gt_to_tracklets.items()
        if len(set(cam for _, cam in tracklets)) > 1
    }
    print(f"Multi-camera GT IDs: {len(multi_cam_gt)}")
    
    # Compute raw cosine similarities for all true cross-camera pairs
    print("\n=== TRUE CROSS-CAMERA PAIR ANALYSIS ===")
    
    true_pairs = []  # (idx_i, idx_j, camera_i, camera_j, cosine_sim, gt_id)
    for gt_id, tracklets in multi_cam_gt.items():
        for ii in range(len(tracklets)):
            for jj in range(ii + 1, len(tracklets)):
                idx_i, cam_i = tracklets[ii]
                idx_j, cam_j = tracklets[jj]
                if cam_i == cam_j:
                    continue
                cos_sim = float(embeddings[idx_i] @ embeddings[idx_j])
                true_pairs.append((idx_i, idx_j, cam_i, cam_j, cos_sim, gt_id))
    
    print(f"Total true cross-camera pairs: {len(true_pairs)}")
    
    # Distribution by camera pair
    pair_sims = defaultdict(list)
    for idx_i, idx_j, cam_i, cam_j, cos_sim, gt_id in true_pairs:
        key = tuple(sorted([cam_i, cam_j]))
        pair_sims[key].append(cos_sim)
    
    print("\nPer-camera-pair true match similarity distribution:")
    print(f"{'Camera Pair':<25} {'Count':>5} {'Min':>6} {'P25':>6} {'Median':>6} {'P75':>6} {'Max':>6}")
    for key in sorted(pair_sims.keys()):
        sims = sorted(pair_sims[key])
        n_pairs = len(sims)
        p25 = sims[max(0, n_pairs//4)]
        med = sims[n_pairs//2]
        p75 = sims[min(n_pairs-1, 3*n_pairs//4)]
        print(f"{key[0]}-{key[1]:<15} {n_pairs:>5} {min(sims):>6.3f} {p25:>6.3f} {med:>6.3f} {p75:>6.3f} {max(sims):>6.3f}")
    
    # How many true pairs are below various thresholds?
    print("\nTrue pairs BELOW threshold (would be fragmented if not boosted):")
    for thresh in [0.30, 0.35, 0.40, 0.45, 0.47, 0.50]:
        below = sum(1 for _, _, _, _, s, _ in true_pairs if s < thresh)
        print(f"  < {thresh:.2f}: {below}/{len(true_pairs)} ({100*below/max(len(true_pairs),1):.1f}%)")
    
    # Now run the FULL similarity pipeline to get combined_sim and check true pair scores
    print("\n=== RUNNING SIMILARITY PIPELINE ===")
    
    from src.stage4_association.fic import per_camera_whiten
    from src.stage4_association.query_expansion import average_query_expansion_batched
    from src.stage3_indexing.faiss_index import FAISSIndex
    from src.stage4_association.similarity import (
        compute_combined_similarity,
        mutual_nearest_neighbor_filter,
    )
    from src.stage4_association.spatial_temporal import SpatioTemporalValidator
    from src.core.io_utils import load_tracklets
    
    stage_cfg = cfg.stage4.association
    
    # Step 0: FIC
    proc_emb = embeddings.copy()
    fic_cfg = stage_cfg.get("fic", {})
    if fic_cfg.get("enabled", False):
        proc_emb = per_camera_whiten(proc_emb, camera_ids,
            regularisation=float(fic_cfg.get("regularisation", 3.0)))
        print("Applied FIC")
    
    # Step 1: FAISS
    faiss_idx = FAISSIndex(index_type="flat_ip")
    faiss_idx.build(proc_emb.astype(np.float32))
    top_k = stage_cfg.top_k
    distances, indices = faiss_idx.search(proc_emb, top_k)
    
    # Step 1b: QE + DBA
    qe_cfg = stage_cfg.get("query_expansion", {})
    if qe_cfg.get("enabled", True):
        qe_k = qe_cfg.get("k", 5)
        qe_alpha = qe_cfg.get("alpha", 5.0)
        proc_emb = average_query_expansion_batched(proc_emb, indices, k=qe_k, alpha=qe_alpha)
        if qe_cfg.get("dba", True):
            dba_index = FAISSIndex(index_type="flat_ip")
            dba_index.build(proc_emb.astype(np.float32))
            distances, indices = dba_index.search(proc_emb, top_k)
            faiss_idx = dba_index
        else:
            distances, indices = faiss_idx.search(proc_emb, top_k)
        print("Applied QE+DBA")
    
    # True pair cosine after FIC+QE
    print("\nTrue pair cosine AFTER FIC+QE+DBA:")
    post_sims = defaultdict(list)
    for idx_i, idx_j, cam_i, cam_j, _, gt_id in true_pairs:
        cos_sim = float(proc_emb[idx_i] @ proc_emb[idx_j])
        post_sims[tuple(sorted([cam_i, cam_j]))].append(cos_sim)
    
    print(f"{'Camera Pair':<25} {'Count':>5} {'Min':>6} {'P25':>6} {'Median':>6} {'P75':>6} {'Max':>6}")
    for key in sorted(post_sims.keys()):
        sims = sorted(post_sims[key])
        n_p = len(sims)
        p25 = sims[max(0, n_p//4)]
        med = sims[n_p//2]
        p75 = sims[min(n_p-1, 3*n_p//4)]
        print(f"{key[0]}-{key[1]:<15} {n_p:>5} {min(sims):>6.3f} {p25:>6.3f} {med:>6.3f} {p75:>6.3f} {max(sims):>6.3f}")
    
    print("\nTrue pairs BELOW threshold after FIC+QE:")
    all_post = []
    for _, sims in post_sims.items():
        all_post.extend(sims)
    for thresh in [0.30, 0.35, 0.40, 0.45, 0.47, 0.50]:
        below = sum(1 for s in all_post if s < thresh)
        print(f"  < {thresh:.2f}: {below}/{len(all_post)} ({100*below/max(len(all_post),1):.1f}%)")
    
    # Step 2: Exhaustive cross-camera pairs
    from src.stage4_association.pipeline import _build_all_cross_camera_pairs, _extract_scene
    candidate_pairs = _build_all_cross_camera_pairs(
        n, proc_emb, camera_ids, class_ids,
        min_similarity=float(stage_cfg.get("exhaustive_min_similarity", 0.0)),
    )
    print(f"\nExhaustive candidate pairs: {len(candidate_pairs)}")
    
    # Check how many true pairs survive exhaustive
    cand_set = {(i, j) for i, j, _ in candidate_pairs}
    true_in_cand = sum(1 for i, j, _, _, _, _ in true_pairs
                       if (i, j) in cand_set or (j, i) in cand_set)
    print(f"True pairs in candidates: {true_in_cand}/{len(true_pairs)}")
    
    # Step 2a: Hard temporal
    from src.stage3_indexing.metadata_store import MetadataStore
    meta_store = MetadataStore(run_dir / "stage3" / "metadata.db")
    start_times, end_times, num_frames_list = [], [], []
    for i in range(n):
        meta = meta_store.get_tracklet(i)
        if meta:
            start_times.append(meta["start_time"])
            end_times.append(meta["end_time"])
            num_frames_list.append(meta.get("num_frames", 1))
        else:
            start_times.append(0.0)
            end_times.append(0.0)
            num_frames_list.append(1)
    
    candidate_pairs = [
        (i, j, sim) for i, j, sim in candidate_pairs
        if not (camera_ids[i] == camera_ids[j]
                and start_times[i] <= end_times[j]
                and start_times[j] <= end_times[i])
    ]
    print(f"After hard temporal: {len(candidate_pairs)}")
    
    # Step 2b: Mutual NN
    mutual_nn_cfg = stage_cfg.get("mutual_nn", {})
    if mutual_nn_cfg.get("enabled", True):
        pre = len(candidate_pairs)
        candidate_pairs = mutual_nearest_neighbor_filter(
            candidate_pairs,
            top_k_per_query=mutual_nn_cfg.get("top_k_per_query", 10),
        )
        print(f"After mutual-NN (top_k={mutual_nn_cfg.get('top_k_per_query', 10)}): {len(candidate_pairs)} ({pre - len(candidate_pairs)} pruned)")
    
    # Check true pairs after mutual-NN
    cand_set2 = {(i, j) for i, j, _ in candidate_pairs}
    true_in_mnn = sum(1 for i, j, _, _, _, _ in true_pairs
                      if (i, j) in cand_set2 or (j, i) in cand_set2)
    true_lost_mnn = true_in_cand - true_in_mnn
    print(f"True pairs after mutual-NN: {true_in_mnn}/{len(true_pairs)} (lost {true_lost_mnn} in mutual-NN)")
    
    # Appearance sim (no reranking)
    appearance_sim = {(i, j): sim for i, j, sim in candidate_pairs}
    
    # Step 4+5: Combined similarity
    st_validator = SpatioTemporalValidator(
        min_time_gap=stage_cfg.spatiotemporal.min_time_gap,
        max_time_gap=stage_cfg.spatiotemporal.max_time_gap,
        camera_transitions=stage_cfg.spatiotemporal.get("camera_transitions"),
    )
    
    combined_sim = compute_combined_similarity(
        appearance_sim=appearance_sim,
        hsv_features=hsv_features,
        start_times=start_times,
        end_times=end_times,
        camera_ids=camera_ids,
        st_validator=st_validator,
        weights=stage_cfg.weights,
        class_ids=class_ids,
        num_frames=num_frames_list,
        temporal_overlap_cfg=stage_cfg.get("temporal_overlap"),
    )
    print(f"\nCombined sim pairs: {len(combined_sim)}")
    
    # Check true pairs in combined_sim
    true_combined = {}
    for idx_i, idx_j, cam_i, cam_j, _, gt_id in true_pairs:
        cs = combined_sim.get((idx_i, idx_j)) or combined_sim.get((idx_j, idx_i))
        if cs is not None:
            true_combined[(idx_i, idx_j)] = (cs, gt_id, cam_i, cam_j)
    
    true_lost_st = true_in_mnn - len(true_combined)
    print(f"True pairs in combined_sim: {len(true_combined)}/{len(true_pairs)} (lost {true_lost_st} in ST validation)")
    
    # Distribution of combined_sim for true pairs
    print("\nCombined sim distribution for true pairs:")
    true_cs_values = [cs for cs, _, _, _ in true_combined.values()]
    if true_cs_values:
        true_cs_values.sort()
        n_tc = len(true_cs_values)
        print(f"  Min={true_cs_values[0]:.3f}, P25={true_cs_values[n_tc//4]:.3f}, "
              f"Median={true_cs_values[n_tc//2]:.3f}, P75={true_cs_values[3*n_tc//4]:.3f}, "
              f"Max={true_cs_values[-1]:.3f}")
    
    # True pairs below graph threshold
    graph_threshold = float(stage_cfg.graph.similarity_threshold)
    below_thresh = [(cs, gt_id, cam_i, cam_j) for cs, gt_id, cam_i, cam_j in true_combined.values()
                    if cs < graph_threshold]
    above_thresh = [(cs, gt_id, cam_i, cam_j) for cs, gt_id, cam_i, cam_j in true_combined.values()
                    if cs >= graph_threshold]
    
    print(f"\nTrue pairs above threshold {graph_threshold}: {len(above_thresh)}/{len(true_combined)}")
    print(f"True pairs BELOW threshold {graph_threshold}: {len(below_thresh)}/{len(true_combined)}")
    
    # Camera pair breakdown of below-threshold true pairs
    below_by_pair = defaultdict(list)
    for cs, gt_id, cam_i, cam_j in below_thresh:
        key = tuple(sorted([cam_i, cam_j]))
        below_by_pair[key].append((cs, gt_id))
    
    print("\nBelow-threshold true pairs by camera pair:")
    for key in sorted(below_by_pair.keys()):
        pairs = below_by_pair[key]
        sims = [cs for cs, _ in pairs]
        print(f"  {key[0]}-{key[1]}: {len(pairs)} pairs, sim range [{min(sims):.3f}, {max(sims):.3f}]")
    
    # FALSE positive analysis: cross-camera pairs NOT matching GT that are above threshold
    false_pairs_above = []
    for (i, j), cs in combined_sim.items():
        if cs < graph_threshold:
            continue
        gt_i = tracklet_gt_map.get(i)
        gt_j = tracklet_gt_map.get(j)
        if gt_i is not None and gt_j is not None and gt_i != gt_j:
            false_pairs_above.append((cs, i, j, gt_i, gt_j, camera_ids[i], camera_ids[j]))
    
    print(f"\nFalse positive pairs above threshold: {len(false_pairs_above)}")
    fp_by_pair = defaultdict(int)
    for cs, i, j, gt_i, gt_j, cam_i, cam_j in false_pairs_above:
        key = tuple(sorted([cam_i, cam_j]))
        fp_by_pair[key] += 1
    for key in sorted(fp_by_pair.keys()):
        print(f"  {key[0]}-{key[1]}: {fp_by_pair[key]} false positives")
    
    # Summary
    total_true = len(true_pairs)
    lost_in_scene_block = total_true - true_in_cand
    lost_in_mnn = true_in_cand - true_in_mnn
    lost_in_st = true_in_mnn - len(true_combined)
    lost_below_thresh = len(below_thresh)
    matched = len(above_thresh)
    
    print(f"\n=== FRAGMENTATION ROOT CAUSE SUMMARY ===")
    print(f"Total true cross-camera pairs: {total_true}")
    print(f"  Lost in scene blocking:     {lost_in_scene_block} ({100*lost_in_scene_block/max(total_true,1):.1f}%)")
    print(f"  Lost in mutual-NN filter:   {lost_in_mnn} ({100*lost_in_mnn/max(total_true,1):.1f}%)")
    print(f"  Lost in ST validation:      {lost_in_st} ({100*lost_in_st/max(total_true,1):.1f}%)")
    print(f"  Lost below threshold {graph_threshold}: {lost_below_thresh} ({100*lost_below_thresh/max(total_true,1):.1f}%)")
    print(f"  MATCHED (above threshold):  {matched} ({100*matched/max(total_true,1):.1f}%)")
    
    # Optimal threshold analysis
    print(f"\n=== OPTIMAL THRESHOLD ANALYSIS ===")
    all_true_cs = true_cs_values
    all_false_cs = [cs for (i, j), cs in combined_sim.items()
                    if tracklet_gt_map.get(i) is not None
                    and tracklet_gt_map.get(j) is not None
                    and tracklet_gt_map.get(i) != tracklet_gt_map.get(j)]
    
    print(f"True positives in combined_sim: {len(all_true_cs)}")
    print(f"False positives in combined_sim: {len(all_false_cs)}")
    
    for thresh in [0.35, 0.40, 0.45, 0.48, 0.50, 0.52, 0.55, 0.58, 0.60, 0.62, 0.65, 0.70]:
        tp = sum(1 for s in all_true_cs if s >= thresh)
        fp = sum(1 for s in all_false_cs if s >= thresh)
        fn = len(all_true_cs) - tp
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-6)
        print(f"  threshold={thresh:.2f}: TP={tp:>3}, FP={fp:>2}, FN={fn:>3}, "
              f"P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")

    # Per-camera-pair TP/FP analysis at various thresholds
    print(f"\n=== PER-CAMERA-PAIR TP/FP BREAKDOWN ===")
    # Build per-pair TP and FP lists
    pair_tp = defaultdict(list)  # cam_pair -> list of combined_sim values for TPs
    pair_fp = defaultdict(list)  # cam_pair -> list of combined_sim values for FPs
    for (i, j), cs in combined_sim.items():
        gt_i = tracklet_gt_map.get(i)
        gt_j = tracklet_gt_map.get(j)
        if gt_i is None or gt_j is None:
            continue
        key = tuple(sorted([camera_ids[i], camera_ids[j]]))
        if gt_i == gt_j:
            pair_tp[key].append(cs)
        else:
            pair_fp[key].append(cs)
    
    all_pairs = sorted(set(list(pair_tp.keys()) + list(pair_fp.keys())))
    for thresh in [0.48, 0.50, 0.52, 0.55, 0.58, 0.60]:
        print(f"\n  At threshold={thresh:.2f}:")
        print(f"  {'Camera Pair':<25} {'TP':>4} {'FP':>4} {'FN':>4} {'Prec':>6} {'Recall':>6}")
        for key in all_pairs:
            tp_vals = pair_tp.get(key, [])
            fp_vals = pair_fp.get(key, [])
            tp = sum(1 for s in tp_vals if s >= thresh)
            fp = sum(1 for s in fp_vals if s >= thresh)
            fn = sum(1 for s in tp_vals if s < thresh)
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            print(f"  {key[0]}-{key[1]:<15} {tp:>4} {fp:>4} {fn:>4} {prec:>6.3f} {rec:>6.3f}")


if __name__ == "__main__":
    main()
