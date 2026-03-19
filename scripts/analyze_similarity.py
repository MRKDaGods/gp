"""Analyze similarity between fragmented pairs to understand threshold issues."""
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

# Load stage4 data
run_dir = Path("data/outputs/run_20260315_v2")

# Load embeddings
embeddings = np.load(run_dir / "stage2" / "embeddings.npy")
# Load metadata    
import sqlite3
db_path = run_dir / "stage3" / "metadata.db"
if not db_path.exists():
    print("No metadata.db found")
    exit(1)

conn = sqlite3.connect(str(db_path))
rows = conn.execute("SELECT index_id, track_id, camera_id FROM tracklets ORDER BY index_id").fetchall()
conn.close()

idx_to_track = {}
idx_to_camera = {}
track_to_idx = {}
for idx, track_id, cam_id in rows:
    idx_to_track[idx] = track_id
    idx_to_camera[idx] = cam_id
    track_to_idx[track_id] = idx

print(f"Loaded {len(rows)} tracklets, embeddings shape: {embeddings.shape}")

# Load global trajectories
trajs = json.load(open(run_dir / "stage4" / "global_trajectories.json"))
track_to_global = {}
for traj in trajs:
    gid = traj.get("global_id", traj.get("id"))
    for tk in traj.get("tracklets", []):
        track_to_global[tk.get("track_id")] = gid

# The fragmented GT cases with their pred IDs (from analysis above)
# Focus on multi-cluster cases where both tracklets ARE in trajectories
fragmented_pairs = {
    57: [73, 220, 269],
    190: [161, 332, 344],
    122: [176, 200, 339],
    34: [2, 228],
    54: [180, 181],
    35: [3, 5],
    12: [11, 253],
    15: [14, 254],
    21: [27, 259],
    25: [31, 185],
    29: [25, 64],
    28: [33, 233],
    20: [24, 188],
    47: [37, 188],
    50: [42, 261],
    56: [45, 266],
    61: [49, 271],
    67: [54, 274],
    53: [62, 186],
    86: [221, 283],
    85: [68, 241],
    71: [59, 182],
}

print(f"\n=== SIMILARITY BETWEEN FRAGMENTED PAIRS ===")
print(f"{'GT ID':>6} {'PredA':>6} {'PredB':>6} {'CamA':>10} {'CamB':>10} {'Cosine':>8} {'GlobA':>6} {'GlobB':>6}")
print("-" * 80)

sims = []
for gt_id, pred_ids in sorted(fragmented_pairs.items()):
    # Compute cosine similarity between all pairs
    for i in range(len(pred_ids)):
        for j in range(i + 1, len(pred_ids)):
            a, b = pred_ids[i], pred_ids[j]
            # Find tracklet indices
            idx_a = track_to_idx.get(a)
            idx_b = track_to_idx.get(b)
            if idx_a is None or idx_b is None:
                continue
            
            cam_a = idx_to_camera.get(idx_a, "?")
            cam_b = idx_to_camera.get(idx_b, "?")
            glob_a = track_to_global.get(a, "?")
            glob_b = track_to_global.get(b, "?")
            
            # Cosine similarity (embeddings should be L2-normalized already)
            emb_a = embeddings[idx_a]
            emb_b = embeddings[idx_b]
            cos_sim = float(np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b) + 1e-8))
            
            sims.append(cos_sim)
            same_cam = "SAME" if cam_a == cam_b else ""
            print(f"{gt_id:>6} {a:>6} {b:>6} {cam_a:>10} {cam_b:>10} {cos_sim:>8.3f} {glob_a!s:>6} {glob_b!s:>6} {same_cam}")

if sims:
    sims_arr = np.array(sims)
    print(f"\n=== SUMMARY ===")
    print(f"Similarity stats for fragmented pairs:")
    print(f"  Count: {len(sims)}")
    print(f"  Min:    {sims_arr.min():.3f}")
    print(f"  Median: {np.median(sims_arr):.3f}")
    print(f"  Mean:   {sims_arr.mean():.3f}")
    print(f"  Max:    {sims_arr.max():.3f}")
    print(f"  Above 0.60: {(sims_arr >= 0.60).sum()}/{len(sims)}")
    print(f"  Above 0.55: {(sims_arr >= 0.55).sum()}/{len(sims)}")
    print(f"  Above 0.50: {(sims_arr >= 0.50).sum()}/{len(sims)}")
    print(f"  Above 0.45: {(sims_arr >= 0.45).sum()}/{len(sims)}")
    print(f"  Above 0.40: {(sims_arr >= 0.40).sum()}/{len(sims)}")
    print(f"  Below 0.30: {(sims_arr < 0.30).sum()}/{len(sims)}")
