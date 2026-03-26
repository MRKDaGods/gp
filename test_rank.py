import json, numpy as np
from pathlib import Path

run_dir = Path("c:/Users/seift/Downloads/gp/outputs/dataset_precompute_s01")

emb = np.load(run_dir / "stage2" / "embeddings.npy")
with open(run_dir / "stage2" / "embedding_index.json") as f:
    emb_idx = json.load(f)

probe_cam = "c001"
probe_tid = 20

# find probe features
probe_indices = [i for i, x in enumerate(emb_idx) if x["camera_id"] == probe_cam and x["track_id"] == probe_tid]
if not probe_indices:
    print("probe not found")
else:
    probe_feats = emb[probe_indices]
    probe_center = probe_feats.mean(axis=0)
    probe_center = probe_center / np.linalg.norm(probe_center)
    
    with open(run_dir / "stage4" / "global_trajectories.json") as f:
        trajs = json.load(f)
        
    scores = []
    for t in trajs:
        # compute similarity
        t_indices = []
        for tr in t["tracklets"]:
            c, tid = tr["camera_id"], tr["track_id"]
            # find in emb_idx
            idx = [i for i, x in enumerate(emb_idx) if x["camera_id"] == c and x["track_id"] == tid]
            t_indices.extend(idx)
        if not t_indices:
            continue
            
        t_feats = emb[t_indices]
        t_center = t_feats.mean(axis=0)
        t_center = t_center / np.linalg.norm(t_center)
        
        sim = float(np.dot(probe_center, t_center))
        scores.append((t["global_id"], sim))
        
    scores.sort(key=lambda x: x[1], reverse=True)
    print("top 5: ", scores[:5])
