"""Analyze cross-camera similarity distribution."""
import sqlite3
import numpy as np

embs = np.load('data/outputs/run_20260315_v2/stage2/embeddings.npy')

# Load metadata from SQLite
conn = sqlite3.connect('data/outputs/run_20260315_v2/stage3/metadata.db')
conn.row_factory = sqlite3.Row
rows = conn.execute("SELECT * FROM tracklets ORDER BY rowid").fetchall()
camera_ids = [r['camera_id'] for r in rows]
conn.close()

# L2-normalize if not already
norms = np.linalg.norm(embs, axis=1, keepdims=True)
embs = embs / (norms + 1e-8)

n = len(embs)
print(f"Total tracklets: {n}")
print(f"Cameras: {set(camera_ids)}")

# Compute cross-camera cosine similarity matrix (only upper triangle)
cross_sims = []
for i in range(n):
    for j in range(i+1, n):
        if camera_ids[i] != camera_ids[j]:
            cos = float(np.dot(embs[i], embs[j]))
            if cos > 0.25:
                cross_sims.append(cos)

cross_sims.sort(reverse=True)
print(f"\nCross-cam pairs with sim>0.25: {len(cross_sims)}")
print(f"Top-30: {[f'{s:.3f}' for s in cross_sims[:30]]}")

for thresh in [0.50, 0.45, 0.40, 0.38, 0.36, 0.35, 0.34, 0.32, 0.30, 0.28, 0.25]:
    cnt = sum(1 for s in cross_sims if s >= thresh)
    print(f"  >= {thresh:.2f}: {cnt} pairs")

# Count per camera
from collections import Counter
cam_counts = Counter(camera_ids)
print(f"\nTracklets per camera: {dict(cam_counts)}")
