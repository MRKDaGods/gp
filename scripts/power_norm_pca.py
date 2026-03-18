"""Generate power-normalized PCA embeddings."""
import numpy as np
from sklearn.decomposition import PCA

raw = np.load("data/outputs/run_20260315_v2/stage2/embeddings_768d_nopca.npy")
print(f"Raw: {raw.shape}")

for alpha in [0.5, 0.3]:
    pn = np.sign(raw) * np.abs(raw) ** alpha
    norms = np.linalg.norm(pn, axis=1, keepdims=True)
    pn = pn / np.maximum(norms, 1e-8)
    
    pca = PCA(n_components=280, whiten=True, random_state=42)
    emb = pca.fit_transform(pn)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / np.maximum(norms, 1e-8)
    ev = pca.explained_variance_ratio_.sum()
    
    tag = str(int(alpha * 10))
    fname = f"embeddings_pca280d_pn{tag}.npy"
    np.save(f"data/outputs/run_20260315_v2/stage2/{fname}", emb)
    print(f"alpha={alpha}: explained_var={ev:.4f}, saved {fname}")

print("Done")
