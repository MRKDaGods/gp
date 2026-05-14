# Multi-Query Track Representation — Implementation Spec

## Goal

Replace the single averaged embedding per tracklet with a multi-query representation that retains the top-K most representative per-crop embeddings. Cross-camera similarity then uses `max(sim(q_i, g_j))` over all K×K pairs, capturing appearance variation within a track (e.g., front vs rear of a vehicle).

**Expected impact**: +0.3–0.8pp MTMC IDF1 (per findings.md gap decomposition).

## Hypothesis

Averaging all crop embeddings into a single vector loses information about intra-track appearance variation. Vehicles seen from different angles produce significantly different embeddings. When two tracklets share at least one similar viewpoint, the max-of-K×K similarity will recover matches that the averaged representation misses — while the max operator is robust to unmatched viewpoints because they only need ONE good pair.

## Current Flow (What Changes)

### Stage 2 (`src/stage2_features/pipeline.py`)

**Current**: `ReIDModel.get_tracklet_embedding_from_scored_crops()` calls `extract_features()` to get per-crop embeddings (N_crops × D), then quality-weighted averages them into a single (D,) vector. This is stored as `TrackletFeatures.embedding`.

**After**: In addition to the averaged embedding (kept for backward compat and FAISS), store the top-K individual crop embeddings per tracklet as a separate artifact.

### Stage 3 (`src/stage3_indexing/pipeline.py`)

**No changes required.** FAISS continues to index the averaged embeddings for candidate retrieval (top-K retrieval and QE/DBA). The multi-query similarity is computed only during the exhaustive cross-camera pair scoring in Stage 4, which already bypasses FAISS for the actual similarity computation.

### Stage 4 (`src/stage4_association/pipeline.py`, `similarity.py`)

**Current**: `_build_all_cross_camera_pairs()` computes `embeddings @ embeddings.T` with single-vector cosine similarity. `appearance_sim` dict feeds into `compute_combined_similarity()`.

**After**: When multi-query embeddings are available, replace the single dot product with `max over K×K` pairwise similarities for each tracklet pair.

## Detailed Changes

### 1. Data Model — `src/core/data_models.py`

Add an optional field to `TrackletFeatures`:

```python
@dataclass
class TrackletFeatures:
    # ... existing fields ...
    multi_query_embeddings: Optional[np.ndarray] = None  # shape: (K, D), top-K crop embeddings
```

### 2. Stage 2 — Extract & Save Multi-Query Embeddings

#### 2a. `src/stage2_features/reid_model.py` — New method

Add a new method `get_tracklet_multi_query_embeddings()` that:
1. Calls `extract_features(crops)` to get all per-crop embeddings (N_crops × D)
2. Applies quality-weighted selection to pick top-K crops (by quality score, not random)
3. Returns the K individual embeddings as (K, D) array, each L2-normalized

```python
def get_tracklet_multi_query_embeddings(
    self,
    scored_crops: List["QualityScoredCrop"],
    k: int = 5,
    cam_id: Optional[int] = None,
) -> Optional[np.ndarray]:
    """Extract top-K representative embeddings for a tracklet.
    
    Selection strategy: pick the K highest-quality crops.
    Each embedding is individually L2-normalized.
    
    Returns:
        (K, D) array, or (N_crops, D) if N_crops < K. None if no crops.
    """
```

#### 2b. `src/stage2_features/pipeline.py` — Per-tracklet extraction

In the main tracklet processing loop (around line 310), after computing `raw_embedding`, also compute multi-query embeddings:

```python
# After: raw_embedding = reid.get_tracklet_embedding_from_scored_crops(...)
multi_query_k = stage_cfg.get("multi_query", {}).get("k", 0)
mq_embeddings = None
if multi_query_k > 0:
    mq_embeddings = reid.get_tracklet_multi_query_embeddings(
        scored_crops, k=multi_query_k, cam_id=sie_cam_id,
    )
```

Store in `TrackletFeatures`:
```python
all_features.append(TrackletFeatures(
    ...,
    multi_query_embeddings=mq_embeddings,
))
```

#### 2c. `src/stage2_features/pipeline.py` — Post-processing & saving

After PCA whitening and L2 normalization of the main embeddings, apply the same transforms to multi-query embeddings:

```python
# After: embeddings = l2_normalize(embeddings)
if multi_query_k > 0:
    # Stack all multi-query embeddings: list of (K_i, D_raw) -> apply camera_bn, PCA, L2
    # Process as a single flat matrix, then reshape back
    mq_list = [f.multi_query_embeddings for f in all_features if f.multi_query_embeddings is not None]
    if mq_list:
        # Track which feature each row belongs to, and the per-tracklet K
        mq_sizes = [mq.shape[0] for mq in mq_list]
        mq_flat = np.concatenate(mq_list, axis=0)  # (sum(K_i), D_raw)
        mq_camera_ids = []
        for feat, sz in zip(all_features, mq_sizes):
            mq_camera_ids.extend([feat.camera_id] * sz)
        
        # Apply same transforms: camera_bn → PCA → L2
        if stage_cfg.get("camera_bn", {}).get("enabled", True):
            mq_flat = camera_aware_batch_normalize(mq_flat, mq_camera_ids)
        if stage_cfg.pca.enabled and whitener is not None:
            mq_flat = whitener.transform(mq_flat)
        mq_flat = l2_normalize(mq_flat)
        
        # Split back and assign
        offset = 0
        feat_idx = 0
        for feat in all_features:
            if feat.multi_query_embeddings is not None:
                sz = mq_sizes[feat_idx]
                feat.multi_query_embeddings = mq_flat[offset:offset+sz]
                offset += sz
                feat_idx += 1
```

#### 2d. `src/core/io_utils.py` — Save/load multi-query embeddings

Add new I/O functions:

```python
def save_multi_query_embeddings(
    mq_embeddings: List[Optional[np.ndarray]],
    output_dir: Path,
) -> None:
    """Save multi-query embeddings as a dict of arrays keyed by tracklet index."""
    mq_dict = {}
    for i, mq in enumerate(mq_embeddings):
        if mq is not None:
            mq_dict[str(i)] = mq
    np.savez_compressed(output_dir / "multi_query_embeddings.npz", **mq_dict)

def load_multi_query_embeddings(input_dir: Path, n: int) -> List[Optional[np.ndarray]]:
    """Load multi-query embeddings. Returns list of length n."""
    path = input_dir / "multi_query_embeddings.npz"
    if not path.exists():
        return [None] * n
    data = np.load(path)
    result = [None] * n
    for key in data.files:
        idx = int(key)
        if idx < n:
            result[idx] = data[key]
    return result
```

### 3. Stage 4 — Multi-Query Similarity

#### 3a. `src/stage4_association/pipeline.py` — Load & use multi-query embeddings

In `run_stage4()`, after building the embedding matrix (around line 80):

```python
# Load multi-query embeddings if available
mq_cfg = stage_cfg.get("multi_query", {})
mq_enabled = mq_cfg.get("enabled", False)
mq_embeddings: List[Optional[np.ndarray]] = [None] * n
if mq_enabled:
    from src.core.io_utils import load_multi_query_embeddings
    mq_dir = Path(stage_cfg.get("multi_query_dir", "")) or output_dir.parent / "stage2"
    mq_embeddings = load_multi_query_embeddings(mq_dir, n)
    mq_count = sum(1 for mq in mq_embeddings if mq is not None)
    logger.info(f"Multi-query embeddings loaded: {mq_count}/{n} tracklets")
```

**Important**: FIC whitening must also be applied to multi-query embeddings if enabled. After the FIC step:

```python
if fic_cfg.get("enabled", False) and mq_enabled:
    # Apply FIC to multi-query embeddings (same per-camera whitening)
    # Flatten → whiten → split back
    ...  # Same flatten/whiten/split pattern as stage2 post-processing
```

#### 3b. `src/stage4_association/pipeline.py` — Replace similarity in `_build_all_cross_camera_pairs`

Add a new function `_build_all_cross_camera_pairs_multi_query()`:

```python
def _build_all_cross_camera_pairs_multi_query(
    n: int,
    embeddings: np.ndarray,         # (N, D) averaged — used as fallback
    mq_embeddings: List[Optional[np.ndarray]],  # list of (K_i, D) or None
    camera_ids: List[str],
    class_ids: List[int],
    min_similarity: float = 0.0,
) -> List[Tuple[int, int, float]]:
    """Exhaustive cross-camera pairs using max-of-K×K multi-query similarity.
    
    For each cross-camera pair (i, j):
    - If both have multi-query embeddings: sim = max(mq_i @ mq_j.T)
    - If only one has them: sim = max(mq @ avg.T) (K×1 max)
    - If neither: sim = avg_i @ avg_j (scalar, same as current)
    """
```

The core computation for a camera pair block:
```python
# For tracklets i (cam_a) and j (cam_b):
if mq_i is not None and mq_j is not None:
    # (K_i, D) @ (D, K_j) -> (K_i, K_j), take max
    sim = (mq_i @ mq_j.T).max()
elif mq_i is not None:
    # (K_i, D) @ (D,) -> (K_i,), take max
    sim = (mq_i @ embeddings[j]).max()
elif mq_j is not None:
    sim = (embeddings[i] @ mq_j.T).max()
else:
    sim = float(embeddings[i] @ embeddings[j])
```

In `run_stage4()`, replace the exhaustive pair building call:
```python
if exhaustive_cfg:
    if mq_enabled and any(mq is not None for mq in mq_embeddings):
        candidate_pairs = _build_all_cross_camera_pairs_multi_query(
            n, embeddings, mq_embeddings, camera_ids, class_ids, min_similarity=min_sim,
        )
    else:
        candidate_pairs = _build_all_cross_camera_pairs(
            n, embeddings, camera_ids, class_ids, min_similarity=min_sim,
        )
```

### 4. Config Additions

#### `configs/default.yaml` — Stage 2

```yaml
stage2:
  # ... existing ...
  multi_query:
    k: 0  # 0 = disabled (backward compat). Set to 3-5 to enable.
```

#### `configs/default.yaml` — Stage 4

```yaml
stage4:
  association:
    # ... existing ...
    multi_query:
      enabled: false
      dir: ""  # path to stage2 output with multi_query_embeddings.npz. Empty = auto-detect.
```

#### Config override examples:
```
stage2.multi_query.k=5
stage4.association.multi_query.enabled=true
```

### 5. Kaggle Notebook Changes

| Notebook | Change | Reason |
|----------|--------|--------|
| **10a** (stages 0-2, GPU) | Add `stage2.multi_query.k=5` override | Extract & save multi-query embeddings |
| **10b** (stage 3, CPU) | No changes | FAISS still indexes averaged embeddings |
| **10c** (stages 4-5, CPU) | Add `stage4.association.multi_query.enabled=true` | Use multi-query similarity |

The `multi_query_embeddings.npz` file flows from 10a output → 10c input alongside the existing `embeddings.npy`.

## Complexity Assessment

| Component | Complexity | Risk |
|-----------|:----------:|:----:|
| Data model change | Low | None — additive optional field |
| Stage 2 extraction | Medium | Must get PCA/L2 transforms right on MQ embeddings |
| Stage 2 I/O | Low | New npz file, no existing format changes |
| Stage 3 | None | No changes needed |
| Stage 4 similarity | Medium | Max-of-K×K is O(K²) per pair, manageable for K≤5 |
| Stage 4 FIC compat | Medium | FIC whitening must be applied to MQ embeddings too |
| Config | Low | Additive, backward compatible |
| Kaggle notebooks | Low | Override additions only |

**Overall: Medium complexity, ~200-300 lines of new code.**

## Risks

1. **Memory**: With K=5 and D=256 (post-PCA), each tracklet's multi-query is 5×256×4 = 5KB. For 800 tracklets = 4MB. Negligible.

2. **PCA alignment**: The multi-query embeddings MUST go through the exact same PCA transform as the averaged embeddings. If they don't, the similarity space is inconsistent. The spec handles this by processing them through the same `whitener.transform()`.

3. **FIC interaction**: FIC whitening in stage4 subtracts per-camera mean and decorrelates. Must apply to MQ embeddings too, or the max-of-K×K operates in a different feature space than the averaged embeddings used for QE/FAISS.

4. **K selection sensitivity**: Too small K (1-2) ≈ just picking the best crop (loses diversity). Too large K (>8) includes low-quality crops that add noise. K=3-5 is the sweet spot based on SOTA literature.

5. **Diminishing returns**: If the ViT already produces viewpoint-invariant features (which CLIP pretraining encourages), the multi-query approach may not help much. The +0.3-0.8pp estimate is uncertain.

6. **No interaction with reranking**: k-reciprocal reranking operates on the averaged embeddings. Multi-query only affects the exhaustive pair similarity. This is fine since reranking is currently disabled.

## Testing Plan

1. **Unit test**: Verify `get_tracklet_multi_query_embeddings()` returns correct shape (K, D) and each row is L2-normalized
2. **Integration test**: Run stage2 with `multi_query.k=5`, verify `multi_query_embeddings.npz` is saved with correct shapes
3. **Ablation on Kaggle**:
   - Baseline: current v80/v44 config (IDF1=0.784)
   - +MQ K=3: `stage2.multi_query.k=3 stage4.association.multi_query.enabled=true`
   - +MQ K=5: `stage2.multi_query.k=5 stage4.association.multi_query.enabled=true`
   - +MQ K=8: `stage2.multi_query.k=8 stage4.association.multi_query.enabled=true`

## Implementation Order

1. `src/core/data_models.py` — Add optional field (1 line)
2. `src/stage2_features/reid_model.py` — Add `get_tracklet_multi_query_embeddings()` (~30 lines)
3. `src/core/io_utils.py` — Add save/load functions (~25 lines)
4. `src/stage2_features/pipeline.py` — Extract, post-process, save MQ embeddings (~50 lines)
5. `src/stage4_association/pipeline.py` — Load MQ, new similarity function (~80 lines)
6. `configs/default.yaml` — Add config sections (~6 lines)
7. Update Kaggle notebook 10a and 10c config overrides
8. Run ablation on Kaggle