# Architecture Overhaul — Closing the 7.36pp Gap to SOTA

> **Date**: 2026-04-17
> **Baseline**: MTMC IDF1 = 0.775 (10c v52, reproducible)
> **Target**: MTMC IDF1 ≥ 0.849 (AIC22 1st place)
> **Gap**: 7.36pp
> **Strategy**: Replicate the AIC22 winning recipe — multi-model ensemble + proper training + camera-pair priors + structural association changes

---

## Root Cause Analysis

### Why We're Stuck at 0.775

The gap decomposes as follows based on AIC22 winner analysis:

| Missing Component | Est. Impact | Evidence |
|---|:---:|---|
| **Single model vs 3-5 ensemble** | 4-6pp | Every AIC winner (2020-2022) uses 3-5 models; our ensemble test with weak secondary (52.77% mAP at 0.30 weight) was neutral because the secondary adds noise, not signal |
| **Wrong ResNet training recipe** | 2-3pp (indirect) | Our ResNet101-IBN-a gets 52.77% mAP; winners get 70-80% with the same backbone using ArcFace+BNNeck+GeM+DMT. This blocks viable ensemble |
| **No timestamp-based CID_BIAS** | 1-2pp | Winners use camera timestamp offsets (from metadata), NOT GT-derived bias. Our CID_BIAS test (-3.3pp) was a different, inferior implementation |
| **No zone/transition modeling** | 0.5-1pp | Winners define entry/exit zones per camera with transition time windows |
| **Track-level vs box-grained matching** | 0.5-1pp | AIC22 1st place's key innovation — per-detection matching preserves fine-grained appearance info |
| **No two-stage clustering** | 0.5-1pp | Winners use AgglomerativeClustering (complete linkage) in two passes with feature re-averaging |

**Critical insight**: Techniques we marked as "dead ends" (384px, DMT, reranking, CID_BIAS) are ensemble-dependent. They fail with 1 model but succeed with 3-5 models because diversity absorbs the noise each technique introduces.

### Why Our ResNet101-IBN-a Only Gets 52.77% mAP

| Our Recipe (09d v18) | AIC Winners' DMT Recipe | Impact |
|---|---|---|
| Triplet + Center loss | **ArcFace(s=30, m=0.5)** + Triplet(margin=0.3) + Center(0.0005) | ArcFace provides much stronger ID supervision |
| Average pooling | **GeM pooling (p=3.0)** | GeM upweights discriminative spatial regions |
| No explicit BNNeck | **BNNeck** (batch norm before classifier) | BNNeck separates metric and classification feature spaces |
| lr=1e-3 | **lr=3e-4**, warmup 10 epochs (factor=0.01), MultiStep (40,70) gamma=0.1 | More conservative LR with proper warmup |
| 384×384 (square) | **384×128** (rectangular) | Vehicle crops are naturally wide; rectangular preserves aspect ratio |
| No label smoothing | **Label smoothing ON** | Prevents overconfident predictions on 128 IDs |
| ImageNet pretrain only | **VeRi-776 pretrain → CityFlowV2 fine-tune** (2-stage) | Progressive domain specialization |
| 120 epochs | **100 epochs** with proper scheduling | Better LR schedule compensates for fewer epochs |
| No flip test | **Flip test**: `feat = model(img) + model(flip(img))` | Free +0.5-1% mAP at inference |

**The recipe is wrong, not the backbone.** ResNet101-IBN-a with the DMT recipe achieves 70-80% mAP on VeRi-776 in published results. Our 52.77% is expected given the training recipe gap.

---

## Priority-Ordered Action Plan

### Phase 1: Fix ResNet101-IBN-a Training (CRITICAL PATH)
**Expected gain**: Enables ensemble (+4-6pp total)
**Compute**: ~8h P100 GPU
**Complexity**: Medium — requires training code changes

#### 1A. VeRi-776 Pretraining with DMT Recipe

**Notebook**: `09d` or new `09i_resnet101ibn_dmt_proper`

**Training Config**:
```yaml
backbone: resnet101_ibn_a
pooling: gem  # p=3.0
neck: bnneck  # BatchNorm before classifier
input_size: [384, 128]  # H×W, rectangular for vehicles
losses:
  id_loss: arcface  # scale=30, margin=0.5
  triplet_loss: soft_margin  # margin=0.3, weight=1.0
  center_loss: 0.0005
optimizer: adamw
lr: 3e-4
warmup:
  epochs: 10
  factor: 0.01
  method: linear
scheduler: multistep
milestones: [40, 70]
gamma: 0.1
epochs: 100
batch_size: 64  # 16 instances/class × 4 classes per batch
label_smoothing: 0.1
augmentations:
  random_flip: 0.5
  random_erasing: 0.5  # area 2-40%, ratio 0.3-3.33
  padding: 10  # pad + random crop
  color_jitter: null  # NOT the augoverhaul recipe — keep baseline
dataset: veri776  # 576 IDs, 37K images
```

**Target**: mAP ≥ 75% on VeRi-776 test set (currently 62.52% with old recipe)

#### 1B. CityFlowV2 Fine-tuning

**Resume from**: Best VeRi-776 checkpoint
**Config changes**:
```yaml
dataset: cityflowv2  # 128 IDs, 7.5K images
lr: 1e-4  # Lower LR for fine-tuning (was 3.5e-4 in failed 09f v3)
warmup:
  epochs: 5
  factor: 0.1
scheduler: cosine  # Cosine annealing smoother for fine-tuning
epochs: 60  # Shorter — small dataset overfits quickly
batch_size: 48  # Smaller batches for 128 IDs
triplet_loss:
  margin: 0.3
  mining: hard  # Hard mining for fine-tuning stage (DMT Stage 2)
```

**Target**: mAP ≥ 68% on CityFlowV2 eval split (currently 52.77%)

**Why this will work when 09f v3 (42.7%) failed**:
1. 09f v3 used `lr=3.5e-4` — too high for fine-tuning, caused catastrophic forgetting
2. 09f v3 had no BNNeck, no GeM, no ArcFace — weaker feature space to begin with
3. 09f v3 pretraining only reached 62.52% (without DMT recipe); proper DMT pretrain should be ≥75%
4. Using cosine annealing instead of MultiStep avoids sharp LR drops that destabilize small datasets

#### 1C. Alternative/Parallel: Train ResNet101-IBN-a Directly on CityFlowV2 with DMT Recipe
- Skip VeRi-776 step; apply ArcFace+BNNeck+GeM+proper scheduling directly
- Faster (~3h) but may yield lower mAP (~60-65%)
- Run as a hedge while VeRi-776 pretrain trains

### Phase 2: Add ResNeXt101-IBN-a as Third Model
**Expected gain**: Architectural diversity for ensemble
**Compute**: ~10h P100 GPU
**Complexity**: Medium — same recipe, different backbone

**Why ResNeXt101-IBN-a**:
- Used by AIC22 2nd place AND AIC21 1st place
- Different architecture family than ResNet (grouped convolutions)
- IBN-a provides built-in domain generalization
- Pretrained weights available from torchvision/timm

**Training**:
- Same DMT recipe as Phase 1 (ArcFace+BNNeck+GeM)
- VeRi-776 pretrain → CityFlowV2 fine-tune
- Backbone: `resnext101_32x8d_ibn_a` (existing in our training code as 09h)
- **Critical fix**: Use the proper DMT recipe, NOT the old recipe that 09h used

**Target**: mAP ≥ 65% on CityFlowV2 eval split

### Phase 3: 3-Model Ensemble Deployment
**Expected gain**: +4-6pp MTMC IDF1 (0.775 → 0.82-0.83)
**Compute**: ~4h P100 GPU (feature extraction)
**Complexity**: Low — infrastructure already exists

#### 3A. Feature Extraction (Stage 2)
```yaml
stage2:
  reid:
    vehicle:             # Model 1: ViT-B/16 CLIP, 768D
      enabled: true
      model_name: transreid
      weights_path: models/reid/transreid_cityflowv2_best.pth
      input_size: [256, 256]
      flip_test: true    # ADD: free +0.5% mAP
    vehicle2:            # Model 2: ResNet101-IBN-a DMT, 2048D
      enabled: true
      save_separate: true
      model_name: resnet101_ibn_a
      weights_path: models/reid/resnet101ibn_dmt_cityflowv2_best.pth
      input_size: [384, 128]  # Rectangular!
      embedding_dim: 2048
      flip_test: true
    vehicle3:            # Model 3: ResNeXt101-IBN-a DMT, 2048D
      enabled: true
      save_separate: true
      model_name: resnext101_ibn_a
      weights_path: models/reid/resnext101ibn_dmt_cityflowv2_best.pth
      input_size: [384, 128]
      embedding_dim: 2048
      flip_test: true
```

#### 3B. Fusion Strategy (Stage 4)

**Method**: L2-normalize → mean-average (AIC22 2nd/3rd place recipe)

```python
# Per-model: L2-normalize embeddings
emb1_norm = emb1 / np.linalg.norm(emb1, axis=1, keepdims=True)  # 768D
emb2_norm = emb2 / np.linalg.norm(emb2, axis=1, keepdims=True)  # 2048D -> PCA 768D
emb3_norm = emb3 / np.linalg.norm(emb3, axis=1, keepdims=True)  # 2048D -> PCA 768D

# PCA each to same dim, then mean-average
emb_fused = np.mean([emb1_pca, emb2_pca, emb3_pca], axis=0)
emb_fused = emb_fused / np.linalg.norm(emb_fused, axis=1, keepdims=True)
```

**Alternative**: Score-level fusion (existing infrastructure)
```yaml
stage4:
  association:
    secondary_embeddings:
      weight: 0.30  # Only viable when secondary mAP ≥ 68%
    tertiary_embeddings:
      weight: 0.20
```

**Crucially**: Apply FIC whitening to EACH model's features separately before fusion. The camera bias is different per model.

#### 3C. Re-enable Reranking (ONLY with ensemble)
```yaml
stage4:
  association:
    reranking:
      enabled: true
      k1: 20
      k2: 6
      lambda: 0.3
```
Previously hurt with single model (-0.5pp). With 3-model ensemble features, k-reciprocal consistently helps in AIC winners (+0.5-1.5pp).

### Phase 4: Timestamp-Based CID_BIAS (Correct Implementation)
**Expected gain**: +1-2pp
**Compute**: CPU only
**Complexity**: Medium

#### What We Got Wrong

Our previous CID_BIAS (-3.3pp) was computed from GT-matched tracklets (only 464/941). The winners derive it from **camera timestamp metadata** — sync offsets that convert frame numbers to wall-clock time.

#### Correct Implementation

1. **Parse CityFlowV2 timestamps**: Each camera has a timestamp offset in the dataset metadata
   ```python
   # From AIC22 2nd place code
   io_time = camera_bias[cam_id] + frame_list[0] / fps
   ```

2. **Compute per-camera-pair time windows**: For each (cam_i, cam_j) pair, determine the valid travel time range
   ```python
   # From AIC22 1st place code — hardcoded per camera pair
   time_thresholds = {
       (42, 43): 180,  # seconds
       (43, 44): 440,
       # ...
   }
   ```

3. **Apply exponential penalty** for associations outside valid time windows:
   ```python
   if travel_time > threshold:
       penalty = exp(alpha * (travel_time - threshold) / long_time_t)
       similarity *= (1.0 - penalty)
   ```

4. **Key difference from our FIC**: FIC is a global per-camera whitening. CID_BIAS is a **pairwise additive bias** that uses temporal constraints. They are complementary, not redundant.

#### CityFlowV2 Camera Topology
- **S01** (c001-c005): 5 cameras at an intersection, vehicles can appear in multiple cameras within ~30-180 seconds
- **S02** (c006-c008): 3 cameras on a highway segment, sequential travel in one direction, ~60-400 seconds between cameras
- S01↔S02 transitions are rare/impossible — apply strong negative bias

### Phase 5: Zone Definitions + Two-Stage Clustering
**Expected gain**: +0.5-1pp
**Compute**: CPU only
**Complexity**: Medium-High

#### 5A. Zone Definitions

Define 2-4 entry/exit zones per camera as pixel bounding boxes:

```python
# Format: {camera_id: [(zone_id, x_min, x_max, y_min, y_max), ...]}
ZONES = {
    "c001": [
        (1, 0, 400, 0, 960),      # Left entry
        (2, 880, 1280, 0, 960),    # Right entry
        (3, 0, 1280, 0, 200),      # Top entry
        (4, 0, 1280, 760, 960),    # Bottom entry
    ],
    "c002": [...],
    # ...
}
```

**How to determine zones**:
1. Extract first/last frame bbox positions from all tracklets
2. Cluster entry positions and exit positions separately using K-means
3. Manually refine based on video inspection
4. Define valid transitions: `(cam_i, zone_exit) → (cam_j, zone_entry)` with time range

#### 5B. Two-Stage AgglomerativeClustering

Replace `conflict_free_cc` with the AIC22 2nd/3rd place two-stage approach:

**Stage 1**: Loose thresholds on adjacent camera pairs
```python
from sklearn.cluster import AgglomerativeClustering

# Per camera-pair thresholds (tunable)
PAIR_THRESHOLDS_S1 = {
    ("c001", "c002"): 0.5,
    ("c002", "c003"): 0.5,
    # ...
}

# Complete linkage — pessimistic, prevents false merges
clustering = AgglomerativeClustering(
    n_clusters=None,
    distance_threshold=threshold,
    linkage='complete',
    metric='precomputed'
)
```

**Feature re-averaging**: After Stage 1, mean-average features of merged tracklets
```python
for cluster_id in unique_clusters:
    member_features = features[labels == cluster_id]
    merged_feature = np.mean(member_features, axis=0)
    merged_feature /= np.linalg.norm(merged_feature)
```

**Stage 2**: Tighter thresholds on merged features
```python
PAIR_THRESHOLDS_S2 = {
    ("c001", "c002"): 0.12,
    ("c002", "c003"): 0.10,
    # ...
}
```

**Why this beats connected components**: CC is all-or-nothing; agglomerative clustering with complete linkage is more conservative about merging dissimilar tracklets and the two-stage approach progressively refines.

### Phase 6: Box-Grained Matching (AIC22 Winner Innovation)
**Expected gain**: +0.5-1pp
**Compute**: ~10% more GPU time in Stage 2
**Complexity**: High

#### Concept

Instead of averaging features over a track and comparing track-vs-track, BGM:
1. Stores per-detection features (not just track-averaged)
2. For cross-camera comparison: builds a box-to-box similarity matrix
3. Uses temporal masks to weight box pairs by travel time validity
4. Requires top-K bidirectional verification (topk=5, r_rate=0.5)

#### Implementation

**Stage 2 changes**:
```yaml
stage2:
  reid:
    vehicle:
      save_per_detection: true  # NEW: save all crop features, not just track-averaged
      max_crops_per_track: 20   # Cap storage (quality-scored top-20)
```

**Stage 4 changes**:
```python
def box_grained_similarity(track_i_features, track_j_features, 
                            track_i_times, track_j_times,
                            time_threshold, topk=5, r_rate=0.5):
    """Compute BGM similarity between two tracklets."""
    # Build N×M cosine similarity matrix
    sim_matrix = track_i_features @ track_j_features.T  # [N, M]
    
    # Apply temporal mask
    time_diff = np.abs(track_i_times[:, None] - track_j_times[None, :])
    temporal_mask = np.exp(-alpha * np.maximum(0, time_diff - time_threshold))
    sim_matrix *= temporal_mask
    
    # Top-K forward matches
    top_k_forward = np.sort(sim_matrix.max(axis=1))[-topk:]
    
    # Bidirectional verification
    top_k_reverse = np.sort(sim_matrix.max(axis=0))[-topk:]
    
    # Require r_rate fraction of reverse matches
    n_verified = np.sum(top_k_reverse > np.median(top_k_forward))
    if n_verified / topk < r_rate:
        return 0.0
    
    return np.mean(top_k_forward)
```

**Storage**: For 929 tracklets × 20 crops × 768D = ~57MB additional. Manageable.

---

## Kaggle Compute Budget

**Available**: ~30 GPU hours/week on P100

| Phase | GPU Hours | Calendar | Can Parallelize? |
|---|:---:|:---:|:---:|
| 1A: ResNet101-IBN-a VeRi-776 pretrain | 4h | Week 1 | — |
| 1B: ResNet101-IBN-a CityFlowV2 fine-tune | 2h | Week 1 | — |
| 1C: ResNet101-IBN-a direct CityFlowV2 (hedge) | 3h | Week 1 | ✅ with 1A (different kernel) |
| 2: ResNeXt101-IBN-a full pipeline | 10h | Week 2 | — |
| 3: 3-model feature extraction (10a) | 4h | Week 2-3 | After 1B+2 |
| 3+: Association experiments (10c) | 2h CPU | Week 3 | — |
| **Total GPU** | **~23h** | **~3 weeks** | |

Phase 4-6 are CPU-only and can run locally in parallel with Kaggle training.

---

## Implementation Complexity

| Phase | Complexity | Code Changes | Files Affected |
|---|:---:|---|---|
| 1: Fix ResNet training | **Medium** | Training notebook rewrite + model.py (ArcFace, BNNeck, GeM) | `notebooks/kaggle/09i*`, `src/training/model.py`, `src/training/losses.py` |
| 2: ResNeXt101 training | **Low** | Same recipe, different backbone name | `notebooks/kaggle/09h*` (reuse with corrected recipe) |
| 3: Ensemble deployment | **Low** | Config changes, minor Stage 2/4 tweaks | `configs/*.yaml`, `src/stage2*/pipeline.py`, `src/stage4*/pipeline.py` |
| 4: CID_BIAS from timestamps | **Medium** | New module for timestamp parsing + temporal penalty | `src/stage4*/cid_bias.py` (new), `src/stage4*/pipeline.py` |
| 5A: Zone definitions | **Medium** | Manual annotation + zone lookup module | `configs/zones/cityflowv2.yaml` (new), `src/stage4*/zones.py` (new) |
| 5B: Two-stage clustering | **Medium-High** | Replace CC solver with AgglomerativeClustering | `src/stage4*/pipeline.py`, `src/stage4*/clustering.py` (new) |
| 6: Box-grained matching | **High** | Per-detection storage + BGM similarity | `src/stage2*/pipeline.py`, `src/stage4*/pipeline.py`, `src/stage4*/bgm.py` (new) |

---

## Parallelization Strategy

```
Week 1:
  GPU: [1A: ResNet101 VeRi pretrain] ─→ [1B: ResNet101 CityFlowV2 fine-tune]
  GPU: [1C: ResNet101 direct CityFlowV2 (hedge)]  # parallel kernel
  CPU: [4: CID_BIAS from timestamps]               # local
  CPU: [5A: Zone annotation]                        # local

Week 2:
  GPU: [2: ResNeXt101 VeRi pretrain → CityFlowV2 fine-tune]
  CPU: [5B: Two-stage clustering implementation]    # local
  CPU: [4: CID_BIAS integration into Stage 4]       # local

Week 3:
  GPU: [3: 3-model feature extraction (10a)]
  CPU: [3+: Association experiments with ensemble features (10c)]
  CPU: [6: BGM implementation (if ensemble shows gains)]

Week 4:
  GPU: [Final full pipeline run with all improvements]
  CPU: [Ablation study: measure contribution of each component]
```

---

## Success Criteria

| Milestone | Target IDF1 | When | Pass/Fail |
|---|:---:|:---:|:---:|
| ResNet101-IBN-a with DMT recipe on CityFlowV2 | mAP ≥ 68% | Week 1 | If < 60%, recipe still wrong |
| ResNeXt101-IBN-a on CityFlowV2 | mAP ≥ 65% | Week 2 | If < 55%, backbone choice wrong |
| 3-model ensemble (score-level fusion) | MTMC IDF1 ≥ 0.81 | Week 3 | If < 0.79, fusion strategy wrong |
| + CID_BIAS + zones | MTMC IDF1 ≥ 0.83 | Week 3-4 | If < 0.81, priors not helping |
| + reranking + BGM | MTMC IDF1 ≥ 0.845 | Week 4 | SOTA territory |

---

## Risk Mitigation

### Risk 1: ResNet101-IBN-a Still Underperforms (< 60% mAP)
**Mitigation**: 
- Try ConvNeXt-Base instead (different architecture family, modern training recipe)
- Try SwinTransformer-Base (used by AIC22 1st place for detection; can be adapted for ReID)
- If IBN-a backbones don't work with our training infra, consider downloading pretrained DMT weights from the open-source AIC21/22 repos

### Risk 2: Ensemble Doesn't Help Despite Strong Individual Models
**Mitigation**:
- Test feature-level fusion (concat→PCA) vs score-level fusion vs rank fusion
- The AIC22 2nd place uses simple L2-norm mean-average — very robust
- If all fusion methods fail, the models themselves may lack diversity (same training data, similar representations)

### Risk 3: P100 Memory Constraints
**Mitigation**:
- ResNet101-IBN-a at 384×128 with batch=64 fits in 16GB P100
- ResNeXt101-IBN-a may need batch=48 or gradient accumulation
- Feature extraction is sequential per model — no memory issue

### Risk 4: CityFlowV2 Has Only 128 IDs — Overfitting
**Mitigation**:
- Use VeRi-776 pretrain (576 IDs) as warm start
- Early stopping on validation mAP
- Label smoothing prevents overconfident predictions
- Random erasing + flip as regularization
- Consider DMT Stage 2 unsupervised domain adaptation (DBSCAN on unlabeled test data)

---

## What NOT to Try (Confirmed Dead Ends)

- ❌ CircleLoss on any backbone (catastrophic: 18.45% mAP, inf loss)
- ❌ 384px as sole model (−2.8pp MTMC IDF1)
- ❌ DMT camera-aware training as sole model (−1.4pp)
- ❌ Reranking with single-model features (always hurts)
- ❌ Feature concatenation without PCA (mixes uncalibrated spaces)
- ❌ CSLS distance (−34.7pp catastrophic)
- ❌ Hierarchical centroid clustering (−1 to −5pp)
- ❌ AFLink motion linking (−3.8 to −13.2pp)
- ❌ SAM2 foreground masking (−8.7pp)
- ❌ More association parameter sweeps (225+ configs exhausted)
- ❌ SGD optimizer (30.27% mAP catastrophic)
- ❌ Augoverhaul augmentations for MTMC (−5.3pp despite +1.45pp mAP)

---

## Concrete Next Step: Week 1 Implementation

### Step 1: Create `09i_resnet101ibn_dmt` Training Notebook

**Purpose**: ResNet101-IBN-a with proper DMT recipe on VeRi-776

**Key code changes needed**:

1. **`src/training/model.py`** — Add:
   - `GeM` pooling layer (p=3.0, trainable)
   - `BNNeck` (BatchNorm1d after pooling, before classifier)
   - `ArcFace` classification head (scale=30, margin=0.5)

2. **`src/training/losses.py`** — Add:
   - `ArcFaceLoss` implementation
   - Modified `CenterLoss` with weight=0.0005

3. **Notebook cells**:
   - Cell 1: Install deps (timm, faiss-cpu)
   - Cell 2: Clone repo, import training modules  
   - Cell 3: Configure VeRi-776 dataset loader (576 IDs, 384×128)
   - Cell 4: Build model with DMT recipe
   - Cell 5: Train 100 epochs with ArcFace+Triplet+Center
   - Cell 6: Evaluate mAP on VeRi-776 test
   - Cell 7: Save best checkpoint

### Step 2: Parse CityFlowV2 Timestamps (Local, CPU)

**Purpose**: Compute camera sync offsets for CID_BIAS

```python
# CityFlowV2 provides timestamps in the dataset metadata
# Parse and compute pairwise offsets
for cam_i, cam_j in camera_pairs:
    offset = timestamps[cam_j] - timestamps[cam_i]
    travel_time_range = (min_travel, max_travel)  # from GT statistics
```

### Step 3: Annotate CityFlowV2 Zones (Local, CPU)

**Purpose**: Define entry/exit regions per camera

1. Extract sample frames from each camera
2. Identify vehicle entry/exit points visually
3. Define 2-4 zones per camera as pixel bounding boxes
4. Store in `configs/zones/cityflowv2.yaml`

---

## Expected Cumulative Gains

| Phase | Component | Individual Gain | Cumulative IDF1 |
|:---:|---|:---:|:---:|
| — | Baseline | — | 0.775 |
| 1-3 | 3-model ensemble (properly trained) | +4-6pp | 0.82-0.83 |
| 4 | CID_BIAS from timestamps | +1-2pp | 0.83-0.84 |
| 3C | Reranking (with ensemble features) | +0.5-1pp | 0.835-0.845 |
| 5 | Zones + two-stage clustering | +0.5-1pp | 0.84-0.85 |
| 6 | Box-grained matching | +0.5-1pp | 0.845-0.86 |
| **Total** | | **+7-11pp** | **0.845-0.86** |

**Conservative projection**: IDF1 ≈ 0.835 (Phases 1-4 only)
**Optimistic projection**: IDF1 ≈ 0.855 (all phases)
**SOTA**: IDF1 = 0.8486

**The 3-model ensemble alone should get us to ~0.82-0.83. Adding CID_BIAS and reranking pushes toward SOTA. Zones, two-stage clustering, and BGM are gravy.**

---

## References

- AIC22 1st: [Yejin0111/AICITY2022-Track1-MTMC](https://github.com/Yejin0111/AICITY2022-Track1-MTMC) — BGM, 5-model ensemble
- AIC22 2nd: [coder-wangzhen/AIC22-MCVT](https://github.com/coder-wangzhen/AIC22-MCVT) — 3-model DMT, CID_BIAS, AgglomerativeClustering
- AIC21 1st: [LCFractal/AIC21-MTMC](https://github.com/LCFractal/AIC21-MTMC) — Zone-based ST, DMT
- DMT: [michuanhaohao/AICITY2021_Track2_DMT](https://github.com/michuanhaohao/AICITY2021_Track2_DMT) — Training recipe reference
