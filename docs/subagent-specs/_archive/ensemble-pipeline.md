# Multi-Model Feature Ensemble Pipeline — Design Spec

> Created: 2026-04-17 | Status: Ready for implementation

## 1. Problem Analysis

### Current State
- **Primary model**: TransReID ViT-B/16 CLIP 256px — mAP=80.14%, R1=92.27% (trained, deployed)
- **Secondary model**: ResNet101-IBN-a — mAP=52.77% (ImageNet→CityFlowV2 direct), training 09i with ArcFace to reach ≥65%
- **Current best MTMC IDF1**: 0.775 (10c v52)
- **SOTA target**: 0.8486 (AIC22 1st place, 5-model ensemble)
- **Gap**: 7.36pp — driven by feature quality (single model), NOT association tuning (225+ configs exhausted)

### Why Ensemble Is the Only Remaining Path
1. **Association is EXHAUSTED**: 225+ configs within 0.3pp of optimal. No more gains from stage 4 tuning.
2. **Single-model upgrades fail**: 384px (-2.8pp), DMT (-1.4pp), augmentation overhaul (-5.3pp), SAM2 masking (-8.7pp). Higher mAP does NOT translate to better MTMC with a single model.
3. **AIC22 universal pattern**: Every top team used 3-5 model ensembles. Ensemble diversity is the key, not a single stronger backbone.
4. **Ensemble-dependent techniques**: Methods like reranking and CID_BIAS that fail with 1 model may work with 3+ models because identity signal survives through model diversity.

### Previous Ensemble Attempt: Why It Failed
- **10c v52 fusion test**: TransReID (80.14% mAP) + ResNet101-IBN (52.77% mAP) at 0.30 weight → **-0.1pp MTMC IDF1**
- **Root cause**: 28pp mAP gap between primary and secondary. The weak secondary adds noise, not complementary signal.
- **Threshold**: Secondary must reach **≥65% mAP** (ideally 70%+) for ensemble benefit. All AIC22 winners used models within 5-10pp mAP of each other.

## 2. Architecture Comparison

### Option A: Feature-Level Fusion (Concatenation)
- **Method**: Concatenate L2-normed embeddings from both models → joint PCA whitening → single similarity computation
- **Pros**: Single similarity matrix; simpler stage 4 logic; PCA can learn cross-model correlations
- **Cons**: Mixes uncalibrated feature spaces; already tested and failed (-1.6pp); requires re-fitting PCA on concatenated features; dimensional imbalance (768D ViT + 2048D ResNet → 2816D raw)
- **Verdict**: **REJECTED** — confirmed dead end in our pipeline. Feature concatenation was already tested and degraded MTMC IDF1 by 1.6pp.

### Option B: Score-Level Fusion (Weighted Similarity Blending)
- **Method**: Extract features from each model independently → separate PCA whitening per model → separate FIC whitening per model → compute cosine similarity independently → weighted average of similarities at stage 4
- **Pros**: Each model operates in its own calibrated feature space; PCA and FIC applied independently per model; easy to tune blending weight; can disable/enable models without re-extracting
- **Cons**: Requires storing separate embedding files; slightly more stage 4 compute (linear in number of models × number of pairs)
- **Verdict**: **RECOMMENDED** — this is exactly what AIC22 top teams did, and our infrastructure already supports it.

### Option C: Rank-Level Fusion
- **Method**: Retrieve top-K matches per query from each model independently → merge ranked lists
- **Pros**: Robust to poorly calibrated similarity scores
- **Cons**: Requires fundamental changes to the graph-based CC association; not compatible with our similarity-threshold-based approach
- **Verdict**: **NOT RECOMMENDED** — requires major architectural changes for uncertain benefit.

## 3. Recommended Approach: Score-Level Fusion

### Architecture Overview

```
Stage 2 (10a, GPU):
  ┌─────────────────────────────────────────────────────────┐
  │ For each tracklet:                                      │
  │   crops = quality_scored_crop_selection(tracklet)        │
  │                                                         │
  │   PRIMARY (TransReID ViT-B/16 CLIP 256px):              │
  │     raw_emb_1 = extract(crops, model=ViT)    → 768D     │
  │     → camera_bn → power_norm → PCA(384D) → L2_norm     │
  │     → embeddings.npy                                    │
  │                                                         │
  │   SECONDARY (ResNet101-IBN-a 384px):                    │
  │     raw_emb_2 = extract(crops, model=ResNet) → 2048D    │
  │     → camera_bn → PCA(384D) → L2_norm                  │
  │     → embeddings_secondary.npy                          │
  └─────────────────────────────────────────────────────────┘

Stage 4 (10c, CPU):
  ┌─────────────────────────────────────────────────────────┐
  │ Load embeddings.npy         (N, 384) — primary          │
  │ Load embeddings_secondary.npy (N, 384) — secondary      │
  │                                                         │
  │ For each cross-camera pair (i, j):                      │
  │   sim_primary   = cosine(emb_1[i], emb_1[j])           │
  │   sim_secondary = cosine(emb_2[i], emb_2[j])           │
  │   sim_combined  = w_pri * sim_primary                   │
  │                 + w_sec * sim_secondary                  │
  │                                                         │
  │ → Graph construction → conflict-free CC → evaluation    │
  └─────────────────────────────────────────────────────────┘
```

### PCA Whitening Strategy

**Per-model PCA whitening (BEFORE fusion)**:
- Primary: 768D → 384D via `pca_transform.pkl` (sklearn PCA, whiten=True)
- Secondary: 2048D → 384D via `pca_transform_secondary.pkl`
- Tertiary (future): 2048D → 384D via `pca_transform_tertiary.pkl`

**Rationale**: Each model has a different feature distribution. Joint PCA on concatenated features was tested and failed (-1.6pp). Per-model PCA ensures each stream is decorrelated and unit-variance in its own space before similarity computation.

**PCA fitting**: Fit on the CityFlowV2 test set embeddings (all tracklets from all cameras). This is the standard MTMC practice — PCA is unsupervised and the test set distribution is what we need to whiten against. Each model gets its own PCA model saved as a separate `.pkl` file.

### FIC (Per-Camera) Whitening Strategy

**Per-model FIC whitening (AT fusion time in stage 4)**:
- Primary embeddings: FIC with regularisation=0.1 (proven optimal for 384D)
- Secondary embeddings: FIC with same regularisation (applied independently)

**Rationale**: Camera-specific distribution bias exists independently in each model's feature space. FIC must be applied per-model, per-camera, which is exactly what the current stage 4 code already does.

### Weight Tuning Strategy

**Starting point**: `w_primary = 0.7, w_secondary = 0.3` (proportional to mAP ratio)

**Sweep plan** (once secondary mAP ≥ 65%):
| w_primary | w_secondary | Notes |
|:---------:|:-----------:|-------|
| 1.0 | 0.0 | Control (single model baseline) |
| 0.9 | 0.1 | Conservative blend |
| 0.8 | 0.2 | Moderate blend |
| 0.7 | 0.3 | Default (AIC22 typical) |
| 0.6 | 0.4 | Aggressive blend |
| 0.5 | 0.5 | Equal weight (unlikely optimal given mAP gap) |

**Three-model sweep** (when tertiary available):
| w_pri | w_sec | w_tert | Notes |
|:-----:|:-----:|:------:|-------|
| 0.6 | 0.25 | 0.15 | Typical AIC22 3-model split |
| 0.5 | 0.3 | 0.2 | More aggressive diversity |
| 0.7 | 0.2 | 0.1 | Primary-dominant |

## 4. Implementation Plan

### Phase 1: Secondary Model Training (09i ArcFace)

**Notebook**: `notebooks/kaggle/09i_resnet101ibn_arcface/09i_resnet101ibn_arcface.ipynb`

**Goal**: Train ResNet101-IBN-a with ArcFace loss to reach ≥65% mAP on CityFlowV2.

**Why ArcFace**: 
- ArcFace adds an angular margin penalty that pushes apart similar-looking vehicles
- More stable than CircleLoss (which caused training collapse on our ViT recipe)
- Used by AIC22 2nd place team for their ResNet101-IBN-a models
- Expected to close the 52.77% → 65%+ mAP gap by learning more discriminative features

**Training config**:
- Backbone: ResNet101-IBN-a (ImageNet pretrained)
- Loss: ArcFace (s=30, m=0.50) + CenterLoss (weight=0.0005)
- Input size: 384×384 (ResNet can benefit from higher resolution unlike ViT's viewpoint-texture issue)
- Optimizer: AdamW (lr=3.5e-4, weight_decay=5e-4)
- Schedule: Cosine annealing, 120 epochs, warmup 10 epochs
- Dataset: CityFlowV2 train split (576 IDs)
- Augmentation: Standard (flip, pad+crop, color jitter, random erasing) — NO augoverhaul

**Kaggle dependencies** (kernel-metadata.json):
```json
{
  "dataset_sources": [
    "thanhnguyenle/data-aicity-2023-track-2",
    "gumfreddy/mtmc-weights"
  ]
}
```

### Phase 2: Model Weight Upload to Kaggle

After 09i training completes with ≥65% mAP:

1. **Download trained weights** from Kaggle output:
   ```bash
   kaggle kernels output gumfreddy/09i-resnet101ibn-arcface -p /tmp/09i_output
   ```

2. **Upload as Kaggle dataset** (for 10a to consume):
   ```bash
   # Create dataset metadata
   mkdir -p /tmp/09i-resnet101ibn-arcface-weights
   cp /tmp/09i_output/resnet101ibn_arcface_best.pth /tmp/09i-resnet101ibn-arcface-weights/
   
   # Create dataset-metadata.json
   cat > /tmp/09i-resnet101ibn-arcface-weights/dataset-metadata.json << 'EOF'
   {
     "title": "09i ResNet101-IBN ArcFace Weights",
     "id": "gumfreddy/09i-resnet101ibn-arcface-weights",
     "licenses": [{"name": "CC0-1.0"}]
   }
   EOF
   
   kaggle datasets create -p /tmp/09i-resnet101ibn-arcface-weights
   ```

3. **Alternative**: Upload to existing `gumfreddy/mtmc-weights` dataset:
   ```bash
   # Add to existing weights dataset (simpler, no new dependency)
   kaggle datasets version -p /path/to/mtmc-weights -m "Add 09i ResNet101-IBN ArcFace weights"
   ```

   **Recommendation**: Use the existing `gumfreddy/mtmc-weights` dataset to avoid adding a new dependency to 10a's kernel-metadata.json. Just add the `.pth` file to the existing weights collection.

### Phase 3: 10a Notebook Changes (Feature Extraction)

**File**: `notebooks/kaggle/10a_stages012/mtmc-10a-stages-0-2-tracking-reid-features.ipynb`

**Changes needed**: Minimal — the pipeline code already supports multi-model extraction.

1. **Model weight mounting** (Cell 8 or equivalent):
   Add a copy command for the secondary model weights:
   ```python
   # Copy secondary model weights (ResNet101-IBN-a ArcFace)
   sec_weights_src = Path("/kaggle/input/mtmc-weights/resnet101ibn_arcface_best.pth")
   sec_weights_dst = Path("models/reid/resnet101ibn_cityflowv2_384px_best.pth")
   if sec_weights_src.exists():
       shutil.copy2(sec_weights_src, sec_weights_dst)
       print(f"Secondary model weights: {sec_weights_dst}")
   ```

2. **Config overrides** (Cell running the pipeline):
   Add these overrides to enable secondary model extraction:
   ```python
   overrides = [
       # ... existing overrides ...
       "stage2.reid.vehicle2.enabled=true",
       "stage2.reid.vehicle2.model_name=resnet101_ibn_a",
       "stage2.reid.vehicle2.weights_path=models/reid/resnet101ibn_cityflowv2_384px_best.pth",
       "stage2.reid.vehicle2.embedding_dim=2048",
       "stage2.reid.vehicle2.input_size=[384,384]",
       "stage2.reid.vehicle2.save_separate=true",
   ]
   ```

3. **VRAM budget check**:
   - Primary ViT-B/16 @ 256px, batch=64, fp16: ~4GB VRAM
   - Secondary ResNet101-IBN @ 384px, batch=64, fp16: ~6GB VRAM
   - Total peak: ~10GB (models are loaded/run sequentially, NOT concurrently)
   - **Kaggle T4/P100 (16GB)**: Fits comfortably with ~6GB headroom
   - **Note**: Stage 2 processes one model at a time per tracklet. Both models are loaded in memory but inference is sequential per crop batch.

4. **Runtime estimate**:
   - Current 10a runtime: ~110 min (stages 0-2 on P100)
   - Additional secondary extraction: ~20-30 min (ResNet is faster than ViT per crop, but 384px input is larger)
   - **Total estimated**: ~130-140 min (well within Kaggle's 12hr limit)

5. **Output includes**:
   - `embeddings.npy` (N, 384) — primary, PCA-whitened, L2-normed
   - `embeddings_secondary.npy` (N, 384) — secondary, PCA-whitened, L2-normed
   - Both included in `checkpoint.tar.gz` for downstream stages

### Phase 4: 10c Notebook Changes (Association + Evaluation)

**File**: `notebooks/kaggle/10c_stages45/mtmc-10c-stages-4-5-association-eval.ipynb`

**Changes needed**: Config overrides only — stage 4 code already supports score-level fusion.

1. **Config overrides** for ensemble evaluation:
   ```python
   overrides = [
       # ... existing overrides ...
       "stage4.association.secondary_embeddings.path=data/outputs/run_latest/stage2/embeddings_secondary.npy",
       "stage4.association.secondary_embeddings.weight=0.3",
   ]
   ```

2. **Weight sweep** (add to the existing sweep logic in 10c):
   ```python
   ENSEMBLE_WEIGHTS = [0.0, 0.1, 0.2, 0.3, 0.4]
   for w_sec in ENSEMBLE_WEIGHTS:
       overrides.append(f"stage4.association.secondary_embeddings.weight={w_sec}")
       # ... run stage 4+5 ...
   ```

### Phase 5: Validation & Tuning

1. **Baseline comparison**: Run with `weight=0.0` (primary only) as control
2. **Weight sweep**: Test 0.1, 0.2, 0.3, 0.4 secondary weight
3. **FIC regularisation re-tune**: The optimal FIC reg may shift from 0.1 to a different value when blending two similarity streams
4. **Threshold re-tune**: Similarity threshold (currently 0.50) may need adjustment since blended similarities have a different distribution

## 5. Code Changes Summary

### No Code Changes Needed (Already Implemented)

The following components are **already fully implemented** and require zero code changes:

| Component | File | Status |
|-----------|------|--------|
| Multi-model loading | `src/stage2_features/pipeline.py` L222-270 | ✅ Supports vehicle2 + vehicle3 |
| Separate feature extraction | `src/stage2_features/pipeline.py` L300-460 | ✅ Extracts per model |
| Per-model PCA whitening | `src/stage2_features/pipeline.py` L568-670 | ✅ Separate PCA per model |
| Per-model camera BN | `src/stage2_features/pipeline.py` L575-585 | ✅ Applied per model |
| Separate embedding save | `src/stage2_features/pipeline.py` L700-720 | ✅ `embeddings_secondary.npy` |
| Score-level fusion loading | `src/stage4_association/pipeline.py` L180-230 | ✅ Loads + L2-norms + FIC |
| Similarity blending | `src/stage4_association/pipeline.py` L507-522 | ✅ Weighted average |
| Config schema (default.yaml) | `configs/default.yaml` L83-97, L161-167 | ✅ vehicle2/vehicle3 + fusion weights |
| Config schema (cityflowv2.yaml) | `configs/datasets/cityflowv2.yaml` L114-135 | ✅ vehicle2 enabled + paths |

### Changes Required (Config + Notebooks Only)

| Change | File | Description |
|--------|------|-------------|
| Model weight path | `configs/datasets/cityflowv2.yaml` L117 | Update `weights_path` if 09i output filename differs |
| 10a model mounting | 10a notebook (model copy cell) | Add `shutil.copy2` for secondary weights |
| 10a config overrides | 10a notebook (pipeline run cell) | Add `stage2.reid.vehicle2.enabled=true` |
| 10c weight sweep | 10c notebook (sweep cell) | Add `secondary_embeddings.weight` to sweep grid |
| Kaggle weights dataset | `gumfreddy/mtmc-weights` | Upload 09i `resnet101ibn_arcface_best.pth` |

## 6. Expected Ensemble IDF1 Gain

### Literature Reference

| Team | Ensemble | Single-Model mAP | Ensemble IDF1 | Gain vs Single |
|------|:--------:|:-----------------:|:-------------:|:--------------:|
| AIC22 1st | 5 models | ~75-82% | 84.86% | +3-5pp estimated |
| AIC22 2nd | 3 models (2×ResNet101-IBN + ResNeXt101-IBN) | ~72-78% | 84.37% | +3-4pp estimated |
| AIC21 1st | 3 models | ~70-75% | 82.5% | +2-3pp estimated |

### Our Projection

**Assumptions**:
- Primary: ViT-B/16 CLIP 256px, mAP=80.14%
- Secondary: ResNet101-IBN-a ArcFace, mAP=65-70% (post-09i training)
- Score-level fusion at optimal weight

**Conservative estimate** (secondary at 65% mAP, 2-model ensemble):
- Expected gain: **+0.5 to +1.5pp MTMC IDF1**
- Projected range: **0.780 - 0.790**
- Reasoning: 15pp mAP gap is narrower than the failed 28pp gap, but still significant. Models share the same training data (CityFlowV2), limiting diversity.

**Optimistic estimate** (secondary at 70%+ mAP, 2-model ensemble):
- Expected gain: **+1.5 to +3.0pp MTMC IDF1**
- Projected range: **0.790 - 0.805**
- Reasoning: 10pp mAP gap is within the range where AIC22 teams saw meaningful ensemble benefit. A ResNet backbone is architecturally different from ViT, providing genuine feature diversity.

**Three-model estimate** (if 09i succeeds and we train a tertiary ResNeXt101-IBN):
- Expected gain: **+2.0 to +4.0pp MTMC IDF1**
- Projected range: **0.795 - 0.815**
- Reasoning: Diminishing returns per additional model, but AIC22 teams consistently gained from 3 models. The diversity between ViT, ResNet, and ResNeXt covers different failure modes.

### Key Uncertainty
The projection assumes that the mAP gap (not diversity) was the sole reason the previous ensemble attempt failed. If architectural diversity between ViT and ResNet is insufficient (e.g., both models fail on the same hard cases), gains will be at the low end. The AIC22 teams mitigated this by using multiple ResNet variants with different training recipes (DMT, different augmentations, different losses).

## 7. Risks and Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|:----------:|:------:|------------|
| 09i training fails to reach 65% mAP | Medium | Blocks everything | Try: (1) longer training, (2) label smoothing, (3) VeRi-776 pretrain→CityFlowV2 fine-tune with ArcFace, (4) different margin (m=0.35 or m=0.65) |
| 10a runtime exceeds 12hr Kaggle limit | Low | Delays pipeline | Reduce batch_size for secondary, or split into separate runs |
| VRAM OOM on T4/P100 | Low | Stalls extraction | Load models sequentially (already the default); reduce batch_size to 32 for secondary |
| Optimal blending weight varies per camera pair | Medium | Suboptimal global weight | Could implement per-camera-pair learned weights, but adds complexity; start with global weight |
| Feature spaces are too correlated (both trained on same data) | Medium | Limited diversity benefit | Train models with different augmentation stacks, or use VeRi-776 pretrain for secondary to inject external knowledge |
| FIC regularisation needs re-tuning for ensemble | High | Suboptimal whitening | Include FIC reg in the stage 4 sweep grid when testing ensemble weights |

### Fallback Plan

If 09i ArcFace training fails to reach 65% mAP:

1. **VeRi-776 pretrain path**: Train ResNet101-IBN on VeRi-776 (776 IDs, different distribution) → fine-tune on CityFlowV2 with ArcFace. The VeRi-776 pretrain already reached 62.52% mAP on VeRi-776 test set, providing a stronger initialization.

2. **Different backbone**: Try ResNeXt101-IBN-a instead of ResNet101-IBN-a. ResNeXt has group convolutions that may capture different feature patterns.

3. **Alternative loss**: If ArcFace causes training instability, fall back to standard CE+LS+Triplet (the recipe that achieved 52.77% mAP) and accept the weaker secondary.

4. **Pivot to center loss on primary**: Center loss for the primary ViT training has never been attempted and could improve the primary model's mAP, indirectly benefiting ensemble quality.

## 8. Implementation Priority & Timeline

| Step | Dependency | Blocking? | Description |
|:----:|:----------:|:---------:|-------------|
| 1 | None | YES | Push 09i notebook and start training |
| 2 | 09i success (≥65% mAP) | YES | Download weights, upload to mtmc-weights dataset |
| 3 | Step 2 | YES | Update 10a notebook with secondary model mounting + config |
| 4 | Step 3 | NO | Push 10a, run stages 0-2 with ensemble extraction |
| 5 | Step 4 → 10b | NO | Run 10b (stage 3, indexing) |
| 6 | Step 5 → 10c | NO | Run 10c with weight sweep (0.0 to 0.4) |
| 7 | Step 6 results | NO | Analyze results, re-tune FIC reg + threshold if needed |
| 8 | Step 7 (if gains > 1pp) | NO | Consider training tertiary ResNeXt101-IBN |

## 9. Config Reference

### Enabling Ensemble in cityflowv2.yaml (already present, just needs `enabled: true`)
```yaml
stage2:
  reid:
    vehicle2:
      enabled: true                # <-- flip this
      save_separate: true
      model_name: "resnet101_ibn_a"
      weights_path: "models/reid/resnet101ibn_cityflowv2_384px_best.pth"
      embedding_dim: 2048
      input_size: [384, 384]
      clip_normalization: false
```

### Stage 4 Fusion Config
```yaml
stage4:
  association:
    secondary_embeddings:
      path: "data/outputs/run_latest/stage2/embeddings_secondary.npy"
      weight: 0.3    # sweep 0.1-0.4
    tertiary_embeddings:
      path: ""
      weight: 0.0
```

### Full Override String for 10c Sweep
```
stage4.association.secondary_embeddings.path=data/outputs/run_latest/stage2/embeddings_secondary.npy
stage4.association.secondary_embeddings.weight=0.3
stage4.association.fic.regularisation=0.1
stage4.association.graph.similarity_threshold=0.50
```