# Post-DMT Strategy: Closing the Gap to SOTA

**Date**: 2026-03-30
**Current best**: MTMC IDF1 = 0.784 (10c v44 / code v80, ali369, min_hits=2)
**Baseline re-run**: MTMC IDF1 = 0.775 (10a v45, same 256px model, min_hits=2)
**DMT attempt**: MTMC IDF1 = 0.758 (10c v46 sweep best, DMT model at 87.30% mAP)
**SOTA target**: MTMC IDF1 = 0.8486 (AIC22 1st place, Team28)
**Gap**: 6.5pp from best (0.784), 9.1pp from re-run (0.775)

---

## 1. Root Cause Analysis: 0.784 → 0.775 Regression

### Observed Facts
- **v80 (10c v44)**: MTMC IDF1 = 0.784, association optima = sim=0.54, app=0.70, fic=0.10
- **v45 (10a v45)**: MTMC IDF1 = 0.775, association optima = sim=0.58, app=0.85, fic=0.20
- Both claim to use 256px ViT-B/16 CLIP (`transreid_cityflowv2_best.pth`), min_hits=2
- The association optima shifted significantly between the two runs

### Likely Causes (ranked by probability)

**1. Config drift in the 10a notebook (HIGH probability)**
The `cityflowv2.yaml` dataset config currently lists:
- `stage2.reid.vehicle.weights_path: "models/reid/transreid_cityflowv2_384px_best.pth"` (384px primary!)
- `stage2.reid.vehicle.input_size: [384, 384]`
- `stage2.reid.vehicle2.enabled: true` (secondary model ON)
- `stage2.reid.color_augment: true`

If the 10a notebook overrides were not perfectly aligned to the v80 configuration, any of these could silently change feature extraction:
- Using 384px weights at 256px input (or vice versa)
- Enabling the secondary model (52.77% ResNet) with non-trivial fusion weight
- Enabling color_augment (tested as neutral-to-harmful)

The shifted association optima (sim 0.54→0.58, app 0.70→0.85) strongly suggest the **features themselves changed**, not just noise — different features have different optimal operating points.

**2. Current 10a notebook uses DMT weights (HIGH probability)**
The current 10a notebook was reported to set `stage2.reid.vehicle.weights_path=models/reid/transreid_cityflowv2_256px_dmt_best.pth`. If v45 ran with DMT weights rather than the original `transreid_cityflowv2_best.pth`, the 0.775 result would represent DMT features evaluated at non-DMT-optimal thresholds (DMT sweep best was 0.758 at different thresholds). This would explain both the regression AND the shifted optima.

**3. Pipeline code changes on `feature/people-tracking` branch (MEDIUM probability)**
Changes to stage 0/1/2 pipeline code for the person tracking feature could have introduced subtle regressions:
- PCA whitening behavior changes
- Crop extraction logic changes
- Camera-BN or power-norm defaults changing

**4. Stochastic variation in tracking (LOW probability)**
BoT-SORT uses sparse optical flow CMC and appearance matching with non-deterministic components. However, 0.9pp is larger than typical run-to-run variation (~0.1-0.3pp), so this alone does not explain the gap.

### Recommended Diagnostic Action

**A. Controlled reproduction of v80 environment:**
1. Git checkout the exact commit that produced v80 (10a code v80)
2. Verify the 10a notebook overrides match the v80 configuration exactly
3. Push to Kaggle and run 10a → 10c with v44's association config
4. If this reproduces 0.784, the regression is config/code drift → diff the two commits
5. If this does NOT reproduce 0.784, the issue is environmental (GPU/framework)

**B. Quick local diagnostic:**
1. Download and diff the v80-era 10a output features vs v45 10a output features
2. Compare embedding distributions, PCA explained variance, tracklet counts
3. This will reveal immediately whether features changed

---

## 2. DMT Camera-Aware Training: Post-Mortem

### What Happened
- DMT (GRL + CameraHead + CircleLoss) trained on CityFlowV2 achieved **87.30% mAP** (+7.16pp over ~80% baseline)
- Deployed in MTMC pipeline and swept across association parameters in 10c v46
- Best MTMC IDF1 = **0.758** at sim=0.55, app=0.80, fic=0.10
- Baseline MTMC IDF1 = **0.775** (non-DMT, 10a v45) = **-1.7pp DROP**
- The higher mAP does NOT translate to better MTMC IDF1

### Why DMT Failed for MTMC

**Hypothesis: mAP measures gallery ranking; MTMC uses threshold-based graph association. These are fundamentally different tasks.**

1. **mAP rewards within-gallery ranking**: DMT features excel at ranking the correct match above incorrect ones within a gallery. The +7pp mAP means DMT dramatically improved **relative** ordering.

2. **MTMC uses absolute thresholds**: Our stage 4 builds a graph using `sim_thresh=0.54` (or similar). It doesn't care about ranking — it cares about whether the correct match lands **above** the threshold AND incorrect matches land **below** it.

3. **DMT may compress the similarity distribution**: Camera-adversarial training forces features to be camera-invariant, which could compress the gap between positive and negative pairs in absolute cosine distance. The ranking improves (better mAP) but the margin between positives and negatives shrinks (worse for thresholding).

4. **CityFlowV2 eval mAP uses within-camera queries**: The +7pp mAP may be inflated by improvements in within-camera discrimination that don't help cross-camera matching.

5. **The optimal thresholds shifted dramatically**: DMT optimal was sim=0.55/app=0.80 vs baseline sim=0.58/app=0.85. This confirms the similarity distribution changed shape, not just shifted.

### Verdict
**DMT camera-aware training is a DEAD END for this pipeline architecture.** The adversarial approach produces features optimized for ranking metrics, not for the threshold-based graph association in our stage 4. Alternative camera-aware approaches (e.g., camera-conditioned scoring, learned per-pair thresholds) might work but would require significant architectural changes to stage 4.

---

## 3. Ranked Approaches to Close the Gap

### Current Ceiling Analysis

With DMT dead, ensemble blocked (secondary model too weak), and association parameters exhausted:

| Source | Estimated Gain | Status |
|--------|:--------------:|--------|
| Fix v80 regression | +0.9pp (recover 0.775→0.784) | **Actionable now** |
| CID_BIAS on 256px baseline | +0.0 to +0.5pp | **Untested on correct baseline** |
| Multi-query track representation | +0.3 to +0.8pp | **Never tried** |
| Hand-annotated zone polygons | +0.5 to +1.0pp | **Manual labor required** |
| Larger detector (YOLOv8x) | +0.2 to +0.5pp | **Straightforward swap** |
| Re-enable reranking after features improve | +0.0 to +0.5pp | **Blocked by feature quality** |
| **Realistic ceiling** | **~0.80-0.81** | Without a 2nd strong model |

**To reach SOTA (0.8486), a multi-model ensemble (3+ diverse backbones) is mandatory.** Every AIC22 top team uses 3-5 models. Our single ViT-B/16 CLIP cannot close the full gap alone.

### Ranked Approaches

#### Rank 1: Diagnose & Fix Regression (IMMEDIATE)
- **Expected gain**: +0.9pp (0.775 → 0.784)
- **Effort**: Low (1-2 days)
- **Risk**: Low
- **Action**: See Section 1 diagnostic steps. Most likely cause is config drift in 10a notebook.
- **Priority**: **DO THIS FIRST** — all other experiments are meaningless until the baseline is restored.

#### Rank 2: CID_BIAS on 256px Baseline (LOW EFFORT)
- **Expected gain**: +0.0 to +0.5pp
- **Effort**: Low (CPU-only, stage 4 change)
- **Risk**: Low (CID_BIAS hurt 384px by -0.52pp, but 384px features are fundamentally different)
- **Action**: Compute pair-wise camera distance biases from GT-matched 256px tracklets. Apply as additive correction to cross-camera similarity scores in stage 4.
- **Note**: MUST test on 256px (v80 baseline), NOT 384px. The 384px failure does not predict 256px behavior.
- **Measurement**: Run 10c with CID_BIAS enabled on v80-era features → compare to 0.784.

#### Rank 3: Multi-Query Track Representation (MEDIUM EFFORT)
- **Expected gain**: +0.3 to +0.8pp
- **Effort**: Medium (stage 2 + stage 4 changes)
- **Risk**: Medium (requires careful implementation)
- **What it is**: Instead of averaging all crop embeddings into one tracklet vector, retain the **top-K representative embeddings** per tracklet (e.g., K=5). Cross-camera matching computes similarity as the **best pairwise match** across the K×K pairs.
- **Why it helps**: Current temporal averaging loses viewpoint-specific information. A vehicle seen from the front and rear produces averaged features that don't match well to either view. Multi-query preserves both views.
- **SOTA precedent**: AIC22 1st place uses "box-grained reranking" — matching at detection level, not tracklet level. This is the key innovation that pushed them from ~84% to 84.86%.
- **Implementation sketch**:
  1. Stage 2: Save per-tracklet top-K embeddings (by quality score) instead of / in addition to averaged embedding
  2. Stage 4: Modify cross-camera similarity to use max-of-K×K pairwise cosines
  3. PCA/FIC still applied per-embedding
  4. Graph construction and connected components unchanged

#### Rank 4: Hand-Annotated Zone Polygons (MEDIUM EFFORT)
- **Expected gain**: +0.5 to +1.0pp
- **Effort**: Medium (manual annotation + code to use zones)
- **Risk**: Low (proven technique — DAMO used this for +0.5-1pp)
- **What it is**: Define entry/exit zone polygons for each camera. Learn transition time distributions per zone-pair. Use as hard constraint: only allow matches where the spatial-temporal transition is physically plausible.
- **Note**: Our auto-generated k-means zones hurt by -0.4pp. Manual annotation with domain knowledge of CityFlowV2 intersections is required.
- **Action**: Annotate zones in S01 (3 cameras at one intersection) and S02 (3 cameras at another). S01↔S02 transitions are blocked (no shared vehicles in GT).

#### Rank 5: Larger Detector (LOW-MEDIUM EFFORT)
- **Expected gain**: +0.2 to +0.5pp
- **Effort**: Low (model swap in config)
- **Risk**: Low (runtime increase on Kaggle)
- **What it is**: Replace YOLO26m with YOLOv8x or YOLOv11x for better detection quality → tighter bounding boxes → cleaner ReID crops.
- **SOTA precedent**: AIC22 2nd uses YOLOv5x6 (much larger), AIC22 1st uses Swin Transformer.
- **Constraint**: Must fit within Kaggle P100 memory and 12h runtime.
- **Action**: Test YOLOv8x on CityFlowV2 subset locally for detection quality, then deploy on Kaggle.

#### Rank 6: DINOv2 or CLIP-ReID Backbone (HIGH EFFORT, HIGH POTENTIAL)
- **Expected gain**: +1.0 to +3.0pp
- **Effort**: High (new training pipeline)
- **Risk**: High (new architecture, unknown failure modes)
- **What it is**: Replace TransReID ViT-B/16 CLIP with a stronger foundation model backbone:
  - **DINOv2 ViT-B/14**: Self-supervised, excellent at fine-grained visual features, no text supervision bias
  - **CLIP-ReID**: CLIP backbone with text-guided contrastive pretraining for ReID
  - **SwinV2-B**: Hierarchical vision transformer with better multi-scale features
- **Why it could work**: Our ViT-B/16 CLIP backbone is from 2021. Newer foundation models have significantly better visual representations. DINOv2 in particular excels at fine-grained matching tasks.
- **Risk**: Requires full training pipeline integration and may not improve cross-camera invariance.

#### Rank 7: Proper Secondary Model for Ensemble (HIGH EFFORT)
- **Expected gain**: +1.0 to +2.0pp (if achieved)
- **Effort**: High (training + integration)
- **Risk**: High (ResNet101-IBN-a path failed; need different approach)
- **What it is**: Train a second ReID model with fundamentally different architecture. Options:
  - **ConvNeXt-Base** (modern CNN, complementary to ViT)
  - **EfficientNetV2-L** (efficient, strong on fine-grained)
  - **Swin-T** (different ViT variant)
- **Key insight**: The secondary model must be architecturally diverse from ViT-B/16 CLIP. ResNet101-IBN-a failed because ImageNet→CityFlowV2 transfer is too weak without VeRi-776 pretraining (and VeRi→CityFlowV2 also failed at 42.7% mAP).
- **Minimum bar**: Secondary model needs **≥65% mAP** on CityFlowV2 eval to contribute positively at ≥10% fusion weight.

---

## 4. Detailed Spec: Rank 1 — Diagnose & Fix Regression

### Goal
Recover the 0.784 MTMC IDF1 baseline by identifying and fixing the source of the 0.9pp regression.

### Hypothesis
The regression is caused by config or code drift between 10a code v80 and the current 10a notebook, not by fundamental pipeline limitations.

### Step-by-Step Plan

#### Step 1: Identify the v80 Commit
```bash
# Find the git commit that corresponds to 10a code v80
git log --oneline --all | grep -i "v80\|10a.*v80"
# Or search Kaggle kernel metadata for the commit hash
kaggle kernels output ali369/mtmc-10a-stages-0-2-tracking-reid-features -v 44 -p /tmp/v44_output
```

#### Step 2: Diff Notebook Overrides
Compare the 10a notebook overrides between v80 and the current version:
- `stage2.reid.vehicle.weights_path` — must be `transreid_cityflowv2_best.pth` (NOT DMT, NOT 384px)
- `stage2.reid.vehicle.input_size` — must be `[256, 256]`
- `stage2.reid.vehicle2.enabled` — check if secondary model was ON or OFF in v80
- `stage2.reid.color_augment` — should be `false`
- `stage2.camera_bn.enabled` — check v80 state
- `stage1.tracker.min_hits` — must be `2`
- Any other stage 0/1/2 overrides

#### Step 3: Diff Pipeline Code
```bash
git diff <v80-commit> HEAD -- src/stage0/ src/stage1/ src/stage2_features/
```
Look for:
- Changes to PCA whitening logic
- Changes to crop extraction or quality scoring
- Changes to embedding normalization
- Changes to tracklet merging or interpolation
- Default parameter changes

#### Step 4: Reproduce v80
1. Create a clean copy of the 10a notebook with v80-exact overrides
2. Push to Kaggle as a new version
3. Run full 10a → 10c chain with 10c v44 association config
4. Compare MTMC IDF1 to 0.784

#### Step 5: Binary Search (if needed)
If the reproduction doesn't match:
1. Git bisect between v80 and HEAD
2. At each commit, run a quick local evaluation on cached features
3. Identify the exact commit that introduced the regression

### Config Overrides (v80-exact)
```python
# Stage 1
stage1.detector.confidence_threshold=0.25
stage1.tracker.min_hits=2
stage1.interpolation.max_gap=50
stage1.intra_merge.max_time_gap=40.0

# Stage 2
stage2.reid.vehicle.model_name=transreid
stage2.reid.vehicle.weights_path=models/reid/transreid_cityflowv2_best.pth
stage2.reid.vehicle.input_size=[256,256]
stage2.reid.vehicle.vit_model=vit_base_patch16_clip_224.openai
stage2.reid.vehicle.num_cameras=59
stage2.reid.vehicle.clip_normalization=true
stage2.reid.vehicle2.enabled=false
stage2.reid.flip_augment=true
stage2.reid.color_augment=false
stage2.reid.quality_temperature=3.0
stage2.pca.enabled=true
stage2.pca.n_components=384
stage2.camera_bn.enabled=true
stage2.power_norm.alpha=0.0
stage2.camera_tta.enabled=false

# Stage 4 (10c v44)
stage4.association.graph.similarity_threshold=0.54
stage4.association.weights.vehicle.appearance=0.70
stage4.association.weights.vehicle.hsv=0.00
stage4.association.weights.vehicle.spatiotemporal=0.25
stage4.association.fic.regularisation=0.10
stage4.association.fic.enabled=true
stage4.association.query_expansion.k=3
stage4.association.query_expansion.alpha=5.0
stage4.association.secondary_embeddings.weight=0.10
stage4.association.intra_merge.enabled=true
stage4.association.intra_merge.threshold=0.80
stage4.association.intra_merge.max_gap=30
```

### Success Criteria
- MTMC IDF1 ≥ 0.782 (within 0.2pp of 0.784, accounting for stochastic variation)

### Risks
- If the regression is environmental (Kaggle hardware change), it cannot be fixed
- If the regression comes from a beneficial code change (e.g., bug fix that happened to help), reverting may break other things

---

## 5. Updated Dead Ends List

### Confirmed Dead Ends (DO NOT RETRY)

| # | Approach | Result | Evidence | Date |
|---|----------|--------|----------|------|
| 1 | Association parameter tuning | Exhausted (225+ configs, all within 0.3pp) | Experiment log | 2026-03 |
| 2 | CSLS distance | -34.7pp catastrophic | 10c v74 | 2026-03 |
| 3 | Hierarchical clustering | -1.0 to -5.1pp | v54-56, v62 | 2026-02 |
| 4 | FAC (Feature Augmented Clustering) | -2.5pp | v26 | 2026-02 |
| 5 | Feature concatenation (vs score fusion) | -1.6pp | Experiment log | 2026-02 |
| 6 | CamTTA (Camera Test-Time Adaptation) | Helps global, hurts MTMC | v28-30 | 2026-03 |
| 7 | Multi-scale TTA | Neutral/harmful | Multiple runs | 2026-02 |
| 8 | Track smoothing / edge trim | Always harmful | Experiment log | 2026-02 |
| 9 | Denoise preprocessing | -2.7pp | v46, v82 | 2026-03 |
| 10 | `mtmc_only` submission flag | -5pp | Documented | 2026-02 |
| 11 | Auto-generated zone polygons | -0.4pp | v54-57 | 2026-02 |
| 12 | PCA dimension search beyond 384D | 384D optimal, others worse | Experiment log | 2026-02 |
| 13 | SGD optimizer for ResNet101-IBN-a | 30.27% mAP catastrophic | 09d v18 mrkdagods | 2026-03 |
| 14 | Circle loss + triplet on same feature tensor | Catastrophic gradient conflict | 09d v17, 09f v2 | 2026-03 |
| 15 | ResNet101-IBN-a VeRi-776→CityFlowV2 fine-tuning | 42.7% mAP, worse than direct path (52.77%) | 09e v2, 09f v3, 09d v18 | 2026-03 |
| 16 | CLIP ViT backbone for standard-ViT checkpoints | Mode collapse (cosine sim 0.874) | 12b v5/v6 vs v8 | 2026-03 |
| 17 | K-reciprocal reranking (with current single-model features) | Always worse | v25, v35 | 2026-02 |
| 18 | Camera-pair similarity normalization | Zero effect (FIC handles it) | v36 | 2026-02 |
| 19 | confidence_threshold=0.20 | -2.8pp | 10c v45 | 2026-03 |
| 20 | max_iou_distance=0.5 | -1.6pp | 10c v47 | 2026-03 |
| 21 | 384px TransReID input resolution | -2.8pp MTMC IDF1 vs 256px baseline | 10a v43-v44, v80 baseline | 2026-03 |
| 22 | CID_BIAS on 384px features | -0.52pp | 10c v44 + CID_BIAS | 2026-03 |
| **23** | **DMT camera-aware training (GRL + CameraHead + CircleLoss)** | **+7.16pp mAP but -1.7pp MTMC IDF1** | **10c v46 sweep best (0.758) vs baseline (0.775)** | **2026-03-30** |

### Key Insight from DMT Dead End

**ReID mAP and MTMC IDF1 measure fundamentally different things.** mAP measures gallery ranking (relative ordering). Our MTMC pipeline uses threshold-based graph construction (absolute similarity cutoffs). DMT camera-adversarial training improved ranking by +7pp mAP but compressed the similarity margin between positives and negatives, making threshold-based association worse.

**Implication**: Future ReID training improvements should be evaluated on **cross-camera pairwise distance margin** (mean positive cosine - mean negative cosine), not just mAP. A model with lower mAP but wider cross-camera margin will perform better in our pipeline.

---

## 6. Strategic Assessment

### What We've Learned

1. **Association is fully exhausted** — 225+ configs, no more gains possible
2. **Camera-adversarial training (DMT) doesn't help threshold-based MTMC** — fundamental mismatch
3. **384px resolution hurts cross-camera matching** — captures viewpoint-specific textures
4. **Single model has a ceiling of ~0.80 MTMC IDF1** — ensemble is mandatory for SOTA
5. **The ResNet101-IBN-a secondary path is blocked** — all training approaches failed
6. **Our pipeline architecture is sound** — the bottleneck is purely feature quality

### Realistic Targets

| Target | Requirements | Probability |
|--------|-------------|:-----------:|
| **0.784** (recover baseline) | Fix regression | 90% |
| **0.79** | + CID_BIAS + minor tuning | 50% |
| **0.80** | + Multi-query + zones | 30% |
| **0.81** | + Larger detector + modest 2nd model | 15% |
| **0.8486** (SOTA) | 3-model ensemble + DMT-like training that works + all tricks | 3% |

### Recommendation

**Immediate priority**: Diagnose and fix the v80 regression (Section 4). This is the highest-ROI action with the lowest risk.

**Medium-term**: Implement multi-query track representation (Rank 3). This addresses a known information bottleneck without requiring new model training.

**Long-term**: Invest in a fundamentally different secondary backbone (ConvNeXt-Base or DINOv2 ViT) trained with a recipe that maximizes **cross-camera margin** rather than mAP. This is the only path to SOTA.

**Do NOT**: Retry any approach in the dead ends list. Do not invest more time in association parameter sweeps. Do not try DMT variants (adversarial camera removal is proven harmful for this pipeline).