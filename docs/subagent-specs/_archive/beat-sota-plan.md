# Strategic Plan: Beat SOTA on Both Vehicle and Person Pipelines

> **Created**: 2026-04-01
> **Account**: gumfreddy (~16h GPU remaining this week, resets Wednesday)
> **Constraint**: T4 machines only; 12b is CPU-only

---

## Executive Summary

| Pipeline | Current | SOTA | Gap | Path to Close |
|----------|:-------:|:----:|:---:|---------------|
| **Vehicle** (CityFlowV2 MTMC IDF1) | 77.5% | 84.86% | 7.36pp | Multi-model ensemble is the ONLY viable path; all single-model and association approaches exhausted |
| **Person** (WILDTRACK IDF1) | 94.7% | 95.3% | 0.6pp | Run 12b tracking on improved 12a v3 detections (MODA 92.1%); near-term closeable |

---

## Part 1: Person Pipeline — Immediate Wins

### Action 1A: Run 12b on 12a v3 Detections (Priority: NOW)

**Goal**: Feed the best-ever detector output (MODA=92.1%, epoch 20/25 from 12a v3 gumfreddy) into the tuned Kalman tracker from 12b v14.

**Hypothesis**: The previous 12b v14 result (IDF1=94.7%) used 12a v26 detections (MODA=90.9%). The new 12a v3 detections have +1.2pp MODA and +1.5pp recall (96.6% vs ~95.1%). Better detections should directly improve tracking IDF1 since the tracker is already well-tuned and the error profile is detection-dominated.

**Expected Impact**: +0.3-0.6pp IDF1, potentially closing the gap to SOTA (95.3%).

**Changes Required**:

1. **12b kernel-metadata.json** — Update upstream reference:
   ```json
   {
     "id": "gumfreddy/12b-wildtrack-mvdetr-tracking-reid",
     "title": "12b WILDTRACK MVDeTr Tracking ReID",
     "code_file": "12b_wildtrack_tracking_reid.ipynb",
     "language": "python",
     "kernel_type": "notebook",
     "is_private": true,
     "enable_gpu": false,
     "enable_tpu": false,
     "enable_internet": true,
     "keywords": [],
     "dataset_sources": [
       "gumfreddy/mtmc-weights",
       "aryashah2k/large-scale-multicamera-detection-dataset"
     ],
     "kernel_sources": [
       "gumfreddy/12a-wildtrack-mvdetr-training"
     ],
     "competition_sources": [],
     "model_sources": []
   }
   ```
   **Key changes**: `id` → gumfreddy, removed `machine_shape`, `kernel_sources` → gumfreddy's 12a.

2. **12b notebook** — Update the upstream kernel slug reference in the markdown header and bootstrap cell:
   - Change `ali369/12a-wildtrack-mvdetr-training` → `gumfreddy/12a-wildtrack-mvdetr-training`
   - Change `mrkdagods/mtmc-weights` → `gumfreddy/mtmc-weights` (if the notebook code references the old slug)

3. **Tracking parameters** — Keep the 12b v14 tuned Kalman config (proven best):
   - `max_age=2`, `min_hits=2`, `distance_gate=20cm`, `q_std=8`, `r_std=8`
   - Interpolation `gap=2`
   - No ReID merge (was tested and didn't help with clean tracks)

**Runtime**: CPU-only, ~20-30 min. **GPU cost: 0h.**

**Push command**:
```bash
kaggle kernels push -p notebooks/kaggle/12b_wildtrack_tracking_reid/
```

**Measurement**:
- Success: IDF1 ≥ 95.0% (within striking distance of SOTA 95.3%)
- Stretch: IDF1 ≥ 95.3% (matches SOTA)

---

### Action 1B: Tracker Parameter Tuning on New Detections (Contingency)

**Goal**: If 1A doesn't fully close the gap, micro-tune the Kalman parameters on the new detection distribution.

**Hypothesis**: The optimal Kalman gate parameters may shift slightly with better detections (higher recall → more candidates → may need tighter gating).

**Sweep Plan** (only if 1A leaves >0.3pp gap):
| Parameter | Current | Sweep Values |
|-----------|---------|--------------|
| `distance_gate` | 20cm | 15, 18, 22, 25 |
| `q_std` | 8 | 5, 6, 10, 12 |
| `r_std` | 8 | 5, 6, 10, 12 |
| `min_hits` | 2 | 1, 3 |
| Interpolation `gap` | 2 | 1, 3 |

Each combination runs in <1 min (CPU), so a 20-point sweep takes ~20 min total.

**Runtime**: CPU-only. **GPU cost: 0h.**

---

### Action 1C: Longer Detector Training — ResNet34 Backbone (If Gap Persists)

**Goal**: Train MVDeTr with ResNet34 backbone for 25 epochs (vs current ResNet18/10-25 epochs).

**Hypothesis**: ResNet34 has ~2x the parameters of ResNet18, providing more capacity for multi-view feature aggregation. The current training peaked at epoch 20/25, suggesting more capacity could help.

**Changes** (from the existing extended-training-09d-12a.md spec):
- 12a notebook: `--arch resnet34`, `--epochs 25`, `--lr 7e-4`
- 12a kernel-metadata: already set to `gumfreddy/12a-wildtrack-mvdetr-training`

**Runtime**: ~3-4h GPU. **GPU cost: ~4h.**

**Only pursue if 1A+1B leave >0.3pp gap.** The 12a v3 epoch-20 MODA=92.1% with ResNet18 may already be sufficient.

---

## Part 2: Vehicle Pipeline — Closing the 7.4pp Gap

### Critical Diagnosis

The findings document is unambiguous:
- **Association is exhausted** (225+ configs, all within 0.3pp of optimal)
- **Single-model improvements DON'T translate** to MTMC IDF1 (384px: -2.8pp despite same mAP; DMT: -1.4pp despite +7pp mAP)
- **Every top AIC method** used a **3-5 model ensemble**
- **Our secondary model** (ResNet101-IBN-a, 52.77% mAP) is too weak for ensemble benefit (need ≥65% mAP)
- **ResNet training path is saturated** at 52.77% (extended training degraded; VeRi-776 pretraining hurt)
- **384px, DMT, CID_BIAS, reranking, CSLS, hierarchical** — all dead ends in single-model regime

### The Only Viable Path: Multi-Model Ensemble

SOTA requires 3-5 complementary ReID models, each with >70% mAP. We currently have:
- **Primary**: TransReID ViT-B/16 CLIP 256px — 80.14% mAP ✅
- **Secondary**: ResNet101-IBN-a — 52.77% mAP ❌ (too weak)
- **Tertiary**: None

The secondary model must reach ≥65% mAP (ideally >70%) before ensemble fusion can help. Until that's achieved, the vehicle pipeline is capped at ~77-78% MTMC IDF1.

### Action 2A: Train ResNet101-IBN-a with DMT via gumfreddy (Priority: HIGH)

**Goal**: Train 09g (ResNet101-IBN-a with DMT camera-aware losses) on gumfreddy's account to produce a stronger secondary model with camera-invariant features.

**Hypothesis**: DMT (Dual-Model Training with camera-adversarial loss) explicitly trains the model to be camera-invariant, which is exactly the failure mode that killed 384px and single-model DMT — but in an **ensemble context**, a DMT-trained secondary model provides **diversity** that complements the non-DMT primary.

The key insight from the findings: DMT hurt as a single model (-1.4pp MTMC IDF1), but AIC22 2nd place used DMT + 3-model ensemble. The technique fails alone but **enables ensemble diversity**.

**Changes Required**:

1. **09g kernel-metadata.json** — Switch to gumfreddy:
   ```json
   {
     "id": "gumfreddy/09g-resnet101-ibn-a-dmt-cityflowv2",
     "title": "09g ResNet101-IBN-a DMT CityFlowV2",
     "code_file": "09g_resnet101ibn_dmt.ipynb",
     "language": "python",
     "kernel_type": "notebook",
     "is_private": true,
     "enable_gpu": true,
     "machine_shape": "NvidiaTeslaT4",
     "enable_internet": true,
     "dataset_sources": [
       "gumfreddy/mtmc-weights",
       "thanhnguyenle/data-aicity-2023-track-2"
     ],
     "kernel_sources": [],
     "competition_sources": []
   }
   ```

2. **09g notebook** — Verify the training recipe:
   - Backbone: ResNet101-IBN-a (ImageNet pretrained via IBN-Net repo)
   - Input: 384×384
   - Losses: ID (label smoothing 0.05) + Triplet (margin 0.3) — **NO circle loss** (confirmed dead end for ResNet)
   - DMT: Camera adversarial loss with GRL (λ warm-up over 15 epochs)
   - Optimizer: AdamW, lr=1e-3 (head), backbone_lr_factor=0.1
   - Epochs: 120
   - Augmentation: RandomHorizontalFlip, Pad+RandomCrop, ColorJitter, RandomErasing(p=0.5)

3. **Critical**: Ensure `circle_weight = 0.0` in the notebook. Circle loss + triplet on the same features was the #1 cause of ResNet training failures (09d v17: 29.6%, 09f v2: 16.2%).

**Runtime**: ~8-10h GPU on T4. **GPU cost: ~10h.**

**Expected mAP**: 55-65% on CityFlowV2 eval split (DMT may improve cross-camera features even if single-camera mAP doesn't jump dramatically).

**Success criterion**: mAP ≥ 60% AND ensemble MTMC IDF1 > 77.5% (any gain over current).

---

### Action 2B: Train ResNeXt101-IBN-a with DMT via gumfreddy (Priority: MEDIUM)

**Goal**: Train 09h (ResNeXt101-IBN-a with DMT) as a tertiary model for 3-model ensemble.

**Hypothesis**: ResNeXt101-IBN uses grouped convolutions (32×4d cardinality) → different feature extraction patterns than both ViT and ResNet101. AIC22 2nd place specifically used ResNeXt101-IBN-a as one of their 3 models.

**Changes Required**: Same as 2A but for 09h notebook:
1. Update `kernel-metadata.json`: `id` → `gumfreddy/09h-resnext101-ibn-a-dmt-cityflowv2`, dataset_sources → `gumfreddy/mtmc-weights`
2. Ensure `circle_weight = 0.0`
3. ResNeXt101-IBN-a can be loaded from `timm` or torch hub

**Runtime**: ~10-12h GPU on T4. **GPU cost: ~11h.**

**⚠️ Budget conflict**: 2A (10h) + 2B (11h) = 21h, but only 16h remain. Must choose one or run sequentially across weeks.

**Recommendation**: Run 2A this week (10h). If successful (mAP ≥ 60%), run 2B next week after quota reset.

---

### Action 2C: Test Ensemble of Primary + DMT ResNet101 (After 2A)

**Goal**: Deploy the 09g output into the 10a→10b→10c pipeline as `vehicle2` and evaluate ensemble MTMC IDF1.

**Changes Required**:

1. Upload 09g's `best_model.pth` to `gumfreddy/mtmc-weights` dataset as `resnet101ibn_dmt_cityflowv2_best.pth`
2. Update 10a notebook:
   ```python
   SECONDARY_WEIGHTS = "models/reid/resnet101ibn_dmt_cityflowv2_best.pth"
   ```
3. Run 10a → 10b → 10c chain with fusion sweep:
   - Sweep `FUSION_WEIGHT` over [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
   - With DMT model, lower weights may work better than 0.30 (since DMT features are designed to be camera-agnostic)

**Runtime**: 10a ~2h GPU + 10b/10c ~10min CPU. **GPU cost: ~2h.**

---

### Action 2D: Untried Approaches Worth Investigating

These are techniques **not in the dead ends list** that could provide incremental gains:

#### 2D-1: Center Loss (NOT TRIED)
The current ViT training uses only ID + Triplet loss. AIC22 winners used **ID + Triplet + Center** (not circle). Center loss pulls same-ID embeddings toward a learnable class center, reducing intra-class variance without the gradient conflict issues of circle loss.

**How to test**: Add center loss to the 09b training recipe with `center_weight=0.005` (the standard weighting). Retrain the primary ViT.

**Risk**: Requires retraining the primary model (~4h GPU), and may not improve cross-camera features.

#### 2D-2: ViT with Different Augmentation (NOT TRIED as ensemble diversity)
Train a second ViT-B/16 with **stronger augmentation** (more aggressive random erasing, autoaugment, or cutout) to create model diversity. Two ViTs trained with different augmentation strategies produce genuinely different feature representations.

**Risk**: Architecture similarity may limit ensemble benefit vs. architecturally different models.

#### 2D-3: Temporal Attention / Tracklet-Level Features (NOT TRIED)
Instead of averaging per-crop embeddings, use attention-weighted tracklet representations that emphasize the most distinctive crops. This addresses the "under-merging" problem (1.69:1 ratio) by providing stronger per-tracklet features.

**Risk**: Requires new stage-2 code; not a config change.

#### 2D-4: Contrastive Learning on Tracklet Pairs (NOVEL)
Use the known GT associations from training cameras to fine-tune embeddings contrastively at the tracklet level (not crop level). This would directly optimize for the cross-camera matching task.

**Risk**: Requires significant new code and may overfit to the 6-camera CityFlowV2 layout.

---

## Part 3: Priority Ordering

### This Week (16h GPU remaining)

| Priority | Action | GPU Hours | Dependency | Expected Impact |
|:--------:|--------|:---------:|:----------:|:---------------:|
| **1** | 1A: Run 12b on 12a v3 detections | 0h (CPU) | None | Person IDF1 +0.3-0.6pp |
| **2** | 2A: Train 09g ResNet101-IBN-a DMT | ~10h | None | Vehicle secondary model for ensemble |
| **3** | 1B: Tune 12b tracker (if needed) | 0h (CPU) | After 1A | Person IDF1 micro-gains |
| **4** | 2C: Test ensemble with 09g output | ~2h | After 2A completes | Vehicle MTMC IDF1 validation |

**Total GPU this week**: ~12h (leaves 4h buffer for reruns/fixes)

### Next Week (fresh 25h quota)

| Priority | Action | GPU Hours | Dependency | Expected Impact |
|:--------:|--------|:---------:|:----------:|:---------------:|
| **5** | 2B: Train 09h ResNeXt101-IBN-a DMT | ~11h | None | Vehicle tertiary model |
| **6** | 2C+: 3-model ensemble test | ~2h | After 2B | Vehicle MTMC IDF1 with 3 models |
| **7** | 1C: ResNet34 MVDeTr (if person gap persists) | ~4h | After 1A results | Person MODA improvement |
| **8** | 2D-1: Center loss ViT retrain | ~4h | After ensemble baseline | Primary model quality |

### Later (if budget allows)

| Priority | Action | GPU Hours | Expected Impact |
|:--------:|--------|:---------:|:---------------:|
| **9** | 2D-2: Different-augmentation ViT for 4-model ensemble | ~4h | Model diversity |
| **10** | 2D-3: Attention-weighted tracklet features | 0h (code only) | Feature quality |

---

## Part 4: Exact Notebook/Config Changes

### 12b — Person Tracking (Action 1A)

**File**: `notebooks/kaggle/12b_wildtrack_tracking_reid/kernel-metadata.json`

```json
{
  "id": "gumfreddy/12b-wildtrack-mvdetr-tracking-reid",
  "title": "12b WILDTRACK MVDeTr Tracking ReID",
  "code_file": "12b_wildtrack_tracking_reid.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": false,
  "enable_tpu": false,
  "enable_internet": true,
  "keywords": [],
  "dataset_sources": [
    "gumfreddy/mtmc-weights",
    "aryashah2k/large-scale-multicamera-detection-dataset"
  ],
  "kernel_sources": [
    "gumfreddy/12a-wildtrack-mvdetr-training"
  ],
  "competition_sources": [],
  "model_sources": []
}
```

**File**: `notebooks/kaggle/12b_wildtrack_tracking_reid/12b_wildtrack_tracking_reid.ipynb`
- In the markdown header cell: change `ali369/12a-wildtrack-mvdetr-training` → `gumfreddy/12a-wildtrack-mvdetr-training`
- In the bootstrap cell: update any `ali369` or `mrkdagods` references to `gumfreddy`
- Tracking config: keep v14 Kalman params unchanged

### 09g — ResNet101-IBN-a DMT (Action 2A)

**File**: `notebooks/kaggle/09g_resnet101ibn_dmt/kernel-metadata.json`

```json
{
  "id": "gumfreddy/09g-resnet101-ibn-a-dmt-cityflowv2",
  "title": "09g ResNet101-IBN-a DMT CityFlowV2",
  "code_file": "09g_resnet101ibn_dmt.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "machine_shape": "NvidiaTeslaT4",
  "enable_internet": true,
  "dataset_sources": [
    "gumfreddy/mtmc-weights",
    "thanhnguyenle/data-aicity-2023-track-2"
  ],
  "kernel_sources": [],
  "competition_sources": []
}
```

**File**: `notebooks/kaggle/09g_resnet101ibn_dmt/09g_resnet101ibn_dmt.ipynb`
- **Critical**: Verify `circle_weight = 0.0` or that circle loss is not used
- Verify DMT (camera adversarial loss) is enabled
- Verify ImageNet pretrained weights loading (use IBN-Net repo URL)
- Output file should be named `resnet101ibn_dmt_cityflowv2_best.pth`

### 09h — ResNeXt101-IBN-a DMT (Action 2B, next week)

**File**: `notebooks/kaggle/09h_resnext101ibn_dmt/kernel-metadata.json`

```json
{
  "id": "gumfreddy/09h-resnext101-ibn-a-dmt-cityflowv2",
  "title": "09h ResNeXt101-IBN-a DMT CityFlowV2",
  "code_file": "09h_resnext101ibn_dmt.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "machine_shape": "NvidiaTeslaT4",
  "enable_internet": true,
  "dataset_sources": [
    "gumfreddy/mtmc-weights",
    "thanhnguyenle/data-aicity-2023-track-2"
  ],
  "kernel_sources": [],
  "competition_sources": []
}
```

### 10a/10c — Vehicle Ensemble Test (Action 2C)

After 09g completes:
1. Download `best_model.pth` from 09g output
2. Upload to `gumfreddy/mtmc-weights` as `resnet101ibn_dmt_cityflowv2_best.pth`
3. In 10a notebook (cell a13), change:
   ```python
   SECONDARY_WEIGHTS = "models/reid/resnet101ibn_dmt_cityflowv2_best.pth"
   ```
4. In 10c notebook (cell c10), enable fusion sweep:
   ```python
   FUSION_WEIGHT = 0.15  # Start lower than 0.30 for DMT model
   ```
5. Push 10a → wait → 10b auto → 10c with sweep

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|:-----------:|:------:|------------|
| 09g DMT ResNet mAP < 55% | Medium | Ensemble won't help | Fall back to retraining primary ViT with center loss |
| 12b on new detections regresses | Low | No person improvement | Debug detection format; fall back to 12a v26 detections |
| gumfreddy GPU quota exceeded | Medium | Delays 09g training | Prioritize 12b (free) first; split 09g across weeks |
| ResNeXt101-IBN-a too slow for T4 | Low | 09h takes >12h | Reduce epochs to 80; or use gradient accumulation |
| Ensemble still capped at ~77-78% | High | Vehicle SOTA unreachable | Accept ceiling; focus on person pipeline for SOTA claim |

---

## Realistic Outcome Expectations

### Optimistic Scenario
- Person: IDF1 ≥ 95.3% (matches SOTA) via 12b on 12a v3 + minor tuning
- Vehicle: MTMC IDF1 ~79-80% via 2-model ensemble (primary ViT + DMT ResNet101)

### Realistic Scenario
- Person: IDF1 95.0-95.2% (within 0.1pp of SOTA)
- Vehicle: MTMC IDF1 77.5-78.5% (ensemble provides marginal gain)

### Pessimistic Scenario
- Person: IDF1 ~94.7% (no improvement — detections good but tracker already optimal)
- Vehicle: MTMC IDF1 ~77.5% (DMT ResNet too weak for ensemble benefit)

### Key Honest Assessment
The vehicle pipeline is very unlikely to reach SOTA (84.86%) without 3+ high-quality models (all >70% mAP). Our best realistic path produces 2 models where the secondary is likely 55-65% mAP. The AIC22 winners had 5 models all trained with VeRi-776 → urban domain adaptation → CityFlowV2, plus box-grained matching. We lack the computational budget and model zoo to replicate that.

The **person pipeline** is the realistic SOTA target. At 94.7% IDF1 with 0.6pp to go, this is achievable with better detections alone.
