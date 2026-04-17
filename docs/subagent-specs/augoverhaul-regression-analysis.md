# Augmentation Overhaul Model — MTMC IDF1 Regression Analysis

**Date**: 2026-04-14  
**Issue**: 09 v2 augmentation-overhaul model (mAP=81.59%) causes -5.3pp MTMC IDF1 regression vs baseline (0.722 vs 0.775)  
**Status**: Root cause identified — confounded experiment with both augmentation AND loss function changes

---

## 1. Timeline of Events

| Step | Version | Result | Notes |
|------|:-------:|:------:|-------|
| Baseline MTMC | 10c v52 | **IDF1=0.775** | Original model (`transreid_cityflowv2_best.pth`) |
| Train augoverhaul | 09 v2 | **mAP=81.59%** | +1.45pp over baseline 80.14% |
| Deploy at 384px (BUG) | 10c v47 | IDF1=0.702 | Config inherited stale 384px default |
| Fix: deploy at 256px | 10a v20 | Confirmed 256px | `input_size=[256,256]` override added |
| Full sweep at 256px | 10c v48 | **IDF1=0.722** | -5.3pp regression, NOT a config bug |

## 2. What Changed Between Baseline and Augoverhaul

The 09 v2 notebook changed **two independent variables simultaneously**, making this a confounded experiment:

### 2a. Augmentation Changes

| Augmentation | Baseline (09b v2) | Augoverhaul (09 v2) |
|-------------|-------------------|---------------------|
| `ColorJitter` | `(0.2, 0.15, 0.1, 0.0)` | `(0.3, 0.25, 0.2, 0.05)` — stronger, adds hue |
| `RandomGrayscale` | None | `p=0.1` — **NEW** |
| `GaussianBlur` | None | `k=5, sigma=(0.1,2.0), p=0.2` — **NEW** |
| `RandomPerspective` | None | `distortion_scale=0.1, p=0.2` — **NEW** |
| `RandomErasing` | `scale=(0.02, 0.33)` | `scale=(0.02, 0.4)` — wider range |
| `Resize` strategy | `(272,272) → Pad(10) → Crop(256)` | `(272,272) → Pad(10) → Crop(256)` — same |

### 2b. Loss Function Change

| Component | Baseline (09b v2) | Augoverhaul (09 v2) |
|-----------|-------------------|---------------------|
| Classification | CE + Label Smoothing (ε=0.05) | CE + Label Smoothing (ε=0.05) — same |
| Metric learning | **TripletLoss** (margin=0.3, hard mining) | **CircleLoss** (m=0.25, γ=128) — **CHANGED** |
| Center loss | CenterLoss (5e-4, delayed@ep15) | CenterLoss (5e-4, delayed@ep15) — same |
| JPM auxiliary | 0.5 × CE(JPM) | 0.5 × CE(JPM) — same |

**Critical finding**: The export metadata at line 1358 of the 09 notebook **falsely lists `"TripletLoss(m=0.3)"`** in the tricks array, but the actual training code at line 1030 instantiates `CircleLoss(m=0.25, gamma=128)`. The TripletLossHardMining class is defined but never used. This metadata bug could mislead future experiments.

### 2c. Everything Else — Identical

Optimizer (AdamW), LR schedule (cosine, 120 epochs, warmup 10), backbone LR (1e-4), head LR (1e-3), LLRD (0.75), weight decay (5e-4), batch size (64, PK 16×4), image size (256×256), model architecture (TransReID ViT-B/16 CLIP, SIE, JPM), init weights (VeRi-776 pretrained) — all the same.

## 3. Root Cause Analysis

### 3a. Primary Hypothesis: Color Invariance Damage

The augmentation overhaul pushes the model to be **more invariant to color, texture, and local detail**:

- **RandomGrayscale(p=0.1)**: 10% of training crops lose ALL color information, teaching the model to match vehicles by shape alone
- **Stronger ColorJitter**: Wider brightness/contrast/saturation/hue ranges further suppress color-dependent features
- **GaussianBlur(p=0.2)**: Destroys fine textures in 20% of crops, biasing toward structural features

In CityFlowV2, many vehicles share the same make/model (e.g., multiple white sedans, silver SUVs). **Color is often the ONLY discriminative cue** between same-model vehicles across cameras. By making the model more color-invariant:

- **mAP goes UP** because the model generalizes better on the closed-set benchmark (128 training IDs with decent diversity)
- **MTMC IDF1 goes DOWN** because same-model different-color vehicles become indistinguishable in the thresholded similarity graph, causing **false merges** (conflated trajectories)

This is **exactly the same failure mode as 384px** (better mAP, worse MTMC) and **DMT camera-aware training** (better mAP, worse MTMC). The pattern is now established: **any change that reduces cross-camera feature specificity hurts MTMC, even if it improves closed-set retrieval metrics.**

### 3b. Secondary Hypothesis: CircleLoss Feature Space Geometry

TripletLoss and CircleLoss optimize fundamentally different objectives:

- **TripletLoss** maximizes the **margin** between hardest positive and hardest negative distances. This directly optimizes the property needed for thresholded matching: a clean separation between same-ID and different-ID similarity scores.
- **CircleLoss** optimizes **pair-level similarity** with adaptive weights, pushing positives toward sim≈1 and negatives toward sim≈0. It can produce better RANKING (higher mAP) while having a NARROWER effective margin in the critical decision region around the threshold.

For MTMC association, which uses **hard cosine similarity thresholds** to build a graph, TripletLoss's explicit margin optimization may produce features that are more amenable to thresholding than CircleLoss's ranking-oriented optimization.

Evidence supporting this:
- The optimal `sim_thresh` shifted from **0.50** (baseline) to **0.45** (augoverhaul)
- Even at the optimal threshold, raw MTMC IDF1 before AQE/AFLink was only **0.651** vs the baseline's **~0.77+**
- The similarity score distribution has fundamentally changed

### 3c. Interaction Effects

The augmentation changes and loss function change may amplify each other:
- CircleLoss with augmentation-invariant features could produce an extremely tight positive-pair cluster that masks legitimate inter-vehicle differences
- The combined effect (-5.3pp) is larger than either 384px (-2.8pp) or DMT (-1.4pp) individually, consistent with compounding damage

## 4. Model File Verification

The model file appears **correct** (base model, not EMA):

- **09 notebook exports**: `vehicle_transreid_vit_base_cityflowv2.pth` = base model (81.59% mAP), `..._ema.pth` = EMA model (39.09% mAP)
- **10a notebook copies**: `vehicle_transreid_vit_base_cityflowv2.pth` → `vehicle_transreid_vit_base_cityflowv2_augv1.pth`
- **Pipeline override**: `stage2.reid.vehicle.weights_path=models/reid/vehicle_transreid_vit_base_cityflowv2_augv1.pth`
- **10a v20 log confirmed**: `ReID model loaded: transreid, dim=768, input=(256, 256)`

If the EMA model (39.09% mAP) had been accidentally deployed, MTMC IDF1 would be MUCH lower than 0.722 — likely below 0.50. The 0.722 result is consistent with a model that has decent (but different) learned representations.

**However, there is one unverified risk**: the Kaggle dataset `09-vehicle-reid-cityflowv2-augoverhaul-ema` may contain outputs from a DIFFERENT 09 v2 run than the one that achieved 81.59%. If the notebook was re-pushed or the exported Kaggle dataset is from an earlier (lower quality) run, the deployed model could have lower mAP than expected. **This should be verified by checking the Kaggle dataset version and training logs.**

## 5. Sweep Behavior Analysis

The 10c v48 sweep reveals the feature quality problem clearly:

| Sweep Stage | Augoverhaul (v48) | Baseline Reference (v52) |
|-------------|:-----------------:|:------------------------:|
| Raw sim_thresh best | 0.630 @ thresh=0.45 | ~0.77 @ thresh=0.50 |
| + appearance_weight | 0.650 | — |
| + FIC regularization | 0.651 | — |
| + AQE (k=3) | 0.675 | — |
| + AFLink | 0.722 | 0.775 (no AFLink) |
| Final best | **0.722** | **0.775** |

Key observations:
1. **The raw similarity threshold sweep is -14pp worse** (0.630 vs ~0.77) — the features produce fundamentally worse cross-camera similarity scores
2. **AFLink helped +4.7pp** (0.675→0.722) whereas in the baseline it hurt -3.95pp — this suggests the augoverhaul features are missing identity-specific signal that AFLink's motion heuristics can partially recover
3. **AQE helped +2.4pp** (0.651→0.675) — query expansion provides more benefit when base features are weaker, consistent with feature quality being the root cause

## 6. Recommended Next Steps

### 6a. Immediate: Ablation Experiments (PRIORITY 1)

Since two variables changed simultaneously, we need to **ablate each independently**:

#### Experiment A: Augmentation-Only (keep TripletLoss)
- Train 09 with the augmentation overhaul transforms BUT with `TripletLossHardMining(margin=0.3)` instead of `CircleLoss`
- Deploy through 10a→10b→10c
- This isolates whether the augmentation changes alone cause the MTMC regression

#### Experiment B: CircleLoss-Only (keep original augmentations)
- Train 09 with the ORIGINAL augmentation transforms BUT with `CircleLoss(m=0.25, gamma=128)` instead of `TripletLoss`
- Deploy through 10a→10b→10c
- This isolates whether the loss function change alone causes the MTMC regression

#### Experiment C: Selective Augmentation (keep TripletLoss)
- Train 09 with only the SAFE augmentations (GaussianBlur, RandomPerspective, wider RandomErasing) BUT **without** RandomGrayscale and without the stronger ColorJitter
- These three augmentations add geometric/occlusion invariance without destroying color information
- This tests whether color-preserving augmentations can improve mAP without hurting MTMC

### 6b. Diagnostic: Feature Distribution Comparison (PRIORITY 2)

Before running expensive retraining, compare the feature distributions:

1. Extract embeddings from both models on the same set of tracklet crops
2. Compare cross-camera cosine similarity distributions:
   - Same-ID pairs: are the augoverhaul similarities lower? (less discriminative)
   - Different-ID same-model pairs: are they higher? (more confused)
   - Plot histograms of same-ID vs different-ID similarity for both models
3. This confirms WHETHER the regression is from color invariance (hypothesis 3a) or feature space geometry (hypothesis 3b)

### 6c. Fix the Export Metadata Bug

Update the 09 notebook export cell (line 1358) to correctly list `CircleLoss(m=0.25, gamma=128)` instead of `TripletLoss(m=0.3)` in the tricks array. This prevents future confusion when tracing model provenance.

### 6d. Verify Kaggle Dataset Version

Confirm that the Kaggle dataset `09-vehicle-reid-cityflowv2-augoverhaul-ema` contains the output from the run that achieved mAP=81.59%, not an earlier/different run. Check the training curves artifact or metadata JSON in the Kaggle dataset.

## 7. Strategic Implications

### The mAP-MTMC Disconnect Is Now a Pattern

| Model Change | mAP Effect | MTMC IDF1 Effect | Mechanism |
|-------------|:----------:|:----------------:|-----------|
| 384px input | +0pp (same) | **-2.8pp** | Viewpoint-specific textures |
| DMT camera-aware | +7pp | **-1.4pp** | Camera-specific features |
| Augmentation overhaul + CircleLoss | **+1.45pp** | **-5.3pp** | Color/texture invariance |

**Conclusion**: Standard ReID mAP on CityFlowV2 is NOT a reliable proxy for MTMC IDF1 performance. Improvements on the closed-set retrieval benchmark can and DO make the thresholded cross-camera matching WORSE. Any future model changes MUST be validated end-to-end through 10a→10b→10c, not just by ReID mAP.

### What This Means for the Remaining Gap

The augoverhaul failure confirms that **single-model feature quality improvements are extremely difficult to translate into MTMC gains**. Three different approaches (384px, DMT, augoverhaul) all improved or preserved mAP while hurting MTMC. The only path that has historically worked is the v80-restored baseline recipe, which represents the current ceiling.

The remaining realistic options are:
1. **Multi-model ensemble** with genuinely complementary models (NOT more single-model tuning)
2. **MTMC-aware training** — train the ReID model with an MTMC-specific objective (e.g., cross-camera triplet sampling, or using MTMC IDF1 as validation metric during training)
3. **Structural association changes** (GNN, network flow) that don't rely on better features

## 8. Key Takeaway

> **Do NOT deploy the augoverhaul model.** Revert to the original `transreid_cityflowv2_best.pth` for all MTMC experiments. The augoverhaul model is only useful as a baseline for ablation studies to understand the mAP-MTMC disconnect.
