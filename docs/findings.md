# MTMC Tracker — Research Findings & Strategic Analysis

> **IMPORTANT**: This is a living document. Update it whenever new experiments are run, new dead ends are discovered, or performance numbers change. Keep the "Current Performance" and "Dead Ends" sections current.

## Current Performance (Last Updated: 2026-03-27)

| Metric | Value | Notes |
|--------|:-----:|-------|
| **Best Kaggle IDF1** | **78.4%** | 10c v44, ali369 account, min_hits=2 |
| **SOTA Target** | 84.86% | AIC22 1st place |
| **Gap to SOTA** | 6.46pp | Feature quality, not association |
| **Primary Model (ViT-B/16 CLIP 256px)** | mAP=80.14% | On CityFlowV2 eval split |
| **Secondary Model (ResNet101-IBN-a)** | mAP=52.77% | On CityFlowV2 eval split, ImageNet→CityFlowV2 only |
| **384px ViT (09b v2)** | mAP=80.14% | Trained, **checkpoint on Kaggle is v1 (44.9%)** — v2 never uploaded |
| **Association configs tested** | 225+ | All within 0.3pp of optimal |

## Gap Decomposition

The 6.46pp gap to SOTA decomposes into:

| Deficiency | Impact | Status |
|------------|:------:|--------|
| Single ReID model vs 3-5 ensemble | -2.5 to -3.5pp | Blocked by weak secondary (52.77%) |
| 256px input vs 384px | -1.0 to -1.5pp | 384px model trained but wrong checkpoint deployed |
| No camera-aware training (DMT) | -1.0 to -1.5pp | Not implemented |
| **Total estimated** | **-4.5 to -6.5pp** | Matches observed gap |

## Critical Discovery: 384px Checkpoint Mismatch

The Kaggle Models dataset `mrkdagods/mtmc-weights` contains `transreid_cityflowv2_384px_best.pth` with metadata showing `best_mAP: 0.449` — this is the **failed 09b v1**, NOT the successful 09b v2 (80.14%). The 10c v48 experiment that "tested 384px" actually tested the wrong model. **The 80.14% 384px model has NEVER been tested in the full pipeline.**

## Critical Discovery: ResNet101-IBN-a 52.77% Is Expected

The ViT achieves high mAP because of 3-stage progressive specialization:
- CLIP (400M image-text pairs) → VeRi-776 (576 IDs, 37K images) → CityFlowV2 (128 IDs, 7.5K images)

The ResNet skips the critical VeRi-776 middle step:
- ImageNet (1.3M generic) → CityFlowV2 (128 IDs, 7.5K images) directly

Published 75-80% mAP baselines for ResNet101-IBN-a are evaluated on **VeRi-776** (576 IDs), NOT on CityFlowV2 (128 IDs). These numbers were never comparable. The 52.77% is reasonable given the massive pretraining disadvantage.

**Fix**: Train ResNet101-IBN-a on VeRi-776 FIRST, then fine-tune on CityFlowV2 (same pattern as the ViT).

## What SOTA Does Differently

| Pattern | AIC22 1st | AIC22 2nd | AIC21 1st | We have? |
|---------|:-:|:-:|:-:|:-:|
| 3+ ReID backbone ensemble | 5 models | 3 models | 3 models | **NO** (1 working) |
| 384×384 input | ✅ | ✅ | ✅ | **NO** (256px deployed) |
| IBN-a backbones | ✅ | ✅ | ✅ | ViT only |
| Camera-pair bias (CID_BIAS) | ROI masks | NPY | NPY | **NO** (FIC only) |
| Reranking | Box-grained | k-reciprocal | k-reciprocal | **Disabled** |
| Camera-aware training (DMT) | ✅ | ✅ | ✅ | **NO** |
| Multiple loss functions | ID+tri+circle+cam | ID+tri+cam | ID+tri+cam | ID+tri |

## Prioritized Action Plan

| Priority | Action | Expected Impact | Status |
|:--------:|--------|:---------------:|--------|
| **1** | Upload correct 09b v2 384px checkpoint & deploy in pipeline | +1.0-2.5pp | **NOT DONE** — wrong checkpoint on Kaggle |
| **2** | Train ResNet101-IBN-a on VeRi-776 → fine-tune CityFlowV2 | +1.5-2.5pp (via ensemble) | NOT STARTED |
| **3** | CID_BIAS per camera-pair calibration | +0.5-1.0pp | NOT STARTED |
| **4** | Box-grained matching (per-detection features) | +0.5-1.5pp | NOT STARTED |
| **5** | Re-enable reranking after feature upgrade | +0.5-1.0pp | Blocked by #1/#2 |

## Conclusive Dead Ends (DO NOT RETRY)

| Approach | Result | Evidence |
|----------|--------|---------|
| Association parameter tuning | Exhausted (225+ configs, all within 0.3pp) | Experiment log |
| CSLS distance | -34.7pp catastrophic | v74 |
| Hierarchical clustering | -1.0 to -5.1pp | v54-56, v62 |
| FAC (Feature Augmented Clustering) | -2.5pp | v26 |
| Feature concatenation (vs score fusion) | -1.6pp | Experiment log |
| CamTTA (Camera Test-Time Adaptation) | Helps global, hurts MTMC | v28-30 |
| Multi-scale TTA | Neutral/harmful | Multiple runs |
| Track smoothing / edge trim | Always harmful | Experiment log |
| Denoise preprocessing | -2.7pp | v46, v82 |
| mtmc_only submission flag | -5pp | Documented |
| Auto-generated zone polygons | -0.4pp | v54-57 |
| PCA dimension search | 384D optimal, others worse | Experiment log |
| Ensemble with 52% secondary at high weight | Dilutes signal | Current state |
| SGD optimizer for ResNet101-IBN-a | 30.27% mAP catastrophic | v18 mrkdagods |
| Circle loss + triplet loss together | Gradient conflict | v17 |
| K-reciprocal reranking (with current features) | Always worse | v25, v35 |
| Camera-pair similarity normalization | Zero effect (FIC handles it) | v36 |
| confidence_threshold=0.20 | -2.8pp | v45 |
| max_iou_distance=0.5 | -1.6pp | v47 |

## Component Health Summary

| Component | Status | Notes |
|-----------|:------:|-------|
| Detection (YOLO26m) | ✅ OK | Not the bottleneck |
| Tracking (BoT-SORT) | ✅ OK | min_hits=2 optimal |
| Feature Extraction (ViT) | ⚠️ Ceiling | Single model at 256px is the limit |
| Feature Processing (PCA) | ✅ OK | 384D optimal |
| Ensemble/Fusion | ❌ Blocked | Secondary model too weak (52.77%) |
| Association (Stage 4) | ✅ Exhausted | 225+ configs, no more gains |
| Evaluation | ✅ OK | Under-merging 1.69:1 ratio = feature quality issue |

## Model Training History

### TransReID ViT-B/16 CLIP (Primary)
- 09b v1: mAP=44.9% (40 epochs from 256px init, too aggressive LR)
- 09b v2: mAP=80.14%, R1=92.27% (VeRi-776 pretrained → CityFlowV2 fine-tune) ← **BEST, but wrong checkpoint on Kaggle**

### ResNet101-IBN-a (Secondary)
- 09d v12: mAP=21.9% (IBN layer3 bug)
- 09d v13: mAP=11.98% at epoch 19 (timed out)
- 09d v17: mAP=29.6% (wrong recipe: lr=3.5e-4 + circle_weight=0.5)
- 09d v18 ali369: mAP=52.77% (AdamW lr=1e-3, best so far)
- 09d v18 mrkdagods: mAP=30.27% (SGD lr=0.008, failed catastrophically)

## Key Insight

**The system is NOT broken.** It's operating at the ceiling of single-model 256px architecture. The codebase is architecturally ready for 84%+ (multi-model support, score fusion, separate PCA all exist). The problem is purely feature quality — deploying the correct 384px model and training a competent second backbone with VeRi-776 pretraining. This is an ML training problem, not a software engineering problem.