# MTMC Tracker — Research Findings & Strategic Analysis

> **IMPORTANT**: This is a living document. Update it whenever new experiments are run, new dead ends are discovered, or performance numbers change. Keep the "Current Performance" and "Dead Ends" sections current.

## Current Performance (Last Updated: 2026-03-28)

### Vehicle Pipeline (CityFlowV2)

| Metric | Value | Notes |
|--------|:-----:|-------|
| **Best Kaggle IDF1** | **78.4%** | 10c v44, ali369 account, min_hits=2 |
| **SOTA Target** | 84.86% | AIC22 1st place |
| **Gap to SOTA** | 6.46pp | Feature quality, not association |
| **Primary Model (ViT-B/16 CLIP 256px)** | mAP=80.14% | On CityFlowV2 eval split |
| **Secondary Model (ResNet101-IBN-a)** | mAP=52.77% | On CityFlowV2 eval split, ImageNet→CityFlowV2 only |
| **384px ViT (09b v2)** | mAP=80.14% | Trained, **checkpoint on Kaggle is v1 (44.9%)** — v2 never uploaded |
| **Association configs tested** | 225+ | All within 0.3pp of optimal |

### Person Pipeline (WILDTRACK) — NEW

| Metric | Value | Notes |
|--------|:-----:|-------|
| **Best IDF1** | **36.8%** | Exp 1, conf=0.55, min_len=8, min_hits=3 |
| **Best MOTA** | **11.8%** | Same run (up from -28.1% after frame ID bug fix) |
| **ReID Model** | TransReID ViT-B/16 CLIP | Market1501 pretrained, not fine-tuned on WILDTRACK |
| **Status** | Early exploration | Extreme fragmentation, low detector precision |

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

## Person Pipeline (WILDTRACK) — New Initiative

### Baseline Performance

| Run | Config Changes | Tracklets | MTMC IDF1 | IDF1 | MOTA |
|-----|---------------|-----------|-----------|------|------|
| Baseline (run_20260327_211115) | Default wildtrack.yaml | 1,339 | 0.176 | 0.316 | -0.281 |
| Exp 1 (run_20260327_224721) | conf=0.55, min_len=8, min_hits=3, match=0.75, merge_gap=40 | 819 | 0.233 | 0.368 | 0.118 |
| Kaggle Baseline (wildtrack_20260328_000939) | 11a→11b→11c chain on Kaggle T4; conf=0.55, min_hits=3, match=0.75, min_tracklet_length=8; ReID 768D→PCA 256D (EV=0.988), HSV 192D, flip_aug=on, camera_bn=on; sim=0.30, louvain=1.5, app=0.80, hsv=0.10, spatio=0.10 | 911 | 0.140 | 0.280 | -0.463 |
| Exp 2 (run_20260327_231511) | conf=0.65, min_len=12, fresh PCA, rerank=off, sim=0.40, louvain=2.0, app=0.90 | INCOMPLETE (interrupted) | — | — | — |

Kaggle baseline details: 911 tracklets across 7 cameras (C1:126, C2:169, C3:154, C4:88, C5:146, C6:121, C7:107), per-camera 2D metrics IDF1=0.280 / MOTA=-0.463 / HOTA=0.000 / IDSW=573, MTMC IDF1=0.140 / MOTA=-0.276 / IDSW=1006, error analysis: 164 fragmented GT IDs, 141 conflated pred IDs, 46 unmatched GT IDs, 141 unmatched pred IDs. Per-camera IDF1/MOTA/IDSW: C1 0.261/-0.012/126, C2 0.191/-0.669/107, C3 0.253/-0.113/127, C4 0.233/-1.468/23, C5 0.316/-0.687/80, C6 0.254/0.004/86, C7 0.450/-0.296/24.

Kaggle underperformed the local Exp 1 baseline (MTMC IDF1 0.140 vs 0.233; per-camera IDF1 0.280 vs 0.368). The likely causes are suboptimal fixed association parameters in 11c and GPU/framework-dependent detection differences. The 911 vs 819 tracklet count gap indicates slightly different detection/tracking behavior on Kaggle hardware/runtime, which plausibly explains part of the per-camera IDF1 regression.

### Key Discoveries (Person Pipeline)
1. **Frame ID off-by-one bug (FIXED)**: WILDTRACK GT was being written with 0-based frame IDs, but predictions use 1-based. Fixed in `scripts/prepare_dataset.py`. This single fix contributed +39.9pp MOTA improvement.
2. **Extreme tracklet fragmentation**: 1,339 tracklets for ~20 people in baseline. Increasing min_hits and min_tracklet_length helped reduce to 819.
3. **Over-detection**: Person detector has low precision (~0.33, 13K FP). Need higher confidence threshold.
4. **PCA model potentially wrong distribution**: Person PCA was trained on vehicle features. Moved to .bak to force refit on WILDTRACK data.
5. **ReID model**: Using TransReID ViT-Base/16 CLIP pretrained on Market1501 (person-specific). Not fine-tuned on WILDTRACK.
6. **All GPU-intensive pipeline stages must run on Kaggle, not locally** (local GTX 1050 Ti too slow).
7. **Kaggle chain currently regresses vs local best**: Same stage-1 thresholds but fixed 11c association settings and possible runtime differences produced 911 tracklets and lower MTMC/per-camera IDF1 than local Exp 1.

### Person Pipeline Next Steps
- Set up person pipeline stages 0-2 on Kaggle (GPU-intensive)
- Run stages 3-5 on Kaggle too (via notebook chain) for convenience
- Fine-tune person ReID on WILDTRACK or EPFL data if better features needed
- Try conf=0.70+ to reduce false positives further

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