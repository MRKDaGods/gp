# Breakthrough Analysis — Why IDF1 Is Stuck at 78.4%

## Three Root Causes Identified

### 1. The 384px ViT (80.14% mAP) Was Never Properly Deployed

The pipeline's best Kaggle run (10c v44, 78.4% IDF1) uses the **256px model**:
```
stage2.reid.vehicle.weights_path=models/reid/transreid_cityflowv2_best.pth
stage2.reid.vehicle.input_size=[256,256]
```

The ONE attempt to deploy 384px (10c v48) used the **WRONG checkpoint** — the failed 09b v1 (mAP=44.9%), NOT the successful 09b v2 (mAP=80.14%).

**The 80.14% mAP 384px model has NEVER been tested in the full MTMC pipeline.**

### 2. ResNet101-IBN-a 52.77% mAP Is EXPECTED (Not a Bug)

The ViT has 3-stage progressive specialization:
- CLIP (400M pairs) → VeRi-776 (576 IDs, 37K images) → CityFlowV2 (128 IDs, 7.5K images)

The ResNet skips the critical middle step:
- ImageNet (1.3M generic) → CityFlowV2 (128 IDs, 7.5K images)

Published 75-80% baselines are on VeRi-776 (576 IDs), NOT CityFlowV2 (128 IDs). The numbers were never comparable.

**Fix: Train ResNet101-IBN-a on VeRi-776 FIRST, then fine-tune on CityFlowV2.**

### 3. Ensemble Dilutes Because Secondary Is Too Weak

52.77% secondary vs 80.14% primary = ~65% discriminative power. Fusion at 10% weight barely helps.

---

## Top 5 Untried Approaches

### Approach 1: Deploy 09b v2 384px Model (QUICK WIN — DO THIS FIRST)
- **Expected**: +1.0-2.5pp IDF1
- **Risk**: LOW — model exists, trained and validated
- **Effort**: Change 2 overrides + refit PCA
- **Steps**: Verify checkpoint → update 10a → run 10a→10b→10c → sweep sim_thresh

### Approach 2: VeRi-776 Pretraining for ResNet101-IBN-a
- **Expected**: +10-20pp mAP (52%→65-75%), enabling +1.5-2.5pp IDF1
- **Risk**: MEDIUM — proven for ResNet50 (NB03), needs adaptation
- **Effort**: New notebook for VeRi-776 training → fine-tune on CityFlowV2

### Approach 3: CID_BIAS Per Camera-Pair Calibration
- **Expected**: +0.5-1.0pp IDF1
- **Risk**: LOW — proven by AIC22 2nd place
- **Effort**: Extract GT pairs → compute per-pair offsets → apply in Stage 4

### Approach 4: Box-Grained Matching
- **Expected**: +0.5-1.5pp IDF1
- **Risk**: MEDIUM — changes Stage 2+4 data flow
- **Effort**: Keep per-crop embeddings, cross-detection similarity matrix

### Approach 5: Re-Enable Reranking After Feature Upgrade
- **Expected**: +0.5-1.0pp (only after Approach 1 or 2)
- **Risk**: LOW — re-sweep after stronger features
- **Effort**: Sweep k1, k2, lambda parameters

---

## Immediate Action: Deploy 384px ViT

1. Verify `transreid_cityflowv2_384px_best.pth` on Kaggle Models = 09b v2 (80.14%)
2. Update 10a: `weights_path=..._384px_best.pth`, `input_size=[384,384]`
3. Refit PCA on 384px features
4. Run 10a→10b→10c
5. Sweep sim_thresh if baseline improves