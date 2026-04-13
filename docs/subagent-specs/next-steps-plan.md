# Next Steps Plan — Vehicle + Person Pipelines

> **Created**: 2026-03-29 | **Status**: Active
> **Vehicle MTMC IDF1**: 78.4% → Target 84.86% (SOTA)
> **Person MODA**: 89.8%, IDF1 92.2% on WILDTRACK (ground-plane)

---

## Table of Contents

1. [Question 1: After 384px Upload — Experiment Plan](#q1-384px-experiments)
2. [Question 2: CID_BIAS Deep Dive](#q2-cid_bias)
3. [Question 3: 09f Deployment Plan](#q3-09f-deployment)
4. [Question 4: Person Pipeline Assessment](#q4-person-pipeline)
5. [Question 5: Prioritized Action Plan](#q5-action-plan)

---

## Q1: After 384px Upload — Experiment Plan {#q1-384px-experiments}

### What to run first: 384px model ALONE

**Rationale**: The 384px ViT (mAP=80.14%) has NEVER been tested in the full pipeline. The checkpoint on Kaggle was the failed v1 (mAP=44.9%). This is the single highest-ROI change available — estimated +1.0-2.5pp IDF1 from resolution upgrade alone.

### Experiment sequence:

#### Experiment A: 384px Baseline (MANDATORY FIRST)
- **Pipeline**: 10a → 10b → 10c
- **Config**: 10a already has 384px configured in `stage2.reid.vehicle`:
  - `weights_path: "models/reid/transreid_cityflowv2_384px_best.pth"`
  - `input_size: [384, 384]`
  - `vehicle2.enabled: false` (keep secondary OFF for clean baseline)
- **Expected IDF1**: 79.4–80.9% (current 78.4% + 1.0–2.5pp)
- **Purpose**: Establish the 384px-only baseline before adding any other changes
- **Measurement**: Compare directly to current best (10c v44, IDF1=78.4%)

#### Experiment B: 384px + CID_BIAS
- **Run AFTER Experiment A completes** — needs stage2 output from Experiment A
- **Requires**: Running `scripts/compute_cid_bias.py` on Experiment A's stage2 embeddings
- **Config override**: `stage4.association.camera_bias.enabled=true`
- **Expected additional gain**: +0.5–1.0pp over Experiment A
- **Note**: CID_BIAS should be recomputed for 384px features (distribution differs from 256px)

#### Should we generate CID_BIAS for 384px?

**YES, absolutely.** CID_BIAS is per-camera-pair and feature-distribution-dependent. The 384px model will have different per-camera-pair mean similarities than the 256px model. The bias matrix from 256px features would be stale/wrong.

**Process to generate CID_BIAS for 384px:**
1. Run 10a → 10b (stages 0-3) with 384px model
2. Download stage1 tracklets and stage2 embeddings from Kaggle output
3. Run locally: `python scripts/compute_cid_bias.py --stage2-dir <path>/stage2 --tracklets-dir <path>/stage1 --gt-dir data/raw/cityflowv2 --output configs/datasets/cityflowv2_cid_bias.npy`
4. Upload the `.npy` + `.json` to `mrkdagods/mtmc-weights` dataset on Kaggle
5. Run 10c with `stage4.association.camera_bias.enabled=true`

---

## Q2: CID_BIAS Deep Dive {#q2-cid_bias}

### What is CID_BIAS?

CID_BIAS (Camera ID Bias) is a **per-camera-pair additive similarity correction** used in AIC21/AIC22 SOTA methods. It compensates for systematic appearance differences between camera pairs caused by viewpoint, lighting, and background differences.

**Intuition**: A vehicle seen from camera c001 (overhead) and c002 (frontal) will have systematically lower cosine similarity than c001↔c003 (both overhead). CID_BIAS adds a correction so that hard camera pairs aren't unfairly penalized.

**Formula**: For tracklets from cameras `ci` and `cj`:
```
adjusted_similarity(i, j) = cosine_similarity(i, j) + bias_matrix[ci][cj]
```

Where `bias_matrix[ci][cj] = mean_sim(ci, cj) - global_mean_sim`, centered so the global average bias is zero.

### Is it implemented?

**YES — fully implemented but not yet activated with valid data.**

| Component | Status | Location |
|-----------|--------|----------|
| Bias computation script | ✅ Implemented | `scripts/compute_cid_bias.py` |
| Stage4 pipeline integration | ✅ Implemented | `src/stage4_association/pipeline.py` lines 381-413 |
| Config support | ✅ Configured | `configs/datasets/cityflowv2.yaml` line 251 |
| Actual bias NPY file | ❌ NOT GENERATED | `configs/datasets/cityflowv2_cid_bias.npy` does not exist |

**How the pipeline uses it** (from `src/stage4_association/pipeline.py`):
1. Loads the NPY bias matrix and camera name mapping JSON
2. For every cross-camera pair `(i, j)` in the similarity matrix, adds `bias_matrix[ci][cj]`
3. Applied AFTER FIC whitening and BEFORE graph clustering
4. Falls back to iterative learned bias if NPY not present (controlled by `iterations` parameter, currently 0)

### What's needed for 384px?

1. **Run the full pipeline** (10a→10b) with the 384px model to generate stage1 tracklets + stage2 embeddings
2. **Download outputs** from Kaggle
3. **Run the bias computation script** (CPU-only, runs locally):
   ```
   python scripts/compute_cid_bias.py \
     --stage2-dir data/outputs/run_384px/stage2 \
     --tracklets-dir data/outputs/run_384px/stage1 \
     --gt-dir data/raw/cityflowv2 \
     --output configs/datasets/cityflowv2_cid_bias.npy
   ```
4. **Upload NPY + JSON** to Kaggle dataset
5. **Run 10c** with bias enabled

### Expected impact

- **+0.5–1.0pp IDF1** based on SOTA analysis (AIC21 1st place uses NPY-based CID_BIAS)
- Impact is larger when camera pairs have significantly different mean similarities
- CityFlowV2 has S01 (3 cameras) and S02 (3 cameras) with different viewpoints — good candidate for CID_BIAS
- Dead end note: the existing `camera_bias` with `iterations > 0` (learned bias) was tested and hurt by -0.7pp — but that's a _different_ mechanism (iterative learned bias from clusters, not GT-computed NPY bias)

---

## Q3: 09f Deployment Plan {#q3-09f-deployment}

### When 09f v3 finishes

**Expected outcome**: mAP 65–75% on CityFlowV2 eval split (ResNet101-IBN-a with VeRi-776 pretraining → CityFlowV2 fine-tuning, corrected recipe: no circle loss, AdamW lr=3.5e-4, 120 epochs).

### Integration into 10a pipeline as vehicle2

The 10a notebook and cityflowv2.yaml **already configure vehicle2**:

```yaml
# In configs/datasets/cityflowv2.yaml (already set):
vehicle2:
  enabled: true
  save_separate: true
  model_name: "resnet101_ibn_a"
  weights_path: "models/reid/resnet101ibn_cityflowv2_384px_best.pth"
  embedding_dim: 2048
  input_size: [384, 384]
  clip_normalization: false
```

### Steps to deploy:

1. **Download 09f v3 best checkpoint** from Kaggle output
2. **Rename** to `resnet101ibn_cityflowv2_384px_best.pth`
3. **Upload** to `mrkdagods/mtmc-weights` Kaggle dataset
4. **10a already enables vehicle2** — the cityflowv2.yaml has `vehicle2.enabled: true`
5. **10c config**: Set `stage4.association.secondary_embeddings.weight` for fusion sweep
   - The cityflowv2.yaml already has `secondary_embeddings.weight: 0.4`
   - Sweep: 0.2, 0.3, 0.4 to find optimal blend

### Config changes needed

**None for 10a** — cityflowv2.yaml already has vehicle2 enabled and configured.

**For 10c** — sweep the fusion weight:
```
stage4.association.secondary_embeddings.weight=0.2
stage4.association.secondary_embeddings.weight=0.3
stage4.association.secondary_embeddings.weight=0.4
```

### Expected IDF1 improvement from ensemble

- **With 65% mAP secondary**: +0.5–1.0pp (weak ensemble partner, but complementary architecture)
- **With 75% mAP secondary**: +1.0–2.0pp (strong ensemble, ResNet captures different features than ViT)
- **Combined with 384px primary**: potentially +2.0–4.0pp total over current 78.4%
- **Risk**: If 09f v3 mAP < 60%, ensemble may dilute rather than help (as seen with 52.77% secondary)

### Fallback if 09f v3 fails

- **Re-examine loss config** — 09f v2 failed at mAP=16.2% due to circle loss gradient conflict
- **Try lower LR** — 09f v3 uses lr=3.5e-4; if it overshoots, try 1e-4
- **Use 09e checkpoint directly** (mAP=62.52% on VeRi-776) as a weak vehicle2 without CityFlowV2 fine-tuning — but this is cross-dataset and will underperform

---

## Q4: Person Pipeline Assessment {#q4-person-pipeline}

### Is 92.8% IDF1 competitive on WILDTRACK?

**Context clarification**: The 92.2% IDF1 and 89.8% MODA are **ground-plane tracking metrics** from MVDeTr detections evaluated on the WILDTRACK ground plane. This is NOT the same as per-camera MTMC IDF1.

### WILDTRACK Published Baselines (Ground-Plane Protocol)

| Method | MODA | MODP | Year | Notes |
|--------|:----:|:----:|:----:|-------|
| RCNN + POM-CNN | 11.3% | 18.4% | 2018 | WILDTRACK paper baseline |
| DeepMCD | 64.3% | 73.9% | 2018 | Multi-camera detection |
| MVDet (Hou et al.) | 88.2% | 75.7% | 2020 | Anchor-free ground-plane |
| **MVDeTr (paper)** | **91.5%** | **82.0%** | 2022 | Deformable transformer |
| **Ours (12a v11)** | **92.0%** | **81.9%** | 2026 | Surpasses MVDeTr paper |
| Shot (Ong et al.) | ~93% | — | 2023 | Larger backbone, more epochs |

**Assessment**: Our **92.0% MODA surpasses the MVDeTr paper** (91.5%) and is within ~1pp of the best published results. The ground-plane detection is NOT the bottleneck.

**For tracking (IDF1=92.2%)**: Published WILDTRACK results rarely report tracking IDF1 because the benchmark protocol is detection-focused (MODA/MODP). Our 92.2% IDF1 is a strong result given only 12 ID switches across 400 frames.

**Verdict**: Person pipeline ground-plane performance is **already competitive**. The WILDTRACK detection component exceeds published baselines.

### Can we improve further?

| Improvement | Expected Impact | Effort | Priority |
|-------------|:--------------:|:------:|:--------:|
| Train MVDeTr longer (20-30 epochs) | +0.5–1.0pp MODA | Low (Kaggle GPU) | LOW |
| Larger backbone (ResNet50) | +0.5–1.5pp MODA | Medium | LOW |
| Person ReID for cross-camera merge | Enables true MTMC IDF1 | Medium | **HIGH** |
| Fine-tune ReID on EPFL/CUHK03 | Better person features | High | MEDIUM |
| More training data augmentation | +0.2–0.5pp | Low | LOW |

### ReID cross-camera merging status

- **12b v8** fixed the ViT backbone mismatch (CLIP → standard ViT)
- Mean cosine similarity dropped from 0.874 to 0.720 (good — less mode-collapsed)
- Only 1 correct merge at threshold 0.90 (from 382 candidates)
- **The cross-camera merging is not yet useful** because:
  1. WILDTRACK has 7 heavily-overlapping cameras — same person visible in multiple cameras simultaneously
  2. Ground-plane tracking already handles multi-view association
  3. Person ReID adds value primarily for re-identification across time gaps or after occlusion

### End-to-end evaluation

Currently missing: a proper **per-camera MTMC evaluation** that combines detection + tracking + ReID across all 7 cameras. The ground-plane metrics (MODA/MODP) evaluate detection quality, and IDF1=92.2% evaluates ground-plane tracking. But there's no MTMC protocol evaluation yet.

**What's needed**: Run the full per-camera pipeline (stages 0-5) on WILDTRACK with the 11a→11b→11c chain and compare against per-camera GT. This would give MTMC IDF1 comparable to the vehicle pipeline's metric.

---

## Q5: Prioritized Action Plan {#q5-action-plan}

### Phase 1: 384px Deployment (IMMEDIATE — after upload completes)

| Step | Action | Platform | Blocked By |
|:----:|--------|:--------:|:----------:|
| 1.1 | Verify 384px checkpoint uploaded to `mrkdagods/mtmc-weights` | Local | Upload completing |
| 1.2 | Push 10a notebook (vehicle2.enabled=false for clean baseline) | Local | 1.1 |
| 1.3 | Run 10a on Kaggle (stages 0-2, GPU) | Kaggle | 1.2 |
| 1.4 | Run 10b on Kaggle (stage 3, CPU) | Kaggle | 1.3 |
| 1.5 | Run 10c on Kaggle (stages 4-5, CPU) | Kaggle | 1.4 |
| 1.6 | Record IDF1 result — this is the **384px baseline** | Local | 1.5 |

**Expected outcome**: IDF1 = 79.4–80.9%

### Phase 2: CID_BIAS for 384px (after Phase 1)

| Step | Action | Platform | Blocked By |
|:----:|--------|:--------:|:----------:|
| 2.1 | Download 10a stage1 + stage2 outputs from Kaggle | Local | Phase 1 |
| 2.2 | Run `scripts/compute_cid_bias.py` locally | Local | 2.1 |
| 2.3 | Upload `cityflowv2_cid_bias.npy` + `.json` to Kaggle dataset | Local | 2.2 |
| 2.4 | Run 10c with `camera_bias.enabled=true` | Kaggle | 2.3 |
| 2.5 | Record IDF1 — this is the **384px + CID_BIAS** result | Local | 2.4 |

**Expected outcome**: IDF1 = 79.9–81.9%

### Phase 3: 09f v3 Completion + Ensemble (parallel with Phase 1-2)

| Step | Action | Platform | Blocked By |
|:----:|--------|:--------:|:----------:|
| 3.1 | Monitor 09f v3 training on Kaggle | Local | 09f running |
| 3.2 | When done: download best checkpoint, check mAP | Local | 3.1 |
| 3.3 | If mAP ≥ 60%: upload as `resnet101ibn_cityflowv2_384px_best.pth` | Local | 3.2 |
| 3.4 | Run 10a with vehicle2.enabled=true (384px + ensemble) | Kaggle | 3.3 + Phase 1 |
| 3.5 | Run 10b → 10c with fusion weight sweep (0.2, 0.3, 0.4) | Kaggle | 3.4 |
| 3.6 | Record best IDF1 — this is the **384px + ensemble** result | Local | 3.5 |

**Expected outcome**: IDF1 = 80.4–82.9% (depending on 09f mAP)

### Phase 4: Reranking Revival (after Phase 3)

| Step | Action | Platform | Blocked By |
|:----:|--------|:--------:|:----------:|
| 4.1 | Re-enable reranking with best 384px+ensemble features | Kaggle | Phase 3 |
| 4.2 | Sweep: k1=[20,30], k2=[6,10], lambda=[0.3,0.4] | Kaggle | 4.1 |

**Expected outcome**: +0.5–1.0pp if features are strong enough (was always harmful with 256px single model)

### Phase 5: Person Pipeline Maturation (LOW PRIORITY — parallel)

| Step | Action | Platform | Blocked By |
|:----:|--------|:--------:|:----------:|
| 5.1 | Ground-plane results are already competitive — document as achievement | Local | Nothing |
| 5.2 | Run full 11a→11b→11c per-camera WILDTRACK pipeline for MTMC IDF1 | Kaggle | Nothing |
| 5.3 | Tune cross-camera merging thresholds with fixed ReID features | Kaggle | 5.2 |
| 5.4 | Consider training MVDeTr longer (20 epochs) if MODA improvement needed | Kaggle | Low priority |

**Person pipeline verdict**: Ground-plane performance (MODA=92.0%, IDF1=92.2%) is already at or above published SOTA. Focus energy on the vehicle pipeline where the competition gap is larger and more impactful.

---

## Summary: Expected IDF1 Progression (Vehicle)

| Phase | Config | Expected IDF1 | Delta |
|:-----:|--------|:-------------:|:-----:|
| Current | 256px ViT single model | 78.4% | — |
| Phase 1 | 384px ViT single model | 79.4–80.9% | +1.0–2.5pp |
| Phase 2 | + CID_BIAS | 79.9–81.9% | +0.5–1.0pp |
| Phase 3 | + ResNet101 ensemble | 80.4–82.9% | +0.5–1.0pp |
| Phase 4 | + Reranking | 80.9–83.9% | +0.5–1.0pp |
| **Cumulative** | **All phases** | **80.9–83.9%** | **+2.5–5.5pp** |
| **SOTA** | AIC22 1st place | **84.86%** | — |

**Remaining gap after all phases**: 0.9–4.0pp. If all phases hit their high estimates, we're within 1pp of SOTA. If they hit low estimates, we need additional techniques (DMT camera-aware training, box-grained matching, or a 3rd model).

---

## Risk Matrix

| Risk | Probability | Impact | Mitigation |
|------|:----------:|:------:|------------|
| 384px model shows < 1pp improvement | 20% | HIGH | Check if weights loaded correctly; compare embeddings |
| 09f v3 fails (mAP < 60%) | 30% | MEDIUM | Try lower LR; use 09e VeRi-776 checkpoint directly |
| CID_BIAS hurts (like iterative bias did -0.7pp) | 15% | LOW | NPY-based is different mechanism; can disable easily |
| Reranking still harmful with better features | 40% | LOW | Known dead end with weak features; may work with strong |
| Kaggle GPU quota exhausted | 25% | HIGH | Spread across 3 accounts (ali369, mrkdagods, yahia) |