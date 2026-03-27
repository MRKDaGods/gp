# Full System Diagnostic — MTMC Tracker

## Executive Summary

The system is **NOT broken**. It's operating at the ceiling of what a single-model 256px architecture can achieve. The 5.7pp gap to SOTA (84.86%) decomposes into three compounding deficiencies:

| Deficiency | Impact | Fix |
|------------|--------|-----|
| Single ReID model (vs 3-5 ensemble) | -2.5 to -3.5pp | Train 2+ additional backbones |
| 256px input (vs 384px) | -1.0 to -1.5pp | Retrain at 384px |
| No camera-aware training (DMT) | -1.0 to -1.5pp | Implement adversarial camera loss |
| **Total** | **-4.5 to -6.5pp** | Matches observed 5.7pp gap |

---

## 1. Component-by-Component Analysis

### 1.1 Detection + Tracking (Stage 0-1) — LOW CONCERN

- YOLO26m + BoT-SORT: Recall is high, detection is not the bottleneck
- min_hits=2 (best config) effectively filters noise
- No evidence of systematic detection failures causing MTMC errors

### 1.2 Feature Extraction (Stage 2) — CRITICAL BOTTLENECK

**TransReID ViT-Base/16 CLIP 256px:**
- mAP=80.14%, R1=92.27% on VeRi-776 — legitimately SOTA for single model
- BUT: single model captures ONE viewpoint encoding
- CLS token global descriptor: front vs rear produces different patch attention maps
- `concat_patch` option exists but NOT used in best config
- SIE DISABLED (training 59 cameras, test 6 — mismatch)
- No viewpoint-aware or camera-aware training losses

**Error profile (44-87 fragmented GT IDs)**: "same vehicle looks different across cameras" — exactly single-model limitation.

**SOTA comparison:**
- AIC22 1st: 5 diverse backbones (ConvNeXt, HRNet-W48, Res2Net200, ResNet50, ResNeXt101) all at 384×384
- AIC22 2nd: 3 IBN-a backbones at 384×384 with DMT training
- AIC21 1st: 3 IBN-a backbones at 384×384 with DMT camera+viewpoint-aware training

**Critical insight:** Architectural diversity matters more than any single model's strength.

### 1.3 Feature Processing: PCA Whitening — LOW CONCERN

- PCA=384D confirmed optimal (256D, 512D both tested and worse)
- Power normalization +0.5pp — good, keep it
- No further gains here

### 1.4 Ensemble/Fusion: ResNet101-IBN-a — CRITICAL (BLOCKED)

Training history:
- v12: mAP=21.9% (IBN layer3 bug)
- v13: mAP=11.98% at epoch 19 (timed out)
- v17: mAP=29.6% (wrong recipe: lr=3.5e-4 + circle_weight=0.5)
- v18 (ali369 AdamW): mAP=52.77% — best but mediocre
- v18 (mrkdagods SGD): mAP=30.27% — SGD failed catastrophically

**The fundamental problem:** Secondary model at 52.77% mAP is ~65% of primary's discriminative power. Score-level fusion dilutes the strong signal. Published ResNet101-IBN-a baselines on VeRi achieve 75-80% mAP. Our 52.77% indicates a training recipe problem.

**This is the single biggest blocking issue.**

### 1.5 Association (Stage 4) — LOW CONCERN (EXHAUSTED)

Pipeline: FIC → QE+DBA → exhaustive cross-camera pairs → hard temporal filter → mutual NN → combined similarity → conflict-free CC → gallery expansion → verification.

**220+ configs prove this is well-optimized.** All within 0.3pp. Remaining ceiling ~1pp from CID_BIAS + hand-annotated zones.

### 1.6 Evaluation — NOT A CONCERN

Error profile:
- Fragmented GT IDs: 44-87 (under-merging — features too dissimilar)
- Conflated pred IDs: 26-35 (over-merging)
- Ratio 1.69:1 → system under-merges → directly implicates feature quality

---

## 2. Component Rankings by Improvement Potential

| Rank | Component | Potential | Current State | Effort |
|:---:|-----------|:---------:|:---:|:---:|
| **1** | Fix ResNet101-IBN-a training | **+1.5-2.0pp** (enables #2) | 52.77% mAP | HIGH |
| **2** | Deploy 2-model ensemble | **+1.5-2.5pp** | 1 model | MEDIUM (after #1) |
| **3** | Train 384px ViT-B/16 from scratch | **+1.0-1.5pp** | 256px | MEDIUM |
| **4** | CID_BIAS per camera pair | **+0.5-1.0pp** | FIC only | MEDIUM |
| **5** | Re-enable reranking (after #2) | **+0.5-1.0pp** | Disabled | LOW |
| **6** | Hand-annotated zone polygons | **+0.5-1.0pp** | Auto-zones hurt | MEDIUM (manual) |
| **7** | DMT camera-aware training | **+1.0-1.5pp** | Not implemented | HIGH |
| **8** | Detection upgrade | **+0-0.3pp** | YOLO26m | LOW |
| **9** | PCA/processing tuning | **+0-0.3pp** | 384D optimal | NONE |
| **10** | Association params | **0pp** | Fully exhausted | NONE |

---

## 3. What SOTA Does Differently

| Pattern | AIC22 1st | AIC22 2nd | AIC21 1st | We have? |
|---------|:-:|:-:|:-:|:-:|
| 3+ ReID backbone ensemble | 5 models | 3 models | 3 models | **NO** (1) |
| 384×384 input | ✅ | ✅ | ✅ | **NO** (256) |
| IBN-a backbones | ✅ | ✅ | ✅ | ViT only |
| Camera-pair bias (CID_BIAS) | ROI masks | NPY | NPY | **NO** (FIC) |
| Reranking | Box-grained | k-reciprocal | k-reciprocal | **Disabled** |
| Camera-aware training (DMT) | ✅ | ✅ | ✅ | **NO** |
| Multiple loss functions | ID+tri+circle+cam | ID+tri+cam | ID+tri+cam | ID+tri |

---

## 4. Top 3 Concrete Actions

### Action 1: Fix ResNet101-IBN-a Training (BLOCKING everything)
- Debug why mAP stalls at 52.77%. Published baselines achieve 75-80%.
- Try: cosine annealing, warmup 10 epochs, random erasing p=0.5, label smoothing 0.1, 384×384 input, batch=64, 150 epochs
- **Target: mAP ≥ 70%**

### Action 2: Deploy 2-Model Score Fusion (after #1)
- Stage 2: vehicle2 enabled, save_separate=true
- Stage 4: secondary_embeddings.weight: 0.30
- Re-sweep sim_thresh + re-enable reranking

### Action 3: Train 384px ViT-B/16 from Scratch
- 09b notebook: input_size=[384, 384], bicubic pos_embed interpolation
- Same recipe as 09b v2 (mAP=80.14%)
- Deploy: update weights_path, refit PCA

---

## 5. Dead Ends to STOP Pursuing

| Approach | Why Stop | Evidence |
|----------|---------|---------|
| Association parameter tuning | Fully exhausted (220+ configs) | Experiment log |
| CSLS | -34.7pp catastrophic | v74 |
| Hierarchical clustering | -1.0 to -5.1pp | v54-56, v62 |
| FAC | -2.5pp | v26 |
| Feature concatenation | -1.6pp | Experiment log |
| CamTTA | Helps GLOBAL, hurts MTMC | v28-30 |
| Multi-scale TTA | Neutral/harmful | Multiple |
| Track smoothing/edge trim | Always harmful | Experiment log |
| Denoise | -2.7pp | v46 |
| mtmc_only submission | -5pp | Documented |
| Auto-generated zones | -0.4pp | v54-57 |
| PCA dimension search | 384D optimal | Experiment log |
| Ensemble with weak secondary | 52% dilutes signal | Current state |
| SGD for ResNet101-IBN-a | 30.27% mAP | v18 mrkdagods |
| Circle loss + triplet | Gradient conflict | v17 |
| K-reciprocal reranking | Always worse (old AND new conditions) | v25, v35 |
| Camera-pair normalization | Zero effect (FIC handles it) | v36 |

---

## 6. The Honest Truth

**Is TransReID bad?** No. 80.14% mAP is SOTA on VeRi-776. The problem is it's **alone**.

**Is fusion bad?** No. Score-level fusion is exactly what AIC22 2nd used. The secondary model quality (52%) is the problem.

**Is the association limited?** Marginally. ~1pp ceiling from CID_BIAS + zones + reranking. Not 5pp.

**Why has nothing worked?** Every attempt was either:
1. Association tuning (exhausted at 220+ configs)
2. Feature post-processing tricks (marginal by nature)
3. Weak secondary model fusion (52% mAP adds noise)

**What's never been done properly:** Training a competent second backbone. That's the lever.

**Bottom line:** The codebase is architecturally ready for 84%+. The implementation quality is high. The problem is purely feature quality — training a competent second model and moving to 384px. This is an **ML training problem**, not a software engineering problem.

---

## 7. Roadmap to 85%

| Phase | Target | Work |
|:---:|:---:|---|
| 1 | 78→82% | Fix IBN-a training (70%+ mAP) + deploy 2-model ensemble |
| 2 | 82→84% | Train 384px ViT + CID_BIAS + reranking |
| 3 | 84→85% | 3rd backbone + DMT training + hand-annotated zones |