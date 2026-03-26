# MTMC Breakthrough Plan — IDF1 0.784 → 0.90+

**Date**: 2026-03-19
**Baseline**: IDF1 = 0.784 (Kaggle v80 / 10c v44 / ali369, current best); historical local 0.8297 claim is unverifiable in the current log
**SOTA**: AIC21 1st place IDF1 = 0.841
**Target**: IDF1 ≥ 0.90 (aspirational)
**Hardware**: GTX 1050 Ti 4GB (local), Kaggle T4 16GB (training/inference)

---

## Current State Summary

### What Works Well
- TransReID ViT-B/16 (CLIP) — 82.2% mAP VeRi-776, 90.5% mAP Market-1501
- FIC per-camera whitening (covariance bug fixed v46)
- QE with self-exclusion fix (v46.1), k=2-3
- Exhaustive cross-camera cosine matching
- conflict_free_cc graph clustering
- Gallery expansion for orphan recovery
- Stationary filter (d150), mtmc_only_submission
- 100% GT recall (0 unmatched GT IDs)

### Algorithm Ceiling Confirmed
220+ config experiments and the tracked Kaggle runs through v80 place the current recorded ceiling at **78.4% IDF1**. No config/threshold change has closed the remaining gap; the bottleneck is **embedding quality**, not association algorithms.

### Error Profile (558 trajectories, best local run)
| Error Type | Count | Impact |
|---|---|---|
| **Fragmented GT IDs** | **87** | Under-merging dominates (2.5× more than conflation) |
| Conflated Pred IDs | 35 | Over-merging |
| Unmatched GT IDs | 0 | Perfect recall |
| Single-cam trajectories | 317 (57%) | All are FP in CityFlowV2 eval |

### Per-Scene Performance Gap
- **S01 average IDF1 = 91.6%** — already excellent
- **S02 average IDF1 = 80.1%** — 11.5pp worse, S02_c006 is catastrophic (IDF1=74.0%)
- S02_c006: FP ratio 6.86×, GT covers only 43% of video frames

### Critical Config Finding: 10c Missing Dataset Config
Notebook 10c (stages 4-5) loads **only default.yaml** — it does NOT merge cityflowv2.yaml. Many carefully tuned cityflowv2.yaml values are silently ignored. This is a source of suboptimality.

---

## Improvement Tiers

### Tier 0: FREE GAINS — Config Fixes (Est. +0.5-1.5pp)

These require zero code changes — just fix configs that are wrong or misaligned.

| # | Fix | Details | Est. Gain | Status |
|---|---|---|---|---|
| 0a | **Add `--dataset-config` to 10c notebook** | Stale blocker note. The current Kaggle chain already reaches 78.4% IDF1, so this is no longer the active explanation for the baseline. | +0.3-0.5pp | **TESTED / SUPERSEDED**: current 10c runs established a higher baseline than the old 0.789 claim. |
| 0b | **Run v47 features on Kaggle** | Kaggle evaluation moved well beyond the stale v42 baseline. The current best is v80 at 78.4% IDF1 with `min_hits=2`. | +2-4pp (stale baseline) | **COMPLETED**: Kaggle chain advanced through v80; stale 0.789 baseline replaced by 0.784 current best. |
| 0c | **Enable 384×384 resolution** | Inference-only 384×384 was tested. | +0.5-1.0pp | **TESTED**: v50 hurt by -1.3pp IDF1; only worth revisiting with a model trained natively at 384. |
| 0d | **Enable multi-scale TTA** | Feature-side augmentation stack is already part of the current Stage 2 pipeline. | +0.3-0.5pp | **TESTED**: no breakthrough recorded in the current 78.4% baseline path. |

### Tier 1: LOW-HANGING FRUIT — Code Exists, Needs Testing (Est. +1-3pp)

Features that are fully implemented in the codebase but have no config entries or were never properly tested with the current v47 pipeline.

| # | Feature | Location | Status | Est. Gain |
|---|---|---|---|---|
| 1a | **CSLS hubness reduction** | `pipeline.py L465` | **TESTED**: v74 was catastrophic at -34.7pp; keep off. | +0.3-0.5pp |
| 1b | **Intra-camera Stage4 ReID merge** | `pipeline.py L544` | **COMPLETED / TESTED**: v72 sweep found threshold=0.80, gap=30 best; carried into the current best chain. | +0.3-0.5pp |
| 1c | **Cluster post-verification** | `pipeline.py L655` | **TESTED**: post-processing on the well-conditioned graph either did nothing or hurt. | +0.3-0.5pp |
| 1d | **Sub-cluster temporal splitting** | Not implemented | **TESTED (conceptually)**: temporal post-processing is not a near-term win on the current graph; keep lower priority. | +0.5-1.0pp |
| 1e | **Per-pair adaptive thresholds** | `pipeline.py L516` | **TESTED / LOW PRIORITY**: exhaustive association sweeps did not beat the fixed tuned threshold path (`sim_thresh=0.53`). | +0.3pp |
| 1f | **Hierarchical centroid expansion** | `pipeline.py L618` | **TESTED**: v54-v56 and v62 hurt by -1.0 to -5.1pp; keep disabled. | +0.5-1.0pp |

### Tier 2: EMBEDDING QUALITY — The Primary Bottleneck (Est. +2-5pp)

#### 2a. Knowledge Distillation (ViT-L → ViT-B) — **HIGHEST PRIORITY**
- **What**: Train a ViT-L/16 (CLIP or EVA-02) teacher on CityFlowV2, then distill into our existing ViT-B student
- **Why**: Every AIC 2024 top-3 team used ViT-L teacher distillation. +2-4% mAP is the largest single gain available
- **How**: α·KD_loss(KL-div on logits + MSE on features) + (1-α)·task_loss
- **Compute**: Teacher 6-8h Kaggle T4x2, Student KD 3-4h
- **Est. gain**: +1.0-2.0pp IDF1
- **Complexity**: Medium

#### 2b. Multi-Model Ensemble — Add CNN Diversity
- **What**: Train ResNet50-IBN or Swin-T on CityFlowV2 alongside TransReID. Average/concat embeddings.
- **Why**: CNN vs ViT capture different features (local texture vs global context). AIC21 used 3-model ensemble.
- **How**: Score-level fusion (weighted average of similarity scores) or feature-level (concat + PCA)
- **Note**: Previously tested TransReID+OSNet ensemble in v26 and it HURT (-1.6pp). But that was with weaker pipeline. Re-test with v47 features and score-level fusion instead of concat.
- **Est. gain**: +0.5-1.5pp IDF1
- **Complexity**: Medium

#### 2c. Part-Aware Transformer (PAT) ReID Head
- **What**: Replace simple CLS/GeM pooling with cross-attention between learnable part tokens and image patches
- **Why**: Vehicles have non-uniform part layouts (license plate, wheels, roof). Part-based features are more discriminative.
- **Reference**: arXiv 2307.02797
- **Est. gain**: +0.5-1.0pp IDF1
- **Complexity**: Medium

#### 2d. SAM2 Foreground Masking
- **What**: Use SAM2 to segment vehicle foreground before ReID. Remove background clutter from crops.
- **Reference**: AIC 2024 2nd place used this
- **Est. gain**: +0.3-0.5pp mAP
- **Complexity**: Medium

### Tier 3: LEARNED ASSOCIATION — Paradigm Shift (Est. +2-5pp)

#### 3a. GNN Edge Classification — **BIGGEST STRUCTURAL IMPROVEMENT**
- **What**: Replace threshold-based graph clustering with a learned GNN that predicts edge weights (same-ID / different-ID) via message passing
- **Why**: Eliminates hand-tuning of 15+ thresholds/weights. Can learn camera-pair-specific biases, transition patterns, and non-linear appearance interactions. Our 36+ experiments prove hand-crafting has hit its ceiling.
- **Reference**: LMGP (arXiv 2104.09018) — specifically for MTMC, +3-5% IDF1 over hand-crafted
- **Training data**: CityFlowV2 GT associations (we have this)
- **Est. gain**: +1.0-3.0pp IDF1
- **Complexity**: High (PyTorch Geometric + training pipeline)

#### 3b. Transformer Cross-Camera Matcher
- **What**: Learn a transformer to predict assignment costs between tracklets
- **Reference**: SUSHI (arXiv 2212.03038)
- **Est. gain**: +1.0-2.0pp IDF1
- **Complexity**: High

### Tier 4: DOMAIN ADAPTATION (Est. +1-2pp)

#### 4a. Camera-Specific Test-Time Adaptation
- **What**: Adapt BN layers per camera at test time using running statistics
- **Why**: We already have camera_bn. TTA makes it automatic and adapts the full normalization.
- **AIC 2024**: 1st and 3rd place teams used this
- **Est. gain**: +0.5-1.0pp IDF1
- **Complexity**: Low

#### 4b. Timestamp Bias Correction
- **What**: Learn per camera-pair timestamp offsets. CityFlowV2 cameras have slight sync errors.
- **How**: Grid search offset per pair, maximize ST agreement
- **Est. gain**: +0.3-0.5pp IDF1
- **Complexity**: Low

#### 4c. Zone-Based ST Constraints (Refined)
- **What**: Hand-annotate tighter entry/exit zone polygons per camera (vs auto-clustered k-means centers)
- **Why**: AIC21 1st place relied on this. Our auto-zones are too coarse.
- **Est. gain**: +0.5-1.5pp IDF1
- **Complexity**: Medium (annotation work)

### Tier 5: ADVANCED TEMPORAL (Est. +1-3pp)

#### 5a. Temporal ReID Transformer
- **What**: Replace quality-weighted crop pooling with a temporal transformer that models appearance dynamics across a tracklet's lifetime
- **Why**: A car seen head-on in cam1 and from behind in cam2 — temporal modeling learns to bridge these
- **Est. gain**: +1.0-2.0pp IDF1
- **Complexity**: High

#### 5b. Traffic Pattern Learning
- **What**: Learn traffic flow distributions per camera pair (not just transition times)
- **How**: Extend existing ST validator with directional flow priors
- **Est. gain**: +0.5-1.0pp IDF1
- **Complexity**: Medium

---

## Implementation Roadmap

### Phase 1: Immediate (Days) — Free Gains + Stale Baseline
**Branch: `v48-config-fixes`**
1. 10c config-path concern: stale note, no longer the main blocker in the current 78.4% baseline
2. Kaggle feature chain: completed beyond the stale v42 baseline, culminating in v80 = 78.4%
3. 384×384 inference: tested and harmful at current training resolution (-1.3pp)
4. CSLS / intra-cam merge / cluster verify: tested; CSLS hurt badly, intra-cam merge is already absorbed, cluster verify did not help
5. **Status**: largely executed and invalidated as a “free gains” phase. Current Kaggle best is already **0.784**, with CSLS failing, 384×384 inference hurting, and intra-cam merge already folded into the best path.

### Phase 2: Quick Wins (1-2 Weeks) — Structural Improvements
**Branch: `v49-association-improvements`**
1. Implement sub-cluster temporal splitting
2. Re-test hierarchical expansion with v47 features
3. Implement timestamp bias correction (CPU grid search)
4. Camera-specific TTA for better normalization
5. **Expected: +1-2pp over Phase 1**

### Phase 3: Embedding Revolution (2-4 Weeks) — Training on Kaggle
**Branch: `v50-embedding-quality`**
1. Train ViT-L teacher on CityFlowV2 → distill to ViT-B
2. Train ResNet50-IBN as complementary CNN model
3. Score-level ensemble fusion
4. 384×384 with part-aware pooling
5. **Expected: +1-3pp over Phase 2**

### Phase 4: Learned Association (4-8 Weeks) — Paradigm Shift
**Branch: `v51-learned-association`**
1. Build GNN edge classifier using CityFlowV2 GT
2. Replace threshold+graph-solver with learned message passing
3. End-to-end differentiable association
4. **Expected: +1-3pp over Phase 3**

---

## Cumulative Projections

| Phase | IDF1 Range | Key Technique |
|---|---|---|
| Current | 0.784 | Kaggle v80 baseline |
| Phase 1 | 0.784 | Most “free gains” already tested; no latent jump from the stale 0.789 baseline |
| Phase 2 | 0.84-0.86 | Association improvements + TTA |
| Phase 3 | 0.86-0.89 | Knowledge distillation + ensemble |
| Phase 4 | 0.89-0.92 | Learned GNN association |

**Conservative estimate**: IDF1 ~0.87 (Phases 1-3)
**Optimistic estimate**: IDF1 ~0.92 (all phases)

---

## Techniques Confirmed NOT Worth Pursuing

| Technique | Why Skip |
|---|---|
| FAC (feature augmentation) | Tested v26, harmful -2.5pp with strong features |
| k-reciprocal re-ranking | Tested, compresses similarity range with PCA features |
| Reciprocal best-match | Tested v27, -1.9pp |
| VehicleX synthetic data | CLIP pretraining already provides superior initialization |
| End-to-end MTMC | No system beats modular pipelines as of 2025 |
| DINOv2 backbone swap | CLIP-ReID outperforms for instance discrimination |
| SOLIDER Swin-B | Doesn't fit 4GB VRAM; CLIP ViT-B already matches it |
| Grounding-DINO detection | Detection not the bottleneck (0 unmatched GT IDs) |

---

## Key Files Reference

| Purpose | Path |
|---|---|
| Default config | `configs/default.yaml` |
| CityFlowV2 config | `configs/datasets/cityflowv2.yaml` |
| 10a notebook (stages 0-2) | `notebooks/kaggle/10a_stages012/` |
| 10c notebook (stages 4-5) | `notebooks/kaggle/10c_stages45/` |
| Stage 4 association | `src/stage4_association/pipeline.py` |
| TransReID model | `src/stage2_features/transreid_model.py` |
| ReID wrapper | `src/stage2_features/reid_model.py` |
| FIC implementation | `src/stage4_association/fic.py` |
| Query expansion | `src/stage4_association/query_expansion.py` |
| Zone scoring | `src/stage4_association/zone_scoring.py` |
| Camera bias | `src/stage4_association/camera_bias.py` |
