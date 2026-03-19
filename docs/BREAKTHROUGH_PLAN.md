# MTMC Breakthrough Plan — IDF1 0.83 → 0.90+

**Date**: 2026-03-19
**Baseline**: IDF1 = 0.8297 (local), 0.789 (Kaggle v42, stale)
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
36+ experiments across 6 rounds confirmed IDF1=0.8297 ceiling. No config/threshold change helps further. The bottleneck is **embedding quality**, not association algorithms.

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

| # | Fix | Details | Est. Gain |
|---|---|---|---|
| 0a | **Add `--dataset-config` to 10c notebook** | 10c currently skips cityflowv2.yaml, falling back to default.yaml for most stage4 params. Many tuned values (gallery thresholds, weights, bridge prune) are silently ignored. | +0.3-0.5pp |
| 0b | **Run v47 features on Kaggle** | v47 (concat_patch 1536D, 48 crops, PCA 384D, power_norm 0.5) is committed but never evaluated. 10a already has all overrides. Current Kaggle score (0.789) is from v42. | +2-4pp (stale baseline) |
| 0c | **Enable 384×384 resolution** | Code ready (v47.1 bicubic pos_embed interpolation). Change `input_size: [384, 384]` in 10a override. Doubles spatial detail. | +0.5-1.0pp |
| 0d | **Enable multi-scale TTA** | 10a already has `multiscale_sizes=[[224,224],[288,288]]` override. Verify it's active and add [384,384] if using 384 base. | +0.3-0.5pp |

### Tier 1: LOW-HANGING FRUIT — Code Exists, Needs Testing (Est. +1-3pp)

Features that are fully implemented in the codebase but have no config entries or were never properly tested with the current v47 pipeline.

| # | Feature | Location | Status | Est. Gain |
|---|---|---|---|---|
| 1a | **CSLS hubness reduction** | `pipeline.py L465` | Fully coded, zero config entries, NEVER tested | +0.3-0.5pp |
| 1b | **Intra-camera Stage4 ReID merge** | `pipeline.py L544` | Fully coded, zero config entries. 10c has override enabling it (threshold=0.75, gap=60s). Verify it works. | +0.3-0.5pp |
| 1c | **Cluster post-verification** | `pipeline.py L655` | Ejects weakly-connected cluster members. No config. | +0.3-0.5pp |
| 1d | **Sub-cluster temporal splitting** | Not implemented | Split large clusters with temporal gaps. AIC21 technique. | +0.5-1.0pp |
| 1e | **Per-pair adaptive thresholds** | `pipeline.py L516` | Coded but no config entry | +0.3pp |
| 1f | **Hierarchical centroid expansion** | `pipeline.py L618` | Full AIC21/22 code, `enabled: false`. Re-test with v47 features. | +0.5-1.0pp |

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
1. Fix 10c to load cityflowv2.yaml dataset config
2. Run v47 features end-to-end on Kaggle (est. jump to ~0.83+ from stale 0.789)
3. Enable 384×384 + multi-scale TTA in 10a
4. Test CSLS, intra-cam merge, cluster verify with simple config additions
5. **Expected total: +3-5pp from stale baseline (0.789 → ~0.83-0.84)**

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
| Current | 0.8297 | Algorithm ceiling |
| Phase 1 | 0.83-0.84 | Config fixes + v47 features on Kaggle |
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
