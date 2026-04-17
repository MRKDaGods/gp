# Next Experiment After AFLink — Strategic Analysis

**Date**: 2026-04-17
**Current Best**: MTMC IDF1 = 0.775 (10c v52, reproducible), 0.7714 (Kaggle controlled retest)
**SOTA Target**: 0.8486 (AIC22 1st place, 5-model ensemble)
**Gap to SOTA**: 7.36pp
**Association configs tested**: 225+ (exhausted, all within 0.3pp of optimal)

---

## 1. Situation Assessment

### What's Exhausted
- **Association parameter tuning**: 225+ configs across 14 dimensions, all within 0.3pp
- **Single-model feature improvements**: augoverhaul (+1.45pp mAP → -5.3pp MTMC), 384px (-2.8pp), DMT (-1.4pp), multi-query (-0.1pp), concat_patch (-0.3pp)
- **Post-association linking**: AFLink (-3.8pp to -13.2pp), hierarchical clustering (-1 to -5pp)
- **Feature-space transforms**: CSLS (-34.7pp), FAC (-2.5pp), reranking (always hurts), CID_BIAS (-3.3pp)
- **Ensemble with weak secondary**: ResNet101-IBN-a at 52.77% mAP gives -0.1pp at 0.30 fusion weight
- **Loss function changes**: CircleLoss (inf loss, 18.45% mAP), circle+triplet (16-30% mAP)
- **ResNet training paths**: VeRi-776 pretrain (42.7% mAP), extended fine-tuning (50.61% mAP), SGD (30.27% mAP)
- **Preprocessing**: SAM2 foreground masking (-8.7pp), CamTTA (hurts MTMC)
- **Model averaging**: EMA (identical to base model)

### Confirmed Dead Ends (18 total)
CSLS, 384px ViT deployment, AFLink, CID_BIAS, DMT, hierarchical clustering, FAC, reranking, feature concatenation, circle loss, SGD, SAM2, score-level ensemble (weak secondary), VeRi-776→CityFlowV2 pretrain, extended ResNet fine-tuning, CamTTA, EMA, augoverhaul+CircleLoss

### Critical Insight
**Higher single-camera mAP does NOT translate to better MTMC IDF1.** Three experiments confirm: augoverhaul (+1.45pp mAP → -5.3pp MTMC), 384px (+same mAP → -2.8pp MTMC), DMT (+7pp mAP → -1.4pp MTMC). The MTMC graph needs features with **clean, thresholdable cross-camera similarity distributions**, not just good ranking on the validation split.

### Error Profile (from v52 baseline)
| Error Type | Count | Interpretation |
|---|---|---|
| Fragmented GT IDs | 87 | Under-merging dominates (2.5× conflation) |
| Conflated Pred IDs | 35 | Over-merging from transitive chains |
| Unmatched GT IDs | 0 | Perfect recall |
| Per-scene gap | S01=91.6% vs S02=80.1% IDF1 | Scene 2 (harder lighting/angles) is the bottleneck |

---

## 2. Analysis of Remaining Viable Approaches

### Tier 1: Low-Effort, Potentially Positive (days, no retraining)

#### A. Timestamp Bias Correction
- **Description**: Grid-search per-camera-pair timestamp offsets (±5s in 0.5s steps) to correct potential CityFlowV2 sync errors, maximizing ST agreement on high-confidence appearance matches
- **Expected impact**: +0.0 to +0.5pp (if cameras have sync errors) or +0.0pp (if already synced)
- **Effort**: Low — simple grid search script, no training
- **Risk**: Very low — does nothing if cameras are synced
- **Status**: Never measured actual camera sync accuracy
- **Why promising**: S02 cameras underperform by 11.5pp; timing errors could cause ST scores to penalize correct matches in difficult scenes

#### B. Per-Camera CLAHE Tuning
- **Description**: Tune CLAHE clip_limit per camera based on exposure analysis, particularly S02_c006 (worst camera at 74% IDF1)
- **Expected impact**: +0.0 to +0.3pp
- **Effort**: Very low — config-only
- **Risk**: Very low
- **Status**: Global clip_limit=2.5 applied uniformly, never analyzed per-camera

### Tier 2: Medium-Effort, Structural Association Change (1-2 weeks)

#### C. Network Flow / Min-Cost-Max-Flow Solver ★ RECOMMENDED
- **Description**: Replace greedy conflict_free_cc with a globally optimal matching formulation. Current CC greedily adds edges in descending similarity order — this creates transitive chains where A↔B and B↔C force A↔C even when sim(A,C) is low. Network flow enforces global consistency.
- **Implementation**: Pairwise bipartite matching (scipy.optimize.linear_sum_assignment) across camera pairs, then merge results with cross-pair consistency resolution
- **Expected impact**: +0.3 to +1.0pp — specifically targets the 35 conflated IDs from transitive merge errors
- **Effort**: Medium — scipy provides the core algorithm, needs multi-camera orchestration
- **Risk**: Medium — the person pipeline's GlobalOptimalTracker was -3.5pp, but that was frame-level assignment competing with Kalman prediction, not cross-camera identity matching. The vehicle MTMC case is a better structural fit for global optimal assignment because there's no temporal prediction component to lose.
- **Why promising**: This is the ONLY untested structural association algorithm. Everything else was threshold tuning on the same greedy algorithm. Network flow is fundamentally different — it finds the globally optimal assignment given all edge weights simultaneously, which CC cannot do.
- **Key difference from person pipeline failure**: The GlobalOptimalTracker failed because Kalman filtering handles temporal dynamics better than immediate frame-level costs. Vehicle MTMC cross-camera association has no temporal dynamics — it's a pure matching problem, which is exactly what flow/assignment algorithms excel at.

#### D. Knowledge Distillation (ViT-L → ViT-B) — Retry
- **Description**: Fix the failed 09c KD implementation (22% mAP due to projector bug) and retry with proper setup: nn.Linear(1024, 768) projector, temperature T=2, MSE feature alignment + KL divergence, initialize student from current best 256px checkpoint
- **Expected impact**: +0.5 to +2.0pp IF the distilled features improve cross-camera invariance (not just mAP)
- **Effort**: Medium-High — 09c exists as starting point but needs substantial fixes
- **Risk**: HIGH — given that mAP improvements do not transfer to MTMC, even a successful KD that improves mAP may not improve MTMC IDF1. The only way KD helps is if the ViT-L teacher provides more viewpoint-invariant features that the student inherits.
- **Caveat**: Must evaluate MTMC IDF1 end-to-end, not just mAP

### Tier 3: High-Effort, Paradigm Shifts (3-6 weeks)

#### E. GNN Edge Classification
- **Description**: Train a GNN/MLP to predict same-identity edges from pair features (cosine sim, HSV sim, temporal gap, camera pair ID, etc.), replacing the fixed similarity_threshold
- **Expected impact**: +1.0 to +3.0pp (per LMGP paper)
- **Effort**: High — PyTorch Geometric, training pipeline, GT label generation
- **Risk**: HIGH — CityFlowV2 has only 128 IDs and 464 GT-matched tracklets. Overfitting is a severe risk (same reason CID_BIAS at -3.3pp failed — too few GT matches for learned camera-pair biases). LMGP results were on datasets with 10× more identities.
- **Mitigation**: Use very simple pair features (5-8D), shallow 2-layer MLP, strong regularization, leave-one-ID-out cross-validation

#### F. Temporal Attention for Tracklet Embedding
- **Description**: Replace quality-weighted crop averaging with a small transformer that models appearance change across a tracklet's lifetime
- **Expected impact**: +0.5 to +1.5pp
- **Effort**: High — requires architecture change + training on CityFlowV2 tracklets
- **Risk**: Medium — proven in video ReID (MARS), but needs sufficient data

#### G. Part-Aware Transformer (PAT) Pooling
- **Description**: Add learnable part tokens (front/rear/side/roof) with cross-attention against ViT patch tokens, creating part-specific features
- **Expected impact**: +0.5 to +1.0pp
- **Effort**: High — architecture change + retraining
- **Risk**: Medium-High — given augoverhaul regression, architecture changes that improve mAP may not improve MTMC

---

## 3. Recommended Next Experiment: Network Flow Solver

### Goal
Replace the greedy `conflict_free_cc` graph algorithm with a globally optimal min-cost matching solver to fix transitive merge errors (35 conflated IDs) and improve cluster quality.

### Hypothesis
The current `conflict_free_cc` algorithm processes edges greedily in descending similarity order. This creates transitive chains: if sim(A,B)=0.8 and sim(B,C)=0.7, both edges are added, implicitly grouping A↔C even if sim(A,C)=0.3. A globally optimal solver would reject this cluster because the total cost is suboptimal. With 35 conflated IDs, fixing even half would yield +0.5-1.0pp.

### Implementation Plan

#### Step 1: Add network flow solver to graph_solver.py
- New function `min_cost_flow_solver(similarity_dict, tracklet_metadata, threshold)`
- For each camera pair (i, j), build bipartite cost matrix from cross-camera similarities
- Run `scipy.optimize.linear_sum_assignment` on negative similarity (minimization)
- Keep only assignments above the similarity threshold
- Result: set of pairwise assignments per camera pair

#### Step 2: Multi-camera merging
- After pairwise bipartite matching, merge results transitively BUT with verification:
  - If A↔B (from cam1-cam2 matching) and B↔C (from cam2-cam3), accept A↔C only if sim(A,C) > `merge_verify_threshold`
  - This prevents the exact transitive chain problem that CC has
- Alternative: solve all cameras simultaneously using a multi-partite flow formulation

#### Step 3: Handle multi-appearance tracklets
- Some vehicles appear in the same camera twice (leave and re-enter)
- Need to allow multiple tracklets per camera in the same identity cluster, as long as they don't temporally overlap (same constraint as current conflict_free_cc)

#### Step 4: Integration
- Add `algorithm: "network_flow"` option in `stage4.association.graph`
- Config: `stage4.association.graph.algorithm=network_flow`
- Additional params: `merge_verify_threshold` (default: same as similarity_threshold)

#### Step 5: Testing
- Run locally with `python scripts/run_pipeline.py` on existing v52 features
- Compare against conflict_free_cc baseline (MTMC IDF1 = 0.775)
- Sweep `merge_verify_threshold` from 0.40 to 0.60

### Config Overrides
```
stage4.association.graph.algorithm=network_flow
stage4.association.graph.similarity_threshold=0.50
stage4.association.graph.merge_verify_threshold=0.45
```

### Files to Modify
1. `src/stage4_association/graph_solver.py` — add `min_cost_flow_solver()` and `network_flow_solver()`
2. `src/stage4_association/pipeline.py` — wire up new algorithm option
3. `configs/default.yaml` — add `merge_verify_threshold` parameter

### Expected Impact
- **Best case**: +0.5 to +1.0pp MTMC IDF1 (fix half of 35 conflated IDs)
- **Likely case**: +0.2 to +0.5pp (modest improvement from better global consistency)
- **Worst case**: -0.2pp (if one-to-one constraints are too strict for multi-camera re-entries)

### Risks
1. **Bipartite assumption**: CityFlowV2 has 6 cameras with overlapping FOVs. Pairwise bipartite matching then merging may miss multi-way interactions. Mitigation: verification step catches bad transitive chains.
2. **Multi-appearance vehicles**: A vehicle leaving and re-entering a camera creates two tracklets that should be in the same cluster. Need temporal non-overlap check (same as current CC).
3. **Overhead from person pipeline failure**: The GlobalOptimalTracker was -3.5pp for person tracking. But that was frame-level assignment vs. Kalman prediction — a fundamentally different problem. Vehicle MTMC is a static matching problem, which is the natural domain for assignment algorithms.
4. **Computational cost**: With 941 tracklets and 6 cameras, the bipartite matrices are at most ~150×150 per pair. scipy handles this in milliseconds.

### Measurement
- **Command**: `python scripts/run_pipeline.py --config configs/default.yaml configs/datasets/cityflowv2.yaml --stages 4,5 --override stage4.association.graph.algorithm=network_flow`
- **Success metric**: MTMC IDF1 > 0.775
- **Failure metric**: MTMC IDF1 < 0.770 (worse than baseline by >0.5pp)

### Secondary Quick Test: Timestamp Bias Correction
As a parallel low-effort test:
1. Write a script that, for all high-confidence same-identity pairs (sim > 0.70), measures the actual time gap distribution per camera pair
2. Compare against the configured transition priors in `spatial_temporal.py`
3. If systematic offsets are found, correct them and re-run Stage 4-5
4. This takes <1 day and is completely independent of the network flow experiment

---

## 4. What NOT to Try Next

| Approach | Reason |
|---|---|
| More association threshold sweeps | 225+ configs exhausted, all within 0.3pp |
| Single-model mAP improvements | mAP ≠ MTMC IDF1 (proven 3× over) |
| AFLink variants | Structurally harmful on CityFlowV2 (-3.8pp even with tight constraints) |
| ResNet101-IBN-a improvements | Path saturated at 52.77% mAP, all training variants tried |
| CircleLoss in any configuration | Catastrophic instability with this TransReID recipe |
| 384px deployment or training | Viewpoint-specific textures hurt cross-camera matching |
| SAM2 or foreground masking | Removes useful context cues (-8.7pp) |
| CamTTA | Helps within-camera, hurts MTMC |

---

## 5. Beyond the Next Experiment: Strategic Outlook

The 7.36pp gap is realistically decomposed as:
- **~4-5pp**: Single model vs 3-5 model ensemble (feature diversity)
- **~1-2pp**: Structural association improvements (network flow, GNN)
- **~0.5-1pp**: Spatio-temporal refinements (timestamps, zones)

**Without a viable second ensemble model, the realistic ceiling is ~78-79% MTMC IDF1.** Network flow + timestamp correction might push to ~78-79%. To reach 82%+, a second ReID model with ≥65% mAP on CityFlowV2 is essential. The most viable path to that is:
1. Train a fundamentally different architecture (SwinTransformer-T or ConvNeXt-T with IBN-a) on the same VeRi-776→CityFlowV2 pipeline
2. Or successfully execute KD from ViT-L to improve the primary model's cross-camera invariance
3. Or find a way to make the augmentation overhaul features work for MTMC (understand why +1.45pp mAP → -5.3pp MTMC)

The network flow experiment is the best next step because it's the only untested structural change that addresses a known error mode (conflation) without requiring model retraining.
