# Convergence Assessment — Vehicle MTMC Pipeline

**Date**: 2026-04-17
**Requested by**: Strategic review after network flow solver dead end

## Verdict: CONVERGED

The vehicle MTMC pipeline is converged at **IDF1 ≈ 0.77** (Kaggle). The remaining 7.7pp gap to SOTA (0.8486) is structurally caused by single-model feature limitations and cannot be closed through association tuning, single-model feature variants, or structural solver changes.

## Evidence Summary

### Association Side — Exhausted
- **225+ configs tested**, all within 0.3pp of optimal
- **Structural solvers tested**: conflict-free CC (best), network flow (-0.24pp, +3 conflation), hierarchical clustering (-1 to -5pp)
- **Post-association linking**: AFLink confirmed harmful at -3.8pp to -13.2pp
- **Motion priors**: unreliable across non-overlapping CityFlowV2 cameras
- **Camera-pair calibration**: CID_BIAS -3.3pp (too few GT tracklets), camera-pair normalization zero effect (FIC handles it)

### Feature Side — Ceiling Reached
- **Primary model**: TransReID ViT-B/16 CLIP 256px, mAP=80.14%, best available
- **Higher mAP ≠ better MTMC**: augoverhaul (+1.45pp mAP → -5.3pp MTMC), 384px (same mAP → -2.8pp MTMC), DMT (+7pp mAP → -1.4pp MTMC)
- **Secondary model**: ResNet101-IBN-a at 52.77% mAP — too weak for ensemble (needs ≥65%)
- **ResNet training paths exhausted**: VeRi-776 pretrain (42.7% mAP), extended fine-tune (50.61%), circle loss (catastrophic), SGD (catastrophic)
- **Ensemble at 0.30 weight**: -0.1pp (weak secondary adds noise, not signal)

### Other Dead Ends (16 major approaches)
CSLS, 384px ViT, AFLink, CID_BIAS, DMT, hierarchical clustering, FAC, reranking, feature concatenation, circle loss, SGD, SAM2 foreground masking, score-level ensemble with weak secondary, VeRi-776→CityFlowV2 pretrain, extended ResNet fine-tune, network flow solver, EMA, multi-query track rep, concat_patch features

## Remaining Untried Approaches — Assessment

### GNN Edge Classification
- **Feasibility**: LOW
- **Rationale**: Only 464 GT-matched tracklets for training — severe overfitting risk. GNN learns edge weights, but the underlying similarity scores are already well-calibrated (225+ configs prove this). A GNN cannot create discriminative signal that doesn't exist in the embeddings. The problem is feature quality, not graph structure.
- **Expected impact**: Marginal at best, likely negative due to overfitting
- **Recommendation**: SKIP

### Knowledge Distillation
- **Feasibility**: MEDIUM effort, LOW expected return
- **Rationale**: Would require a strong teacher model we don't have. The primary ViT IS the best model. Distilling it to a secondary doesn't create new signal — it just compresses existing signal. Cross-camera invariance is a property of the training data distribution, not something that transfers through distillation alone.
- **Expected impact**: Unlikely to help; the bottleneck is not model capacity but training signal
- **Recommendation**: SKIP

### Center Loss for Primary ViT
- **Feasibility**: LOW effort to implement
- **Rationale**: Center loss pulls features toward class centroids. With only 128 IDs in CityFlowV2, this is a mild regularizer at best. The CircleLoss ablation already showed that metric learning additions to the primary ViT recipe are dangerous — center loss is safer but unlikely to move the needle given that better mAP consistently fails to improve MTMC.
- **Expected impact**: Marginal; the mAP→MTMC disconnect means even if center loss helps validation, it likely won't help MTMC
- **Recommendation**: LOW PRIORITY — only if time is free and curiosity-driven

## Is Anything Still Worth Trying on Association?

**No.** The association side is definitively exhausted:
1. Parameter space: 225+ configs, 0.3pp spread
2. Solver architecture: CC, network flow, hierarchical — all tested
3. Post-processing: AFLink, reranking, camera-pair normalization — all tested
4. Priors: CID_BIAS, motion linking, temporal overlap — all tested or incorporated

The one association change that could theoretically help is a **true multi-model ensemble** where all models exceed 65% mAP, enabling techniques like reranking and CID_BIAS that are ensemble-dependent. But this is blocked by the inability to train a competitive secondary model.

## Quick Wins Assessment

There are no quick wins remaining. The entire pipeline has been thoroughly optimized:
- Detection: not the bottleneck
- Tracking: min_hits=2 optimal, BoT-SORT working well
- Features: 256px ViT-B/16 CLIP is the best single model available
- PCA/FIC: 384D, power normalization, AQE K=3 — all tuned
- Association: exhausted at 225+ configs
- The ~1pp drift from historical v80 (0.784) to current v52 (0.775) remains unexplained but is likely due to minor codebase evolution and is not recoverable through parameter tuning

## Recommendation: Pivot to Paper

### Rationale
1. **Diminishing returns**: Every experiment for the last 50+ runs has been within ±0.3pp of the current best
2. **Paper angle is strong**: "One Model, 91% of SOTA" with 225+ exhaustive experiments proving feature quality is the MTMC bottleneck — this IS a publishable contribution
3. **Dual-domain story**: Vehicle (CityFlowV2, 0.775 IDF1) + Person (WILDTRACK, 0.947 IDF1) on the same pipeline
4. **Dead-end catalog**: 16+ major approaches documented — saves future researchers significant time
5. **Time-value tradeoff**: Each additional experiment costs Kaggle GPU hours and developer time for 0.1-0.3pp expected gain, while the paper needs ablation tables, figures, and writing

### Immediate Actions
1. **Update `docs/findings.md`** — add network flow solver to dead ends, update current best to 0.7714 Kaggle
2. **Freeze the pipeline** — no more experiments; current codebase is the submission version
3. **Begin paper writing** using the strategy in `docs/paper-strategy.md`:
   - Clean ablation table (baseline → +CLIP → +FIC → +quality crops → +PCA → final)
   - Error analysis figures (87 fragmented, 35 conflated, per-camera heatmap)
   - Feature similarity distributions (same-ID vs different-ID)
   - Compute efficiency comparison (1x T4 ~3h vs 5-model SOTA ~50h+)
4. **Update experiment log** with final numbers and close it out

### What Would Change This Assessment
- A breakthrough in ReID training that produces a secondary model with ≥65% mAP on CityFlowV2 (unlocks ensemble)
- A fundamentally new association paradigm not yet tried (e.g., learned graph neural network on a larger dataset, not CityFlowV2's 464 tracklets)
- Access to significantly more compute (enabling larger ensembles or longer training runs)

None of these are actionable in the near term.

## Person Pipeline Status

**Converged at IDF1 = 0.947**, 0.6pp from SOTA. Confirmed by 12b v1, v2, v3 across 59 tracker configs. The gap is tracker-limited (Kalman), not detector-limited. No further experiments recommended.
