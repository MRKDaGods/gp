# Cross-Camera (MTMC) Performance Analysis

**Date**: 2026-03-24
**Current MTMC IDF1**: 76.5% (v85, CamTTA experiment)
**Current GLOBAL IDF1**: 80.5% (v85)
**SOTA Target**: 84.1% (AIC21 1st place)
**True Gap**: 7.6pp (MTMC-to-MTMC comparison)

## 1. Critical Metric Clarification

### GLOBAL IDF1 ≠ MTMC IDF1

| Metric | Function | Method | Measures |
|--------|----------|--------|----------|
| **GLOBAL IDF1** | `evaluate_mot()` | Per-camera accumulators merged via TrackEval | Within + cross camera |
| **MTMC IDF1** | `evaluate_mtmc()` | Single global accumulator, globally-unique GT IDs | Cross-camera only |

AIC21 SOTA (84.1%) uses the single-global-accumulator protocol = our `evaluate_mtmc`.
- Our comparable number is **76.5% MTMC IDF1**, NOT 80.5% GLOBAL IDF1
- v80 "best" of 78.4% was likely GLOBAL IDF1
- **True gap to SOTA: 7.6pp**

## 2. Error Profile (v85)

| Error Type | Count | Meaning |
|---|---|---|
| Fragmented GT IDs | 44 | Under-merging — features too dissimilar across cameras |
| Conflated pred IDs | 26 | Over-merging — threshold too loose for some pairs |
| Unmatched GT IDs | 10 | Missed tracklets or ST filtering too aggressive |
| Unmatched pred IDs | 4 | Spurious tracklets |
| ID Switches | 152 (MTMC) | Frame-level identity swaps |

**Fragmentation:Conflation = 1.69:1** → system under-merges. Root cause: same-identity tracklets across cameras lack sufficient similarity. Feature quality problem, not threshold.

## 3. CamTTA Was Counterproductive

CamTTA adapts BN stats per camera → features become camera-specific:
- Within-camera: more discriminative → GLOBAL IDF1 ↑ (80.5%)
- Across cameras: more camera-variant → MTMC IDF1 ↓ (76.5%)
- FIC whitening in Stage 4 is the correct camera-bias correction (post-extraction)
- **Verdict: Keep CamTTA disabled for MTMC**

## 4. Gap Attribution

| Source | IDF1 Cost | Fix |
|--------|-----------|-----|
| Feature quality (256px, single model) | ~4-5pp | 384px model, ensemble |
| Spatio-temporal imprecision | ~1.5-2pp | Hand-annotated zones, timestamp calibration |
| Over-merging (same-model vehicles) | ~1-1.5pp | Better fine-grained features |

## 5. Association Algorithm Is NOT the Bottleneck

- 220+ config sweeps prove diminishing returns
- CC vs conflict_free_cc vs Louvain vs agglomerative < 0.3pp difference
- Graph structure is well-conditioned at threshold=0.53
- Problem is which edges exist (feature similarity), not how they're clustered

## 6. Next Steps

Pending proper literature review of AIC21-24 competition winners and cross-camera ReID papers before committing to implementation strategy.
