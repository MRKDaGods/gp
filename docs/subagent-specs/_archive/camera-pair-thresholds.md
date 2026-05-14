# Camera-Pair Specific Similarity Thresholds

**Status:** Analysis complete — implementation already exists but never tested  
**Priority:** LOW — honest assessment: marginal gains expected (+0.0 to +0.3pp)  
**Risk:** LOW — code already in pipeline.py, just needs config  

---

## 1. Executive Summary

Per-camera-pair similarity thresholds are **already implemented** in `src/stage4_association/pipeline.py` (Step 5e, lines ~740-760) but have **never been activated** in any of the 225+ experiments. This spec analyzes whether activating them is worth the experiment time.

**Honest verdict:** Probably not worth prioritizing. FIC whitening already removes the camera-specific distribution bias that would make per-pair thresholds necessary. The remaining gap is feature-quality-limited, not threshold-limited. However, since the code already exists, a single test costs almost nothing.

---

## 2. How Thresholds Are Currently Applied

### Current Flow (Single Global Threshold)

```
compute_combined_similarity()          → Dict[(i,j), score]
    ↓
camera_pair_norm (disabled)            → optional distribution centering
camera_bias / CID_BIAS (disabled)      → optional additive bias
camera_pair_boost (disabled)           → optional targeted boosts  
pair_thresholds (disabled)             → per-pair edge filtering ← THIS FEATURE
    ↓
GraphSolver.solve()                    → if sim >= self.similarity_threshold: add_edge
    ↓
conflict_free_cc                       → clusters
```

The global threshold is applied in `graph_solver.py` line ~70:
```python
if sim >= self.similarity_threshold:
    G.add_edge(i, j, weight=sim)
```

Current value: `0.55` (cityflowv2.yaml) or `0.60` (default.yaml).

### Existing Per-Pair Threshold Code (Step 5e)

Already in `pipeline.py` lines ~740-760:
```python
pair_thresholds_cfg = stage_cfg.get("pair_thresholds", {})
if pair_thresholds_cfg.get("enabled", False):
    pair_thresh_map = pair_thresholds_cfg.get("thresholds", {})
    if pair_thresh_map:
        # Pre-filter edges below pair-specific threshold
        for (i, j), sim in solve_sim.items():
            cam_pair = tuple(sorted([camera_ids[i], camera_ids[j]]))
            pair_key = f"{cam_pair[0]}-{cam_pair[1]}"
            pair_thresh = pair_thresh_map.get(pair_key, graph_threshold)
            if sim < pair_thresh:
                del solve_sim[key]
        # Set graph_threshold to min(pair thresholds) to avoid double-filtering
        graph_threshold = min(graph_threshold, min(pair_thresh_map.values()))
```

This is **exactly** what we need. No new code required.

---

## 3. How This Differs from CID_BIAS (Dead End)

| Aspect | CID_BIAS (Dead End) | Per-Pair Thresholds |
|--------|:---:|:---:|
| **Mechanism** | Additive bias to similarity scores | Different acceptance cutoff per pair |
| **Effect on scores** | Changes edge weights | Does NOT change edge weights |
| **Effect on graph** | Shifts entire distribution → distorts FIC calibration | Only changes which edges enter the graph |
| **Why CID_BIAS failed** | Warps FIC-calibrated similarity geometry | Does not warp anything — binary accept/reject |
| **Interaction with FIC** | Destructive (undoes whitening) | Orthogonal (operates after all score computation) |

**Key insight:** CID_BIAS failed because it modifies the similarity values that FIC carefully calibrated. Per-pair thresholds operate AFTER all score computation — they only decide the accept/reject boundary per pair, leaving the relative ordering intact.

---

## 4. CityFlowV2 Camera Pair Analysis

### 6 Active Camera Pairs (cross-scene blocked)

| Pair | Mean Transit Time | Viewpoint Relationship | Expected Match Difficulty |
|------|:-:|---|---|
| S01_c001 ↔ S01_c003 | 0.5s | Overlapping FOV, near-identical angle | EASY — high similarity for true matches |
| S01_c001 ↔ S01_c002 | 1.9s | Adjacent, moderate angle change | MODERATE |
| S01_c002 ↔ S01_c003 | 1.4s | Adjacent, moderate angle change | MODERATE |
| S02_c006 ↔ S02_c007 | 0.9s | Overlapping FOV | EASY |
| S02_c007 ↔ S02_c008 | 5.8s | Sequential, moderate distance | MODERATE-HARD |
| S02_c006 ↔ S02_c008 | 8.1s | Farthest apart, largest viewpoint change | HARD |

**Hypothesis:** S02_c006↔S02_c008 (mean 8.1s transit, hardest pair) might benefit from a LOWER threshold (more permissive) because true matches have lower similarity due to larger viewpoint change. Easy pairs (S01_c001↔S01_c003) might benefit from a HIGHER threshold to suppress false positives between visually similar vehicles seen from the same angle.

---

## 5. Why FIC Already Mostly Handles This

FIC (per-camera whitening) transforms each camera's embedding distribution to be zero-mean with identity-like covariance. This means:

1. **Per-camera bias removed:** Each camera's embeddings are centered and normalized
2. **Cross-camera comparability:** Cosine similarity between cameras A and B is now calibrated relative to each camera's own distribution
3. **Viewpoint normalization:** The covariance inversion rotates the feature space to compensate for camera-specific viewpoint effects

**What FIC does NOT do:**
- It doesn't guarantee identical difficulty across all pairs
- With Tikhonov regularization (reg=0.1), it's a compromise — some residual per-pair bias may remain
- It operates on EMBEDDINGS, not on the combined similarity score (which includes HSV, spatiotemporal, temporal overlap)

**Evidence from findings.md:**
> "FIC whitening already handles the useful camera calibration, and extra CID_BIAS offsets only warp the calibrated similarity geometry."

This was about CID_BIAS (additive), but the implication is clear: FIC already does most of the camera calibration work.

---

## 6. Three Approaches (If We Decide to Test)

### 6A. GT-Calibrated Thresholds (Needs GT)

Learn optimal threshold per camera pair from ground truth:
1. For each camera pair, compute similarity distribution of TRUE matches vs FALSE matches
2. Find the threshold that maximizes F1 (or IDF1 proxy) per pair
3. Deploy learned thresholds

**Pros:** Optimal by definition  
**Cons:** Overfits to GT; same concern as GT-learned CID_BIAS (-3.3pp); only 6 pairs = very few data points  
**Risk:** HIGH — GT-learned CID_BIAS already catastrophically failed

### 6B. Distribution-Based Thresholds (No GT)

For each camera pair, compute the similarity distribution of all candidate edges, then set threshold at a fixed percentile (e.g., 75th percentile):
```python
for pair in camera_pairs:
    sims = [s for (i,j), s in combined_sim.items() 
            if sorted([cam[i], cam[j]]) == pair]
    pair_threshold = np.percentile(sims, 75)
```

**Pros:** No GT needed; adapts to each pair's distribution  
**Cons:** Percentile doesn't distinguish true from false matches; if a pair has 90% false matches, the 75th percentile is still in the false region  
**Risk:** MEDIUM — could hurt if the percentile is wrong

### 6C. Hand-Tuned ± Delta from Global (Simplest)

Start from the global threshold (0.55) and adjust ±0.05 for specific pairs based on transit time:
```yaml
pair_thresholds:
  enabled: true
  thresholds:
    S01_c001-S01_c003: 0.58  # easy pair, raise threshold (+0.03)
    S01_c001-S01_c002: 0.55  # moderate, keep global
    S01_c002-S01_c003: 0.55  # moderate, keep global
    S02_c006-S02_c007: 0.55  # easy pair, keep global
    S02_c006-S02_c008: 0.50  # hardest pair, lower threshold (-0.05)
    S02_c007-S02_c008: 0.53  # moderate-hard, slightly lower (-0.02)
```

**Pros:** Simplest; uses domain knowledge; easy to tune; uses existing code  
**Cons:** Manual; only 6 pairs to tune = might overfit  
**Risk:** LOW — easy to revert; at most 0.05 delta from proven global threshold

---

## 7. Recommended Test (If Proceeding)

### Single Experiment: 6C with Conservative Deltas

**Config override for 10c:**
```
stage4.association.pair_thresholds.enabled=true
stage4.association.pair_thresholds.thresholds.S01_c001-S01_c003=0.58
stage4.association.pair_thresholds.thresholds.S01_c001-S01_c002=0.55
stage4.association.pair_thresholds.thresholds.S01_c002-S01_c003=0.55
stage4.association.pair_thresholds.thresholds.S02_c006-S02_c007=0.55
stage4.association.pair_thresholds.thresholds.S02_c006-S02_c008=0.50
stage4.association.pair_thresholds.thresholds.S02_c007-S02_c008=0.53
```

**Expected outcome:** +0.0 to +0.3pp MTMC IDF1  
**Success criterion:** MTMC IDF1 > 0.778 (vs 0.775 baseline)  
**Measurement:** Compare fragmented/conflated ID counts vs 10c v52 baseline

### What to Watch For
- If S02_c006-S02_c008 threshold=0.50 recovers fragmented IDs → the hard pair was under-served
- If easy pair threshold=0.58 reduces conflated IDs → false positives were leaking through
- If neutral → FIC already handles this, close the investigation

---

## 8. Honest Assessment

### Why This Probably Won't Help Much

1. **Association is exhausted.** 225+ configs, all within 0.3pp. The threshold landscape is flat — there's no hidden optimum per pair that differs dramatically from the global one.

2. **FIC already calibrates.** Per-camera whitening removes the distribution shift that would make per-pair thresholds necessary. After FIC, all pairs should have roughly comparable similarity distributions.

3. **Only 6 pairs.** With only 6 active camera pairs, there's almost no degree of freedom. The global threshold is already a good compromise for 6 pairs — you can't improve much by splitting 1 parameter into 6.

4. **Error profile is feature-limited.** 87 fragmented / 35 conflated. The fragmented IDs are caused by embeddings that simply aren't similar enough (different viewpoints → different features). A lower threshold lets them in but also lets in false matches → conflation rises → net zero.

5. **CID_BIAS precedent.** While per-pair thresholds are mechanically different from CID_BIAS, the underlying assumption is the same: "different camera pairs need different treatment." FIC+global threshold already achieves this implicitly. Every explicit per-pair adjustment has hurt.

### Why It Might Still Be Worth One Test

1. **Code already exists.** Zero implementation effort — just a config change.
2. **Never tested.** The 225+ experiments never activated `pair_thresholds`. It's a genuine blind spot.
3. **S02_c006↔S02_c008 is an outlier.** 8.1s mean transit, farthest apart — might be the one pair where global threshold is suboptimal.
4. **Low risk.** Conservative deltas (±0.03–0.05) can't cause catastrophic regression.

### Bottom Line

**Worth a single test? Yes** — it costs nothing (code exists, config-only change).  
**Worth investing significant time? No** — the expected gain ceiling is +0.3pp, and the gap to SOTA (7.36pp) is feature-quality-limited.  
**Should this block other work? Absolutely not** — run it as a side experiment while working on feature quality improvements.

---

## 9. Implementation Checklist

- [ ] No code changes needed — `pair_thresholds` is already in pipeline.py Step 5e
- [ ] Add config to 10c notebook as CLI override
- [ ] Run single test with 6C config (conservative ±0.05 deltas)
- [ ] Compare MTMC IDF1, fragmented IDs, conflated IDs vs 10c v52 baseline
- [ ] If neutral: close investigation, document in findings.md
- [ ] If positive: run 2-3 more tests varying delta magnitude (±0.03, ±0.08)
- [ ] Update findings.md with result