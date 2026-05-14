# Phase 5A: Zone-Based Spatio-Temporal Constraints

> **Status**: Spec complete — recommends Simplified Option (Option 3)
> **Estimated effort**: 1-2 hours (simplified), 1-2 days (full)
> **Expected gain**: +0.0 to +0.3pp (simplified), +0.5-1.0pp (full, IF manually annotated)
> **Risk**: Low (simplified), Medium-High (full)

---

## Critical Assessment: What Already Exists

Before designing anything new, we must recognize that our **temporal constraint system is already comprehensive**:

### Existing Infrastructure (pipeline.py + similarity.py + spatial_temporal.py)

| Feature | Location | Status |
|---------|----------|--------|
| Per-camera-pair transition windows (min/max/mean/std) | `cityflowv2.yaml` → `SpatioTemporalValidator` | ✅ GT-calibrated |
| Hard blocking of impossible pairs (S01↔S02) | `SpatioTemporalValidator.is_valid_transition()` | ✅ Via missing transitions |
| Hard same-camera temporal overlap filter | pipeline.py Step 2a | ✅ Removes provably-impossible pairs |
| Gaussian temporal scoring (per-pair priors) | `SpatioTemporalValidator.transition_score()` | ✅ Mean+std from GT |
| Temporal overlap bonus (overlapping-FOV pairs) | `compute_combined_similarity()` | ✅ Configurable |
| `st_score <= 0 → continue` hard gate | `compute_combined_similarity()` line ~206 | ✅ Blocks invalid transitions |
| ZoneScorer code skeleton | `zone_scoring.py` | ✅ Code exists, needs zone data |
| Global max_time_gap = 300s | `cityflowv2.yaml` | ✅ 5-minute cutoff |

### Key Observation

The `compute_combined_similarity()` function already **hard-skips** pairs where `transition_score() <= 0`. Since `transition_score()` returns 0.0 whenever `is_valid_transition()` returns False, impossible camera pairs (S01↔S02) are already eliminated — they never enter the similarity graph.

### What Failed: Auto-Generated Zones (v54-57)

Auto-generated zone polygons resulted in **-0.4pp MTMC IDF1** (documented in findings.md). The approach:
- K-means clustered first/last bbox centers into entry/exit zone IDs
- Learned transition counts between zone pairs from initial clustering
- Applied bonus/penalty based on transition validity

**Why it failed**: K-means discretization is noisy with only ~100 tracklets per camera per scene. Cluster boundaries are arbitrary, and the transition counts are too sparse to be reliable (many zone-pair combinations have 0 observations).

---

## Three Options

### Option 1: Full Manual Zone Annotation System

**What**: Define 2-4 entry/exit zones per camera as pixel bounding boxes. Build a travel time matrix per (cam_i_zone, cam_j_zone) pair. Apply as hard gate + soft scoring.

**Pros**: This is what AIC22 winners did. Zone-level temporal windows are tighter than camera-level ones (e.g., "exit left of c001 → enter right of c002" takes 3-8s, vs camera-level 0-20s).

**Cons**:
- Requires video inspection for **all 8 cameras** to define zones — hours of manual annotation
- Zone boundaries are subjective and need iterative refinement
- Travel time matrices need manual calibration or GT-based learning (small sample sizes)
- The ZoneScorer infrastructure already exists but hurt with auto-generated data
- Our camera count is small (8 cameras, ~15 valid camera pairs) — zone-level precision adds marginal value over camera-level precision

**Effort**: 1-2 days (annotation + calibration + testing)
**Expected gain**: +0.5-1.0pp IF zones are well-annotated. Risk of -0.4pp if not.
**Verdict**: Not recommendable for 1-2 hour timebox. Revisit after ensemble (Phase 3) if needed.

### Option 2: Auto-Generated Zones (Already Tested)

**Result**: -0.4pp (v54-57). **Do not retry.**

### Option 3: Temporal Feasibility Tightening (RECOMMENDED)

**What**: Strengthen the existing temporal system with three targeted improvements that require no manual annotation and no new zone infrastructure:

1. **Asymmetric time gap enforcement** — Currently `abs(time_diff)` makes all transitions direction-agnostic. For S02 sequential cameras, we know vehicles flow in one predominant direction. Add an optional directional penalty.

2. **Per-camera-pair hard max_time pre-filter in pair generation** — Currently, impossible temporal matches are caught in `compute_combined_similarity()` via `transition_score() → continue`. But the exhaustive pair builder (`_build_all_cross_camera_pairs`) still computes cosine similarity for ALL cross-camera pairs before this filter runs. Adding temporal pre-filtering in the pair builder avoids wasted computation AND prevents any edge case where high appearance similarity could bypass soft temporal scoring.

3. **Entry/exit edge classification** (lightweight zone proxy) — Instead of discretizing positions into K-means zones, classify each tracklet's entry/exit to a **frame edge** (top/bottom/left/right) based on which edge the first/last bbox center is closest to. This is deterministic, requires no training, and captures the major directional signal. Score compatibility: if tracklet exits right edge of cam_i and enters left edge of cam_j, that's the expected flow. If both exit the same edge, that's suspicious.

**Effort**: 1-2 hours
**Expected gain**: +0.0 to +0.3pp (conservative — existing temporal system already captures most signal)
**Risk**: Very low — purely additive soft scoring, disabled by default

---

## Recommended Implementation: Option 3

### 3.1 Per-Camera-Pair Temporal Pre-Filter

**Location**: `pipeline.py`, inside `_build_all_cross_camera_pairs()` (and `_build_all_cross_camera_pairs_multi_query()`)

**Current state**: These functions compute `cosine_sim = embeddings[i] @ embeddings[j]` for ALL cross-camera pairs, then filter by `min_similarity`. Temporal filtering happens later in `compute_combined_similarity()`.

**Change**: Before computing cosine similarity, check if the pair is temporally feasible using the same `camera_transitions` config. Skip impossible pairs early.

```python
def _build_all_cross_camera_pairs(
    n, embeddings, camera_ids, class_ids,
    min_similarity, start_times, end_times, st_validator,  # NEW params
):
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if camera_ids[i] == camera_ids[j]:
                continue
            if class_ids[i] != class_ids[j]:
                continue

            # NEW: temporal pre-filter
            if st_validator is not None:
                if not st_validator.is_valid_transition(
                    camera_ids[i], camera_ids[j],
                    start_times[i], start_times[j],
                ):
                    continue

            sim = float(embeddings[i] @ embeddings[j])
            if sim >= min_similarity:
                pairs.append((i, j, sim))
    return pairs
```

**Impact**: Reduces pair count (and computation) by eliminating S01↔S02 pairs and pairs exceeding per-camera-pair max_time_gap before computing cosine similarity. This is an optimization that also adds safety — no appearance similarity, however high, can resurrect a temporally impossible pair.

**Config**: No new config needed. Uses existing `camera_transitions`.

### 3.2 Entry/Exit Edge Classification

**Location**: New function in `similarity.py` or small addition to `zone_scoring.py`

**Concept**: For each tracklet, determine which edge of the frame it enters/exits from:
- Compute entry position = center of first bbox
- Compute exit position = center of last bbox
- Classify to nearest frame edge: 0=top, 1=right, 2=bottom, 3=left

```python
def classify_frame_edge(
    cx: float, cy: float,
    frame_width: float = 1920.0,
    frame_height: float = 1080.0,
) -> int:
    """Classify a position to the nearest frame edge.

    Returns: 0=top, 1=right, 2=bottom, 3=left
    """
    distances = [
        cy,                    # top
        frame_width - cx,      # right (negative = far from right)
        frame_height - cy,     # bottom
        cx,                    # left
    ]
    return int(np.argmin(distances))
```

**Edge compatibility scoring**: Define a per-camera-pair expected edge mapping. For example, if vehicles exiting the right edge of c006 typically enter the left edge of c007, then:

```yaml
stage4:
  association:
    edge_constraints:
      enabled: false  # off by default
      bonus: 0.02
      penalty: 0.02
      frame_size: [1920, 1080]
      # Per camera-pair expected edge transitions
      # Format: {exit_edge_cam_i: [valid_entry_edges_cam_j]}
      # 0=top, 1=right, 2=bottom, 3=left
      transitions:
        S01_c001-S01_c002:
          exit_edges: {1: [3], 2: [0]}  # right→left, bottom→top
        S01_c002-S01_c003:
          exit_edges: {1: [3]}
        S02_c006-S02_c007:
          exit_edges: {1: [3]}  # highway flow: right→left
        S02_c007-S02_c008:
          exit_edges: {1: [3]}
```

**Problem**: We'd need to determine these edge mappings. Without video inspection, we don't know which edges correspond to which traffic flow. We COULD learn them from GT matches, but the sample sizes per camera pair are small (46-100 matches).

**Practical approach**: Start with edge classification as a **logged diagnostic** only. Log the entry/exit edge distribution per camera and per camera pair. Use this to determine if edge mappings are learnable, THEN enable scoring.

### 3.3 Diagnostic-First Implementation Plan

Rather than guessing at edge mappings and potentially hurting performance (like auto-zones did), take a diagnostic approach:

**Step 1** (30 min): Add temporal pre-filter to pair builders (Section 3.1)
- Pure optimization + safety improvement
- Zero risk of quality regression

**Step 2** (30 min): Add entry/exit edge classification as diagnostic logging
- Compute entry/exit edges for all tracklets
- Log distribution per camera: "c001: entries from top=12%, right=45%, bottom=8%, left=35%"
- Log distribution per GT-matched camera pair: "c001→c002 GT matches: exit_right→entry_left=78%"
- This data will tell us if edge constraints are viable

**Step 3** (30 min): If diagnostics show clear edge patterns, add soft scoring
- Only if Step 2 reveals >70% concentration in specific edge combinations
- Use bonus/penalty scheme similar to existing ZoneScorer

**Step 4** (optional, 30 min): Per-camera-pair directional bias
- If S02 traffic is predominantly one-direction, add asymmetric scoring
- For cameras where time_a < time_b implies "forward" direction, give a small bonus
- Configurable and disabled by default

---

## Config Schema

```yaml
stage4:
  association:
    # Existing — no changes
    spatiotemporal:
      max_time_gap: 300
      min_time_gap: 0
      camera_transitions: { ... }  # existing per-pair priors

    # NEW: Temporal pre-filter in pair generation
    temporal_prefilter:
      enabled: true  # safe to enable — uses existing camera_transitions
      # When true, _build_all_cross_camera_pairs checks is_valid_transition
      # before computing cosine similarity. Reduces pair count and prevents
      # any high-sim pair from surviving temporal impossibility.

    # NEW: Entry/exit edge diagnostics and scoring
    edge_constraints:
      enabled: false  # off by default — enable after diagnostic analysis
      diagnostic_only: true  # when true, log edge distributions but don't modify scores
      bonus: 0.02
      penalty: 0.02
      frame_size: [1920, 1080]
      transitions: {}  # populated after diagnostic analysis
```

---

## Files to Modify

| File | Change | Complexity |
|------|--------|-----------|
| `src/stage4_association/pipeline.py` | Pass st_validator + times to pair builders; add edge diagnostic logging | Low |
| `src/stage4_association/similarity.py` | Add `classify_frame_edge()` utility | Trivial |
| `configs/datasets/cityflowv2.yaml` | Add `temporal_prefilter` and `edge_constraints` sections | Trivial |
| `src/stage4_association/pipeline.py` `_build_all_cross_camera_pairs()` | Add temporal feasibility check before cosine computation | Low |

---

## Why NOT Full Zones Right Now

1. **Auto-generated zones already failed** (-0.4pp, v54-57)
2. **Current temporal system is already GT-calibrated** with per-pair mean/std
3. **8 cameras × 2-4 zones × manual annotation** = hours of video inspection we can't do without the videos
4. **Zone-pair travel times** require even more manual calibration
5. **Marginal gain** over existing system is small (existing priors already capture camera-pair transitions)
6. **Better ROI for that time**: Ensemble (Phase 3) is worth +4-6pp vs zones at +0.5pp

## When to Revisit Full Zones

- After Phase 3 (ensemble) is deployed and the baseline is at IDF1 ≥ 0.82
- If diagnostic logging (Step 2) reveals strong edge patterns that could tighten transitions
- If error analysis shows remaining conflation errors are temporally distinguishable at the sub-camera level

---

## Success Criteria

| Change | Metric | Pass |
|--------|--------|------|
| Temporal pre-filter | Same IDF1, fewer computed pairs | Pair count reduced by ≥10% |
| Edge classification diagnostics | Edge distributions logged | Distributions show >60% concentration |
| Edge scoring (if enabled) | IDF1 change | ≥ 0.0pp (must not regress) |

---

## Dependencies

- **Requires**: Stage 1 tracklet data with bbox positions (already available via `Tracklet.frames[0].bbox`)
- **Requires**: Stage 3 metadata with start_time/end_time (already in `MetadataStore`)
- **Requires**: `camera_transitions` config (already in `cityflowv2.yaml`)
- **No new dependencies**: Uses existing numpy, omegaconf

## Relationship to Other Phases

- **Phase 4 (CID_BIAS from timestamps)**: Complementary. CID_BIAS adjusts similarity scores per camera pair; temporal pre-filter removes impossible pairs. Both can coexist.
- **Phase 5B (Two-stage clustering)**: Independent. Zone constraints modify the similarity matrix; two-stage clustering modifies the graph solver. Can be developed in parallel.
- **Phase 3 (Ensemble)**: Temporal pre-filter is feature-agnostic and will work with any embedding. No conflict.
