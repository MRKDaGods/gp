# 12b Cross-Camera Identity Merging — Design Spec

**Date**: 2026-03-29
**Status**: Ready for implementation
**Target**: 12b notebook next version (v9+)

## 1. Current State Analysis

### What 12b Does Today

The 12b pipeline has **three stages**:

1. **MVDeTr ground-plane detection** (from 12a): Multi-view fused detections in world coordinates (x_cm, y_cm) per frame
2. **Hungarian tracking on ground plane**: Temporal association using L2 distance in world coordinates → produces `GroundPlaneTrack` objects (42 tracks from 40 test frames)
3. **ReID feature extraction**: TransReID ViT-Base/16 (Market1501 pretrained) extracts 768-dim L2-normalized embeddings per trajectory, averaged across all camera crops

Each ground-plane track becomes exactly one `GlobalTrajectory` with tracklets projected into all 7 cameras. The `global_id` equals the `track_id`.

### ReID Features Are Computed But Never Used

- **`reid_features.npz`**: 42 trajectories × 768-dim embeddings (L2-normalized)
- **`reid_merge_candidates.json`**: 382 pairs with cosine similarity ≥ 0.75
- **The merge candidates are saved but NEVER applied to modify trajectories**
- Evaluation (`evaluate_wildtrack_ground_plane()`) receives the **original** unmerged trajectories

### Feature Quality (12b v8)

| Metric | Value |
|--------|-------|
| Mean off-diagonal cosine similarity | 0.720 |
| Similarity range | [0.215, 1.000] |
| Candidates above 0.75 | 382 / 861 pairs (44%) |
| Total trajectory pairs | 42 × 41 / 2 = 861 |

The 0.720 mean and 44% above-threshold rate are **high** — this means many different people also have high similarity. The Market1501-pretrained model hasn't been fine-tuned on WILDTRACK and the outdoor clothing patterns are less discriminative than vehicle appearance. A conservative threshold is essential.

### Evaluation Pipeline

`evaluate_wildtrack_ground_plane()` in `src/stage5_evaluation/ground_plane_eval.py`:
- Builds ground-plane predictions from `GlobalTrajectory` objects using `traj.global_id` as the identity
- Back-projects foot positions from all available cameras, averages them per frame
- Applies NMS (DBSCAN, 50cm radius)
- Matches to GT using L2 distance (50cm threshold) via `motmetrics.MOTAccumulator`
- Computes MODA, IDF1, precision, recall, ID switches

**Key**: IDF1 is directly affected by identity correctness. If the tracker fragments person A into tracks 5 and 12, motmetrics sees different IDs for the same GT person → ID switches → lower IDF1.

### Current Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| MODA | 89.8% | Detection quality metric |
| IDF1 | 92.2% | Identity consistency metric |
| Precision | 97.1% | Few false positives |
| Recall | 93.9% | Some missed detections |
| ID Switches | 12 | Fragmentation instances |
| Tracks | 42 | For ~20-25 distinct people |

With only **12 ID switches**, fragmentation is already low. The 42 tracks likely represent ~25-30 unique people plus ~12-17 fragments.

## 2. Key Question: Will Merging Help?

### Yes, But With Caveats

**Why it can help:**
- 12 ID switches = 12 fragmentation events. If ReID correctly merges even some of these, IDF1 improves.
- 42 tracks for ~20-25 people means ~17-22 tracks are fragments. Merging them reduces over-counting.
- IDF1 directly rewards correct identity assignment — merging two fragments of the same person into one ID is a pure win.

**Why the impact may be limited:**
- IDF1 is already 92.2%. The theoretical headroom from fixing 12 ID switches is small (~1-3pp).
- With 0.720 mean cosine similarity, false merges are a real risk. Merging two different people is **worse** than leaving fragments unmerged.
- MODA won't change (it only cares about detection coverage, not identity).

**Expected impact**: IDF1 improvement of **0.5–2.0pp** (92.2% → 92.7–94.2%) if merging is conservative enough to avoid false merges. A bad threshold could **decrease** IDF1.

## 3. Proposed Design

### Approach: Graph-Based Merging with Temporal Non-Overlap Constraint

Reuse the vehicle pipeline's `GraphSolver` pattern but adapted for the WILDTRACK ground-plane context.

#### Step 1: Hard Constraint — Temporal Non-Overlap

Two ground-plane tracks that **overlap in time** cannot be the same person (each person occupies exactly one position per frame on the ground plane). This is an absolute constraint that eliminates many false merge candidates.

```
For each candidate pair (A, B):
  frames_A = set of frame_ids where track A has detections
  frames_B = set of frame_ids where track B has detections
  if frames_A ∩ frames_B is non-empty → REJECT (different people)
```

This should dramatically reduce the 382 candidates. Most high-similarity pairs are likely **concurrent** tracks of different people who look similar.

#### Step 2: Similarity Graph Construction

From the remaining candidates (temporal non-overlap confirmed):
- Build a graph where nodes = trajectory global_ids
- Add edges for pairs with cosine similarity ≥ `merge_threshold`
- Weight edges by cosine similarity

#### Step 3: Connected Components Clustering

Use simple connected components (not Louvain — with temporal constraints already applied, aggressive community detection isn't needed and risks over-merging).

#### Step 4: Trajectory Merging

For each connected component with >1 node:
- Assign all member trajectories the same global_id (lowest in the component)
- Combine tracklet lists from all member trajectories
- Recompute confidence as mean pairwise cosine similarity

#### Step 5: Ground-Plane Position Averaging

When the merged trajectory has detections at the same frame from different original tracks (shouldn't happen due to temporal non-overlap constraint, but defensive), average the ground-plane positions.

### Threshold Selection

Given feature quality:
- Mean off-diagonal sim: 0.720 → many different-person pairs score high
- After temporal filtering, most high-sim concurrent pairs are eliminated
- Recommended sweep: **[0.75, 0.80, 0.85, 0.90]**
- Starting point: **0.85** (conservative — only merge very confident matches)
- The temporal constraint does the heavy lifting; the threshold is a secondary filter

### Configuration

Add to the 12b notebook as constants (not in YAML since this is notebook-internal logic):

```python
# Cross-camera identity merging
REID_MERGE_ENABLED = True
REID_MERGE_THRESHOLD = 0.85      # cosine similarity threshold
REID_MERGE_MIN_CANDIDATES = 1    # minimum edges per component to merge
REID_MERGE_MAX_COMPONENT = 5     # safety cap on cluster size
```

## 4. Implementation Plan

### Phase 1: Add Merging Logic to 12b Notebook (v9)

**New cell** between the current merge-candidates cell and the evaluation cell:

```
Cell: "Apply ReID-based identity merging"
1. Load merge_candidates from the previous cell
2. For each candidate pair, check temporal non-overlap:
   - Build frame_id sets per trajectory from ground_plane_tracks
   - Reject pairs with overlapping frames
3. Build similarity graph (networkx) from surviving pairs above threshold
4. Find connected components
5. Cap components at max size (drop weakest edges if exceeded)
6. Create merged trajectories:
   - New global_id = min(component member IDs)
   - Combine tracklet lists
   - Update trajectory objects in-place (or create new list)
7. Report: N merges applied, original tracks → merged tracks
```

**Modified evaluation cell**: Pass the merged trajectories to `evaluate_wildtrack_ground_plane()` instead of the original ones.

**New output artifact**: `reid_merges_applied.json` documenting which tracks were merged and why.

### Phase 2: Sweep Merge Threshold

Add to the existing hyperparameter sweep:
- Sweep `REID_MERGE_THRESHOLD` over [0.75, 0.80, 0.85, 0.90, 0.95]
- For each threshold, apply merging and evaluate
- Report IDF1 and ID switches for each threshold
- Also report a "no merge" baseline for comparison

### Phase 3: Evaluate Impact

Compare:
1. **Baseline** (no merging): current MODA=89.8%, IDF1=92.2%
2. **With merging** at each threshold: track MODA, IDF1, precision, recall, ID switches
3. **Diagnostic**: For each merge, check against GT whether both tracks belong to the same person

### Files to Modify

| File | Change |
|------|--------|
| `notebooks/kaggle/12b_wildtrack_tracking_reid/12b_wildtrack_tracking_reid.ipynb` | Add merging cell, modify eval to use merged trajectories, add threshold sweep |

No changes needed to:
- `src/stage4_association/` (the vehicle pipeline is separate)
- `src/stage5_evaluation/ground_plane_eval.py` (already uses `traj.global_id`)
- `configs/datasets/wildtrack.yaml` (merging is notebook-internal)

## 5. Risks

| Risk | Mitigation |
|------|------------|
| False merges decrease IDF1 | Conservative threshold (0.85), temporal constraint, max component size cap |
| Too few merges to matter | Expected given only 12 ID switches — this is incremental improvement, not a breakthrough |
| Temporal constraint filters ALL candidates | If all 382 candidates involve concurrent tracks, no merges possible — would indicate features are finding same-frame similar-looking people, not fragments |
| Person ReID not discriminative enough | Market1501 model without WILDTRACK fine-tuning; mean sim 0.72 is high. May need fine-tuning for meaningful merging |

## 6. Success Criteria

- **Minimum**: IDF1 ≥ 92.2% (no regression from merging)
- **Target**: IDF1 ≥ 93.0% (+0.8pp from fixing some fragmentations)
- **Stretch**: IDF1 ≥ 94.0% (+1.8pp from fixing most fragmentations)
- **Diagnostic**: Report how many of the 12 ID switches were corrected

## 7. Open Questions

1. **How many of the 382 candidates survive temporal filtering?** This is the critical unknown. If most high-sim pairs are concurrent (different people who look similar), merging won't help much. Running the temporal filter first will reveal the answer.

2. **Should we use ground-plane distance as an additional constraint?** Two fragments of the same person should have their last/first detections within plausible walking distance. E.g., if track A ends at (300, 400) and track B starts at (800, 1200) 2 frames later, the person would need to teleport — reject. This could be a secondary constraint but temporal non-overlap is the primary one.

3. **Is the feature quality sufficient for merging?** With mean off-diagonal cosine sim of 0.720, different people are quite similar. The temporal constraint helps but the **remaining** post-temporal candidates need enough similarity spread to distinguish true matches from false ones. If post-temporal candidates still have mean sim ~0.85, it's workable; if all post-temporal candidates are >0.90, we can't discriminate.

4. **Would fine-tuning ReID on WILDTRACK help more?** Almost certainly yes — but that's a separate effort (requires labeled person ReID data from WILDTRACK, which doesn't exist as a standard ReID benchmark). Market1501 → WILDTRACK domain gap is significant.