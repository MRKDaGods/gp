# Vehicle Pipeline Code Drift Investigation — Spec

**Date**: 2026-03-30
**Status**: Ready for investigation
**Problem**: MTMC IDF1 dropped from 0.784 (v80) to 0.772 (current) — 1.2pp code drift

## 1. Problem Statement

The vehicle pipeline historically achieved MTMC IDF1 = 0.784 in the v80-era run on the ali369 account. The same repository now reproduces only about 0.772 MTMC IDF1 in current 10c v50/v51 runs. That is a regression of about 1.2 percentage points.

Per [docs/findings.md](../findings.md), the current reproducible best is 77.2% while the historical best remains 78.4% but is not reproducible on the current codebase. This spec assumes the missing 1.2pp is primarily code drift or config drift between the v80 commit and current HEAD, rather than a new fundamental limit in the model.

The objective of this investigation is to identify the minimal set of code and config differences that explains the regression and restore reproducibility of the 0.784 MTMC IDF1 result.

## 2. Evidence

Current evidence already narrows the search space:

- Historical best: MTMC IDF1 0.784 from v80-era code and config.
- Current reproducible best: MTMC IDF1 0.772 from 10c v50/v51.
- Association sweeps on the current codebase are already exhausted, so the missing 1.2pp is unlikely to come from merely retuning stage 4 thresholds.
- [docs/findings.md](../findings.md) explicitly records that the v80 result is no longer reproducible and frames the issue as code drift.

Evidence already confirmed in config and code:

- [configs/datasets/cityflowv2.yaml](../../configs/datasets/cityflowv2.yaml) line 91 sets `stage1.tracker.min_hits: 3`, while the historical v80 reference used `min_hits=2`.
- [configs/datasets/cityflowv2.yaml](../../configs/datasets/cityflowv2.yaml) lines 195-201 set `stage4.association.graph.algorithm: "conflict_free_cc"` and `max_component_size: 12`.
- [src/stage4_association/graph_solver.py](../../src/stage4_association/graph_solver.py) lines 200-264 implement `conflict_free_cc` as a greedy conflict-avoiding graph builder, which is materially different from standard connected components.
- [src/stage2_features/pipeline.py](../../src/stage2_features/pipeline.py) lines 153-176 actively read and pass `color_augment` into the ReID model path.
- [configs/datasets/cityflowv2.yaml](../../configs/datasets/cityflowv2.yaml) lines 103-138 enable `color_augment: true`, `camera_bn.enabled: true`, and `laplacian_min_var: 30.0`.
- [src/stage5_evaluation/pipeline.py](../../src/stage5_evaluation/pipeline.py) lines 75-201 contain several score-sensitive submission and GT-assisted filters.
- [configs/datasets/cityflowv2.yaml](../../configs/datasets/cityflowv2.yaml) lines 264-307 currently set `mtmc_only_submission: true`, `stationary_filter.enabled: true`, `track_smoothing.enabled: true`, and GT filters disabled by default.

This is enough to justify a structured archaeology-plus-ablation investigation rather than more blind scan runs.

## 3. Suspect Analysis

### Tier 1 — Most Likely Causes (~0.5-1.0pp each)

#### 1. `min_hits=3` vs historical `min_hits=2`

- File: [configs/datasets/cityflowv2.yaml](../../configs/datasets/cityflowv2.yaml)
- Reference: line 91
- Hypothesis: if the v80 run explicitly used `min_hits=2` and the current reproduction path forgets to override the now-default `min_hits: 3`, stage 1 emits a different tracklet set and the entire downstream feature and association graph changes.
- Why it matters: this is already a confirmed historical difference and can plausibly explain a small but real portion of the drift on its own.

#### 2. `conflict_free_cc` vs `connected_components`

- Files:
  - [configs/datasets/cityflowv2.yaml](../../configs/datasets/cityflowv2.yaml)
  - [src/stage4_association/graph_solver.py](../../src/stage4_association/graph_solver.py)
- References:
  - config lines 195-201
  - solver lines 200-264
- Hypothesis: if v80 used `connected_components` and the live config later switched to `conflict_free_cc`, the cluster construction logic changed fundamentally. `conflict_free_cc` greedily blocks merges that would create same-camera temporal conflicts, while `connected_components` allows more transitive chains and relies on later cleanup.
- Why it matters: a graph algorithm change is large enough to account for a non-trivial share of the missing 1.2pp by itself.

#### 3. Stage 2 feature-space drift from augmentation and crop filtering

- Files:
  - [configs/datasets/cityflowv2.yaml](../../configs/datasets/cityflowv2.yaml)
  - [src/stage2_features/pipeline.py](../../src/stage2_features/pipeline.py)
- References:
  - config lines 103-138
  - pipeline lines 142-176 and 512-514
- Suspect settings:
  - `color_augment: true`
  - `laplacian_min_var: 30.0`
  - `camera_bn.enabled: true`
- Hypothesis: even when association code is held constant, the feature distribution may have drifted relative to the v80-era GPU export if color TTA, blur gating, or camera-aware normalization were introduced or changed later.
- Why it matters: stage 4 is downstream of stage 2. A small embedding-distribution shift can change candidate ordering, graph edges, and confidence scores everywhere.

#### 4. GT-assisted evaluation/filter implementation drift

- File: [src/stage5_evaluation/pipeline.py](../../src/stage5_evaluation/pipeline.py)
- References: lines 177-201
- Suspect settings:
  - `gt_zone_filter`
  - `gt_frame_clip`
- Hypothesis: if the historical 0.784 number was produced with GT-assisted filtering and the current implementation of those filters changed slightly, the amount of score inflation can differ even with otherwise similar trajectories.
- Why it matters: [docs/findings.md](../findings.md) states these filters can inflate scores by about 1-3pp, so even a partial implementation drift here is large enough to explain a substantial part of the gap.

### Tier 2 — Possible Contributors (~0.1-0.3pp each)

#### 5. Gallery expansion thresholds

- Files:
  - [configs/datasets/cityflowv2.yaml](../../configs/datasets/cityflowv2.yaml)
  - [src/stage4_association/pipeline.py](../../src/stage4_association/pipeline.py)
- References:
  - config lines 213-217
  - pipeline lines 803-835 and 1760-1850
- Hypothesis: `gallery_expansion.threshold` and `orphan_match_threshold` may have drifted since v80. The current config uses `0.50` and `0.40`, and the code path is active.

#### 6. Hard temporal constraint pre-filter

- File: [src/stage4_association/pipeline.py](../../src/stage4_association/pipeline.py)
- Reference: lines 340-385, especially the same-camera overlap removal block
- Hypothesis: the pre-filter now removes impossible same-camera overlapping pairs before mutual-NN filtering. If this was added after v80, it changes graph connectivity before clustering and may block links that older code admitted.

#### 7. Scene blocking in exhaustive cross-camera pair generation

- File: [src/stage4_association/pipeline.py](../../src/stage4_association/pipeline.py)
- Reference: pair generation and candidate filtering path before combined similarity, centered around lines 299-338 and the helper logic used by exhaustive pair generation
- Hypothesis: any stricter scene extraction or camera-pair blocking logic can reduce candidate recall. This is less likely than Tier 1, but still capable of contributing a few tenths.

#### 8. `cross_id_nms_iou` in submission writing

- Files:
  - [configs/datasets/cityflowv2.yaml](../../configs/datasets/cityflowv2.yaml)
  - [src/stage5_evaluation/format_converter.py](../../src/stage5_evaluation/format_converter.py)
- References:
  - config line 282
  - converter lines 113-118
- Hypothesis: if the threshold used during the v80 submission export differed from the current path, duplicate suppression can alter IDFP and IDTP counts.

#### 9. Same-camera conflict resolution order

- File: [src/stage4_association/pipeline.py](../../src/stage4_association/pipeline.py)
- Reference: lines 760-900 and downstream re-resolution passes after clustering, gallery expansion, and temporal split
- Hypothesis: `_resolve_same_camera_conflicts` may now be called at different points or behave differently than in v80, which can slightly alter final cluster composition.

### Tier 3 — Unlikely but Worth Checking

#### 10. `max_component_size`

- Files:
  - [configs/datasets/cityflowv2.yaml](../../configs/datasets/cityflowv2.yaml)
  - [src/stage4_association/graph_solver.py](../../src/stage4_association/graph_solver.py)
- References:
  - config line 201
  - solver lines 117-148
- Hypothesis: if the component-size cap was introduced or changed after v80, it could split clusters that older code kept intact.

#### 11. FAISS rebuild timing after FIC/FAC

- File: [src/stage4_association/pipeline.py](../../src/stage4_association/pipeline.py)
- Reference: lines 252-259
- Hypothesis: the current code explicitly rebuilds FAISS after FIC/FAC-transformed embeddings. If earlier code did not, later retrieval stages would operate on a different neighborhood structure.

#### 12. Currently disabled but historically toggled options

- Files:
  - [src/stage4_association/pipeline.py](../../src/stage4_association/pipeline.py)
  - [configs/datasets/cityflowv2.yaml](../../configs/datasets/cityflowv2.yaml)
  - [src/stage5_evaluation/pipeline.py](../../src/stage5_evaluation/pipeline.py)
- Suspects:
  - `cluster_verify`
  - `temporal_split`
  - `csls`
  - `pair_thresholds`
  - `track_edge_trim`
  - `track_smoothing`
- Hypothesis: most are currently disabled or obviously not the main cause, but v80 may have been run during a period when one of these was temporarily enabled or implemented differently.

## 4. Investigation Plan

The investigation should be staged. Do not begin with broad parameter sweeps. The goal is to isolate drift, not re-optimize the whole pipeline.

### Phase 1: Git Archaeology

Objective: identify the exact v80-era commit and define the drift window.

Steps:

1. Identify the exact commit hash associated with the historical v80 result, using experiment notes, git history, and any notebook or log artifacts that mention v80.
2. Record the exact config and override set used for the v80 run, especially `min_hits`, stage 4 thresholds, stage 5 filters, and any GT-assisted toggles.
3. Diff the v80 commit against current HEAD for:
   - [src/stage2_features/pipeline.py](../../src/stage2_features/pipeline.py)
   - [src/stage4_association/pipeline.py](../../src/stage4_association/pipeline.py)
   - [src/stage4_association/graph_solver.py](../../src/stage4_association/graph_solver.py)
   - [src/stage5_evaluation/pipeline.py](../../src/stage5_evaluation/pipeline.py)
   - [src/stage5_evaluation/format_converter.py](../../src/stage5_evaluation/format_converter.py)
   - [configs/datasets/cityflowv2.yaml](../../configs/datasets/cityflowv2.yaml)
4. Build a dated timeline of when the major suspect features landed, especially:
   - switch to `conflict_free_cc`
   - introduction of hard temporal pre-filter
   - stage 2 `color_augment`
   - stage 2 `laplacian_min_var`
   - stage 5 GT filter behavior changes

Deliverable:

- A short archaeology note listing the v80 commit, the suspect commit range, and the first commit where each Tier 1 suspect appears.

### Phase 2: Config Comparison

Objective: separate config drift from code drift.

Steps:

1. Reconstruct the exact v80 YAML and CLI override state.
2. Compare it against current [configs/datasets/cityflowv2.yaml](../../configs/datasets/cityflowv2.yaml) lines 80-307.
3. Produce a focused diff covering only behavior-changing keys, including:
   - `stage1.tracker.min_hits`
   - `stage2.reid.color_augment`
   - `stage2.crop.laplacian_min_var`
   - `stage2.camera_bn.enabled`
   - `stage4.association.secondary_embeddings.weight`
   - `stage4.association.query_expansion.k`
   - `stage4.association.weights.length_weight_power`
   - `stage4.association.graph.algorithm`
   - `stage4.association.graph.max_component_size`
   - `stage4.association.gallery_expansion.*`
   - `stage5.cross_id_nms_iou`
   - `stage5.track_smoothing.*`
   - `stage5.gt_zone_filter`
   - `stage5.gt_frame_clip`
4. Re-run the current codebase with a reconstructed v80 config before changing any code. This tests whether the drift is mostly in config.

Deliverable:

- A config diff table classifying each key as confirmed changed, unknown, or unchanged.

### Phase 3: Targeted Ablation

Objective: measure each Tier 1 suspect independently.

Run order, highest priority first:

1. Force `stage1.tracker.min_hits=2` and compare against the current default path.
2. Force `stage4.association.graph.algorithm=connected_components` and compare against `conflict_free_cc`.
3. Disable stage 2 drift candidates one at a time:
   - `color_augment=false`
   - `laplacian_min_var=0.0`
   - `camera_bn.enabled=false`
4. Reproduce the historical stage 5 filter state explicitly, especially:
   - `gt_zone_filter`
   - `gt_frame_clip`
   - `track_smoothing.enabled`
   - `cross_id_nms_iou`

Rules for ablation:

- Change one suspect at a time where possible.
- Reuse the same feature export when testing stage 4 and stage 5 logic, so results are not confounded by stage 2 changes.
- Log the exact override set for every run.
- Prioritize MTMC IDF1 deltas over secondary metrics when ranking suspects.

Deliverable:

- A suspect ranking table with measured delta from baseline for each ablation.

### Phase 4: Binary Search on Code Changes

Objective: if config reconstruction does not recover the gap, locate the responsible code change window.

Steps:

1. Use git bisect or manual midpoint checkout between the v80 commit and HEAD.
2. For each midpoint, run the narrowest viable reproduction path that still measures MTMC IDF1 reliably.
3. Start with stage 4 and stage 5 files if stage 2 export reuse is available. Use full stage 0-5 reproduction only when required.
4. Once the regression window is found, diff only the changed functions in that window.

Likely high-value functions to inspect first:

- `GraphSolver.solve` and `_conflict_free_greedy` in [src/stage4_association/graph_solver.py](../../src/stage4_association/graph_solver.py)
- candidate-pair filtering and graph-prep logic in [src/stage4_association/pipeline.py](../../src/stage4_association/pipeline.py)
- stage 2 crop/augmentation path in [src/stage2_features/pipeline.py](../../src/stage2_features/pipeline.py)
- GT-assisted filters in [src/stage5_evaluation/pipeline.py](../../src/stage5_evaluation/pipeline.py)

Deliverable:

- A narrowed commit window and a root-cause candidate diff.

### Phase 5: Fix Verification

Objective: prove the regression is understood and corrected.

Steps:

1. Apply the minimal config and/or code changes needed to match the v80 behavior.
2. Re-run the reproduction path on the same data path used for current best runs.
3. Verify MTMC IDF1 returns to approximately 0.784.
4. If the recovery exceeds or undershoots by more than about 0.2pp, inspect remaining Tier 2 suspects.
5. Lock the recovered behavior into documented config defaults or documented required overrides.

Deliverable:

- A verified reproduction report showing the recovered score and the exact fix set.

## 5. Expected Outcome

The expected outcome is restoration of the historical 0.784 MTMC IDF1 result, or at minimum a clear accounting of where the missing 1.2pp went.

Best-case outcome:

- One or two high-impact changes explain most of the drift, such as `min_hits=2` plus graph algorithm parity or GT-filter parity.

Acceptable outcome:

- The investigation proves that the gap is the sum of several smaller drifts and produces a reproducible v80-compatibility recipe.

Failure condition:

- No combination of code and config parity reproduces 0.784, implying the historical run also depended on external runtime artifacts or exported feature files that are no longer reproducible.

## 6. Risk Assessment

- The historical 0.784 run may depend on exact Kaggle runtime behavior, model artifact versions, or exported feature files that are not preserved in git.
- Some drift may come from stage 2 feature exports rather than only code in the current repository, which makes binary search slower.
- GT-assisted metrics complicate interpretation because small implementation changes can cause disproportionate score shifts.
- The current config in [configs/datasets/cityflowv2.yaml](../../configs/datasets/cityflowv2.yaml) has evolved toward a different operating point than v80, so naive rollback of one change may not recover the result unless related settings are rolled back together.

## 7. Recommended First Moves

The first investigation cycle should be deliberately narrow:

1. Reconstruct the exact v80 overrides, especially `min_hits=2` and stage 5 GT-assisted toggles.
2. Run the current codebase with those exact overrides.
3. If still below 0.784, switch only `stage4.association.graph.algorithm` from `conflict_free_cc` to `connected_components`.
4. If still below target, hold stage 4 fixed and ablate stage 2 feature-space drift from `color_augment`, `laplacian_min_var`, and `camera_bn`.

That order is the shortest path to determining whether the regression is mostly tracker/config drift, clustering drift, feature drift, or evaluation drift.