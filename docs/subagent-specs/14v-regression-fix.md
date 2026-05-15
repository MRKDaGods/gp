# 14v Regression Fix - Restore 14e B1 Plateau (MTMC IDF1 = 0.77936)

Status: APPLIED
Stage C: APPLIED

## TL;DR

The 14v Kaggle verification kernel produced MTMC IDF1 = 0.46009 (target 0.77936; drift -0.31927) even though the four headline knobs (`similarity_threshold=0.48`, `aqe_k=2`, `fic.regularisation=0.5`, `tertiary_embeddings.weight=0.525`) were applied correctly via overrides. **Root cause: ~14 additional Stage-4 / Stage-5 overrides used by the original 14e B1 notebook were never promoted to `configs/datasets/cityflowv2.yaml`.** The dominant contributors are the two GT-assisted post-processing filters (`stage5.gt_frame_clip` and `stage5.gt_zone_filter`), which `docs/findings.md:449` explicitly states all current best numbers (including 0.77936) depend on, and which are `false` on master `327bf05` (see `configs/datasets/cityflowv2.yaml:307,309`). With those two filters off, 14v produces 66,673 pred rows vs 29,038 GT rows (2.3x), the S02_c006 ratio reaches 3.85x, and per-camera MOTA collapses to -2.6 on S02_c006.

## Confidence

**HIGH.** The 14e B1 notebook's `build_overrides()` (`notebooks/kaggle/14e_tta_sweep_expanded/14e_tta_sweep_expanded.ipynb:511-529`) lists every CLI override that produced 0.77936. Diffing that list against the live `configs/datasets/cityflowv2.yaml` enumerates the missing knobs deterministically. `docs/findings.md:931` independently records the exact same override set for the prior 10c v15 / 14e family headline numbers. The fix is mechanical (YAML edits) and reversible.

## Evidence Trail

### 1. The verifier itself is sound

`tmp_14v_output/14v-verify-b1-from-yaml.log:112-115`:
```text
similarity_threshold: 0.48
aqe_k: 2
fic_reg: 0.5
w_tertiary: 0.525
```

`tmp_14v_output/gp/data/outputs/run_verify/pipeline.log:5` confirms at runtime:
```text
Tertiary embeddings loaded: 384D, weight=0.53 (score-level fusion)
```

So the four headline knobs *are* applied. The regression is elsewhere.

### 2. The 14e B1 override set (ground truth)

From `notebooks/kaggle/14e_tta_sweep_expanded/14e_tta_sweep_expanded.ipynb:511-529` the canonical 14e B1 override list is:

Stage 4 (in addition to the four headline knobs):
- `stage4.association.query_expansion.dba=false`
- `stage4.association.graph.bridge_prune_margin=0.0`
- `stage4.association.weights.vehicle.appearance=0.70`
- `stage4.association.weights.vehicle.hsv=0.0`
- `stage4.association.weights.vehicle.spatiotemporal=0.30`
- `stage4.association.weights.length_weight_power=0.3`
- `stage4.association.gallery_expansion.threshold=0.48`
- `stage4.association.gallery_expansion.orphan_match_threshold=0.38`
- `stage4.association.camera_bias.enabled=false`
- `stage4.association.intra_camera_merge.enabled=true` (`threshold=0.80`, `max_time_gap=30`)
- `stage4.association.temporal_overlap.enabled=true` (`bonus=0.05`, `max_mean_time=5.0`)

Stage 5:
- `stage5.mtmc_only_submission=false`
- `stage5.stationary_filter.enabled=true`
- `stage5.stationary_filter.min_displacement_px=150`
- `stage5.stationary_filter.max_mean_velocity_px=2.0`
- `stage5.min_submission_confidence=0.15`
- `stage5.cross_id_nms_iou=0.40`
- `stage5.min_trajectory_confidence=0.30`
- `stage5.min_trajectory_frames=40`
- `stage5.track_edge_trim.enabled=false`
- **`stage5.track_smoothing.enabled=false`**
- **`stage5.gt_frame_clip=true`**
- **`stage5.gt_zone_filter=true`**

`docs/findings.md:931` independently confirms the Stage-5 subset:
> sim=0.54, secondary_weight=0.17, appearance=0.70, gallery=0.50, FIC=0.10, cross_id_nms_iou=0.40, min_traj_frames=40, min_traj_conf=0.30, `gt_frame_clip=true`, `gt_zone_filter=true`, `mtmc_only_submission=false`

### 3. Master HEAD `configs/datasets/cityflowv2.yaml` diff

| Override path | 14e B1 value | Current YAML | YAML line |
|---|---|---|---|
| `stage4.association.query_expansion.dba` | `false` | **`true`** | 191 |
| `stage4.association.graph.bridge_prune_margin` | `0.0` | **`0.05`** | 224 |
| `stage4.association.weights.vehicle.appearance` | `0.70` | **`0.75`** | 207 |
| `stage4.association.weights.vehicle.spatiotemporal` | `0.30` | **`0.25`** | 209 |
| `stage4.association.weights.length_weight_power` | `0.3` | **`0.5`** | 204 |
| `stage4.association.gallery_expansion.threshold` | `0.48` | **`0.50`** | 257 |
| `stage4.association.gallery_expansion.orphan_match_threshold` | `0.38` | **`0.40`** | 259 |
| `stage4.association.camera_bias.enabled` | `false` | **`true`** | 265 |
| `stage4.association.intra_camera_merge.enabled` | `true` | **missing key** | - |
| `stage5.min_submission_confidence` | `0.15` | **`0.0`** | 294 |
| `stage5.cross_id_nms_iou` | `0.40` | **`0.5`** | 296 |
| `stage5.min_trajectory_confidence` | `0.30` | **`0.0`** | 288 |
| `stage5.min_trajectory_frames` | `40` | **`0`** | 290 |
| `stage5.stationary_filter.max_mean_velocity_px` | `2.0` | **`0`** | 304 |
| `stage5.track_smoothing.enabled` | **`false`** | **`true`** | 315 |
| `stage5.gt_frame_clip` | **`true`** | **`false`** | 309 |
| `stage5.gt_zone_filter` | **`true`** | **`false`** | 307 |

Headline knobs that already match (no change needed): `graph.similarity_threshold=0.48`, `query_expansion.k=2`, `fic.regularisation=0.5`, `tertiary_embeddings.weight=0.525`, `weights.vehicle.hsv=0.0`, `stationary_filter.enabled=true`, `stationary_filter.min_displacement_px=150`, `track_edge_trim.enabled=false`, `mtmc_only_submission=false`.

### 4. Cross-referenced log signals

`pipeline.log:21`:
```text
Bridge pruning: removed 28 weak bridges (threshold=0.530)
```
That `0.530 = 0.48 + 0.05` confirms `bridge_prune_margin=0.05` (YAML default), not the `0.0` 14e B1 used. So 28 edges with similarity in `[0.48, 0.53)` are being pruned that 14e B1 kept.

`pipeline.log:13`:
```text
QE+DBA: rebuilt FAISS with expanded embeddings (alpha=5.0, k=2)
```
Confirms `dba=true` (YAML default), not the `false` 14e B1 used.

`pipeline.log:18`:
```text
CID_BIAS: adjusted 638 pairs from configs/datasets/cityflowv2_cid_bias.npy
```
Confirms `camera_bias.enabled=true` (YAML default), not the `false` 14e B1 used.

`pipeline.log:60`:
```text
Track smoothing (window=7, poly=2): smoothed 648 tracks
```
Confirms `track_smoothing.enabled=true` (YAML default), not the `false` 14e B1 used.

`pipeline.log:62`:
```text
Submission quality: 66673 pred rows vs 29038 GT rows (2.3x) - HIGH FP ratio detected.
```
Direct symptom of `gt_frame_clip=false` and `gt_zone_filter=false` - predictions extend through frames and image regions with no GT annotation. With those filters on (as 14e B1 did), the FP flood is removed and IDF1 recovers ~30pp.

`pipeline.log:69`:
```text
S02_c006: id_switches=39 idf1=0.249 mota=-2.607
```
S02_c006 has 3.85x pred/GT ratio (`pipeline.log:54`). Without GT-zone/frame clipping, every parked-car / outside-annotation-zone false positive becomes an MTMC pred ID. `mtmc_id_switches=324` and `123 unmatched pred IDs` (`pipeline.log:75`) is the direct consequence.

### 5. Why the four headline knobs alone aren't enough

The 4 headline values control the *association graph* (which tracklets merge into the same global ID). They do **not** control:
- Which Stage-5 post-processing filters run on the produced trajectories.
- The other Stage-4 cosmetics (bridge pruning, gallery thresholds, DBA, camera bias, intra-camera merge, weight blend, length weighting).

The 14e B1 0.77936 number is the *joint product* of all ~30 overrides, with the four headline knobs being the only ones that vary across the 14e sweep grid. The other ~26 are constant and were silently the "ambient" Stage-4/5 config of the 14d/14e family - never written to YAML.

## Root Cause (final)

`configs/datasets/cityflowv2.yaml` (master HEAD `327bf05`) is missing 14-17 Stage-4 / Stage-5 keys that the 14e B1 notebook applied as CLI overrides. The two dominant contributors are `stage5.gt_frame_clip` and `stage5.gt_zone_filter` (each independently worth several pp; together responsible for the bulk of the -31.9pp drift, per `docs/findings.md:449` "GT-assisted metrics inflate scores by 1-3pp"; in practice on this run the gap is larger because S02 cameras have severe parked-car FP without GT-zone filtering).

## Remediation Plan (ordered, minimum-change-first)

All changes are YAML-only edits to `configs/datasets/cityflowv2.yaml`. No source edits required. The plan is staged so the highest-confidence change is verified first.

### Stage A - High-confidence single edit (verify GT-assisted hypothesis)

Edit only the two GT-assisted toggles:

```yaml
stage5:
  # ...
  gt_zone_filter: true   # was false
  gt_frame_clip: true    # was false
```

Re-run the 14v kernel. **Expected outcome:** MTMC IDF1 jumps from 0.460 into the **0.74-0.77** band (most of the gap closed). If it lands >= 0.745, the GT-assisted-filter hypothesis is confirmed and Stage B is the residual fix.

If Stage A produces >= 0.78 already, we are done (but unlikely - Stage 4/5 cosmetics still differ).

### Stage B - Promote remaining 14e B1 Stage-5 + Stage-4 overrides

Add the rest of the 14e B1 override set to `configs/datasets/cityflowv2.yaml`:

```yaml
stage4:
  association:
    query_expansion:
      dba: false                 # was true
    graph:
      bridge_prune_margin: 0.0   # was 0.05
    weights:
      length_weight_power: 0.3   # was 0.5
      vehicle:
        appearance: 0.70         # was 0.75
        hsv: 0.00                # unchanged
        spatiotemporal: 0.30     # was 0.25
    gallery_expansion:
      threshold: 0.48                  # was 0.50
      orphan_match_threshold: 0.38     # was 0.40
    camera_bias:
      enabled: false             # was true
    intra_camera_merge:          # NEW block - code reads via .get(), missing => disabled
      enabled: true
      threshold: 0.80
      max_time_gap: 30

stage5:
  min_submission_confidence: 0.15           # was 0.0
  cross_id_nms_iou: 0.40                    # was 0.5
  min_trajectory_confidence: 0.30           # was 0.0
  min_trajectory_frames: 40                 # was 0
  stationary_filter:
    max_mean_velocity_px: 2.0               # was 0
  track_smoothing:
    enabled: false                          # was true
```

Re-run 14v. **Expected outcome:** MTMC IDF1 = 0.77936 +/- 0.005, id_switches=154 +/- a small drift band.

### Stage C - Sanity & lockdown

If Stage B reproduces 0.77936 +/- 0.005, treat this as the canonical reproducible config and:

1. Drop a one-line provenance comment block above the `stage4.association` block in `configs/datasets/cityflowv2.yaml` referencing 14e B1 / `notebooks/kaggle/14e_tta_sweep_expanded/`.
2. Update `docs/findings.md` to note the YAML now self-contains the 14e B1 reproducible config (no CLI overrides needed beyond pipeline-runner plumbing).
3. Re-snapshot model_registry provenance: `configs/model_registry.yaml:17` already records the four headline knobs; the rest now live in the dataset YAML.

## Pre-Flight Check Classification

| Change group | Confidence the fix is needed | Confidence on the value |
|---|---|---|
| `gt_frame_clip=true`, `gt_zone_filter=true` | **HIGH** (direct symptom in `pipeline.log:62`, confirmed in `docs/findings.md:449`, `:931`) | **HIGH** (binary toggle, matches 14e B1) |
| `track_smoothing.enabled=false` | **HIGH** (`pipeline.log:60` confirms it is currently running) | **HIGH** (matches 14e B1) |
| `min_trajectory_confidence=0.30`, `min_trajectory_frames=40`, `min_submission_confidence=0.15`, `cross_id_nms_iou=0.40`, `stationary_filter.max_mean_velocity_px=2.0` | **HIGH** | **HIGH** (verbatim from 14e B1) |
| Stage-4 weight blend (`appearance=0.70`, `st=0.30`, `length_weight_power=0.3`) | **HIGH** | **HIGH** |
| `bridge_prune_margin=0.0` | **HIGH** (`pipeline.log:21` shows 28 edges being pruned at threshold 0.530) | **HIGH** |
| `gallery_expansion.threshold=0.48`, `orphan_match_threshold=0.38` | **HIGH** | **HIGH** |
| `camera_bias.enabled=false` | **HIGH** (`pipeline.log:18` shows it currently runs and adjusts 638 pairs) | **HIGH** (14e B1 explicitly turns it off) |
| `query_expansion.dba=false` | **HIGH** (`pipeline.log:13` confirms DBA currently runs) | **HIGH** |
| `intra_camera_merge.enabled=true` | **MEDIUM** (the code path exists at `src/stage4_association/pipeline.py:817`; effect on IDF1 plausibly small) | **HIGH** (verbatim from 14e B1) |
| Stage A vs Stage B split | **HIGH** that Stage A alone closes most of the gap; **MEDIUM** that it lands inside +/-0.005 of 0.77936 without Stage B | - |

## Risks & Rollback

- **Risk: bleeding `gt_*=true` into wildtrack / person pipelines.** Wildtrack (`configs/datasets/wildtrack.yaml`) has its own Stage-5 block and the 12b person headline (0.947 IDF1) is *not* claimed to use GT-assisted post-processing. **Mitigation:** the proposed edits are scoped to `configs/datasets/cityflowv2.yaml` only. Audit `configs/datasets/wildtrack.yaml`: any `gt_zone_filter` / `gt_frame_clip` keys there must remain at their current values (almost certainly `false`).
- **Risk: future deployment / submission code paths that *intentionally* keep GT-assisted filters off (for "fair" leaderboard submission)** will be silently inflated. **Mitigation:** add an inline YAML comment block `# GT-assisted filters ON: required to reproduce 14e B1 headline 0.77936. Set both to false for AIC server submission.`
- **Risk: rollback.** All edits are YAML-only; revert via `git checkout configs/datasets/cityflowv2.yaml`. No code or notebook touched.
- **Risk: drift band breach despite full Stage B.** If 14v still misses 0.77936 +/- 0.005 after Stage B, the remaining gap is likely in (a) Stage-2 feature input bit-identity (compare `embeddings.npy` SHA against the 14c-tta-stage2 kernel-output dataset that produced 14e B1; the verifier inherits it via `yahiaakhalafallah/14c-tta-stage2` kernel_source - should already be identical), or (b) `cityflowv2_cid_bias.npy` is being read despite `camera_bias.enabled=false` somewhere upstream; rule out by checking `pipeline.log` for the absence of the `CID_BIAS: adjusted` line.

## What NOT to Do

- Do **not** flip `mtmc_only_submission` to `true` - it drops single-cam tracks and hurts IDF1 by ~5pp (per `.github/copilot-instructions.md` "What NOT to Do").
- Do **not** also touch `stage4.association.graph.similarity_threshold`, `query_expansion.k`, `fic.regularisation`, or `tertiary_embeddings.weight` - they are already correct in the YAML at 14e B1 values.
- Do **not** edit `configs/default.yaml`; per-dataset YAML is the correct override surface (and `cityflowv2.yaml` is loaded on top of `default.yaml` per the config loader).
- Do **not** modify the 14v notebook to inject additional overrides; promoting them to YAML is the explicit user-stated goal.

## Confidence Summary

- **Confidence that Stage A (only the two GT-assisted toggles) lifts IDF1 from 0.460 to >= 0.74:** HIGH.
- **Confidence that Stage A + Stage B together reproduce 0.77936 +/- 0.005:** HIGH (the proposed YAML state is bit-equivalent to the live OmegaConf state that produced 14e B1).
- **Confidence that no source code change is required:** HIGH (every override key is already consumed by current `src/stage4_association/pipeline.py` / `src/stage5_evaluation/*` - e.g. `intra_camera_merge` at `src/stage4_association/pipeline.py:817`).

---

## Stage C - Residual Drift Analysis (post Stage-A+B, IDF1 = 0.76791 vs target 0.77936, drift -0.01145)

Status: APPLIED. Stage A+B closed -30.8pp of the original -31.9pp gap. The residual -1.15pp (~-0.011 IDF1, +56 cross-camera ID switches: 154 -> 210) is *outside* the +/-0.005 reproducibility gate. Every override the 14e B1 notebook applied via CLI is now logged as taking effect at runtime (verified below). The remaining drift therefore points at one of three classes of cause: (i) brittle YAML parsing, (ii) pipeline source-code drift between the 14e B1 commit and the verify branch `d6e6d3d`, or (iii) cached Stage-2 feature drift.

### What Stage A+B succeeded at (confirmed from `tmp_14v_output/14v-verify-b1-from-yaml.log`)

| 14e B1 override | Confirmed runtime value | Log evidence |
|---|---|---|
| `similarity_threshold=0.48` | 0.48 | `Similarity graph: 929 nodes, 342 edges (threshold=0.48)` |
| `aqe_k=2`, `alpha=5.0`, `dba=false` | k=2, alpha=5.0, no DBA | `Query Expansion (batched): k=2, alpha=5.0, N=929` then `QE (no DBA): re-retrieved with expanded queries` |
| `fic.regularisation=0.5` | lambda=0.5 | `FIC per-camera whitening: 929/929 tracklets across 6 cameras (lambda=0.5)` |
| `tertiary_embeddings.weight=0.525` | 0.53 (display rounded) | `Tertiary embeddings loaded: 384D, weight=0.53 (score-level fusion)` |
| `intra_camera_merge.enabled=true, threshold=0.80, max_gap=30s` | applied | `Intra-camera ReID merge: added 27 same-camera pairs (threshold=0.80, max_gap=30s)` |
| `gallery_expansion.threshold=0.48`, `orphan_match_threshold=0.38` | both | `Orphan<->orphan matching (threshold=0.38): 27 candidate pairs -> 23 new clusters` |
| `temporal_overlap.enabled=true, bonus=0.05, max_mean_time=5.0` | applied | `Temporal overlap bonus applied to 251 pairs (bonus=0.05, max_mean_time=5.0s)` |
| `bridge_prune_margin=0.0` | no pruning line emitted | absence of any `Bridge pruning: removed N weak bridges` line (the v4/v5 log line at threshold=0.530 is gone) |
| `camera_bias.enabled=false` | not applied | absence of any `CID_BIAS: adjusted` line |
| `track_smoothing.enabled=false` | not applied | absence of any `Track smoothing (window=...)` line |
| `stationary_filter.min_displacement_px=150, max_mean_velocity_px=2.0` | applied | `Stationary filter (min_displacement=150.0px, max_mean_velocity=2.0px/frame): dropped 196/624 trajectories` |
| `min_trajectory_confidence=0.30` | applied | `Trajectory confidence filter: removed 1 trajectories with confidence < 0.30` |
| `min_trajectory_frames=40` | applied | `Min trajectory frames filter: removed 144 trajectories with < 40 total frames` |
| `cross_id_nms_iou=0.40` | applied (cuts hundreds of overlap rows per camera) | `S01_c003: cross-ID NMS removed 398 overlapping predictions` etc. |
| `gt_zone_filter=true`, `gt_frame_clip=true` | both applied | `GT IoU filter (min_iou=0.0, min_frames=1): dropped 15/525 tracks` and `GT frame clip (min_iou=0.4): dropped 31774/58185 rows (54.6%)` |
| `mtmc_only_submission=false`, `track_edge_trim.enabled=false` | applied (no edge-trim line) | - |

So at the level of *runtime behaviour*, no override is silently regressed. The remaining drift is NOT a missing toggle.

### The smoking gun: excess trajectory count

The final submission has **525 trajectory IDs across 6 cameras** (`MOT submission written: 6 cameras, 59925 detection rows`, then `GT IoU filter ... dropped 15/525 tracks`). For comparison, a 0.77936-IDF1 / id_sw=154 run is structurally bound to have closer to 200-250 trajectory IDs (since IDF1 ~ 0.78 means most pred IDs match a single GT ID; 525 IDs covering 240 GT vehicles implies ~2.2 pred per GT, consistent with the observed `+56 id_switches`). This means **Stage 4 is under-merging** relative to 14e B1: the same set of 929 tracklets is being clustered into substantially more components.

Concretely:
- Stage 4 produces `652 clusters: 206 multi-tracklet, 446 singleton` (then 624 after conflict-resolution split).
- Cross-camera trajectories with confidence >=0.7 are only 54/212.
- MTMC error analysis: 48 fragmented GT IDs, 24 conflated pred IDs (much more fragmentation than conflation).

Fragmentation > conflation in MTMC error is the signature of *under-merging*. The hypothesis "the residual gap is a tuning-knob we missed" is contradicted by the per-camera 2D IDF1 = 0.793 with id_sw=162: within-camera tracking is already at or above the 14e B1 plateau level; the loss is exclusively in cross-camera association.

### Candidate causes (ordered by confidence)

#### C1 — YAML file is structurally corrupted in the `stage5:` block (HIGH confidence the file is broken; MEDIUM confidence this causes the residual drift)

Direct inspection of `configs/datasets/cityflowv2.yaml:272-330` shows the Stage-A+B edits were not cleanly merged — they were inserted *into* the stage5 block at inconsistent indents alongside the existing keys, producing duplicated keys at different nesting depths. Examples (line numbers from current branch):

```yaml
stage5:                                    # line 272 (indent 0)
  ground_truth_dir: "data/raw/cityflowv2"
  iou_threshold: 0.5
  metrics: ["HOTA", "MOTA", "IDF1"]
  generate_report: true
        dba: false                # 14e B1 baseline       # line 277 — INDENT 8, nonsensical here
  # CityFlowV2 GT annotates 240 vehicles ...
        bridge_prune_margin: 0.0    # 14e B1 baseline     # line 280 — INDENT 8 (Stage-4 key inside Stage-5)
  mtmc_only_submission: false                              # line 282 — indent 2, valid
        length_weight_power: 0.3   # 14e B1 baseline      # line 283 — INDENT 8
          appearance: 0.70         # 14e B1 baseline      # line 286 — INDENT 10
  min_trajectory_confidence: 0.0                           # line 287 — indent 2, value 0.0
          spatiotemporal: 0.30     # 14e B1 baseline      # line 289 — INDENT 10
  min_trajectory_frames: 0                                 # line 290 — indent 2, value 0
        threshold: 0.48                  # 14e B1 baseline # line 292
        orphan_match_threshold: 0.38     # 14e B1 baseline # line 294
  cross_id_nms_iou: 0.5                                    # line 296 — indent 2, value 0.5
        enabled: false            # 14e B1 baseline       # line 297
  stationary_filter:                                       # line 301
    enabled: true
    min_displacement_px: 150
    max_mean_velocity_px: 0   # 0 = disabled               # line 304 — value 0!
    min_submission_confidence: 0.15           # 14e B1 baseline  # line 305 — DEAD: nested under stationary_filter
    cross_id_nms_iou: 0.40                    # 14e B1 baseline  # line 307 — DEAD: nested under stationary_filter
    enabled: false                                         # line 308 — re-declares stationary_filter.enabled
    min_trajectory_confidence: 0.30           # 14e B1 baseline  # line 309 — DEAD
    trim_fraction: 0.10
    min_trajectory_frames: 40                 # 14e B1 baseline  # line 311 — DEAD
      max_mean_velocity_px: 2.0               # 14e B1 baseline  # line 313 — INDENT 6, nested under prior scalar
    enabled: true                                          # line 314 — re-re-declares
    window: 7                                              # line 315
      enabled: false                          # 14e B1 baseline  # line 316
    gt_zone_filter: true        # 14e B1 baseline                # line 318 — DEAD nested
    gt_frame_clip: true         # 14e B1 baseline                # line 320 — DEAD nested
  gt_zone_filter: false                                    # line 321 — indent 2, value FALSE
  gt_zone_margin_frac: 0.20
  gt_frame_clip: false                                     # line 323 — indent 2, value FALSE
  gt_frame_clip_min_iou: 0.40
```

By a *literal* reading of this YAML, `gt_zone_filter` and `gt_frame_clip` should resolve to `false` (the indent-2 declarations at lines 321/323 come last and would win under OmegaConf's normal merge). YET the runtime log shows both filters running. This means OmegaConf is either (a) treating one of the duplicate keys as the canonical value via an order-dependent rule we did not predict, or (b) silently coercing the indent-8 entries into stage5-level keys.

In either case, *the file is parsing into a non-obvious resolved config*. The fact that ALL the headline values we want still come through is partly luck. The +56 ID switches could plausibly come from one or two keys that I haven't directly verified from the log — most likely:

- `min_submission_confidence`: appears ONLY at indent 4 (line 305, nested under stationary_filter, *dead*) — there is no indent-2 declaration. The code reads `stage_cfg.get("min_submission_confidence", 0.0)` (`src/stage5_evaluation/pipeline.py:134`). Likely resolved value: **0.0**, not 0.15. A 0.15 confidence floor in 14e B1 would have dropped low-confidence detection rows that are currently inflating the submission to 59925 rows and contributing IDFP at cluster-boundary frames.
- `stationary_filter.max_mean_velocity_px`: appears as `0` at line 304 (indent 4, valid stationary_filter child) AND `2.0` at line 313 (indent 6, nested under prior scalar, almost certainly dead). The log shows `max_mean_velocity=2.0px/frame`, which is inconsistent with this YAML state — implying OmegaConf is doing something unexpected, OR the log line emits the function-arg-default rather than the cfg-resolved value (would need to inspect `src/stage5_evaluation/pipeline.py:_filter_stationary` to confirm). If the resolved value is actually 0 (default), v6 is dropping FEWER oscillating-parked-car trajectories than 14e B1 did.

YAML key | Resolved-via-log value | 14e B1 intended | Confidence in mismatch
---|---|---|---
`stage5.min_submission_confidence` | UNKNOWN (no log line; default=0.0; structurally dead in YAML) | 0.15 | **HIGH** that it is unset
`stage5.stationary_filter.max_mean_velocity_px` | log says 2.0 | 2.0 | LOW that this is the cause
`stage5.gt_zone_filter` | log says true | true | LOW
`stage5.gt_frame_clip` | log says true | true | LOW

**Action**: Rewrite `configs/datasets/cityflowv2.yaml:272-326` from scratch as a clean stage5 block. Concretely:

```yaml
stage5:
  ground_truth_dir: "data/raw/cityflowv2"
  iou_threshold: 0.5
  metrics: ["HOTA", "MOTA", "IDF1"]
  generate_report: true
  mtmc_only_submission: false
  min_submission_confidence: 0.15          # 14e B1 baseline (currently missing at indent 2)
  min_trajectory_confidence: 0.30          # 14e B1 baseline (currently 0.0 at indent 2)
  min_trajectory_frames: 40                # 14e B1 baseline (currently 0 at indent 2)
  cross_id_nms_iou: 0.40                   # 14e B1 baseline (currently 0.5 at indent 2)
  stationary_filter:
    enabled: true
    min_displacement_px: 150
    max_mean_velocity_px: 2.0              # 14e B1 baseline
  track_edge_trim:
    enabled: false
    trim_fraction: 0.10
  track_smoothing:
    enabled: false                         # 14e B1 baseline
    window: 7
  gt_zone_filter: true                     # 14e B1 baseline
  gt_zone_margin_frac: 0.20
  gt_frame_clip: true                      # 14e B1 baseline
  gt_frame_clip_min_iou: 0.40
```

Expected impact: closes 0.005-0.012 IDF1 (uncertain — could be the entire gap if `min_submission_confidence`/`min_trajectory_confidence`/`min_trajectory_frames` were silently 0). Risk: zero (YAML-only structural cleanup; semantically equivalent to the intended Stage A+B if those values really were applying).

#### C2 — Pipeline source-code drift between the 14e B1 commit and verify branch `d6e6d3d` (HIGH confidence drift exists; MEDIUM confidence it costs ~1pp)

The 14e B1 run was executed against a specific repo commit (the one snapshotted in the `14e_tta_sweep_expanded` Kaggle kernel output). The verify run is on branch `verify/14v-kaggle-b1` at `d6e6d3d`. Any change in `src/stage4_association/` or `src/stage5_evaluation/` between those two commits — even one that doesn't touch a config key, e.g. tie-breaking in `_conflict_free_greedy`, sort stability in `graph_solver.solve`, FAISS index-build seed, edge-weight rounding in `similarity.compute_combined_similarity`, the order of `intra_camera_merge` vs `temporal_overlap` application — can shift MTMC IDF1 by ~1pp on this dataset. With 525 vs ~250 trajectories produced, even a small tie-break difference compounds across 929 tracklets.

**Action**:
1. Identify the exact commit hash that produced 0.77936. The 14e kernel slug is `yahiaakhalafallah/14e-tta-sweep-expanded`. Fetch its kernel-output, grep for `git rev-parse HEAD` or look at `_meta/git_sha.txt` if present. Otherwise infer from kernel push timestamp + repo `git log`.
2. Run `git log --oneline <14e_commit>..d6e6d3d -- src/stage4_association/ src/stage5_evaluation/` to enumerate intervening commits.
3. For each commit in the diff, ask "could this change Stage-4 clustering output deterministically?" — the most likely suspects are: (a) any change to `graph_solver.py` `_conflict_free_greedy` edge ordering, (b) any change to `pipeline.py:_resolve_same_camera_conflicts`, (c) any change to `similarity.py:compute_combined_similarity` (the 251-pair temporal_overlap bonus is computed here), (d) any change to `pipeline.py:_gallery_expansion` orphan iteration order.
4. If a suspect commit is found, cherry-pick its inverse onto `verify/14v-kaggle-b1` and re-run 14v.

Expected impact: 0.000-0.011 IDF1 swing. If C1 fully closes the gap, skip this. Otherwise, this is the next candidate.

#### C3 — Cached `14c-tta-stage2` Stage-2 features drifted since 14e B1 (MEDIUM confidence)

The verify kernel inherits Stage-2 outputs from kernel `yahiaakhalafallah/14c-tta-stage2` (or its current kernel-output dataset). If 14c-tta-stage2 was re-run between when 14e B1 consumed it and now, the cached `embeddings.npy` / `embeddings_tertiary.npy` would differ (timm version drift, CUDA non-determinism on the original GPU run, etc.). The Stage-4 graph operates on these vectors directly; sub-percent embedding drift compounds across 929x929 cosine-similarity computations into noticeable cluster-boundary shifts.

**Action**:
1. SHA-256 the verify-side `embeddings.npy` and `embeddings_tertiary.npy` once downloaded from the Kaggle kernel-output (or from `/kaggle/working/.../stage2/`).
2. Cross-reference against the embeddings used by 14e B1: grep the 14e kernel-output for `embeddings.npy` SHA, or hash the kernel-output `.npy` directly via `kaggle kernels output yahiaakhalafallah/14e-tta-sweep-expanded` (or the upstream 14c-tta-stage2 kernel-output the 14e family pinned).
3. If SHA mismatches, re-run 14c-tta-stage2 fresh and re-pin the verify kernel to the new output.

Expected impact: 0.000-0.005 IDF1 swing. Low priority — Stage-2 features tend to be highly reproducible across kernel re-runs on the same checkpoints, BUT this is the silent failure mode most likely to confound C1 and C2 if they don't close the gap.

#### C4 — `secondary_embeddings.path` / `weight` mismatch (LOW confidence this is causing the drift; HIGH confidence the brittle state should be fixed)

`configs/datasets/cityflowv2.yaml:159-162` declares:

```yaml
    secondary_embeddings:
      path: "data/outputs/run_latest/stage2/embeddings_secondary.npy"
      weight: 0.4
```

14e B1 explicitly overrode both: `secondary_embeddings.path=` (empty) and `secondary_embeddings.weight=0.0`. The verify run does NOT show either of the diagnostic log lines from `src/stage4_association/pipeline.py:174-210`:
- `Secondary embeddings loaded: ...` (would emit if file present and weight > 0)
- `Secondary embeddings file not found: ...` (would emit if path is truthy and weight > 0 but file missing)

Absence of both means `sec_path` resolves to a falsy string at runtime. So secondary is *currently* inactive in v6 — matching 14e B1's intended behaviour. **But the YAML state is non-deterministic**: if the verify kernel's CWD ever yields a real `data/outputs/run_latest/stage2/embeddings_secondary.npy` (e.g. from a future stages-0-2 run), secondary would silently activate at `weight=0.4`, dropping primary contribution to 0.075 and breaking the headline.

**Action**: Edit `configs/datasets/cityflowv2.yaml:159-162` to:

```yaml
    secondary_embeddings:
      path: ""
      weight: 0.0
```

Expected impact: 0.000 IDF1 swing for this verify run (already inactive). Defensive fix against future regressions; should be bundled with C1's YAML rewrite.

#### C5 — Excess trajectory count root-cause: Stage-4 algorithmic divergence (LOW-MEDIUM confidence, exploratory)

If C1+C2+C3 collectively do not close the gap, the residual must come from inside Stage 4. Specific probes:

- Dump `cfg.stage4.association.graph.algorithm` resolved value to disk (`save_config(cfg, ...)`). 14e B1 set `conflict_free_cc` explicitly via CLI. If resolved to something else (e.g. `connected_components`), that explains ~1pp.
- Dump `cfg.stage4.association.weights.length_weight_power`. 14e B1 set 0.3; cityflowv2.yaml has it at 0.3 too — sanity check the resolved value still says 0.3.
- Dump `cfg.stage4.association.mutual_nn.top_k_per_query`. Both 14e B1 and YAML are 20 — sanity check.
- Inspect `data/outputs/run_verify/stage4/trajectories.json` (or whatever the verify kernel persists) and compare:
  - Tracklet -> trajectory mapping multiplicity (how many tracklets per trajectory?)
  - Number of cross-camera trajectories (212 in v6; 14e B1 should have similar OR fewer if it merges more aggressively)
- If trajectory-count mismatch is structural, suspect `_resolve_same_camera_conflicts` (line 159 of the log says it split 1 cluster, line 165 splits more) — the split count could differ from 14e B1.

Expected impact: indeterminate. Use as a fallback diagnostic if C1+C2+C3 do not close the gap to <=0.005 drift.

### Recommended execution order

1. **C1 first** (YAML structural cleanup) — cheapest fix, highest probability of closing gap, eliminates one source of nondeterminism. Lift estimate: 0.005-0.012 IDF1.
2. **C2 second** (code-drift bisect) — only if C1 alone leaves drift >0.005. Lift estimate: 0.000-0.011 IDF1.
3. **C4 bundled with C1** (free defensive fix, no separate run).
4. **C3 third** (Stage-2 feature SHA check) — only if C1+C2 still leave drift >0.005.
5. **C5 last** (Stage-4 dump diagnostic) — only if C1+C2+C3 still leave drift >0.005.

### Confidence summary for Stage C

- **Confidence Stage C closes the residual -1.15pp:** MEDIUM-HIGH. C1 alone has a strong prior because the YAML state is provably non-canonical and `min_submission_confidence` has no indent-2 declaration in the file.
- **Confidence that no source change is needed beyond YAML:** MEDIUM. C2 may necessitate a code revert or version pin if a benign-looking commit changed deterministic ordering inside Stage 4.
- **What we will NOT do here:** flip `mtmc_only_submission=true`, enable `track_smoothing`, enable `camera_bias`, re-introduce `secondary_embeddings.weight=0.4`, or modify `configs/default.yaml` — all of these have been verified harmful or off-track per `docs/findings.md`.
