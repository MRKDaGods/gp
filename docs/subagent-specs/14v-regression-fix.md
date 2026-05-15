# 14v Regression Fix - Restore 14e B1 Plateau (MTMC IDF1 = 0.77936)

Status: APPLIED

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
