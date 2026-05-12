# Post-14a Next Experiment Spec — Multi-Crop TTA for Stage 2 ReID

**Date**: 2026-05-07
**Author**: MTMC Planner
**Status**: PROPOSED — ready for Coder implementation

## Decision

After 13d CLIP-SENet fusion, 13h CLIP-SENet CityFlow fine-tune fusion, and 14a SAM2 foreground masking all failed to clear the production-best **0.7703 MTMC IDF1**, the next experiment is **multi-crop test-time augmentation (TTA) for Stage 2 ReID feature extraction**. Stage 2 currently extracts a single feature per detection with no TTA. Standard ReID test-time tricks such as horizontal-flip averaging and multi-scale crops are documented to add roughly **+0.3-1.0pp R1** on VeRi-776 and have never been tried in this pipeline.

Implementation cost is small: only Stage 2 changes, reusing the existing TransReID checkpoint and DINOv2 checkpoint while keeping the downstream chain unchanged. Walltime fits one overnight: roughly **3-4x Stage 2 cost** for one forward pass plus 3-7 augmented views, about **30-50 min on P100** for the primary extractor, with multiple variants queueable. This is a higher-EV overnight candidate than a GNN edge classifier, which cannot finish in one shift per the post-13h analysis.

### Why not the alternatives

- **GNN edge classifier**: high-ceiling but multi-shift implementation and validation cost; better as the next escalation if TTA is neutral or dead.
- **Person pipeline pivot**: tracker-limited at ~0.947 IDF1 and already exhaustively swept; not aligned with the current vehicle breakthrough target.
- **Rank-fusion of existing features**: cheap enough to run as a fallback in the same kernel, but it reuses the same exhausted fixed-checkpoint feature streams and is less likely to change the embedding noise floor.
- **Paper pivot**: violates the user's NEVER-EXIT mandate for continued experiment search.

## Hypothesis

TTA averages out viewpoint- and pose-specific artifacts, producing more invariant tracklet means after the existing per-tracklet mean pooling stage. This is orthogonal to backbone capacity, which was the 13d/13h failure mode, and orthogonal to input cleanup, which was the 14a failure mode. It targets the per-detection embedding noise floor rather than adding another correlated feature stream or deleting context.

## Implementation Plan

Single Kaggle GPU kernel: `yahiaakhalafallah/14c-tta-stage2`. Reuse 10a v7 Stage 0/1 outputs from `yahiaakhalafallah/mtmc-10a-stages-0-2` v7, including detections and tracklets. For each detection crop, extract TransReID embeddings under N augmented views and mean-pool L2-normalized features.

Views:
1. View 0: original 256² center crop (current behaviour).
2. View 1: horizontal flip.
3. View 2: scale 0.95x, center crop 256².
4. View 3: scale 1.05x, center crop 256².
5. Optional Views 4-5: +/-5% bbox jitter plus center crop, with hflip variants.

Apply the same TTA principle to DINOv2 tertiary features, but use **original + hflip only** because DINOv2 is more sensitive to scale shifts. Per-tracklet pooling remains unchanged; each per-detection feature is the L2-normalized mean of the per-view features.

## Hyperparameters

- `stage2.tta.enabled = true`
- `stage2.tta.views = ["original", "hflip", "scale_0.95", "scale_1.05"]`
- `stage2.tta.dinov2_views = ["original", "hflip"]`
- `stage2.tta.aggregation = "mean_l2"`

Start with four TransReID views and ablate to two views only if walltime is tight. For `mean_l2`, L2-normalize each view embedding, mean the normalized vectors, then L2-normalize the mean.

## Downstream Chain

Use the production chain unchanged: 10b v6, then 10c v17 with `w_tertiary=0.60`, AQE `k=3`, FIC regularisation `0.50`, PCA 384D, gallery expansion, temporal overlap bonus, and `mtmc_only_submission=false`. `outputs/14a_v8_summary/14a_summary.json` records `similarity_threshold=0.50`; verify against the 13h fusion control or 10c v15 before push if production-best should use `0.40`.

## Stop Criteria

1. **WIN**: MTMC IDF1 >= **0.7720**. This clears production by >0.17pp, beyond run-to-run noise of ~0.24pp. Push an ablation kernel that drops from four views to two views to find the minimum-cost config.
2. **NEUTRAL**: MTMC IDF1 in **0.7680-0.7720**. Mark as MARGINAL and move to GNN.
3. **DEAD**: MTMC IDF1 < **0.7680**. Record likely cause, such as over-smoothing or the wrong augmentation set, close the branch, and escalate to a GNN edge classifier.
4. **WALLTIME**: cap Stage 2 at **4h on P100**. If not done, restart with two views only.

## Expected Impact Range

- **Optimistic**: +0.3 to +0.8pp -> **0.7733-0.7783**.
- **Central**: +0.0 to +0.3pp -> **0.7703-0.7733**.
- **Pessimistic**: -0.5 to +0.0pp -> over-smoothing destroys discriminative detail.

## Estimated Walltime

- Stage 2 primary TransReID: **~30-50 min** for four TTA views, ~99k detections, P100 batch 64.
- DINOv2 tertiary: **~30 min** for two views.
- 10b/10c CPU chain: **~1-2h**.
- **Total**: **~2-3.5h**, fitting one overnight slot easily.

## Coder Handoff Checklist

1. Build `notebooks/kaggle/14c_tta_stage2/` from a copy of the existing 10a Stage-2 cell.
2. Verify production-best 10c config from `outputs/14a_v8_summary/14a_summary.json` or the 13h fusion control.
3. Push **once**, watch for `not valid dataset sources`, and cancel + refix if seen.
4. Append the result to `docs/findings.md` and `docs/experiment-log.md`, then update `.github/copilot-instructions.md`.
5. On WIN: push a two-view ablation. On DEAD: write the next spec for a GNN edge classifier.