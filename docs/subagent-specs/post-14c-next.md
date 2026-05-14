# Post-14c Next Experiment Spec — CPU-Only Fusion Sweep on TTA Features

**Date**: 2026-05-07
**Author**: MTMC Planner
**Status**: PROPOSED — ready for Coder implementation

## Decision

14c v2 (multi-crop TTA at Stage 2) landed at **MTMC IDF1 = 0.77085**, which falls in the NEUTRAL band (0.7680–0.7720) per the post-14a spec. The lift over production 0.7703 is **+0.05pp**, well below the 0.7720 WIN threshold and within run-to-run noise of ~0.24pp. Before escalating to a multi-shift GNN edge classifier, run a **CPU-only fusion-weight sweep on the existing 14c v2 TTA features** as kernel `14d`. The TTA features are already extracted and uploaded; no GPU is needed and no Stage 0/1/2 work is repeated.

The hypothesis is narrow: TTA shifts the primary embedding distribution (smoother, lower-variance per detection), which may shift the optimal `w_tertiary` away from the production-tuned 0.60. Cost is ~30 min/config on CPU, with zero GPU slot consumption. If the best sweep configuration still does not clear 0.7720, the TTA family is closed and the next spec escalates to a GNN edge classifier.

### Why not the alternatives

- **GNN edge classifier**: high-ceiling but multi-shift implementation cost (~5-8h training plus validation). Better as the next escalation if 14d also lands NEUTRAL/DEAD.
- **More TTA view variants** (bigger scale span, max-pool, weighted aggregation): each variant requires a fresh GPU Stage 2 pass (~99 min) and TTA already shows monotonically diminishing returns; better to first confirm the existing features cannot be re-tuned downstream.
- **Person-pipeline pivot**: tracker-limited at ~0.947 IDF1 and exhaustively swept; no near-term breakthrough surface.
- **Paper pivot**: violates the user's NEVER-EXIT mandate and 14d closure is needed to finalize the TTA section of any future write-up.

## Hypothesis

Multi-crop TTA averaging produces per-detection embeddings that are slightly smoother and less viewpoint-specific than the single-view production features. The CLIP-primary / DINOv2-tertiary score fusion was tuned with `w_tertiary=0.60` on single-view features; TTA reduces primary noise, so the fusion optimum may shift toward higher CLIP weight (lower `w_tertiary`). A small reweighting could convert the +0.05pp marginal result into a clear WIN.

## Implementation Plan

Single Kaggle CPU kernel: `yahiaakhalafallah/14d-tta-fusion-sweep`. Reuse 14c v2 Stage 2 outputs (primary + tertiary embeddings, tracklets, detections) plus the same 10b v6 FAISS index inputs. Re-run only Stages 3 (re-index if PCA differs), 4 (association), and 5 (eval) under each fusion configuration.

### Sweep matrix (8 configs, sequential CPU)

| Config | `w_tertiary` | `similarity_threshold` | Notes |
|--------|:------------:|:----------------------:|-------|
| C0 | 0.60 | 0.50 | Control: replicate 14c v2 production-fusion |
| C1 | 0.55 | 0.50 | -0.05 around prod |
| C2 | 0.65 | 0.50 | +0.05 around prod |
| C3 | 0.50 | 0.50 | Wider exploration |
| C4 | 0.70 | 0.50 | Wider exploration |
| C5 | 0.60 | 0.40 | Lower threshold (10c v15-style) |
| C6 | 0.55 | 0.40 | Most likely combined optimum |
| C7 | 0.65 | 0.40 | Symmetric |

All other parameters held at production values: `aqe_k=3`, `fic_regularisation=0.50`, `pca_components=384`, `algorithm=conflict_free_cc`, gallery expansion enabled, temporal overlap bonus enabled, `mtmc_only_submission=false`.

If C0 does not reproduce 0.77085 ±0.001, halt and investigate before trusting the sweep.

## Stop Criteria

1. **WIN**: any config ≥ **0.7720**. Push a tighter sweep around the winning point and ablate threshold separately. Update production `w_tertiary` or `similarity_threshold` only after the winner replicates on a second seed/run.
2. **MARGINAL**: best config in **0.7680–0.7720**. Mark the TTA family closed and write `docs/subagent-specs/post-14d-next.md` for a GNN edge classifier escalation.
3. **DEAD**: best config < **0.7680**. Same closure as MARGINAL, plus record likely cause (TTA destroyed discriminative detail more than fusion can recover).
4. **WALLTIME**: cap at **3h on Kaggle CPU**. If incomplete, prioritize C0–C4 (single-axis `w_tertiary` sweep) and defer threshold axis.

## Expected Impact Range

- **Optimistic**: +0.10 to +0.30pp → **0.7718–0.7738** (clears WIN at the upper end).
- **Central**: +0.00 to +0.10pp → confirms TTA is genuinely noise-level and closes the family.
- **Pessimistic**: -0.10 to +0.00pp → fusion reweighting cannot help; TTA features carry the same information as single-view.

## Estimated Walltime

- 8 configs × ~30 min CPU = **~4h**. With C5–C7 deprioritized under the 3h cap, expect **~2.5h** for C0–C4.
- Zero GPU slot consumption.

## Coder Handoff Checklist

1. Build `notebooks/kaggle/14d_tta_fusion_sweep/` from a copy of the 10b/10c CPU chain. Wire `kernel_sources` to depend on `yahiaakhalafallah/14c-tta-stage2` for Stage 2 features and the existing 10b v6 dataset for FAISS inputs.
2. Verify the C0 control replicates 14c v2's 0.77085 before trusting C1–C7.
3. Push **once**, watch for `not valid dataset sources`, and cancel + refix if seen.
4. Append the result table (config × IDF1) to `docs/findings.md` under a new "14d Fusion Sweep on TTA Features" subsection and to `docs/experiment-log.md` as section 2.11.
5. On WIN: push a tighter ±0.025 sweep around the winner. On MARGINAL/DEAD: write `docs/subagent-specs/post-14d-next.md` proposing a GNN edge classifier (input: tracklet pair similarities + camera/time features; output: edge keep/drop probability; training: GT-derived edges from CityFlowV2 train split).