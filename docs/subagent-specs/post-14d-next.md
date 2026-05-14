# Post-14d Next Experiment Spec — Tighter Fusion + AQE/FIC Sweep on TTA Features (`14e`)

**Date**: 2026-05-07
**Author**: MTMC Planner
**Status**: PROPOSED — ready for Coder implementation

## Decision

14d v1 produced a **MARGINAL POSITIVE** result on the 14c v2 TTA features:

| Config | `w_tertiary` | `sim_thresh` | MTMC IDF1 | Δ vs control |
|:------:|:------------:|:------------:|:---------:|:-----------:|
| C0 (control) | 0.60 | 0.50 | 0.77085 | — |
| C1 | 0.55 | 0.50 | 0.77149 | +0.06pp |
| C2 | 0.65 | 0.50 | 0.77124 | +0.04pp |
| **C3 (best)** | **0.50** | **0.50** | **0.77155** | **+0.07pp** |
| C4 | 0.70 | 0.50 | 0.77115 | +0.03pp |
| C5–C7 | 0.55–0.65 | 0.40 | ~0.7566 | -1.4pp |

Versus production 0.7703, C3 lifts MTMC IDF1 by **+0.13pp** to 0.77155. This is below the 0.7720 WIN threshold from `post-14c-next.md` but the **trackeval IDF1** increased from 0.7866 (production) to 0.7897 (+0.31pp), the optimum shifted from `w_t=0.60` (production-tuned on single-view features) to `w_t=0.50` on TTA features (a real signal that TTA changed the embedding distribution), and the +0.05–0.13pp lift is consistent across the full `w_t∈{0.50,0.55,0.60,0.65,0.70}` family at `thr=0.50`. Declaring DEAD with this much consistency would be premature. Declaring WIN would be premature too — the lift is within the ~0.24pp run-to-run noise band.

The next move is a **second CPU-only sweep on the same 14c v2 features** that combines (a) a tighter `w_tertiary × sim_thresh` grid around the C3 optimum with (d) an AQE/FIC axis untouched on TTA features. Both are independent levers with zero GPU cost. Total cost ~10-15 min CPU, 16-18 configs, single Kaggle kernel. If best result still falls in 0.7715–0.7720 (NEUTRAL), declare TTA family closed and write `post-14e-next.md` proposing the GNN edge classifier escalation.

### Why not the alternatives

- **Aggressive 5-crop TTA** (90/270 rotation, random crops, max-pool): each variant requires fresh GPU Stage 2 (~99 min) and the diminishing returns from 14c → 14d argue against more view variants without first ruling out parameter mis-tuning on the existing features.
- **GNN edge classifier**: high-ceiling but multi-shift implementation cost (~5-8h training plus integration). 14e is a 15-minute investment that, if it produces ≥0.7720, makes the GNN unnecessary; if it plateaus, the TTA family is decisively closed and the GNN spec follows immediately.
- **Paper pivot**: violates the user's NEVER-EXIT mandate; one more cheap CPU sweep is justified before the production-best moves to TTA features.

## Hypothesis

Two independent corrections may stack on the C3 optimum:

1. **Sub-step around C3**: with thr=0.50 fixed, the IDF1 surface across `w_t∈{0.50, 0.55, 0.60, 0.65, 0.70}` reads {0.77155, 0.77149, 0.77085, 0.77124, 0.77115} — a noisy but unimodal curve peaking at 0.50. A finer grid `w_t∈{0.45, 0.475, 0.50, 0.525}` may locate a slightly sharper peak. Threshold should also be probed at `thr∈{0.48, 0.50, 0.52}` since the production `thr=0.50` was tuned on different feature variance.
2. **AQE/FIC re-tune**: production has `aqe_k=3, fic_reg=0.50`. AQE expansion smooths queries via top-k neighbours; FIC regularisation controls the weight of feature-correlation whitening. TTA already smooths the per-detection embedding, so the production AQE k may be over-smoothing. Sweep `aqe_k∈{2, 3, 4}` and `fic_reg∈{0.30, 0.50, 0.70}` at the C3 fusion point.

The two corrections are independent levers (fusion vs query-side processing); if either contributes meaningful lift, combining the best of each may push past 0.7720.

## Implementation Plan

Single Kaggle CPU kernel: `yahiaakhalafallah/14e-tta-fusion-aqe-fic-sweep`. Reuse the 14d kernel structure exactly; the kernel must NOT re-run Stage 0/1/2. Inputs:

- `kernel_sources`: `yahiaakhalafallah/14c-tta-stage2` (Stage 2 features) + the same 10b FAISS dataset 14d used.
- `dataset_sources`: same as 14d.
- GPU: **disabled** (`enable_gpu: false`); enable_internet still true if needed for Kaggle CLI fallbacks.

Sweep matrix (run sequentially, ~30-60s each on CPU):

### Block A — fine `w_tertiary × sim_thresh` grid (12 configs)

| Label | `w_tertiary` | `similarity_threshold` |
|:-----:|:------------:|:----------------------:|
| A1 | 0.45 | 0.48 |
| A2 | 0.45 | 0.50 |
| A3 | 0.45 | 0.52 |
| A4 | 0.475 | 0.48 |
| A5 | 0.475 | 0.50 |
| A6 | 0.475 | 0.52 |
| A7 | 0.50 | 0.48 |
| A8 | 0.50 | 0.50 (replicate C3) |
| A9 | 0.50 | 0.52 |
| A10 | 0.525 | 0.48 |
| A11 | 0.525 | 0.50 |
| A12 | 0.525 | 0.52 |

A8 is the C3 replicate and must reproduce 0.77155 ±0.001. If it does not, halt before trusting Block B.

### Block B — AQE/FIC sweep at the best Block A point (4 configs)

After Block A finishes, identify the best `(w_t, thr)` point. Run 4 configs at that point:

| Label | `aqe_k` | `fic_regularisation` |
|:-----:|:-------:|:--------------------:|
| B1 | 2 | 0.50 |
| B2 | 4 | 0.50 |
| B3 | 3 | 0.30 |
| B4 | 3 | 0.70 |

(Block A already covers `aqe_k=3, fic_reg=0.50` at the chosen fusion point, so no need to repeat.)

If the Block A best is itself worse than the C3 control (0.77155), skip Block B, write a brief diagnosis, and declare the TTA family closed.

All other parameters held at production values: `pca_components=384`, `algorithm=conflict_free_cc`, gallery expansion enabled (`thresh=0.48`, `orphan=0.38`), temporal overlap bonus enabled (`bonus=0.05`), intra-merge enabled (`thresh=0.80`, `gap=30`), `mtmc_only_submission=false`.

## Stop Criteria

1. **WIN**: any config ≥ **0.7720**. Production fusion config moves to TTA features at the winning `(w_t, thr, aqe_k, fic_reg)`. Replicate on a second seed before declaring as new headline.
2. **MARGINAL**: best config in **0.7715–0.7720**. Mark TTA family closed; write `docs/subagent-specs/post-14e-next.md` for the GNN edge classifier escalation. Note 14e best as the new reproducible best but do NOT promote to headline production.
3. **NEUTRAL**: best config in **0.7705–0.7715**. Same closure as MARGINAL. Note that TTA features are confirmed unchanged-from-original-information-content-but-different-distribution.
4. **DEAD**: best config < **0.7705**. Same closure plus record that even fine-grained re-tuning cannot recover the +0.05pp 14c lift, suggesting most of the 14d C3 win was noise.
5. **WALLTIME**: cap at **30 min on Kaggle CPU**. If incomplete, prioritize Block A; defer Block B.

## Expected Impact Range

- **Optimistic**: +0.15 to +0.30pp vs 14d C3 → **0.7730–0.7745** (clears WIN by 0.1pp).
- **Central**: +0.00 to +0.10pp → **0.77155–0.77255** (most likely lands MARGINAL).
- **Pessimistic**: -0.10 to +0.00pp → no improvement; TTA family closed.

## Estimated Walltime

- Block A: 12 configs × ~30-45s CPU = ~6-9 min.
- Block B: 4 configs × ~30-45s CPU = ~2-3 min.
- Total: **~10-15 min**, well under the 30-min cap. Zero GPU slot consumption.

## Coder Handoff Checklist

1. Build `notebooks/kaggle/14e_tta_fusion_aqe_fic_sweep/` from a copy of the 14d kernel. Strip the `enable_gpu: true` flag in `kernel-metadata.json` to `false` (this is CPU only). Wire `kernel_sources` to depend on `yahiaakhalafallah/14c-tta-stage2` (Stage 2 features) and the existing 10b FAISS-index dataset 14d used. **Do NOT add 14d as a kernel source** — 14e re-runs Stages 3-5 from the same 14c features that 14d used.
2. Implement Block A as the first sweep loop; verify A8 replicates 14d C3's 0.77155 ±0.001 before proceeding to Block B. If A8 does not replicate, halt the sweep and write a diagnosis cell.
3. After Block A, programmatically pick the `(w_t, thr)` with highest MTMC IDF1, then run Block B at that point.
4. Push **once**, watch for `not valid dataset sources` warnings, and cancel + refix if seen.
5. Append the result tables (Block A, Block B) to `docs/findings.md` under a new "14e Tighter TTA Fusion + AQE/FIC Sweep" subsection and to `docs/experiment-log.md` as section 2.12.
6. On WIN: replicate on a second seed (re-run the winning config) and only then update production / `.github/copilot-instructions.md` headline.
7. On MARGINAL/NEUTRAL/DEAD: write `docs/subagent-specs/post-14e-next.md` proposing the GNN edge classifier (input: tracklet-pair similarities + camera-ID + temporal features; output: edge keep/drop probability; training: GT-derived edges from CityFlowV2 train split; integration: replaces/augments the conflict_free_cc edge weighting in Stage 4).