# Post-14e Next Experiment Spec — Confirm AQE k=2 Win + k=1 Probes (`14f`)

**Date**: 2026-05-07
**Author**: MTMC Planner
**Status**: PROPOSED — ready for Coder implementation

## Decision

14e produced a clear **WIN**: B1 (`aqe_k=2, fic_reg=0.5, w_t=0.525, thr=0.48`) reached **MTMC IDF1 = 0.77936** on 14c v2 TTA features, **+0.91pp vs production 0.7703** and **+0.78pp vs the 14d C3 floor of 0.77155**. ID switches dropped 213 → 154 (-28%) and trackeval IDF1 rose from 0.7866 (production) to 0.7946 (+0.80pp). Block B's AQE axis is monotonic (k=4 → 0.77052, k=3 → 0.77171, k=2 → **0.77936**), so the natural next move is to (a) confirm B1 on a tighter local grid and (b) probe whether `k=1` continues the trend or breaks AQE entirely.

## Hypothesis

1. **AQE k=2 confirmation**: B1 was a single point. The tight sweep `(k=2) × (FIC ∈ {0.3, 0.4, 0.5, 0.6, 0.7}) × (thr ∈ {0.46, 0.48, 0.50}) × (w_t ∈ {0.50, 0.525, 0.55})` = **45 configs** maps the local optimum and reveals whether B1's ID-switch reduction is robust or a fluke at exactly that point. Including the B1-replicate cell (k=2, FIC=0.5, thr=0.48, w_t=0.525) provides the drift check.
2. **AQE k=1 probes**: TTA already L2-mean-pools 4 primary views per detection, so the query embedding is itself a small ensemble. Reducing AQE k from 3 → 2 cut ID switches sharply; k=1 (no neighbour expansion at all, just FIC+similarity) is the logical extreme. Two outcomes are possible: either the trend continues and k=1 wins by another +0.1–0.5pp (TTA is sufficient smoothing on its own), or AQE collapses entirely because some neighbour expansion is still needed for hard cross-camera matches. Either result is a clean signal.

## Implementation Plan

Single Kaggle CPU kernel: `yahiaakhalafallah/14f-tta-aqe-confirmation-sweep`. Inputs:

- `kernel_sources`: `yahiaakhalafallah/14c-tta-stage2` (Stage 2 TTA features, v2) + the same 10b FAISS-index dataset 14d/14e used.
- `dataset_sources`: same as 14d/14e.
- GPU: **disabled** (`enable_gpu: false`).
- Reuse the 14e kernel structure exactly; do NOT add 14e as a kernel source — 14f re-runs Stages 3–5 from the same 14c v2 features.

### Sweep matrix — 54 configs

#### Block A — `aqe_k=2` confirmation grid (45 configs)

For each `w_t ∈ {0.50, 0.525, 0.55}` (outer), `fic_reg ∈ {0.3, 0.4, 0.5, 0.6, 0.7}` (middle), `thr ∈ {0.46, 0.48, 0.50}` (inner). Labels A1..A45 with `aqe_k=2` fixed for the entire block. The cell `(w_t=0.525, fic_reg=0.5, thr=0.48)` is the B1-replicate (label **A20**).

#### Block B — `aqe_k=1` probes (9 configs)

For each `thr ∈ {0.46, 0.48, 0.50}` × `w_t ∈ {0.50, 0.525, 0.55}` at `fic_reg=0.5` and `aqe_k=1`. Labels B1..B9.

### Configs as JSON list

```json
[
  {"label":"A1","aqe_k":2,"fic_regularisation":0.3,"similarity_threshold":0.46,"w_tertiary":0.50},
  {"label":"A2","aqe_k":2,"fic_regularisation":0.3,"similarity_threshold":0.48,"w_tertiary":0.50},
  {"label":"A3","aqe_k":2,"fic_regularisation":0.3,"similarity_threshold":0.50,"w_tertiary":0.50},
  {"label":"A4","aqe_k":2,"fic_regularisation":0.4,"similarity_threshold":0.46,"w_tertiary":0.50},
  {"label":"A5","aqe_k":2,"fic_regularisation":0.4,"similarity_threshold":0.48,"w_tertiary":0.50},
  {"label":"A6","aqe_k":2,"fic_regularisation":0.4,"similarity_threshold":0.50,"w_tertiary":0.50},
  {"label":"A7","aqe_k":2,"fic_regularisation":0.5,"similarity_threshold":0.46,"w_tertiary":0.50},
  {"label":"A8","aqe_k":2,"fic_regularisation":0.5,"similarity_threshold":0.48,"w_tertiary":0.50},
  {"label":"A9","aqe_k":2,"fic_regularisation":0.5,"similarity_threshold":0.50,"w_tertiary":0.50},
  {"label":"A10","aqe_k":2,"fic_regularisation":0.6,"similarity_threshold":0.46,"w_tertiary":0.50},
  {"label":"A11","aqe_k":2,"fic_regularisation":0.6,"similarity_threshold":0.48,"w_tertiary":0.50},
  {"label":"A12","aqe_k":2,"fic_regularisation":0.6,"similarity_threshold":0.50,"w_tertiary":0.50},
  {"label":"A13","aqe_k":2,"fic_regularisation":0.7,"similarity_threshold":0.46,"w_tertiary":0.50},
  {"label":"A14","aqe_k":2,"fic_regularisation":0.7,"similarity_threshold":0.48,"w_tertiary":0.50},
  {"label":"A15","aqe_k":2,"fic_regularisation":0.7,"similarity_threshold":0.50,"w_tertiary":0.50},
  {"label":"A16","aqe_k":2,"fic_regularisation":0.3,"similarity_threshold":0.46,"w_tertiary":0.525},
  {"label":"A17","aqe_k":2,"fic_regularisation":0.3,"similarity_threshold":0.48,"w_tertiary":0.525},
  {"label":"A18","aqe_k":2,"fic_regularisation":0.3,"similarity_threshold":0.50,"w_tertiary":0.525},
  {"label":"A19","aqe_k":2,"fic_regularisation":0.4,"similarity_threshold":0.46,"w_tertiary":0.525},
  {"label":"A20","aqe_k":2,"fic_regularisation":0.5,"similarity_threshold":0.48,"w_tertiary":0.525,"notes":"B1 REPLICATE — drift check, must reproduce 0.77936 ±0.001"},
  {"label":"A21","aqe_k":2,"fic_regularisation":0.4,"similarity_threshold":0.48,"w_tertiary":0.525},
  {"label":"A22","aqe_k":2,"fic_regularisation":0.4,"similarity_threshold":0.50,"w_tertiary":0.525},
  {"label":"A23","aqe_k":2,"fic_regularisation":0.5,"similarity_threshold":0.46,"w_tertiary":0.525},
  {"label":"A24","aqe_k":2,"fic_regularisation":0.5,"similarity_threshold":0.50,"w_tertiary":0.525},
  {"label":"A25","aqe_k":2,"fic_regularisation":0.6,"similarity_threshold":0.46,"w_tertiary":0.525},
  {"label":"A26","aqe_k":2,"fic_regularisation":0.6,"similarity_threshold":0.48,"w_tertiary":0.525},
  {"label":"A27","aqe_k":2,"fic_regularisation":0.6,"similarity_threshold":0.50,"w_tertiary":0.525},
  {"label":"A28","aqe_k":2,"fic_regularisation":0.7,"similarity_threshold":0.46,"w_tertiary":0.525},
  {"label":"A29","aqe_k":2,"fic_regularisation":0.7,"similarity_threshold":0.48,"w_tertiary":0.525},
  {"label":"A30","aqe_k":2,"fic_regularisation":0.7,"similarity_threshold":0.50,"w_tertiary":0.525},
  {"label":"A31","aqe_k":2,"fic_regularisation":0.3,"similarity_threshold":0.46,"w_tertiary":0.55},
  {"label":"A32","aqe_k":2,"fic_regularisation":0.3,"similarity_threshold":0.48,"w_tertiary":0.55},
  {"label":"A33","aqe_k":2,"fic_regularisation":0.3,"similarity_threshold":0.50,"w_tertiary":0.55},
  {"label":"A34","aqe_k":2,"fic_regularisation":0.4,"similarity_threshold":0.46,"w_tertiary":0.55},
  {"label":"A35","aqe_k":2,"fic_regularisation":0.4,"similarity_threshold":0.48,"w_tertiary":0.55},
  {"label":"A36","aqe_k":2,"fic_regularisation":0.4,"similarity_threshold":0.50,"w_tertiary":0.55},
  {"label":"A37","aqe_k":2,"fic_regularisation":0.5,"similarity_threshold":0.46,"w_tertiary":0.55},
  {"label":"A38","aqe_k":2,"fic_regularisation":0.5,"similarity_threshold":0.48,"w_tertiary":0.55},
  {"label":"A39","aqe_k":2,"fic_regularisation":0.5,"similarity_threshold":0.50,"w_tertiary":0.55},
  {"label":"A40","aqe_k":2,"fic_regularisation":0.6,"similarity_threshold":0.46,"w_tertiary":0.55},
  {"label":"A41","aqe_k":2,"fic_regularisation":0.6,"similarity_threshold":0.48,"w_tertiary":0.55},
  {"label":"A42","aqe_k":2,"fic_regularisation":0.6,"similarity_threshold":0.50,"w_tertiary":0.55},
  {"label":"A43","aqe_k":2,"fic_regularisation":0.7,"similarity_threshold":0.46,"w_tertiary":0.55},
  {"label":"A44","aqe_k":2,"fic_regularisation":0.7,"similarity_threshold":0.48,"w_tertiary":0.55},
  {"label":"A45","aqe_k":2,"fic_regularisation":0.7,"similarity_threshold":0.50,"w_tertiary":0.55},
  {"label":"B1","aqe_k":1,"fic_regularisation":0.5,"similarity_threshold":0.46,"w_tertiary":0.50},
  {"label":"B2","aqe_k":1,"fic_regularisation":0.5,"similarity_threshold":0.48,"w_tertiary":0.50},
  {"label":"B3","aqe_k":1,"fic_regularisation":0.5,"similarity_threshold":0.50,"w_tertiary":0.50},
  {"label":"B4","aqe_k":1,"fic_regularisation":0.5,"similarity_threshold":0.46,"w_tertiary":0.525},
  {"label":"B5","aqe_k":1,"fic_regularisation":0.5,"similarity_threshold":0.48,"w_tertiary":0.525},
  {"label":"B6","aqe_k":1,"fic_regularisation":0.5,"similarity_threshold":0.50,"w_tertiary":0.525},
  {"label":"B7","aqe_k":1,"fic_regularisation":0.5,"similarity_threshold":0.46,"w_tertiary":0.55},
  {"label":"B8","aqe_k":1,"fic_regularisation":0.5,"similarity_threshold":0.48,"w_tertiary":0.55},
  {"label":"B9","aqe_k":1,"fic_regularisation":0.5,"similarity_threshold":0.50,"w_tertiary":0.55}
]
```

All other parameters held at production values: `pca_components=384`, `algorithm=conflict_free_cc`, gallery expansion enabled (`thresh=0.48`, `orphan=0.38`), temporal overlap bonus enabled (`bonus=0.05`), intra-merge enabled (`thresh=0.80`, `gap=30`), `mtmc_only_submission=false`, `w_secondary=0.0`, `w_primary = 1 - w_tertiary`.

Implement Block A first; the kernel must run **A20 first or check it explicitly** to perform the drift check before trusting the rest of the sweep. If A20 does not reproduce 0.77936 ±0.001, **mark the entire sweep as drift-affected** and write a diagnosis cell instead of promoting any config.

## Stop Criteria (relative to 14e B1 = 0.77936)

| Verdict | Best Block A or B IDF1 | Action |
|:-------:|:----------------------:|:------:|
| **WIN** | ≥ **0.7810** (+0.16pp) | Promote new config to headline; update findings, experiment-log, copilot-instructions; replicate on a second seed before declaring final |
| **MARGINAL** | 0.7795–0.7810 | Add to docs as sub-optimum; keep B1 (0.77936) as headline; consider averaging best 2–3 configs to reduce noise |
| **NEUTRAL** | 0.7785–0.7795 | Keep B1 headline; declare AQE/FIC axis confirmed; next step is feature-side (GNN edge classifier or 5-view TTA) |
| **DEAD** | < 0.7785 | B1 was an outlier — keep 0.7703 production headline only if A20 also fails to replicate; otherwise treat B1 as borderline-noise win and commission a 3-seed replication of B1 alone before promoting |
| **DRIFT** | A20 not within 0.001 of 0.77936 | Halt sweep; do not promote any config; investigate kernel drift before re-running |

## Expected Walltime

Per-config Stage-3+4+5 on Kaggle CPU is ~30–40 s (per 14e timings: 0.55 min/config). 54 configs × ~35 s = **~30 min**, slightly over the 30-min cap. Run Block A (45 configs) first to guarantee the drift check + main grid land within the cap; Block B (9 k=1 probes) is appended last so it gracefully truncates if walltime is hit.

- Estimated walltime: **~25–35 min CPU**
- Zero GPU slot consumption.

## Coder Handoff Checklist

1. Build `notebooks/kaggle/14f_tta_aqe_confirmation_sweep/` from a copy of the 14e kernel. Set `enable_gpu: false`. Wire `kernel_sources` to `yahiaakhalafallah/14c-tta-stage2` (v2) plus the same 10b FAISS dataset.
2. Run **A20 first** (or always include it and check immediately after); if `|IDF1 - 0.77936| > 0.001`, halt and write a drift-diagnosis cell. Do not promote any config.
3. After Block A completes (and A20 passed), run Block B (k=1 probes).
4. Push **once**; watch for `not valid dataset sources` warnings; cancel + refix if seen.
5. On WIN (≥0.7810): re-run the winning config on a second seed before promoting to headline. Update `docs/findings.md`, `docs/experiment-log.md` § 2.13, and `.github/copilot-instructions.md` with the new headline.
6. On MARGINAL/NEUTRAL: keep B1 (0.77936) as headline; write `docs/subagent-specs/post-14f-next.md` proposing either (a) 5-view TTA at Stage 2 (extra rotations) or (b) GNN edge classifier on the new TTA+AQE-2 baseline.
7. On DEAD/DRIFT: write the diagnosis and pause the TTA/AQE thread.
