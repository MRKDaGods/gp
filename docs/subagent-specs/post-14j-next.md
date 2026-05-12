# Post-14j Next Experiment Spec — Extended R50-IBN Quaternary Sweep (`14k`)

**Date**: 2026-05-08
**Author**: MTMC Planner
**Status**: PROPOSED — ready for Coder implementation

## Decision

Run **14k**: extend the 14j v1 4-way score-fusion sweep into the upper `w_quaternary` range that 14j did not cover. 14j peaked at **MTMC IDF1 = 0.78032** (W14, `w_q=0.30, thr=0.48`) sitting on the upper boundary of the swept grid, classed **MARGINAL** in 14j's verdict bands (0.7795–0.7810). The lift is +0.00097 over the 14e B1 plateau (0.77936) but +0.07pp short of the 0.7810 WIN threshold. The 14j evidence supports one more cheap CPU-only sweep before declaring the 4-way fusion family closed.

### Why extend the sweep

1. **Boundary effect**: W14 is at the maximum `w_q` tested. Three of the five swept `w_q` values are below the optimum; the upper half of the curve is unsampled.
2. **Monotonic at thr=0.46**: 0.77713 → 0.77742 → 0.77828 → 0.77856 → 0.77917 across `w_q ∈ {0.05, 0.10, 0.15, 0.20, 0.30}`. No turnover seen.
3. **Two-axis confirmation at thr=0.48**: after the regime change at `w_q=0.10` (where `id_switches` jumps from 154 to 200), IDF1 climbs cleanly from 0.77727 → 0.77796 → 0.77833 → 0.78032. The dip at `w_q=0.10` is a regime change, not noise — `id_switches` saturates at 206–207 from there onward, and within that regime the trend is monotonic.
4. **Cost**: one CPU-only kernel, no GPU slot, ~10–15 min walltime. Negligible.
5. **Decision value**: a clean WIN at `w_q ∈ {0.35, 0.40, 0.45}` would close out the 0.78936 plateau; a NEUTRAL/DEAD result combined with K13 closes the 4-way family with high confidence and routes directly to 14k EVA-02 fine-tune (the previously-deferred GPU-heavy lever).

### Counter-argument and mitigation (K13 sanity probe)

At `w_q=0.30`, `w_primary=0.175`. At `w_q=0.50`, `w_primary=0.025` — primary CLIP TransReID is essentially zero'd out. If the lift continues into that range, the apparent improvement may be **rebalancing of expert weights** (R50-IBN + DINOv2 alone happens to outperform CLIP + DINOv2 on this anchor) rather than additive 4-way ensemble diversity. To distinguish:

- **K13 (sanity)**: `w_primary=0.30, w_tertiary=0.30, w_quaternary=0.40`. Three streams roughly balanced, primary not suppressed below `w_t`. If K13 lifts to ≥ peak of K1–K12, the lift is real ensemble effect. If K13 underperforms K1–K12 strongly, the lift is primary-suppression and should not be promoted even if it crosses the WIN threshold.

## Hypothesis

The 14j signal is genuine 4-way fusion benefit at the harder 14e B1 anchor; the 0.78032 peak is on the rising side of a curve that turns over somewhere in `w_q ∈ [0.30, 0.50]`. Extending the sweep should locate the turnover and either (a) push above 0.7810 WIN, (b) plateau in the MARGINAL band 0.7795–0.7810, or (c) regress monotonically (closing the family).

Expected outcome band: **−0.20pp to +0.30pp** vs 14j W14 0.78032. Most likely MARGINAL plateau or modest lift to ~0.781.

## Implementation Plan

Single CPU-only kernel: `notebooks/kaggle/14k_r50ibn_fusion_extended/`. Must reuse the **14j v1 fusion-sweep notebook structure** (same AST + name-simulation harness, same per-config config-dir layout, same Stage 3–5 chaining, same summary-JSON schema).

### Sweep grid (13 configs)

Anchor (fixed across K0–K13): tertiary fixed at `w_tertiary=0.525`, `aqe_k=2`, `fic_regularisation=0.5`, source 14h v3 Stage-2 outputs, R50-IBN secondary stream from 14j features.

| Label | Block | `w_quaternary` | `similarity_threshold` | `w_primary` | `w_tertiary` | Notes |
|:-----:|:------|:--------------:|:----------------------:|:-----------:|:------------:|:------|
| **K0** | drift | **0.00** | **0.48** | **0.475** | **0.525** | drift gate; must reproduce 0.77936 / id_switches=154 |
| K1 | quaternary_grid | 0.35 | 0.46 | 0.150 | 0.500 | rescaled `w_p:w_t = 0.475:0.525` to sum (1−w_q) |
| K2 | quaternary_grid | 0.35 | 0.48 | 0.150 | 0.500 | |
| K3 | quaternary_grid | 0.35 | 0.50 | 0.150 | 0.500 | |
| K4 | quaternary_grid | 0.40 | 0.46 | 0.125 | 0.475 | |
| K5 | quaternary_grid | 0.40 | 0.48 | 0.125 | 0.475 | |
| K6 | quaternary_grid | 0.40 | 0.50 | 0.125 | 0.475 | |
| K7 | quaternary_grid | 0.45 | 0.46 | 0.100 | 0.450 | |
| K8 | quaternary_grid | 0.45 | 0.48 | 0.100 | 0.450 | |
| K9 | quaternary_grid | 0.45 | 0.50 | 0.100 | 0.450 | |
| K10 | quaternary_grid | 0.50 | 0.46 | 0.075 | 0.425 | primary nearly zero — diagnostic |
| K11 | quaternary_grid | 0.50 | 0.48 | 0.075 | 0.425 | |
| K12 | quaternary_grid | 0.50 | 0.50 | 0.075 | 0.425 | |
| **K13** | **sanity** | **0.40** | **0.48** | **0.30** | **0.30** | three-stream rebalance probe; **does NOT preserve `w_p:w_t = 0.475:0.525` ratio**; tests primary-suppression hypothesis |

Note on weight rescaling for K1–K12: the convention from 14j W1–W15 is preserved — at each `w_q`, the residual `(1 − w_q)` is split between primary and tertiary in the **0.475 : 0.525** ratio (i.e., `w_p = 0.475 × (1−w_q)`, `w_t = 0.525 × (1−w_q)`). K13 is the only deliberate departure from this convention.

### Stages 3–5 anchor (fixed across K0–K13)
- `stage4.association.aqe.k=2`
- `stage4.association.fic.regularisation=0.5`
- `stage4.association.gallery_expansion.threshold=0.48`
- `stage4.association.gallery_expansion.orphan_threshold=0.38`
- `stage4.association.intra_merge.threshold=0.80`
- `stage4.association.intra_merge.gap=30`
- `stage4.association.temporal_overlap_bonus=0.05`
- `stage4.association.algorithm=conflict_free_cc`
- `stage4.association.pca.n_components=384`
- `stage5.mtmc_only_submission=false`
- `stage4.association.graph.similarity_threshold` is the only Stage-4 knob varied (0.46 / 0.48 / 0.50 per config).

### Drift gate (K0)

K0 (`w_q=0.0`) MUST reproduce **0.77936** with `id_switches=154 EXACT`, tolerance ±0.001. Same gate as 14e B1 / 14f / 14g / 14h / 14i F0 / 14j W0. If K0 deviates, halt and diagnose — the rescaling or the secondary feature loading is wrong.

### Pre-flight check (Coder must verify before push)

At K10–K12 (`w_q=0.50`), `w_primary=0.075` and `w_tertiary=0.425` — the primary CLIP TransReID stream contributes only **7.5%** of the fused similarity score. This is a deliberate diagnostic regime, not a deployment recommendation. Do not interpret a high IDF1 at K10–K12 alone as a WIN. The K13 sanity probe is the gate that distinguishes "real 4-way ensemble lift" from "primary suppression rebalances toward a stronger expert pair." Coder must NOT skip K13.

### Outputs

Persist `outputs/14k_v1_summary/14k_summary.json` mirroring the 14j summary schema (per-config: `mtmc_idf1`, `id_switches`, `mota`, `trackeval_idf1`, fusion weights, plus `best`/`overall_best` and the full sweep grid).

### Kaggle metadata
- **kernel slug**: `yahiaakhalafallah/14k-r50ibn-fusion-extended`
- **kernel_sources**: `yahiaakhalafallah/14h-robust-tracklet-pooling` + `yahiaakhalafallah/14j-r50-ibn-features`
- **enable_gpu**: false (CPU-only)
- **enable_internet**: as required for pip installs only
- Push once per `.github/copilot-instructions.md` Kaggle Push Safety Rules.

## Stop Criteria (relative to 14e B1 = 0.77936)

| Verdict | Best of K1..K13 MTMC IDF1 | Action |
|:-------:|:-------------------------:|:------:|
| **WIN** | ≥ **0.7810** (+0.16pp) AND K13 ≥ best of K1..K12 − 0.0015 | Promote: re-run the winning `(w_q, thr)` on a second seed, sweep `w_q` neighborhood at ±0.025 resolution, then commit headline change. |
| **PRIMARY-SUPPRESSION (degenerate WIN)** | best of K1..K12 ≥ 0.7810 AND K13 < best − 0.0015 | Do NOT promote. The lift is rebalancing, not ensemble. Document and pivot to 14k EVA-02. |
| **MARGINAL** | 0.7795–0.7810 | Keep 0.77936 headline; document the best fusion point as a sub-optimum; close the 4-way family. |
| **NEUTRAL** | 0.7785–0.7795 | 4-way fusion family closed. Pivot to 14k EVA-02 ViT-Large fine-tune from scratch on CityFlowV2. |
| **DEAD** | < 0.7785 | Family closed; pivot directly to 14k EVA-02. |
| **DRIFT_FAIL** | K0 ∉ [0.77836, 0.78036] | Halt sweep; investigate fusion plumbing. Do not promote any config. |

## Expected Walltime
- CPU sweep: **~10–15 min CPU**, no GPU. 13 configs at ~30–60s each plus per-config Stage-4/5.

## Coder Handoff Checklist

1. Build the kernel from a copy of the 14j v1 fusion-sweep notebook (same AST + name-simulation harness, same Stage-3/4/5 chaining, same summary-JSON schema). Do NOT change the harness — only the sweep grid.
2. Update kernel-metadata.json: `id=yahiaakhalafallah/14k-r50ibn-fusion-extended`, title, code_file, `enable_gpu=false`, kernel_sources as specified above.
3. Implement the K0–K13 sweep grid exactly as specified. Verify K13 weights `(0.30, 0.30, 0.40)` are NOT rescaled to the `0.475:0.525` ratio.
4. Run K0 FIRST. On drift failure, write a diagnosis cell and DO NOT push the rest of the sweep.
5. Push the kernel once. Watch for `not valid dataset sources` warnings; cancel + refix if seen.
6. After the run, persist `outputs/14k_v1_summary/14k_summary.json`.
7. On WIN (with K13 confirmation): run second-seed re-validation and a tighter local sweep before updating findings/experiment-log/copilot-instructions.
8. On MARGINAL/NEUTRAL/DEAD/PRIMARY-SUPPRESSION: keep 0.77936 headline. Write `docs/subagent-specs/post-14k-next.md` proposing 14L EVA-02 ViT-Large fine-tune.
9. On DRIFT_FAIL: write the diagnosis. Halt the fusion thread.

## On-deck if 14k NEUTRAL/DEAD/PRIMARY-SUPPRESSION

**14L (formerly 14k EVA-02)**: EVA-02 ViT-Large fine-tuned on CityFlowV2 as a 3rd or replacement feature stream. ~6–8 hr GPU fine-tune + ~1.5–2 hr GPU Stage-2 + CPU 4-way sweep, multi-day. See `docs/subagent-specs/post-14i-next.md` "On-deck" section for the existing pre-spec.
