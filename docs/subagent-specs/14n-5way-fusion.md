# 14n — 5-Way Score-Fusion Sweep (CLIP + DINOv2 + R50-IBN + OSNet-IBN)

**Date**: 2026-05-08
**Parent specs**: docs/subagent-specs/post-14k-next.md (Candidate 1 fusion arm) and docs/subagent-specs/14m-osnet-ibn-train.md
**Status**: READY FOR IMPLEMENTATION (gated on 14m-extract success)
**Goal**: Confirm or refute that the OSNet-IBN-x1.0 quinary stream is architecturally diverse enough to break the 0.78079 4-way plateau and clear the 0.7920 WIN bar.

## Pre-condition
- 14m-extract has emitted a valid `embeddings_quinary.npy` with shape `(929, 512)`.
- The dropped-indices set is exactly `[280, 286, 481]`.
- `embedding_index.json` compare-equals the 14h v3 index.

## Kernel Identity
- **Slug**: `<active-account>/14n-5way-fusion-sweep`
- `title`: `14n 5-Way Fusion Sweep`
- `code_file`: `14n_5way_fusion_sweep.ipynb`
- `language`: `python`
- `kernel_type`: `notebook`
- `is_private`: `true`
- `enable_gpu`: `false` (CPU-only Stages 3/4/5)
- `enable_internet`: `true`
- `enable_tpu`: `false`
- `dataset_sources`: `[
  "thanhnguyenle/data-aicity-2023-track-2"
]` (ground truth)
- `kernel_sources`:
  - `yahiaakhalafallah/14h-robust-tracklet-pooling` (Stage-1 plus primary and tertiary Stage-2)
  - `yahiaakhalafallah/14j-r50-ibn-features` (quaternary stream)
  - `<active-account>/14m-extract-osnet-features` (quinary stream)
- `competition_sources`: `[]`
- `model_sources`: `[]`

## Workspace Files
- Notebook: [notebooks/kaggle/14n_5way_fusion_sweep/14n_5way_fusion_sweep.ipynb](notebooks/kaggle/14n_5way_fusion_sweep/14n_5way_fusion_sweep.ipynb)
- Metadata: [notebooks/kaggle/14n_5way_fusion_sweep/kernel-metadata.json](notebooks/kaggle/14n_5way_fusion_sweep/kernel-metadata.json)
- Builder script: `_build_14n_5way_fusion_sweep.py` modeled on `_build_14j_4way_fusion_sweep.py`

## Score-Fusion Math
Extend 14j 4-way score fusion to 5-way additive cosine fusion:

```text
sim_total = w_p*sim_primary + w_t*sim_tertiary + w_q*sim_quaternary + w_o*sim_quinary
subject to: w_p + w_t + w_q + w_o = 1.0 AND w_p, w_t, w_q, w_o >= 0
```

- `w_p` is derived as `w_p = 1.0 - w_t - w_q - w_o`; there must be no separate primary-weight knob.
- The notebook must assert `abs(w_p + w_t + w_q + w_o - 1.0) < 1e-6` for every config and `w_p >= 0`, raising on negative weights.
- Quinary stream is plumbed through the existing tertiary path or through a new quinary override pathway.
- Inspect Stage 4 association code in `src.stage4_association` before implementation to confirm there is room for a fourth additive cosine term.
- If current Stage 4 code only supports primary + secondary + tertiary, the implementation MUST include a required code change in `src/stage4_association/` extending additive fusion to a fourth `quinary_embeddings` field with `quinary_embeddings.weight`.
- The override keys exposed by that change must be `stage4.association.quinary_embeddings.path` and `stage4.association.quinary_embeddings.weight`.
- The builder script must verify the override path resolves cleanly via OmegaConf load before push.

## Anchor Configs (drift gates — MUST reproduce EXACT)
- **N0 drift gate** (14e B1 baseline): `w_t=0.525`, `w_q=0.0`, `w_o=0.0`, `w_p=0.475`, `sim_thresh=0.48`, AQE `k=2`, FIC `reg=0.5`. Quaternary path empty and quinary path empty. Must reproduce MTMC IDF1 `0.77936` and `id_switches=154` EXACT, with IDF1 tolerance +/-0.001 and equality on id_switches. If drift occurs, halt and do NOT run the sweep.
- **N1 sanity gate** (14k K7 4-way plateau): `w_p=0.10`, `w_t=0.45`, `w_q=0.45`, `w_o=0.0`, `sim_thresh=0.46`, AQE `k=2`, FIC `reg=0.5`. Quinary path empty. Must reproduce MTMC IDF1 `0.78079` and `id_switches=213` EXACT, with IDF1 tolerance +/-0.001 and equality on id_switches. This validates that the 4-way to 5-way refactor did not break existing fusion arithmetic.

## Sweep Design
Target **18 active configs** plus 2 drift gates, for **20 total**. The Coder must produce exactly the grid the implementation spec settles on; do not silently shrink. The authoritative budget is 18-20 configs total including N0 and N1 drift gates.

### Anchor A — 14e B1 base
- Base: `w_t=0.525`, `w_q=0.0`.
- Vary `w_o in {0.10, 0.15, 0.20, 0.25, 0.30, 0.35}` and `thr in {0.46, 0.48, 0.50}`.
- Constraint: `w_p = 0.475 - w_o`; reject if `w_p < 0`.
- Drop `w_o=0.35, thr=0.50` if it produces `w_p < 0`.
- Nominal grid is 6 x 3 = 18 configs; the implementation should use the pre-registered subset needed to keep total active configs in budget.

### Anchor B — K7 plateau base
- Literal K7 base `w_t=0.45`, `w_q=0.45` with `w_o in {0.10, 0.15, 0.20}` only leaves `w_o=0.10` valid at `w_p=0.0`, giving 1 x 2 active configs for `thr in {0.46, 0.48}`.
- Preferred realization: rebalance to `w_t=0.40`, `w_q=0.40`, and `w_o in {0.10, 0.15, 0.20}` so `w_p=0.10, 0.05, 0.00` respectively, with `thr in {0.46, 0.48}`. This gives 3 x 2 = 6 configs.

### Anchor C — balanced 4+1
- Base: `w_t=0.30`, `w_q=0.30`.
- Vary `w_o in {0.10, 0.15, 0.20, 0.25, 0.30}` and `thr in {0.46, 0.48, 0.50}`.
- Constraint: `w_p = 0.40 - w_o >= 0.10` for all listed `w_o <= 0.30`.
- Nominal grid is 5 x 3 = 15 configs; the implementation should use the pre-registered subset needed to keep total active configs in budget.

All configs share the locked anchor: AQE `k=2`, FIC `reg=0.5`, `conflict_free_cc`, gallery expansion enabled at 0.48 with orphan 0.38, intra-merge enabled at 0.80 / 30, `mtmc_only=false`, stationary filter on, GT-clip + zone-filter on, track smoothing OFF, edge trim OFF, AFLink OFF, CSLS OFF, rerank OFF, FAC OFF, CID_BIAS OFF, and hierarchical OFF.

## Verdict Bands (pre-registered)
| Verdict | 14n MTMC IDF1 | Action |
|:-------:|:-------------:|:------|
| **WIN** | >= **0.7920** | 5-way ensemble breakthrough. Promote, deploy, update the headline in `.github/copilot-instructions.md` and `docs/findings.md`, then schedule a single confirmation re-run on a fresh CPU kernel before final promotion. |
| **MARGINAL** | >= **0.7820** and < **0.7920** | Matches or refines the 14k plateau. Log in `docs/experiment-log.md` and `docs/findings.md` as MARGINAL but do NOT promote. |
| **FAIL** | < **0.7820** | OSNet-IBN stream failed to add net diversity. Log dead-end, close architecture-diverse fusion branch, and escalate to GNN edge classifier per `docs/subagent-specs/post-14k-next.md`. |
| **DRIFT_FAIL** | N0 or N1 mismatch | Halt sweep without running remaining configs; treat as a bug, not an experiment outcome. |

## Output Contract
- `/kaggle/working/outputs/14n_5way_summary.json` with a full summary dict mirroring 14j v4: `run_name`, `experiment`, `kernel`, `verdict_band`, `drift_detected`, `drift_check_config_id`, `drift_check_result`, `configs`, `planned_configs`, `results`, `sweep_results`, `overall_best`, `best`, top-5 by IDF1, `halt_reason`, `fixed_config`, `sweep_grid`, `stop_criteria`, and `paths`.
- Add the literal-third sanity probe analogous to 14k K13: `w_p=0.30`, `w_t=0.30`, `w_q=0.30`, `w_o=0.10`, `thr=0.48`. This confirms any IDF1 lift is real ensemble signal rather than primary suppression. The literal-third config must produce IDF1 >=0.7795 to count as ensemble-valid.
- `/kaggle/working/14n_5way_summary.json` root-level alias.
- `/kaggle/working/outputs/14n_5way_recovery/<config_id>/` for each non-drift config, containing `config.yaml`, `stage4/global_trajectories.json`, `stage4/forensic_report.json`, `stage5/evaluation_report.json`, and `stage5/predictions_mot/`.
- Per-config rows include `config_id`, `block`, `w_primary`, `w_tertiary`, `w_quaternary`, `w_quinary`, `similarity_threshold`, `aqe_k`, `fic_regularisation`, `mtmc_idf1`, `idp`, `idr`, `id_switches`, `fragmentations`, `mota`, `trackeval_idf1`, `hota`, `conflations`, `num_pred_ids`, and `num_trajectories`.
- `block` must be one of `drift`, `anchor_a`, `anchor_b`, `anchor_c`, or `literal_third`.

## ID-switch ensemble-validity check
- At reporting time, classify the best config by `id_switches` relative to N0 (154) and N1 (213).
- A real lift requires `id_switches` between 154 and 213, or lower.
- A config with `id_switches > 220` AND `IDF1 > 0.79` is suspicious, likely conflations being miscounted as IDF1 wins, and must be flagged in the summary as `ensemble_validity="suspect"`.

## ETA
- ~5-10 min CPU only. Stage 3 is FAISS index build, Stage 4 is the only nontrivial cost; per-config runtime is ~30s on a 6-camera / 929-tracklet setup, so ~20 configs should fit around 10 min wall-clock.

## Pre-flight Checks (single-push policy)
1. **AST validation** on the on-disk notebook.
2. **Name simulation** of import-only cells.
3. **Static contracts**: literal strings `"embeddings_quinary.npy"`, `"w_p + w_t + w_q + w_o"`, `"0.77936"`, `"0.78079"`, `"0.7920"`, `"AQE k=2"` or canonical equivalent, `"154"`, `"213"`, `"WIN"`, `"MARGINAL"`, `"FAIL"`, and `"DRIFT_FAIL"` must be present in the notebook source.
4. **OmegaConf override sanity**:
   ```bash
   python -c "from src.core.config import load_config; load_config('configs/default.yaml', dataset_config='configs/datasets/cityflowv2.yaml', overrides=[<one full override list>])"
   ```
   This must succeed without raising; it catches any missing `quinary_embeddings` config key before the kernel push.
5. **Active account check + GPU slot check**: CPU kernel, but still respect the 2-concurrent-session limit. 14m-extract must be `complete` first.
6. **Push-once rule**: validate metadata, push once, poll `kaggle kernels status <slug>` to completion. Cancel-on-warning if any input source is reported invalid.

## Hard Constraints
- Do NOT enable AFLink; confirmed -3.8 to -13.2pp dead end.
- Do NOT enable CSLS; confirmed -34.7pp catastrophic.
- Do NOT enable reranking; it always hurts on current features.
- Do NOT use feature concatenation; fusion must be score-level only.
- Do NOT use 384px features.
- Do NOT set `mtmc_only=True`.
- Do NOT enable track smoothing or edge trim.
- Do NOT add CID_BIAS, hierarchical clustering, FAC, or network-flow solver; all are confirmed dead ends.
- Do NOT alter AQE `k=2` or FIC `reg=0.5`; these are the locked 14e B1 anchor.
- Do NOT silently lower `w_t` below 0.30 or above 0.55, outside the empirically-validated band.

## Handoff
- **WIN**: create `docs/subagent-specs/post-14n-next.md` planning a confirmation re-run plus headline-promotion PR.
- **MARGINAL**: log in `docs/experiment-log.md`, do NOT promote, and escalate to GNN edge classifier per the parent spec.
- **FAIL**: close the architecture-diverse-stream branch, escalate to GNN edge classifier, and consider EVA-02-L/14 only if GNN also fails.