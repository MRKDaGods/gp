# Post-14h Next Experiment Spec — Track-Quality Pre-Filter Sweep (`14i`)

**Date**: 2026-05-08
**Author**: MTMC Planner
**Status**: PROPOSED — ready for Coder implementation

## Decision

Choose **B (track-quality pre-filter)** as the next lever. This is CPU only and deliberately scoped as a cheap hedge after the feature-side plateau became hard to ignore:

- 14e WIN exhausted Stage-4 association tuning at the **0.77936** plateau.
- 14f confirmed the plateau bit-identically (A20 = **0.77936 exact**, `id_switches=154 exact`).
- 14g eliminated tertiary view expansion as a lever (`id_sw=154` unchanged across every `aqe_k=2` config).
- 14h eliminated robust per-tracklet aggregation as a lever (M0 = 14e B1 exact; all 8 robust modes worse; medoid showed the "stable but wrong" pattern).

The remaining cheap CPU axis is **what tracklets enter association in the first place**. The current pipeline runs `min_hits=2` at the tracker stage but does NOT filter on per-tracklet aggregate detection confidence or short-tracklet length at the Stage-3/4 boundary. If the residual 154 ID-switch error concentrates on low-quality short tracklets (which is plausible — short tracklets give the multi-query path fewer rows to pool, and low-confidence tracklets have noisier embeddings), pre-filtering them out before the association graph is built may raise IDP without sacrificing IDR enough to offset the gain.

Explicitly **defer 14j (re-fuse CLIP-SENet-FT against 14e B1 features)**:

- 13h sweep at the easier production baseline (0.7703) peaked at 0.7691 — already DEAD END.
- The 14e B1 lift (+0.91pp) came from AQE k=2 + TTA-smoothed primary, neither of which CLIP-SENet-FT participates in. The marginal value of CLIP-SENet-FT against the harder baseline is structurally lower, not higher.
- CLIP-SENet-FT tracklet features are NOT confirmed available aligned to the 14e/14h tracklet set (`run_kaggle_20260425_202123`); a fresh GPU re-extraction would likely be required (~1.5–2 hr P100), violating the CPU-only hedge criterion.
- Reconsider 14j only if 14i is also NEUTRAL/DEAD AND a new architecture stream (option A) is too expensive to attempt next.

Defer indefinitely: A (new architecture stream), C (GNN edge classifier), D (cross-camera self-training) — same reasoning as the post-14g spec. These are valid long-term directions, but they should not displace a 15-minute CPU hedge that can falsify the last cheap data-quality hypothesis.

## Hypothesis

The current 154 ID-switch floor is concentrated on a small subset of tracklets where the pooled embedding is unreliable for cross-camera matching. Two unreliability sources have NOT yet been pre-filtered:

1. **Short tracklets** (`length < L_min`): with a small number of detections, the softmax-quality pool effectively reduces to the top-1 quality-weighted frame, which is high-variance for cross-camera matching.
2. **Low average detection confidence**: tracklets where the underlying YOLO detector was uncertain across the trajectory have noisier per-frame embeddings, and TTA pre-smoothing only mitigates per-frame noise *given* the detection box — it cannot rescue a tracklet whose box pool is systematically off-target.

Stage-4 currently uses ALL 929 tracklets unconditionally. If a meaningful fraction of these are unreliable, dropping them before the association graph is built should raise IDP (fewer false matches via noisy tracklets) at a known cost to IDR (the dropped GT IDs that those tracklets covered are now missed). Net IDF1 = 2·IDP·IDR / (IDP + IDR) is sensitive to which side dominates.

Expected outcome band: **−0.30pp to +0.30pp** vs 14e B1 0.77936. Most likely NEUTRAL (filtering tracklets in CityFlowV2 typically just shifts the IDP/IDR balance without net gain). A clean WIN (≥+0.16pp to ≥0.7810) is unlikely but not impossible; a clean DEAD (≤−0.30pp) is also unlikely because mild filtering thresholds rarely drop more than ~5% of tracklets. The asymmetric upside justifies the cheap test.

## Implementation Plan

Single CPU kernel (no GPU). Use the existing 14h Stage-2 features in `outputs/14h_v3_summary/...` (or the on-disk Kaggle dataset published from the 14h kernel run) — do NOT rebuild Stage 2.

### New small CPU helper: `scripts/filter_tracklets.py`

- Reads the same per-camera tracklet metadata + `embeddings.npy` that 14h's `repool_stage2.py` reads.
- For each tracklet:
  - Compute `length = number_of_frames`.
  - Compute `avg_conf = mean(detection_confidence_per_frame)` — read from the existing per-frame quality / confidence record produced by Stage 2.
- Drop tracklets where `length < L_min` OR `avg_conf < τ_c`.
- Write filtered `embeddings_filtered.npy`, filtered tracklet metadata, and a `filter_summary.json` recording: total in, total out, drops by length, drops by confidence, drops by both, per-camera count delta.
- Atomic-swap convention identical to 14h: leave the original `embeddings.npy` untouched; Stages 3–5 read from `embeddings_filtered.npy` via existing override pattern (or via the same backup-and-overwrite convention used in 14h).

### Sweep grid (20 configs)

`L_min ∈ {3, 5, 8, 12}` × `τ_c ∈ {0.30, 0.35, 0.40, 0.45, 0.50}`. Plus one **F0 control** at `(L_min=0, τ_c=0.0)` (no filtering, must reproduce 14e B1 0.77936 exactly) — total 21 configs.

### Stages 3–5 anchor (fixed across all 21 configs)

- `stage4.association.aqe.k=2`
- `stage4.association.fic.regularisation=0.5`
- `stage4.association.graph.similarity_threshold=0.48`
- `stage4.association.fusion.w_tertiary=0.525`
- `stage4.association.fusion.w_primary=0.475`
- `stage4.association.fusion.w_secondary=0.0`
- `stage4.association.gallery_expansion.threshold=0.48`
- `stage4.association.gallery_expansion.orphan_threshold=0.38`
- `stage4.association.intra_merge.threshold=0.80`
- `stage4.association.intra_merge.gap=30`
- `stage4.association.temporal_overlap_bonus=0.05`
- `stage4.association.algorithm=conflict_free_cc`
- `stage4.association.pca.n_components=384`
- `stage5.mtmc_only_submission=false`

Same fixed config as 14e B1, 14f A20, 14g S0, 14h M0.

### Drift gate (F0)

F0 (`L_min=0, τ_c=0.0`) MUST reproduce **0.77936 EXACT** with `id_switches=154 EXACT`. Tolerance ±0.001 (tighter than 14g/14h ±0.005 because no Stage-2 rebuild is involved — just an identity filter). If F0 deviates, halt and diagnose (filter code is altering the tracklet ordering or dropping rows incorrectly).

### Outputs

Persist `outputs/14i_v1_summary/14i_summary.json` mirroring the 14h summary schema (per-config: `mtmc_idf1`, `id_switches`, `mota`, `trackeval_idf1`, dropped counts, stage timings, plus a top-level `best`/`overall_best` and `filter_grid`).

## Stop Criteria (relative to 14e B1 = 0.77936)

| Verdict | Best of {F0..F20} MTMC IDF1 | Action |
|:-------:|:--------------------------:|:------:|
| **WIN** | ≥ **0.7810** (+0.16pp) | Promote: re-run the winning `(L_min, τ_c)` on a second seed, sweep the immediate neighborhood, then commit headline change. |
| **MARGINAL** | 0.7795–0.7810 | Keep 0.77936 headline; document the best filter point as a sub-optimum; consider second seed before deciding. |
| **NEUTRAL** | 0.7785–0.7795 | Track-quality pre-filtering is not a lever on the current feature build. Pivot to **14j: new third feature stream (EVA-02 ViT-Large)**, GPU-heavy, multi-day. |
| **DEAD** | < 0.7785 | Filtering actively hurts (recall loss dominates precision gain). Revert; pivot directly to 14j. |
| **DRIFT** | F0 not within ±0.001 of 0.77936 | Halt sweep; investigate filter pipeline. Do not promote any filtered config. |

## Expected Walltime

- Filter computation per config: ~5 s CPU.
- Stages 3–5 per config: ~30 s CPU (matches 14g/14h timings).
- 21 configs total: **~12–15 min CPU**, no GPU.

## Coder Handoff Checklist

1. Create `scripts/filter_tracklets.py` per the spec above. Add a unit test verifying F0 (no-op filter) produces identical tracklet count, embedding rows, and metadata as the input.
2. Verify per-frame detection confidence is available in the existing Stage-2 output (it should be in tracklet JSON / metadata; check `data/outputs/.../stage2/tracklets_*.json` schema).
3. Build kernel `notebooks/kaggle/14i_track_quality_filter/` from a copy of `notebooks/kaggle/14h_robust_pooling/`. Update kernel-metadata.json (id `yahiaakhalafallah/14i-track-quality-filter`, title, code_file). Set `enable_gpu: false`, `enable_internet: true`, `machine_shape: cpu`. `kernel_sources` must include the **14h kernel output dataset** (so we can reuse its `embeddings.npy` and `embeddings_tertiary.npy` directly without rebuilding Stage 2). Do NOT include the 10a Stage-1 dataset — Stage 1 is not re-executed.
4. Apply the filter sweep cell. Run F0 drift gate FIRST. On drift failure (F0 ≠ 0.77936 ± 0.001), write a diagnosis cell and DO NOT push the rest of the sweep.
5. Push **once** per `.github/copilot-instructions.md` Kaggle Push Safety Rules. Watch for `not valid dataset sources` warnings; cancel + refix if seen.
6. Persist `outputs/14i_v1_summary/14i_summary.json`.
7. On WIN: run second-seed re-validation and a tighter local sweep around the winner's `(L_min, τ_c)` before updating findings/experiment-log/copilot-instructions.
8. On MARGINAL/NEUTRAL/DEAD: keep 0.77936 headline. Write `docs/subagent-specs/post-14i-next.md` proposing **14j: EVA-02 ViT-Large fine-tuned on CityFlowV2 as third feature stream**, GPU-heavy, multi-day (download + fine-tune + Stage 2 extract + 3-way fusion sweep).
9. On DRIFT: write the diagnosis. Halt the filtering thread.

## On-deck if 14i NEUTRAL/DEAD

**14j (DEFERRED until 14i decided)** — choice between:

- **14j-A**: EVA-02 ViT-Large (different pretraining = MIM + image-text contrastive), CityFlowV2 fine-tune, integrate as 3rd score-fusion stream OR replace DINOv2 tertiary. HIGH effort: ~6–8 hr GPU for fine-tune + ~2 hr GPU for Stage 2 + CPU fusion sweep. Best prior signal: not yet attempted in *this* configuration; 09o EVA02 ViT-B/16 was a 256px CLIP variant and gave only 48.17% mAP (different recipe — base/256 vs Large/likely-336).
- **14j-B**: Re-fuse CLIP-SENet-FT (13f checkpoint) with 14e B1 features as a 3-way score fusion. Requires CLIP-SENet-FT tracklet feature alignment to the 14e/14h tracklet set; if 13h Kaggle output features are not recoverable on this tracklet set, requires a fresh GPU re-extraction (~1.5–2 hr P100). Prior signal at the 0.7703 baseline was −0.12pp DEAD END; the 14e B1 baseline is harder, so the prior is even weaker. Run only if 14j-A is judged too expensive AND 13h features are recoverable cheaply.

Pick between 14j-A and 14j-B after seeing 14i results. 14j-A is the preferred path because the 14e/14g/14h evidence consistently points to **feature diversity**, not feature reweighting, as the missing lever.