# Post-14g Next Experiment Spec — Robust Tracklet Pooling Sweep (`14h`)

**Date**: 2026-05-08
**Author**: MTMC Planner
**Status**: PROPOSED — ready for Coder implementation

## Decision

14g delivered the most informative null result of the TTA campaign: symmetrizing the tertiary DINOv2 stream to 4 views produced **zero change in association decisions** (every `aqe_k=2` config landed at `id_switches=154` exact, matching the 14e B1 / 14f A20 plateau). That single observation eliminates "more views" as an IDF1 lever — both primary (14c v2) and tertiary (14g) 4-view TTA are now confirmed saturated. The 0.77936 plateau is **feature-diversity limited and tracklet-aggregation limited**, not view-coverage limited.

Four candidate directions were ranked:

| Option | Lever | Effort | Prior signal | Pick |
|:------:|:-----:|:------:|:------------:|:----:|
| **D+F. Robust tracklet pooling** | Stage-2 aggregation across frames | 1 GPU rerun + CPU sweep | Multi-query path already in code; current pool is softmax-quality mean (outlier-sensitive) | ✅ |
| B. Track-quality pre-filter | Drop short / low-confidence tracklets | CPU only | None — speculative; tracker already runs `min_hits=2` | ❌ Defer to 14i |
| A. New 3rd architecture stream | Feature diversity | HIGH (download + train + integrate) | All prior 3rd-stream attempts (CLIP-SENet, OSNet, ResNet101, ArcFace, EVA02) **DEAD** | ❌ Defer |
| C. GNN edge classifier | Learned association | VERY HIGH (training data + model + integration) | Long-term ambitious target | ❌ Defer |
| E. CityFlow self-training | Pseudo-label fine-tune | HIGH (GPU + threshold tuning) | Domain gap fix lifted CLIP-SENet 0.6855→0.7099 standalone but fusion peak 0.7691 < 0.7703 (13f/13h DEAD END) | ❌ Defer |

**Choose D+F.** Cheapest available lever with a positive structural prior: every cross-camera ID switch in the 154-floor is anchored to a per-tracklet pooled embedding. The current pool is `softmax(quality_score · T) ⋅ frame_embedding`, which is robust to *quality* outliers (blur, small boxes) but NOT to *embedding* outliers (occluded but sharp views, detection drift onto a neighbouring vehicle, partial truncation). One bad high-quality frame can pull the mean off-axis and flip a cross-camera match. Robust pooling — median, medoid, trimmed mean — directly addresses this.

If D+F is also NEUTRAL, escalate to (B) track-quality pre-filter (cheap, no positive prior but easy to test). Only after both D+F and B fail do we commit GPU/engineering to (A) a new architecture stream, then (C) GNN.

## Hypothesis

Stage-2 currently aggregates per-tracklet via `softmax(quality · T) ⋅ embedding` (T=3.0 default). The TTA expansion in 14c v2 makes each *frame* embedding more stable (averaged across `{original, hflip, scale_0.95, scale_1.05}`), but the *cross-frame* aggregation is unchanged — and `softmax(quality)` is a weighted *mean*, so it is still vulnerable to embedding-space outliers that happen to come from high-quality crops. With TTA already smoothing per-frame noise, the residual error source on the 154 ID-switch floor is most likely a small subset of tracklets where a sharp-but-wrong-view frame dominates the pooled embedding.

Three structural points support this:

1. **14g signal**: tertiary view expansion changed `id_switches` by zero. Per-frame stability is no longer the bottleneck.
2. **Existing `multi_query` path**: the codebase already supports saving the top-K highest-quality TTA-smoothed embeddings per tracklet (`get_tracklet_multi_query_embeddings`). Stage 4 has a multi-query max-similarity branch (`pipeline.py:1171`). The infrastructure for K-element robust aggregation is mostly in place.
3. **Robust statistics theory**: replacing a mean with a median / geometric-median / medoid asymptotically removes the influence of bounded-ratio outliers without sacrificing convergence rate on inliers. On normalized unit-sphere embeddings the relevant operation is on the *direction* (cosine), making medoid and L2-median the natural choices.

Expected outcome: small positive lift (+0.05–0.30pp) most likely in the **MARGINAL** band. A clean WIN (≥0.16pp lift to ≥0.7810) is plausible if even ~20% of the residual ID switches are outlier-driven. Risk: median/medoid over short tracklets (length < 5) reduces effective sample size and may add variance — mitigation is to use the robust aggregate ONLY when tracklet length ≥ K_min, and fall back to the existing softmax-quality mean otherwise.

## Implementation Plan

Single Kaggle GPU kernel: `yahiaakhalafallah/14h-robust-tracklet-pooling`. Inputs:

- `kernel_sources`:
  - `yahiaakhalafallah/mtmc-10a-stages-0-2` (Stage 1 outputs to copy as-is — no redo of detection/tracking)
  - The 10b FAISS-index dataset used by 14c/14e/14f/14g
  - Same ReID weights datasets (CLIP TransReID + DINOv2 ViT-L/14) used by 14g
- `dataset_sources`: same as 14g v1.
- GPU: **enabled**, `machine_shape: NvidiaTeslaP100`, `enable_internet: true`.

### Stage-2 change (Coder)

Reuse the **14c v2** TTA recipe (the proven WIN feature build — primary 4-view + tertiary 2-view; do NOT use 14g's tertiary-4-view recipe since it produced zero net change and would only add walltime). Add ONE config knob:

```yaml
stage2:
  multi_query:
    k: 24            # save top-24 highest-quality TTA-smoothed embeddings per tracklet
```

This activates the existing `get_tracklet_multi_query_embeddings` path. No new model code, no new TTA logic. The Stage-2 outputs gain `multi_query_embeddings.npy` alongside the existing `embeddings.npy`. Expected walltime: same as 14c v2 (~2 hr on P100) since multi-query selects from already-extracted per-frame embeddings (no extra forward passes).

### Aggregation post-processor (NEW small module)

Add `src/stage2_features/robust_pool.py` with these aggregation functions (all operate on a `(K, D)` block of L2-normalized TTA-smoothed embeddings and return a single `(D,)` L2-normalized vector):

| Mode | Formula | Notes |
|:----:|:--------|:------|
| `mean` | mean(rows); L2-renorm | Drift control — must reproduce the existing softmax pool to within ±0.001 IDF1 when fed the same K embeddings |
| `median` | per-dim median; L2-renorm | Cheap robust baseline; not directionally optimal on the sphere but easy |
| `geo_median` | Weiszfeld iteration (≤20 steps, ε=1e-6); L2-renorm | Direction-aware L2-median |
| `medoid` | argmax_i(Σ_j cos(x_i, x_j)) | Most central existing embedding; pure exemplar selection |
| `trimmed_mean_10` | drop bottom-10% by cos-to-mean; mean of rest; L2-renorm | Mild outlier rejection |
| `trimmed_mean_25` | drop bottom-25% by cos-to-mean; mean of rest; L2-renorm | Stronger outlier rejection |
| `top12_to_mean` | keep top-12 nearest-to-mean (cos); mean; L2-renorm | Hard truncation toward consensus |
| `top12_to_medoid` | keep top-12 nearest-to-medoid (cos); mean; L2-renorm | Combines exemplar-selection with averaging |

Length fallback: if a tracklet has fewer than `min_K=8` saved multi-query rows, fall back to the existing softmax-quality pool (do not over-trim short tracklets).

### Driver script

Add a CPU helper `scripts/repool_stage2.py` that:
1. Reads `multi_query_embeddings.npy` and the existing tracklet metadata for each camera.
2. For a given `--mode {mean, median, geo_median, medoid, trimmed_mean_10, trimmed_mean_25, top12_to_mean, top12_to_medoid}`, computes the new pooled embedding per tracklet.
3. Writes a new `embeddings_<mode>.npy` alongside the original (does NOT overwrite).
4. Stage 3-5 then run with `stage3.embeddings_filename=embeddings_<mode>.npy` (new override; thread through `src/stage3_indexing/pipeline.py`).

If the override threading is more invasive than expected, the simpler implementation is to atomically swap `embeddings.npy` per sweep cell (back up original, overwrite, run, restore). The notebook can do this in a try/finally block.

### Run plan

1. Build kernel `notebooks/kaggle/14h_robust_tracklet_pooling/` from a copy of `notebooks/kaggle/14c_tta_stage2/`. Adjust kernel-metadata (id `yahiaakhalafallah/14h-robust-tracklet-pooling`, title, code_file). Keep `enable_gpu: true`, `enable_internet: true`, `machine_shape: NvidiaTeslaP100`. Same kernel_sources + dataset_sources as 14c v2 (NOT 14g — we want primary 4-view + tertiary 2-view).
2. Confirm `MTMC_TTA_RECIPE = "14c_v1"` (NOT `14g_v1`).
3. Add the multi-query enable cell: `os.environ["MTMC_STAGE2_MULTI_QUERY_K"] = "24"` (or thread via the existing OmegaConf path `stage2.multi_query.k=24`).
4. Stage 0/1 are reused: copy `stage1/` from the mounted 10a kernel run dir as in 14c.
5. Run Stage 2 with the 14c v2 recipe + multi-query k=24. Expected walltime: **~2 hr on P100**. Verify `multi_query_embeddings.npy` is produced for each camera.
6. **Drift gate (M0)**: run Stages 3–5 with the **existing** `embeddings.npy` (the softmax-quality pool) at the 14e B1 anchor. Expected: ≥ 0.7790 (within ±0.005 of 0.77936, identical to 14g S0 since it's the same primary feature build). If M0 < 0.770, halt and diagnose — the multi-query enable should NOT have perturbed the primary pooled embedding.
7. **Aggregation sweep (CPU, ~12 min)**: for each mode `M ∈ {median, geo_median, medoid, trimmed_mean_10, trimmed_mean_25, top12_to_mean, top12_to_medoid}`, recompute pooled embeddings via `repool_stage2.py`, then run Stages 3–5 at the 14e B1 anchor:
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
8. Persist outputs: write `outputs/14h_v1_summary/14h_summary.json` mirroring 14c/14e/14f/14g schema. Include for each mode: `mtmc_idf1`, `id_switches`, `mota`, `trackeval_idf1`, the count of tracklets that hit the `min_K=8` fallback, and stage timings.
9. **If a mode WINS** (best ≥ 0.7810): re-run that winning mode's S-sweep around the 14e B1 anchor (`w_t ∈ {0.500, 0.525, 0.550}` × `thr ∈ {0.46, 0.48, 0.50}`, 9 configs) to confirm the optimum on the new pooled embedding. Robust pooling may shift the optimum because the tracklet embedding distribution has changed.

Total: 1 GPU Stage-2 build (~2 hr) + 1 drift CPU run + 7 aggregation CPU runs + (optional) 9-config refinement sweep on the winner.

## Stop Criteria (relative to 14e B1 = 0.77936)

| Verdict | Best of {M0..M7} MTMC IDF1 | Action |
|:-------:|:--------------------------:|:------:|
| **WIN** | ≥ **0.7810** (+0.16pp) | Promote: run the 9-config refinement sweep around the winning mode at the 14e B1 anchor before final headline change. Replicate the winning (mode + sweep) on a second seed. Update findings, experiment-log § 2.15, copilot-instructions. |
| **MARGINAL** | 0.7795–0.7810 | Keep 0.77936 headline; document the best robust-pool mode as a sub-optimum; consider a second seed before deciding. |
| **NEUTRAL** | 0.7785–0.7795 | Robust pooling does not move the floor on the current feature build. Pivot to **14i: track-quality pre-filter** (CPU only, ~30 min). |
| **DEAD** | < 0.7785 | Robust pooling actively hurts (likely over-aggressive trimming on short tracklets). Revert; pivot directly to **14i: track-quality pre-filter**. |
| **DRIFT** | M0 not within ±0.005 of 0.77902 (the 14g S0 anchor on the same primary 4-view build) | Halt sweep; investigate whether enabling multi-query perturbed the existing pooled embedding. Do not promote any aggregation mode. |

Drift tolerance is ±0.005 (matching 14g) because the multi-query enable is theoretically a no-op on `embeddings.npy` but in practice may perturb floating-point ordering during selection.

## Expected Walltime

- Stage 2 rebuild on P100 (14c v2 recipe + multi-query k=24): **~2 hr** (multi-query selection is free on already-extracted features).
- Drift gate M0: ~30 s CPU.
- Aggregation sweep (7 modes × ~30 s + 7 small repool computations × ~30 s): **~7–10 min CPU**.
- (Optional) 9-config winner refinement: **~5 min CPU**.
- **Total: ~2.25 hr GPU + ~15 min CPU**, single P100 slot.

GPU budget: one P100 slot for ~2.25 hr. No conflict with other live work as long as no other GPU kernel is queued on the same Kaggle account.

## Coder Handoff Checklist

1. Create `src/stage2_features/robust_pool.py` with the 8 aggregation functions described above. Add a unit test `tests/stage2_features/test_robust_pool.py` covering: (a) `mean` recovers L2-normalized arithmetic mean; (b) `medoid` returns one of the input rows; (c) `geo_median` converges and lies in the convex cone of inputs; (d) length fallback to softmax-quality mean when K < 8.
2. Create `scripts/repool_stage2.py` per the spec above. Verify on a single camera that all 7 non-default modes produce L2-unit outputs of shape `(N_tracklets, D)` matching the original `embeddings.npy`.
3. Build `notebooks/kaggle/14h_robust_tracklet_pooling/` from a copy of `notebooks/kaggle/14c_tta_stage2/`. Update kernel-metadata.json (id `yahiaakhalafallah/14h-robust-tracklet-pooling`, title, code_file). Keep `enable_gpu: true`, `enable_internet: true`, `machine_shape: NvidiaTeslaP100`. Same kernel_sources + dataset_sources as 14c v2.
4. Apply the multi-query enable (`stage2.multi_query.k=24`). Verify on disk with the `json.load() → modify → json.dump()` pattern per `.github/copilot-instructions.md` notebook-editing rule. Confirm `MTMC_TTA_RECIPE` remains `14c_v1` (NOT `14g_v1`).
5. Add the aggregation sweep cell AFTER the Stage 2 run completes. Run M0 (drift gate) FIRST. On drift failure (M0 < 0.770 or > 0.785), write a diagnosis cell and DO NOT push the rest of the sweep.
6. Push **once** per `.github/copilot-instructions.md` Kaggle Push Safety Rules. Watch for `not valid dataset sources` warnings; cancel + refix if seen. After push, confirm Stage 2 starts running before walking away. Cancel duplicates.
7. Persist outputs: write `outputs/14h_v1_summary/14h_summary.json`.
8. On **WIN**: run the 9-config refinement sweep on the winning mode, then re-run on a second seed. Update `docs/findings.md`, `docs/experiment-log.md` § 2.15, `.github/copilot-instructions.md` (headline metric + "What Actually Worked" bullet).
9. On **MARGINAL/NEUTRAL/DEAD**: keep 0.77936 headline. Write `docs/subagent-specs/post-14h-next.md` proposing **14i: track-quality pre-filter** (CPU-only, drop tracklets with confidence < τ_c or length < L_min, sweep τ_c ∈ {0.30, 0.35, 0.40, 0.45, 0.50} × L_min ∈ {3, 5, 8, 12}, 20 configs at the 14e B1 anchor).
10. On **DRIFT**: write the diagnosis. Pause the tracklet-aggregation thread until the multi-query enable is shown to be a no-op on `embeddings.npy`.