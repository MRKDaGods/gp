# Post-14f Next Experiment Spec — DINOv2 4-View TTA Expansion (`14g`)

**Date**: 2026-05-07
**Author**: MTMC Planner
**Status**: PROPOSED — ready for Coder implementation

## Decision

14f confirmed 14e B1 = **0.77936** as a reproducible plateau and proved the Stage-4 association axis (`w_t × thr × FIC × AQE`) is **fully saturated** on the current TTA features. The next IDF1 lever must be on a **different axis** — specifically, **feature quality on the tertiary stream**.

The 14c v2 TTA recipe is asymmetric: the primary CLIP TransReID stream uses **4 views** `{original, hflip, scale_0.95, scale_1.05}` while the tertiary DINOv2 ViT-L/14 stream uses only **2 views** `{original, hflip}`. After the 14e WIN the fusion weight is `w_t = 0.525` (tertiary is essentially half the fused score), so the tertiary stream's noise floor is now the dominant residual error source on cross-camera matches. Symmetrizing DINOv2 to 4 views is the single cheapest, most directly-targeted intervention available — same proven axis (Stage-2 multi-crop TTA) that just delivered +0.91pp.

Three candidate directions were considered:

| Option | Lever | Effort | Prior signal | Pick |
|:------:|:-----:|:------:|:------------:|:----:|
| **A. DINOv2 4-view TTA expansion** | Tertiary feature quality | 1 GPU P100 kernel, ~2.5–3 hr | Same axis as 14e WIN; tertiary now half the fusion | ✅ |
| B. Track-quality pre-filter | Drop low-quality tracklets | CPU only, ~15 min | None — speculative; tracker already runs `min_hits=2` | ❌ Defer |
| C. GNN edge classifier | Learned association | Multi-week — training data, model, integration | Long-term ambitious target | ❌ Defer |

**Choose A.** Cheap GPU spend with a positive prior on the same axis. If A is NEUTRAL or DEAD, escalate to richer primary TTA (6/8 views with rotations) before committing to B (track filtering, no positive prior) or C (GNN, multi-week).

## Hypothesis

The 14c v2 design used 2 DINOv2 views because DINOv2 ViT-L/14 is the largest model in the pipeline and the original walltime budget made 4 views expensive. With `w_t = 0.525`, every percentage point of DINOv2 embedding noise contributes ~0.525× to the fused similarity. The 14e WIN came from reducing AQE from k=3 to k=2 — i.e., trusting the per-tracklet TTA-smoothed embedding more and the FAISS neighbours less. That move only paid off on the primary stream because the primary stream had already been smoothed by 4-view TTA. The tertiary stream did not get the same smoothing and may now be the bottleneck on the residual 154 ID switches.

Concretely:
1. Adding `scale_0.95` and `scale_1.05` to DINOv2 should produce a more stable per-tracklet tertiary embedding (lower per-detection variance), narrowing the cross-camera distribution.
2. Because `w_t = 0.525` and AQE is already at the k=2 optimum, any tertiary noise reduction propagates almost linearly into fused score reliability.
3. Risk: as with `aqe_k=1`, more smoothing can over-blur fine-grained discriminative signal. Mitigated because we are adding scale views (not rotations), which preserve top–bottom semantics that vehicles depend on (wheels, roof, windshield).

Expected outcome: small positive lift (+0.05–0.30pp), most likely in the **MARGINAL band**. A clean WIN (≥0.16pp lift to ≥0.7810) is plausible but not guaranteed.

## Implementation Plan

Single Kaggle GPU kernel: `yahiaakhalafallah/14g-dinov2-4view-tta-stage2`. Inputs:

- `kernel_sources`:
  - `yahiaakhalafallah/mtmc-10a-stages-0-2` (Stage 1 outputs to copy as-is — no redo of detection/tracking)
  - `yahiaakhalafallah/09s-dinov2-large-cityflowv2` (DINOv2 weights)
  - Same 10b FAISS-index dataset that 14c/14e/14f used (for Stage-3-5 wiring after the new Stage-2 features are built)
- `dataset_sources`: same as 14c v2.
- GPU: **enabled**, `machine_shape: NvidiaTeslaP100`, `enable_internet: true`.

### Code change (Coder must apply this single patch)

Reuse the 14c v2 notebook structure exactly. The TTA recipe is selected by the env var `MTMC_TTA_RECIPE`. Add a new recipe value `14g_v1` and patch `src/stage2_features/reid_model.py` so that when `MTMC_TTA_RECIPE == "14g_v1"` and the model is DINOv2:

```python
self.tta_views = ["original", "hflip", "scale_0.95", "scale_1.05"]
```

(Primary CLIP TransReID retains the same 4 views as 14c v1.)

Practically: copy the 14c TTA-patch cell and change `os.environ["MTMC_TTA_RECIPE"] = "14c_v1"` → `"14g_v1"`, and add a parallel branch in the patch's view-selection block:

```python
if tta_recipe in ("14c_v1", "14g_v1") and self.is_transreid:
    if "dinov2" in vit_model.lower():
        if tta_recipe == "14g_v1":
            self.tta_views = ["original", "hflip", "scale_0.95", "scale_1.05"]
        else:
            self.tta_views = ["original", "hflip"]
    else:
        self.tta_views = ["original", "hflip", "scale_0.95", "scale_1.05"]
    self.normalize_views = True
```

All other TTA pipeline code (mean_l2 aggregation, normalization, view application via `_apply_tta_view`) is unchanged — it already iterates over `self.tta_views` generically.

### Run plan

1. Build kernel `notebooks/kaggle/14g_dinov2_4view_tta_stage2/` from a copy of `notebooks/kaggle/14c_tta_stage2/`. Adjust kernel-metadata (id, title, code_file). Set `enable_gpu: true`. Wire kernel sources as above.
2. Stage 0/1 are reused: copy `stage1/` from the mounted 10a kernel run dir as in 14c.
3. Run Stage 2 with the new `14g_v1` recipe. Expected walltime: 14c v2 took ~2 hr on P100; adding 2 DINOv2 views adds ~30–60 min for an estimated total of **~2.5–3 hr**. If walltime exceeds 4 hr, abort and re-scope.
4. After Stage 2 completes, run Stages 3–5 with the **14e B1 anchor config** as the baseline drift check:
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
5. **Drift gate**: if the tertiary-stream change broke nothing the new feature run at the 14e B1 anchor should land within ~0.005 of 0.77936. If it lands below 0.770, halt and write a diagnosis cell — do NOT run the mini-sweep.
6. **CPU mini-sweep** (8 configs) to find the new optimum (the tertiary geometry has changed, so the fusion/AQE optimum may shift):

| Label | aqe_k | w_tertiary | similarity_threshold | fic_reg |
|:-----:|:-----:|:----------:|:--------------------:|:-------:|
| S0 (anchor) | 2 | 0.525 | 0.48 | 0.5 |
| S1 | 2 | 0.500 | 0.48 | 0.5 |
| S2 | 2 | 0.550 | 0.48 | 0.5 |
| S3 | 2 | 0.575 | 0.48 | 0.5 |
| S4 | 2 | 0.525 | 0.46 | 0.5 |
| S5 | 2 | 0.525 | 0.50 | 0.5 |
| S6 | 3 | 0.525 | 0.48 | 0.5 |  ← probe in case more views push optimum back to k=3
| S7 | 2 | 0.525 | 0.48 | 0.4 |

S0 IS the drift check. Run S0 first; if it fails the gate above, halt.

Total: 1 GPU Stage-2 build + 8 CPU Stage-3-5 runs (~30 s each).

## Stop Criteria (relative to 14e B1 = 0.77936)

| Verdict | Best of {S0..S7} MTMC IDF1 | Action |
|:-------:|:--------------------------:|:------:|
| **WIN** | ≥ **0.7810** (+0.16pp) | Promote to new headline. Update findings, experiment-log §2.14, copilot-instructions. Replicate winning config on a second seed before final promotion. |
| **MARGINAL** | 0.7795–0.7810 | Keep 0.77936 headline; document tertiary 4-view as sub-optimum; consider a follow-up that combines DINOv2 4-view with a 2-seed average for noise reduction. |
| **NEUTRAL** | 0.7785–0.7795 | Keep 0.77936 headline; tertiary TTA expansion confirmed null; escalate to **14h: richer primary TTA** (6-view: + rotation ±5°, OR + color jitter) before pivoting to track-filter or GNN. |
| **DEAD** | < 0.7785 | Tertiary 4-view actively hurts (likely over-smoothing, mirroring `aqe_k=1` regression). Revert; pivot directly to **14h: track-quality pre-filter** (the previous deferred Option B). |
| **DRIFT** | S0 not within ±0.005 of 0.77936 | Halt sweep; investigate Stage-2 build correctness (TTA recipe wiring, DINOv2 weights mount, run dir copy). Do not promote any config. |

Wider drift tolerance (±0.005 vs 14f's ±0.001) because the tertiary embedding actually *changed* — exact reproduction is not expected. The gate is "did the larger DINOv2 TTA cause catastrophic regression" not "is the kernel deterministic".

## Expected Walltime

- Stage 2 rebuild on P100: **~2.5–3 hr** (vs 14c v2 ~2 hr; +50% on DINOv2 views, primary unchanged).
- Stages 3–5 mini-sweep (8 configs × ~35 s): **~5 min CPU** appended to the same kernel.
- **Total: ~2.5–3.5 hr GPU**, single P100 slot.

GPU budget: one P100 slot for ~3 hr. No conflict with other live work as long as no other GPU kernel is queued on the same Kaggle account.

## Coder Handoff Checklist

1. Build `notebooks/kaggle/14g_dinov2_4view_tta_stage2/` from a copy of `notebooks/kaggle/14c_tta_stage2/`. Update kernel-metadata.json (id `yahiaakhalafallah/14g-dinov2-4view-tta-stage2`, title, code_file). Keep `enable_gpu: true`, `enable_internet: true`, `machine_shape: NvidiaTeslaP100`. Same kernel_sources + dataset_sources as 14c v2.
2. Apply the single TTA-patch change (env var `14g_v1` + DINOv2 view branch). Verify on disk with `python -c "import json; nb=json.load(open(...)); ..."` per `.github/copilot-instructions.md` notebook-editing rule.
3. Add a CPU mini-sweep cell (8 configs above) AFTER the Stage 2 run completes. Run S0 first; gate on S0 ∈ [0.77436, 0.78436]. On failure, write a diagnosis cell and DO NOT push more configs.
4. Push **once** per `.github/copilot-instructions.md` Kaggle Push Safety Rules. Watch for `not valid dataset sources` warnings; cancel + refix if seen. After push, confirm Stage 2 starts running before walking away. Cancel duplicates.
5. Persist outputs: write `outputs/14g_v1_summary/14g_summary.json` mirroring 14c/14e/14f schema.
6. On **WIN**: re-run the winning config on a second seed before promoting. Update `docs/findings.md`, `docs/experiment-log.md` § 2.14, `.github/copilot-instructions.md` headline metric and "What Actually Worked" bullet.
7. On **MARGINAL/NEUTRAL/DEAD**: keep 0.77936 headline. Write `docs/subagent-specs/post-14g-next.md` proposing **14h: richer primary TTA expansion (6-view rotation/color)** if NEUTRAL, or **14h: track-quality pre-filter** if DEAD. Update findings + experiment-log § 2.14 with the result.
8. On **DRIFT**: write the diagnosis. Pause the TTA thread until Stage-2 reproducibility is restored.
