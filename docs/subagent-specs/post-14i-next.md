# Post-14i Next Experiment Spec — FastReID R50-IBN as 4-Way Score-Fusion Stream (`14j`)

**Date**: 2026-05-08
**Author**: MTMC Planner
**Status**: PROPOSED — ready for Coder implementation

## Decision

Choose **A (FastReID R50-IBN as a 4th score-fusion stream)** as the next lever. This is the cheapest available test of "genuinely new signal" using existing CityFlowV2-trained assets. The 14e/14f/14g/14h/14i evidence has now confirmed a 4-axis plateau at **0.77936** (Stage-4 tuning, tertiary view expansion, tracklet aggregation, track-quality pre-filter all NEUTRAL/MARGINAL). All cheap CPU-only experiments are exhausted; the only remaining levers all require GPU work.

### Why R50-IBN over other candidates

The current 14e B1 feature stack is **all-ViT**: primary CLIP ViT-B/16 (256px, TTA-smoothed) + tertiary DINOv2 ViT-L/14. Among existing trained CityFlowV2 checkpoints:

- **09l v3 LAION-2B CLIP ViT-B/16** (mAP 78.61%): DEAD via 10c v56 — fusing two ViT CLIP models is too correlated.
- **09o v1 EVA02 ViT-B/16 CLIP** (mAP 48.17%): DEAD — well below ensemble threshold.
- **09k v1 ViT-Small/16 IN-21k** (mAP 48.66%): DEAD — too weak.
- **09m v2 CLIP RN50x4** (mAP 1.55%): DEAD — catastrophic training failure.
- **13f CLIP-SENet-FT** (CityFlow fine-tune of v6): DEAD via 13h fusion at the 0.7703 baseline (peak −0.12pp); harder 14e B1 anchor weakens the prior further.
- **09s DINOv2 ViT-L/14**: already deployed as tertiary stream.
- **09n / 09p FastReID R50-IBN** (mAP 63.64% / improved variant): **the only existing CNN-architecture CityFlow checkpoint**. Tested in 10c v60/v61 fusion sweeps at the easier 0.7703 baseline → peak +0.06pp at `w=0.10` (within noise but not negative).

09p R50-IBN brings **genuinely different inductive bias** to the current all-ViT stack (convolutional locality, IBN normalization, FastReID head — none of which are present in CLIP/DINOv2). The harder 14e B1 anchor at `w_tertiary=0.525, aqe_k=2, thr=0.48` has different score calibration than the production fusion that produced the +0.06pp prior result, so a re-test is genuinely untested. This is the only candidate where the existing dead-end signal has both an architectural-diversity argument AND a measurement-context argument for retesting.

Explicitly **defer 14k (EVA-02 ViT-Large CityFlow fine-tune)**: 6–8 hr GPU fine-tune + 2 hr GPU Stage 2 + CPU sweep, multi-day. Reconsider 14k only if 14j is also NEUTRAL/DEAD.

Defer indefinitely: GNN edge classifier, pseudo-label self-training. These are valid long-term directions, but they should not displace a ~1 hr GPU + ~20 min CPU test that can falsify the last cheap-ish existing-checkpoint hypothesis.

## Hypothesis

The 14e/14g/14h/14i evidence consistently points to **feature diversity** (not feature reweighting, not aggregation, not data filtering) as the missing lever. The current stack contains two **ViT-architecture** feature streams that share substantial inductive bias (transformer self-attention, patch embeddings, large-scale image-or-image+text pretraining). A CNN-architecture stream provides:

1. **Locality bias**: convolutions encode neighborhood structure that ViT self-attention does not impose.
2. **IBN normalization**: instance normalization in early layers improves cross-domain invariance — directly relevant for non-overlapping cross-camera matching.
3. **Different failure modes**: where ViT features confuse vehicles by global colour/shape signatures, R50-IBN features should fail differently (e.g. by texture or local detail mismatch), so score-level fusion has decorrelated error patterns to exploit.

Even if the standalone R50-IBN MTMC IDF1 is well below the all-ViT stack's 0.77936, a small fusion weight (w_secondary ≈ 0.05–0.15) could shift IDS-borderline cases without dragging the global score. The 10c v60/v61 prior signal of +0.06pp at the easier baseline is consistent with this — the question is whether the 14e B1 anchor's different calibration pushes that signal above noise.

Expected outcome band: **−0.30pp to +0.30pp** vs 14e B1 0.77936. Most likely NEUTRAL/MARGINAL. A clean WIN (≥+0.16pp to ≥0.7810) would imply CNN diversity is the unlock; a clean DEAD would close the existing-checkpoint hypothesis and route directly to 14k EVA-02.

## Implementation Plan

Two-kernel chain: GPU feature extraction kernel, then CPU 4-way fusion sweep kernel.

### Kernel 1 (GPU, ~30–45 min P100): `notebooks/kaggle/14j_r50ibn_extract/`

Build from a copy of `notebooks/kaggle/14g_dinov2_4view_tta/` (it already does the right pattern of "extract a single tertiary stream over the existing tracklet box set"). Replace the DINOv2 model loading with:

- Load FastReID R50-IBN backbone from the **most recent 09p checkpoint dataset** (verify which exists via `kaggle datasets list -m -s 09p`; expected source is `gumfreddy/09p-fastreid-r50-extended-cityflowv2` or similar). If 09p output is missing, fall back to the 09n checkpoint dataset.
- Run inference on the **same 14c v2 tracklet box list** that produced `embeddings.npy` for 14e/14f/14g/14h. Same 929 tracklets, same per-tracklet softmax-quality pooling, same TTA views {original, hflip, scale_0.95, scale_1.05} **only if 09p was trained for that augmentation distribution; otherwise use original-only TTA** to match the model's training setup. (TTA across an underadapted model may inject noise.)
- Output: `embeddings_secondary.npy` shape `(929, D_r50)`, where `D_r50` is the FastReID R50-IBN embedding dimension (typically 2048 before BNNeck or as projected by the FastReID head — check the 09p inference code and match it).
- Push the kernel output as a Kaggle dataset (e.g. `yahiaakhalafallah/14j-r50ibn-features`) for consumption by Kernel 2.

Drift gate for Kernel 1: confirm tracklet count is 929 (match 14h v3 input), embedding rows match, no NaNs, embeddings are L2-normalised before saving.

### Kernel 2 (CPU, ~15–20 min): `notebooks/kaggle/14j_4way_fusion_sweep/`

Build from a copy of `notebooks/kaggle/14i_track_quality_prefilter/` (it already chains the 14h v3 features + Stages 3–5). Add the new secondary feature stream and sweep `w_secondary` while rescaling primary/tertiary weights to maintain unit sum.

Sweep grid (15 configs + 1 control):

- **W0 (drift control)**: `w_p=0.475, w_secondary=0.00, w_t=0.525` — must reproduce 14e B1 0.77936 ± 0.001.
- **W1–W15**: `w_secondary ∈ {0.05, 0.10, 0.15, 0.20, 0.25}` × `similarity_threshold ∈ {0.46, 0.48, 0.50}`. At each `w_secondary`, rescale `w_p` and `w_t` proportionally so they still sum to `(1 − w_secondary)` while preserving the `w_p : w_t = 0.475 : 0.525` ratio. Keep all other 14e B1 anchor params fixed.

Stages 3–5 anchor (fixed across W0–W15):
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

### Drift gate (W0)

W0 (`w_secondary=0.0`) MUST reproduce **0.77936** with `id_switches=154 EXACT`, tolerance ±0.001 (same gate as 14g/14h/14i). If W0 deviates, halt and diagnose — the 4-way fusion plumbing is altering the score even at zero secondary weight, which means the rescaling is wrong or the secondary stream is leaking through a non-fused code path.

### Outputs

Persist `outputs/14j_v1_summary/14j_summary.json` mirroring the 14i summary schema (per-config: `mtmc_idf1`, `id_switches`, `mota`, `trackeval_idf1`, fusion weights, plus `best`/`overall_best` and the full sweep grid).

## Stop Criteria (relative to 14e B1 = 0.77936)

| Verdict | Best of W0..W15 MTMC IDF1 | Action |
|:-------:|:------------------------:|:------:|
| **WIN** | ≥ **0.7810** (+0.16pp) | Promote: re-run the winning `(w_secondary, thr)` on a second seed, sweep `w_secondary` neighborhood at ±0.025 resolution, then commit headline change. |
| **MARGINAL** | 0.7795–0.7810 | Keep 0.77936 headline; document the best fusion point as a sub-optimum; second-seed before deciding. |
| **NEUTRAL** | 0.7785–0.7795 | CNN-vs-ViT diversity is not a lever on this feature build. Pivot to **14k: EVA-02 ViT-Large fine-tune from scratch on CityFlowV2** (multi-day GPU, last cheap-ish lever). |
| **DEAD** | < 0.7785 | R50-IBN actively hurts (correlated noise dominates). Revert; pivot directly to 14k. Strongly consider that ALL existing-checkpoint fusion options are now fully closed. |
| **DRIFT** | W0 not within ±0.001 of 0.77936 | Halt sweep; investigate fusion plumbing (rescaling bug, wrong feature normalization, or leakage). Do not promote any config. |

## Expected Walltime

- Kernel 1 (R50-IBN feature extraction): **~30–45 min P100 GPU**, 1 GPU slot.
- Kernel 2 (4-way fusion sweep): **~15–20 min CPU**, no GPU. 16 configs at ~30s each plus per-config Stage-4/5.
- Total: ~1 hr wall-clock, 1 GPU slot, 1 CPU slot.

## Coder Handoff Checklist

1. Verify the 09p checkpoint and its inference code are available as a Kaggle dataset. Run `kaggle datasets list -m -s 09p-fastreid` (gumfreddy account) and `kaggle datasets list -m -s 09n-fastreid` to confirm. If neither is recoverable as a usable model checkpoint, halt 14j and switch to 14k (EVA-02 fine-tune) directly.
2. Build Kernel 1 from a 14g copy. Update kernel-metadata.json (`id`: `yahiaakhalafallah/14j-r50ibn-features`, title, code_file). Set `enable_gpu: true`, `enable_internet: true`, `machine_shape: NvidiaTeslaP4` or P100 as available. `kernel_sources` must include the **14h v3 kernel output dataset** (for the tracklet box list and 14c v2 TTA crops) and the **09p checkpoint dataset**. Validate that the tracklet count is 929 before pushing.
3. Push Kernel 1 once per `.github/copilot-instructions.md` Kaggle Push Safety Rules. Watch for `not valid dataset sources` warnings; cancel + refix if seen.
4. After Kernel 1 completes, push the output as a Kaggle dataset.
5. Build Kernel 2 from a 14i copy. Add the secondary feature stream loading. Apply the W0 drift control + W1–W15 sweep cells. Run W0 FIRST. On drift failure, write a diagnosis cell and DO NOT push the rest of the sweep.
6. Push Kernel 2 once. Persist `outputs/14j_v1_summary/14j_summary.json`.
7. On WIN: run second-seed re-validation and a tighter local sweep around the winner before updating findings/experiment-log/copilot-instructions.
8. On MARGINAL/NEUTRAL/DEAD: keep 0.77936 headline. Write `docs/subagent-specs/post-14j-next.md` proposing **14k: EVA-02 ViT-Large fine-tuned from scratch on CityFlowV2 as a third feature stream**, GPU-heavy, multi-day.
9. On DRIFT: write the diagnosis. Halt the fusion thread.

## On-deck if 14j NEUTRAL/DEAD

**14k (DEFERRED until 14j decided)**: EVA-02 ViT-Large fine-tuned on CityFlowV2 as a 3rd feature stream OR replacement for DINOv2 tertiary.

- Pretraining: MIM + image-text contrastive — different recipe from both CLIP (image-text only) and DINOv2 (MIM only).
- Effort: ~6–8 hr GPU fine-tune (P100) + ~1.5–2 hr GPU Stage-2 inference + CPU 4-way fusion sweep.
- Risk: 09o v1 EVA02 ViT-B/16 reached only 48.17% mAP under the current recipe. EVA-02 Large at 224px or 336px might cross the practical ensemble bar (≥65% mAP) but requires hyperparameter retuning.
- Pre-spec: confirm the EVA-02 Large checkpoint and the timm/transformers loader compatibility before committing GPU time.

If 14k also NEUTRAL/DEAD, the existing-checkpoint and new-feature-stream paths are fully exhausted and the only remaining levers are GNN edge classifier (multi-week) and pseudo-label self-training (multi-week, careful threshold selection required).