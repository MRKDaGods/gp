# Post-13h Next Experiment Spec — SAM2 Foreground Masking for Vehicle ReID

**Date**: 2026-05-07
**Author**: MTMC Planner
**Status**: PROPOSED — ready for Coder implementation

## Decision

After 13h fine-tune fusion landed at peak **0.7691 MTMC IDF1** (−0.12pp below production **0.7703**), the next experiment is **SAM2 foreground masking applied at Stage 2 ReID feature extraction**. This is the only candidate that (a) is genuinely untried per `docs/findings.md`, (b) attacks the documented feature-quality bottleneck along an orthogonal axis (input cleanliness), and (c) fits inside a single overnight Kaggle shift without retraining any ReID model.

### Why not the alternatives

- **Retrain 13f longer (24 ep)**: best-case extrapolation lifts standalone to ~0.73 and fusion to ~0.770–0.775; most likely lands at or below 0.7703. Negative-EV.
- **GNN edge classifier**: high-ceiling but multi-shift implementation cost; cannot complete in one overnight.
- **Person pipeline pivot**: capped at +0.6pp tracker-limited; already exhaustively swept.
- **Paper-writing pivot**: explicitly contradicts the user's "NEVER EXIT" experiment loop mandate.

## Hypothesis

CityFlowV2 vehicle crops contain heavy background context (road, other vehicles, buildings) that varies dramatically across non-overlapping cameras. TransReID ViT-B/16 CLIP at 256² has been shown to encode some of this background signal — evidenced by the **DINOv2 86.79% mAP → 0.744 MTMC IDF1** result, which proved that single-camera mAP gain does not equal cross-camera invariance. If we mask the foreground vehicle pixels with **SAM2** before feeding crops to TransReID, the resulting embeddings should be more camera-invariant. The same hypothesis predicted the historical AIC22 1st-place pipeline's use of foreground segmentation at Stage 2.

Expected behavior:
- Single-camera mAP may **decrease slightly** (loss of contextual cues).
- Cross-camera MTMC IDF1 should **increase**, ideally clearing **0.7703**.
- This is the inverse of the DINOv2 paradox and would be the first piece of evidence that we can rebalance the mAP-vs-MTMC tradeoff without retraining ReID.

## Implementation Plan

### Pipeline structure (single new Kaggle kernel, GPU)

Kernel slug: `yahiaakhalafallah/14a-sam2-masked-stage2`. Replaces only Stage 2 of the 10a chain — Stage 0/1 outputs from `yahiaakhalafallah/mtmc-10a-stages-0-2` v7 are reused by re-running detection-aligned crops through SAM2, then through the unchanged TransReID extractor.

Steps:
1. **Inputs** (Kaggle datasets):
   - `yahiaakhalafallah/mtmc-10a-stages-0-2` v7 (existing tracking outputs: per-frame detection JSON, tracklet JSON)
   - `yahiaakhalafallah/mtmc-data-cityflow` (raw frames or cached crops)
   - `yahiaakhalafallah/mtmc-weights` (TransReID ViT-B/16 CLIP checkpoint)
   - SAM2 weights: download `sam2_hiera_base_plus.pt` from Meta's public release into a fresh dataset `yahiaakhalafallah/sam2-weights` (one-time upload by Coder before push).
2. **SAM2 inference** (per detection):
   - Use `sam2_hiera_base_plus` (smallest checkpoint that maintains accuracy; ~80M params).
   - For each tracked detection bbox, prompt SAM2 with the bbox as a box-prompt; take the highest-IoU output mask.
   - Apply mask: zero out background pixels in the crop (do NOT crop tighter — preserve crop dimensions to keep TransReID's positional encoding aligned).
   - Optional refinement: dilate mask by 5px to avoid hard edges (set as config flag, default ON).
3. **TransReID feature extraction**: identical to existing 10a Stage 2 path; same checkpoint, same 256² resize, same per-tracklet mean pooling.
4. **Outputs**: emit `tracklets.json` with masked-feature embeddings; format matches the existing 10b input contract so 10b/10c can run unchanged.

### Hyperparameters / flags

- `sam2.checkpoint = sam2_hiera_base_plus.pt`
- `sam2.dilate_px = 5` (try 0 in ablation if walltime allows)
- `sam2.bbox_expand = 1.10` (10% padding before SAM2 prompt to give the mask predictor breathing room)
- `sam2.batch_size = 8` (P100 16GB constraint with image_size=512 inside SAM2)
- `stage2.image_size = 256` (UNCHANGED — match existing TransReID training)

### Downstream chain

Once 14a finishes, re-run:
- `yahiaakhalafallah/mtmc-10b-stage-3-faiss-indexing` v6 → unchanged
- `yahiaakhalafallah/mtmc-10c-stages-4-5-association-eval` v17 → unchanged, with the production-best config: `w_primary=0.40, w_dinov2=0.60, w_cs=0.0`, `stage4.association.graph.similarity_threshold=0.40` (or whichever value 10c v15 used; Coder must verify from `outputs/.../config.json`).

### Ablation (only if walltime allows after the main run)

A second kernel `14b-sam2-fusion-sweep` would replicate the 13d/13h fusion form, fusing **masked-TransReID** with **unmasked-TransReID** (treating masking as a feature-stream weighting rather than a hard replacement), at `w_mask ∈ {0.30, 0.50, 0.70, 1.00}`. Skip this if 14a alone clears 0.7703.

## Expected Impact Range

- **Optimistic**: +0.5 to +1.5pp MTMC IDF1 → **0.775–0.785**, clearing production and approaching the historical 0.784 v80 ceiling.
- **Central**: +0.0 to +0.5pp → **0.7703–0.7750**, marginal but informative.
- **Pessimistic**: −1.0 to 0.0pp → background context turns out to encode useful camera-invariant cues (e.g., shadow/orientation), or hard-edge artifacts hurt TransReID more than background noise.

The pessimistic case is informative regardless: it would reduce the GNN candidate's expected value by ruling out "input cleanup" as the missing ingredient, narrowing the remaining hypothesis space to learned association.

## Stop Criteria

### Milestone Update — 2026-05-07

14a v6 (`yahiaakhalafallah/14a-sam2-masked-stage2`) reached Kaggle completion and emitted `14a_summary.json` plus `checkpoint.tar.gz`. A local checkpoint audit found that the current tarball contains Stage 1 tracklets and GT annotations but no Stage 2 embedding artifacts (`embeddings.npy`, `embedding_index.json`, `hsv_features.npy`, or `embeddings_tertiary.npy`). The follow-up CPU evaluator has been pushed as 14b (`yahiaakhalafallah/14b-sam2-masked-eval`) v1 and reported `RUNNING`; it will produce the SAM2 MTMC IDF1 verdict as soon as a complete Stage 2 checkpoint is available.

Stop and update findings if any of these trigger:
1. **WIN**: 14a MTMC IDF1 ≥ **0.7720** (clears production by >0.17pp, beyond run-to-run noise of ~0.24pp). → Mark as live, push 14b ablation, plan a paper update.
2. **NEUTRAL**: 14a MTMC IDF1 ∈ [0.7680, 0.7720]. → Within noise. Mark as MARGINAL in findings, do not push 14b. Move to next candidate (GNN edge classifier).
3. **REGRESSION**: 14a MTMC IDF1 < **0.7680**. → Mark as DEAD END. Hypothesize whether mask edge artifacts or contextual signal loss caused the drop; record in findings. Move to GNN candidate.
4. **WALLTIME EXCEEDED**: SAM2 inference > 8h on P100 → reduce to a single CityFlowV2 scene (S02) for a partial-pipeline check; if that subset shows ≥+0.5pp gain, escalate to a multi-shift run. Otherwise mark as INFEASIBLE-AT-THIS-COST.

## Estimated Walltime

- SAM2 inference on ~99k detections at batch_size=8 on P100: ~3–5h (empirical SAM2-base throughput is ~5–10 images/s with box-prompt batching).
- TransReID re-extraction: ~30–45 min (same as existing Stage 2).
- 10b + 10c CPU chain: ~1–2h.
- **Total**: 5–8h, fits in a single overnight Kaggle GPU slot.

## Coder Handoff Checklist

The Coder agent that picks this up must:
1. Verify SAM2 weights upload to a Kaggle dataset and confirm dataset slug.
2. Build `notebooks/kaggle/14a_sam2_masked_stage2/` from a copy of the existing `10a_stages_0_2` Stage-2 cell as the starting template.
3. Verify the production-best 10c config from `outputs/10c_v15_logs.txt` or equivalent before running 10c.
4. Push **once**, watch for `not valid dataset sources` warnings, cancel and refix if seen.
5. After completion, append results to `docs/findings.md` and `docs/experiment-log.md` and update `.github/copilot-instructions.md` performance state.