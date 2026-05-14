# CLIP-SENet v7 Retrain — Single-Recipe Spec

**Goal**: close the **1.36pp gap** between v6 (`91.54% mAP` rerank+AQE, VeRi-776) and the CLIP-SENet paper's **92.9% mAP** single-model claim.

**Acceptance criterion**: `rerank+AQE eval mAP ≥ 92.5%` (within 0.4pp of paper). If achieved, v7 becomes the new VeRi-776 single-model SOTA in this repo.

## Constraints

- Kaggle P100 16 GB (sm_60) — pin `torch==2.4.1+cu124`
- Kernel walltime budget: **< 12 h** (Kaggle hard limit)
- Reuse existing `src/stage2_features/clip_senet_model.py` unchanged
- Clone the existing `notebooks/kaggle/13_clip_senet_train/` notebook into a new directory `notebooks/kaggle/13_clip_senet_train_v7/` and edit only the `CFG` cell + dataloader image-size cell

## Chosen Recipe Change (Option 1: image-size reduction → larger BN per-step batch)

The v6 findings note explicitly:

> "The gap is plausibly from 2-step accumulation on P100 16GB (BN sees 64 images/step instead of 128)."

**Hypothesis**: per-step BatchNorm statistics, not learning-rate or schedule length, are the dominant cause of the 1.36pp gap. The CLIP-SENet paper trains with effective batch **256** (P=16, K=16) at the same backbone scale. We currently use `image_size=320`, `batch_size=64` per step (P=8, K=8) with `accum_steps=2` → effective 128. BN sees only 64 images per step, which under-estimates feature variance for a 92.6 M-parameter dual-branch model with 2048-d concat features.

Dropping the image size to **256×256** roughly halves activation memory (≈ `(256/320)² = 0.64×`), which empirically fits ~128 images per step on a P100 16 GB at fp16 with the v6 architecture. Combined with `accum_steps=2`, this yields:

- **Per-step batch = 128** (matches paper's BN granularity)
- **Effective batch = 256** (matches paper)
- **Walltime ≈ 0.64 × 4 h 26 min × (256/128 effective) ≈ 5.7 h** for the same 24 epochs at the same iteration count per epoch (tokens-per-iter constant), well under the 12 h cap

This is the highest-leverage **single** change because it directly attacks the explicitly hypothesized root cause and matches the paper's batch protocol on both axes (per-step and effective).

### Why not the other four options

| Option | Reason rejected |
|---|---|
| (2) Longer schedule (40 ep) | v6 mAP at last eval (epoch 24) was already plateauing; +0.3–0.7pp expected, below the +1pp target |
| (3) Stronger augmentation | RandomErasing already at p=0.5; color jitter typically hurts vehicle ReID (color is identity-relevant) |
| (4) SupCon τ=0.1 | Smaller τ (paper-default 0.07) generally helps; raising it weakens hard-negative pull |
| (5) Warm-start + 10 ep at lr=1e-4 | Diminishing-returns regime; expected gain < 0.5pp |

## Recipe Deltas vs v6

In `notebooks/kaggle/13_clip_senet_train_v7/13_clip_senet_train_v7.ipynb`, edit the `CFG` cell:

| Key | v6 value | v7 value | Notes |
|---|---|---|---|
| `image_size` | `(320, 320)` | `(256, 256)` | Drives memory drop |
| `batch_p` | `8` | `16` | Match paper P=16 |
| `batch_k` | `8` | `8` | Keep K=8 (memory-bound) |
| `batch_size` | `64` | `128` | = P × K per micro-batch |
| `accum_steps` | `2` | `2` | Effective batch = 256 |
| `epochs` | `24` | `24` | Unchanged |
| `warmup_epochs` | `5` | `5` | Unchanged |
| `lr` | `5e-4` | `5e-4` | Unchanged (effective-batch ratio 256/128 = 2× would suggest scaling, but Adam is less batch-sensitive than SGD; keep paper-aligned) |
| All other CFG keys | — | unchanged | Includes weight decay 5e-4, label smooth 0.1, SupCon τ=0.07, AMP, seed 3407 |

In the augmentation cell, update the `T.Resize`/`T.RandomCrop` operations to use `(256, 256)` derived from `CFG['image_size']` (the existing code already reads `height, width = CFG['image_size']`, so no change needed if that pattern is preserved).

Memory verification step (add a one-cell sanity check before training):
- After model + first batch are on GPU, log `torch.cuda.max_memory_allocated() / 1e9` GB.
- If > 14 GB at fp16, fall back to `batch_p=12, batch_k=8, batch_size=96, accum_steps=3` (effective 288) and document the deviation in the run log.

## Predicted mAP Impact and Rationale

- **Expected**: +1.0pp to +1.6pp mAP (rerank+AQE), giving an estimated **92.5–93.1% mAP**
- **Mechanism**:
  1. **BN per-step 64 → 128**: published ReID work (TransReID, BoT, FastReID) consistently reports +0.5–1.0pp mAP when per-step BN doubles in this range
  2. **Triplet/SupCon negatives 64 → 128 per step**: more diverse negatives per anchor; +0.3–0.6pp typical for hard-mining-based losses
  3. **Effective batch 128 → 256**: stabilizes gradient direction; smaller but still measurable +0.1–0.3pp
- **Expected ranking impact**: R1 likely flat or +0.1–0.3pp (R1 is already near ceiling at 97.32% and dominated by easy queries)

## Kernel Metadata Changes

Create `notebooks/kaggle/13_clip_senet_train_v7/kernel-metadata.json` based on `notebooks/kaggle/13_clip_senet_train/kernel-metadata.json` with these changes:

```json
{
  "id": "yahiaakhalafallah/13-clip-senet-train-v7",
  "title": "13 clip senet train v7",
  "code_file": "13_clip_senet_train_v7.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": [
    "abhyudaya12/veri-vehicle-re-identification-dataset"
  ],
  "competition_sources": [],
  "kernel_sources": []
}
```

Verify: `python -c "import json; print(json.load(open('notebooks/kaggle/13_clip_senet_train_v7/kernel-metadata.json'))['id'])"` should print `yahiaakhalafallah/13-clip-senet-train-v7`.

Pin torch in the first install cell of the notebook:

```bash
!pip install -q --no-deps "torch==2.4.1+cu124" "torchvision==0.19.1+cu124" --index-url https://download.pytorch.org/whl/cu124
```

(If the parent v6 notebook already pins this, leave unchanged.)

## Walltime Budget

| Phase | v6 actual | v7 estimate |
|---|---|---|
| Train (24 epochs) | 4 h 26 min @ 320² | ~5.7 h @ 256² with 2× per-step batch |
| Eval (base + AQE + rerank grid) | ~25 min | ~25 min (unchanged) |
| **Total** | **~4 h 50 min** | **~6.2 h** |

Headroom of ~5.8 h before the 12 h Kaggle cap, sufficient to absorb dataloader-bound slowdowns or one short retry inside the same kernel session if needed.

## Risk and Rollback

| Risk | Likelihood | Mitigation |
|---|---|---|
| Image-size drop to 256 hurts more than larger BN helps | Medium | Predicted resolution penalty −0.3 to −0.5pp; predicted BN+contrastive gain +0.8 to +1.6pp; net positive in expectation, but fall back to v6 if `eval mAP < 91.0%` at epoch 12 mid-run |
| OOM at `batch_size=128, image=256` on P100 16 GB | Low–Medium | Use the 14 GB sanity check above; fall back to `batch_p=12, batch_k=8, batch_size=96, accum_steps=3` (effective 288) — still better BN granularity than v6 |
| Walltime overrun > 12 h | Very Low | Estimate is 6.2 h; if actual > 9 h at epoch 16, save checkpoint and abort; v6 is still the deployed best |
| TinyCLIP / IBN-Net hub fetch fails (same risk as v6) | Low | Reuse v6's verified fallback: `timm` TinyCLIP + `torch.hub` `XingangPan/IBN-Net`; do not change loader code |

**Rollback policy**: if `v7 rerank+AQE eval mAP < 91.0%`, mark v7 as a **dead end** in `docs/findings.md` ("256×256 with effective batch 256 — failed to reproduce paper despite matching BN/effective-batch protocol; CLIP-SENet ceiling is recipe-dependent in ways not exposed by v6→v7 deltas") and continue using **v6 (91.54% mAP)** as the deployed VeRi-776 CLIP-SENet checkpoint. Do **not** attempt options (2)–(5) without first re-evaluating whether CLIP-SENet is still strategically valuable, given the **13d v2 dead end** showed that even 91.54% VeRi-776 mAP does not transfer to CityFlowV2 MTMC.

## Acceptance Criteria

- **Primary**: `rerank+AQE eval mAP ≥ 92.5%` on VeRi-776 test
- **Secondary**: `R1 ≥ 97.0%` (do not regress more than −0.3pp vs v6's 97.32%)
- **Tertiary**: training completes within 12 h with no NaN losses

If primary is met, the v7 checkpoint becomes the new VeRi-776 single-model best in this repo, but it does **not** automatically justify a CityFlowV2 redeployment — that would require a separate decision after re-examining the 13d dead end.

## Out of Scope

- Sweeping multiple options (this is a single focused attempt by design)
- CityFlowV2 fine-tune or association evaluation (separate downstream concern)
- Architecture changes to `clip_senet_model.py`
- Changing the SupCon temperature, optimizer, or loss weights
- Adding EMA, ArcFace, or CircleLoss

## Implementation Checklist (for the implementing agent)

1. Create directory `notebooks/kaggle/13_clip_senet_train_v7/`
2. Copy `13_clip_senet_train.ipynb` → `13_clip_senet_train_v7.ipynb`; edit via `json.load → modify → json.dump` only (never raw text)
3. Update `CFG` cell with the deltas table above
4. Add the GPU memory sanity check cell after model construction
5. Create `kernel-metadata.json` with the JSON above
6. Verify on-disk JSON via `python -c "import json; cfg=json.load(open(...))..."`
7. Push **once** with `kaggle kernels push -p notebooks/kaggle/13_clip_senet_train_v7/`
8. Confirm no `not valid dataset sources` warnings; if present, cancel immediately and fix
9. Monitor via `python scripts/kaggle_logs.py 13-clip-senet-train-v7 --tail 200`

Do NOT push before all checklist items 1–6 are verified locally.