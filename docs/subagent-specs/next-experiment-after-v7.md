# Next Experiment After CLIP-SENet v7 Regression

**Date**: 2026-05-07
**Author**: MTMC Planner
**Status**: Proposed — awaiting Coder execution
**Predecessor results**: 13 v6 (82.34% mAP / 91.54% post-rerank, canonical), 13 v7 (81.36% mAP, REGRESSION/DEAD END), 13d v2 (CLIP-SENet × CityFlowV2 cross-domain fusion DEAD END)

## TL;DR

**Chosen approach: Fine-tune CLIP-SENet v6 on CityFlowV2 train split (in-domain transfer), then re-test score-level fusion with TransReID + DINOv2.**

Rationale in one line: every prior fusion failure (DINOv2, OSNet, CLIP-SENet v6, LAION-2B CLIP) confirms that *cross-camera invariance is determined by training methodology / in-domain supervision*, not single-distribution mAP — so the only fusion stream we have **never tried** is one that is itself trained on CityFlowV2 cross-camera labels with a different backbone family from TransReID.

## Goal

Produce a **secondary ReID feature stream** that is genuinely complementary to TransReID ViT-B/16 CLIP **on CityFlowV2 cross-camera matching** (not just on a single-camera mAP benchmark), so that score-level fusion finally yields a *positive* delta vs the **0.7703** TransReID + DINOv2 baseline.

Target: MTMC IDF1 ∈ [0.78, 0.80] (closing 1–3pp of the 7.83pp SOTA gap).

## Hypothesis

Per `docs/findings.md`, three fusion partners have failed:
1. DINOv2 ViT-L/14 (86.79% CityFlowV2 mAP, but **−3.1pp** standalone MTMC IDF1) — wrong training methodology.
2. CLIP-SENet v6 (91.54% VeRi-776 mAP, but **0.6855** standalone MTMC IDF1) — wrong domain.
3. ResNet101-IBN-a 09d (52.77% CityFlowV2 mAP) — too weak.

What is missing: a backbone trained **with the TransReID-style cross-camera recipe on CityFlowV2 itself**, but with a different inductive bias (CNN+attention hybrid vs pure ViT). CLIP-SENet v6 fine-tuned on CityFlowV2 supplies exactly that:
- Strong VeRi-776 init (92.6M params, 91.54% post-rerank) gives a much better warm-start than ImageNet (which only got R101-IBN to 52.77%).
- ResNet101-IBN-a + AFEM injects CNN locality + channel-attention features that are *architecturally distinct* from ViT-B/16 patches.
- In-domain CityFlowV2 supervision finally gives the secondary model the cross-camera invariance signal that VeRi-776 lacks.

If the in-domain fine-tune lands at ≥65% CityFlowV2 mAP (the established ensemble threshold), and the resulting features are genuinely uncorrelated with TransReID (different backbone family + different pre-training corpus), score-level fusion has a credible path to net-positive impact for the first time since the v80 OSNet result.

## Why not the alternatives

| Candidate | Verdict | Reason |
|---|---|---|
| (a) **Fine-tune CLIP-SENet on CityFlowV2 train** | **CHOSEN** | Highest EV × feasibility — see hypothesis. Reuses 13 training infra. |
| (b) Train CLIP-SENet on VehicleX synthetic + VeRi-776 | Rejected (this round) | VehicleX availability/licensing on Kaggle unverified, and synthetic→real still leaves a CityFlowV2 domain gap; (a) attacks the gap directly. |
| (c) Different secondary (Swin-Vehicle / DINOv2-vehicle finetune) | Rejected | No verified public checkpoint matches our compute budget; DINOv2 already failed at the standalone-MTMC test. |
| (d) Calibration MLP on frozen TransReID features using CityFlow GT pairs | Defer | Lower upper bound (single-stream re-projection cannot exceed the underlying feature manifold); good follow-up if (a) plateaus. |
| (e) GNN edge classifier for stage4 association | Reject (now) | `findings.md` shows association tuning is exhausted within 0.3pp; the bottleneck is feature quality, not the graph algorithm. |
| (f) SAM2 foreground masking before ReID | Defer | Plausible but high engineering cost; appropriate after exhausting backbone-side wins. |
| (g) Graph multi-view tracking | Reject | Designed for overlapping-camera person datasets; CityFlowV2 cameras are non-overlapping. |

## Concrete Implementation Plan

### Stage A — In-domain fine-tune ("13f")

**New Kaggle kernel**: `notebooks/kaggle/13f_clip_senet_cityflow_finetune/` (mirror the layout of the existing 13 kernel).

**Build script**: `_build_13f_notebook.py` (clone of the v6 build script with the diffs below).

**Inputs**:
- Dataset: `cityflowv2-vehicle-reid-crops` (already used by 09v / 09d) — train split only (S03/S04/S05 scenarios, ~128 IDs).
- Pretrained checkpoint: `13-clip-senet-train` v6 best (`outputs/13_v6/best.pth`, mAP=82.34 / R1=96.54).
- Tokenizer/backbone weights identical to v6 (TinyCLIP timm fallback documented in v6 findings).

**Training recipe** (deltas vs v6):
- `image_size: 320` (KEEP — v7 proved 256 hurts).
- `P=8, K=8`, `batch_size=128`, `accum_steps=2` (KEEP).
- Optimizer: Adam, **lr=1e-4** (10× smaller than v6's 5e-4 for fine-tune stability on small dataset).
- Schedule: cosine, **12 epochs** (half of v6) + 2-epoch warmup — small dataset, avoid overfit.
- Losses: CE label-smoothing 0.1 + SupCon τ=0.07 (KEEP).
- Augmentation: KEEP v6 stack (RandomErasing + HFlip + Pad + RandomCrop).
- Init: load v6 backbone+head; reset only the final classifier FC to match CityFlowV2 ID count.
- AMP fp16 (KEEP).

**Eval kernel**: `13g-clip-senet-cityflow-eval` (clone of `13e-v7-clip-senet-eval` but pointing at CityFlowV2 query/gallery, not VeRi-776).

**Acceptance gate (Stage A)**:
- Single-camera CityFlowV2 mAP ≥ **65%**. If <65%, declare DEAD END (insufficient ensemble strength) and pivot to candidate (d).
- Single-camera CityFlowV2 R1 ≥ **78%**.

### Stage B — Score-level fusion sweep ("13h")

If Stage A passes, extract per-detection features on the CityFlowV2 eval split and re-run the **13d** fusion harness with a third stream:

- Reuse `outputs/13d_v2/` infrastructure (tracklet alignment, mean-pool per-tracklet, `feature_dim=2048`).
- 3-way score-level fusion: `w_primary` (TransReID), `w_dinov2`, `w_cs_ft` (new fine-tuned CLIP-SENet).
- Search grid (small, focused):
  - Anchor at the current 0.7703 optimum (`w_primary=0.4, w_dinov2=0.6`).
  - Sweep `w_cs_ft ∈ {0.10, 0.15, 0.20, 0.25, 0.30}` with primary/dinov2 rescaled by `(1 − w_cs_ft)`.
  - 5 runs total — cheap.

**Acceptance gate (Stage B)**:
- MTMC IDF1 > **0.7703** at any `w_cs_ft`. If yes → SUCCESS, log result, run a finer sweep ±0.05 around the winner.
- If all 5 weights regress → DEAD END (in-domain fine-tune still insufficient as a fusion partner; document and stop).

### Files to create
- `_build_13f_notebook.py` — at repo root.
- `notebooks/kaggle/13f_clip_senet_cityflow_finetune/kernel-metadata.json`
- `notebooks/kaggle/13f_clip_senet_cityflow_finetune/13f_clip_senet_cityflow_finetune.ipynb`
- `notebooks/kaggle/13g_clip_senet_cityflow_eval/kernel-metadata.json`
- `notebooks/kaggle/13g_clip_senet_cityflow_eval/13g_clip_senet_cityflow_eval.ipynb`
- `notebooks/kaggle/13h_clip_senet_cityflow_fusion/kernel-metadata.json`
- `notebooks/kaggle/13h_clip_senet_cityflow_fusion/13h_clip_senet_cityflow_fusion.ipynb` (clone of 13d v2 with 3rd stream)

### Files to modify
- None outside the new kernels (no `src/` changes — fine-tune is self-contained on Kaggle; fusion harness reuses 13d code paths).
- After completion: append results to `docs/findings.md` and `docs/experiment-log.md`.

## Expected Impact

**Lower bound**: 0.7703 (no improvement; secondary still too weak or correlated). Probability ≈ 40% based on prior fusion failures.

**Realistic mid case**: +0.5 to +1.5pp MTMC IDF1 → **0.775 – 0.785**. Probability ≈ 45%. Justification: in-domain TransReID alone already hit 0.775+ standalone; an architecturally distinct in-domain partner has been the missing fusion ingredient since the v80 OSNet loss.

**Upper bound**: +2 to +3pp MTMC IDF1 → **0.79 – 0.80**, recovering ground toward the historical v80 0.784 result without depending on the lost OSNet checkpoint. Probability ≈ 15%.

**Tail risk**: catastrophic overfit on the small CityFlowV2 train split (128 IDs vs 92.6M params) → fine-tune mAP <50% → Stage A fails the gate cleanly, no Stage B time wasted.

## Risks

1. **Overfitting on small dataset.** Mitigated by 10× lower LR, halved schedule, and the strong VeRi-776 warm-start (the 09d result shows ~52% mAP is achievable from ImageNet init alone, so a far stronger init starting at near-zero training loss should plateau at a higher ceiling).
2. **Fine-tune destroys VeRi-776 generalization without gaining CityFlowV2 strength.** Mitigated by Stage A acceptance gate at 65% mAP; if not met, cleanly declared dead end.
3. **Cross-camera correlation with TransReID.** Both models share CLIP-family pre-training (CLIP-SENet uses TinyCLIP semantic branch, TransReID uses ViT-B/16 CLIP). If features end up highly correlated, fusion will not help. Mitigated by the ResNet101-IBN-a appearance branch, which dominates the 2048-d output and is architecturally distinct from ViT.
4. **Kaggle GPU budget.** Stage A is one P100-hours run (~2–3h estimated, halved schedule); Stage B is CPU-only fusion sweep. Total well within budget.

## Rollback Plan

- If Stage A fails the 65% mAP gate: stop. Mark candidate (a) as DEAD END in `findings.md`. Promote candidate (d) — train a small calibration MLP on top of frozen TransReID features using CityFlowV2 train cross-camera GT pairs — to the next experiment slot.
- If Stage A passes but Stage B regresses at all 5 weights: stop. Log the in-domain mAP win as a standalone result (still useful as a paper data point on cross-camera invariance vs single-cam mAP), and pivot to candidate (d).
- No `src/` or config changes are made by this experiment, so rollback is purely "delete the new kernel folders" — zero risk to the production pipeline.

## Stop Criteria

- **SUCCESS**: any `w_cs_ft` in Stage B yields MTMC IDF1 ≥ 0.7750 (≥+0.5pp). Run finer sweep, declare new best, update `findings.md`.
- **PARTIAL**: Stage A passes (≥65% CityFlowV2 mAP) but Stage B never beats 0.7703. Log; pivot to (d).
- **FAILURE**: Stage A fine-tune does not reach 65% mAP within 12 epochs. Declare DEAD END; pivot to (d).
- **HARD STOP**: total Kaggle GPU time exceeds 6 P100-hours across Stage A retries — abandon this candidate regardless of partial gains.

## Next-After-This

If this experiment succeeds, the natural follow-up is to scale: train a third complementary backbone (Swin or ConvNeXt) on CityFlowV2 with the same recipe and aim for a 4-way fusion (primary + DINOv2 + CLIP-SENet-FT + new). If it fails or is partial, candidate (d) — the calibration MLP — becomes the next experiment, followed by candidate (f) SAM2 masking if (d) also stalls.