---

# 14q Spec — TransReID ViT-B/16 CLIP @ 256² Higher-Resolution Retrain on VeRi-776

**Date**: 2026-05-10
**Author**: MTMC Planner
**Status**: PROPOSED — ready for Coder implementation
**Predecessor**: 14p3 (`MRKDaGods/14p3-...`) — TransReID ViT-L/14 CLIP @ 224² **FAILED** (post-rerank mAP=87.95%, R1=97.32% — ~3.6pp below 09v v17 ViT-B post-rerank 91.54%, ~1.0pp below R1 ceiling 98.33%). ViT-L/14 (304M params) overfit VeRi-776's 576 IDs / 37,778 images even with LLRD@0.65, BACKBONE_LR=1.5e-4, and 100 epochs.
**Account**: MRKDaGods (~18h quota remaining; spec budgeted at ≤12h for >6h buffer)
**Hardware**: Kaggle T4 (16GB), single GPU
**Goal**: Cross the WIN gate of post-rerank mAP ≥ 91.54% AND R1 ≥ 98.33% — beating both 09v v17 ViT-B/16 (89.97% / 98.33%) and CLIP-SENet v6 (91.54% post-rerank) using a single ViT-B/16 CLIP TransReID checkpoint, then port to CityFlowV2.

---

## 1. Decision: **Option B — Higher-Resolution ViT-B/16 CLIP TransReID Retrain at 256²**

### Why B beats A, D, E, F (and why ViT-L is closed)

| Option | Backbone / lever | Expected post-rerank mAP | T4 fit | Verdict |
|:------:|:----------------:|:------------------------:|:------:|:-------:|
| A. Extended ViT-B/16 @ 224² + stronger aug | Same as 09v v17, +60 epochs, +RandAugment/MixUp | 89.97% → ~90.5–91.0%; **near current ceiling** because 09v v17 already saturated the 224² recipe | ~6h | **FALLBACK** (used iff B blows budget) |
| **B. ViT-B/16 @ 256² extended** | Same backbone + same TransReID recipe + **higher resolution** + LLRD@0.65 (carried from 14p3 fix) + 160 epochs + mild RandAugment | **91.5–92.5%** — primary untried lever; resolution scaling is the standard SOTA path on fine-grained vehicle ReID | ~9–10h | ✅ **PICK** |
| C. ViT-B/16 @ 320² or 384² | Same as B, larger crops | Higher upside (~92–93%) but training cost ~2× B; may not fit budget | ~16h+ | ❌ Doesn't fit budget; defer to 14r if B WINs |
| D. CLIP-ReID two-stage (text-prompt) | ViT-B/16 + learnable ID prompts via CLIP text encoder, 60ep stage-1 + 120ep stage-2 | Paper claims ~90.5% mAP on VeRi-776, BUT 09v v17 already at 89.97% via eval tricks — narrow margin and adds open_clip dependency + new code path | ~10h + integration risk | ❌ High implementation cost, marginal upside |
| E. DINOv2 ViT-B/14 standalone on VeRi | New backbone | Unknown VeRi mAP; findings.md flags DINOv2 as inferior on cross-camera invariance and ViT-L on small-data; ViT-B may follow same overfit pattern | ~7h | ❌ Negative prior |
| F. IBN-Net-101 + Bag of Tricks | Classic Strong Baseline | Published ceiling ~85–87% on VeRi-776; no path to 91.54% | ~5h | ❌ Cannot reach WIN bar |

### Evidence-based rationale

1. **ViT-L overfits 37k images.** 14p3 confirmed: even with proper LLRD, lower LR, and AMP, the 304M-parameter backbone produced a strictly worse VeRi feature space than the 86M ViT-B/16 baseline (87.95% vs 91.54% post-rerank). The capacity axis is closed for VeRi-776 — the dataset is too small. The next axis must be **resolution**, not capacity.
2. **Resolution is the untried lever.** 09v v17's 89.97% mAP was achieved at 224². The findings.md note that "256² costs −0.12pp R1 vs 224 in 09v eval" is an **eval-time mismatch** result (checkpoint trained at 224, evaluated at 256 → upsampling artifact). It says nothing about training at 256². CLIP-SENet v6 was trained at **320²** and reached 91.54% post-rerank — direct in-house evidence that higher training resolution helps on this dataset and infrastructure.
3. **256² is the budget sweet spot.** ViT-B/16 at 224² has 196 patch tokens; at 256² it has 256 tokens (+30%); at 320² it has 400 tokens (+104%). 256² delivers most of the resolution benefit at ~1.4× the per-iter cost of 224², leaving budget for 160 epochs vs 14p3's 100 — the **second** untried lever.
4. **Re-uses the 14p3 scaffolding verbatim.** The 14p3 notebook already has correct LLRD, AMP fp16, PK sampler, JPM 4-group, SIE, BNNeck, CLIP-norm, RandomErasing, periodic eval, and the full concat-patch-flip + AQE(k=2) + rerank evaluator. Only the backbone constant, image_size constant, batch shape, and epoch count change. Minimal regression risk.
5. **CityFlowV2 port story is unchanged.** The deployed primary on CityFlowV2 is also ViT-B/16 CLIP TransReID. A 256²-trained ViT-B/16 with higher mAP is a direct drop-in replacement for the existing Stage-2 feature extractor; the production fusion config (`w_t=0.525, thr=0.48, aqe_k=2, fic_reg=0.5`) and 0.77936 MTMC IDF1 plateau both stay valid as a transfer baseline. (Note: 384px deployment was a CityFlowV2-specific cross-camera dead end; we are NOT proposing 384² for CityFlow Stage-2 — only for VeRi training and eval.)

### Why this is NOT a re-run of 09v's recipe at higher res

The 09v v17 / kernel 08 baseline was trained at 224² with NO LLRD, batch 96 on 2× T4 with DataParallel, 140 epochs, no RandAugment. 14q is single-T4 with LLRD@0.65 carried from 14p3, 256², 160 epochs, mild RandAugment, batch 64 (P=16, K=4). It is a distinct recipe combining the **proven backbone** (ViT-B/16 CLIP) with the **proven scaffolding** (14p3 LLRD + scheduler + eval) and the **two genuinely untried levers** (resolution=256², epochs=160, RandAugment).

---

## 2. Recipe (full hyperparameter spec)

**Backbone**: `vit_base_patch16_clip_224.openai` via `timm.create_model(..., pretrained=True, num_classes=0, img_size=256)`. Verify `model.patch_embed.num_patches == 256` (16×16) at `img_size=256`. Embed dim 768. 12 layers. ~86M params.

**Head** (unchanged from 14p3):
- SIE camera embedding, `num_cameras=20` for VeRi-776 (cameras 1–20).
- JPM 4 groups; for 256 patches, each group holds 64 patches (clean partition).
- BNNeck on 768-d global CLS feature. Linear classifier (no bias), 576 IDs.
- Triplet on raw global feature; CE-LS on BN feature; per-JPM-group CE-LS on BN'd group features. Loss weights `λ_global_ce = λ_global_tri = λ_jpm_ce = 1.0`.

**Optimizer**: AdamW, betas (0.9, 0.999), weight_decay=1e-4.
- **BACKBONE_LR = 3.5e-4** (revert to 08-baseline value — ViT-B is the right-sized model that the 09v ceiling was already established with, and LLRD@0.65 prevents the destabilization that bit ViT-L).
- **HEAD_LR = 3.5e-3** (BNNeck/classifier/JPM heads/SIE token).
- **LLRD_FACTOR = 0.65** carried verbatim from 14p3's `build_optimizer`. With 12 ViT-B blocks, deepest-block LR = 3.5e-4 × 0.65^12 ≈ 2.6e-6; stem LR ≈ 1.7e-6. Block-by-block decay protects early CLIP features.

**LR schedule**: 10-epoch linear warmup `1e-7 → base lr`; cosine decay over remaining 150 epochs to `min_lr=1e-6`.

**Epochs**: **160**. Eval mAP/R1 (base only, no rerank/AQE) every 10 epochs starting epoch 30. Save best-by-mAP and best-by-R1 + last.

**Image size**: **256×256** (must be divisible by 16; 256/16 = 16 patches/side = 256 tokens). Bicubic resize.

**Augmentations** (mild expansion of 14p3 stack):
- Resize 256² (bicubic)
- RandomHorizontalFlip(p=0.5)
- Pad(10) → RandomCrop(256)
- **NEW: RandAugment(num_ops=2, magnitude=9)** — a single conservative augmentation expansion vs 09v/14p3 baselines. Excludes color-jitter primitives that have caused training collapse historically (per findings.md: "Augmentation overhaul (color jitter, random rotation, AutoAugment) — caused training collapse"). Use the torchvision RandAugment with `num_magnitude_bins=31` and exclude `Color`, `Contrast`, `Saturation`, `Hue` from the op list — keep only geometric (TranslateX/Y, ShearX/Y, Rotate, Posterize, Sharpness, AutoContrast, Equalize, Solarize, Brightness). Apply BEFORE Normalize and AFTER RandomCrop.
- RandomErasing(p=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465])
- Normalize with **CLIP** mean/std `[0.48145466, 0.4578275, 0.40821073] / [0.26862954, 0.26130258, 0.27577711]`

NO MixUp, NO CutMix, NO color jitter, NO random rotation as a standalone aug — these are pre-registered dead ends.

**Sampler**: PK identity-balanced, **P=16, K=4, batch_size=64**. With 576 train IDs, this gives 36 batches/epoch ID coverage (exactly the 09v v17 / kernel 08 effective batch size, but on a single T4 instead of 2× T4 + DataParallel). **Memory check**: ViT-B @ 256², batch 64, AMP fp16 → est. ~9–10 GB activations + ~1 GB weights + ~1 GB optimizer state = ~12 GB on T4 16 GB. Headroom OK.
- **Backup if OOM**: drop to P=12, K=4, batch=48 (no accumulation needed — SIE+BNNeck+JPM are all LN/BN-on-features layers, not in-batch BN, so batch reduction is safe).

**Mixed precision**: `torch.cuda.amp` with `GradScaler`. fp16 forward/backward, fp32 master weights, identical to 14p3.

**Triplet margin**: 0.3. **Label smoothing**: ε=0.1.

**Eval (final, on best-mAP checkpoint)** — replicate 09v v17 eval pipeline at 256²:
1. Single-flip CLS 768-d, no rerank, no AQE — **the base mAP that drives MTMC IDF1**.
2. Single-flip CLS 768-d, AQE(k=2) + rerank(k1=80, k2=15, λ=0.2).
3. Concat-patch-flip 1536-d (CLS + GeM-pooled patch tokens), AQE(k=2) + rerank(k1=80, k2=15, λ=0.2) — **the headline post-rerank number**.
4. Concat-patch-flip 1536-d, AQE(k=3) + rerank(k1=80, k2=15, λ=0.2) — secondary check (09v v17's best-mAP row used k=3).

Report all four rows in `eval_results.json`; verdict gate is row 3 OR row 4, whichever is higher.

---

## 3. Pre-registered Verdict Bands

Primary metric: best of rows 3 / 4 from the eval table. Secondary metric: row 3 R1.

| Band | Post-rerank mAP | R1 | Action |
|:----:|:---------------:|:--:|:------:|
| **WIN** | ≥ 91.54% | ≥ 98.33% | Promote checkpoint to canonical VeRi baseline. Save `vehicle_transreid_vit_base_clip_veri776_256.pth` to weights dataset. Queue **14r CityFlowV2 port**: re-run Stage 2 feature extraction with the new checkpoint at the existing CityFlow image_size, then 10b → 10c with production fusion config. Goal: clear MTMC IDF1 ≥ 0.781 (current plateau 0.77936). |
| MARGINAL | 90.5% – 91.54% | ≥ 98.0% | Do NOT consume more MRKDaGods budget. Compare base mAP (row 1) to ViT-B/16's 85.14% — if base mAP ≥ 87% it is a real training-time gain, otherwise eval-time inflation. Document in findings.md and reconvene with planner before any 14r/14s. |
| **FAIL** | < 90.5% OR R1 < 98.0% | — | Close the higher-resolution ViT-B branch on VeRi. Pivot recommendation: **CLIP-ReID two-stage (Option D)** or **mark VeRi single-model SOTA recreation as exhausted at 91.54% (CLIP-SENet v6 ceiling)** and switch the paper angle from "we recreated VeRi SOTA" to "we systematically demonstrate the recreation ceiling for SOTA-class single-model VeRi ReID under reproducible compute". |

Pre-registration locks: the verdict is determined ONLY by the four rows of the final eval, computed exactly as specified above. No post-hoc rerank-grid sweeping. No swapping `λ` to find a better number. AQE k ∈ {2, 3} only. Rerank k1=80, k2=15, λ=0.2 only. This is the same eval contract 09v v17 used.

---

## 4. Compute Budget Breakdown (T4, AMP fp16)

| Phase | Time |
|:------|:----:|
| Boot + dataset attach + env install (`timm>=1.0`, `torch>=2.4`) | 6 min |
| Backbone init (`vit_base_patch16_clip_224.openai` weights ~340 MB; `img_size=256` interpolates pos_embed) | 2 min |
| Smoke S1–S4 from 14p2-fix.md (PK batch / LLRD application / feature variance / triplet+CE move under one step) | 4 min |
| Training: 160 epochs × ~3.0 min/epoch (37,778 imgs / batch 64 ≈ 590 iters; ViT-B fp16 fwd ~110ms + bwd ~220ms per batch on T4 at 256² ≈ 200ms incl. data loading) → ~3.0 min/epoch | **~8.0h** |
| Periodic eval at epochs {30,40,50,60,70,80,90,100,110,120,130,140,150,160}: 14 × ~2 min (base mAP/R1 only, no rerank, no AQE) | 28 min |
| Final eval (4 rows: row 1 base, rows 2–4 with AQE+rerank) on best-mAP checkpoint | 30 min |
| Save checkpoints + recipe.json + train_log.json + eval_results.json + 14q_summary.json | 5 min |
| **Total** | **~9h35m** |

**Hard cutoff**: 12h. Spec leaves ~2.5h slack within budget and ~6h slack within MRKDaGods quota. **Watchdog**: if any 10-epoch window exceeds 1.5× the per-epoch budget (>4.5 min/epoch sustained), abort training and dump current best checkpoint + run final eval on it.

---

## 5. Files & Outputs

- **Notebook**: `notebooks/kaggle/14q_veri_vit_b_256/14q_veri_vit_b_256.ipynb`
- **Build script**: `_build_14q_notebook.py` (use `json.load → modify → json.dump` pattern; `ensure_ascii=True`; each `source` line ends with `\n` except last). Strongly recommend cloning `_build_14p_notebook.py` as a starting point and changing only:
  - `MODEL_NAME = "vit_base_patch16_clip_224.openai"` (was `vit_large_patch14_clip_224.openai`)
  - `IMG_SIZE = 256` (was 224)
  - `PATCH_SIZE = 16` (was 14)
  - `BACKBONE_LR = 3.5e-4` (was 1.5e-4)
  - `EPOCHS = 160` (was 100)
  - `P_IDS = 16, K_INSTANCES = 4, BATCH_SIZE = 64` (was 8/4/32)
  - **NEW**: RandAugment transform inserted between RandomCrop and RandomErasing, with the geometric-only op list specified above
  - Output filenames: `transreid_vit_b16_clip_veri776_256_*.pth`
- **Kernel slug**: `MRKDaGods/14q-veri-vit-b-256-train`
- **kernel-metadata.json**:
  ```json
  {
    "id": "MRKDaGods/14q-veri-vit-b-256-train",
    "title": "14q VeRi ViT-B/16 CLIP @ 256 TransReID Train",
    "code_file": "14q_veri_vit_b_256.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": true,
    "enable_gpu": true,
    "enable_internet": true,
    "machine_shape": "NvidiaTeslaT4",
    "dataset_sources": [
      "abhyudaya12/veri-vehicle-re-identification-dataset"
    ],
    "competition_sources": [],
    "kernel_sources": [],
    "model_sources": []
  }
  ```
- **Auth**: `$env:KAGGLE_API_TOKEN = (Get-Content $HOME/.kaggle/MRKDaGods__access_token -Raw).Trim()` (note double underscore in filename per repo memory).
- **Outputs to save** in `/kaggle/working/`:
  - `transreid_vit_b16_clip_veri776_256_best_mAP.pth` (state_dict only)
  - `transreid_vit_b16_clip_veri776_256_best_R1.pth`
  - `transreid_vit_b16_clip_veri776_256_last.pth`
  - `train_log.json`
  - `eval_results.json` (4 rows)
  - `recipe.json`
  - `14q_summary.json` (verdict band, primary mAP, primary R1, walltime)
- **Local mirror** after run: `outputs/14q_v1_summary/14q_summary.json`

---

## 6. Hard Constraints (do NOT re-test these dead ends)

Citing `docs/findings.md`:

- ❌ **ViT-L/14 CLIP TransReID on VeRi** (14p3 FAIL — 87.95% post-rerank). Do NOT retry larger-than-ViT-B backbones on VeRi-776 in this branch.
- ❌ **OSNet-IBN-x1.0 from-scratch on CityFlowV2** (14m FAIL — 23.80% mAP).
- ❌ **SGD on transformers** (catastrophic; 09 ablation produced 30.27% mAP).
- ❌ **CircleLoss** (caused `inf` loss in two independent runs; 09 ablation v1 → 18.45%; 09l v1 → 20.36%).
- ❌ **CLIP-SENet @ 256² / P=16** (v7 → −0.98pp mAP regression vs v6 320²). Do not interpret 14q's 256² as evidence about CLIP-SENet's resolution sensitivity — different architectures.
- ❌ **Color jitter, random rotation as standalone aug, AutoAugment** (caused training collapse in 09 ablations). The RandAugment op list above explicitly excludes color primitives.
- ❌ **MixUp / CutMix on ReID training** — historically degrades hardest-positive triplet mining because the synthetic positive pairs no longer share an identity. Not in this recipe.
- ❌ **k-reciprocal rerank as deployed feature stream** — only as eval-time post-processing.
- ❌ **DMT camera-aware training** (-1.4pp single-model on CityFlow; 09g → 43.8% mAP).
- ❌ **384px ViT deployment on CityFlowV2** — this is for VeRi training only; CityFlow Stage-2 stays at the existing image_size in 14r.
- ❌ **Conflating single-cam VeRi mAP with cross-cam MTMC IDF1**. 14q's WIN gate is VeRi-only. 14r is the CityFlow port and has its own gate (≥ 0.781 MTMC IDF1) computed in production fusion config.

---

## 7. Risks + Rollback

### Fast-fail abort signals (Coder MUST monitor and halt)

1. **OOM at smoke S2 batch=64**: rebuild notebook with `P_IDS=12, K_INSTANCES=4, BATCH_SIZE=48`, push **once** more. Do NOT auto-retry beyond one fallback.
2. **Smoke S4 fails (loss does not move under one optimizer step)**: do NOT push. Report and stop. This indicates an optimizer-wiring regression vs 14p2-fix.
3. **Epoch-1 triplet > 0.305 AND CE > 6.50**: training has not started. Indicates the LLRD param-group construction missed BNNeck/classifier/JPM heads (head_lr group is empty). Coder should validate via S2 before the long run; if epoch-1 already shows the symptom, abort and inspect optimizer.param_groups.
4. **Per-epoch wall-clock > 4.5 min sustained for 10 epochs**: abort training. Either dataloader is the bottleneck (raise `num_workers`) or memory is paging — drop to batch 48 and resume.
5. **Periodic eval at epoch 80 base mAP < 75%**: training is on an under-converged trajectory that 80 more epochs cannot rescue. Save current best, stop training, run final eval on what we have, mark MARGINAL/FAIL accordingly.
6. **Kernel push triggers `not valid dataset sources` warning**: cancel via `kaggle kernels cancel MRKDaGods/14q-veri-vit-b-256-train`. If cancel CLI unavailable, poll `kaggle kernels status` every 60s until `cancelled`/`error`/`complete` per repo Kaggle Push Safety Rules. Do NOT auto-repush.

### Train_log.json signs to watch (mid-run sanity)

- Epoch 1 triplet ∈ [0.30, 0.40], CE ∈ [6.30, 6.50] — normal init.
- Epoch 30 triplet ∈ [0.20, 0.28], CE ∈ [3.5, 4.5] — normal mid-training.
- Epoch 30 base mAP ≥ 70% — normal trajectory toward 88%+ base by epoch 160.
- Epoch 80 base mAP ≥ 82% — on track for WIN.
- Epoch 160 base mAP target: ≥ 88% (matches 09v v17's eval-time-only ceiling and exceeds it via training resolution).

### Rollback policy

- If FAIL: keep 09v v17 / `vehicle_transreid_vit_base_veri776.pth` as the canonical VeRi-776 checkpoint. CityFlowV2 production stays at 0.77936 plateau. Update findings.md to record that ViT-B @ 256² training did not cross 91.54% post-rerank, closing the resolution-axis lever for ViT-B/16.
- If MARGINAL: do not deploy to CityFlowV2. Reconvene with planner to decide between (a) escalating to 320² (Option C, 14r-alt) at higher compute risk, or (b) pivoting to CLIP-ReID two-stage (Option D), or (c) declaring single-model VeRi recreation exhausted and pivoting paper angle.
- If WIN: immediately queue 14r — Stage-2 CityFlowV2 feature extraction with the new checkpoint, then 10b → 10c with production fusion (`w_t=0.525, thr=0.48, aqe_k=2, fic_reg=0.5`). 14r is a separate spec with its own gate.

---

## 8. Coder Handoff (single-push, no auto-retry)

**Phase 0 — Preflight**:
1. Activate `.venv` (Python 3.11.9): `.\.venv\Scripts\activate`.
2. Verify `~/.kaggle/MRKDaGods__access_token` exists and contains a `KGAT_` token.
3. Check active MRKDaGods GPU sessions: `$env:KAGGLE_API_TOKEN = (Get-Content $HOME/.kaggle/MRKDaGods__access_token -Raw).Trim(); kaggle kernels list --mine`. If ≥1 GPU kernel `running`, STOP.

**Phase 1 — Build notebook**: clone `_build_14p_notebook.py` to `_build_14q_notebook.py`, apply the constant changes from §5, add the geometric-only RandAugment transform between RandomCrop and RandomErasing, change all output filenames and the kernel id. Build the notebook via `json.load → modify → json.dump`, `ensure_ascii=True`, each source line ends with `\n` except last. Verify on-disk after build.

**Phase 2 — CPU smoke**: run a CPU smoke that exercises the patched dataset + sampler + model + optimizer with a 32-image fake-VeRi loader. Required: smoke S1 (PK batch composition), S2 (LLRD application, with `num_blocks=12` for ViT-B), S3 (per-row feature variance), S4 (triplet+CE move under one optimizer step). All four MUST PASS before push.

**Phase 3 — Single push**: `kaggle kernels push -p notebooks/kaggle/14q_veri_vit_b_256/`. Inspect startup log for `not valid dataset sources` warnings; cancel and stop if present. Do NOT auto-repush.

**Phase 4 — Monitor**: poll `kaggle kernels status MRKDaGods/14q-veri-vit-b-256-train` and tail logs via `python scripts/kaggle_logs.py 14q-veri-vit-b-256-train --tail 200`. On completion, download outputs to `outputs/14q_v1_summary/`. Do not run any post-hoc rerank-grid sweeps; the eval contract is fixed in §2.

**Phase 5 — Update docs**: append a `## 14q TransReID ViT-B/16 CLIP @ 256² on VeRi-776` section to `docs/findings.md` with verdict, exact numbers in all 4 eval rows, walltime, and 14r gating decision. Append the 14q row to `docs/experiment-log.md`.
