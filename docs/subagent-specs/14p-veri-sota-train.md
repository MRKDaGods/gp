---

# 14p Spec — TransReID ViT-L/14 CLIP on VeRi-776 (SOTA Recreation Run)

**Date**: 2026-05-08
**Author**: MTMC Planner
**Status**: PROPOSED — ready for Coder implementation
**Account**: gumfreddy (~15h GPU quota; reserve MRKDaGods 30h for follow-up)
**Hardware**: Kaggle T4 (16GB), single GPU
**Goal**: Beat the current best VeRi-776 single-model ReID checkpoint and produce a portable backbone for CityFlowV2 transfer.

## Decision: **Option B — TransReID ViT-L/14 CLIP** on VeRi-776

### Why B over A/C/D

| Option | Backbone | Recipe | Single-run upside | T4 budget fit | Verdict |
|:------:|:--------:|:------:|:-----------------:|:-------------:|:-------:|
| A. CLIP-SENet v8 | ResNet101-IBN-a + TinyCLIP (92.6M) | v6 + longer schedule + better augs | Low — v7 retrain at 256² already regressed (-0.98pp); the recipe is sensitive and the dominant residual gap to paper claim 92.9% mAP comes from missing TinyCLIP-ViT-40M-32 weights and full-batch BN, neither of which is fixable on T4 16GB | Fits (~6–8h) | ❌ Saturated |
| **B. TransReID ViT-L/14 CLIP** | `vit_large_patch14_clip_224.openai` (~304M) + SIE + JPM | Exact 08-kernel recipe scaled to L/14 | **Highest** — ViT-B/16 CLIP already at R1=98.33% / mAP=89.97% (09v v17). Same family + 2.6× capacity + same proven CLIP pretraining → expected mAP 91–93%, R1 98.5–99% | **Tight, fits** at 224² with AMP fp16 (~12–14h for 100 epochs) | ✅ **PICK** |
| C. EVA-02-L/14 | EVA-02-L (~305M) + EVA pretrain | New pretraining family | Medium — but unfamiliar recipe; original estimate 24–36h | ❌ Doesn't fit | Defer |
| D. DINOv2 ViT-L/14 on VeRi-776 | `vit_large_patch14_dinov2.lvd142m` | TransReID recipe | Medium — same backbone size as B, but DINOv2 already underperformed on CityFlowV2 (mAP=86.79% but MTMC IDF1=0.744, **−3.1pp**), strongly suggesting DINOv2's self-supervised features lack the cross-camera invariance that CLIP image-text contrastive supplies | Fits (~12h) | ❌ Negative prior on cross-camera transferability |

### Evidence-based rationale (citations from `docs/findings.md`)

1. **CLIP pretraining is the proven cross-camera invariance signal.** DINOv2 ViT-L/14 reached 86.79% CityFlowV2 mAP / 96.15% R1 (09s v1) — a +6.65pp mAP jump — yet its MTMC IDF1 = 0.744 was **−3.1pp below** the deployed CLIP-based 0.7703. The "training methodology for cross-camera invariance" finding is decisive: B keeps CLIP, D abandons it. This is the canonical reason to pick CLIP over DINOv2 for any vehicle ReID checkpoint that we intend to port to CityFlowV2.
2. **TransReID ViT-B/16 CLIP is already 1pp from SOTA.** 09v v17 best R1=98.33% / mAP=89.97% (concat-patch-flip AQE+rerank, k1=80,k2=15,λ=0.2), joint optimum 98.15% / 89.71%. Published TransReID paper VeRi-776: ~82% mAP / 97.1% R1 (no rerank). Our base mAP at 224² is well above paper. The remaining gap to 92–93% is plausibly closed by going to ViT-L/14 with the same recipe.
3. **CLIP-SENet v6 vs v7 closes Option A.** v6 = 82.34% / 96.54% (320², P=8/K=8). v7 retrain at 256²/P=16 regressed to 81.36% / 95.71% (`-0.98pp mAP, -0.83pp R1`). The CLIP-SENet recipe is sensitive and we have already explored its neighborhood. Closing the 1.36pp gap to paper-claimed 92.9% requires unobtainable TinyCLIP-ViT-40M-32 weights and full batch=128 BN, neither of which is fixable on T4 16GB.
4. **The 09v v17 R1 ceiling at 98.33% is reached by eval-time techniques.** Any further single-model gain on VeRi-776 must come from training-time changes — i.e., a stronger backbone trained with the same recipe.
5. **CityFlowV2 portability requirement.** The current deployed primary on CityFlowV2 is also TransReID ViT-B/16 CLIP. A ViT-L/14 CLIP TransReID checkpoint is the natural, drop-in upgrade for both Stage 2 feature extraction and the existing 09v concat-patch-flip eval recipe — minimal integration risk.

## Target Metric and Verdict Bands

Primary metric: **VeRi-776 mAP** with concat-patch-flip features at 224², single-flip TTA, AQE(k=2)+rerank(k1=80, k2=15, λ=0.2). Secondary metric: R1 (joint-optimum row).

| Band | mAP | R1 | Action |
|:----:|:---:|:--:|:------:|
| **WIN** | ≥ 91.5% | ≥ 98.6% | Promote checkpoint to canonical VeRi baseline; queue MRKDaGods follow-up to (i) eval at 336², (ii) Stage-2 port to CityFlowV2 |
| MARGINAL | 90.5–91.5% | ≥ 98.4% | Confirm with one extra eval (concat-patch GeM); compare base mAP (no rerank) ≥ 88% — if yes promote, else keep ViT-B/16 as VeRi baseline; do NOT consume MRKDaGods budget |
| **FAIL** | < 90.5% OR R1 < 98.3% | — | Treat as dead end; close the ViT-L/14 CLIP TransReID branch on VeRi-776 and pivot to A (CLIP-SENet v8 with full batch via gradient checkpointing) or pure association/data-side work |

Base (no rerank, no AQE) mAP must be reported alongside; this is the number that actually drives MTMC IDF1 (per the mAP-vs-MTMC paradox). Target base mAP ≥ 88% (vs ViT-B/16's ~85.14% single-flip).

## Architecture & Recipe

**Backbone**: `vit_large_patch14_clip_224.openai` via `timm.create_model(..., pretrained=True, num_classes=0)`. Patch 14, embed_dim=1024, 24 layers, ~304M params. CLIP image-text contrastive pretraining on WIT-400M.

**Head**:
- Side Information Embedding (SIE): learnable camera token added to patch+CLS embeddings; `num_cameras=20` for VeRi-776 (cameras 1–20).
- Jigsaw Patch Module (JPM): final-block patch shuffle + group-wise BNNeck heads. Use 4 JPM groups as in 08 baseline.
- BNNeck on global CLS feature (1024-d). Linear classifier (no bias), 576 IDs.
- Triplet on raw global feature; CE-LS on BN feature; per-JPM-group CE-LS on BN'd group features. Loss weights `λ_global_ce=1.0, λ_global_tri=1.0, λ_jpm_ce=1.0` (sum-mean over 4 groups), matching 08 v8 best config.

**Optimizer**: AdamW, base lr `3.5e-4` for backbone, `3.5e-3` for head (BNNeck/classifier/JPM heads/SIE token). Weight decay `1e-4`. Betas (0.9, 0.999).

**LR schedule**: 10-epoch linear warmup from `1e-7` → base lr; cosine decay over the remaining 90 epochs to `1e-6`.

**Epochs**: **100** (down from 08's 140; verified in 08 v9 that 140→180 caused mild overfit, and L/14's larger capacity converges faster on 37,778 train images). Eval mAP/R1 every 10 epochs starting epoch 30; save best-by-mAP and best-by-R1 checkpoints + last.

**Image size**: **224×224** (must be divisible by 14 for ViT-L/14 patch size). Bicubic resize.

**Augmentation** (08 v8 stack, validated 224² recipe):
- RandomHorizontalFlip(p=0.5)
- Pad(10) → RandomCrop(224)
- RandomErasing(p=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465])
- Normalize with **CLIP** mean/std `[0.48145466, 0.4578275, 0.40821073] / [0.26862954, 0.26130258, 0.27577711]` (CRITICAL — CLIP-pretrained backbones must use CLIP norm, not ImageNet)
- No color jitter, no random rotation (08 v8 recipe).

**Sampler**: PK identity-balanced sampler `P=8, K=4`, batch_size=32. Effective gradient batch = 32 (no accumulation; T4 16GB can hold ViT-L/14 + JPM + 32 imgs at 224² with AMP fp16 — ~12 GB).
- Backup if OOM: `P=4, K=4` batch=16, accum_steps=2 → effective 32. Acceptable since BN runs on the 1024-d BNNeck only (no in-batch BN normalization on backbone — ViT uses LN), so accumulation does not degrade statistics here.

**Mixed precision**: `torch.cuda.amp` with `GradScaler`. fp16 forward/backward, fp32 master weights.

**Triplet margin**: 0.3 (08 v8 baseline).
**Label smoothing**: ε=0.1.

**Eval (final, on best-mAP checkpoint)**: replicate 09v v17 concat-patch-flip pipeline at 224². Report:
- Single-flip CLS 1024-d, no rerank
- Single-flip CLS 1024-d, AQE(k=2) + rerank(k1=80, k2=15, λ=0.2)
- Concat-patch-flip 2048-d (CLS+GeM-patch), AQE(k=2) + rerank(k1=80, k2=15, λ=0.2)

## Time Budget Breakdown (T4, AMP fp16)

| Phase | Time |
|:------|:----:|
| Kernel boot, dataset attach, environment install (`timm>=1.0`, `torch>=2.4`) | 6 min |
| Backbone init (download `vit_large_patch14_clip_224.openai` weights ~1.2GB) | 4 min |
| Smoke test (1 mini-epoch on 256 images, validate forward/backward, AMP, sampler) | 3 min |
| Training: 100 epochs × ~7 min/epoch (37,778 imgs / batch 32 ≈ 1180 iters; ViT-L/14 forward ~330ms/batch on T4 fp16, backward ~660ms/batch → ~7 min/epoch incl. data loading) | **11h40m** |
| Periodic eval at epochs {30,40,50,60,70,80,90,100}: 8 × ~2.5 min (no rerank, base mAP only) | 20 min |
| Final eval with rerank+AQE on best checkpoint | 25 min |
| Save checkpoints to `/kaggle/working/` and verify download | 5 min |
| **Total** | **~13h** |

**Safety margin**: ~2h vs the 15h gumfreddy quota. **Hard cutoff**: if any 10-epoch window shows >1.5× the per-epoch budget, abort and dump current best checkpoint.

## Risk Analysis (≥3 named risks with mitigation)

1. **OOM on T4 at batch 32 with AMP fp16.** ViT-L/14 has 24 layers × 1024-d × 16 heads. Activation memory at 224² ≈ 9–11 GB; SIE+JPM heads + classifier add ~1 GB. Could spike past 16 GB if `torch.cuda.amp` autocast region is too wide.
   - **Mitigation**: smoke-test fits batch 32 first (3-min preflight). If OOM, fall back to `P=4, K=4` batch=16 with accum_steps=2 (effective 32). Enable `torch.utils.checkpoint` on the last 6 ViT blocks if still OOM (≈10% time hit, ~30% memory cut). Use `torch.backends.cudnn.benchmark=True`.
2. **CLIP normalization mismatch silently degrades performance.** A common bug when porting from BoT/TransReID examples is using ImageNet mean/std on a CLIP-pretrained backbone. This hurts ~2–4pp mAP and the loss curve looks normal.
   - **Mitigation**: explicitly pass CLIP mean/std `[0.48145466, 0.4578275, 0.40821073] / [0.26862954, 0.26130258, 0.27577711]` and assert it in the smoke test. Also assert `timm.data.resolve_model_data_config(model)["mean"]` matches.
3. **JPM patch shuffle on patch-14 grid is wrong size.** TransReID's original JPM was designed for ViT-B/16 with 14×14 patch grid at 224². ViT-L/14 at 224² has 16×16 patch grid (224/14=16). The 4-group JPM shuffle and the per-group BNNeck must operate on a 16×16=256 token sequence, not the 14×14=196 of the 08 baseline. Hard-coded shapes in 08 will break.
   - **Mitigation**: parameterize JPM by `num_patches = (img_size//patch_size)**2` from `model.patch_embed`. Smoke test asserts `vit.patch_embed.num_patches == 256` and that the 4 groups partition 256 cleanly (256/4 = 64 patches per group).
4. **Periodic eval blow-up: rerank at every checkpoint is too slow.** Rerank is O(N²) on 11,579 gallery × 1,678 query — ~25 min standalone. Doing it at every 10-epoch eval would add ~3.5h.
   - **Mitigation**: during training, run only base mAP/R1 (no rerank, no AQE) at periodic evals. Run the full rerank+AQE only once on the final best-mAP checkpoint at end of training.
5. **timm version skew.** `vit_large_patch14_clip_224.openai` requires `timm>=0.9.16`. Older Kaggle base images may have `timm==0.6.x`.
   - **Mitigation**: pin `timm>=1.0,<2.0` in the kernel install cell; assert `timm.create_model("vit_large_patch14_clip_224.openai", pretrained=True)` succeeds in the smoke test before launching the main training loop.

## Hard Constraints (do NOT re-test these dead ends)

The following are confirmed dead ends in `docs/findings.md` and **must not appear** in this run:

- ❌ DMT camera-aware training (-1.4pp single-model on CityFlow; on VeRi: 09g 43.8% mAP)
- ❌ ArcFace warm-start (6 variants exhausted at the 52.77% ceiling on R101-IBN; conflict with the existing CE+Triplet recipe)
- ❌ ResNeXt101-IBN-a ArcFace (36.88% mAP; weight-loading mismatch)
- ❌ OSNet training from scratch
- ❌ 384px deployment of ViT (-2.8pp on CityFlow; viewpoint-specific texture issue). 224 is mandatory; 256 has been shown to be -0.12pp R1 vs 224 on the 09v eval.
- ❌ CSLS post-hoc reweighting (-34.7pp catastrophic on MTMC; likely also harmful on single-cam ranking)
- ❌ AFLink motion linking (irrelevant — this is a single-camera training, but listed for completeness)
- ❌ Hierarchical clustering or FAC at training time
- ❌ k-reciprocal rerank as the deployed stream (only as eval-time post-processing)
- ❌ Score fusion with weak (<65% mAP VeRi) secondaries
- ❌ CircleLoss (caused `inf` loss in two independent runs: 09 ablation v1 → 18.45% mAP; 09l v1 → 20.36% mAP)
- ❌ CLIP-SENet recipe at 256² / P=16 (v7 -0.98pp regression)
- ❌ Augmentation overhaul (color jitter, random rotation, AutoAugment) — caused training collapse in earlier 09 experiments
- ❌ ViT-Large AugReg backbone without CLIP/DINOv2 pretraining (09r v7: 60.38% mAP)

## Success Criteria for Promotion to Deployed VeRi Baseline

A run is promoted to the canonical VeRi-776 baseline iff **all** of:

1. Concat-patch-flip AQE(k=2)+rerank(k1=80,k2=15,λ=0.2) **mAP ≥ 91.5%** (WIN band).
2. Single-flip base **mAP ≥ 88%** (no rerank, no AQE) — exceeds ViT-B/16's 85.14%, demonstrating training-time gain not eval-time inflation.
3. Joint-optimum **R1 ≥ 98.6%** (beats 09v v17's 98.33% R1 ceiling).
4. Checkpoint is loadable in PyTorch 2.4.1+cu124 (the Kaggle 10a stack) with no missing/unexpected keys outside the classifier head, demonstrated via a 1-image inference test in the same notebook.
5. Final feature dimensionality is 1024 (CLS) or 2048 (concat-patch); both must be saved separately for downstream compatibility with the existing 10a Stage-2 pipeline.

If WIN: queue **MRKDaGods follow-up (14q)** chain:
- (i) 14q-eval at 336² (single concat-patch eval, ~30 min) — only if the backbone supports `dynamic_img_size` or via `vit_large_patch14_clip_336.openai` re-init.
- (ii) 14q-cityflow-port — Stage-2 feature extraction on CityFlowV2 with the new checkpoint, then 10b → 10c with production fusion config (`w_t=0.525, thr=0.48, aqe_k=2, fic_reg=0.5`). Goal: clear MTMC IDF1 ≥ 0.781 (current plateau 0.77936). This is the headline experiment that turns a VeRi SOTA recreation into a CityFlow improvement; it MUST be gated on the 14p WIN above.

If MARGINAL or FAIL: do NOT consume MRKDaGods budget. Close the ViT-L/14 CLIP branch on VeRi-776 and reconvene with the planner.

## Files & Outputs

- **Notebook**: `notebooks/kaggle/14p_veri_vit_l_train/14p_veri_vit_l_train.ipynb`
- **Kernel slug**: `gumfreddy/14p-veri-vit-l-train`
- **kernel-metadata.json template**:
  ```json
  {
    "id": "gumfreddy/14p-veri-vit-l-train",
    "title": "14p VeRi ViT-L/14 CLIP TransReID Train",
    "code_file": "14p_veri_vit_l_train.ipynb",
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
- **Outputs to save** (in `/kaggle/working/`):
  - `transreid_vit_l14_clip_veri776_best_mAP.pth` (state_dict only)
  - `transreid_vit_l14_clip_veri776_best_R1.pth`
  - `transreid_vit_l14_clip_veri776_last.pth`
  - `train_log.json` (per-epoch loss, lr, periodic mAP/R1)
  - `eval_results.json` (final base + AQE+rerank table)
  - `recipe.json` (full hparams snapshot for reproducibility)
- **Local result mirror**: `outputs/14p_v1_summary/14p_summary.json`

## Update findings.md After Run

Append a `## 14p TransReID ViT-L/14 CLIP on VeRi-776` section with verdict (WIN / MARGINAL / FAIL), exact mAP/R1 in each eval mode, training walltime, and the 14q gating decision.

---

## Coder Handoff Prompt (ready-to-paste)

You are the @coder agent. Implement spec **`docs/subagent-specs/14p-veri-sota-train.md`**. **Read the spec in full before writing code.**

**Phase 0 — Preflight (local, no Kaggle push)**:
1. Activate `.venv` (Python 3.11.9): `.\.venv\Scripts\activate`.
2. Verify `~/.kaggle/gumfreddy_access_token` exists and contains a `KGAT_` token. If missing, STOP and report.
3. Confirm `kaggle` CLI version: `kaggle --version` (must be ≥ 2.0.0).
4. Check active gumfreddy GPU sessions: `$env:KAGGLE_API_TOKEN = (Get-Content $HOME/.kaggle/gumfreddy_access_token -Raw).Trim(); kaggle kernels list --mine`. If ≥1 GPU kernel is currently `running`, STOP — do not push a second one.

**Phase 1 — Build notebook**:
1. Create `notebooks/kaggle/14p_veri_vit_l_train/` with `kernel-metadata.json` (use the template in the spec) and `14p_veri_vit_l_train.ipynb`.
2. Build the notebook via a `_build_14p_notebook.py` helper script using `json.load → modify → json.dump` (NEVER use text replace on `.ipynb`). Use `ensure_ascii=True`. Each `source` line ends with `\n` except the last.
3. Cells:
   - **Cell 1 (markdown)**: title + spec reference + WIN/MARGINAL/FAIL bands.
   - **Cell 2 (code)**: env install — pin `timm>=1.0,<2.0`, `torch>=2.4`, `torchvision>=0.19`. Print versions.
   - **Cell 3 (code)**: dataset paths, set seeds (random / numpy / torch) to 42, set `cudnn.benchmark=True`.
   - **Cell 4 (code)**: VeRi-776 dataset class + PK sampler (P=8, K=4) + transforms (CLIP mean/std assertion + RandomErasing + Pad+RandomCrop).
   - **Cell 5 (code)**: TransReID model — `timm.create_model("vit_large_patch14_clip_224.openai", pretrained=True, num_classes=0)` + SIE camera token (20 cameras) + JPM (4 groups, 64 patches/group from 16×16 grid) + BNNeck heads. **Assert `vit.patch_embed.num_patches == 256` and `vit.embed_dim == 1024`.**
   - **Cell 6 (code)**: Losses — TripletLoss(margin=0.3) + CrossEntropyLabelSmooth(0.1, 576 IDs) + per-JPM-group CE.
   - **Cell 7 (code)**: Optimizer (AdamW, lr 3.5e-4 backbone / 3.5e-3 head, wd 1e-4) + cosine schedule with 10-epoch linear warmup from 1e-7.
   - **Cell 8 (code)**: **Smoke test** — load 256 images, run 1 forward+backward+step at full batch, verify no OOM, assert loss is finite, time 10 iterations and project per-epoch walltime; abort if projected total >14h.
   - **Cell 9 (code)**: Main 100-epoch training loop with AMP, periodic mAP/R1 eval at epochs 30/40/.../100 (base only, no rerank), checkpoint saving (best-mAP / best-R1 / last).
   - **Cell 10 (code)**: Final eval pipeline — replicate 09v v17 concat-patch-flip at 224², three rows (base, single-flip+AQE+rerank, concat-patch+AQE+rerank). Save `eval_results.json`.
   - **Cell 11 (code)**: Save `recipe.json` (full hparams snapshot) and download instructions.

**Phase 2 — Validate locally**:
1. Run the build script.
2. Open the notebook in VS Code (do NOT edit) and confirm cell count, cell types, and that JSON parses cleanly: `python -c "import json; json.load(open('notebooks/kaggle/14p_veri_vit_l_train/14p_veri_vit_l_train.ipynb', encoding='utf-8'))"`.
3. Validate kernel-metadata.json: `kaggle kernels metadata -p notebooks/kaggle/14p_veri_vit_l_train/` (should not error). NEVER edit the file with text replace.

**Phase 3 — Push (gumfreddy account)**:
```powershell
$env:KAGGLE_API_TOKEN = (Get-Content $HOME/.kaggle/gumfreddy_access_token -Raw).Trim()
kaggle kernels push -p notebooks/kaggle/14p_veri_vit_l_train/
```
Then poll status every 60s: `kaggle kernels status gumfreddy/14p-veri-vit-l-train`. Watch for the `not valid dataset sources` warning — if it appears, immediately cancel and re-check `dataset_sources` in `kernel-metadata.json`.

**Phase 4 — Monitor**:
- Tail logs every 10–15 min: `python scripts/kaggle_logs.py gumfreddy/14p-veri-vit-l-train --tail 200`.
- Hard cutoff: if any 10-epoch window exceeds 1.5× the per-epoch budget (~10 min/epoch), prepare an abort plan and inform the planner.

**DO NOT push a second kernel until this run is complete or cancelled.** Report status updates to the planner agent for verdict-band decision.

---

(End of file)

---
