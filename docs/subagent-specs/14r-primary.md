# 14r-primary - CLIP-ReID Text-Prompt 2-Stage Training on VeRi-776

**Date**: 2026-05-10  
**Status**: PROPOSED - ready for future @coder implementation  
**Account**: MRKDaGods, using the remaining ~15h Kaggle T4 quota  
**Hardware**: Kaggle T4 16GB, single GPU  
**Goal**: Produce a VeRi-776 single-model checkpoint that beats the canonical 09v v17 TransReID ViT-B/16 CLIP baseline by adding an architecturally orthogonal CLIP text-tower training signal, not more image-branch scale.

## 1. Choice Rationale

Two consecutive scale-axis runs failed on the same image-only CE+triplet recipe. 14p3 scaled capacity to TransReID ViT-L/14 CLIP at 224x224 and reached only 87.95% mAP / 97.32% R1 after concat-patch-flip AQE+rerank. 14q kept ViT-B/16 CLIP but scaled resolution/schedule to 256x256 and 160 epochs, reaching 89.15% mAP / 97.20% R1. The canonical untouched baseline remains 09v v17 TransReID ViT-B/16 CLIP at 224x224: base 89.97% mAP / 98.33% R1, and post-rerank+AQE 91.54% mAP.

The next lever must therefore be orthogonal to both capacity and crop-size. CLIP-ReID is the primary pick because it trains against the CLIP text tower, which this project has not used on VeRi-776. Li et al., "CLIP-ReID: Exploiting Vision-Language Model for Image Re-Identification without Concrete Text Labels" (AAAI 2023, arXiv:2211.13977), reports about 85.7-88.9% mAP / 97.6% R1 on VeRi-776 using a ViT-B/16 CLIP backbone without rerank/AQE. Given this repository's consistent base-to-post-rerank lift around +5pp on 09v v17, the projected post-rerank band is roughly 91-93% mAP, enough to clear the 91.54% WIN bar if the paper signal transfers.

| Option | Orthogonal to 14p3/14q scale fails | Published VeRi-776 evidence | 15h MRKDaGods T4 fit | Implementation cost | Expected upside | Verdict |
|---|---:|---:|---:|---:|---:|---|
| CLIP-ReID text-prompt 2-stage | High - adds CLIP text-tower supervision | High - AAAI 2023 reports ~85.7-88.9% mAP / 97.6% R1 without rerank/AQE | High - ~10-12h total | Medium - prompt learner plus i2tce head | High - projected 91-93% post-rerank mAP | PICK |
| TransReID-SSL | Medium - changes pretraining objective | Medium in general ReID, weak for our available vehicle corpus | Low - needs large unlabeled pretraining | High | Medium but not feasible | Reject |
| DINOv2 backbone | High - self-supervised distillation pretraining | Unknown for VeRi-776 in this exact recipe | High as a cheap run | Low | Medium | Probe only |
| 5-way eval-only ensemble | Low - combines correlated failed checkpoints | Eval-only, no single-model claim | High | Low | Low to medium | Reject |
| CAL/HRCN | Medium - different architecture/loss family | Some vehicle ReID evidence | Low in 15h because codebase unfamiliar | High | Medium | Reject |
| CityFlow MTMC pivot | Orthogonal to VeRi training, but abandons target | Not relevant to VeRi SOTA recreation | High | Medium | Low for current paper direction | Reject |

Rejected alternatives are deliberately kept out of scope. TransReID-SSL needs a LUPerson-scale unlabeled corpus; this workspace has only the 37k-image VeRi-776 training set. DINOv2 is the insurance probe, not the primary, because DINOv2 ViT-L/14 regressed by about 3.1pp MTMC IDF1 on CityFlow despite strong ReID mAP. Eval-only ensembling does not create the SOTA-band single-model checkpoint needed for the paper direction, and the failed checkpoints are likely correlated. CAL/HRCN would require unfamiliar code under a tight quota. Pivoting away from VeRi training abandons the user-stated direction: VeRi-776 SOTA recreation first, then CityFlowV2 port.

Risk is bounded. Stage 1 is cheap because only prompt tokens are trained. Stage 2 reuses the established TransReID-style ViT-B/16 CLIP recipe, with i2tce as the single new training signal. Even if Stage 2 underperforms, the learned prompts can be ablated or used for evaluation-time analysis.

## 2. Concrete Recipe - Full Hyperparameter Spec

**Backbone**: `vit_base_patch16_clip_224.openai` through `timm`, initialized from OpenAI CLIP image-encoder weights. Image embed_dim is 768. Use image size 224x224 only. Do not use 256x256; 14q already showed that bigger crops and more epochs do not unlock this recipe.

**Text encoder**: OpenAI CLIP text transformer, loaded through `open_clip` or the `clip` package. Freeze all text-encoder parameters in both stages. In Stage 1, train only prompt tokens. In Stage 2, load cached learned text features and keep the text side frozen.

**Stage 1 - prompt learning, frozen image and text encoders**:

- Learn ID-specific prompts using 4 shared learnable context tokens plus 1 learnable token per identity, total `(4 + 576) x 512` trainable parameters.
- Prompt template: `X X X X [ID_TOKEN_i] vehicle`, where the four `X` tokens are shared context tokens and `[ID_TOKEN_i]` is identity-specific.
- Encode text with the frozen CLIP text encoder and images with the frozen CLIP image encoder.
- Loss: bidirectional image-to-text contrastive InfoNCE over the batch identities, matching official CLIP-ReID Stage 1 behavior.
- Optimizer: Adam, lr `3.5e-4`, weight decay `1e-4`, prompts only.
- Schedule: 120 epochs, P=8/K=4, batch size 32.
- Augmentation: horizontal flip plus Pad(10)+RandomCrop only. No RandAugment and no RandomErasing in Stage 1; preserve stable image-text alignment.
- Output: `learned_prompts.pt`, containing 576 ID-specific text feature vectors after CLIP projection, dim 512.

**Stage 2 - image-branch finetune, frozen text side**:

- Initialize the image branch from OpenAI CLIP weights, not from Stage 1. Stage 1 leaves the image branch untouched.
- Add SIE camera token with `num_cameras=20`, JPM with 4 groups, and BNNeck, matching the 09v v17 / 14p3 TransReID-style recipe for direct comparability.
- Add an image-text alignment head: a learnable linear projection from 768 to 512 used only for i2tce. Initialize with Xavier. Use head lr. Keep CE/triplet/JPM losses on the original 768-d image feature path.
- Losses:
  - Standard CE with label smoothing epsilon=0.1 on the BNNeck CLS feature, 576 classes.
  - TripletLoss margin 0.3 on the raw CLS feature.
  - i2tce: cross-entropy over cosine similarity between projected image features and the 576 cached learned text features, temperature 0.07, target equal to ground-truth ID.
  - Per-JPM-group CE with label smoothing.
- Loss weights: `lambda_ce=1.0`, `lambda_tri=1.0`, `lambda_i2tce=1.0`, `lambda_jpm=1.0`. Treat `lambda_i2tce` as the only planned lever; sweep `{0.5, 1.0, 1.5}` only if the run lands MARGINAL.
- Optimizer: AdamW, backbone lr `3.5e-4`, head lr `3.5e-3`, weight decay `1e-4`, betas `(0.9, 0.999)`.
- Schedule: 10-epoch linear warmup from `1e-7` to base lr, then cosine decay to `1e-6` over the remaining 110 epochs.
- Epochs: 120 total.
- Sampler: P=8/K=4, batch size 32, AMP fp16.
- Augmentation: identical to 09v v17: Pad(10)+RandomCrop+HorizontalFlip+RandomErasing. No geometric RandAugment; 14q tested it without unlocking the recipe, and this run should isolate i2tce as the changed lever.

**Evaluation**: exactly match 09v v17. Report all rows on the best-mAP checkpoint:

- `single_flip_cls_base`
- `single_flip_cls_aqe2_rerank`
- `concat_patch_flip_aqe2_rerank`
- `concat_patch_flip_aqe3_rerank`

Use rerank parameters `k1=80`, `k2=15`, `lambda=0.2`. The verdict is based on the best concat-patch-flip post-rerank+AQE row.

## 3. Pre-Registered Verdict Bands

| Band | Criteria | Action |
|---|---|---|
| WIN | concat_patch_flip post-rerank+AQE mAP >= 91.54% and R1 >= 98.33% | Promote checkpoint as the canonical VeRi baseline. Queue a follow-up CityFlow port: Stage 2 feature extraction with the new checkpoint, then 10b/10c with production fusion. |
| MARGINAL | 90.5% <= mAP < 91.54% or 98.0% <= R1 < 98.33% | Run exactly one `lambda_i2tce` ablation sweep with 3 configs and about 1h eval total before deciding. |
| FAIL | Otherwise | Close the CLIP-ReID branch on VeRi-776 and reconvene with planner. |

## 4. Compute Budget

| Phase | Estimate |
|---|---:|
| Stage 1 prompt learning | 120 epochs x ~50 sec/epoch = ~1.7h |
| Stage 2 image finetune | 120 epochs x ~3.5 min/epoch = ~7h |
| Periodic Stage 2 eval, base only | 6 x 2 min = ~12 min |
| Final eval, 4 rows with rerank+AQE | ~30 min |
| Boot, installs, save, artifact checks | ~30 min |
| Total | ~9.5-10.5h |

This fits the MRKDaGods ~15h budget with margin. Hard cutoff: abort if Stage 2 walltime projects beyond 14h.

## 5. Files To Create / Modify

- New notebook: `notebooks/kaggle/14r_clip_reid_veri/14r_clip_reid_veri.ipynb`
- New metadata: `notebooks/kaggle/14r_clip_reid_veri/kernel-metadata.json`
  - `id`: `mrkdagods/14r-clip-reid-veri-train`
  - `title`: `14r CLIP-ReID VeRi-776 Train`
  - GPU enabled, T4, internet enabled
  - `dataset_sources`: `["abhyudaya12/veri-vehicle-re-identification-dataset"]`
- New builder: `_build_14r_clip_reid_notebook.py`, using `json.load -> modify -> json.dump` with `ensure_ascii=True`. Never text-replace raw `.ipynb` JSON.

Expected Kaggle `/kaggle/working/` outputs:

- `clip_reid_vit_b16_veri776_stage1_prompts.pt`
- `clip_reid_vit_b16_veri776_stage2_best_mAP.pth`
- `clip_reid_vit_b16_veri776_stage2_best_R1.pth`
- `clip_reid_vit_b16_veri776_stage2_last.pth`
- `train_log.json`
- `eval_results.json`
- `recipe.json`

## 6. Fast-Fail Abort Signals

- Stage 1 prompt loss is flat after epoch 5: abort because prompt learning is broken.
- Stage 2 epoch 1 base mAP is below 50% on a 1000-image quick eval: abort because the recipe regressed versus the 09v v17 epoch-1 sanity expectation of about 70%+.
- Smoke test OOM at batch 32: fall back to batch 16 with gradient accumulation 2. If that still OOMs, abort.
- `kernel-metadata.json` produces a "not valid dataset sources" warning after push: cancel the kernel immediately with `kaggle kernels cancel`, fix metadata, and repush once.
- `timm` cannot load `vit_base_patch16_clip_224.openai`: pin `timm>=1.0,<2.0`, rebuild, and revalidate before any push.

## 7. Risk + Rollback

**R1 - orthogonality fails**: i2tce may duplicate CE supervision instead of adding a new cross-modal signal. Published CLIP-ReID gains are about 1-2pp mAP, so this could land MARGINAL instead of WIN. Rollback: ablate i2tce off with `lambda_i2tce=0`, confirm the baseline path reproduces, then close the branch.

**R2 - text integration bugs**: CLIP text features are 512-d while image CLS features are 768-d. Mitigation: use the dedicated 768-to-512 image-text alignment head only for i2tce, leaving BNNeck CE/triplet on the original 768-d feature. Initialize with zero-mean Xavier and train at head lr.

**R3 - prompt collapse**: ID-specific prompts may converge to similar vectors. Mitigation: add a 0.1-weight diversity regularizer based on pairwise cosine distance between learned prompts only if Stage 1 loss or prompt diagnostics show collapse. Do not add it by default.

**Hard rollback path**: if 14r-primary FAILs, the canonical VeRi baseline remains untouched 09v v17. No production deployment changes are made. The CityFlow MTMC plateau at 0.77936 is unaffected.

## Coder Handoff Prompt

You are the @coder agent. Implement spec `docs/subagent-specs/14r-primary.md`. Read the full spec before writing code. Do not edit `docs/findings.md` until a real run has completed and results exist.

**Phase 0 - Preflight, local, no Kaggle push**:

1. Activate `.venv` with Python 3.11.9: `.\.venv\Scripts\activate`.
2. Verify `$HOME/.kaggle/MRKDaGods__access_token` exists and contains a `KGAT_` token. If missing, stop and report.
3. Confirm `kaggle --version` is at least 2.0.0.
4. Set the account token with `$env:KAGGLE_API_TOKEN = (Get-Content $HOME/.kaggle/MRKDaGods__access_token -Raw).Trim()`.
5. Check active MRKDaGods GPU sessions using `kaggle kernels list --mine`. If a GPU kernel is already running on this account, stop and report. A separate gumfreddy 14r-probe run is acceptable because it uses a different account.

**Phase 1 - Build notebook**:

1. Create `notebooks/kaggle/14r_clip_reid_veri/` with `kernel-metadata.json` and `14r_clip_reid_veri.ipynb`.
2. Create `_build_14r_clip_reid_notebook.py` and build the notebook through Python JSON manipulation: `json.load -> modify -> json.dump(ensure_ascii=True)`. Never use raw text find/replace on `.ipynb` files.
3. Notebook cells should include: title and verdict bands; installs with pinned `timm>=1.0,<2.0` plus `open_clip_torch` or `clip`; dataset paths and seeds; VeRi dataset class and PK sampler; CLIP-ReID prompt learner; TransReID-style image model with SIE/JPM/BNNeck; losses including i2tce; optimizer and scheduler; smoke test; Stage 1 training; Stage 2 training; final 4-row eval; artifact save.

**Phase 2 - Validate locally**:

1. Run the builder.
2. Parse the notebook from disk with `python -c "import json; json.load(open('notebooks/kaggle/14r_clip_reid_veri/14r_clip_reid_veri.ipynb', encoding='utf-8'))"`.
3. Validate notebook cell count and metadata in VS Code without editing it.
4. Validate Kaggle metadata locally with `kaggle kernels metadata -p notebooks/kaggle/14r_clip_reid_veri/`.

**Phase 3 - Push with MRKDaGods token**:

```powershell
$env:KAGGLE_API_TOKEN = (Get-Content $HOME/.kaggle/MRKDaGods__access_token -Raw).Trim()
kaggle kernels push -p notebooks/kaggle/14r_clip_reid_veri/
```

After push, poll `kaggle kernels status mrkdagods/14r-clip-reid-veri-train`. If the push output includes "not valid dataset sources", immediately cancel the kernel, fix `kernel-metadata.json`, and repush once after validation. Do not push a second kernel while the first is still running.

**Phase 4 - Monitor**:

Tail logs every 10-15 minutes with `python scripts/kaggle_logs.py mrkdagods/14r-clip-reid-veri-train --tail 200`. Enforce the fast-fail and 14h projected walltime cutoff rules from this spec.

