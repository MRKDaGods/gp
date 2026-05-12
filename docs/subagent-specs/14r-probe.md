# 14r-probe - DINOv2 ViT-B/14 Standalone VeRi-776 Insurance Probe

**Date**: 2026-05-10  
**Status**: PROPOSED - ready for future @coder implementation  
**Account**: gumfreddy, about 6h of the remaining ~9h Kaggle T4 quota  
**Hardware**: Kaggle T4 16GB, single GPU  
**Goal**: Run a cheap, parallel insurance probe while 14r-primary uses MRKDaGods.

## 1. Choice Rationale

14r-probe tests DINOv2 ViT-B/14 as a standalone VeRi-776 single-model checkpoint under the proven 09v v17 TransReID-style recipe. It is cheap insurance against the primary CLIP-ReID run failing: DINOv2 is architecturally orthogonal in pretraining objective, using self-supervised distillation instead of CLIP image-text contrastive pretraining, and it requires no large new code beyond a `timm` backbone swap plus the correct normalization.

The prior is mixed but worth measuring. DINOv2 ViT-L/14 improved CityFlowV2 ReID mAP but regressed MTMC IDF1 by about 3.1pp, so cross-camera invariance is suspect. However, standalone VeRi-776 mAP for `vit_base_patch14_dinov2.lvd142m` with the 09v v17 recipe is unmeasured. If WIN, it gives a second SOTA-band checkpoint. If MARGINAL, it becomes a future fusion-stream candidate. If FAIL, close DINOv2 standalone for VeRi forever. The negative result is still publishable evidence that CLIP pretraining is necessary, not merely sufficient.

Do not repeat known dead ends: no CSLS, AFLink, ViT-L scale-up, OSNet small-ID branch, SGD on transformers, 384px deployment, MixUp/CutMix, color jitter, or CircleLoss.

## 2. Concrete Recipe

**Backbone**: `vit_base_patch14_dinov2.lvd142m` via `timm`, embed_dim 768, patch size 14, 224x224 input, 16x16 patch grid, `num_patches=256`.

**Image size**: 224x224, matching 09v v17. Orthogonality comes from the pretraining objective, not a scale change.

**Training recipe**: exact clone of 09v v17 / 14p except for the backbone and normalization:

- TransReID head with SIE `num_cameras=20`, JPM 4 groups, and BNNeck.
- AdamW, backbone lr `3.5e-4`, head lr `3.5e-3`, weight decay `1e-4`, betas `(0.9, 0.999)`.
- 100 epochs, 10-epoch warmup from `1e-7` to base lr, cosine decay to `1e-6`.
- PK sampler P=8/K=4, batch size 32, AMP fp16.
- CE with label smoothing epsilon=0.1, TripletLoss margin 0.3, and per-JPM-group CE.
- Augmentation: Pad(10)+RandomCrop+HorizontalFlip+RandomErasing only. No geometric RandAugment.

**Critical normalization**: use DINOv2/ImageNet stats, not CLIP stats: mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`. Assert via `timm.data.resolve_model_data_config(model)` and write the resolved config into `recipe.json`.

**Evaluation**: same 4-row 09v v17 suite: `single_flip_cls_base`, `single_flip_cls_aqe2_rerank`, `concat_patch_flip_aqe2_rerank`, `concat_patch_flip_aqe3_rerank`, with rerank `k1=80`, `k2=15`, `lambda=0.2`.

## 3. Verdict Bands

| Band | Criteria | Action |
|---|---|---|
| WIN | concat_patch_flip post-rerank+AQE mAP >= 91.54% and R1 >= 98.33% | Promote as a SOTA-band VeRi candidate; compare against 14r-primary; queue CityFlow port only after planner approval. |
| MARGINAL | mAP >= 90.5% and R1 >= 98.0%, below WIN on either metric | Keep checkpoint as candidate fusion stream; do not promote to canonical VeRi baseline. |
| FAIL | Otherwise | Close DINOv2 standalone for VeRi unless a future non-scale rationale reopens it. |

## 4. Compute Budget

| Phase | Estimate |
|---|---:|
| Training | 100 epochs x ~3 min/epoch = ~5h |
| Periodic eval plus final rerank+AQE | ~30 min |
| Boot, installs, saves, artifact checks | ~20 min |
| Total | ~6h |

This fits gumfreddy's ~9h budget and does not block MRKDaGods 14r-primary.

## 5. Files

- Notebook: `notebooks/kaggle/14r_probe_dinov2_veri/14r_probe_dinov2_veri.ipynb`
- Metadata: `notebooks/kaggle/14r_probe_dinov2_veri/kernel-metadata.json`
  - `id`: `gumfreddy/14r-probe-dinov2-veri`
  - `title`: `14r Probe DINOv2 VeRi-776 Train`
  - GPU enabled, T4, internet enabled
  - `dataset_sources`: `["abhyudaya12/veri-vehicle-re-identification-dataset"]`
- Builder: `_build_14r_probe_notebook.py`, using `json.load -> modify -> json.dump(ensure_ascii=True)`. Never text-replace `.ipynb` JSON.

Expected Kaggle `/kaggle/working/` outputs: `dinov2_vit_b14_veri776_best_mAP.pth`, `_best_R1.pth`, `_last.pth`, `train_log.json`, `eval_results.json`, `recipe.json`.

## 6. Fast-Fail Signals

- Epoch-1 quick eval mAP below 40%: abort; this indicates broken labels, transforms, wiring, or normalization.
- DINOv2 weights download fails: pin a fallback Kaggle dataset mirror if available; abort if both routes fail.
- Triplet loss is `inf` or `nan` in epoch 1: abort and dump diagnostics.
- OOM at batch 32: fall back to batch 16 with accumulation 2. If still OOM, abort.

## 7. Risk + Rollback

**R1 - weak cross-camera invariance**: likely outcome is MARGINAL because DINOv2 regressed on CityFlow. Rollback: keep only as fusion-stream candidate; do not promote unless WIN.

**R2 - JPM shape mismatch**: patch 14 at 224x224 gives 256 tokens, already handled in 14p3. Mitigation: reuse parameterized 14p3 JPM code and assert `num_patches == 256`.

**R3 - normalization mismatch**: using CLIP stats invalidates the probe. Mitigation: assert ImageNet stats before training.

**Hard rollback**: on FAIL, close DINOv2 standalone on VeRi. Do not spend more gumfreddy quota on DINOv2 scale-ups or 384px variants.

## Coder Handoff Prompt

You are the @coder agent. Implement `docs/subagent-specs/14r-probe.md`. Read the spec in full. This is a probe only; do not delay 14r-primary.

**Phase 0 - Preflight, local, no Kaggle push**:

1. Activate `.venv`: `.\.venv\Scripts\activate`.
2. Verify `$HOME/.kaggle/gumfreddy_access_token` exists and contains `KGAT_`.
3. Confirm `kaggle --version` is at least 2.0.0.
4. Set `$env:KAGGLE_API_TOKEN = (Get-Content $HOME/.kaggle/gumfreddy_access_token -Raw).Trim()`.
5. Check gumfreddy GPU sessions with `kaggle kernels list --mine`; require 0 active gumfreddy GPU sessions. MRKDaGods 14r-primary running separately is fine.

**Phase 1 - Build notebook**: create `notebooks/kaggle/14r_probe_dinov2_veri/`, `kernel-metadata.json`, `14r_probe_dinov2_veri.ipynb`, and `_build_14r_probe_notebook.py`. Generate the notebook with Python JSON manipulation, `ensure_ascii=True`, and cells for title, installs, data, DINOv2 model, normalization assertions, losses, optimizer, smoke test, train loop, final eval, and saves.

**Phase 2 - Validate locally**: run the builder, parse the notebook JSON from disk, and validate metadata with `kaggle kernels metadata -p notebooks/kaggle/14r_probe_dinov2_veri/`.

**Phase 3 - Push with gumfreddy token**:

```powershell
$env:KAGGLE_API_TOKEN = (Get-Content $HOME/.kaggle/gumfreddy_access_token -Raw).Trim()
kaggle kernels push -p notebooks/kaggle/14r_probe_dinov2_veri/
```

Poll `kaggle kernels status gumfreddy/14r-probe-dinov2-veri`. If push output includes "not valid dataset sources", cancel immediately, fix metadata, and repush only after validation.

**Phase 4 - Monitor**: tail logs with `python scripts/kaggle_logs.py gumfreddy/14r-probe-dinov2-veri --tail 200`, enforce fast-fail rules, and report final verdict against the bands.