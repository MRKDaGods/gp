# Spec: NB09b v2 — Fixed 384px TransReID CityFlowV2 Fine-Tuning

## Goal
Retrain 384x384 TransReID ViT-Base/16 CLIP on CityFlowV2 with full proven NB09 recipe.
Target: mAP > 78% (v1 got 62.13% due to stripped techniques).

## Key Delta from NB09 (256px)
- H,W = 384,384 (not 256)
- img_size=384 in TransReID constructor (24x24 patch grid, 576 patches)  
- BATCH = 14 * NUM_GPUS (reduced from 32 for memory)
- P_IDS = 7 * NUM_GPUS
- Everything else IDENTICAL to NB09

## What NB09b v1 Was Missing (ALL must be restored)
1. SIE (camera embeddings) - num_cameras=59
2. JPM (jigsaw patch module)
3. LLRD (layer-wise LR decay, factor=0.75)
4. Center Loss (5e-4, delayed@ep15, SGD lr=0.5)
5. VeRi-776 pretrained init (mrkdagods/transreid-veri/other/default/1)
6. AdamW (not Adam), wd=5e-4
7. Gradient clipping max_norm=5.0
8. backbone_lr=1e-4, head_lr=1e-3
9. Label smoothing epsilon=0.05 (not 0.10)
10. 120 epochs (not 80), warmup=10

## Implementation Strategy
Copy NB09 notebook structure wholesale. Change only: resolution (384), batch size (14*GPUs), img_size param. Keep NB09b's crop extraction cell (Cell 3) which has the streaming fix.

## VeRi Weight Loading
Skip keys: cls_head, jpm_cls, sie_embed, pos_embed (shape mismatches)
Path: /kaggle/input/models/mrkdagods/transreid-veri/other/default/1/transreid_veri_best.pth

## kernel-metadata.json
- model_sources: ["mrkdagods/transreid-veri/other/default/1"]
- dataset_sources: ["thanhnguyenle/data-aicity-2023-track-2"]
- machine_shape: NvidiaTeslaT4 (T4x2)

## CityFlowV2 Path Fix
Use fallback pattern (same as 09d v8):
for _subpath in ["AIC22_Track1_MTMC_Tracking/train", "train"]:
    RAW_ROOT = _CITYFLOW_BASE / _subpath
    if RAW_ROOT.exists(): break
