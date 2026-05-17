# Kernel Log Findings: Model Card Recovery

Date: 2026-05-17

Scope: recovery pass for fields marked `NOT RECORDED IN REPO` in `docs/model-cards.md` after PR #51, using only the six requested Kaggle kernels.

## Recovered

- `yahiaakhalafallah/09v-veri-776-eval-transreid-rerank`: `veri776_eval_results_v10.json` records the best-mAP row as R5=98.45, R10=98.81 and the joint row as R5=98.51, R10=98.75.
- `yahiaakhalafallah/14t-veri-fusion-clip-senet-x-transreid`: `14t_summary.json` records runtime_sec=2971.8576, best score-fusion R5=98.75/R10=99.05, and best concat R5=98.57/R10=98.93.
- `gumfreddy/09-vehicle-reid-cityflowv2-augoverhaul-ema`: `vehicle_reid_cityflowv2_metadata.json` records 466 train IDs, 200 eval IDs, 38,537 train images, 909 query images, and 15,472 gallery images.
- `gumfreddy/12a-wildtrack-mvdetr-training`: `MVDeTr/logs/wildtrack/.../log.txt` records `augmentation=True`, best detector MODA=92.4% at epoch 17, epoch-20 MODA=92.1%, and train/test loop time 23,492.727 sec / 6.53h. `MVDeTr/README.md` describes the enabled upstream augmentation as view-coherent affine transforms with inverse feature maps for multiview coherency.
- `yahiaakhalafallah/09s-dinov2-large-cityflowv2`: pulled notebook source records AdamW with LLRD=0.75, backbone_lr=1.5e-5, head_lr=1.5e-4, weight_decay=1e-4, batch=32 as P=8/K=4, 120 epochs, 10 warmup epochs, CE label smoothing 0.05, triplet margin 0.3, delayed center loss weight 5e-4, and the full transform stack. Kernel metadata/log record `NvidiaTeslaT4` and elapsed kernel time through notebook conversion of 11,759.98 sec / 3.27h.

## Still Not Recorded

- `yahiaakhalafallah/09v-veri-776-eval-transreid-rerank`: verifier commit SHA and original training kernel/notebook remain absent from pulled metadata/source/results.
- `yahiaakhalafallah/14t-veri-fusion-clip-senet-x-transreid`: verifier commit SHA was not present in `14t_summary.json`, `14t_fusion_results.json`, `kernel-metadata.json`, or notebook source.
- `gumfreddy/09-vehicle-reid-cityflowv2-augoverhaul-ema`: R5/R10 and commit SHA were not present in the retained metadata; the first output pull exposed only metadata plus checkpoint/curve artifacts before checkpoint cleanup.
- `gumfreddy/12a-wildtrack-mvdetr-training`: exact affine augmentation parameter ranges and verifier commit SHA were not present in pulled logs/metadata/source.
- `yahiaakhalafallah/09s-dinov2-large-cityflowv2`: R5/R10, commit SHA, train image count, ID split count, exact split file, and a dedicated train-loop walltime artifact were not present in pulled logs/metadata/source.
- `yahiaakhalafallah/13-clip-senet-train`: v6 final training loss curve summary and R20 were not present in `vehicle_clip_senet_veri776_metadata.json`, `kernel-metadata.json`, or notebook source.