# Model Cards

Per-model architecture, training recipe, and verified metrics for the
checkpoints registered in `configs/model_registry.yaml`. Model weights are not
committed; see [../README.md](../README.md) and
[../models/reid/README.md](../models/reid/README.md) for how to download or
regenerate them.

All ReID metrics are single-camera retrieval (R1 / mAP). MTMC metrics are
cross-camera IDF1. Person detection uses MODA.

---

## cityflow_transreid (vehicle MTMC primary stream)

| Field | Value |
| --- | --- |
| Task | CityFlowV2 single-camera vehicle ReID and primary MTMC feature extractor |
| Checkpoint | `transreid_cityflowv2_best.pth` |
| Dataset | CityFlowV2 (AI City 2022, Track 1) |
| Headline | single-camera mAP 81.53, R1 92.41 |

- **Architecture.** TransReID ViT-B/16 on the OpenAI CLIP backbone
  (`vit_base_patch16_clip_224.openai`), 256x256 input, SIE camera embeddings (59
  cameras), JPM (4 groups), BNNeck. The Stage-2 deployment path pools CLS + GeM
  patch tokens to 1536D, then PCA-whitens to 384D for association.
- **Training.** AdamW, backbone lr 3.5e-4, head lr 3.5e-3, layer-wise LR decay
  0.65, cosine schedule with 10-epoch warmup, 120 epochs, AMP fp16. Sampling
  P=16, K=4 (batch 64). Loss: cross-entropy (label smoothing 0.05) + triplet
  (margin 0.3) + center loss (5e-4, from epoch 15), EMA 0.999. AugOverhaul
  augmentation stack (grayscale, colour jitter, blur, perspective, erasing).
- **Inference (MTMC 14e B1).** TTA Stage-2 features, AQE k=2, DINOv2 tertiary
  weight 0.525, graph similarity threshold 0.48, FIC regularisation 0.5.
- **Notes.** Single-camera mAP does not predict MTMC IDF1 in this project; this
  checkpoint is deployed for feature-stream diversity in the score-fusion stack.

## veri776_09v_v17_transreid (VeRi-776 ReID reference)

| Field | Value |
| --- | --- |
| Task | VeRi-776 single-camera vehicle ReID |
| Checkpoint | `vehicle_transreid_vit_base_veri776.pth` |
| Dataset | VeRi-776 (576 train IDs; 1,678 query, 11,579 gallery) |
| Headline | mAP 89.97, R1 98.33 |

- **Architecture.** TransReID ViT-B/16 CLIP (`vit_base_patch16_clip_224.openai`),
  768D global feature (1536D concat-patch bundle for the best-mAP eval), SIE
  camera embeddings, JPM (4 groups), BNNeck.
- **Inference.** Best-R1 row: single-flip + k-reciprocal rerank (k1=24, k2=8,
  lambda=0.2). Best-mAP row: concat-patch-flip + AQE k=3 + rerank (k1=80, k2=15,
  lambda=0.2).
- **Notes.** Eval-time tricks no longer improve the checkpoint beyond the 98.33
  R1 ceiling; ten-crop TTA was harmful.

## veri776_clipsenet_v6 (VeRi-776 ReID second stream)

| Field | Value |
| --- | --- |
| Task | VeRi-776 single-camera vehicle ReID |
| Checkpoint | `clipsenet_v6_veri776_best.pth` |
| Dataset | VeRi-776 (576 train IDs) |
| Headline | base mAP 82.34; rerank+AQE mAP 91.54, R1 97.32 |

- **Architecture.** ResNet101-IBN-a appearance branch (2048D) plus a TinyCLIP
  semantic branch (512D), fused through an AFEM (G=32) / SENet path with BNNeck.
- **Training.** Adam, lr 5e-4, weight decay 5e-4, cosine schedule after 5 warmup
  epochs, 24 epochs. Effective batch 128 (P=8, K=8, accumulate 2). Loss:
  cross-entropy (label smoothing 0.1) + supervised contrastive (temp 0.07).
  Input 320x320, standard flip/pad/crop/erasing, AMP fp16.
- **Inference.** Base cosine; AQE k=10; rerank (k1=50, k2=10, lambda=0.1) for the
  91.54 mAP reference.
- **Notes.** The 256x256 / P=16 v7 retrain regressed and is closed. The
  VeRi-trained model does not transfer to CityFlowV2 MTMC.

## veri776_14t_fusion (VeRi-776 two-stream fusion)

| Field | Value |
| --- | --- |
| Task | VeRi-776 single-camera score-level fusion (inference only) |
| Checkpoints | `vehicle_transreid_vit_base_veri776.pth` + `clipsenet_v6_veri776_best.pth` |
| Dataset | VeRi-776 |
| Headline | mAP 93.30, R1 98.45 |

- **Method.** Per-model AQE k=3, score fusion
  `score = 0.7 * (q_cs @ g_cs.T) + 0.3 * (q_tr @ g_tr.T)`, then k-reciprocal
  rerank (k1=80, k2=15, lambda=0.2). No new trained head.
- **Notes.** Inference-only; not a CityFlowV2 MTMC feature stream. Evaluated by
  `scripts/eval/eval_14t_fusion_veri776.py`.

## YOLO26m + BoxMOT (vehicle detection and tracking)

| Field | Value |
| --- | --- |
| Task | CityFlowV2 vehicle detection + single-camera tracking |
| Checkpoint | `models/detection/yolo26m.pt` (COCO-pretrained, no project fine-tuning) |
| Dataset | CityFlowV2 (inference) |

- **Architecture.** Ultralytics YOLO26m detector + BoxMOT BoT-SORT tracker.
- **Inference.** YOLO confidence 0.25, NMS IoU 0.65, class-agnostic NMS, fp16,
  1280x1280; vehicle classes car/bus/truck. BoT-SORT: track buffer 450, max age
  450, min hits 3, match thresh 0.85; tracker ReID uses
  `models/tracker/osnet_x0_25_msmt17.pt` (separate from the Stage-2 ReID).
- **Notes.** No standalone detector metric is recorded; used within the 14e B1
  MTMC stack (IDF1 0.779).

## person_detector_12a_mvdetr (WILDTRACK person detection)

| Field | Value |
| --- | --- |
| Task | WILDTRACK multi-view ground-plane person detection |
| Checkpoint | `MultiviewDetector.pth` |
| Dataset | WILDTRACK (~400 frames x 7 cameras) |
| Headline | MODA 0.913, precision 0.947, recall 0.966 (exported model) |

- **Architecture.** MVDeTr with an ImageNet-pretrained ResNet18 backbone and a
  deformable-transformer multi-view fusion, with heatmap / offset / size heads
  (no identity head).
- **Training.** Adam, base lr 7e-4, backbone lr 7e-5, OneCycleLR for 25 epochs,
  weight decay 1e-4, batch size 1. Loss: CornerNet-style focal heatmap loss + L1
  offset + L1 size (weight 0.1). View-coherent affine augmentation.
- **Inference.** Followed by a Kalman tracker (max age 2, min hits 2, distance
  gate 25.0). Final person MTMC IDF1 0.947, MODA 0.900.
- **Notes.** Better detector MODA did not move WILDTRACK IDF1 beyond the
  converged 0.947 tracker plateau.

## DINOv2 ViT-L/14 (vehicle MTMC tertiary stream)

| Field | Value |
| --- | --- |
| Task | CityFlowV2 vehicle ReID tertiary score-fusion stream |
| Checkpoint | `vehicle_transreid_dinov2_large_cityflowv2_final.pth` |
| Dataset | CityFlowV2 |
| Headline | single-camera mAP 86.79, R1 96.15; standalone MTMC IDF1 0.744 |

- **Architecture.** DINOv2 ViT-L/14 (`vit_large_patch14_dinov2.lvd142m`)
  fine-tuned on CityFlowV2, 1024D CLS-token feature.
- **Training.** AdamW with layer-wise LR decay 0.75, backbone lr 1.5e-5, head lr
  1.5e-4, 10-epoch warmup, cosine decay, 120 epochs (best epoch 115). Sampling
  P=8, K=4 (batch 32). Loss: cross-entropy (label smoothing 0.05) + batch-hard
  triplet (margin 0.3) + delayed center loss (5e-4). Input 252x252 (stride 14).
- **Inference.** Tertiary stream in 14e B1 at weight 0.525 (AQE k=2, similarity
  threshold 0.48, FIC regularisation 0.5).
- **Notes.** Higher single-camera mAP did not transfer to better standalone
  MTMC (0.744 IDF1 despite 86.79 mAP); it contributes only as a fusion stream.
