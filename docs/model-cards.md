# Model Cards

<!-- markdownlint-disable MD013 MD024 MD060 -->

Last verified: 2026-05-17

Inventory cross-reference: [docs/models.md](docs/models.md).

These cards capture per-model recipe and provenance details that are deeper than the checkpoint inventory. Missing fields are intentionally marked as "not recorded in repo" rather than inferred.

## veri776_09v_v17_transreid

### Identity

| Field | Value |
|---|---|
| model_id | `veri776_09v_v17_transreid` |
| task | VeRi-776 single-camera vehicle ReID |
| registry location | [configs/model_registry.yaml#L382](configs/model_registry.yaml#L382) |
| checkpoint filename | `vehicle_transreid_vit_base_veri776.pth` |
| dataset | VeRi-776 |
| headline metric | R1=98.33%, mAP=89.97% |

### Architecture

| Field | Value |
|---|---|
| backbone | TransReID ViT-B/16 CLIP, `vit_base_patch16_clip_224.openai` |
| embedding dim | 768D global feature; 1536D concat-patch feature bundle used for the best-mAP eval row |
| special heads/tokens | SIE camera embeddings; JPM/patch feature path available through concat-patch GeM evaluation; BNNeck details for the original training are not recorded in repo |

### Training Data

| Field | Value |
|---|---|
| dataset name + version | VeRi-776 from `abhyudaya12/veri-vehicle-re-identification-dataset` for verifier access |
| ID count | 576 train IDs recorded in findings; full train/test ID counts beyond the standard split are not recorded in repo |
| image count | approximately 37k train images recorded in findings; exact train/query/gallery counts for the 09v training run are not recorded in repo |
| train/test split | `image_train`, `image_query`, `image_test`; exact original training split metadata is not recorded in repo |

### Training Recipe

The repo contains the 09v v17 verifier/evaluation notebook, not the original training notebook. Training fields below are limited to what is recorded in docs and notebook-derived evaluation setup.

| Hyperparameter | Value |
|---|---|
| optimizer + LR + schedule | not recorded in repo for the original checkpoint |
| batch size + sampling strategy | not recorded in repo for the original checkpoint |
| epochs + warmup | not recorded in repo for the original checkpoint |
| loss functions | standard CE + triplet is described for the saturated CLIP TransReID family, but the exact 09v training loss config is not recorded in repo |
| augmentations | not recorded in repo for training; verifier tested single_flip, concat_patch_flip, AQE, rerank, and rejected ten-crop TTA |
| hardware + approximate training time | not recorded in repo |

### Inference Hyperparameters

| Setting | Value |
|---|---|
| input image size | 224x224 in the canonical v17 verifier |
| feature dim | 768D single_flip; 1536D concat_patch_flip |
| normalization | L2-normalized features |
| post-processing | best R1 row: single_flip + rerank k1=24, k2=8, lambda=0.2; best mAP row: concat_patch_flip + AQE k=3 + rerank k1=80, k2=15, lambda=0.2; joint row: concat_patch_flip + AQE k=2 + same rerank |

### Verified Metrics

| Scope | R1 | R5 | R10 | mAP | IDF1 | HOTA | MOTA | IDSW | Kernel | Verification commit SHA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| Best R1 row | 98.33 | 99.05 | 99.34 | 85.14 | N/A | N/A | N/A | N/A | `yahiaakhalafallah/09v-veri-776-eval-transreid-rerank` v17 | not recorded in repo |
| Best mAP row | 97.80 | not recorded in repo | not recorded in repo | 89.97 | N/A | N/A | N/A | N/A | `yahiaakhalafallah/09v-veri-776-eval-transreid-rerank` v17 | not recorded in repo |
| Joint row | 98.15 | not recorded in repo | not recorded in repo | 89.71 | N/A | N/A | N/A | N/A | `yahiaakhalafallah/09v-veri-776-eval-transreid-rerank` v17 | not recorded in repo |

### Provenance

| Field | Value |
|---|---|
| training notebook path | not recorded in repo; verifier path is [notebooks/kaggle/09v_veri776_eval/09v-veri776-eval.ipynb](notebooks/kaggle/09v_veri776_eval/09v-veri776-eval.ipynb) |
| Kaggle training kernel slug | `yahiaakhalafallah/09v-veri-776-eval-transreid-rerank` is recorded as the source/verifier slug in registry; original training kernel is not separately recorded |
| verifier kernel slug | `yahiaakhalafallah/09v-veri-776-eval-transreid-rerank` |
| date of best result | canonical v17 result recorded by 2026-05-11 in [docs/experiment-log.md#L727](docs/experiment-log.md#L727) |
| author/account | `yahiaakhalafallah` for verifier; checkpoint hosted via `mrkdagods/mtmc-weights` and `gumfreddy/mtmc-weights` |

### Known Limitations

- Eval-time tricks no longer improve the checkpoint beyond the 98.33% R1 ceiling; ten-crop TTA was harmful.
- The remembered historical 0.984505 value is recorded as R5, not R1, for the relevant AQE/rerank row.
- Original training recipe details are incomplete in the repo, so reproduction is verifier-based rather than full retraining-based.

## veri776_clipsenet_v6

### Identity

| Field | Value |
|---|---|
| model_id | `veri776_clipsenet_v6` |
| task | VeRi-776 single-camera vehicle ReID |
| registry location | [configs/model_registry.yaml#L434](configs/model_registry.yaml#L434) |
| checkpoint filename | `clipsenet_v6_veri776_best.pth` from kernel output `best.pth` |
| dataset | VeRi-776 |
| headline metric | cosine/base mAP=82.34%, R1=96.54%; rerank+AQE mAP=91.54%, R1=97.32% |

### Architecture

| Field | Value |
|---|---|
| backbone | ResNet101-IBN-a appearance branch plus TinyCLIP semantic branch |
| embedding dim | 2048D output feature in verifier/fusion notebooks |
| special heads/tokens | SENet/AFEM path with BNNeck; appearance branch 2048D plus TinyCLIP 512D semantic branch, concat to FC, `T_u`, AFEM with G=32, `T_s'`, and `T = T_u + T_s'` |

### Training Data

| Field | Value |
|---|---|
| dataset name + version | VeRi-776 from `abhyudaya12/veri-vehicle-re-identification-dataset` |
| ID count | 576 train IDs in the canonical split |
| image count | repo notebook prints train/query/gallery counts at runtime; exact values are not recorded in static repo output |
| train/test split | `image_train`, `image_query`, `image_test` with `name_train.txt`, `name_query.txt`, `name_test.txt` |

### Training Recipe

| Hyperparameter | Value |
|---|---|
| optimizer + LR + schedule | Adam, lr=5e-4, weight_decay=5e-4, cosine schedule via LambdaLR after 5 warmup epochs |
| batch size + sampling strategy | micro-batch P=8, K=8, batch_size=64, accum_steps=2; effective batch 128 with P_effective=16, K=8 |
| epochs + warmup | 24 epochs, 5 warmup epochs, eval every 2 epochs |
| loss functions | CrossEntropy with label smoothing epsilon=0.1 plus SupCon temperature=0.07 |
| augmentations | Resize 320x320, RandomHorizontalFlip p=0.5, Pad(10), RandomCrop 320x320, ImageNet normalization, RandomErasing p=0.5 with random value |
| hardware + approximate training time | P100, approximately 4h26min for v6; AMP fp16 enabled |

### Inference Hyperparameters

| Setting | Value |
|---|---|
| input image size | 320x320 |
| feature dim | 2048D |
| normalization | L2-normalized features |
| post-processing | base cosine; AQE k=10; preferred rerank row k1=50, k2=10, lambda=0.1 after AQE for the 91.54 mAP reference |

### Verified Metrics

| Scope | R1 | R5 | R10 | mAP | IDF1 | HOTA | MOTA | IDSW | Kernel | Verification commit SHA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| Base cosine | 96.54 | 98.51 | 99.11 | 82.34 | N/A | N/A | N/A | N/A | `yahiaakhalafallah/13-clip-senet-train` v6 | train `53c2947`; eval `3df0915` |
| AQE k=10 | 96.90 | 98.03 | 98.75 | 89.21 | N/A | N/A | N/A | N/A | `yahiaakhalafallah/13-clip-senet-train` v6 | train `53c2947`; eval `3df0915` |
| Rerank k1=50, k2=10, lambda=0.1 | 97.32 | 98.09 | 98.69 | 91.54 | N/A | N/A | N/A | N/A | `yahiaakhalafallah/13-clip-senet-train` v6 | train `53c2947`; eval `3df0915` |

### Provenance

| Field | Value |
|---|---|
| training notebook path | [notebooks/kaggle/13_clip_senet_train/13_clip_senet_train.ipynb](notebooks/kaggle/13_clip_senet_train/13_clip_senet_train.ipynb) |
| Kaggle training kernel slug | `yahiaakhalafallah/13-clip-senet-train` |
| verifier kernel slug | `yahiaakhalafallah/13-clip-senet-train`; v7 eval slug `yahiaakhalafallah/13e-v7-clip-senet-eval` is for the rejected v7 variant |
| date of best result | 2026-05-06 |
| author/account | `yahiaakhalafallah` |

### Known Limitations

- The v7 256x256/P=16 retrain regressed versus v6 and is closed.
- VeRi-trained CLIP-SENet does not transfer to CityFlowV2 MTMC: 13d score-fusion degraded monotonically and standalone CityFlow MTMC reached only 0.6855 IDF1.
- The paper-claim gap is plausibly tied to unavailable TinyCLIP weights and P100 batch/BN constraints, but the exact missing ingredient is not recorded.

## cityflow_transreid

### Identity

| Field | Value |
|---|---|
| model_id | `cityflow_transreid` |
| task | CityFlowV2 single-camera vehicle ReID reference and primary MTMC feature extractor |
| registry location | [configs/model_registry.yaml#L74](configs/model_registry.yaml#L74) |
| checkpoint filename | `transreid_cityflowv2_best.pth` |
| dataset | CityFlowV2 |
| headline metric | mAP=81.53%; R1=92.41% in registry and experiment log, while some high-level docs still carry R1=92.27% as an unverified claim |

### Architecture

| Field | Value |
|---|---|
| backbone | TransReID ViT-B/16 CLIP 256px |
| embedding dim | 768D before downstream Stage-2 pooling; deployed Stage-2 path uses CLS + GeM patches to 1536D then PCA to 384D for association |
| special heads/tokens | SIE/JPM-style TransReID behavior is used in the project family; exact training head config for this checkpoint is not recorded in repo |

### Training Data

| Field | Value |
|---|---|
| dataset name + version | CityFlowV2 / AI City Challenge 2022 Track 1 |
| ID count | 666 CityFlowV2 IDs are recorded for related CityFlow fine-tuning; exact train/test ID split for this checkpoint is not recorded in repo |
| image count | not recorded in repo for this checkpoint |
| train/test split | ad-hoc CityFlowV2 ReID benchmark split; exact split file is not recorded in repo |

### Training Recipe

| Hyperparameter | Value |
|---|---|
| optimizer + LR + schedule | AugOverhaul + EMA kernel is recorded, but exact optimizer/LR/schedule are not recorded in repo |
| batch size + sampling strategy | not recorded in repo |
| epochs + warmup | not recorded in repo |
| loss functions | not recorded in repo for this checkpoint |
| augmentations | AugOverhaul is recorded; exact augmentation list is not recorded in repo |
| hardware + approximate training time | GPU training on Kaggle; exact GPU type and walltime are not recorded in repo |

### Inference Hyperparameters

| Setting | Value |
|---|---|
| input image size | 256x256 |
| feature dim | 768D model feature; 384D association vector after deployed PCA whitening path |
| normalization | L2/cosine features with FIC whitening, power normalization, and PCA in the MTMC pipeline |
| post-processing | For 14e B1 MTMC: TTA Stage-2 features, AQE k=2, DINOv2 tertiary weight=0.525, graph similarity threshold=0.48, FIC regularisation=0.5 |

### Verified Metrics

| Scope | R1 | R5 | R10 | mAP | IDF1 | HOTA | MOTA | IDSW | Kernel | Verification commit SHA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| Single-camera CityFlowV2 reference | 92.41 | not recorded in repo | not recorded in repo | 81.53 | N/A | N/A | N/A | N/A | `gumfreddy/09-vehicle-reid-cityflowv2-augoverhaul-ema` | not recorded in repo |
| MTMC 14e B1 stack using this primary stream | N/A | N/A | N/A | N/A | 0.77936 | 0.79461 trackeval IDF1 recorded, HOTA not recorded for this exact row | not recorded in repo for this exact row | 154 | `yahiaakhalafallah/14e-tta-fusion-aqe-fic-sweep` | not recorded in repo |

### Provenance

| Field | Value |
|---|---|
| training notebook path | not present under `notebooks/kaggle/` in this repo |
| Kaggle training kernel slug | `gumfreddy/09-vehicle-reid-cityflowv2-augoverhaul-ema` |
| verifier kernel slug | docs cite retained kernel metadata/logs; standalone verifier notebook path is not recorded in repo |
| date of best result | 2026-04 lineage, promoted into registry on 2026-05-17 |
| author/account | `gumfreddy` |

### Known Limitations

- Single-camera mAP does not predict MTMC IDF1 in this project; stronger DINOv2 mAP regressed as a standalone MTMC stream.
- Several docs disagree on R1: registry/experiment-log use 92.41%, while high-level instructions call 92.27% unverified.
- GPU-heavy Stage 0-2 inference should not be run locally on the GTX 1050 Ti environment.

## veri776_14t_fusion

### Identity

| Field | Value |
|---|---|
| model_id | `veri776_14t_fusion` |
| task | VeRi-776 single-camera score-fusion evaluation |
| registry location | [configs/model_registry.yaml#L321](configs/model_registry.yaml#L321) |
| checkpoint filename | Uses `vehicle_transreid_vit_base_veri776.pth` plus `clipsenet_v6_veri776_best.pth`; no fused checkpoint file is produced |
| dataset | VeRi-776 |
| headline metric | mAP=93.30%, R1=98.45% |

### Architecture

| Field | Value |
|---|---|
| backbone | Score-level fusion of TransReID 09v v17 and CLIP-SENet v6 |
| embedding dim | TransReID 768D global-token stream plus CLIP-SENet 2048D stream; concat experiments also evaluated 1536D TransReID paths |
| special heads/tokens | Fusion uses parent model outputs; no new trained head |

### Training Data

| Field | Value |
|---|---|
| dataset name + version | VeRi-776 |
| ID count | inherits parent-model VeRi-776 splits; no new training IDs |
| image count | inherits parent-model VeRi-776 query/gallery data; no new training images |
| train/test split | inference-only query/gallery evaluation; no fusion training split |

### Training Recipe

This is an inference-only fusion experiment, not a trained model.

| Hyperparameter | Value |
|---|---|
| optimizer + LR + schedule | N/A, inference-only |
| batch size + sampling strategy | TransReID batch 64; CLIP-SENet batch 32 for feature extraction |
| epochs + warmup | N/A, inference-only |
| loss functions | N/A, inference-only |
| augmentations | TransReID single_flip feature extraction; CLIP-SENet standard evaluation resize/normalization |
| hardware + approximate training time | T4, approximately 49.5 min runtime for the fusion sweep |

### Inference Hyperparameters

| Setting | Value |
|---|---|
| input image size | TransReID 224x224; CLIP-SENet 320x320 |
| feature dim | TransReID 768D selected stream; CLIP-SENet 2048D |
| normalization | L2-normalized parent features and score matrices |
| post-processing | score fusion w_clipsenet=0.7, w_transreid=0.3, AQE k=3, rerank k1=80, k2=15, lambda=0.2 |

### Verified Metrics

| Scope | R1 | R5 | R10 | mAP | IDF1 | HOTA | MOTA | IDSW | Kernel | Verification commit SHA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| Best score-fusion row | 98.45 | not recorded in repo | not recorded in repo | 93.30 | N/A | N/A | N/A | N/A | `yahiaakhalafallah/14t-veri-fusion-clip-senet-x-transreid` | not recorded in repo |
| Best concat row | 98.27 | not recorded in repo | not recorded in repo | 93.19 | N/A | N/A | N/A | N/A | `yahiaakhalafallah/14t-veri-fusion-clip-senet-x-transreid` | not recorded in repo |

### Provenance

| Field | Value |
|---|---|
| training notebook path | N/A, inference-only; notebook path is [notebooks/kaggle/14t_veri_fusion/14t_veri_fusion.ipynb](notebooks/kaggle/14t_veri_fusion/14t_veri_fusion.ipynb) |
| Kaggle training kernel slug | N/A for fusion; parent kernels are `yahiaakhalafallah/09v-veri-776-eval-transreid-rerank` and `yahiaakhalafallah/13-clip-senet-train` |
| verifier kernel slug | `yahiaakhalafallah/14t-veri-fusion-clip-senet-x-transreid` |
| date of best result | 2026-05-11 |
| author/account | `yahiaakhalafallah` |

### Known Limitations

- The successful VeRi-776 fusion does not port to CityFlowV2: 14u peaked at 0.77995 IDF1, only +0.00059 versus the drift gate and below the promotion bar.
- It should remain scoped to VeRi-776 single-camera evaluation and verification, not as a CityFlow MTMC replacement feature stream.
- It depends on two parent checkpoints and therefore inherits both parent provenance gaps.

## YOLO26m + BoT-SORT

### Identity

| Field | Value |
|---|---|
| model_id | no standalone registry id; detector role appears under `vehicle_mtmc_14e_b1` |
| task | CityFlowV2 vehicle detection plus single-camera tracking |
| registry location | [configs/model_registry.yaml#L23](configs/model_registry.yaml#L23) |
| checkpoint filename | `yolo26m.pt` |
| dataset | CityFlowV2 for pipeline use; detector provenance is pretrained COCO with no project fine-tuning found |
| headline metric | no standalone detector metric recorded; used in the 14e B1 MTMC stack with IDF1=0.77936 |

### Architecture

| Field | Value |
|---|---|
| backbone | YOLO26m detector plus BoxMOT BoT-SORT tracker |
| embedding dim | N/A for detector/tracker; downstream ReID embeddings are produced in Stage 2 |
| special heads/tokens | BoT-SORT Kalman/data-association logic; no project-trained tracker head recorded |

### Training Data

| Field | Value |
|---|---|
| dataset name + version | pretrained COCO for YOLO26m; CityFlowV2 videos for inference |
| ID count | N/A for detector; CityFlowV2 track IDs are consumed downstream |
| image count | not recorded in repo for detector pretraining |
| train/test split | no project detector fine-tuning split recorded |

### Training Recipe

| Hyperparameter | Value |
|---|---|
| optimizer + LR + schedule | not recorded in repo; no project fine-tuning found |
| batch size + sampling strategy | not recorded in repo |
| epochs + warmup | not recorded in repo |
| loss functions | not recorded in repo |
| augmentations | not recorded in repo |
| hardware + approximate training time | not recorded in repo; local GPU-heavy detection/tracking is discouraged by project instructions |

### Inference Hyperparameters

| Setting | Value |
|---|---|
| input image size | not recorded in registry/docs for the model card sources |
| feature dim | N/A |
| normalization | N/A |
| post-processing | BoT-SORT tracker; historical tracker settings include min_hits=2 as important, but the complete production tracker config should be read from dataset config when running |

### Verified Metrics

| Scope | R1 | R5 | R10 | mAP | IDF1 | HOTA | MOTA | IDSW | Kernel | Verification commit SHA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| Detector/tracker standalone | N/A | N/A | N/A | not recorded in repo | not recorded in repo | not recorded in repo | not recorded in repo | not recorded in repo | not recorded in repo | not recorded in repo |
| Full vehicle MTMC 14e B1 stack using this detector/tracker | N/A | N/A | N/A | N/A | 0.77936 | not recorded in repo for this card | not recorded in repo for this card | 154 | `yahiaakhalafallah/14e-tta-fusion-aqe-fic-sweep` | not recorded in repo |

### Provenance

| Field | Value |
|---|---|
| training notebook path | not recorded in repo |
| Kaggle training kernel slug | null in registry for detector source training |
| verifier kernel slug | full-stack verification via `yahiaakhalafallah/14e-tta-fusion-aqe-fic-sweep` |
| date of best result | 2026-05-07 for 14e B1 stack |
| author/account | detector source author not recorded; project verification account `yahiaakhalafallah` |

### Known Limitations

- The repo does not record detector-only mAP/precision/recall for YOLO26m on CityFlowV2.
- The local machine is explicitly not intended for GPU-heavy Stage 0-2 runs.
- Tracker changes can hurt MTMC IDF1; `mtmc_only_submission=true`, track smoothing, and edge trim are documented as harmful or not recommended elsewhere in project instructions.

## MVDeTr ResNet18

### Identity

| Field | Value |
|---|---|
| model_id | `person_detector_12a_mvdetr` |
| task | WILDTRACK multi-view person detection |
| registry location | [configs/model_registry.yaml#L262](configs/model_registry.yaml#L262) |
| checkpoint filename | `MultiviewDetector.pth` |
| dataset | WILDTRACK |
| headline metric | MODA=0.921 epoch-20 detector claim; exported loaded-model line verifies MODA=0.913, precision=0.947, recall=0.966 |

### Architecture

| Field | Value |
|---|---|
| backbone | MVDeTr ResNet18 |
| embedding dim | N/A detector output is ground-plane detection, not ReID embedding |
| special heads/tokens | `deform_trans` MVDeTr setup with multi-view deformable transformer components |

### Training Data

| Field | Value |
|---|---|
| dataset name + version | WILDTRACK via `aryashah2k/large-scale-multicamera-detection-dataset` |
| ID count | not recorded in repo for detector training |
| image count | not recorded in repo for detector training |
| train/test split | WILDTRACK official frame/calibration/annotation layout; exact detector split files are not recorded in repo |

### Training Recipe

| Hyperparameter | Value |
|---|---|
| optimizer + LR + schedule | MVDeTr training command records lr=7e-4; optimizer and schedule are delegated to upstream MVDeTr and not recorded in repo |
| batch size + sampling strategy | batch_size=1 |
| epochs + warmup | 25 epochs for 12a v3; best MODA claim at epoch 20; warmup not recorded in repo |
| loss functions | upstream MVDeTr losses, not detailed in repo |
| augmentations | not recorded in repo |
| hardware + approximate training time | Kaggle T4 according to kernel metadata; approximate training time not recorded in repo |

### Inference Hyperparameters

| Setting | Value |
|---|---|
| input image size | WILDTRACK camera frames; exact network resize is controlled by upstream MVDeTr and not recorded in repo card sources |
| feature dim | N/A |
| normalization | N/A |
| post-processing | 12a conversion uses ground-plane detections; downstream 12b Kalman tracker selected max_age=2, min_hits=2, distance_gate=25.0, q_std=5.0, r_std=10.0, interpolation_max_gap=1 |

### Verified Metrics

| Scope | R1 | R5 | R10 | mAP | IDF1 | HOTA | MOTA/MODA | IDSW | Kernel | Verification commit SHA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| Detector epoch-20 claim | N/A | N/A | N/A | N/A | N/A | N/A | MODA=0.921 | N/A | `gumfreddy/12a-wildtrack-mvdetr-training` | not recorded in repo |
| Exported loaded-model line | N/A | N/A | N/A | N/A | N/A | N/A | MODA=0.913, MODP=0.818, precision=0.947, recall=0.966 | N/A | `gumfreddy/12a-wildtrack-mvdetr-training` | not recorded in repo |
| 12b tracking rerun on 12a v3 detections | N/A | N/A | N/A | N/A | 0.947 | not recorded in repo | MODA=0.900 | 5 | `gumfreddy/12b-wildtrack-mvdetr-tracking-reid` | not recorded in repo |

### Provenance

| Field | Value |
|---|---|
| training notebook path | [notebooks/kaggle/12a_wildtrack_mvdetr/12a_wildtrack_mvdetr.ipynb](notebooks/kaggle/12a_wildtrack_mvdetr/12a_wildtrack_mvdetr.ipynb) |
| Kaggle training kernel slug | `gumfreddy/12a-wildtrack-mvdetr-training` |
| verifier kernel slug | `gumfreddy/12b-wildtrack-mvdetr-tracking-reid` for downstream tracking; 12a kernel also runs detector export/eval |
| date of best result | 2026-05 era, exact detector best date not recorded beyond docs |
| author/account | `gumfreddy` |

### Known Limitations

- The registry intentionally marks MODA=0.921 as unverified because it is an epoch-20 claim, while the exported loaded-model line is lower at MODA=0.913.
- Better detector MODA did not move WILDTRACK IDF1 beyond the converged 0.947 tracker plateau.
- Person pipeline is tracker-limited; global optimal and naive tracker variants underperformed the Kalman setup.

## DINOv2 ViT-L/14

### Identity

| Field | Value |
|---|---|
| model_id | no standalone registry id; tertiary ReID role appears under `vehicle_mtmc_14e_b1` |
| task | CityFlowV2 vehicle ReID feature stream and tertiary score-fusion stream |
| registry location | [configs/model_registry.yaml#L52](configs/model_registry.yaml#L52) |
| checkpoint filename | `vehicle_transreid_dinov2_large_cityflowv2_final.pth` |
| dataset | CityFlowV2 |
| headline metric | CityFlowV2 ReID mAP=86.79%, R1=96.15%; standalone MTMC IDF1=0.744, but tertiary fusion helps the 14e B1 stack |

### Architecture

| Field | Value |
|---|---|
| backbone | DINOv2 ViT-L/14, `vit_large_patch14_dinov2.lvd142m` |
| embedding dim | 1024D recorded for the deployed tertiary stream |
| special heads/tokens | TransReID-style wrapper around DINOv2 features is implied by checkpoint naming; exact special head details are not recorded in repo |

### Training Data

| Field | Value |
|---|---|
| dataset name + version | CityFlowV2 |
| ID count | not recorded in repo for the 09s training split |
| image count | not recorded in repo for the 09s training split |
| train/test split | CityFlowV2 ReID split; exact split metadata is not recorded in repo |

### Training Recipe

| Hyperparameter | Value |
|---|---|
| optimizer + LR + schedule | not recorded in repo |
| batch size + sampling strategy | batch=32 |
| epochs + warmup | 120 epochs, best epoch 115/120; warmup not recorded in repo |
| loss functions | stable ID loss + batch-hard triplet + delayed center loss |
| augmentations | not recorded in repo |
| hardware + approximate training time | Kaggle GPU; exact machine shape and training time not recorded in repo |

### Inference Hyperparameters

| Setting | Value |
|---|---|
| input image size | 252x252, stride=14 for 09s training/eval; 14g tested tertiary TTA views original, hflip, scale_0.95, scale_1.05 |
| feature dim | 1024D |
| normalization | L2/cosine feature stream before score fusion |
| post-processing | As tertiary stream in 14e B1: weight=0.525, AQE k=2 at Stage 4, similarity_threshold=0.48, FIC regularisation=0.5; standalone DINOv2 MTMC best used AFLink gap=150, dir_cos=0.85 but remained below CLIP-based MTMC |

### Verified Metrics

| Scope | R1 | R5 | R10 | mAP | IDF1 | HOTA | MOTA | IDSW | Kernel | Verification commit SHA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| CityFlowV2 single-camera ReID 09s v1 | 96.15 | not recorded in repo | not recorded in repo | 86.79 | N/A | N/A | N/A | N/A | `yahiaakhalafallah/09s-dinov2-large-cityflowv2` | not recorded in repo |
| DINOv2 standalone MTMC best with AFLink | N/A | N/A | N/A | N/A | 0.744 | 0.547 | 0.624 | not recorded in repo | `yahiaakhalafallah/mtmc-10c-dinov2-stages-4-5-association-eval` v2 | not recorded in repo |
| 14g tertiary 4-view TTA anchor | N/A | N/A | N/A | N/A | 0.77902 | not recorded in repo | not recorded in repo | 154 | `yahiaakhalafallah/14g-dinov2-4view-tta-stage2` | not recorded in repo |

### Provenance

| Field | Value |
|---|---|
| training notebook path | not present under `notebooks/kaggle/` in this repo; producing kernel is recorded in docs and registry |
| Kaggle training kernel slug | `yahiaakhalafallah/09s-dinov2-large-cityflowv2` |
| verifier kernel slug | `yahiaakhalafallah/mtmc-10c-dinov2-stages-4-5-association-eval` for standalone MTMC; `yahiaakhalafallah/14g-dinov2-4view-tta-stage2` for tertiary TTA saturation |
| date of best result | 2026-04-25 for 09s ReID and standalone MTMC; 2026-05-08 for 14g tertiary TTA check |
| author/account | `yahiaakhalafallah` |

### Known Limitations

- Higher single-camera mAP did not transfer to better standalone MTMC; the best DINOv2 standalone MTMC result was 0.744 IDF1 despite mAP=86.79%.
- The deployed tertiary checkpoint hosting remains partially unresolved in inventory docs: the producing kernel is known, but the hosted dataset was unresolved in the inventory pass.
- 14g showed the tertiary TTA stream is saturated: adding scale views did not change the ID-switch floor relative to the 14e B1 plateau.
