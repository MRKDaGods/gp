# Model Cards

<!-- markdownlint-disable MD013 MD024 MD060 -->

Last verified: 2026-05-17

Inventory cross-reference: [docs/models.md](docs/models.md).

These cards capture per-model recipe and provenance details that are deeper than the checkpoint inventory. Missing fields are intentionally marked as "NOT RECORDED IN REPO — [searched: ...]" rather than inferred.

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
| special heads/tokens | SIE camera embeddings with 20 VeRi cameras, JPM with 4 groups, BNNeck on the 768D global feature before classifier |

### Training Data

| Field | Value |
|---|---|
| dataset name + version | VeRi-776 from `abhyudaya12/veri-vehicle-re-identification-dataset` for verifier access |
| ID count | 576 train IDs; query/gallery use the standard VeRi-776 test split |
| image count | approximately 37,778 train images, 1,678 query images, 11,579 gallery images |
| train/test split | `image_train`, `image_query`, `image_test`; exact original training split metadata beyond the standard directory split is NOT RECORDED IN REPO — [searched: `docs/findings.md`, `docs/experiment-log.md`, `docs/subagent-specs/14q-veri-next.md`, `notebooks/kaggle/09v_veri776_eval/`, `scripts/eval/eval_09v_transreid_veri776.py`] |

### Training Recipe

The repo contains the 09v v17 verifier/evaluation notebook, not the original training notebook. Training fields below are limited to what is recorded in docs and notebook-derived evaluation setup.

| Hyperparameter | Value |
|---|---|
| optimizer + LR + schedule | AdamW family reconstruction; exact betas, weight decay, backbone LR, head LR, and minimum LR are NOT RECORDED IN REPO — [searched: `docs/findings.md`, `docs/experiment-log.md`, `docs/subagent-specs/14q-veri-next.md`, `docs/models.md`, `notebooks/kaggle/09v_veri776_eval/`, `scripts/eval/eval_09v_transreid_veri776.py`] |
| batch size + sampling strategy | effective batch 96 on 2xT4 DataParallel; exact P x K tuple is NOT RECORDED IN REPO — [searched: `docs/subagent-specs/14q-veri-next.md`, `docs/findings.md`, `docs/experiment-log.md`] |
| epochs + warmup | 140 epochs; warmup duration recorded as 10 epochs in the family reconstruction, exact warmup curve is NOT RECORDED IN REPO — [searched: `docs/subagent-specs/14q-veri-next.md`, `docs/findings.md`, `docs/experiment-log.md`] |
| loss functions | CE + triplet family; exact label smoothing epsilon, triplet margin, center-loss usage/weight, and loss weights are NOT RECORDED IN REPO — [searched: `docs/findings.md`, `docs/experiment-log.md`, `docs/subagent-specs/14q-veri-next.md`, `notebooks/kaggle/09v_veri776_eval/`] |
| augmentations | Resize 224x224, RandomHorizontalFlip, Pad+RandomCrop, CLIP normalization, likely RandomErasing; exact full training augmentation stack is NOT RECORDED IN REPO — [searched: `docs/subagent-specs/14q-veri-next.md`, `scripts/eval/eval_09v_transreid_veri776.py`, `notebooks/kaggle/09v_veri776_eval/`] |
| hardware + approximate training time | 2x Kaggle T4 with DataParallel; approximate training time 2.5-3.0 GPU-hours; exact walltime log is NOT RECORDED IN REPO — [searched: `docs/subagent-specs/14q-veri-next.md`, `docs/experiment-log.md`] |

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
| Best R1 row | 98.33 | 99.05 | 99.34 | 85.14 | N/A | N/A | N/A | N/A | `yahiaakhalafallah/09v-veri-776-eval-transreid-rerank` v17 | NOT RECORDED IN REPO — [searched: `docs/experiment-log.md`, `docs/models.md`, `configs/model_registry.yaml`] |
| Best mAP row | 97.80 | 98.45 (recovered from Kaggle kernel `yahiaakhalafallah/09v-veri-776-eval-transreid-rerank` `veri776_eval_results_v10.json`) | 98.81 (recovered from Kaggle kernel `yahiaakhalafallah/09v-veri-776-eval-transreid-rerank` `veri776_eval_results_v10.json`) | 89.97 | N/A | N/A | N/A | N/A | `yahiaakhalafallah/09v-veri-776-eval-transreid-rerank` v17 | NOT RECORDED IN REPO — [searched: `docs/experiment-log.md`, `docs/models.md`, `configs/model_registry.yaml`, Kaggle pull `yahiaakhalafallah/09v-veri-776-eval-transreid-rerank` `kernel-metadata.json`, `09v-veri-776-eval-transreid-rerank.ipynb`, `veri776_eval_results_v10.json`] |
| Joint row | 98.15 | 98.51 (recovered from Kaggle kernel `yahiaakhalafallah/09v-veri-776-eval-transreid-rerank` `veri776_eval_results_v10.json`) | 98.75 (recovered from Kaggle kernel `yahiaakhalafallah/09v-veri-776-eval-transreid-rerank` `veri776_eval_results_v10.json`) | 89.71 | N/A | N/A | N/A | N/A | `yahiaakhalafallah/09v-veri-776-eval-transreid-rerank` v17 | NOT RECORDED IN REPO — [searched: `docs/experiment-log.md`, `docs/models.md`, `configs/model_registry.yaml`, Kaggle pull `yahiaakhalafallah/09v-veri-776-eval-transreid-rerank` `kernel-metadata.json`, `09v-veri-776-eval-transreid-rerank.ipynb`, `veri776_eval_results_v10.json`] |

### Provenance

| Field | Value |
|---|---|
| training notebook path | NOT RECORDED IN REPO — [searched: `notebooks/kaggle/09v_veri776_eval/`, `docs/models.md`, `configs/model_registry.yaml`]; verifier path is [notebooks/kaggle/09v_veri776_eval/09v-veri776-eval.ipynb](notebooks/kaggle/09v_veri776_eval/09v-veri776-eval.ipynb) |
| Kaggle training kernel slug | original training slug NOT RECORDED IN REPO — [searched: `configs/model_registry.yaml`, `docs/models.md`, `docs/experiment-log.md`]; private/inaccessible `mrkdagods/09v-veri776-transreid` appears only in verification notes |
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
| image count | 37,681 train images, 1,678 query images, 11,579 gallery images |
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
- The paper-claim gap is plausibly tied to unavailable TinyCLIP weights and P100 batch/BN constraints, but the exact missing ingredient and R20 are NOT RECORDED IN REPO — [searched: `notebooks/kaggle/13_clip_senet_train/`, `docs/findings.md`, `docs/experiment-log.md`, Kaggle pull `yahiaakhalafallah/13-clip-senet-train` `vehicle_clip_senet_veri776_metadata.json`, `kernel-metadata.json`, `13-clip-senet-train.ipynb`].

## cityflow_transreid

### Identity

| Field | Value |
|---|---|
| model_id | `cityflow_transreid` |
| task | CityFlowV2 single-camera vehicle ReID reference and primary MTMC feature extractor |
| registry location | [configs/model_registry.yaml#L74](configs/model_registry.yaml#L74) |
| checkpoint filename | `transreid_cityflowv2_best.pth` |
| dataset | CityFlowV2 |
| headline metric | mAP=81.53%, post-rerank mAP=82.80%, R1=92.41%; EMA branch mAP=81.44%, R1=92.74% |

### Architecture

| Field | Value |
|---|---|
| backbone | TransReID ViT-B/16 with OpenAI CLIP backbone `vit_base_patch16_clip_224.openai`, 256x256 |
| embedding dim | 768D before downstream Stage-2 pooling; deployed Stage-2 path uses CLS + GeM patches to 1536D then PCA to 384D for association |
| special heads/tokens | SIE enabled for 59 cameras, JPM enabled with 4 groups, BNNeck |

### Training Data

| Field | Value |
|---|---|
| dataset name + version | CityFlowV2 / AI City Challenge 2022 Track 1 |
| ID count | 466 train IDs and 200 eval IDs (recovered from Kaggle kernel `gumfreddy/09-vehicle-reid-cityflowv2-augoverhaul-ema` `vehicle_reid_cityflowv2_metadata.json`) |
| image count | 38,537 train images, 909 query images, 15,472 gallery images (recovered from Kaggle kernel `gumfreddy/09-vehicle-reid-cityflowv2-augoverhaul-ema` `vehicle_reid_cityflowv2_metadata.json`) |
| train/test split | CityFlowV2 ReID benchmark over the kernel's 59 listed cameras; exact split file is NOT RECORDED IN REPO — [searched: `exported_models/vehicle_reid_cityflowv2_metadata.json`, `configs/model_registry.yaml`, `docs/findings.md`, `docs/models.md`, Kaggle pull `gumfreddy/09-vehicle-reid-cityflowv2-augoverhaul-ema` `vehicle_reid_cityflowv2_metadata.json`] |

### Training Recipe

| Hyperparameter | Value |
|---|---|
| optimizer + LR + schedule | AdamW; backbone_lr=3.5e-4, head_lr=3.5e-3, LLRD=0.65, cosine schedule with 10-epoch linear warmup |
| batch size + sampling strategy | P=16, K=4, total batch=64; eval batch=32 |
| epochs + warmup | 120 epochs, 10-epoch linear warmup; AMP fp16 |
| loss functions | CE with label smoothing epsilon=0.05, triplet margin=0.3, center loss weight=5e-4 delayed until epoch 15, EMA decay=0.999 |
| augmentations | AugOverhaul: RandomGrayscale p=0.1, ColorJitter brightness=0.3 contrast=0.25 saturation=0.2 hue=0.05, GaussianBlur k=5 p=0.2, RandomPerspective distortion=0.1 p=0.2, extended RandomErasing, RandomHorizontalFlip, Pad+RandomCrop, ImageNet normalization |
| hardware + approximate training time | single Kaggle T4, approximately 3-4h training time |

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
| Single-camera CityFlowV2 reference | 92.41 | NOT RECORDED IN REPO — [searched: `exported_models/vehicle_reid_cityflowv2_metadata.json`, `docs/findings.md`, `configs/model_registry.yaml`, Kaggle pull `gumfreddy/09-vehicle-reid-cityflowv2-augoverhaul-ema` `vehicle_reid_cityflowv2_metadata.json`] | NOT RECORDED IN REPO — [searched: `exported_models/vehicle_reid_cityflowv2_metadata.json`, `docs/findings.md`, `configs/model_registry.yaml`, Kaggle pull `gumfreddy/09-vehicle-reid-cityflowv2-augoverhaul-ema` `vehicle_reid_cityflowv2_metadata.json`] | 81.53 | N/A | N/A | N/A | N/A | `gumfreddy/09-vehicle-reid-cityflowv2-augoverhaul-ema` | NOT RECORDED IN REPO — [searched: `exported_models/vehicle_reid_cityflowv2_metadata.json`, `configs/model_registry.yaml`, `docs/models.md`, Kaggle pull `gumfreddy/09-vehicle-reid-cityflowv2-augoverhaul-ema` `vehicle_reid_cityflowv2_metadata.json`] |
| MTMC 14e B1 stack using this primary stream | N/A | N/A | N/A | N/A | 0.77936 | 0.5747 | NOT RECORDED IN REPO — [searched: `docs/findings.md`, `docs/experiment-log.md`, `docs/_data/kaggle_kernel_summaries.json`] | 154 | `yahiaakhalafallah/14e-tta-fusion-aqe-fic-sweep`; verifier `yahiaakhalafallah/14v-verify-b1-from-yaml` | NOT RECORDED IN REPO — [searched: `docs/findings.md`, `docs/experiment-log.md`, `configs/model_registry.yaml`] |

### Provenance

| Field | Value |
|---|---|
| training notebook path | NOT RECORDED IN REPO — [searched: `notebooks/kaggle/`, `docs/models.md`, `configs/model_registry.yaml`]; Kaggle kernel-only artifact |
| Kaggle training kernel slug | `gumfreddy/09-vehicle-reid-cityflowv2-augoverhaul-ema` |
| verifier kernel slug | `yahiaakhalafallah/14v-verify-b1-from-yaml` reproduces downstream MTMC IDF1=0.77936; standalone single-camera verifier path is NOT RECORDED IN REPO — [searched: `notebooks/kaggle/`, `docs/models.md`, `configs/model_registry.yaml`] |
| date of best result | 2026-04 lineage, promoted into registry on 2026-05-17 |
| author/account | `gumfreddy` |

### Known Limitations

- Single-camera mAP does not predict MTMC IDF1 in this project; stronger DINOv2 mAP regressed as a standalone MTMC stream.
- AugOverhaul improved single-camera mAP by +1.39pp but does not improve MTMC by itself; this checkpoint is deployed primarily for feature-stream diversity in the 14e B1 score-fusion stack.
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
| hardware + approximate training time | T4, runtime_sec=2971.8576 / 49.53 min for the fusion sweep (recovered from Kaggle kernel `yahiaakhalafallah/14t-veri-fusion-clip-senet-x-transreid` `14t_summary.json`) |

### Inference Hyperparameters

| Setting | Value |
|---|---|
| input image size | TransReID 224x224; CLIP-SENet 320x320 |
| feature dim | TransReID 768D selected stream; CLIP-SENet 2048D |
| normalization | L2-normalized parent features and score matrices |
| post-processing | Per-model AQE k=3, score fusion `score = 0.7 * (q_cs @ g_cs.T) + 0.3 * (q_tr @ g_tr.T)`, then k-reciprocal rerank k1=80, k2=15, lambda=0.2 |

### Verified Metrics

| Scope | R1 | R5 | R10 | mAP | IDF1 | HOTA | MOTA | IDSW | Kernel | Verification commit SHA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| Best score-fusion row | 98.45 | 98.75 (recovered from Kaggle kernel `yahiaakhalafallah/14t-veri-fusion-clip-senet-x-transreid` `14t_summary.json`) | 99.05 (recovered from Kaggle kernel `yahiaakhalafallah/14t-veri-fusion-clip-senet-x-transreid` `14t_summary.json`) | 93.30 | N/A | N/A | N/A | N/A | `yahiaakhalafallah/14t-veri-fusion-clip-senet-x-transreid` | NOT RECORDED IN REPO — [searched: `scripts/eval/eval_14t_fusion_veri776.py`, `configs/model_registry.yaml`, `docs/models.md`, Kaggle pull `yahiaakhalafallah/14t-veri-fusion-clip-senet-x-transreid` `14t_summary.json`, `14t_fusion_results.json`, `kernel-metadata.json`, `14t-veri-fusion-clip-senet-x-transreid.ipynb`] |
| Best concat row | 98.27 | 98.57 (recovered from Kaggle kernel `yahiaakhalafallah/14t-veri-fusion-clip-senet-x-transreid` `14t_summary.json`) | 98.93 (recovered from Kaggle kernel `yahiaakhalafallah/14t-veri-fusion-clip-senet-x-transreid` `14t_summary.json`) | 93.19 | N/A | N/A | N/A | N/A | `yahiaakhalafallah/14t-veri-fusion-clip-senet-x-transreid` | NOT RECORDED IN REPO — [searched: `scripts/eval/eval_14t_fusion_veri776.py`, `configs/model_registry.yaml`, `docs/models.md`, Kaggle pull `yahiaakhalafallah/14t-veri-fusion-clip-senet-x-transreid` `14t_summary.json`, `14t_fusion_results.json`, `kernel-metadata.json`, `14t-veri-fusion-clip-senet-x-transreid.ipynb`] |

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

## Detection / Tracking models

### YOLO26m + BoT-SORT

#### Identity

| Field | Value |
|---|---|
| model_id | no standalone registry id; detector role appears under `vehicle_mtmc_14e_b1` |
| task | CityFlowV2 vehicle detection plus single-camera tracking |
| registry location | [configs/model_registry.yaml#L23](configs/model_registry.yaml#L23) |
| checkpoint filename | `models/detection/yolo26m.pt` (44.3 MB) |
| dataset | CityFlowV2 for pipeline use; detector is out-of-box COCO pretrained YOLO26m with no project fine-tuning |
| headline metric | no standalone detector metric recorded; used in the 14e B1 MTMC stack with IDF1=0.77936 |

#### Architecture

| Field | Value |
|---|---|
| backbone | Ultralytics YOLO26m detector plus BoxMOT BoT-SORT tracker |
| embedding dim | N/A for detector/tracker; downstream ReID embeddings are produced in Stage 2 |
| special heads/tokens | BoT-SORT Kalman/data-association logic; no project-trained tracker head recorded |

#### Training Data

| Field | Value |
|---|---|
| dataset name + version | pretrained COCO for YOLO26m; CityFlowV2 videos for inference |
| ID count | N/A for detector; CityFlowV2 track IDs are consumed downstream |
| image count | COCO pretraining image count NOT RECORDED IN REPO — [searched: `configs/datasets/cityflowv2.yaml`, `configs/model_registry.yaml`, `docs/models.md`] |
| train/test split | N/A, no project detector fine-tuning performed |

#### Training Recipe

| Hyperparameter | Value |
|---|---|
| optimizer + LR + schedule | N/A, no project fine-tuning performed; COCO pretraining recipe is NOT RECORDED IN REPO — [searched: `configs/datasets/cityflowv2.yaml`, `docs/models.md`, `configs/model_registry.yaml`] |
| batch size + sampling strategy | N/A, no project fine-tuning performed |
| epochs + warmup | N/A, no project fine-tuning performed |
| loss functions | N/A, no project fine-tuning performed; COCO pretraining losses are NOT RECORDED IN REPO — [searched: `configs/datasets/cityflowv2.yaml`, `docs/models.md`, `configs/model_registry.yaml`] |
| augmentations | N/A, no project fine-tuning performed; COCO pretraining augmentations are NOT RECORDED IN REPO — [searched: `configs/datasets/cityflowv2.yaml`, `docs/models.md`, `configs/model_registry.yaml`] |
| hardware + approximate training time | Kaggle T4 for full pipeline Stage 0-2 run, approximately 3h total; local GPU-heavy detection/tracking is discouraged by project instructions |

#### Inference Hyperparameters

| Setting | Value |
|---|---|
| input image size | 1280x1280 |
| feature dim | N/A |
| normalization | N/A |
| post-processing | YOLO confidence=0.25, NMS IoU=0.65, agnostic NMS=true, fp16; vehicle classes COCO car(2), bus(5), truck(7). BoT-SORT: track_high_thresh=0.25, track_low_thresh=0.05, new_track_thresh=0.25, track_buffer=450, max_age=450, max_obs=460, min_hits=3, match_thresh=0.85, appearance_thresh=0.5, cmc_method=`sof`; tracker ReID uses `models/tracker/osnet_x0_25_msmt17.pt` and is not the main Stage-2 ReID |

#### Verified Metrics

| Scope | R1 | R5 | R10 | mAP | IDF1 | HOTA | MOTA | IDSW | Kernel | Verification commit SHA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| Detector/tracker standalone | N/A | N/A | N/A | NOT MEASURED — [searched: `configs/datasets/cityflowv2.yaml`, `docs/findings.md`, `docs/experiment-log.md`, `configs/model_registry.yaml`] | NOT MEASURED — [searched: `configs/datasets/cityflowv2.yaml`, `docs/findings.md`, `docs/experiment-log.md`, `configs/model_registry.yaml`] | NOT MEASURED — [searched: `configs/datasets/cityflowv2.yaml`, `docs/findings.md`, `docs/experiment-log.md`, `configs/model_registry.yaml`] | NOT MEASURED — [searched: `configs/datasets/cityflowv2.yaml`, `docs/findings.md`, `docs/experiment-log.md`, `configs/model_registry.yaml`] | NOT MEASURED — [searched: `configs/datasets/cityflowv2.yaml`, `docs/findings.md`, `docs/experiment-log.md`, `configs/model_registry.yaml`] | no standalone detector eval kernel recorded | N/A |
| Full vehicle MTMC 14e B1 stack using this detector/tracker | N/A | N/A | N/A | N/A | 0.77936 | 0.5747 | NOT RECORDED IN REPO — [searched: `docs/findings.md`, `docs/experiment-log.md`, `docs/_data/kaggle_kernel_summaries.json`] | 154 | `yahiaakhalafallah/14e-tta-fusion-aqe-fic-sweep`; verifier `yahiaakhalafallah/14v-verify-b1-from-yaml` | NOT RECORDED IN REPO — [searched: `configs/model_registry.yaml`, `docs/findings.md`, `docs/experiment-log.md`] |

#### Provenance

| Field | Value |
|---|---|
| training notebook path | N/A, no project fine-tuning performed |
| Kaggle training kernel slug | N/A, no project fine-tuning performed; registry source training is null |
| verifier kernel slug | full-stack verification via `yahiaakhalafallah/14e-tta-fusion-aqe-fic-sweep` |
| date of best result | 2026-05-07 for 14e B1 stack |
| author/account | detector source author NOT RECORDED IN REPO — [searched: `configs/model_registry.yaml`, `docs/models.md`]; project verification account `yahiaakhalafallah` |

#### Known Limitations

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
| backbone | MVDeTr with ImageNet-pretrained ResNet18 backbone and dilated convolutions |
| embedding dim | N/A detector output is ground-plane detection, not ReID embedding |
| special heads/tokens | `deform_trans` MVDeTr setup with deformable transformer multi-view fusion, heatmap head, offset head, and size head; no ID classification head |

### Training Data

| Field | Value |
|---|---|
| dataset name + version | WILDTRACK via `aryashah2k/large-scale-multicamera-detection-dataset` |
| ID count | N/A for detector training; MVDeTr predicts ground-plane detections without identity classification |
| image count | approximately 400 annotated frames x 7 cameras = 2,800 multi-view images |
| train/test split | WILDTRACK official frame/calibration/annotation layout; exact detector split files beyond the upstream MVDeTr convention are NOT RECORDED IN REPO — [searched: `notebooks/kaggle/12a_wildtrack_mvdetr/`, `docs/models.md`, `docs/pipeline-person.md`, `configs/model_registry.yaml`] |

### Training Recipe

| Hyperparameter | Value |
|---|---|
| optimizer + LR + schedule | Adam, base_lr=7e-4, backbone_lr=7e-5 (base_lr_ratio=0.1), OneCycleLR for 25 epochs, weight_decay=1e-4 |
| batch size + sampling strategy | batch_size=1 |
| epochs + warmup | 25 epochs for 12a v3; best logged detector MODA occurs at epoch 17, while epoch 20 logs MODA=0.921; warmup is implicit via OneCycleLR with no separate warmup epochs (recovered from Kaggle kernel `gumfreddy/12a-wildtrack-mvdetr-training` `MVDeTr/logs/wildtrack/.../log.txt`) |
| loss functions | CornerNet-style focal loss on heatmap plus L1 offset loss plus L1 size loss with size weight 0.1; per-view image losses alpha=1.0 |
| augmentations | enabled via `--augmentation true`; upstream README describes view-coherent affine transformations on per-view inputs with inverse per-view feature maps for multiview coherency, but exact sampled parameter ranges are NOT RECORDED IN REPO — [searched: `notebooks/kaggle/12a_wildtrack_mvdetr/`, `docs/models.md`, `docs/pipeline-person.md`, Kaggle pull `gumfreddy/12a-wildtrack-mvdetr-training` `MVDeTr/README.md`, `MVDeTr/logs/wildtrack/.../log.txt`, `12a-wildtrack-mvdetr-training.ipynb`] |
| hardware + approximate training time | Kaggle T4, logged train/test loop time 23,492.727 sec / 6.53h (recovered from Kaggle kernel `gumfreddy/12a-wildtrack-mvdetr-training` `MVDeTr/logs/wildtrack/.../log.txt`) |

### Inference Hyperparameters

| Setting | Value |
|---|---|
| input image size | WILDTRACK 1920x1080 frames; ResNet18 feature map 240x135 at 512D, `img_reduce=12` per-camera 160x90, `world_reduce=4` world feature 120x360 over 480x1440 grid |
| feature dim | N/A |
| normalization | N/A |
| post-processing | 12a conversion uses ground-plane detections; downstream 12b Kalman tracker selected max_age=2, min_hits=2, distance_gate=25.0, q_std=5.0, r_std=10.0, interpolation_max_gap=1 |

### Verified Metrics

| Scope | R1 | R5 | R10 | mAP | IDF1 | HOTA | MOTA/MODA | IDSW | Kernel | Verification commit SHA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| Detector training-time best | N/A | N/A | N/A | N/A | N/A | N/A | MODA=0.924 at epoch 17; epoch 20 logs MODA=0.921 | N/A | `gumfreddy/12a-wildtrack-mvdetr-training` | NOT RECORDED IN REPO — [searched: `notebooks/kaggle/12a_wildtrack_mvdetr/`, `docs/models.md`, `docs/pipeline-person.md`, Kaggle pull `gumfreddy/12a-wildtrack-mvdetr-training` `kernel-metadata.json`, `12a-wildtrack-mvdetr-training.ipynb`, `MVDeTr/logs/wildtrack/.../log.txt`] |
| Exported loaded-model line | N/A | N/A | N/A | N/A | N/A | N/A | MODA=0.913, MODP=0.818, precision=0.947, recall=0.966 | N/A | `gumfreddy/12a-wildtrack-mvdetr-training` | NOT RECORDED IN REPO — [searched: `notebooks/kaggle/12a_wildtrack_mvdetr/`, `docs/models.md`, `docs/pipeline-person.md`] |
| 12b tracking rerun on 12a v3 detections | N/A | N/A | N/A | N/A | 0.947 | NOT RECORDED IN REPO — [searched: `docs/pipeline-person.md`, `docs/findings.md`, `docs/_data/kaggle_kernel_summaries.json`] | MODA=0.900 | 5 | `gumfreddy/12b-wildtrack-mvdetr-tracking-reid` | NOT RECORDED IN REPO — [searched: `docs/pipeline-person.md`, `docs/findings.md`, `docs/_data/kaggle_kernel_summaries.json`] |

### Provenance

| Field | Value |
|---|---|
| training notebook path | [notebooks/kaggle/12a_wildtrack_mvdetr/12a_wildtrack_mvdetr.ipynb](notebooks/kaggle/12a_wildtrack_mvdetr/12a_wildtrack_mvdetr.ipynb) |
| Kaggle training kernel slug | `gumfreddy/12a-wildtrack-mvdetr-training` |
| verifier kernel slug | `gumfreddy/12b-wildtrack-mvdetr-tracking-reid` for downstream tracking; 12a kernel also runs detector export/eval |
| date of best result | 2026-05 era; exact detector best date NOT RECORDED IN REPO — [searched: `docs/models.md`, `docs/pipeline-person.md`, `docs/_data/kaggle_kernel_summaries.json`, Kaggle pull `gumfreddy/12a-wildtrack-mvdetr-training` `kernel-metadata.json`, `MVDeTr/logs/wildtrack/.../log.txt`] |
| author/account | `gumfreddy` |

### Known Limitations

- The registry keeps both detector values because the training log peaks at MODA=0.924 on epoch 17 and records MODA=0.921 at epoch 20, while the exported loaded-model line verifies MODA=0.913.
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
| headline metric | Fine-tuned CityFlowV2 DINOv2 ReID mAP=86.79%, R1=96.15%; standalone MTMC IDF1=0.744, but tertiary fusion helps the 14e B1 stack |

### Architecture

| Field | Value |
|---|---|
| backbone | Two variants are documented: pure frozen timm `vit_large_patch14_dinov2.lvd142m`, and fine-tuned CityFlowV2 checkpoint `vehicle_transreid_dinov2_large_cityflowv2_final.pth`; 14e B1 production uses the fine-tuned checkpoint, not pure frozen DINOv2 |
| variant roles | Pure frozen DINOv2 is the Meta LVD-142M self-supervised timm backbone with no project training; fine-tuned 09s wraps that backbone for CityFlowV2 vehicle ReID and is the deployed tertiary stream in 14e B1 |
| embedding dim | 1024D recorded for the deployed tertiary stream |
| special heads/tokens | 1024D CLS token stream from DINOv2 ViT-L/14; patch tokens are available in the wrapper but not used in 14e B1; exact fine-tuned head details are NOT RECORDED IN REPO — [searched: `configs/datasets/cityflowv2.yaml`, `src/stage2_features/`, `configs/model_registry.yaml`, `docs/findings.md`] |

### Training Data

| Field | Value |
|---|---|
| dataset name + version | Fine-tuned variant: CityFlowV2. Pure frozen variant: Meta DINOv2 LVD-142M pretraining via timm |
| ID count | CityFlowV2 ReID split ID count for 09s is NOT RECORDED IN REPO — [searched: `configs/model_registry.yaml`, `docs/findings.md`, `docs/models.md`, `configs/datasets/cityflowv2.yaml`] |
| image count | 09s training image count is NOT RECORDED IN REPO — [searched: `configs/model_registry.yaml`, `docs/findings.md`, `docs/models.md`, `configs/datasets/cityflowv2.yaml`] |
| train/test split | CityFlowV2 ReID split; exact split metadata is NOT RECORDED IN REPO — [searched: `configs/model_registry.yaml`, `docs/findings.md`, `docs/models.md`, `configs/datasets/cityflowv2.yaml`] |

### Training Recipe

| Hyperparameter | Value |
|---|---|
| optimizer + LR + schedule | AdamW with layer-wise LR decay 0.75, backbone_lr=1.5e-5, head_lr=1.5e-4, weight_decay=1e-4, 10-epoch linear warmup, cosine decay, plus center-loss SGD lr=0.5 (recovered from Kaggle kernel `yahiaakhalafallah/09s-dinov2-large-cityflowv2` notebook source) |
| batch size + sampling strategy | batch=32, PK sampling P=8 identities x K=4 instances (recovered from Kaggle kernel `yahiaakhalafallah/09s-dinov2-large-cityflowv2` notebook source) |
| epochs + warmup | 120 epochs, best epoch 115/120, 10 warmup epochs (recovered from Kaggle kernel `yahiaakhalafallah/09s-dinov2-large-cityflowv2` notebook source) |
| loss functions | CrossEntropy with label smoothing epsilon=0.05 + batch-hard triplet margin=0.3 + delayed center loss weight=5e-4 starting after warmup at epoch index 10 (recovered from Kaggle kernel `yahiaakhalafallah/09s-dinov2-large-cityflowv2` notebook source) |
| augmentations | Resize 268x268 bicubic, RandomHorizontalFlip p=0.5, Pad(10), RandomCrop 252x252, ColorJitter brightness=0.2 contrast=0.15 saturation=0.1 hue=0.0, CLIP normalization, RandomErasing p=0.5 scale=(0.02,0.33) ratio=(0.3,3.3) value=random (recovered from Kaggle kernel `yahiaakhalafallah/09s-dinov2-large-cityflowv2` notebook source) |
| hardware + approximate training time | Kaggle `NvidiaTeslaT4`; kernel log reaches 11,759.98 sec / 3.27h through notebook HTML conversion, but a dedicated train-loop walltime artifact is NOT RECORDED IN REPO — [searched: Kaggle pull `yahiaakhalafallah/09s-dinov2-large-cityflowv2` `kernel-metadata.json`, `09s-dinov2-large-cityflowv2.log`, `09s-dinov2-large-cityflowv2.ipynb`] |

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
| CityFlowV2 single-camera ReID 09s v1 | 96.15 | NOT RECORDED IN REPO — [searched: `docs/findings.md`, `configs/model_registry.yaml`, `docs/models.md`, Kaggle pull `yahiaakhalafallah/09s-dinov2-large-cityflowv2` `kernel-metadata.json`, `09s-dinov2-large-cityflowv2.log`, `09s-dinov2-large-cityflowv2.ipynb`] | NOT RECORDED IN REPO — [searched: `docs/findings.md`, `configs/model_registry.yaml`, `docs/models.md`, Kaggle pull `yahiaakhalafallah/09s-dinov2-large-cityflowv2` `kernel-metadata.json`, `09s-dinov2-large-cityflowv2.log`, `09s-dinov2-large-cityflowv2.ipynb`] | 86.79 | N/A | N/A | N/A | N/A | `yahiaakhalafallah/09s-dinov2-large-cityflowv2` | NOT RECORDED IN REPO — [searched: `docs/findings.md`, `configs/model_registry.yaml`, `docs/models.md`, Kaggle pull `yahiaakhalafallah/09s-dinov2-large-cityflowv2` `kernel-metadata.json`, `09s-dinov2-large-cityflowv2.log`, `09s-dinov2-large-cityflowv2.ipynb`] |
| DINOv2 standalone MTMC best with AFLink | N/A | N/A | N/A | N/A | 0.744 | 0.547 | 0.624 | NOT RECORDED IN REPO — [searched: `docs/findings.md`, `docs/experiment-log.md`, `docs/_data/kaggle_kernel_summaries.json`] | `yahiaakhalafallah/mtmc-10c-dinov2-stages-4-5-association-eval` v2 | NOT RECORDED IN REPO — [searched: `docs/findings.md`, `docs/experiment-log.md`, `docs/_data/kaggle_kernel_summaries.json`] |
| 14g tertiary 4-view TTA anchor | N/A | N/A | N/A | N/A | 0.77902 | NOT RECORDED IN REPO — [searched: `docs/findings.md`, `docs/experiment-log.md`, `docs/_data/kaggle_kernel_summaries.json`] | NOT RECORDED IN REPO — [searched: `docs/findings.md`, `docs/experiment-log.md`, `docs/_data/kaggle_kernel_summaries.json`] | 154 | `yahiaakhalafallah/14g-dinov2-4view-tta-stage2` | NOT RECORDED IN REPO — [searched: `docs/findings.md`, `docs/experiment-log.md`, `docs/_data/kaggle_kernel_summaries.json`] |

### Provenance

| Field | Value |
|---|---|
| training notebook path | NOT RECORDED IN REPO — [searched: `notebooks/kaggle/`, `configs/model_registry.yaml`, `docs/models.md`, `docs/findings.md`]; producing kernel is recorded in docs and registry |
| Kaggle training kernel slug | `yahiaakhalafallah/09s-dinov2-large-cityflowv2` |
| verifier kernel slug | `yahiaakhalafallah/mtmc-10c-dinov2-stages-4-5-association-eval` for standalone MTMC; `yahiaakhalafallah/14g-dinov2-4view-tta-stage2` for tertiary TTA saturation |
| date of best result | 2026-04-25 for 09s ReID and standalone MTMC; 2026-05-08 for 14g tertiary TTA check |
| author/account | `yahiaakhalafallah` |

### Known Limitations

- Higher single-camera mAP did not transfer to better standalone MTMC; the best DINOv2 standalone MTMC result was 0.744 IDF1 despite mAP=86.79%.
- The deployed tertiary checkpoint is hosted only via the producing kernel's output dataset yahiaakhalafallah/09s-dinov2-large-cityflowv2; it is not mirrored into any */mtmc-weights aggregate dataset, so downloads must reference the producing kernel directly.
- 14g showed the tertiary TTA stream is saturated: adding scale views did not change the ID-switch floor relative to the 14e B1 plateau.
