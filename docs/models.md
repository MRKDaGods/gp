# Models & Checkpoints - Canonical Reference

> Last verified: 2026-05-15
> Source data: docs/_data/kernel_inventory.json, docs/_data/checkpoint_inventory.json

## Section 1 - Active deployed models

### Vehicle pipeline (CityFlowV2, target MTMC IDF1=0.77936)

#### Detector - YOLO26m

- Architecture: YOLO26m
- Local: models/detection/yolo26m.pt (44,255,705 bytes; 44.3 MB decimal)
- Provenance: pretrained COCO, no project fine-tuning found
- Hosted: yahiaakhalafallah/mtmc-weights, gumfreddy/mtmc-weights, mrkdagods/mtmc-weights (identical 44,255,705 bytes)
- Used by: src/stage1_tracking/pipeline.py

#### Primary ReID - TransReID ViT-B/16 CLIP 256px

- Local file: models/reid/transreid_cityflowv2_best.pth (346,518,635 bytes; 346.5 MB decimal)
- Hosted: yahiaakhalafallah/mtmc-weights, gumfreddy/mtmc-weights, mrkdagods/mtmc-weights as reid/transreid_cityflowv2_best.pth
- Source training kernel: gumfreddy/09-vehicle-reid-cityflowv2-augoverhaul-ema, selected by highest verified candidate raw mAP among the three 09-* candidates
- Verified metric: mAP=0.8152743047017524, R1=UNVERIFIED (cite: gumfreddy/09-vehicle-reid-cityflowv2-augoverhaul-ema/exported_models/vehicle_reid_cityflowv2_metadata.json:44)
- Additional parsed claims: best_ema_mAP=0.8144469802052257, best_mAP_rr=0.8279662735373268, best_ema_mAP_rr=0.8355198487541973 (same metadata lines 45-47)
- Embedding dim: 768D
- Used at: Stage 2 primary feature, w_primary=0.475 in 14e B1
- Yaml: configs/datasets/cityflowv2.yaml lines 95-105

#### Tertiary ReID - DINOv2 ViT-L/14

- Local file: models/reid/dinov2_large_cityflowv2.pth (NOT ON LOCAL DISK)
- Deployed Kaggle file: vehicle_transreid_dinov2_large_cityflowv2_final.pth
- Hosted: DATASET UNRESOLVED. Visible datasets across gumfreddy, mrkdagods, ali369, and yahiaakhalafallah only expose the mtmc-weights datasets, and none contains the DINOv2 tertiary checkpoint. The deployed path is a Kaggle notebook output source: /kaggle/input/09s-dinov2-large-cityflowv2/vehicle_transreid_dinov2_large_cityflowv2_final.pth.
- Source training: yahiaakhalafallah/09s-dinov2-large-cityflowv2 -> Kaggle notebook output source -> Stage 2 tertiary file. The local notebook defines vehicle_transreid_dinov2_large_cityflowv2_best.pth, vehicle_transreid_dinov2_large_cityflowv2_final.pth, and vehicle_transreid_dinov2_large_cityflowv2_summary.json.
- Verified metric: UNVERIFIED. The kernel log was accessible but contained no mAP/R1 lines; the output API paged through crop images before the summary JSON and did not expose it in this pass.
- Embedding dim: 1024D
- Used at: Stage 2 tertiary, w_tertiary=0.525 in 14e B1
- Yaml: configs/datasets/cityflowv2.yaml lines 120-130
- Download: kaggle kernels output yahiaakhalafallah/09s-dinov2-large-cityflowv2 --file-pattern '^vehicle_transreid_dinov2_large_cityflowv2_final\.pth$' -p models/reid/

#### Secondary ReID (DISABLED) - ResNet101-IBN-a CityFlowV2

- Local: models/reid/resnet101ibn_cityflowv2_384px_best.pth (171,701,980 bytes)
- Hosted: gumfreddy/mtmc-weights and mrkdagods/mtmc-weights as reid/resnet101ibn_cityflowv2_384px_best.pth
- Source: historical 09d v18 ali369 record; exact primary output was not visible in this inventory pass
- Verified metric: UNVERIFIED for the exact hosted checkpoint. Current accessible follow-up logs report a loaded previous best mAP=0.5061, while docs/findings.md records the historical 52.77% claim.
- Status: DISABLED in 14e B1 (too weak; w_secondary=0.0)

### Vehicle MTMC fusion configs (deployed states)

| Config tag | MTMC IDF1 | Stage 4 knobs | Source kernel |
|---|---:|---|---|
| 10c v15 (previous baseline) | 0.7703 | aqe_k=3, w_t=0.60, sim_thr=0.55 | 10c v15 |
| 14e B1 (current best) | 0.77936 | aqe_k=2, w_t=0.525, sim_thr=0.48, fic_reg=0.5 | yahiaakhalafallah/14e-tta-fusion-aqe-fic-sweep (verified from 14e_summary.json) |

### Person pipeline (WILDTRACK, target ground-plane IDF1=0.947)

#### Detector - MVDeTr ResNet18

- Local: NOT ON DISK (kernel output only)
- Hosted: gumfreddy/12a-wildtrack-mvdetr-training (kernel output)
- Source: 12a kernel
- Verified metric: exported loaded-model log line reports MODA=0.913, MODP=0.818, precision=0.947, recall=0.966; epoch-20 line reports MODA=0.921 but is not the final exported-checkpoint line
- Output filename: MultiviewDetector.pth
- Download: kaggle kernels output gumfreddy/12a-wildtrack-mvdetr-training -p models/person_detection/

#### Tracker - BoT-SORT Kalman (no model file)

- Best params (verified from 12b tracking_sweep_best.json): max_age=2, min_hits=2, distance_gate=25.0, q_std=5.0, r_std=10.0, interpolation_max_gap=1
- Verified IDF1: 0.9467 (ground-plane)
- Yaml: configs/datasets/wildtrack.yaml

## Section 2 - Active fusion approaches

### 14t - VeRi-776 single-cam SOTA (CLIP-SENet x TransReID score fusion)

- mAP=0.93304, R1=0.98451 (verified in docs/experiment-log.md from 14t_summary.json)
- Source kernel: yahiaakhalafallah/14t-veri-fusion-clip-senet-x-transreid
- Recipe: CLIP-SENet v6 + TransReID 09v v17, score fusion 0.7/0.3, AQE k=3 + rerank (k1=80, k2=15, lambda=0.2)
- Does NOT improve CityFlow MTMC: 14u Option C port failed at 0.77995 vs 14e B1's 0.77936 baseline.
- This is a single-camera workflow on VeRi-776 only.

## Section 3 - TODO - see next subagent commit

- Approach catalog (every TransReID/CLIP-SENet/DINOv2/ResNet variant)
- Dead-end summary
- System integration map (yaml -> backend -> frontend)
- Reproduction recipes (link to pipeline-vehicle.md, pipeline-person.md)# Models & Checkpoints - Canonical Reference

> Last verified: 2026-05-15. Verification used Kaggle metadata, Kaggle kernel logs / small JSON outputs, and local notebook source only. No checkpoint downloads or local pipeline stages were run.

## Kaggle Inventory

### Datasets With Checkpoint-Like Files

| Slug | Verified files |
| --- | --- |
| `yahiaakhalafallah/mtmc-weights` | `detection/yolo26m.pt` 44,255,705 bytes; `reid/transreid_cityflowv2_best.pth` 346,518,635; `reid/person_transreid_vit_base_market1501.pth` 347,713,209; `reid/vehicle_osnet_veri776.pth` 30,159,023; `reid/vehicle_transreid_vit_base_veri776.pth` 346,889,637; PCA files and metadata. Does not include the DINOv2 tertiary checkpoint. |
| `gumfreddy/mtmc-weights` | `detection/yolo26m.pt` 44,255,705 bytes; `reid/transreid_cityflowv2_best.pth` 346,518,635; `reid/resnet101ibn_cityflowv2_384px_best.pth` 171,701,980; `reid/person_transreid_vit_base_market1501.pth` 345,390,449; PCA files and metadata. Does not include the DINOv2 tertiary checkpoint. |
| `mrkdagods/mtmc-weights` | `detection/yolo26m.pt` 44,255,705 bytes; `reid/transreid_cityflowv2_best.pth` 346,518,635; `reid/resnet101ibn_cityflowv2_384px_best.pth` 171,701,980; `reid/transreid_cityflowv2_256px_dmt_best.pth` 347,199,765; `reid/transreid_cityflowv2_384px_best.pth` 345,353,127; `reid/person_transreid_vit_base_market1501.pth` 345,390,449. No `vehicle_osnet_veri776.pth`. |
| `gumfreddy/09p-r50-ibn-cityflowv2-extended-checkpoint` | `fastreid_r50_ibn_cityflowv2_extended_final.pth` 99,857,872 bytes. |

### Kernel Outputs Used As Sources

| Slug | Verified output evidence |
| --- | --- |
| `yahiaakhalafallah/14c-tta-stage2` | `14c_summary.json` downloaded: Stage 2 TTA feature build, MTMC IDF1 0.770846, primary 4-view TTA and DINOv2 2-view TTA. |
| `yahiaakhalafallah/14e-tta-fusion-aqe-fic-sweep` | `14e_summary.json` downloaded: B1 is MTMC IDF1 0.7793596227569698 at `w_tertiary=0.525`, `similarity_threshold=0.48`, `aqe_k=2`, `fic_regularisation=0.5`. |
| `gumfreddy/12a-wildtrack-mvdetr-training` | Kernel log downloaded. It confirms `MultiviewDetector.pth` export and final loaded-model test line `moda: 91.3%`; it also contains an epoch-20 line `moda: 92.1%`. No `ground_plane_eval_summary.json` was produced by this notebook run because repo conversion failed. |
| `gumfreddy/12b-wildtrack-mvdetr-tracking-reid` | `evaluation_summary.json`, `tracking_sweep_best.json`, and `reid_merge_sweep_best.json` downloaded. Best ground-plane IDF1 is 0.9467084639498433 with Kalman params below. |
| `yahiaakhalafallah/09s-dinov2-large-cityflowv2` | Kernel output metadata pages confirm many output crop files; exact summary JSON was not reachable via the first paged output results. Notebook source confirms checkpoint filenames, but the 86.79% metric remains unverified from primary output in this pass. |

The ali_369 token listed no owned datasets. Switching via `KAGGLE_API_TOKEN` worked for Yahia, Gumfreddy, and MRKDaGods.

## Vehicle Pipeline (CityFlowV2 -> MTMC IDF1 = 0.77936)

### Detector: YOLO26m

- Local: `models/detection/yolo26m.pt`, 44,255,705 bytes on disk.
- Kaggle: present in `yahiaakhalafallah/mtmc-weights`, `gumfreddy/mtmc-weights`, and `mrkdagods/mtmc-weights` as `detection/yolo26m.pt` with the same size.
- Provenance: standard Ultralytics YOLO26m COCO checkpoint, no project fine-tuning found.
- Used by: Stage 1 vehicle detection in [configs/datasets/cityflowv2.yaml](../configs/datasets/cityflowv2.yaml).

### Primary ReID: TransReID ViT-B/16 CLIP 256px

- Local: `models/reid/transreid_cityflowv2_best.pth`, 346,518,635 bytes on disk.
- Kaggle: `gumfreddy/mtmc-weights/reid/transreid_cityflowv2_best.pth` and matching files in `yahiaakhalafallah/mtmc-weights` and `mrkdagods/mtmc-weights`.
- Used at: Stage 2 primary ReID; in 14e B1 the effective Stage 4 primary score weight is 0.475.
- Claimed metric: mAP 80.14%, R1 92.27% in [docs/experiment-log.md](experiment-log.md) and [docs/findings.md](findings.md).
- Primary-source status: UNVERIFIED for this pass. The requested notebook, [notebooks/kaggle/09_vehicle_reid_cityflowv2/09_vehicle_reid_cityflowv2.ipynb](../notebooks/kaggle/09_vehicle_reid_cityflowv2/09_vehicle_reid_cityflowv2.ipynb), is not the 80.14% source in its current kernel output. Its downloaded `vehicle_reid_cityflowv2_metadata.json` says `v4_circleloss_ablation` with `best_mAP=0.1844796256388448`.
- Reconciled conclusion: the deployed file exists and is used by the pipeline, but the 80.14% / 92.27% claim must be treated as a documented historical claim until the actual 09b v2 primary output is found.

### Tertiary ReID: DINOv2 ViT-L/14

- Local: `models/reid/dinov2_large_cityflowv2.pth` is missing locally.
- Kaggle: expected as kernel output `yahiaakhalafallah/09s-dinov2-large-cityflowv2/vehicle_transreid_dinov2_large_cityflowv2_final.pth`.
- Trained by: [notebooks/kaggle/09s_dinov2_large/09s_dinov2_large_cityflowv2.ipynb](../notebooks/kaggle/09s_dinov2_large/09s_dinov2_large_cityflowv2.ipynb), Cell 5 (`242772ef`) defines `vehicle_transreid_dinov2_large_cityflowv2_best.pth`, `vehicle_transreid_dinov2_large_cityflowv2_final.pth`, and the summary path; Cell 9 (`97c55d77`) saves the best state; Cell 11 (`d838dc3f`) exports the final state.
- Claimed metric: mAP 86.79%, R1 96.15%, best epoch 115/120.
- Primary-source status: UNVERIFIED for this pass. The metric is recorded in [docs/experiment-log.md](experiment-log.md), but the Kaggle output API did not expose the summary JSON before many crop-image pages, and no metric line was present in the downloaded log.
- Used at: Stage 2 tertiary stream and Stage 4 `tertiary_embeddings.weight=0.525` in [configs/datasets/cityflowv2.yaml](../configs/datasets/cityflowv2.yaml).

### Secondary ReID: ResNet101-IBN-a CityFlowV2 (Disabled)

- Local: `models/reid/resnet101ibn_cityflowv2_384px_best.pth`, 171,701,980 bytes on disk.
- Kaggle: present in `gumfreddy/mtmc-weights` and `mrkdagods/mtmc-weights` as `reid/resnet101ibn_cityflowv2_384px_best.pth`.
- Used at: disabled `stage2.reid.vehicle2` reference stream in [configs/datasets/cityflowv2.yaml](../configs/datasets/cityflowv2.yaml).
- Claimed best metric: mAP 52.77% from historical `09d v18 ali369`.
- Primary-source status: PARTIALLY VERIFIED / DISCREPANT. The current accessible Gumfreddy 09d kernel log is an extended fine-tune that loads this checkpoint and reports `Previous best mAP: 0.5061`, then finishes `Best mAP: 0.5061` without saving new weights. [docs/experiment-log.md](experiment-log.md) records 52.77% for `09d v18 ali369`, but that exact primary output was not accessible from the current tokens.

### Promoted 14e B1 Association Result

- Feature source: `yahiaakhalafallah/14c-tta-stage2`, verified by downloaded `14c_summary.json`.
- Association/eval source: `yahiaakhalafallah/14e-tta-fusion-aqe-fic-sweep`, verified by downloaded `14e_summary.json`.
- Verified best: B1 `mtmc_idf1=0.7793596227569698`, `trackeval_idf1=0.7946139234279311`, `id_switches=154`.
- Exact B1 params: `w_primary=0.475`, `w_tertiary=0.525`, `similarity_threshold=0.48`, `aqe_k=2`, `fic_regularisation=0.5`.

## Person Pipeline (WILDTRACK -> Ground-Plane IDF1 = 0.947)

### Detector: MVDeTr ResNet18

- Local: `models/person_detection/MultiviewDetector.pth` is not on disk.
- Kaggle: kernel output `gumfreddy/12a-wildtrack-mvdetr-training` exports `/kaggle/working/MultiviewDetector.pth`; the output listing could not be cleanly listed by this Kaggle CLI version, but the downloaded kernel log confirms the export.
- Trained by: [notebooks/kaggle/12a_wildtrack_mvdetr/12a_wildtrack_mvdetr.ipynb](../notebooks/kaggle/12a_wildtrack_mvdetr/12a_wildtrack_mvdetr.ipynb), Cell 6 (`#VSC-698ce737`) runs MVDeTr training for 25 epochs, Cell 7 (`#VSC-46fae1e7`) resolves `MultiviewDetector.pth`, and Cell 8 (`#VSC-731816ad`) copies it to `/kaggle/working/MultiviewDetector.pth`.
- Verified metric nuance: the log contains `moda: 92.1%, modp: 81.7%, prec: 95.7%, recall: 96.4%` at epoch 20, but the final `Test loaded model...` line reports `moda: 91.3%, modp: 81.8%, prec: 94.7%, recall: 96.6%` for the exported run. Treat 92.1% as an epoch-line claim, not as verified exported-checkpoint performance.

### Tracker: Kalman Ground-Plane Tracker (No Model File)

- Trained by / evaluated in: [notebooks/kaggle/12b_wildtrack_tracking_reid/12b_wildtrack_tracking_reid.ipynb](../notebooks/kaggle/12b_wildtrack_tracking_reid/12b_wildtrack_tracking_reid.ipynb).
- Source cells: Cell 5 (`12b-deps-config`) defines the baseline Kalman values; Cell 11 (`VSC-12b-kalman-sweep`) runs the sweep and writes `tracking_sweep_best.json`; Cell 13 (`12b-evaluate`) writes `evaluation_summary.json`.
- Verified best from downloaded `tracking_sweep_best.json` and `evaluation_summary.json`:
  - `max_age=2`
  - `min_hits=2`
  - `distance_gate=25.0`
  - `max_euclidean_cm=200.0`
  - `q_std=5.0`
  - `r_std=10.0`
  - `interpolation_enabled=true`
  - `interpolation_max_gap=1`
  - `detection_conf_threshold=0.25`
- Verified metrics: `idf1=0.9467084639498433`, `moda=0.9002100840336135`, `precision=0.9480249480249481`, `recall=0.957983193277311`, `id_switches=5`.
- Reconciled discrepancy: `distance_gate=20.0`, `q_std=8.0`, `r_std=8.0` are only the 12b baseline values; they are not the selected best. The person branch config already uses the verified best `25.0 / 5.0 / 10.0`, so no config fix is needed.

## Unresolved Unknowns

- The exact primary-source notebook/log for the deployed 80.14% / 92.27% `transreid_cityflowv2_best.pth` was not located. The current `09_vehicle_reid_cityflowv2` kernel output contradicts that claim.
- The 09s DINOv2 metric JSON was not reachable through the paged Kaggle output API within this pass, so 86.79% / 96.15% remains documented but not primary-source verified here.
- The exact `09d v18 ali369` primary output for 52.77% mAP was not available from the current tokens; only the hosted checkpoint file and later Gumfreddy extended-run log were verified.