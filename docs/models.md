# Models & Checkpoints - Canonical Reference

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