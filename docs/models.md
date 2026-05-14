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

- Local file: models/reid/vehicle_transreid_dinov2_large_cityflowv2_final.pth (NOT ON LOCAL DISK)
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

## Section 3 - Approach catalog, integration map, and reproduction recipes

### Section 3.1 - Approach catalog: Vehicle ReID

#### TransReID variants

| Variant | Trained on | Best verified mAP / R1 | Source kernel | Deployed? | Notes |
|---|---|---:|---|---|---|
| ViT-B/16 CLIP 256px | CityFlowV2 | 0.8153 / UNVERIFIED | `gumfreddy/09-vehicle-reid-cityflowv2-augoverhaul-ema` | Yes - primary 14e B1 | AugOverhaul+EMA fine-tune; supersedes the older 80.14% documentation claim. R1 was not retained in the verified metadata. |
| ViT-B/16 CLIP 256px | VeRi-776 | 0.8997 / 0.9833 | `mrkdagods` 09v v17 | Yes - via 14t fusion | Checkpoint `vehicle_transreid_vit_base_veri776.pth`; one of the two 14t VeRi-776 experts. |
| ViT-B/16 CLIP 384px | CityFlowV2 | 0.8014 / 0.9227 documented; primary artifact UNVERIFIED | `mrkdagods/09b-vehicle-reid-384px-vit-training-v2` | No - DEAD END | MTMC v43/v44 fell to 0.7585/0.7562, about -2.8pp; 384px features captured viewpoint-specific texture that harmed cross-camera association. |
| ViT-L/14 CLIP | VeRi-776 | 0.8090 / 0.9690 base; 0.8795 / 0.9732 post-rerank | `mrkdagods/14p3-veri-vit-l-14-clip-clean-train` | No - probe only | Larger CLIP TransReID did not beat 09v. The earlier 14p mAP=1.0 signal was treated as overfitting/buggy and not a valid model claim. |

#### CLIP-SENet variants

| Variant | Trained on | Best verified metric | Source kernel | Deployed? | Notes |
|---|---|---:|---|---|---|
| CLIP-SENet v6 320px, P=8, K=8 | VeRi-776 | 0.8234 mAP base; 0.9154 mAP post-rerank | `yahiaakhalafallah/13-clip-senet-train` | Yes - via 14t fusion | Canonical CLIP-SENet checkpoint for VeRi-776 score fusion. |
| CLIP-SENet v7 256px, P=16 | VeRi-776 | 0.8136 mAP / 0.9571 R1 | `yahiaakhalafallah/13-clip-senet-train` v7 and `yahiaakhalafallah/13e-v7-clip-senet-eval` | No - DEAD END | Smaller crops lost fine vehicle texture; post-rerank also regressed versus v6. |
| CLIP-SENet fine-tune | CityFlowV2, 12 epochs | standalone MTMC IDF1=0.7099 | `yahiaakhalafallah/13f-clip-senet-cityflow-finetune` plus `yahiaakhalafallah/13h-clip-senet-ft-fusion` | No - DEAD END | Fusion sweep peaked at 0.7691, still -0.12pp below the 10c v15 production baseline. |

#### DINOv2 variants

| Variant | Trained on | Best verified metric | Source kernel | Deployed? | Notes |
|---|---|---:|---|---|---|
| ViT-L/14 | CityFlowV2 | UNVERIFIED; documented claim 0.8679 mAP / 0.9615 R1 | `yahiaakhalafallah/09s-dinov2-large-cityflowv2` | Yes - tertiary 14e B1 | Upstream filename is `vehicle_transreid_dinov2_large_cityflowv2_final.pth`. The config path is fixed on PR #4 to match that filename. |
| ViT-B/14 | VeRi-776 | 0.8927 mAP / 0.9815 R1 post-rerank | `gumfreddy/14r-probe-dinov2-veri-776-train` | No - probe only | Standalone DINOv2 SSL pretraining underperformed the VeRi-776 CLIP-based experts. |

#### ResNet variants

| Variant | Trained on | Best verified metric | Source kernel | Deployed? | Notes |
|---|---|---:|---|---|---|
| ResNet101-IBN-a 384px | CityFlowV2 | UNVERIFIED; documented claim 0.5277 mAP | `ali369/09d-vehicle-reid-resnet101-ibn-a-training` or `mrkdagods/09d-vehicle-reid-resnet101-ibn-a-training` | No - disabled (`w_secondary=0.0`) | Too weak for ensemble use; later accessible logs only verified a 0.5061 loaded previous best. |
| ResNet101-IBN-a | VeRi-776 | 0.6252 mAP documented | `ali369/09e-vehicle-reid-resnet101-ibn-a-veri-776-pretrain` | No - reference only | VeRi pretraining succeeded, but CityFlowV2 fine-tune regressed to 0.427 mAP. |
| FastReID R50-IBN-a | CityFlowV2 | 0.6364 mAP / 0.7869 R1 documented for 09n; 09p exact metric UNVERIFIED | `gumfreddy/09p-fastreid-r50-extended-cityflowv2` | Warning - 14k quaternary only | 14k K7 reached 0.78079 MTMC IDF1, a marginal +0.0014 over 14e B1, not promoted. |

#### Other approaches tried

- DMT camera-aware training (09g): 43.8% mAP and about -1.4pp MTMC IDF1, too weak.
- ResNeXt101-IBN-a ArcFace (09j): 36.88% mAP; partial/mismatched pretrained weights left large backbone regions effectively random.
- ArcFace on ResNet101-IBN-a (09i): 50.80% mAP; warm-starting CE geometry into angular-margin loss overfit and missed the 52.77% baseline.
- Circle loss plus triplet on ResNet: 16-30% mAP; conflicting gradients and unstable metric-learning recipe.
- SGD for ResNet101-IBN-a: 30.27% mAP; AdamW was essential for these small-data vehicle runs.

### Section 3.2 - Approach catalog: Person

- Detector: MVDeTr ResNet18 from `gumfreddy/12a-wildtrack-mvdetr-training`. The epoch-20 log line claims MODA=92.1%, but the final loaded-model line for the exported run verifies MODA=91.3%, MODP=81.8%, precision=94.7%, recall=96.6%.
- Tracker: BoT-SORT-style Kalman ground-plane tracker. The selected 12b operating point is `max_age=2`, `min_hits=2`, `distance_gate=25.0`, `q_std=5.0`, `r_std=10.0`, plus `interpolation_max_gap=1`, giving verified IDF1=0.9467084639498433.
- Dead-end alternatives: global optimal assignment trailed Kalman by about -3.5pp IDF1; the naive tracker was worse; extended Kalman sweeps across 59+ configs stayed within about +/-0.0004 IDF1 of the same ceiling.

### Section 3.3 - Vehicle MTMC fusion family

| Config | MTMC IDF1 | Source | Status |
|---|---:|---|---|
| 10c v15 production (CLIP+DINOv2, AQE k=3, w_t=0.60, sim_thr=0.55) | 0.7703 | 10c v15 | Previous baseline |
| 14e B1 (TTA + AQE k=2, w_t=0.525, sim_thr=0.48, fic_reg=0.5) | 0.77936 | `14e_summary.json` from `yahiaakhalafallah/14e-tta-fusion-aqe-fic-sweep` | Current best |
| 14k v1 K7 (+R50-IBN quaternary) | 0.78079 | `outputs/14k_extended/14k_extended_summary.json` | MARGINAL, NOT promoted |
| 14u Option C (VeRi-fusion port) | 0.77995 | `tmp_14u_outputs/14u_summary.json` | FAIL - does not transfer |

### Section 3.4 - Dead-end summary

- Reranking: always hurt CityFlowV2 MTMC because k-reciprocal sets include false positives with the current features.
- Feature concatenation: mixed uncalibrated feature spaces and lost the calibrated score-fusion behavior.
- Hierarchical clustering: centroid averaging erased discriminative per-tracklet signal.
- CSLS and CID_BIAS: penalized genuine vehicle-type hubs and distorted FIC-calibrated similarities.
- AFLink: lost -3.8pp to -13.2pp MTMC IDF1 in clean retests; motion consistency is unreliable across non-overlapping CityFlowV2 cameras.
- 384px ViT: lost about -2.8pp MTMC IDF1 because high-resolution features overfit viewpoint-specific texture.
- Network flow solver: lost -0.24pp MTMC IDF1 and increased conflation rather than reducing it.
- Weak secondary fusion: ResNet101-IBN-a, CLIP-SENet CityFlow fine-tune, and VeRi-fusion ports added correlated or weak signal; the best CLIP-SENet fine-tune fusion still sat -0.12pp below production.
- Robust pooling and track-quality filtering: neutral on IDF1 despite lower ID switches, confirming the current plateau is feature-quality limited rather than aggregation-limited.

### Section 3.5 - System integration map

```text
+-------------+                 +---------------------+
| Frontend    | POST /api/...    | Backend (FastAPI)   |
| Next.js     |----------------->| /api/pipeline/run   |
| ATHAR UI    |                  | ?dataset=cityflow   |
+-------------+                  |        |            |
                                 |        v            |
                                 | pipeline_service    |
                                 | builds CLI:         |
                                 | python scripts/     |
                                 |  run_pipeline.py    |
                                 |  --config <yaml>    |
                                 +--------+------------+
                                          |
                                          v
                         +------------------------------+
                         | cityflowv2.yaml (vehicle)    |
                         | stage1: yolo26m + BoT-SORT   |
                         | stage2: TransReID + DINOv2   |
                         | stage4: AQE k=2, w_t=0.525   |
                         +------------------------------+
                                          |
                                          | OR
                                          v
                         +------------------------------+
                         | wildtrack.yaml (person)      |
                         | MVDeTr ResNet18              |
                         | Kalman max_age=2, gate=25    |
                         +------------------------------+
```

Pipeline contracts:

- Vehicle: `dataset=cityflowv2` maps to `--config configs/datasets/cityflowv2.yaml`.
- Person: `dataset=wildtrack` maps to `--config configs/datasets/wildtrack.yaml`.
- Default/manual smoke: use `--config configs/default.yaml` unless a dataset-specific config is required.

Checkpoint locality:

- Local vehicle runs need `models/detection/yolo26m.pt`, `models/reid/transreid_cityflowv2_best.pth`, and `models/reid/vehicle_transreid_dinov2_large_cityflowv2_final.pth`; Stage 0-2 should still run on Kaggle for performance.
- Local person runs need `models/person_detection/MultiviewDetector.pth` only if doing detector inference locally, which is not recommended; the canonical run consumes the Kaggle 12a output from the 12b kernel.
- Kaggle reproduction needs the CityFlowV2 or WILDTRACK datasets mounted, plus the relevant mtmc-weights dataset or upstream checkpoint-output kernels listed above.

### Section 3.6 - Reproduction recipes

- Vehicle Stage 0-6: see [docs/pipeline-vehicle.md](pipeline-vehicle.md).
- Person 12a-12b chain: see [docs/pipeline-person.md](pipeline-person.md).
- 14t VeRi-776 single-cam workflow: mount CLIP-SENet v6 from `yahiaakhalafallah/13-clip-senet-train` and TransReID 09v v17 `vehicle_transreid_vit_base_veri776.pth`; run `yahiaakhalafallah/14t-veri-fusion-clip-senet-x-transreid` with `w_clipsenet=0.7`, `w_transreid=0.3`, AQE k=3, and rerank k1=80/k2=15/lambda=0.2. Verified result is mAP=0.93304 and R1=0.98451.

### Section 3.7 - Provenance gaps

- `vehicle_transreid_dinov2_large_cityflowv2_final.pth` source dataset remains unresolved; only the producing kernel `yahiaakhalafallah/09s-dinov2-large-cityflowv2` is known.
- `transreid_cityflowv2_best.pth` R1 is UNVERIFIED in retained primary logs. The best retained primary-source metric is mAP=0.8153 from AugOverhaul+EMA metadata.
- Several `09d-*` and `09p-*` extended-training claims are not present in retained primary log fragments; keep them documented claims unless the exact output JSON/log is recovered.
- The inventory spans four Kaggle accounts; some kernels may be archived/private and absent from the visible 154-kernel inventory.
- `copilot-instructions.md` still contains older vehicle mAP wording in places; this pass intentionally leaves agent instructions unchanged and records the corrected number here.