# Model Reproduction Guide

Single source of truth for verifying every entry in `configs/model_registry.yaml`, locating its weights, and understanding how it is exposed through the app.

## TL;DR

Production deployments are `vehicle_mtmc_14e_b1` for CityFlowV2 vehicle MTMC and `person_mtmc_12b` for WILDTRACK person MTMC. GPU-heavy stages 0-2 are reproduced on Kaggle; local verification is for source-cited metrics, registry validation, API wiring, and CPU-only downstream replay when artifacts already exist.

## Quick Start

```powershell
.\.venv\Scripts\python.exe scripts/verify_registry_numbers.py
.\.venv\Scripts\python.exe -c "from backend.services.model_registry import list_models; print('\n'.join(f'{m.id}\t{m.name}\t{m.status}' for m in list_models(include_dead_ends=True)))"
.\.venv\Scripts\python.exe scripts/run_pipeline.py --config configs/datasets/cityflowv2.yaml --stages 3,4,5 --override stage4.association.query_expansion.k=2 --override stage4.association.tertiary_embeddings.weight=0.525 --override stage4.association.graph.similarity_threshold=0.48 --override stage4.association.fic.regularisation=0.5
```

## How To Add A New Model

1. Add one complete entry to `configs/model_registry.yaml`, including metrics, source citations, checkpoint refs, status, and runnable-local policy.
2. Cite only existing source files or captured Kaggle summary artifacts; do not add uncited metric claims.
3. Run `scripts/validate_model_registry.py` and `scripts/verify_registry_numbers.py --entry-id <entry_id>`.
4. Regenerate `docs/models.generated.md` with `scripts/gen_models_doc.py` and update this guide.
5. Add or update backend smoke coverage so `/api/models`, `/api/models/{entry_id}`, and `/api/pipeline/run` stay in sync.

### Vehicle MTMC 14e B1 production (`vehicle_mtmc_14e_b1`)

**Task**: mtmc_vehicle  
**Dataset**: cityflowv2  
**Status**: production  
**Verified**: yes - mtmc_idf1

**Metrics** (from source-cited registry):
| Metric | Value | Verified | Source |
|---|---:|:---:|---|
| mtmc_idf1 | 0.77936 | ✓ | docs/models.md:L138 |

**Weights**:
- `yolo26m.pt` - hosted at `yahiaakhalafallah/mtmc-weights:yolo26m.pt`, `gumfreddy/mtmc-weights:yolo26m.pt`, `mrkdagods/mtmc-weights:yolo26m.pt` - size: 42.2 MB - local path: `models/detection/yolo26m.pt` (not in repo)
- `transreid_cityflowv2_best.pth` - hosted at `yahiaakhalafallah/mtmc-weights:reid/transreid_cityflowv2_best.pth`, `gumfreddy/mtmc-weights:reid/transreid_cityflowv2_best.pth`, `mrkdagods/mtmc-weights:reid/transreid_cityflowv2_best.pth` - size: 330.5 MB - local path: `models/reid/transreid_cityflowv2_best.pth` (not in repo)
- `vehicle_transreid_dinov2_large_cityflowv2_final.pth` - hosted at `yahiaakhalafallah/09s-dinov2-large-cityflowv2:vehicle_transreid_dinov2_large_cityflowv2_final.pth` - size: unknown - local path: `models/reid/vehicle_transreid_dinov2_large_cityflowv2_final.pth` (not in repo)

**How to verify the headline metric**:
```powershell
.\.venv\Scripts\python.exe scripts/verify_registry_numbers.py --entry-id vehicle_mtmc_14e_b1
```

**How to actually reproduce on Kaggle**:
```powershell
kaggle kernels output yahiaakhalafallah/14e-tta-fusion-aqe-fic-sweep -p _tmp_14e-tta-fusion-aqe-fic-sweep
cat _tmp_14e-tta-fusion-aqe-fic-sweep/14e_summary.json | jq .
```

**How to run through the app**:
- Backend: `POST /api/pipeline/run` with body `{"model_id": "vehicle_mtmc_14e_b1"}`.
- Frontend: select "Vehicle MTMC 14e B1 production" from the inference stage dropdown.
- This model is accepted by the local API resolver with `configs/datasets/cityflowv2.yaml` plus the registry overrides. Keep GPU-heavy stages 0-2 on Kaggle unless running a tiny smoke path on suitable hardware.

### Vehicle MTMC 14k v1 K7 R50-IBN research (`vehicle_mtmc_14k_v1_k7`)

**Task**: mtmc_vehicle  
**Dataset**: cityflowv2  
**Status**: research  
**Verified**: yes - mtmc_idf1

**Metrics** (from source-cited registry):
| Metric | Value | Verified | Source |
|---|---:|:---:|---|
| mtmc_idf1 | 0.78079 | ✓ | docs/models.md:L139 |

**Weights**:
- `yolo26m.pt` - hosted at `yahiaakhalafallah/mtmc-weights:yolo26m.pt`, `gumfreddy/mtmc-weights:yolo26m.pt`, `mrkdagods/mtmc-weights:yolo26m.pt` - size: 42.2 MB - local path: `models/detection/yolo26m.pt` (not in repo)
- `transreid_cityflowv2_best.pth` - hosted at `yahiaakhalafallah/mtmc-weights:reid/transreid_cityflowv2_best.pth`, `gumfreddy/mtmc-weights:reid/transreid_cityflowv2_best.pth`, `mrkdagods/mtmc-weights:reid/transreid_cityflowv2_best.pth` - size: 330.5 MB - local path: `models/reid/transreid_cityflowv2_best.pth` (not in repo)
- `vehicle_transreid_dinov2_large_cityflowv2_final.pth` - hosted at `yahiaakhalafallah/09s-dinov2-large-cityflowv2:vehicle_transreid_dinov2_large_cityflowv2_final.pth` - size: unknown - local path: `models/reid/vehicle_transreid_dinov2_large_cityflowv2_final.pth` (not in repo)
- `final_model.pth` - hosted at `gumfreddy/09p-fastreid-r50-extended-cityflowv2:final_model.pth` - size: unknown - local path: `models/reid/fastreid_r50_ibn_cityflowv2_final.pth` (not in repo)

**How to verify the headline metric**:
```powershell
.\.venv\Scripts\python.exe scripts/verify_registry_numbers.py --entry-id vehicle_mtmc_14k_v1_k7
```

**How to actually reproduce on Kaggle**:
```powershell
# No notebook_or_kernel_ref is recorded for this registry entry.
# Use the cited local source line instead: docs/models.md:L139.
```

**How to run through the app**:
- Backend: `POST /api/pipeline/run` with body `{"model_id": "vehicle_mtmc_14k_v1_k7"}`.
- Frontend: select "Vehicle MTMC 14k v1 K7 R50-IBN research" from the inference stage dropdown.
- This model is not runnable locally; reproduce via the cited source `docs/models.md:L139` because no Kaggle kernel ref is recorded.

### Person MTMC 12b production (`person_mtmc_12b`)

**Task**: mtmc_person  
**Dataset**: wildtrack  
**Status**: production  
**Verified**: yes - idf1_groundplane, moda_groundplane

**Metrics** (from source-cited registry):
| Metric | Value | Verified | Source |
|---|---:|:---:|---|
| idf1_groundplane | 0.947 | ✓ | docs/paper-draft.md:L198 |
| moda_groundplane | 0.903 | ✓ | docs/paper-draft.md:L198 |

**Weights**:
- `MultiviewDetector.pth` - hosted at `gumfreddy/12a-wildtrack-mvdetr-training:MultiviewDetector.pth` - size: unknown - local path: `models/person_detection/MultiviewDetector.pth` (not in repo)

**How to verify the headline metric**:
```powershell
.\.venv\Scripts\python.exe scripts/verify_registry_numbers.py --entry-id person_mtmc_12b
```

**How to actually reproduce on Kaggle**:
```powershell
kaggle kernels output gumfreddy/12b-wildtrack-tracking-reid -p _tmp_12b-wildtrack-tracking-reid
cat _tmp_12b-wildtrack-tracking-reid/<summary>.json | jq .
```

**How to run through the app**:
- Backend: `POST /api/pipeline/run` with body `{"model_id": "person_mtmc_12b"}`.
- Frontend: select "Person MTMC 12b production" from the inference stage dropdown.
- This model is accepted by the local API resolver with `configs/datasets/wildtrack.yaml` plus the registry overrides. Keep GPU-heavy detector work on Kaggle unless running a tiny smoke path on suitable hardware.

### Person detector 12a MVDeTr reference (`person_detector_12a_mvdetr`)

**Task**: detector_only  
**Dataset**: wildtrack  
**Status**: reference  
**Verified**: partial - precision, recall; moda is retained as unverified

**Metrics** (from source-cited registry):
| Metric | Value | Verified | Source |
|---|---:|:---:|---|
| moda | 0.921 | ✗ | docs/models.md:L129 |
| precision | 0.947 | ✓ | docs/models.md:L129 |
| recall | 0.966 | ✓ | docs/models.md:L129 |

**Weights**:
- `MultiviewDetector.pth` - hosted at `gumfreddy/12a-wildtrack-mvdetr-training:MultiviewDetector.pth` - size: unknown - local path: `models/person_detection/MultiviewDetector.pth` (not in repo)

**How to verify the headline metric**:
```powershell
.\.venv\Scripts\python.exe scripts/verify_registry_numbers.py --entry-id person_detector_12a_mvdetr
```

**How to actually reproduce on Kaggle**:
```powershell
kaggle kernels output gumfreddy/12a-wildtrack-mvdetr-training -p _tmp_12a-wildtrack-mvdetr-training
cat _tmp_12a-wildtrack-mvdetr-training/<summary>.json | jq .
```

**How to run through the app**:
- Backend: `POST /api/pipeline/run` with body `{"model_id": "person_detector_12a_mvdetr"}`.
- Frontend: select "Person detector 12a MVDeTr reference" from the inference stage dropdown.
- This model is not runnable locally; reproduce via Kaggle kernel `gumfreddy/12a-wildtrack-mvdetr-training`.

### VeRi-776 14t CLIP-SENet x TransReID fusion (`veri776_14t_fusion`)

**Task**: single_cam_reid  
**Dataset**: veri776  
**Status**: research  
**Verified**: yes - map, r1

**Metrics** (from source-cited registry):
| Metric | Value | Verified | Source |
|---|---:|:---:|---|
| map | 0.9330 | ✓ | docs/models.md:L77 |
| r1 | 0.9845 | ✓ | docs/models.md:L77 |

**Weights**:
- `vehicle_transreid_vit_base_veri776.pth` - hosted at `mrkdagods/mtmc-weights:reid/vehicle_transreid_vit_base_veri776.pth`, `gumfreddy/mtmc-weights:reid/vehicle_transreid_vit_base_veri776.pth` - size: unknown - local path: `models/reid/vehicle_transreid_vit_base_veri776.pth` (not in repo)
- `best.pth` - hosted at `yahiaakhalafallah/13-clip-senet-train:best.pth` - size: unknown - local path: `models/reid/clipsenet_v6_veri776_best.pth` (not in repo)

**How to verify the headline metric**:
```powershell
.\.venv\Scripts\python.exe scripts/verify_registry_numbers.py --entry-id veri776_14t_fusion
```

**How to actually reproduce on Kaggle**:
```powershell
kaggle kernels output yahiaakhalafallah/14t-veri-fusion-clip-senet-x-transreid -p _tmp_14t-veri-fusion-clip-senet-x-transreid
cat _tmp_14t-veri-fusion-clip-senet-x-transreid/14t_summary.json | jq .
```

**How to run through the app**:
- Backend: `POST /api/pipeline/run` with body `{"model_id": "veri776_14t_fusion"}`.
- Frontend: select "VeRi-776 14t CLIP-SENet x TransReID fusion" from the inference stage dropdown.
- This model is not runnable locally; reproduce via Kaggle kernel `yahiaakhalafallah/14t-veri-fusion-clip-senet-x-transreid`.

### VeRi-776 09v v17 TransReID baseline (`veri776_09v_v17_transreid`)

**Task**: single_cam_reid  
**Dataset**: veri776  
**Status**: production  
**Verified**: yes - map, r1

**Metrics** (from source-cited registry):
| Metric | Value | Verified | Source |
|---|---:|:---:|---|
| map | 0.8997 | ✓ | docs/models.md:L92 |
| r1 | 0.9833 | ✓ | docs/models.md:L92 |

**Weights**:
- `vehicle_transreid_vit_base_veri776.pth` - hosted at `mrkdagods/mtmc-weights:reid/vehicle_transreid_vit_base_veri776.pth`, `gumfreddy/mtmc-weights:reid/vehicle_transreid_vit_base_veri776.pth` - size: unknown - local path: `models/reid/vehicle_transreid_vit_base_veri776.pth` (not in repo)

**How to verify the headline metric**:
```powershell
.\.venv\Scripts\python.exe scripts/verify_registry_numbers.py --entry-id veri776_09v_v17_transreid
```

**How to actually reproduce on Kaggle**:
```powershell
kaggle kernels output mrkdagods/09v-veri776-transreid -p _tmp_09v-veri776-transreid
cat _tmp_09v-veri776-transreid/<summary>.json | jq .
```

**How to run through the app**:
- Backend: `POST /api/pipeline/run` with body `{"model_id": "veri776_09v_v17_transreid"}`.
- Frontend: select "VeRi-776 09v v17 TransReID baseline" from the inference stage dropdown.
- This model is not runnable locally; reproduce via Kaggle kernel `mrkdagods/09v-veri776-transreid`.

### VeRi-776 CLIP-SENet v6 expert (`veri776_clipsenet_v6`)

**Task**: single_cam_reid  
**Dataset**: veri776  
**Status**: research  
**Verified**: yes - map_post_rerank, r1

**Metrics** (from source-cited registry):
| Metric | Value | Verified | Source |
|---|---:|:---:|---|
| map_post_rerank | 0.9154 | ✓ | docs/models.md:L100 |
| r1 | 0.9732 | ✓ | docs/findings.md:L340 |

**Weights**:
- `best.pth` - hosted at `yahiaakhalafallah/13-clip-senet-train:best.pth` - size: unknown - local path: `models/reid/clipsenet_v6_veri776_best.pth` (not in repo)

**How to verify the headline metric**:
```powershell
.\.venv\Scripts\python.exe scripts/verify_registry_numbers.py --entry-id veri776_clipsenet_v6
```

**How to actually reproduce on Kaggle**:
```powershell
kaggle kernels output yahiaakhalafallah/13-clip-senet-train -p _tmp_13-clip-senet-train
cat _tmp_13-clip-senet-train/<summary>.json | jq .
```

**How to run through the app**:
- Backend: `POST /api/pipeline/run` with body `{"model_id": "veri776_clipsenet_v6"}`.
- Frontend: select "VeRi-776 CLIP-SENet v6 expert" from the inference stage dropdown.
- This model is not runnable locally; reproduce via Kaggle kernel `yahiaakhalafallah/13-clip-senet-train`.

### Dead end vehicle CSLS association (`deadend_vehicle_csls`)

**Task**: mtmc_vehicle  
**Dataset**: cityflowv2  
**Status**: dead_end  
**Verified**: yes - mtmc_idf1_delta_pp

**Metrics** (from source-cited registry):
| Metric | Value | Verified | Source |
|---|---:|:---:|---|
| mtmc_idf1_delta_pp | -34.7 | ✓ | docs/findings.md:L997 |

**Weights**:
- None.

**How to verify the headline metric**:
```powershell
.\.venv\Scripts\python.exe scripts/verify_registry_numbers.py --entry-id deadend_vehicle_csls
```

**How to actually reproduce on Kaggle**:
```powershell
# No notebook_or_kernel_ref is recorded for this tombstone entry.
# Use the cited local source line instead: docs/findings.md:L997.
```

**How to run through the app**:
- Backend: `POST /api/pipeline/run` with body `{"model_id": "deadend_vehicle_csls"}`.
- Frontend: select "Dead end vehicle CSLS association" from the inference stage dropdown when dead-end entries are included.
- This model is not runnable locally; reproduce via the cited source `docs/findings.md:L997` because no Kaggle kernel ref is recorded.

### Dead end vehicle AFLink motion linking (`deadend_vehicle_aflink`)

**Task**: mtmc_vehicle  
**Dataset**: cityflowv2  
**Status**: dead_end  
**Verified**: yes - mtmc_idf1_delta_pp

**Metrics** (from source-cited registry):
| Metric | Value | Verified | Source |
|---|---:|:---:|---|
| mtmc_idf1_delta_pp | -3.82 | ✓ | docs/findings.md:L1044 |

**Weights**:
- None.

**How to verify the headline metric**:
```powershell
.\.venv\Scripts\python.exe scripts/verify_registry_numbers.py --entry-id deadend_vehicle_aflink
```

**How to actually reproduce on Kaggle**:
```powershell
# No notebook_or_kernel_ref is recorded for this tombstone entry.
# Use the cited local source line instead: docs/findings.md:L1044.
```

**How to run through the app**:
- Backend: `POST /api/pipeline/run` with body `{"model_id": "deadend_vehicle_aflink"}`.
- Frontend: select "Dead end vehicle AFLink motion linking" from the inference stage dropdown when dead-end entries are included.
- This model is not runnable locally; reproduce via the cited source `docs/findings.md:L1044` because no Kaggle kernel ref is recorded.