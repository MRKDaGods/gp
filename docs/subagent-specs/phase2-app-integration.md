# Phase 2 — Application Integration of Verified Models & Pipelines

## 1. Status / Scope / Non-Goals

### Status
- Baseline: Phase 1 on master `776a2ff`.
- Backend: FastAPI under `backend/`.
- Frontend: Next.js ATHAR under `frontend/`.
- Offline pipeline: Stage 0-6 MTMC code under `src/`, invoked by `scripts/run_pipeline.py`.
- Registry: `configs/model_registry.yaml`, loaded by `backend/services/model_registry.py`.
- Registry API: `backend/routers/models.py` exposes `GET /api/models` and `GET /api/models/{model_id}`.
- App wiring: `backend/app.py` currently registers health, locations, results, tracklets, export, crops, videos, detections, frames, runs, datasets, models, pipeline, search, and timeline routers.
- Pipeline service: `backend/services/pipeline_service.py` already has `DATASET_CONFIG_BY_NAME` and `DATASET_TASK_BY_NAME`, including `veri776: single_cam_reid`.
- Frontend root: `frontend/src/app/` currently has only `layout.tsx`, `page.tsx`, `providers.tsx`, and `globals.css`.
- Frontend registry UI: `frontend/src/components/ModelPicker.tsx` and `frontend/src/services/models.ts` already exist.

### Verified models to expose
| Model id | Task type | Dataset | Verified metric | Existing eval entry point |
| --- | --- | --- | --- | --- |
| `veri776_09v_v17_transreid` | `single_cam_reid` | `veri776` | R1=98.33, mAP=89.97 | `scripts/eval/eval_09v_transreid_veri776.py` |
| `veri776_clipsenet_v6` | `single_cam_reid` | `veri776` | cosine mAP=82.34, rerank+AQE mAP=91.54 | `scripts/eval/eval_clip_senet_veri776.py` |
| `veri776_14t_fusion` | `single_cam_reid` | `veri776` | mAP=93.30 | `scripts/eval/eval_14t_fusion_veri776.py` |
| CityFlow TransReID | `mtmc_vehicle` | `cityflowv2` | mAP=81.53 | `scripts/eval_cityflowv2_reid.py` |
| `person_detector_12a_mvdetr` | `detector_only` | `wildtrack` | MODA=0.913 | registry + WILDTRACK verifier artifacts |

### Scope
- Add backend serving design for VeRi-776 single-camera ReID.
- Add backend serving design for score fusion over 2+ `single_cam_reid` models.
- Add controlled eval job submission and HTTP polling.
- Add frontend routes for ReID playground, fusion playground, and eval runner.
- Add global frontend dataset switching for CityFlowV2 and WILDTRACK.
- Reuse verified eval-script model construction and feature extraction helpers.

### Non-goals
- No training, fine-tuning, Kaggle orchestration, GNN edge classifier, GPU multi-tenancy, model sharding, auth, or multi-user support.
- No MVDeTr detector-only endpoint in Phase 2.
- No changes to registry metric values, checkpoint provenance, frame-id conventions, or MOT submission formatting.

## 2. Inventory of What's Already Wired vs. What's Not

| Area | Already wired | Not wired yet | Phase 2 action |
| --- | --- | --- | --- |
| FastAPI app | `backend/app.py` registers existing routers | No ReID router | Add `backend/routers/reid.py` and include it |
| Registry API | `backend/routers/models.py` read-only endpoints | Registry not used for inference validation | Validate through `get_model()` |
| Registry service | `backend/services/model_registry.py` loads YAML | No serving cache | Add `backend/services/reid_service.py` |
| Pipeline service | Dataset/task maps include `veri776: single_cam_reid` | No eval-script dispatch | Add `single_cam_reid` and `score_fusion` eval mapping |
| Backend schemas | `backend/models/requests.py`, `backend/models/registry.py` | No ReID/eval job schemas | Add request schemas and `backend/models/reid.py` |
| Eval helpers | Helpers exist in eval scripts | Helpers are script-local | Promote to `src/serving/reid_loaders.py` |
| Fusion math | `tests/test_stage2/test_14t_fusion_math.py` exists | No endpoint/service fusion test | Reuse deterministic fusion checks |
| Frontend routes | Only root route files | No `/reid`, `/fusion`, `/eval` | Add new app routes |
| ModelPicker | Task/status grouping exists | No explicit task filter or multi-select prop | Extend `ModelPicker.tsx` |
| Store | `frontend/src/store/index.ts` has pipeline/video state | No global dataset selector | Add dataset state/actions |

## 3. Backend Design

### 3.1 New Pydantic request/response schemas in `backend/models/requests.py` and `backend/models/reid.py`
Add small request models to `backend/models/requests.py`.

`ReIDImageInput`:
- `id: str | None`
- `image_base64: str | None`
- `path: str | None`
- `metadata: dict[str, Any] = {}`
- Exactly one of `image_base64` or `path` must be supplied.

`SingleCamReIDRequest`:
- `model_id: str = Field(alias="modelId")`
- `queries: list[ReIDImageInput]`
- `gallery: list[ReIDImageInput]`
- `top_k: int = Field(default=20, alias="topK", ge=1, le=100)`
- `rerank: bool = False`
- `aqe_k: int = Field(default=0, alias="aqeK", ge=0, le=20)`
- `normalize: bool = True`
- Validation: at least one query, at least one gallery image, max 50 queries, max 500 gallery images.

`FusionReIDRequest`:
- `model_ids: list[str] = Field(alias="modelIds")`
- `weights: list[float]`
- same query/gallery/topK/rerank/aqeK fields as `SingleCamReIDRequest`
- Validation: at least two models, matching weight count, all weights finite and non-negative, sum `1.0 ± 1e-6`.

`EvalRunRequest`:
- `model_id: str = Field(alias="modelId")`
- `dataset: str`
- `task_type: str = Field(alias="taskType")`
- `config: dict[str, Any] = {}`
- `use_cpu: bool = Field(default=False, alias="useCpu")`
- `limit: int | None = Field(default=None, ge=1)`
- `notes: str | None = None`

Create `backend/models/reid.py` for large response payloads:
- `ReIDImageRef`: `id`, `source`, `metadata`.
- `ReIDRankedMatch`: `galleryId`, `rank`, `score`, optional `distance`, metadata.
- `ReIDQueryResult`: `queryId`, `matches`, `latencyMs`.
- `SingleCamReIDResponse`: `success`, `modelId`, `device`, `featureDim`, `queryCount`, `galleryCount`, `results`, `latencyMs`.
- `FusionReIDResponse`: same plus `modelIds`, `weights`, optional component score metadata.
- `EvalJobResponse`: `jobId`, `status`.
- `EvalJobStatusResponse`: `jobId`, `status`, timestamps, `error`, `progress`.
- `EvalJobResultResponse`: `jobId`, `status`, `result`.

Use camelCase aliases, matching `PipelineRunRequest` in `backend/models/requests.py`.

### 3.2 New router `backend/routers/reid.py`
Create `backend/routers/reid.py` with `APIRouter(prefix="/api/v1", tags=["reid"])`.

#### `POST /api/v1/reid/single_cam`
Body: `SingleCamReIDRequest`.

Validation order:
- Resolve `modelId` through `backend/services/model_registry.py::get_model()`.
- Return 404 `model_not_found` if absent.
- Return 422 `dead_end_model_not_served` if `status == "dead_end"`.
- Return 422 `unsupported_task_type` if `task_type != "single_cam_reid"`.
- Return 422 or 503 `checkpoint_missing` if required checkpoint refs are absent.
- Return 422 for no query/gallery images or invalid image source fields.
- Return 413 `payload_too_large` if base64 caps are exceeded.
- Return 422 `path_outside_upload_dir` if a path escapes `UPLOAD_DIR`.
- Return 500 `inference_failed` for unexpected model/inference failures.

Response shape:
```json
{
  "success": true,
  "modelId": "veri776_09v_v17_transreid",
  "device": "cuda:0",
  "featureDim": 768,
  "queryCount": 2,
  "galleryCount": 10,
  "results": [
    {"queryId": "q0", "latencyMs": 42.1, "matches": [{"galleryId": "g3", "rank": 1, "score": 0.8123, "distance": 0.1877}]}
  ],
  "latencyMs": 391.6
}
```

#### `POST /api/v1/reid/fusion`
Body: `FusionReIDRequest`.

Rules:
- Support any 2+ registry models with `task_type == "single_cam_reid"` and `status != "dead_end"`.
- Reject non-ReID models, dead-end models, negative weights, non-finite weights, and weights not summing to 1.
- Each component model returns a query-gallery cosine similarity matrix.
- Fuse score matrices, not concatenated feature vectors.

Illustrative fusion math:
```python
fused_scores = sum(weight_i * score_matrix_i for weight_i, score_matrix_i in outputs)
```

Response shape:
```json
{
  "success": true,
  "modelIds": ["veri776_09v_v17_transreid", "veri776_clipsenet_v6"],
  "weights": [0.3, 0.7],
  "device": "cuda:0",
  "queryCount": 1,
  "galleryCount": 10,
  "results": [{"queryId": "q0", "matches": [{"galleryId": "g1", "rank": 1, "score": 0.9012}], "latencyMs": 81.4}],
  "latencyMs": 812.5
}
```

#### `POST /api/v1/eval/run`
Purpose: submit a controlled eval job and return immediately.

Allowed mapping:
- `veri776_09v_v17_transreid` -> `scripts/eval/eval_09v_transreid_veri776.py`
- `veri776_clipsenet_v6` -> `scripts/eval/eval_clip_senet_veri776.py`
- `veri776_14t_fusion` -> `scripts/eval/eval_14t_fusion_veri776.py`
- CityFlow TransReID registry model -> `scripts/eval_cityflowv2_reid.py`

Do not accept arbitrary script paths from the client.

Submit response:
```json
{"jobId": "eval_20260516_abc123", "status": "queued"}
```

#### `GET /api/v1/eval/{job_id}`
Return in-memory or persisted status:
```json
{
  "jobId": "eval_20260516_abc123",
  "status": "running",
  "createdAt": "2026-05-16T12:00:00Z",
  "startedAt": "2026-05-16T12:00:04Z",
  "finishedAt": null,
  "error": null,
  "progress": {"stage": "extracting_features"}
}
```
Return 404 if job is unknown.

#### `GET /api/v1/eval/{job_id}/result`
Return final JSON for succeeded jobs.

Errors:
- 404 unknown job.
- 409 queued/running job.
- 500 failed job with sanitized error.

### 3.3 New service `backend/services/reid_service.py`
Responsibilities:
- Validate model eligibility.
- Load checkpoints from registry refs.
- Cache loaded models with LRU eviction keyed by `model_id`.
- Decode base64 images and load path inputs safely.
- Extract features.
- Rank gallery images by cosine similarity.
- Fuse scores across 2+ models.

Create `src/serving/reid_loaders.py` and promote reusable helpers from:
- `scripts/eval/eval_09v_transreid_veri776.py::build_09v_model`
- `scripts/eval/eval_09v_transreid_veri776.py::extract_09v_features_with_metadata`
- `scripts/eval/eval_09v_transreid_veri776.py::parse_split`
- `scripts/eval/eval_clip_senet_veri776.py::build_clipsenet_model`
- `scripts/eval/eval_clip_senet_veri776.py::extract_clipsenet_features`
- `scripts/eval/eval_clip_senet_veri776.py::parse_veri_split`

Both eval scripts and the service should import from `src/serving/reid_loaders.py` to avoid duplicated model construction and preserve verifier parity.

Recommended mapping:
| Registry id | Loader | Extractor |
| --- | --- | --- |
| `veri776_09v_v17_transreid` | `load_transreid_09v` | `extract_transreid_09v` |
| `veri776_clipsenet_v6` | `load_clipsenet_v6` | `extract_clipsenet_v6` |

LRU cache:
- Env var `REID_MODEL_CACHE_SIZE`, default 2.
- Store `model_id`, model object, device, checkpoint path, feature dim, loaded time, and last-used time.
- Evict by recency.
- On CUDA eviction, delete references and call `torch.cuda.empty_cache()`.

Device selection:
- If `USE_CPU` is truthy, use CPU.
- Else use `cuda:0` when `torch.cuda.is_available()`.
- Else use CPU.
- Include selected device in every response.

Image input safety:
- Accept base64 bytes or a path.
- Resolve path inputs under `UPLOAD_DIR` only.
- Use `Path.resolve().is_relative_to(UPLOAD_DIR.resolve())`.
- Cap base64 payloads at 25MB/image and 50 images/request.
- Return 413 before decode if caps are exceeded.
- Convert decoded images to RGB.

Public functions:
- `load_model(model_id)`
- `extract_features(model, images)`
- `rank_gallery(q_feats, g_feats, rerank, aqe_k)`
- `fuse_scores(models_outputs, weights)`
- `run_single_cam(request)`
- `run_fusion(request)`

Ranking details:
- L2-normalize features before cosine similarity.
- Similarity matrix is `q_feats @ g_feats.T`.
- With rerank disabled, scores must be finite and in `[-1, 1]`.
- If rerank is unsupported, return 422 `rerank_not_supported`.

### 3.4 New service `backend/services/job_service.py`
Use an in-memory `asyncio.Queue` and dict of `Job` dataclasses.

`Job` fields:
- `id`, `status`, `created_at`, `started_at`, `finished_at`
- `request`, `result`, `error`, `progress`

Behavior:
- Persist each state transition to `data/jobs/{job_id}.json`.
- Load job JSONs at startup.
- Mark interrupted `running` jobs as failed after backend restart.
- Spawn one worker task from `app.on_event('startup')`.
- Jobs serialize; this is acceptable for local dev.
- Scale-out path is Redis + RQ or Celery.

Public functions:
- `start_worker()`
- `submit_eval_job(request)`
- `get_job(job_id)`
- `get_job_result(job_id)`
- `run_eval_job(job)`

### 3.5 Extend `backend/services/pipeline_service.py`
Add a separate eval dispatch map for `POST /api/v1/eval/run`.

Task support:
- `single_cam_reid` invokes verified single-model eval scripts.
- `score_fusion` invokes `scripts/eval/eval_14t_fusion_veri776.py`.

Do not run single-camera ReID through `scripts/run_pipeline.py` unless a dedicated verified pipeline config exists.

### 3.6 Register `reid.router` in `backend/app.py`
Implementation edit later:
- Add `reid` to the `from backend.routers import (...)` list.
- Add `app.include_router(reid.router)` near `models.router` and `pipeline.router`.
- Start the job worker in the existing startup event.
- Preserve `_scan_startup_videos()` and `_background_precompute_dataset()`.

### 3.7 Auth & security
- No new auth tier.
- Honor the existing CORS list in `backend/app.py`.
- Validate every `model_id` through `get_model()`.
- Reject `status == "dead_end"`.
- Restrict fusion to `task_type == "single_cam_reid"`.
- Constrain image paths to `UPLOAD_DIR` with resolved-path subtree checks.
- Cap base64 payloads and return 413 for cap violations.
- Do not expose arbitrary eval scripts.
- Log internal exception details server-side and return sanitized 500s.

## 4. Frontend Design

### 4.1 Convert `frontend/src/app/page.tsx` to a navigation hub and add routes
Add routes:
- `frontend/src/app/reid/page.tsx` — single-cam ReID playground.
- `frontend/src/app/fusion/page.tsx` — fusion playground.
- `frontend/src/app/eval/page.tsx` — eval runner.

Route requirements:
- Use `'use client';` where hooks, file uploads, or polling are used.
- Reuse `frontend/src/components/ui/` and `frontend/src/components/layout/`.
- Keep screens compact and operational.
- Root hub links to the existing MTMC workflow plus `/reid`, `/fusion`, and `/eval`.

### 4.2 New components under `frontend/src/components/reid/`
`ImageUploader.tsx`:
- Query and gallery dropzones.
- Multi-file support.
- Base64 conversion for Phase 2a.
- Thumbnail, filename, size, and client-side image validation.

`RankedResults.tsx`:
- Top-K grid per query.
- Query thumbnail beside ranked gallery cards.
- Similarity scores with 3 decimals.
- Optional CMC chart when labels are available.
- Empty, loading, and error states.

`FusionWeightSliders.tsx`:
- Per-model sliders from 0.0 to 1.0.
- Normalized weight display.
- Normalize button.
- Disable submit until weights are finite and sum to 1.

`EvalProgressPanel.tsx`:
- Poll `/api/v1/eval/{job_id}` every 2 seconds by default.
- Stop on succeeded, failed, or cancelled.
- Fetch final JSON via `/api/v1/eval/{job_id}/result`.
- Show status, timestamps, error, and collapsible JSON.

### 4.3 Reuse `ModelPicker.tsx`
Extend `frontend/src/components/ModelPicker.tsx`.

New props:
- `taskType?: ModelTaskType`
- `multiSelect?: boolean`
- `selectedIds?: string[]`
- `onMultiSelect?: (modelIds: string[]) => void`
- `includeReference?: boolean`

Compatibility:
- Existing `selectedId`, `onSelect`, and `onModelChange` behavior must keep working.
- `/reid` uses `taskType="single_cam_reid"` in single-select mode.
- `/fusion` uses `taskType="single_cam_reid"` in multi-select mode.
- `/eval` derives filtering from selected eval type.

### 4.4 Global dataset switcher
Create `frontend/src/components/layout/DatasetSwitcher.tsx`.

Store extension in `frontend/src/store/index.ts`:
- `dataset: 'cityflowv2' | 'wildtrack'`
- `setDataset(dataset): void`

Propagation:
- Pipeline run requests include selected `dataset`.
- ModelPicker task filter maps `cityflowv2` to `mtmc_vehicle`.
- ModelPicker task filter maps `wildtrack` to `mtmc_person`.
- ReID routes explicitly use `single_cam_reid` and can show `veri776` context independently.

### 4.5 New service `frontend/src/services/reid.ts`
Add typed wrappers:
- `runSingleCamReid(request)`
- `runFusionReid(request)`
- `submitEvalJob(request)`
- `fetchEvalJob(jobId)`
- `fetchEvalJobResult(jobId)`

Types:
- `UploadedImagePayload`, `SingleCamReIDRequest`, `FusionReIDRequest`
- `ReIDRankedMatch`, `ReIDQueryResult`, `SingleCamReIDResponse`, `FusionReIDResponse`
- `EvalRunRequest`, `EvalJobResponse`, `EvalJobStatusResponse`, `EvalJobResultResponse`

Base URL caution:
- `frontend/src/services/models.ts` defaults to `http://localhost:8004/api`.
- New endpoints are `/api/v1/...`.
- Normalize the base once so the frontend does not call `/api/api/v1/...`.

## 5. Tests

### 5.1 `tests/integration/test_reid_endpoints.py`
Use pytest with `httpx.AsyncClient` or FastAPI `TestClient`.

Required tests:
- Happy path `POST /api/v1/reid/single_cam` with mocked model and deterministic features.
- Happy path `POST /api/v1/reid/fusion` with two mocked models and known weighted ranking.
- 404 `model_not_found`.
- 422 bad weights: sum not 1, negative, count mismatch.
- 422 no query images and 422 no gallery images.
- 413 payload too large.
- 422 dead-end model.
- 422 unsupported task type using detector-only or mocked non-ReID model.

Fixtures:
- 32x32 RGB PNG as base64.
- Deterministic feature vectors: query `[1,0,0]`, same-id gallery `[1,0,0]`, other gallery `[0,1,0]`.

### 5.2 `tests/integration/test_eval_jobs.py`
Required tests:
- Submit job and assert queued response.
- Poll queued/running/succeeded status.
- Fetch result for succeeded job.
- Failure path with mocked subprocess error.
- Unknown job returns 404.
- Result for unfinished job returns 409.

### 5.3 `tests/e2e/test_phase2_smoke.py`
Purpose:
- Verify route registration and request/response plumbing.
- Avoid real checkpoint loading by mocking service calls.

Smoke endpoints:
- `POST /api/v1/reid/single_cam` with 32x32 dummy image.
- `POST /api/v1/reid/fusion` with two mocked models.
- `POST /api/v1/eval/run`.
- `GET /api/v1/eval/{job_id}`.
- `GET /api/v1/eval/{job_id}/result`.

### 5.4 Unit test for `backend/services/reid_service.py` fusion math
Create `tests/unit/test_reid_service.py` or colocate with existing test style.

Assertions:
- Weighted sum is deterministic.
- Invalid weights raise validation error.
- Ranking order is stable.
- Reuse logic from `tests/test_stage2/test_14t_fusion_math.py` where practical.

## 6. Deployment / Runbook

Backend:
```powershell
.\.venv\Scripts\activate
python -m uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000
```

Frontend:
```powershell
cd frontend
pnpm dev
```

Checkpoint expectations, with `models/reid/README.md` as provenance reference:
| Local path | Purpose | Note |
| --- | --- | --- |
| `models/reid/transreid_09v_best.pth` | Proposed serving alias for VeRi-776 TransReID | ViT-B/16, roughly 340MB |
| `models/reid/vehicle_transreid_vit_base_veri776.pth` | Documented VeRi-776 TransReID 09v checkpoint | Consumed by 14t eval per `models/reid/README.md` |
| `models/reid/clip_senet_v6_best.pth` | Proposed serving alias for CLIP-SENet v6 | Size TBD from actual checkpoint |
| `models/reid/clipsenet_v6_veri776_best.pth` | Documented CLIP-SENet v6 checkpoint | Consumed by 14t eval per `models/reid/README.md` |
| `models/person_detection/MultiviewDetector.pth` | WILDTRACK MVDeTr detector | Not exposed as Phase 2 endpoint |
| `models/reid/transreid_cityflowv2_best.pth` | CityFlowV2 TransReID primary | Used by CityFlow MTMC pipeline |

Always prefer registry `checkpoint_refs.local_path` from `configs/model_registry.yaml` over hard-coded aliases.

Cold-start estimates:
- TransReID ViT-B/16 CPU load: roughly 5-20s; GPU load: roughly 2-8s.
- CLIP-SENet v6 CPU load: roughly 5-20s; GPU load: roughly 2-8s.
- CPU inference can be about 10x slower than GPU.

Memory budget:
- TransReID ViT-B/16: roughly 1.5-2GB GPU.
- CLIP-SENet: roughly 1.5GB GPU.
- LRU cache size 2 can peak around 4GB GPU with buffers.
- Use `REID_MODEL_CACHE_SIZE=1` for constrained GPUs.
- Use `USE_CPU=1` for CPU-only serving.

## 7. Risks & Mitigations

| Risk | Mitigation |
| --- | --- |
| OOM with multiple models | LRU cache, `REID_MODEL_CACHE_SIZE`, `USE_CPU`, explicit CUDA cleanup |
| Job concurrency | Single-worker serialization; scale out with Redis/RQ/Celery later |
| Missing checkpoint | Startup health warning and endpoint `checkpoint_missing` before inference |
| Path traversal | `Path.resolve().is_relative_to(UPLOAD_DIR.resolve())` guard |
| Large payloads | 25MB/image and 50-image caps, return 413 |
| Hydration mismatch | `'use client'` on upload/polling pages |
| API base mismatch | Normalize `/api` versus `/api/v1` in `frontend/src/services/reid.ts` |
| Fusion math drift | Shared helpers and parity tests against `scripts/eval/eval_14t_fusion_veri776.py` |
| Restart during job | Persist `data/jobs/{job_id}.json`, mark interrupted jobs failed |

Frame ID convention:
- Internal MTMC stages use 0-based frame ids and MOT output uses 1-based ids.
- ReID image ranking does not use frame ids, so this convention is not relevant here.
- Eval jobs invoking existing MTMC scripts must preserve existing frame behavior.

## 8. Phasing & Effort

### 8.1 Phase 2a (must): backend single-cam ReID endpoint
Files to create:
- `backend/routers/reid.py`
- `backend/models/reid.py`
- `backend/services/reid_service.py`
- `src/serving/reid_loaders.py`
- `tests/integration/test_reid_endpoints.py`
- `tests/unit/test_reid_service.py` if unit layout is accepted

Files to edit:
- `backend/app.py`
- `backend/models/requests.py`
- `scripts/eval/eval_09v_transreid_veri776.py`
- `scripts/eval/eval_clip_senet_veri776.py`

Registry/test entry points:
- `configs/model_registry.yaml`
- `scripts/eval/eval_09v_transreid_veri776.py`
- `scripts/eval/eval_clip_senet_veri776.py`
- `models/reid/README.md`

Acceptance:
- Curl `single_cam` with two query images and a 10-image gallery returns ranked indices.
- Similarities are finite and in `[-1,1]` when rerank is false.
- Unknown model returns 404, dead-end model returns 422, too-large payload returns 413, path outside `UPLOAD_DIR` returns 422.

### 8.2 Phase 2b (should): fusion endpoint and frontend playgrounds
Files to create:
- `frontend/src/app/reid/page.tsx`
- `frontend/src/app/fusion/page.tsx`
- `frontend/src/components/reid/ImageUploader.tsx`
- `frontend/src/components/reid/RankedResults.tsx`
- `frontend/src/components/reid/FusionWeightSliders.tsx`
- `frontend/src/services/reid.ts`

Files to edit:
- `backend/routers/reid.py`
- `backend/services/reid_service.py`
- `frontend/src/components/ModelPicker.tsx`
- `frontend/src/app/page.tsx`
- `tests/integration/test_reid_endpoints.py`

Registry/test entry points:
- `veri776_14t_fusion` in `configs/model_registry.yaml`
- `scripts/eval/eval_14t_fusion_veri776.py`
- `tests/test_stage2/test_14t_fusion_math.py`

Acceptance:
- Fusion accepts `veri776_09v_v17_transreid` + `veri776_clipsenet_v6`.
- Fusion rejects non-`single_cam_reid` models and invalid weights.
- `/reid` and `/fusion` render model selection, upload controls, and ranked results.
- 14t fusion via API matches CLI mAP within `1e-4` on the same split/settings.

### 8.3 Phase 2c (nice): eval jobs, dataset switcher, integration suite
Files to create:
- `backend/services/job_service.py`
- `frontend/src/app/eval/page.tsx`
- `frontend/src/components/reid/EvalProgressPanel.tsx`
- `frontend/src/components/layout/DatasetSwitcher.tsx`
- `tests/integration/test_eval_jobs.py`
- `tests/e2e/test_phase2_smoke.py`

Files to edit:
- `backend/app.py`
- `backend/routers/reid.py`
- `backend/services/pipeline_service.py`
- `backend/models/requests.py`
- `backend/models/reid.py`
- `frontend/src/store/index.ts`
- `frontend/src/app/layout.tsx` or existing layout component
- `frontend/src/app/page.tsx`
- `frontend/src/services/reid.ts`

Acceptance:
- Eval submission returns queued status, polling returns queued/running/succeeded/failed, final result endpoint returns JSON, failed jobs surface sanitized errors.
- Dataset switcher changes MTMC model filtering and pipeline requests.

## 9. Acceptance Criteria

Phase 2a curl sketch:
```powershell
$body = Get-Content .\tmp_reid_request.json -Raw
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/v1/reid/single_cam -ContentType 'application/json' -Body $body
```

Request shape:
```json
{
  "modelId": "veri776_09v_v17_transreid",
  "queries": [{"id": "q0", "image_base64": "BASE64_PNG"}],
  "gallery": [{"id": "g0", "image_base64": "BASE64_PNG"}],
  "topK": 10,
  "rerank": false,
  "aqeK": 0
}
```

Manual verification:
- Load TransReID, embed 5 VeRi-776 query images, and include same-id gallery matches.
- Top-1 should be same-id at cosine above `0.6` for clean crop pairs.
- Repeated warm-cache calls should return stable ordering.

Phase 2b curl sketch:
```powershell
$body = Get-Content .\tmp_fusion_request.json -Raw
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/v1/reid/fusion -ContentType 'application/json' -Body $body
```

Request shape:
```json
{
  "modelIds": ["veri776_09v_v17_transreid", "veri776_clipsenet_v6"],
  "weights": [0.3, 0.7],
  "queries": [{"id": "q0", "image_base64": "BASE64_PNG"}],
  "gallery": [{"id": "g0", "image_base64": "BASE64_PNG"}],
  "topK": 10,
  "rerank": true,
  "aqeK": 3
}
```

Manual verification:
- Run fusion endpoint with canonical TransReID + CLIP-SENet pair.
- Run `scripts/eval/eval_14t_fusion_veri776.py` with same split/settings.
- API fusion mAP should match CLI within `1e-4`.

Phase 2c curl sketches:
```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/v1/eval/run -ContentType 'application/json' -Body '{"modelId":"veri776_14t_fusion","dataset":"veri776","taskType":"score_fusion","useCpu":true}'
Invoke-RestMethod http://127.0.0.1:8000/api/v1/eval/eval_20260516_abc123
Invoke-RestMethod http://127.0.0.1:8000/api/v1/eval/eval_20260516_abc123/result
```

Expected submit response:
```json
{"jobId": "eval_20260516_abc123", "status": "queued"}
```

Manual verification:
- Submit limited VeRi-776 eval job, confirm queued -> running -> succeeded, and confirm `data/jobs/{job_id}.json` exists.
- Restart backend during a running job and confirm the interrupted job is marked failed.

## 10. Open Questions / Decisions Needed

### WebSocket for progress or HTTP polling only?
Recommendation: HTTP polling for Phase 2c; defer WebSocket.

### Should fusion accept arbitrary models or restrict to `task_type == "single_cam_reid"`?
Recommendation: restrict to `single_cam_reid`.

### Where should query/gallery uploads live?
Recommendation: `data/uploads/reid/{request_id}/`, garbage collected after 24 hours.

### Should MVDeTr detector-only be exposed as an endpoint?
Recommendation: skip for Phase 2; it is not a user-facing primitive without tracker/eval context.

### Should registry ids or checkpoint aliases be renamed?
Recommendation: no. Use registry ids and display friendly registry `name` fields in the UI.

## 11. Out of Scope

- Training/fine-tuning endpoints.
- GNN edge classifier.
- Kaggle orchestration.
- GPU multi-tenancy/model sharding.
- Auth/multi-user.
- Production object storage.
- WebSocket progress streaming.
- MVDeTr detector-only API.
- Frame ID convention changes.
- MOT submission formatting changes.
- Registry metric/provenance changes.
