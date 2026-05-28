# Spec: Per-stage Kaggle Offload (optional)

Status: Spec only — implementation will continue on branch `feature/pipeline-model-integration` as Phases 8+ after the fusion-integration spec lands (Phases P1–P7).

Goal: Let the user opt-in to running individual pipeline stages (especially the GPU-heavy 0/1/2 and the CPU-bound but parallelizable 4) on Kaggle instead of locally, while keeping local execution as the default for every stage. Backend manages the full Kaggle lifecycle: dataset prep, kernel push, polling, output download, and integration into the local run-artifacts directory under `e:\dev\src\gp\data\outputs\<run_id>\stage<N>\`.

## Section A — Architecture

### A.1 Per-stage offload data flow

```text
Frontend stage component (e.g. DetectionStage)
  └─ user toggles "Run on Kaggle" for stage 1
      └─ POST /api/pipeline/run-stage/1 with body.kaggle = { target: "kaggle", username?, key?, datasetSlug? }
          └─ backend/routers/pipeline.py:run_stage dispatches to kaggle_run_service.run_stage_on_kaggle(...)
              ├─ resolve_kaggle_credentials(request) → (username, key) or 401
              ├─ ensure_input_dataset(run_id, video_id|dataset_name, datasetSlug) → kaggle_dataset_slug
              ├─ render kernel notebook + kernel-metadata.json from stage template
              ├─ kaggle_service.push_kernel(...) → kernel_slug
              ├─ persist data/outputs/<run_id>/kaggle_job.json
              ├─ register slug into state.active_kaggle_kernels
              └─ return { kaggle: { kernel_slug, kernel_url, dataset_slug, status: "queued" } }

Background polling worker (asyncio task in backend.app)
  ├─ every 60s for each active kernel:
  │    ├─ kaggle_service.kernel_status(slug)
  │    ├─ update kaggle_job.json
  │    ├─ emit websocket event on the existing pipeline run channel
  │    └─ on terminal status:
  │         ├─ kaggle_service.kernel_output(slug, dest=data/outputs/<run_id>/_kaggle_dl/)
  │         ├─ finalize: copy artifacts into data/outputs/<run_id>/stage<N>/ via existing
  │         │    backend.services.pipeline_service._materialize_import_tree(...)
  │         ├─ remove slug from state.active_kaggle_kernels
  │         └─ mark stage progress complete in app_state.active_runs[run_id]
```

### A.2 State location

- Frontend execution-target state lives in `e:\dev\src\gp\frontend\src\store\index.ts`, extending `usePipelineStore` with:
  - `stageExecutionTargets: Record<number, 'local' | 'kaggle'>`
  - `setStageExecutionTarget(stage: StageNumber, target: 'local' | 'kaggle'): void`
  - Persisted via the existing `persist()` middleware (workflow preference, NOT pipeline output).

- Frontend Kaggle credentials live in a SEPARATE store `useKaggleCredentialsStore` (also persisted to localStorage). Keeping it separate from `usePipelineStore` avoids coupling auth state to pipeline reset/flush flows.

- Backend authoritative state for in-flight Kaggle jobs:
  - `e:\dev\src\gp\backend\state.py`: extend `AppState` with `active_kaggle_kernels: Dict[str, KaggleJobRecord]` keyed by kernel slug.
  - On-disk truth: `e:\dev\src\gp\data\outputs\<run_id>\kaggle_job.json` (per stage; multiple entries allowed if user offloads multiple stages of the same run).

### A.3 Credential flow

User-supplied credentials (preferred when present):
1. Frontend collects `username` + `key` via the credentials modal (Section C.2).
2. On every stage-run request that targets Kaggle, frontend includes them in the request body under `kaggle.username` and `kaggle.key`.
3. Backend never persists these to disk. They are written into a `tempfile.TemporaryDirectory()` as `kaggle.json` only for the lifetime of the subprocess that calls `kaggle` CLI / SDK; `KAGGLE_CONFIG_DIR` env var points at that tempdir.
4. The temp dir is `shutil.rmtree()`-ed in a `try/finally` immediately after the subprocess returns, even on error.

Server fallback:
1. If `kaggle.username` and `kaggle.key` are absent in the request body, backend uses the server's `~/.kaggle/kaggle.json` (currently the `gumfreddy` token).
2. Resolution order is strict: request body > server-side `~/.kaggle/kaggle.json` > 401 error.

`kaggle.json` schema (documented, not read by backend):
```json
{ "username": "<kaggle-username>", "key": "<kaggle-api-key>" }
```

## Section B — Backend implementation

### B.1 Kaggle CLI / SDK wrapper

New file: `e:\dev\src\gp\backend\services\kaggle_service.py`

Reuses patterns already proven in `e:\dev\src\gp\scripts\kaggle_logs.py` (which uses the official `kaggle.api.kaggle_api_extended.KaggleApi` SDK directly, not subprocess `kaggle` CLI) for status polling. For push/output/dataset operations, prefer SDK calls where supported; fall back to subprocess `kaggle` CLI (via `subprocess.run([...])`) only when an SDK equivalent is not exposed.

Public functions (all accept an optional `credentials: Optional[KaggleCredentials]` arg; when provided, the function injects them as a tempdir + `KAGGLE_CONFIG_DIR` env override for the duration of the call):

| Function | Returns | Notes |
|---|---|---|
| `push_kernel(metadata: dict, source_files: dict[str, bytes \| str], *, credentials)` | `KernelPushResult(slug, url, version)` | Stages files into a tempdir matching Kaggle's expected `kernel-metadata.json` + code file layout, runs `kaggle kernels push -p <tempdir>`, parses warnings (e.g. "not valid dataset sources") and raises `KaggleKernelValidationError` with the parsed details. |
| `kernel_status(slug, *, credentials)` | `KernelStatus(status: Literal['queued','running','complete','error','cancelled','unknown'], failure_message: Optional[str])` | Wrapper around the SDK call already used by `scripts/kaggle_logs.py:get_status`. Normalize the raw `KernelWorkerStatus.*` strings into the lowercase enum above. |
| `kernel_output(slug, dest_dir: Path, *, credentials)` | `list[Path]` | Wraps `kaggle kernels output <slug> -p <dest_dir>` (subprocess; SDK lacks a clean equivalent). Returns the list of downloaded paths. |
| `dataset_create_or_update(slug, files: list[Path], description: str, *, credentials)` | `DatasetUpsertResult(slug, version)` | Idempotent: tries `kaggle datasets version` first; if the dataset doesn't exist, falls back to `kaggle datasets create`. Auto-bumps the version note. |
| `cancel_kernel(slug, *, credentials)` | `CancelResult(method: 'cli' \| 'poll-only', final_status)` | Try `kaggle kernels cancel` if the installed CLI version supports it. If `subprocess.CalledProcessError` says "no such command" or returns the help banner, fall back to polling `kernel_status` every 60s until the run reaches a terminal state, per the Kaggle Push Safety Rules in `e:\dev\src\gp\.github\copilot-instructions.md`. |

Error mapping (raised inside `kaggle_run_service`, caught by router and translated to HTTP):

| Internal exception | HTTP status | Body shape |
|---|---|---|
| `KaggleConcurrencyError` (2 active GPU sessions detected) | 429 | `{ "code": "kaggle_busy", "active_kernels": [...] }` |
| `KaggleAuthError` (missing creds OR invalid token) | 401 | `{ "code": "kaggle_auth", "message": "...", "configure_url": "/settings/kaggle" }` |
| `KaggleKernelValidationError` (e.g. "not valid dataset sources") | 400 | `{ "code": "kaggle_kernel_invalid", "validation": {...} }` |
| `KaggleQuotaError` (storage / dataset size limits) | 413 | `{ "code": "kaggle_quota", "limit": ... }` |

Concurrency-limit detection happens BEFORE push: call `kaggle_service.list_my_running_kernels(credentials)` (SDK), count GPU-enabled ones; if `>= 2` → `KaggleConcurrencyError`. Post-push, also re-check active set and update `state.active_kaggle_kernels`.

> **OPEN QUESTION:** The Kaggle SDK does not currently expose a clean `list_my_running_kernels` filtered by GPU. We may need to maintain our own active-set view in `state.active_kaggle_kernels` (seeded from on-disk `kaggle_job.json` files at startup) and only consult the API for verification. User decision: trust local state with API verification on push, or always query API first?

### B.2 Kernel template generation

New file: `e:\dev\src\gp\backend\services\kaggle_kernel_templates.py`

Per-stage builders (no Jinja — straightforward Python f-strings + JSON dicts; the existing `_build_*_notebook.py` scripts at the repo root demonstrate the pattern):

| Stage | Builder | GPU? | Inputs (dataset_sources) | Output artifact mounted by next stage |
|---|---|---|---|---|
| 0 | `build_stage0_kernel(run_id, dataset_slug, config_path, model_overrides) → (notebook_json, metadata_json)` | T4/P100 (configurable; default T4 to match `notebooks/kaggle/10a_stages012/kernel-metadata.json`) | user video dataset OR pre-existing `<owner>/cityflow-v2` | `data/outputs/<run_id>/stage0/` zip |
| 1 | `build_stage1_kernel(...)` | T4/P100 | stage0 output kernel ref OR pre-existing dataset | `data/outputs/<run_id>/stage1/` |
| 2 | `build_stage2_kernel(..., model_id)` | T4/P100 | stage1 output kernel ref + ReID checkpoint dataset (`gumfreddy/mtmc-weights` or fusion-aware multi-checkpoint set) | `data/outputs/<run_id>/stage2/` |
| 4 | `build_stage4_kernel(..., association_overrides)` | CPU only | stage2 output kernel ref OR a user-uploaded artifact dataset (when stage2 ran locally) | `data/outputs/<run_id>/stage4/` |

Each builder produces:
- A notebook JSON whose first code cell `pip install`s pinned project deps from a small inline list (PyTorch 2.4.1+cu124 for P100, see copilot-instructions), then `git clone`s the project at the current commit SHA (or pulls from a private mirror — see open question), then `cd`s in and runs:
  ```text
  python scripts/run_pipeline.py --config <config_path> --stages <stage> --override project.run_name=<run_id> [other overrides]
  ```
- A `kernel-metadata.json` matching the convention seen in `e:\dev\src\gp\notebooks\kaggle\10a_stages012\kernel-metadata.json`:
  - `"id": "<kaggle_username>/mtmc-run-<run_id>-stage-<N>"` (deterministic, derived from the resolved kaggle username + run_id + stage)
  - `"is_private": true` (always)
  - `"enable_gpu": <bool>` per stage above
  - `"enable_internet": true`
  - `"dataset_sources"`, `"kernel_sources"` per the table above
- A small `params.json` artifact written into the kernel's working directory so the kernel reads its parameters deterministically and the same notebook code path is used regardless of stage.

> **OPEN QUESTION:** How does the kernel get the current project source? Three options:
> (a) `git clone https://github.com/<...>/gp.git && git checkout <sha>` — requires the repo to be public OR a deploy key in the kernel (Kaggle "secrets" feature).
> (b) Bundle the entire `src/`, `scripts/`, `configs/` tree into a Kaggle dataset that is upserted on every push (slow, eats dataset storage).
> (c) Use `kernel_sources` chained from a "base" kernel that always contains the latest `src/`. Requires a separate auto-pushed base kernel.
> Pick **before Phase 9**.

### B.3 Job model

Extend `e:\dev\src\gp\backend\models\requests.py`:

```python
class StageExecutionTarget(str, Enum):
    local = "local"
    kaggle = "kaggle"

class KaggleConfig(BaseModel):
    target: StageExecutionTarget = StageExecutionTarget.local
    username: Optional[str] = None
    key: Optional[str] = None
    datasetSlug: Optional[str] = None  # pre-existing Kaggle dataset, e.g. "thanhnguyenle/data-aicity-2023-track-2"
    enableGpu: Optional[bool] = None   # override per stage default

class PipelineRunRequest(BaseModel):
    # existing fields preserved verbatim from current requests.py
    runId: Optional[str] = None
    videoId: Optional[str] = None
    cameraId: Optional[str] = None
    dataset: Optional[str] = None
    model_id: Optional[str] = Field(default=None, alias="modelId")
    smokeTest: bool = False
    useCpu: bool = False
    config: Optional[Dict[str, Any]] = None
    # NEW:
    kaggle: Optional[KaggleConfig] = None
```

When `kaggle` is absent or `kaggle.target == "local"`, behaviour is identical to today.

Persisted job state at `e:\dev\src\gp\data\outputs\<run_id>\kaggle_job.json`:
```json
{
  "runId": "<run_id>",
  "stage": 1,
  "kernelSlug": "gumfreddy/mtmc-run-42-stage-1",
  "kernelUrl": "https://www.kaggle.com/code/gumfreddy/mtmc-run-42-stage-1",
  "datasetSlug": "gumfreddy/mtmc-user-video-42",
  "status": "running",
  "createdAt": "2026-05-20T15:01:00Z",
  "lastPolledAt": "2026-05-20T15:02:00Z",
  "downloadDir": "data/outputs/42/_kaggle_dl/stage-1",
  "credentialsSource": "request | server"
}
```

### B.4 Polling worker

New module-level asyncio task started from `e:\dev\src\gp\backend\app.py` startup hook (alongside `job_service.start_worker()`):

- Loop: every 60s, iterate `state.active_kaggle_kernels.values()`, call `kaggle_service.kernel_status(slug, credentials=record.credentials_proxy)`.
- On terminal status: schedule `kaggle_service.kernel_output(...)` then `pipeline_service._materialize_import_tree(...)` to fold the downloaded `data/outputs/<run_id>/_kaggle_dl/stage-<N>/` zip/tree into the proper `data/outputs/<run_id>/stage<N>/` layout. Reuse the existing import logic from `e:\dev\src\gp\backend\routers\runs.py:import_kaggle_run_artifacts` rather than duplicating it.
- Emit websocket events on the existing pipeline run channel (the WebSocket handler near `run_stage` in `e:\dev\src\gp\backend\routers\pipeline.py` is the integration point — discover the existing event shape and add a `kaggle_status` variant).
- Network failure handling: exponential backoff per slug starting at 60s, doubling to a max of 600s. After 5 consecutive failures, surface `kaggle.status = "polling_degraded"` to the UI but keep retrying.
- Credentials in the worker: cache the credentials USED at push time on the in-memory `KaggleJobRecord`; never re-read them from disk. If the backend restarts mid-poll, the worker re-loads `kaggle_job.json` and falls back to `~/.kaggle/kaggle.json` for that record (since the original user-supplied creds were never persisted) — surfaces a `credentials_lost` warning to the UI when this happens.

### B.5 Stage handler dispatch

Modify `e:\dev\src\gp\backend\routers\pipeline.py:run_stage`:

```python
# After payload = request or PipelineRunRequest()
if payload.kaggle and payload.kaggle.target == StageExecutionTarget.kaggle:
    return await kaggle_run_service.run_stage_on_kaggle(
        stage=stage, payload=payload, state=state, run_id=run_id, resolution=resolution,
    )
# else: existing local path unchanged
```

`kaggle_run_service.run_stage_on_kaggle` returns:
```json
{
  "runId": "<run_id>",
  "stage": <N>,
  "status": "queued",
  "kaggle": {
    "kernelSlug": "...",
    "kernelUrl": "...",
    "datasetSlug": "...",
    "status": "queued",
    "credentialsSource": "request"
  }
}
```

The local response shape is preserved as a strict subset so the frontend can handle either result with one type guard.

### B.6 Dataset auto-upload

For user-uploaded videos (when `kaggle.datasetSlug` is absent and `payload.videoId` references an upload):
- Slug pattern: `<resolved_kaggle_username>/mtmc-user-video-<run_id>` (deterministic, scoped per run).
- First time: `dataset_create_or_update` with `description="Auto-uploaded by MTMC backend for run <run_id>"`.
- Subsequent stages of the same run: same slug, no re-upload (the previous upload is still valid).
- For full-dataset runs (e.g. CityFlowV2): use the registry mapping below (open question on exact slugs).

Hardcoded registry constant (in `kaggle_kernel_templates.py`):
```python
KAGGLE_DATASET_REGISTRY: dict[str, str] = {
    "cityflowv2": "thanhnguyenle/data-aicity-2023-track-2",
    "wildtrack":   "<TBD>",
    # extended via configs/datasets/<name>.yaml `kaggle.dataset_slug` if present
}
```

> **OPEN QUESTION:** Confirm `thanhnguyenle/data-aicity-2023-track-2` is the slug we want as the canonical CityFlowV2 source (it is what `notebooks/kaggle/10a_stages012/kernel-metadata.json` uses). For WILDTRACK, no existing kernel uses it — we likely need to upload it as a new private dataset under the server account. Pre-Phase 8 user decision required.

## Section C — Frontend implementation

### C.1 Per-stage execution-target store

Edit `e:\dev\src\gp\frontend\src\store\index.ts`, extending `PipelineState`:

```ts
stageExecutionTargets: Record<number, 'local' | 'kaggle'>;
setStageExecutionTarget: (stage: StageNumber, target: 'local' | 'kaggle') => void;
```

Defaults (set in the store initializer, motivated by where Kaggle actually pays off):

| Stage | Default | Toggle visible? |
|---|---|---|
| 0 (Ingestion) | local | yes |
| 1 (Detection & Tracking) | local | **yes** (highest-value Kaggle stage — GPU-bound) |
| 2 (Feature Extraction) | local | **yes** (also GPU-bound) |
| 3 (Indexing) | local | yes (CPU; useful when stage 2 ran on Kaggle so we don't have to download embeddings) |
| 4 (Association) | local | yes |
| 5 (Evaluation) | local | no — local is always fast enough |
| 6 (Visualization) | local | no — local is the only sensible target (renders to user's browser) |

Persist in localStorage via `persist()` (already used by `usePipelineStore`). Reset on `usePipelineStore.reset()` is **not** desired — execution-target is a workflow preference, not run state. Keep it across resets; clear only via an explicit user "reset preferences" action.

### C.2 Kaggle credentials UI

New file: `e:\dev\src\gp\frontend\src\components\settings\kaggle-credentials-modal.tsx` (or wherever the existing settings modals live; check `e:\dev\src\gp\frontend\src\components\settings\` first — create that folder if it doesn't exist yet).

New store file: `e:\dev\src\gp\frontend\src\store\kaggle-credentials.ts` (or extend `index.ts` with a second store if the project convention is co-located stores — verify by inspecting whether the dataset switcher store is in `index.ts` or a sibling file):

```ts
interface KaggleCredentialsState {
  username: string | null;
  key: string | null;
  hasCredentials: () => boolean;
  setCredentials: (username: string, key: string) => void;
  clearCredentials: () => void;
}
```

Persisted via `persist()`. The persisted entry is in localStorage only — never sent to the backend except in stage-run request bodies.

UI: a small gear / cloud-config icon in the sidebar header opens a modal with two fields (username, password-style key input) and a copy-button "How to get an API key" help link to `https://www.kaggle.com/settings`. Includes a clear warning:

> "Your Kaggle username and API key are stored only in this browser's local storage. They are submitted to the MTMC backend per-request to authenticate Kaggle operations on your behalf, and are never written to backend disk. Clear them at any time with 'Sign out of Kaggle'."

### C.3 Per-stage toggle

Each visible stage component listed in Section C.1 (`detection-stage.tsx`, `inference-stage.tsx`, etc. under `e:\dev\src\gp\frontend\src\components\stages\`) gets a small "Run on Kaggle" `Switch` (from the existing UI primitives) placed adjacent to the run button.

- When toggled on AND `useKaggleCredentialsStore.hasCredentials()` is `false`: show an inline warning chip "No Kaggle credentials configured" with a button that opens the credentials modal. The run button remains enabled — the backend will fall back to its own credentials if the user proceeds, so the chip is informational.
- When toggled on AND credentials exist: chip reads "Using your Kaggle account: `<username>`".
- When toggled off (default): no chip, run button behaves as today.

### C.4 Kaggle status panel

New component: `e:\dev\src\gp\frontend\src\components\stages\kaggle-status-panel.tsx`

Shown in place of the local progress bar whenever the active stage was launched on Kaggle (check via the `kaggle` block in the run-stage response, persisted into `usePipelineStore.stages[stage].kaggle`).

Displays:
- Kernel slug (monospace) + external link icon → opens `kernelUrl` in a new tab.
- Live status badge: queued / running / complete / failed / cancelled (color-coded).
- Time elapsed since `createdAt`.
- Optional "Cancel run" button → calls a new `POST /api/pipeline/kaggle/cancel/{runId}/{stage}` endpoint (Section D).
- On terminal status, the panel auto-collapses and the normal local results view takes over (since the polling worker has already integrated the outputs into `data/outputs/<run_id>/stage<N>/`).

### C.5 Sidebar enhancement

Edit `e:\dev\src\gp\frontend\src\components\layout\main-dashboard.tsx`. Beside each stage label in the sidebar, render a 12px `Server` (lucide-react) icon when `stageExecutionTargets[stage] === 'local'` and a `Cloud` icon when it equals `'kaggle'`. Tooltip: "Local execution" / "Kaggle execution".

## Section D — Concurrency / failure handling

### D.1 Two-slot Kaggle limit

- Source of truth: `state.active_kaggle_kernels: Dict[str, KaggleJobRecord]`.
- Pre-push check in `kaggle_run_service.run_stage_on_kaggle`:
  ```text
  active = [r for r in state.active_kaggle_kernels.values() if r.gpu_enabled and r.status not in TERMINAL]
  if len(active) >= 2 and request_is_gpu_enabled:
      raise KaggleConcurrencyError(active=[r.slug for r in active])
  ```
- The check runs against credentials-resolved running kernels, NOT a per-user count, because Kaggle enforces the limit per account and our server may share the `gumfreddy` account across users.
- Frontend: 429 response → toast "Both your Kaggle GPU slots are busy. Cancel an existing run to start a new one." with a "View active runs" action.

### D.2 Cancellation

New endpoint: `POST /api/pipeline/kaggle/cancel/{run_id}/{stage}`:
1. Resolve `kaggle_job.json`, fetch slug.
2. `kaggle_service.kernel_status(slug, ...)`. If terminal already, return immediately.
3. Try `kaggle_service.cancel_kernel(slug, ...)`.
4. If CLI lacks `cancel`: enter the documented poll loop (60s) until terminal, surfacing each transition via websocket.
5. On confirmation: remove from `state.active_kaggle_kernels`, update `kaggle_job.json` to `status: "cancelled"`, emit final websocket event.

### D.3 Network failures during polling

Per Section B.4: exponential backoff 60s → 600s, max 5 consecutive failures before surfacing degraded status, infinite retries afterward (Kaggle outages are real but transient).

### D.4 Authentication failures

- 401 from any Kaggle call → surface to UI with a `code: "kaggle_auth"` body and a `configure_url` hint. Frontend opens the credentials modal with a "Re-enter credentials" header.
- If the failure occurred during the polling worker (not the original push), the active job is marked `status: "auth_failed"` and the worker stops polling that slug; user can re-enter credentials and call a new `POST /api/pipeline/kaggle/resume/{run_id}/{stage}` endpoint to retry polling with fresh creds.

## Section E — Phased commit plan

Branch: `feature/pipeline-model-integration` (continuation after Phases P1–P7 of the fusion spec). Each phase is a separate revertible commit (or small commit set).

| Phase | Title | Files touched | Validation gate | User-visible |
|---|---|---|---|---|
| 8 | `kaggle_service.py` CLI/SDK wrapper | new `backend/services/kaggle_service.py`, new `tests/backend/test_kaggle_service.py` | unit tests with mocked `subprocess.run` and mocked `KaggleApi`; assert credential injection via tempdir + `KAGGLE_CONFIG_DIR` | none |
| 9 | Kernel templates | new `backend/services/kaggle_kernel_templates.py`, new fixture notebooks under `tests/fixtures/kaggle_templates/` | golden-file tests: render notebook + metadata for each stage and assert exact JSON bytes | none |
| 10 | Request shape + dispatcher (no UI yet) | edit `backend/models/requests.py`, edit `backend/routers/pipeline.py:run_stage`, new `backend/services/kaggle_run_service.py` | manual curl test with a `kaggle` body still hitting the local fallback path; smoke that targeting Kaggle returns a `kernel_slug` (using a pre-canned mocked kernel for CI) | none |
| 11 | Polling worker + websocket events | edit `backend/app.py`, edit `backend/state.py`, edit websocket handler in `backend/routers/pipeline.py` | integration test that spins up a fake `kaggle_service` returning `running`→`complete` and asserts the stage finalize step writes into `data/outputs/<run_id>/stage<N>/` | none |
| 12 | Frontend credentials store + modal | new `frontend/src/store/kaggle-credentials.ts`, new `frontend/src/components/settings/kaggle-credentials-modal.tsx`, edit sidebar header in `main-dashboard.tsx` | manual: open modal, save creds, reload page, assert persistence; clear creds, assert cleared | yes (settings only) |
| 13 | Per-stage toggle + execution-target store | edit `frontend/src/store/index.ts` (`stageExecutionTargets`), edit each visible stage component in `frontend/src/components/stages/` | manual: toggle each stage independently, refresh, assert toggles persist | yes |
| 14 | Kaggle status panel | new `frontend/src/components/stages/kaggle-status-panel.tsx`, hook into stage components | manual: kick off a real Stage 1 Kaggle run with the mocked backend, observe panel updates from queued → running → complete; cancel button works | yes |
| 15 | Sidebar icons + e2e validation | edit `main-dashboard.tsx`, add `tests/e2e/kaggle_offload.spec.ts` (or whatever e2e harness exists — verify before committing) | full e2e: upload video → toggle Stage 1 to Kaggle → run → observe panel → cancel mid-run → re-run to completion → confirm artifacts in `data/outputs/<run_id>/stage1/` | yes |

## Section F — Risks & open questions

1. **Kaggle CLI version drift** — `cancel` subcommand absent in 2.0.1. Workaround documented in copilot-instructions and codified in `kaggle_service.cancel_kernel`. ACCEPT.
2. **Dataset upload size limits** — Kaggle's per-file limit is ~5 GB; the per-private-dataset limit is currently 20 GB. CityFlowV2 plus its preprocessed crops fit, but a future user-uploaded multi-camera dataset may not. Mitigation: pre-flight size check in `dataset_create_or_update`; raise `KaggleQuotaError` if total > 19 GB. ACCEPT.
3. **Polling rate limits** — Kaggle's API allows ~100 req/min unauthenticated; authenticated is far higher. With max 2 concurrent kernels polled at 60s each, we are well under any plausible limit. ACCEPT.
4. **User navigates away mid-job** — Backend continues polling and finalizing outputs because `state.active_kaggle_kernels` and `kaggle_job.json` are server-side. When the client reconnects, it `GET /api/runs/<run_id>` and reads the persisted Kaggle status from `kaggle_job.json` to repopulate the status panel. ACCEPT.
5. **Dataset slug collisions across users sharing the server account** — Slug pattern `<server_username>/mtmc-user-video-<run_id>` is unique per `run_id`, and `run_id` is allocated server-side via `_allocate_numeric_run_id`. No collisions are possible within one server. **OPEN QUESTION:** if the same backend is later replicated (multi-instance deploy), `run_id` is no longer globally unique → recommend prefixing with a server-instance UUID before that ever happens. Defer.
6. **Output caching for re-runs** — Repeating a Stage 4 sweep on the same Stage 2 outputs shouldn't require re-downloading from Kaggle every time. Defer to Phase 15+: simple content-hash key into `data/outputs/_kaggle_cache/<sha>/`. Not in MVP.
7. **Pre-existing Kaggle dataset slug registry** — Section B.6 hardcodes a small map. **OPEN QUESTION:** for WILDTRACK we have no canonical Kaggle dataset; user must decide between (a) upload-once-then-cache or (b) require user to specify `kaggle.datasetSlug` for WILDTRACK. Pre-Phase 9 user decision required.
8. **Project source delivery into the kernel** — see open question in Section B.2. Pre-Phase 9 user decision required.
9. **Two-slot limit when the server account is also being used by humans pushing kernels manually** — our `state.active_kaggle_kernels` won't see those. Mitigation: also call `list_my_running_kernels` SDK API at push time (Section B.1 open question). ACCEPT with the open question's resolution.
10. **Auth-lost-on-restart** — Section B.4 documents the fallback. ACCEPT (rare; manageable via the `resume` endpoint).

## Section G — Out of scope (explicit)

- Multi-user authentication / authorization on the backend (no per-user identity exists today).
- Kaggle Notebooks scheduled runs.
- Kaggle paid Inference endpoints.
- Automatic model checkpoint sync between local and Kaggle. Kaggle datasets like `gumfreddy/mtmc-weights` are assumed to already contain the active checkpoints; the registry in `e:\dev\src\gp\backend\services\model_registry.py` already records `notebook_or_kernel_ref` for traceability.
- Cross-account Kaggle aggregation (e.g. running on `mrkdagods` while user is logged in as `gumfreddy`). One credential set per request.
- Caching of intermediate Kaggle outputs across runs (deferred per Risk #6).

## Blocking decisions before Phase 8 starts

1. **Project source delivery into the kernel** (Section B.2 open question): pick (a) clone-from-git, (b) bundle-as-dataset, or (c) base-kernel chain.
2. **Kaggle dataset slug for WILDTRACK** (Section B.6 / Risk #7).
3. **Concurrency source-of-truth** (Section B.1 open question / Risk #9): trust local `state.active_kaggle_kernels` with API verification, or always query the API first?
