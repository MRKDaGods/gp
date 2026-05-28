
# Pipeline Model + Fusion Integration Spec (v2 — full live-pipeline fusion)

**Status:** Spec only — implementation in branch `feature/pipeline-model-integration`
**Author context:** MTMC Planner. Consumed by MTMC Coder.
**Date:** 2026-05-20 (v2 revision: real backend fusion, not stub)
**Scope (user-confirmed):**
1. Add Single↔Fusion mode toggle to Stage 3 (`InferenceStage`).
2. Promote selected model + fusion config to global Zustand state (persist across stage navigation).
3. Render selected-model badge in the sidebar (expanded + collapsed states).
4. **Build a real end-to-end live-pipeline fusion path** (multi-model Stage 2 extraction + Stage 4 score-level fusion). NOT a warning stub.
5. Pre-seed Fusion mode with the canonical 14t pair (TransReID VeRi-776 + CLIP-SENet VeRi-776) on first open (CityFlowV2 dataset). Other datasets start empty.
6. Fusion picker shows **all task types** — no filter. The user may mix MTMC and single-cam ReID checkpoints; the spec calls out the cross-domain risk explicitly.

**NOT in scope:** deleting `/reid` or `/fusion` standalone pages; locking dataset to `model.dataset`; modifying `ModelPicker` layout; adding test-bench shortcut links.

---

## Section A — Architecture decisions

### A1. State location
Extend `usePipelineStore` (the live-pipeline store at [frontend/src/store/index.ts](frontend/src/store/index.ts)). It already owns `runId`, `stages`, `isRunning`, `currentStage`, so model selection that drives a pipeline run belongs here. Do NOT use `useDatasetStore` (that is dataset-only) and do NOT create a new store.

New fields on `usePipelineStore`:

```ts
type PipelineModelMode = "single" | "fusion";

interface FusionModelEntry {
  modelId: string;
  weight: number; // 0..1, normalized so sum==1 across selectedModelIds
}

interface PipelineModelSelection {
  mode: PipelineModelMode;        // default "single"
  selectedModelId: string | null; // single mode
  fusion: {
    selectedModelIds: string[];   // length >= 2 to be valid; max 4 (see B'.2)
    weights: Record<string, number>; // raw, un-normalized; normalize on read
    aqeK: number;                 // default 2 (14e B1 canonical for live MTMC; 14t VeRi uses 3 — see B'.4)
    rerank: boolean;              // default false (live MTMC pipeline reranking is off; see B'.3)
    seeded: boolean;              // true once we've auto-applied the dataset preset
  };
  selectedModelMeta: {
    name: string;
    dataset: string;
    headlineMetric: { name: string; value: number } | null;
  } | null;
}
```

### A2. Persistence policy
- Do NOT wrap `usePipelineStore` in `persist()` — it already holds transient run state (`runId`, `isRunning`) that must NOT survive a hard reload.
- Cross-stage-navigation persistence is automatic because the store is a singleton in the React tree; the user never leaves the page, only switches stage panels (see `MainDashboard` keeping panels mounted via `visitedPipelineStages`).
- Hard-reload reset is acceptable. The `seeded` flag also resets on reload, so the pre-seed runs again on the next first open.

### A3. Toggle scope
The Single↔Fusion toggle controls **both UI and the backend call shape** in Stage 3. The `runStage(2,...)` and `runStage(3,...)` calls in `inference-stage.tsx` will branch:

- **Single mode** (existing): send `{ model_id }`. No regression risk.
- **Fusion mode** (new): send `{ model_id: <primary>, fusion: { models: [...], aqeK, rerank } }` as a **top-level** request field (NOT nested under `config`). See B'.4. Backend validates, materialises Stage 2 multi-model overrides + Stage 4 score-fusion overrides, and runs the real fused pipeline.

The "primary" model in fusion mode is the model with the largest normalized weight (deterministic tiebreak: first in `selectedModelIds`). This keeps `model_id` resolution + dataset inference compatible with the existing `resolve_pipeline_model()` flow.

### A4. Sidebar badge rendering
Add a compact badge to `MainDashboard` sidebar.

- **Expanded sidebar (sidebarOpen=true):** small card below the stage list, above any future settings. Two lines: model name (truncate) + headline metric "IDF1 0.7794" (or "FUSION · 2 models" if fusion mode). Border + muted bg.
- **Collapsed sidebar (sidebarOpen=false):** small Cpu icon with a Tooltip showing the same info; click routes to Stage 3.

Badge is read-only; editing happens only on Stage 3.

### A5. Pre-seed defaults
On Stage 3 mount, if `usePipelineStore.fusion.seeded === false` AND the active dataset (from `useDatasetStore`) is `cityflowv2`, populate:

```ts
fusion.selectedModelIds = ["veri776_09v_v17_transreid", "veri776_clipsenet_v6"]
fusion.weights = { veri776_09v_v17_transreid: 0.30, veri776_clipsenet_v6: 0.70 }  // 14t canonical w_clipsenet=0.7
fusion.aqeK = 3            // 14t canonical
fusion.rerank = true       // 14t canonical (NOTE: only effective when backend wires rerank into Stage 4 — see B'.3 caveat)
fusion.seeded = true
```

For `wildtrack` and any other dataset: leave `selectedModelIds` empty + show the "Pick at least 2 models" hint. Set `seeded = true` so we don't keep retrying.

**Registry IDs (verified in [configs/model_registry.yaml](configs/model_registry.yaml)):**
- TransReID VeRi-776 expert → `veri776_09v_v17_transreid` (line 382)
- CLIP-SENet VeRi-776 expert → `veri776_clipsenet_v6` (line 434)

**Cross-domain caveat:** these are VeRi-776 single-cam checkpoints. Running them as the live MTMC pipeline's appearance models on CityFlowV2 footage is **expected to underperform** the deployed `vehicle_mtmc_14e_b1` baseline (0.77936 IDF1) because of the VeRi-776 → CityFlowV2 domain gap (see [docs/findings.md](docs/findings.md), 13d/13f dead ends). The pre-seed is a **canonical-recipe demonstration**, not a recommended production config. The frontend MUST surface this with an info banner: *"This preset reproduces the 14t VeRi-776 single-camera fusion recipe. On CityFlowV2 MTMC, expect lower IDF1 than the production single-model baseline due to the cross-domain gap (see findings.md 13d/13f)."*

### A6. Picker `task_type` filter
Per user decision: **no filter**. The Fusion mode multi-select picker calls `fetchModels()` without a `task_type` constraint and lists every runnable model. Users may mix `mtmc_vehicle`, `single_cam_reid`, `mtmc_person` checkpoints. The pre-seed and the cross-domain banner provide the only guardrail.

---

## Section B — Backend reality check (current state)

### B1. What exists today
- `POST /api/v1/reid/fusion` ([backend/services/reid_service.py](backend/services/reid_service.py)) operates on **uploaded query+gallery base64 images only**. It is the engine behind the `/fusion` page. It does NOT run Stages 0–4 of the live MTMC pipeline.
- `POST /api/pipeline/run-stage/{stage}` ([backend/routers/pipeline.py](backend/routers/pipeline.py)) accepts only a single `model_id` via `PipelineRunRequest` and resolves it through `resolve_pipeline_model()` ([backend/services/pipeline_service.py](backend/services/pipeline_service.py#L91)) → one `pipeline_config` YAML + `model_overrides` list.
- The live offline Stage 2 / Stage 4 code path **already supports up to 4 ReID streams** today via these existing knobs (verified in [src/stage2_features/pipeline.py](src/stage2_features/pipeline.py) and [src/stage4_association/pipeline.py](src/stage4_association/pipeline.py)):

| Slot | Stage 2 config block | Output `.npy` | Stage 4 fusion key |
|------|----------------------|---------------|--------------------|
| Primary | `stage2.reid.vehicle` | `embeddings.npy` | implicit weight `1 - sum(others)` |
| Secondary | `stage2.reid.vehicle2` (`enabled=true`, `save_separate=true`) | `embeddings_secondary.npy` | `stage4.association.secondary_embeddings.{path,weight}` |
| Tertiary | `stage2.reid.vehicle3` (`enabled=true`, `save_separate=true`) | `embeddings_tertiary.npy` | `stage4.association.tertiary_embeddings.{path,weight}` |
| Quaternary | *no `vehicle4` extractor exists* | (would be `embeddings_quaternary.npy`) | `stage4.association.quaternary_embeddings.{path,weight}` (loader exists; producer does not) |

The 14e B1 production headline (0.77936) is in fact this exact code path: TransReID primary + DINOv2 tertiary score fusion (see [configs/datasets/cityflowv2.yaml#L160-L172](configs/datasets/cityflowv2.yaml#L160-L172)).

### B2. What's missing
- The router does NOT accept a `fusion` field — the existing knobs are reachable only via static YAML edits or hand-crafted `--override` strings.
- There is no `vehicle4` slot in Stage 2; the quaternary loader in Stage 4 is currently only fed by Kaggle-precomputed `.npy` artifacts (e.g. 14k R50-IBN). For now we cap fusion at **3 models** (primary + secondary + tertiary). The 4th slot is a follow-up.
- Sequential vs. parallel model loading is not configurable. On a GTX 1050 Ti (4 GB) loading TransReID + CLIP-SENet ViT-B simultaneously is borderline OOM; on Kaggle P100 (16 GB) it's fine.

---

## Section B' — Backend implementation plan (real fusion, no stub)

### B'.1 Reusable assets (lift directly)

From [scripts/eval/eval_14t_fusion_veri776.py](scripts/eval/eval_14t_fusion_veri776.py):
- `score_similarity()` (weighted sum of two L2-normed cosine matrices) — *the math is identical to what Stage 4 already does in its score-level fusion path*; no need to import.
- `average_query_expansion(features, k, iterations=1)` — *equivalent to* [src/stage4_association/query_expansion.py](src/stage4_association/query_expansion.py)::`average_query_expansion_batched`. No need to import.
- `compute_reranking_torch()` (k-reciprocal Jaccard rerank) — *equivalent to* [src/stage4_association/reranking.py](src/stage4_association/reranking.py)::`k_reciprocal_rerank`.
- The 14t **canonical hyperparameter shape** (`AQE_K=3`, `k1=80`, `k2=15`, `lambda=0.2`, `w_clipsenet=0.7`) which we use for the pre-seed.

What we do NOT lift:
- The eval script's argparse / VeRi-776 split parser / per-image feature extraction loop. The live pipeline already extracts per-tracklet features via Stage 2.
- The "extract once, evaluate many weights" workflow. The live pipeline runs once per request.

**Verdict:** the math is already implemented inside the live pipeline. The ONLY work is plumbing the request → Stage 2 multi-model config → Stage 4 fusion config.

### B'.2 Stage 2 multi-model extraction

**Existing capability:** Stage 2 already loads up to three vehicle models (`vehicle`, `vehicle2`, `vehicle3`) when their `enabled` flag is true and writes `embeddings_secondary.npy` / `embeddings_tertiary.npy` when `save_separate=true`. See [src/stage2_features/pipeline.py#L227-L290](src/stage2_features/pipeline.py).

**No code change required to Stage 2 itself for the 2- and 3-model cases.** The router / `resolve_pipeline_model` layer materialises the right `--override` strings.

**Memory budget note:**
- Kaggle P100 (16 GB): 2× ViT-B/16 fits with batch_size=64, half=true. 3× borderline at 256² input.
- Local GTX 1050 Ti (4 GB): only 1 ViT-B/16 at a time fits with half=true. Per project rules ("NEVER run GPU-intensive pipeline stages locally"), live fusion is **Kaggle-only or CPU-only on smoke clips** for local development. The frontend already surfaces a CPU-mode warning; we extend it to mention fusion. **No new sequential-extraction mode is required for this PR** — defer to a future PR if the local CPU smoke path is too slow.

**Config schema additions:** none. We reuse `cfg.stage2.reid.vehicle / vehicle2 / vehicle3` exactly as today.

### B'.3 Stage 4 score-level fusion

**Existing capability:** Stage 4 already computes per-stream FIC whitening, loads `secondary_embeddings.path/weight`, `tertiary_embeddings.path/weight`, `quaternary_embeddings.path/weight`, and validates `sec + tert + quat ≤ 1.0`. The fused similarity is then fed into AQE, mutual-NN filtering, and the conflict-free CC graph solver. See [src/stage4_association/pipeline.py#L162-L275](src/stage4_association/pipeline.py).

**Reranking caveat:** the 14t VeRi recipe applies k-reciprocal rerank AFTER AQE. The live MTMC Stage 4 has rerank disabled by default ([configs/datasets/cityflowv2.yaml](configs/datasets/cityflowv2.yaml) `reranking.enabled: false`) because rerank consistently hurts MTMC IDF1 (see findings.md "Reranking: Always hurts"). When the user toggles `fusion.rerank=true` in the frontend, the backend MUST surface a warning: *"Reranking is enabled in your fusion config. The live MTMC pipeline has reranking disabled by default because it has historically hurt cross-camera IDF1 by 1–3pp on CityFlowV2 (findings.md). The 14t recipe's rerank=on is a single-camera-VeRi-776 result and may not transfer."* The override is honored regardless — user's call.

**No code change required to Stage 4 itself.** The router materialises:
- `stage4.association.secondary_embeddings.path=outputs/<run_id>/stage2/embeddings_secondary.npy`
- `stage4.association.secondary_embeddings.weight=<w_secondary>`
- `stage4.association.tertiary_embeddings.path=...` (when 3 models)
- `stage4.association.query_expansion.k=<aqeK>`
- `stage4.association.reranking.enabled=<rerank>` (and its k1/k2/lambda when rerank=true)

### B'.4 Backend request shape

Top-level optional `fusion` field on `PipelineRunRequest` (defined in [backend/models/requests.py](backend/models/requests.py)). Add a typed pydantic submodel:

```python
class FusionModelEntry(BaseModel):
    model_id: str
    weight: float = Field(ge=0.0, le=1.0)

class FusionConfig(BaseModel):
    models: List[FusionModelEntry]      # length 2 or 3 (cap)
    aqe_k: int = Field(default=3, ge=1, le=20)
    k1: int = Field(default=80, ge=1, le=500)
    k2: int = Field(default=15, ge=1, le=200)
    lambda_value: float = Field(default=0.2, ge=0.0, le=1.0, alias="lambda")
    rerank: bool = True

class PipelineRunRequest(BaseModel):
    # ... existing fields ...
    fusion: Optional[FusionConfig] = None  # presence triggers fusion mode
```

Validation rules (router-side, on top of pydantic):
- `len(models) >= 2` and `<= 3` — return **422** if violated (3-model cap is the documented current ceiling; expand later).
- All `model_id` values exist in the registry.
- All resolved registry models have a `runnable_locally=true` flag (or the existing `_lookup_registry_model` returns a model with `pipeline_config` and `weights[].local_path`).
- `sum(weights) > 0`. Auto-normalize so `sum == 1.0`.
- Duplicate `model_id` values → 422.

### B'.5 `resolve_pipeline_model` extension

Extend [backend/services/pipeline_service.py](backend/services/pipeline_service.py)::`resolve_pipeline_model` to optionally accept a `fusion: Optional[FusionConfig]`:

```python
def resolve_pipeline_model(
    model_id: Optional[str] = None,
    dataset: Optional[str] = None,
    fusion: Optional[FusionConfig] = None,
) -> PipelineModelResolution:
    ...
```

When `fusion is not None`:

1. Sort `fusion.models` by weight descending; tie-break by request order.
2. The **primary** model becomes `model_id` for normal resolution. Run the existing single-model path on it to get `pipeline_config` + base `applied_overrides`.
3. For each non-primary model `M` at index `i ∈ {1, 2}`, look it up in the registry, then **append the following overrides** to `applied_overrides` (slot = `vehicle2` for i=1, `vehicle3` for i=2):
   ```
   stage2.reid.{slot}.enabled=true
   stage2.reid.{slot}.save_separate=true
   stage2.reid.{slot}.model_name=<M.weights[?].arch_name>      # from registry weights metadata
   stage2.reid.{slot}.weights_path=<M.weights[?].local_path>
   stage2.reid.{slot}.embedding_dim=<M.weights[?].embedding_dim>
   stage2.reid.{slot}.input_size=<M.weights[?].input_size>
   stage2.reid.{slot}.vit_model=<M.weights[?].vit_model or default>
   stage2.reid.{slot}.clip_normalization=<bool>
   ```
   The exact keys mirror the existing `vehicle2` / `vehicle3` blocks at [configs/default.yaml#L89-L107](configs/default.yaml#L89-L107). Where the registry doesn't carry an explicit `arch_name`, fall back to the weight's `local_path` filename heuristics (e.g. `transreid` → `vehicle_transreid_clip`, `clipsenet` → `clipsenet_v6`).
4. Compute fused weights: with primary weight `w_p = 1 - Σ w_others_normalized` (since Stage 4 treats primary as `1 - sum(others)`), append:
   ```
   stage4.association.secondary_embeddings.path=${project.output_dir}/${project.run_name}/stage2/embeddings_secondary.npy
   stage4.association.secondary_embeddings.weight=<w_secondary_normalized>
   ```
   And same for tertiary if present. The path values reuse the existing `${project.output_dir}/${project.run_name}` interpolation already produced by `_build_pipeline_cmd`.
5. Append AQE / rerank overrides:
   ```
   stage4.association.query_expansion.k=<aqe_k>
   stage4.association.reranking.enabled=<rerank>
   stage4.association.reranking.k1=<k1>      # only if rerank=true
   stage4.association.reranking.k2=<k2>
   stage4.association.reranking.lambda_value=<lambda>
   ```
6. Add a `fusion_summary` to the returned `PipelineModelResolution` dataclass:
   ```python
   fusion_summary: Optional[Dict[str, Any]] = None  # {primary, secondary, tertiary, aqe_k, rerank, ...}
   ```
   This is what the frontend "Effective Config" panel reads to render the fusion description.

The returned `applied_overrides` already lists every fusion-related override, so the existing UI "Effective Config" panel surfaces them without further changes.

### B'.6 Pre-seed defaults (frontend ↔ backend contract)

The pre-seed is purely client-side (Section A5). The backend doesn't pre-fill anything. On the wire:

```jsonc
{
  "runId": "42",
  "videoId": "...",
  "cameraId": "S02_c008",
  "dataset": "cityflowv2",
  "model_id": "veri776_clipsenet_v6",   // primary = highest weight (0.70)
  "fusion": {
    "models": [
      {"model_id": "veri776_clipsenet_v6", "weight": 0.70},
      {"model_id": "veri776_09v_v17_transreid", "weight": 0.30}
    ],
    "aqe_k": 3,
    "k1": 80,
    "k2": 15,
    "lambda": 0.2,
    "rerank": true
  },
  "config": { "dataset": "cityflowv2", "datasetName": "S02" }
}
```

### B'.7 Phased commit plan

Branch: `feature/pipeline-model-integration`. Each phase = one or more commits, each independently revertible.

| Phase | Title | Files touched | Validation gate | User-visible |
|-------|-------|---------------|-----------------|--------------|
| **P1** | Frontend: Zustand store extension + sidebar badge | [frontend/src/store/index.ts](frontend/src/store/index.ts), [frontend/src/components/layout/main-dashboard.tsx](frontend/src/components/layout/main-dashboard.tsx) | `npx tsc --noEmit && npx next lint --max-warnings 0` | Sidebar shows "No model selected" badge (expanded) / Cpu icon (collapsed). |
| **P2** | Frontend: Single↔Fusion toggle + multi-select picker + weight sliders + pre-seed | [frontend/src/components/stages/inference-stage.tsx](frontend/src/components/stages/inference-stage.tsx), new file `frontend/src/components/stages/fusion-model-panel.tsx` | tsc + lint clean; manual smoke: toggling Fusion on a fresh CityFlowV2 session shows the 14t pre-seed | Fusion UI works visually but Run Inference still sends Single-mode payload. |
| **P3** | Backend: extend `PipelineRunRequest` + `resolve_pipeline_model` validation only | [backend/models/requests.py](backend/models/requests.py), [backend/services/pipeline_service.py](backend/services/pipeline_service.py), [backend/routers/pipeline.py](backend/routers/pipeline.py), new pytest in `backend/tests/test_pipeline_router.py` | `pytest backend/tests/ -v -k pipeline` | Backend accepts `fusion` field, validates it, returns `applied_overrides` with the new fusion-related lines. NOT yet wired into subprocess. Returns `warning` listing what would have been overridden. |
| **P4** | Backend: materialise Stage 2 multi-model overrides | [backend/services/pipeline_service.py](backend/services/pipeline_service.py) (`resolve_pipeline_model` registry → `vehicle2/vehicle3` mapping), schema additions to [backend/models/registry.py](backend/models/registry.py) for arch metadata if missing, possible patches to [configs/model_registry.yaml](configs/model_registry.yaml) to expose `arch_name` / `vit_model` / `clip_normalization` per weight | pytest + manual smoke: run a 2-model CPU smoke pipeline on a 30-frame clip; verify `embeddings_secondary.npy` lands in stage2 output | Stage 2 produces 2 feature streams. Stage 4 still uses primary only because no `secondary_embeddings.weight` override yet. |
| **P5** | Backend: materialise Stage 4 score-fusion overrides + AQE/rerank | same files as P4 | pytest + manual smoke: same 2-model run; verify `stage4.json` reflects the fused similarity and IDF1 differs from primary-only run | End-to-end fusion runs. Frontend still shows the P3 warning until P6. |
| **P6** | Frontend: send `fusion` field on Run; remove "not executed" warnings; surface backend `fusion_summary` in Effective Config panel | [frontend/src/components/stages/inference-stage.tsx](frontend/src/components/stages/inference-stage.tsx) | tsc + lint + manual full-stack smoke (CityFlowV2 single camera, 14t pre-seed) | User can run real fusion pipelines from the UI. |
| **P7** | Validation: tiny end-to-end fusion smoke test | new `backend/tests/test_pipeline_fusion_smoke.py` (uses `--smoke-test` flag and existing tiny CityFlowV2 sample) | `pytest backend/tests/test_pipeline_fusion_smoke.py -v` runs in <60 s in CI on CPU | CI gate exists for fusion regressions. |

Tag the merge commit `pipeline-fusion-v2` so the whole feature can be reverted as a unit.

### B'.8 Honest constraints / blockers

- **Local GPU OOM:** GTX 1050 Ti (4 GB) cannot run two ViT-B/16 ReID models simultaneously on the live pipeline. Mitigation: forced CPU mode for fusion runs locally OR Kaggle-only execution. The frontend warning banner mentions this.
- **Quaternary slot:** the existing `vehicle3` is the last Stage 2 producer slot. We cap UI fusion at 3 models. Adding a `vehicle4` extractor is mechanical (~30 lines mirroring vehicle3) but is OUT OF SCOPE for this PR.
- **Registry weight metadata:** today the registry stores `arch_name` / `vit_model` / `clip_normalization` only implicitly (via path conventions and per-config defaults). P4 may need a small registry-schema patch to surface these explicitly so `resolve_pipeline_model` can build correct `vehicle2/vehicle3` blocks for arbitrary models. If the patch is too invasive for one PR, fall back to a per-model-id lookup table baked into `pipeline_service.py` for the 14t pair plus the 14e DINOv2 tertiary; document the limitation and ship.
- **Person pipeline:** `cfg.stage2.reid.person` has no `person2` / `person3` slots. Fusion mode for `mtmc_person` checkpoints would no-op silently. P4 must validate that all selected models map to compatible Stage 2 slots; if any selected model has `task_type=mtmc_person` and the dataset is not one supporting person-fusion, return 422 with a clear message.
- **Cross-domain pre-seed:** documented in A5. The frontend banner is mandatory.

---

## Section C — File-by-file change list

### C1. [frontend/src/store/index.ts](frontend/src/store/index.ts) — MODIFY
**Lines:** add to `PipelineState` interface (after `error: string | null;`, before `downstreamInvalidateGeneration`), and to the `create()` initial state + actions.

Add types:
```ts
export type PipelineModelMode = "single" | "fusion";
export interface PipelineFusionConfig {
  selectedModelIds: string[];
  weights: Record<string, number>;
  aqeK: number;        // default 2 (live MTMC); pre-seed sets to 3 for 14t
  rerank: boolean;     // default false; pre-seed sets to true for 14t
  k1: number;          // default 80
  k2: number;          // default 15
  lambdaValue: number; // default 0.2
  seeded: boolean;
}
export interface PipelineSelectedModelMeta {
  name: string;
  dataset: string;
  headlineMetric: { name: string; value: number } | null;
}
```

Add to `PipelineState`:
```ts
modelMode: PipelineModelMode;
selectedModelId: string | null;
selectedModelMeta: PipelineSelectedModelMeta | null;
fusion: PipelineFusionConfig;

setModelMode: (mode: PipelineModelMode) => void;
setSelectedModelId: (id: string | null) => void;
setSelectedModelMeta: (meta: PipelineSelectedModelMeta | null) => void;
setFusionConfig: (cfg: Partial<PipelineFusionConfig>) => void;
```

Default fusion config:
```ts
const DEFAULT_FUSION: PipelineFusionConfig = {
  selectedModelIds: [],
  weights: {},
  aqeK: 2,           // live MTMC default
  rerank: false,
  k1: 80,
  k2: 15,
  lambdaValue: 0.2,
  seeded: false,
};
```

Initial values in `create()`:
```ts
modelMode: "single",
selectedModelId: null,
selectedModelMeta: null,
fusion: DEFAULT_FUSION,
```

Extend `reset()` to also reset these four fields.

### C2. [frontend/src/components/stages/inference-stage.tsx](frontend/src/components/stages/inference-stage.tsx) — MODIFY
**Lines:** ~115–135 (state declarations), ~226–365 (`handleRunInference`), ~620–650 (Model Registry card).

Replace local `useState` for `selectedModelId` and `selectedRegistryModel` with reads from `usePipelineStore`. Keep a small local `useState<ModelEntry | null>` for the full `ModelEntry` object; mirror its headline metric into the store via `setSelectedModelMeta`.

**Add a Mode toggle Card** between "Dataset Source" Card and "Location Filter" Card:
```tsx
<Card>
  <CardHeader className="pb-2">
    <CardTitle className="flex items-center gap-2 text-sm">
      <Cpu className="h-4 w-4" /> Model Mode
    </CardTitle>
  </CardHeader>
  <CardContent>
    <div className="inline-flex rounded-md border p-0.5">
      <button onClick={() => setModelMode("single")} className={...}>Single</button>
      <button onClick={() => setModelMode("fusion")} className={...}>Fusion</button>
    </div>
    <p className="mt-2 text-xs text-muted-foreground">
      {modelMode === "single"
        ? "Run inference with a single ReID model."
        : "Run inference with weighted score-level fusion across 2–3 ReID models. Live pipeline executes the full multi-model path."}
    </p>
    {modelMode === "fusion" && datasetIsCityFlow && (
      <div className="mt-2 rounded-md border border-amber-500/40 bg-amber-500/10 p-2 text-xs text-amber-200">
        The default 14t preset uses VeRi-776 single-cam experts. On CityFlowV2 MTMC, expect lower IDF1 than the production 14e B1 baseline due to the cross-domain gap (see findings.md 13d/13f).
      </div>
    )}
  </CardContent>
</Card>
```

**Pre-seed effect (Section A5):**
```tsx
useEffect(() => {
  if (fusion.seeded) return;
  if (modelMode !== "fusion") return; // only seed on first toggle to fusion
  if (datasetSlug === "cityflowv2") {
    setFusionConfig({
      selectedModelIds: ["veri776_clipsenet_v6", "veri776_09v_v17_transreid"],
      weights: { veri776_clipsenet_v6: 0.70, veri776_09v_v17_transreid: 0.30 },
      aqeK: 3, rerank: true, k1: 80, k2: 15, lambdaValue: 0.2,
      seeded: true,
    });
  } else {
    setFusionConfig({ seeded: true });
  }
}, [modelMode, datasetSlug, fusion.seeded, setFusionConfig]);
```

**Modify "Model Registry" Card content** to switch on `modelMode`:
- Single (existing): `<ModelPicker selectedId={selectedModelId} onSelect={setSelectedModelId} onModelChange={...} />`
- Fusion: `<FusionModelPanel />` (Section C5).

**Modify `handleRunInference`:**
- Compute `effectiveModelId`:
  - Single → `selectedModelId`
  - Fusion → highest-weight model; tiebreak first in `selectedModelIds`
- Build top-level `fusion` payload (NOT under `config`):
  ```ts
  const fusionPayload = modelMode === "fusion" && fusion.selectedModelIds.length >= 2
    ? {
        models: fusion.selectedModelIds.map((id) => ({ model_id: id, weight: normalizedWeights[id] ?? 0 })),
        aqe_k: fusion.aqeK,
        rerank: fusion.rerank,
        k1: fusion.k1, k2: fusion.k2, lambda: fusion.lambdaValue,
      }
    : null;
  ```
- Pass `fusion: fusionPayload` to BOTH `runStage(2, ...)` and `runStage(3, ...)` as a top-level request field.
- Guard: if `modelMode === "fusion"` and `fusion.selectedModelIds.length < 2`, set a local error and abort.

### C3. [frontend/src/components/layout/main-dashboard.tsx](frontend/src/components/layout/main-dashboard.tsx) — MODIFY
**Lines:** ~145–165 (after the Dataset button, before closing `</nav>`).

Add badge identical to the original spec (expanded card vs collapsed Cpu icon, click → `setCurrentStage(3)`). State reads:
```ts
const modelMode = usePipelineStore((s) => s.modelMode);
const selectedModelId = usePipelineStore((s) => s.selectedModelId);
const selectedModelMeta = usePipelineStore((s) => s.selectedModelMeta);
const fusionIds = usePipelineStore((s) => s.fusion.selectedModelIds);
```

### C4. `frontend/src/components/reid/FusionWeightSliders.tsx` — NO CHANGE

### C5. `frontend/src/components/stages/fusion-model-panel.tsx` — CREATE NEW
Encapsulates Stage 3 fusion UI. Reads + writes via `usePipelineStore`. Structure:
1. `<ModelPicker multiSelect selectedIds={...} onMultiSelect={...} compact />` — **no `taskType` filter** (Section A6).
2. `<FusionWeightSliders models={selectedFullEntries} weights={fusion.weights} onChange={...} />`
3. AQE K slider (1–20, default 2; 14t preset = 3).
4. Rerank checkbox (default off; 14t preset = on). When on, expose k1/k2/lambda inputs (default 80/15/0.2).
5. Validation hint when `<2` models selected.
6. Effective config preview (read-only) listing models + normalized weights.

Fetches `ModelEntry[]` once via `fetchModels()` (no filter).

### C6. [backend/models/requests.py](backend/models/requests.py) — MODIFY
Add `FusionModelEntry` and `FusionConfig` pydantic models (see B'.4). Add `fusion: Optional[FusionConfig] = None` to `PipelineRunRequest`. Use `Field(alias="lambda")` for `lambda_value` to keep the wire shape clean.

### C7. [backend/services/pipeline_service.py](backend/services/pipeline_service.py) — MODIFY
**Phase P3:** extend `resolve_pipeline_model(..., fusion=None)` signature. When `fusion` is non-None, validate (length 2–3, unique IDs, weights ≥ 0, sum > 0, all IDs runnable, all task_type compatible with dataset slot) and store the validated fusion summary on the returned dataclass; do NOT yet append overrides.

**Phase P4:** when `fusion` non-None, append `stage2.reid.vehicle2.*` and (if 3-model) `stage2.reid.vehicle3.*` overrides built from the registry weight metadata of each non-primary model. Where registry metadata is missing fields, fall back to a small per-model-id lookup table (initially populated for `veri776_clipsenet_v6`, `veri776_09v_v17_transreid`, and the 14e DINOv2 tertiary).

**Phase P5:** also append `stage4.association.{secondary,tertiary}_embeddings.{path,weight}` and `stage4.association.query_expansion.k`, `stage4.association.reranking.{enabled,k1,k2,lambda_value}` overrides. Use `${project.output_dir}/${project.run_name}/stage2/embeddings_secondary.npy` for the secondary path so it stays consistent with the run's actual output dir.

### C8. [backend/routers/pipeline.py](backend/routers/pipeline.py) — MODIFY
In `run_stage` and `run_full_pipeline`, accept the new `payload.fusion` field, pass it to `resolve_pipeline_model()`. Persist `fusion` and `fusion_summary` into `state.active_runs[run_id]` and into `run_context.json`. On validation errors raise 422 with a clear message.

### C9. [backend/tests/test_pipeline_router.py](backend/tests/test_pipeline_router.py) — MODIFY (or create if missing)
Add unit tests for: 2-model valid fusion → 200 + `applied_overrides` contains `stage2.reid.vehicle2.*` and `stage4.association.secondary_embeddings.*`; 1-model fusion → 422; 4-model fusion → 422; duplicate model_ids → 422; weights summing to 0 → 422; unknown model_id → 422.

### C10. [backend/tests/test_pipeline_fusion_smoke.py](backend/tests/test_pipeline_fusion_smoke.py) — CREATE NEW (P7)
End-to-end smoke that posts a fusion run with the 14t pre-seed, runs `--smoke-test` on a 30-frame CPU clip, and asserts `embeddings_secondary.npy` exists and `stage4.json.global_trajectories` is non-empty.

---

## Section D — UI/UX mockup (Stage 3, top-to-bottom)

```
┌───────────────────────────────────────────────────────────┐
│ Header: "Stage 2-3: Inference"           [N objects]      │
├───────────────────────────────────────────────────────────┤
│ Error banner (if any) / Warning banner                    │
├───────────────────────────────────────────────────────────┤
│ ┌─ Dataset Source ────────────────────────────────┐       │
│ │ [Uploaded] [S01] [S02] [WILDTRACK] ...          │       │
│ └──────────────────────────────────────────────────┘       │
│                                                            │
│ ┌─ Model Mode ───────────────────────── NEW ─────┐        │
│ │ ( Single | Fusion )                            │        │
│ │ helper text                                    │        │
│ │ [amber banner: cross-domain caveat for 14t]    │        │
│ └─────────────────────────────────────────────────┘        │
│                                                            │
│ ┌─ Location Filter ────────────────────────────────┐      │
│ └───────────────────────────────────────────────────┘      │
│                                                            │
│ ┌─ Model Registry ─────────────────────────────────┐      │
│ │ if mode==single: <ModelPicker single>            │      │
│ │ if mode==fusion: <FusionModelPanel>              │      │
│ │   - <ModelPicker multi compact, no taskType>     │      │
│ │   - <FusionWeightSliders>                        │      │
│ │   - AQE K slider (default 2; preset 3 for 14t)   │      │
│ │   - [ ] Rerank (default off; preset on for 14t)  │      │
│ │     when on: k1=80, k2=15, lambda=0.2 inputs     │      │
│ │   - Validation: "Pick at least 2 models"         │      │
│ │   - Effective config preview                     │      │
│ └───────────────────────────────────────────────────┘      │
│                                                            │
│ ┌─ Effective Config ──────────────────────────────┐       │
│ │ Model / Pipeline YAML / Applied overrides        │       │
│ │ (now includes the synthesized fusion overrides)  │       │
│ └──────────────────────────────────────────────────┘       │
│                                                            │
│ ┌─ Active Pipeline Parameters (read-only) ────────┐       │
│ └──────────────────────────────────────────────────┘       │
│                                                            │
│ ┌─ Date & Time Range ──────────────────────────────┐      │
│ └───────────────────────────────────────────────────┘      │
│                                                            │
│            [ Run Inference → ]                             │
└────────────────────────────────────────────────────────────┘
```

**Sidebar (expanded):** unchanged from v1 — small card with model name + headline metric, or "Fusion · N models" + "aqeK X, rerank on/off".

**Sidebar (collapsed):** small `Cpu` icon; tooltip on hover; click → setCurrentStage(3).

---

## Section E — Validation plan

### E1. Frontend
- `cd frontend; npx tsc --noEmit` → exit 0
- `cd frontend; npx next lint --max-warnings 0` → exit 0
- `frontend/src/app/fusion/page.tsx` still type-checks unchanged.

### E2. Backend
- `pytest backend/tests/ -v -k pipeline` — must pass.
- New tests in C9 + C10.

### E3. Manual smoke (Single mode) — unchanged from v1.

### E4. Manual smoke (Fusion mode, end-to-end)
1. CityFlowV2 dataset; Stage 3; toggle to Fusion → 14t pre-seed appears.
2. Click Run Inference → backend returns 200 with `fusion_summary` populated and `applied_overrides` listing `stage2.reid.vehicle2.*` + `stage4.association.secondary_embeddings.*` + `stage4.association.query_expansion.k=3` + `stage4.association.reranking.enabled=true`.
3. Stage 2 produces both `embeddings.npy` and `embeddings_secondary.npy`.
4. Stage 4 logs include `Secondary embeddings loaded: ... (score-level fusion)` and `weight=0.30`.
5. Final `global_trajectories.json` is non-empty.
6. Sidebar badge shows "Fusion · 2 models · aqeK 3, rerank on".

### E5. Negative tests
- Fusion with 1 model → frontend Run button disabled; backend returns 422 if the request is hand-crafted.
- Fusion with `mtmc_person` checkpoints on `cityflowv2` dataset → 422 with the slot-incompatibility message.

---

## Section F — Risks & residual open questions

### F1. Resolved decisions (no longer open)
1. ✅ Fusion is a **real backend feature**, not a stub.
2. ✅ Pre-seed = 14t canonical (TransReID + CLIP-SENet VeRi-776) on CityFlowV2; empty on other datasets.
3. ✅ Fusion picker has **no task_type filter** — all task types selectable.
4. ✅ Cross-domain banner is mandatory for the CityFlowV2 14t preset.

### F2. Remaining open questions
1. **3-model cap.** OK to ship at 2–3 models for now and defer the `vehicle4` extractor to a follow-up? Spec assumes yes.
2. **Registry metadata patch.** Is it acceptable to add explicit `arch_name` / `vit_model` / `clip_normalization` fields to per-weight metadata in [configs/model_registry.yaml](configs/model_registry.yaml), OR should we ship with a hardcoded id→arch lookup table inside `pipeline_service.py`? Spec assumes the latter for v2; the former is a follow-up.
3. **Local fusion runnability.** Confirm the project is OK with fusion mode being effectively Kaggle-only locally on the 1050 Ti, with CPU mode only for tiny smoke clips.

### F3. Backwards-compat
- Existing single-model runs unchanged — `fusion=None` is the default.
- `/fusion` page (`/api/v1/reid/fusion`) is independent and untouched.
- `PipelineRunRequest.fusion` is optional; existing clients that don't send it work as before.

### F4. Risks
- **Risk:** registry weight metadata insufficient to construct correct `vehicle2/vehicle3` blocks for every selected model. **Mitigation:** hardcoded id→arch lookup table for the 14t pair (and 14e DINOv2) in P4; clear error if a user picks an unsupported id.
- **Risk:** 14t pre-seed underperforms expectations on CityFlowV2 → user impression of broken fusion. **Mitigation:** the cross-domain banner.
- **Risk:** Stage 2 OOM with two ViT-B/16 models on Kaggle P100 if both run at 320² simultaneously. **Mitigation:** force `stage2.reid.{slot}.input_size=256x256` and `batch_size=32` in fusion mode by default; document in the warning banner.
- **Risk:** rerank=true in the live pipeline regresses MTMC IDF1. **Mitigation:** the rerank=on warning + user opt-in.

---

## Section G — Rollback plan / commit chain

See B'.7 for the full phased plan (P1–P7). Each phase is independently revertible.

Tag the merge commit `pipeline-fusion-v2`.

