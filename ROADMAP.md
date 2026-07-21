# ATHAR v2 — Rebuild Roadmap & Living Checklist

> **This file is the single source of truth** for the rebuild: decisions, phase
> checklist, parity gates, and open items. Update it in every working session —
> mark items done, add discoveries, never delete decision history.
>
> Last updated: **2026-07-21** · Current phase: **Phase 1 — Skeleton & contracts**

---

## 1. Vision & non-negotiables

Enterprise/intelligence-grade forensic tracking platform (target users: major
forensic teams, e.g. Egyptian MOI; sites: malls closed+open, home compounds,
smart cities, streets).

- **Multi-class**: detect AND track people + vehicles simultaneously from the
  same video. Dataset-agnostic by default; all dataset/site calibration is an
  optional plugin that improves accuracy when present.
- **Flagship workflow**: batch-preprocess offline videos into searchable
  galleries → search them with a reference (probe) video or crop → cross-camera,
  cross-video trajectory reconstruction → case report.
- **Offline forensic now, live-ready seams** (`FrameSource` protocol + windowed
  association) — RTSP/live and blacklist alerting are **post-v1**.
- **Forensic honesty**: ranked hypotheses with evidence, calibrated confidence
  (never raw cosine as certainty), investigator confirm/reject with audit trail,
  full reproducibility chain (video SHA → config hash → model SHA → result).
- **Air-gap ready**: no cloud calls at runtime; remote compute (Kaggle) is an
  opt-in job executor for non-sensitive data only.
- **UI/UX**: professional enterprise-grade dashboard; Arabic/RTL support planned
  from the start.

## 2. Decision record (locked)

| # | Date | Decision |
|---|------|----------|
| D1 | 07-21 | **"New chassis, transplanted engine"**: port v1 algorithm kernels verbatim; rewrite orchestration, state, backend, frontend. Council verdict (4 members, unanimous). |
| D2 | 07-21 | Parity gates before/during port: VeRi mAP 93.3 → CityFlow IDF1 0.779 → WILDTRACK 0.946/0.903 (tolerance ±0.002 IDF1). Backend rebuilt only after gates pass. |
| D3 | 07-21 | Multi-class runs (person+vehicle in one pass, per-class branch pipelines) replace per-run single-class profiles. |
| D4 | 07-21 | Three-tier model strategy: (1) frozen prod set + DINOv2 generalist stream + per-camera statistical calibration; (2) ONE joint multi-domain retrain (VeRi+CityFlow+VeRi-Wild+VehicleID, domain-balanced — never sequential finetune); (3) per-deployment unsupervised adaptation ("calibration mode") gated by eval harness. |
| D5 | 07-21 | Model registry = lifecycle DB (candidate→validated→production→retired), SQLite source of truth, YAML authoring only, content-addressed weights (SHA-256). |
| D6 | 07-21 | Config: every run freezes fully-resolved config + per-key provenance (profile/deployment/case/run layer) + config hash in the manifest. Typed (pydantic), fail-at-submit. |
| D7 | 07-21 | Identity model: case-level **Target** spans appearance clusters via ranked hypothesis edges; clothes-change cues = face + gait + CC-ReID (face+gait IN scope, pretrained); person↔vehicle boarding/alighting via interaction-event detector → multi-entity graph. |
| D8 | 07-21 | Plugin planes: pipeline components / analytic apps (watchlist, LPR, 3D viz — post-v1) / UI panels. Signed + permissioned manifests for deployments. |
| D9 | 07-21 | Spatial model is a plugin: GPS (haversine reachability — port v1 `geospatial.py`) \| floor-plan graph (indoor malls) \| learned transition-time topology (mined from footage). Per-camera ground-plane homography optional plugin. |
| D10 | 07-21 | TimeBase is a first-class per-camera artifact (source: assumed/manual/OCR/event-alignment) — never bare `time_offsets: {}`. |
| D11 | 07-21 | Normalize-on-ingest boundary: hash original evidence, transcode to canonical copy, record transforms; fisheye dewarp as calibration plugin. Frames decoded on demand — never extracted wholesale to disk. |
| D12 | 07-21 | Licensing: **assume no restrictions** (user decision) — keep Ultralytics YOLO + BoxMOT lineage. |
| D13 | 07-21 | Compute: 4 Kaggle accounts × 30 h/wk (120 GPU-h/wk, 8 concurrent). Sensitive/MOI footage never leaves premises → training/adaptation jobs are executor-agnostic (kaggle \| local). |
| D14 | 07-21 | Shorouk dataset (14 synced 1080p cams, ~21.5 min, GPS coords) = **ATHAR-Bench v0** eval seed (annotate in-house), NOT training data. |
| D15 | 07-21 | Night/IR: per-segment IR detection + dynamic stream reweighting (HSV weight → 0 on IR). IR footage required in bench before night claims. |
| D16 | 07-21 | Branch layout: `athar-v2` = orphan rebuild branch; v1 stays on `seif_final`; legacy access via `git restore --source=seif_final`. |
| D17 | 07-21 | **Stack researched & pinned** (live-verified 2026-07-21; full rationale in [docs/STACK.md](docs/STACK.md)): Python 3.13 + uv; torch 2.13 cu130; ultralytics 8.4 / **YOLO26**; boxmot 22 / **BoostTrack**; torchcodec (frame-exact decode — cv2 seeking retired); faiss-cpu exact IP; opencv headless 4.13 (5.x blocked by boxmot); insightface 1.0 buffalo_l; OpenGait; CION ReIDZoo as person-ReID retrain base. Dropped as dead: decord, py-motmetrics, upstream TrackEval (sn-trackeval fork for parity), torchreid, passlib. |
| D18 | 07-21 | **Parity profile ≠ production profile.** Parity gates P1–P3 pin v1 components (YOLO26m, v1 tracker+config, TrackEval-compatible metrics). Upgrades (YOLO26x @1280, BoostTrack, optional RF-DETR-XL second pass) live in production profiles and are benchmarked against the parity baseline — never silently swapped. |
| D19 | 07-21 | App stack: FastAPI 0.139 + uvicorn; SQLAlchemy 2 on SQLite WAL; jobs = own SQLite file + `UPDATE…RETURNING` claims in separate worker processes; **server-side sessions** + pwdlib argon2 + dependency-based RBAC (fastapi-users/passlib dead); SSE for progress; structlog + hash-chained audit; **Playwright PDF** (WeasyPrint Arabic-broken). Frontend: Next.js 16 + React 19, Tailwind 4 (logical props → RTL), **shadcn/ui on Base UI**, next-intl, TanStack Table v8, **MapLibre + PMTiles** (air-gapped maps), Recharts+ECharts, Konva video overlays + custom canvas timeline, hey-api client, Node 24 + pnpm, TS 5.9 pinned. |

## 3. Phase checklist

### Phase 0 — Safety net & hygiene
- [x] Council review (4 members) + salvage audit + red team + hygiene inventory
- [x] Fresh orphan branch `athar-v2`
- [x] Junk purge (caches/logs/tmp deleted; zips + untracked sweep scripts + kaggle notebooks preserved in `_legacy_archive/`)
- [x] Fresh `.gitignore`, `README.md`, this roadmap
- [ ] Golden artifacts: CityFlow stage2/stage4 goldens do NOT exist locally (v1 generated them on Kaggle) — freeze VeRi feature dumps from the parity run instead; CityFlow goldens come from the first local v1 pipeline run (long GPU job, schedule deliberately)
- [x] Checkpoints verified: all 8 manifest SHA-256s match on disk; both TransReID checkpoints load through the PORTED module with the exact v1 key contract (tests/test_checkpoint_contract.py)
- [ ] VeRi 93.3 reproduction: full eval RUNNING in background (v1 worktree `../gp-v1`, GTX 1050 Ti, smoke passed) → becomes Gate P1 baseline
- [x] Local branch pruning: deleted 7 safe branches (merged and/or identical to origin). KEPT pending user decision: `feature/people-tracking` + `master-legacy` (local differs from origin — possible unpushed work), `repro/osnet-secondary` (exists nowhere else). Remote pruning still needs user OK.
- [x] v1 reference worktree at `../gp-v1` (seif_final checkout) for running legacy code + verbatim ports

### Phase 1 — Skeleton & core contracts  ← CURRENT
- [x] `athar/` package layout + `pyproject.toml`
- [x] Core ids/types (`EntityClass`, `TrackKey`, `BBox`, `Tracklet`, `Trajectory`)
- [x] `TimeBase` contract (D10)
- [x] `ResolvedConfig` with per-key provenance + hash (D6)
- [x] `RunManifest` + artifact records + filesystem run store (single root `data/runs/`)
- [x] Component protocols (FrameSource, Detector, MultiViewDetector, Tracker, Embedder, FeatureRefiner, ScoreTerm, Solver, SpatialModel, InteractionEventDetector) + registry
- [x] Multi-class `RunProfile` / `ClassBranch` (D3)
- [x] Typed pipeline events (no stdout regex ever again)
- [x] Case/Gallery/Probe/Target + HypothesisEdge models (D7)
- [x] Model lifecycle registry types (D5)
- [x] Contract unit tests green
- [x] Dependency/stack research (live-verified) → [docs/STACK.md](docs/STACK.md) + pinned pyproject groups (D17–D19)
- [x] Config authoring loader (YAML → validated layers → ResolvedConfig); `athar config resolve` shows per-key provenance
- [x] `athar` CLI: subcommand scaffold (`config resolve` live; `run`/`models`/`migrate` stubs point at their phases)
- [ ] uv lockfile (`uv lock`) with cu130/cpu torch indexes — first real install (upgrade local uv 0.8.2 → current first; do at Phase 2 start)

### Phase 2 — Pipeline port (parity-gated)
- [x] Port `configs/model_registry.yaml` + `weights_manifest.yaml` + schema (verbatim) and `scripts/download_weights.py`
- [x] Port TransReID model **verbatim** (`athar/components/embedders/transreid_model.py`; sole change: loguru → stdlib logging) + checkpoint-contract tests (both frozen checkpoints load; only cls_head/jpm_cls drops; unit-norm 768-d output)
- [ ] Ingest: normalize-on-ingest (hash, transcode, VFR handling), on-demand frame decode (D11)
- [x] Port stage1 kernels verbatim: `detector`, `tracker`, `tracklet_builder`, `bidirectional`, `bidirectional_merge`, `ssa` → `athar/components/tracking/` + 3 v1 test modules (12 tests). Note: written against ultralytics 8.4.23 / boxmot 12 APIs — re-validate wrappers against boxmot 22 at uv-lock time
- [ ] Wrap stage1/stage2 kernels behind the v2 component protocols (Detector/Tracker/Embedder adapters)
- [ ] Port remaining stage2 kernels: CLIP-SENet (fix HF-hub-first load order — air-gap bug found during smoke), DINOv2, HSV extractor, PCA whitener (pickled PCA = checkpoint)
- [ ] Port stage3: FAISS index + tracklet catalog (SQLite)
- [x] Port stage4 kernels **verbatim** (9 of 11): `similarity`, `reranking`, `query_expansion`, `graph_solver`, `camera_bias`, `fic`, `geospatial`, `spatial_temporal`, `zone_scoring` → `athar/components/associators/` + `hsv_extractor`/`pca_whitening` → embedders + `athar/core/constants.py`; 6 v1 test modules carried (101 tests green). Sole change everywhere: loguru → stdlib logging (verified no loguru-only APIs, all f-strings)
- [x] Port `aflink` + `occlusion` + `global_trajectories` + v1 `data_models`/`video_utils` (athar/core), `faiss_index`/`metadata_store` (athar/components/indexing), `format_converter`/`metrics`/`ground_plane_eval`/`evaluate_reid` (athar/evaluation), `clip_senet_model`/`reid_model`/`robust_pool`/`crop_extractor` (embedders), `training/model.py` (IBN archs) — 64 more v1 tests carried
- [x] Port VeRi eval trio (`scripts/eval/eval_{09v_transreid,clip_senet,14t_fusion}_veri776.py`) + `athar/serving/reid_loaders.py` byte-faithful (loader↔script cycle kept intact for parity; clean inversion = Phase 4 serving refactor) + `test_14t_fusion_math`
- [x] **Gate P1 test written**: `tests/parity/test_veri_fusion.py` runs the PORTED evaluator end-to-end (`ATHAR_RUN_PARITY=1 pytest -m parity`), asserts mAP 93.32 ± 0.2pt
- [ ] `test_multi_query` (needs stage2/stage3 pipeline glue)
- [ ] New DAG runner with resume (stage- and chunk-level checkpointing)
- [ ] Port stage5: TrackEval integration + format converter
- [ ] **GATE P1: VeRi-776 fusion mAP = 93.3 ± 0.2** (pytest `-m parity`)
- [ ] **GATE P2: CityFlowV2 MTMC IDF1 = 0.779 ± 0.002**
- [ ] IR/grayscale segment detection + dynamic stream reweighting (D15)

### Phase 3 — Multi-class & profiles
- [ ] `vehicle+person` joint profile: one detection pass (COCO 0,2,5,7), per-class branches
- [ ] MVDeTr as MultiViewDetector component; `person_multiview` profile
- [ ] **GATE P3: WILDTRACK IDF1 0.946 / MODA 0.903**
- [ ] **GATE P4 (new): generic person tracking benchmark** (MOT17-style or WILDTRACK-as-plain-video) — no number exists today; establish baseline
- [ ] Spatial-model plugins: GPS (port `geospatial.py`), learned transition-time topology; floor-plan graph interface

### Phase 4 — Serving, jobs, API
- [ ] Unified model loader + refcounted LRU cache with VRAM budget + DeviceManager
- [ ] Model lifecycle registry impl (SQLite) + promote/rollback + eval-gate enforcement
- [ ] JobService: SQLite queue, worker subprocess, typed event pipe, cancel/resume, executor-agnostic (local | kaggle) (D13)
- [ ] FastAPI app: thin routers → services → RunRepository; OpenAPI → generated TS client
- [ ] AuthN/AuthZ (users, roles, case ownership) + tamper-evident append-only audit log
- [ ] Search engine: probe-vs-gallery with embedding-provenance compatibility checks (explicit 409, no silent PCA fallback); calibrated score→probability

### Phase 5 — Frontend (enterprise-grade)
- [ ] Next.js App Router scaffold, design system (shadcn/ui + Tailwind), dark/light, **Arabic/RTL i18n scaffolding from day 1**
- [ ] Case workspace: galleries, probes, search, results review (confirm/reject hypotheses with audit)
- [ ] Cross-camera timeline + map view (GPS cameras; Shorouk coords) + evidence clips
- [ ] Job monitoring (live typed events), run detail with resolved-config + provenance viewer
- [ ] Report export (PDF, chain-of-custody chain: video SHA → config hash → model SHA → results)

### Phase 6 — Domain accuracy campaign
- [ ] ATHAR-Bench v0: annotate Shorouk (cross-camera identities, sparse — cheap) (D14)
- [ ] Cross-domain eval matrix in CI (train-domain × test-domain incl. VeRi-Wild held out)
- [ ] Joint multi-domain vehicle ReID retrain on Kaggle (D4 tier 2)
- [ ] Person: CC-ReID embedder + face (pretrained ArcFace-class) + gait (pretrained OpenGait-class) score terms
- [ ] Unsupervised adaptation job ("calibration mode") + per-camera statistical calibration
- [ ] Collect: mall interior, open mall, street, night/IR footage → bench v1

### Phase 7 — Hardening & delivery
- [ ] Ingest fuzzing (malformed/truncated/VFR video), kill-and-resume chaos tests, scale test (≥14 cams)
- [ ] Locked envs (uv/conda-lock) + Docker + offline air-gapped installer bundle
- [ ] Determinism policy doc (seeds, CUDA flags, tolerance bands)
- [ ] Drift observability (det/min, embedding-norm, match-rate profiles per run)
- [ ] Security review (path traversal, plugin permissions, encryption at rest)

### Post-v1 (designed-for, deferred)
- [ ] Live RTSP ingestion + windowed association | [ ] Watchlist/blacklist alerting | [ ] LPR (Arabic plates) | [ ] 3D visualization | [ ] Floor-plan editor

## 4. Parity gates

| Gate | Benchmark | Metric | Target | Status |
|------|-----------|--------|--------|--------|
| P1 | VeRi-776 two-stream fusion | mAP | 93.3 ± 0.2 | ☐ not run |
| P2 | CityFlowV2 MTMC | IDF1 | 0.779 ± 0.002 | ☐ not run |
| P3 | WILDTRACK (MVDeTr) | IDF1 / MODA | 0.946 / 0.903 | ☐ not run |
| P4 | Generic person tracking | IDF1 | establish baseline | ☐ no number exists |
| P5 | ATHAR-Bench v0 (Shorouk) | IDF1 + retrieval | establish baseline | ☐ needs annotation |

## 5. Port map (from council salvage audit)

- **Verbatim**: `stage4_association/{similarity,reranking,query_expansion,graph_solver}.py` + helpers; `stage2_features/{transreid_model,hsv_extractor,pca_whitening}.py`; `stage5_evaluation/{metrics,format_converter}.py`; `scripts/eval/eval_14t_fusion_veri776.py`; `configs/{model_registry,weights_manifest}.yaml`; v1 `tests/test_stage4/`, TransReID key-contract tests.
- **Refactor**: `serving/reid_loaders.py` (kill module globals + fixed lru_cache); stage helper configs → typed schema; MVDeTr txt-parsing + ground-plane tracking (extract functions first).
- **Rewrite**: all `pipeline.py` orchestrators; backend entirely; run identity/state; config layering; frontend.
- **Drop**: Kaggle-dispatch-as-default, Streamlit apps, dead config flags (v25–v49), `backend_api.py` shim.

## 6. Open items needing user input

- [ ] Remote branch pruning approval (`origin/backedn`, stale feature/fix/verify branches)
- [ ] Shorouk annotation plan (who labels, which tool — CVAT?)
- [ ] Night/IR + mall footage acquisition
- [ ] Deployment hardware target (server spec, GPU) — sizes DeviceManager & installer
- [ ] Confirm keeping `_legacy_archive/` on disk or moving it out of the repo folder

## 7. Working agreements

- Update this file every session (checkboxes + decision record + discoveries).
- Persistent context also mirrored in Claude memory (`athar-rebuild-council` note).
- Legacy code access: `git show seif_final:<path>` / `git restore --source=seif_final -- <path>`.
- Conventional commits (`feat:`, `fix:`, `port:`, `docs:`, `test:`); parity-affecting ports must cite the v1 source path in the commit body.
- Never hand-edit resolved configs or registry DB — authoring formats only.
