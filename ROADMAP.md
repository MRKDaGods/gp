# ATHAR v2 — Rebuild Roadmap & Living Checklist

> **This file is the single source of truth** for the rebuild: decisions, phase
> checklist, parity gates, and open items. Update it in every working session —
> mark items done, add discoveries, never delete decision history.
>
> Last updated: **2026-07-22** · Current phase: **Phase 5 — Frontend (core COMPLETE: workspace/monitoring/provenance/report; timeline+map view open)** · clipsenet fusion in `production` profile DONE → next: DINOv2 (09s kernel port) or Phase 6 (gates P1–P4 settled)

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
- [x] **VeRi 93.3 REPRODUCED on current env** (v1 worktree, GTX 1050 Ti, 41.6 min): fused mAP **93.268** / R1 98.451 (@ w_clipsenet=0.7; Kaggle reference at same w: 93.214; manifest headline 93.32 @ best-w). Drift parents match: TransReID solo 90.014 (registry 89.97), CLIP-SENet solo 91.362. Baseline frozen: `tests/parity/baselines/veri776_fusion_v1env_20260721.json`
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
- [x] uv lockfile: 151 packages resolved clean (uv 0.8.2 — upgrade blocked by a locked uvx.exe; re-lock after upgrade)
- [x] **`.venv-v2` runtime env live** (2026-07-21): `UV_PROJECT_ENVIRONMENT=.venv-v2 uv sync` with ml/backend/eval/dev extras; old `.venv` untouched (v1 parity env). CPU torch by design (driver 572.47 < cu130's ≥580; local = smoke only). Full suite: **261 passed** under .venv-v2, 251 under v1 env. torchcodec works on Windows via PyAV-bundled FFmpeg DLLs (STACK.md conflict #7); boxmot 22 quirks documented (conflict #8)
- **RULE (2026-07-21, permanent)**: no co-author trailers on commits (history rewritten); GPU-intensive jobs run on **Kaggle, never locally** → Gate P2 CityFlow golden run must be a Kaggle kernel

### Phase 2 — Pipeline port (parity-gated)
- [x] Port `configs/model_registry.yaml` + `weights_manifest.yaml` + schema (verbatim) and `scripts/download_weights.py`
- [x] Port TransReID model **verbatim** (`athar/components/embedders/transreid_model.py`; sole change: loguru → stdlib logging) + checkpoint-contract tests (both frozen checkpoints load; only cls_head/jpm_cls drops; unit-norm 768-d output)
- [x] Ingest boundary v1: SHA-256 evidence hashing, PyAV/cv2 probe, TimeBase declaration, manifest population (`athar/pipeline/ingest.py`, 7 tests). Transcode-to-canonical + fisheye dewarp remain pluggable TODOs; FrameSource decode (torchcodec) next
- [x] FrameSource implementations (D11): torchcodec frame-exact random access (primary) + PyAV sequential + OpenCV sequential-only (seeking retired) + image-dir; original frame indices preserved through sampling; pts carried on batches (`pts_deviation_s` flags VFR); registry-wired (`video` auto-best). 23 tests across the full decoder matrix
- [x] Port stage1 kernels verbatim: `detector`, `tracker`, `tracklet_builder`, `bidirectional`, `bidirectional_merge`, `ssa` → `athar/components/tracking/` + 3 v1 test modules (12 tests)
- [x] **boxmot 22 revalidation** (2026-07-21): `TrackerWrapper` speaks both generations — `TRACKER_DEFINITIONS` class paths (≥20) + casing fallback (11-12); reid via `boxmot.reid.core.reid.ReID` from LOCAL weights only (public `ReIDModel` chain is broken upstream); `boosttrack` added; smoke-validated bytetrack/botsort(+osnet reid)/boosttrack. omegaconf fully decoupled from the ported tree
- [x] Detector/Tracker adapters (`athar/components/adapters/`): `yolo_v1` (one all-classes pass, COCO→EntityClass, scene-time Detections), `boxmot_v1` (stateful per camera, drain() → v2 Tracklets + observations)
- [x] **detect_track stage** (`athar/pipeline/stages/detect_track.py`): FrameSource → detector → per-branch trackers → `tracklets.<cam>` artifact per camera; branch id namespacing; camera-level checkpoint resume; 5 tests + **end-to-end smoke on real Shorouk CCTV green** (torchcodec + YOLO26m + botsort/osnet on boxmot 22 — first full v2 ingest→detect_track run)
- [x] Embedder adapters (`transreid_v1` — flip TTA + softmax-quality pooling, byte-faithful; `hsv_v1`) + **embed stage**: deterministic v1 candidate-frame plan → sparse decode (FrameSource explicit-indices plans) → verbatim `extract_crops_from_frames` → per-branch `embeddings.<cam>.<stream>` npz + `embed.summary`; camera-level resume; bounded memory. 4 tests + real-footage smoke (TransReID 768-d healthy cosine spread + HSV 192-d)
- [x] **index stage**: run-global tracklet catalog (ported MetadataStore, stable index_ids, HSV blobs) + FAISS exact-IP per appearance stream (ported FAISSIndex) with row sidecars; 3 tests + smoke search round-trip (self-match 0.98) — **the gallery half of the D5 flagship workflow runs end-to-end in v2**
- [x] **associate stage**: windowed engine seam (offline = one window; `window_s != 0` refused until live path) composing ported kernels — FAISS candidates → mutual-NN → SpatioTemporalValidator → class-adaptive combined similarity → GraphSolver; cross-camera + same-branch pairs only; Trajectory artifacts w/ per-term evidence (D6). 5 tests (identities recovered, class/time gating verified). **Full DAG ingest→…→associate green on real footage**
- [x] **Gallery search engine** (`athar/search/engine.py`): probe→gallery FAISS search w/ catalog join, compatibility guards (dim + projection lineage — the v1 global-PCA bug class), ranked APPEARANCE HypothesisEdges on Targets w/ attributed decide(). 7 tests. **D5 flagship loop closed end-to-end**
- [x] **CLI: `athar run` + `athar search`** — builtin `multiclass` profile (athar/profiles/builtin.py) or custom profile YAML; `--set` run overrides; `--resume RUN_ID`; console progress from the event stream. Validated from the CLI on real footage: c017 gallery + c018 probe + search — hits agree with associate's independent cross-camera merges (same identity groupings via both paths)
- [x] **CLIP-SENet embedder adapter** — DONE 2026-07-22 (`clipsenet_v1`, offline construction UNBLOCKED): equivalence check (`scripts/eval/check_clipsenet_offline_build.py`) proved pretrained=False + strict ckpt load is state-dict-BITWISE-identical to the canonical download build (forward drift ~6e-8 = kernel-order only; two same-mode runs are bitwise equal). Air-gap fix: vendored-IBN loader candidate added AFTER torch.hub (online selection unchanged; offline builds IBN-a from `athar/training/model.py` instead of falling to structure-mismatched plain resnet101). Discovery: the hf-hub TinyCLIP repo 404s upstream and open_clip 3.3 dropped the builtin config — timm `vit_medium_patch32_clip_224.tinyclip_laion400m` is the working provider on- AND offline. Adapter pools tracklets with the exact v1 softmax(quality x T) semantics. **Wired into the `production` builtin 2026-07-22** (see below)
- [x] **Weighted appearance-stream fusion + `production` profile — DONE 2026-07-22**: `associate.stream_weights` rescoring (weighted mean of per-stream cosines, renormalized over streams carrying BOTH tracklets; zero weight drops a stream; UNSET keeps max-merge bit-identical — parity gates untouched, D18). Builtin `production` = multiclass + `clipsenet_v1` on the vehicle branch @ 14t reference weighting (transreid_primary 1.0 / clipsenet 0.7)
- [ ] DINOv2 generalist stream (D9) — the v1 14e tertiary ckpt (`vehicle_transreid_dinov2_large_cityflowv2_final.pth`, on disk) needs its arch builder ported VERBATIM from the `yahiaakhalafallah/09s-dinov2-large-cityflowv2` kernel (not in the repo tree — do not improvise); alternatively the D4 raw-generalist DINOv2 needs vendored timm weights for air-gap
- [x] **package stage** (2026-07-21, thin v1): best-crop JPEG thumbnail per tracklet (highest-confidence observation, sparse decode) + `report_inputs.json` — the chain-of-custody skeleton (evidence sha256 -> config hash -> identities w/ per-member time spans + thumbnail paths; `clip` field reserved for evidence clips, which need the serving-phase transcode plumbing). Wired as the 5th CLI stage; camera-level resume. Validated on real Shorouk footage (21 thumbs, person crops verified visually)
- [x] **person-ReID embedder adapter** (2026-07-21): the Market1501 TransReID checkpoint (256x128, 6 cams, plain ImageNet ViT — `vit_base_patch16_224`, no CLIP normalization) wired through the same `transreid_v1` adapter as vehicles; person branch of `multiclass` now ships `transreid_person` (768-d) + HSV. Checkpoint-contract test added (tolerates the checkpoint's absent train-only `bn_jpm.*`). Associate stage generalized: FAISS candidates swept per appearance stream (each branch's stream indexes only its rows; single-stream runs bit-identical). Validated on real Shorouk pedestrians end to end
- [ ] Port remaining stage2 kernels: CLIP-SENet (fix HF-hub-first load order — air-gap bug found during smoke), DINOv2, HSV extractor, PCA whitener (pickled PCA = checkpoint)
- [ ] Port stage3: FAISS index + tracklet catalog (SQLite)
- [x] Port stage4 kernels **verbatim** (9 of 11): `similarity`, `reranking`, `query_expansion`, `graph_solver`, `camera_bias`, `fic`, `geospatial`, `spatial_temporal`, `zone_scoring` → `athar/components/associators/` + `hsv_extractor`/`pca_whitening` → embedders + `athar/core/constants.py`; 6 v1 test modules carried (101 tests green). Sole change everywhere: loguru → stdlib logging (verified no loguru-only APIs, all f-strings)
- [x] Port `aflink` + `occlusion` + `global_trajectories` + v1 `data_models`/`video_utils` (athar/core), `faiss_index`/`metadata_store` (athar/components/indexing), `format_converter`/`metrics`/`ground_plane_eval`/`evaluate_reid` (athar/evaluation), `clip_senet_model`/`reid_model`/`robust_pool`/`crop_extractor` (embedders), `training/model.py` (IBN archs) — 64 more v1 tests carried
- [x] Port VeRi eval trio (`scripts/eval/eval_{09v_transreid,clip_senet,14t_fusion}_veri776.py`) + `athar/serving/reid_loaders.py` byte-faithful (loader↔script cycle kept intact for parity; clean inversion = Phase 4 serving refactor) + `test_14t_fusion_math`
- [x] **Gate P1 test written**: `tests/parity/test_veri_fusion.py` runs the PORTED evaluator end-to-end (`ATHAR_RUN_PARITY=1 pytest -m parity`), asserts mAP 93.32 ± 0.2pt
- [ ] `test_multi_query` (needs stage2/stage3 pipeline glue)
- [x] DAG runner with two-level resume (`athar/pipeline/runner.py`): frozen-config guard, `is_complete` stage skip, atomic per-stage chunk checkpoints, CancellationToken, per-run `events.jsonl` + pluggable sinks, failures recorded on the manifest. 11 tests
- [ ] Port stage5: TrackEval integration + format converter
- [x] **GATE P1: VeRi-776 fusion mAP = 93.3 ± 0.2 — PASSED 2026-07-21** (ported tree, `ATHAR_RUN_PARITY=1 pytest -m parity`, 45:48)
- [x] **GATE P2: CityFlowV2 MTMC IDF1 = 0.779 ± 0.002 — PASSED 2026-07-21** (EXACT: mtmc_idf1 0.77936, id_switches 154 — matches the Kaggle 14v drift gate bit-for-bit; local CPU stages 3-5 in 33.7s). Baseline frozen: `tests/parity/baselines/cityflow_b1_local_20260721.json` (incl. golden sha256s + provenance)
  - [x] Harness: `tests/parity/test_cityflow_association.py` — v1 stages 3-5 on CPU against goldens, pinned to commit `24e85f31` (worktree `../gp-v1-b1`, auto-created; KEEP)
  - [x] Goldens: fetched from `yahiaakhalafallah/14c-tta-stage2` kernel output → `data/goldens/cityflow_b1_goldens/` (stage1 tracklets ×6 cams + TTA stage2 features). Kaggle auth for teammate accounts: `KAGGLE_API_TOKEN=$(cat ~/.kaggle/<name>_access_token)` — env vars KAGGLE_USERNAME/KEY and KAGGLE_CONFIG_DIR are IGNORED by CLI 2.0.1
  - [x] Golden-packaging kernel (`scripts/kaggle/p2_cityflow_goldens/`) + fetch script kept for regeneration
- [ ] IR/grayscale segment detection + dynamic stream reweighting (D15)

### Phase 3 — Multi-class & profiles
- [ ] `vehicle+person` joint profile: one detection pass (COCO 0,2,5,7), per-class branches
- [ ] MVDeTr as MultiViewDetector component; `person_multiview` profile
- [x] **GATE P3: WILDTRACK ground-plane IDF1 0.946 / MODA 0.903 — PASSED 2026-07-21** (EXACT bit-for-bit vs the Kaggle 14w verify reference: idf1 0.9456066945606695 / moda 0.9033613445378151 / IDSW 5; local CPU ~12s). Baseline frozen: `tests/parity/baselines/wildtrack_b1_local_20260721.json`
  - [x] Harness: `tests/parity/test_wildtrack_ground_plane.py` — v1 wildtrack single-shot route (cached 12a MVDeTr test.txt -> ground-plane Kalman -> ground-plane eval) pinned to commit `8c181472` (worktree `../gp-v1-wt`, auto-created; KEEP — seif_final later slimmed stage5, so the pin matters)
  - [x] Goldens: test.txt fetched from `yahiaakhalafallah/12a-resume-emit-wildtrack-test-txt` kernel output -> `data/goldens/wildtrack_b1_goldens/` (fetch script `scripts/kaggle/fetch_p3_goldens.py` verifies pinned sha256s)
  - Reference: kernel `gumfreddy/14w-verify-wildtrack-b1` validated the same recipe at the same SHA (drift -0.0014/+0.0004 vs 0.947/0.903 targets, match=true); MVDeTr detector parity itself was gated upstream by kernel 14z
- [x] **GATE P4 (new): generic person tracking baseline — ESTABLISHED 2026-07-21** (WILDTRACK-as-plain-video, Kaggle T4 kernel, v2 DAG end-to-end incl. image-dir ingest + person stream): SCT IDF1 0.324 / MTMC IDF1 0.192 vs the v1-recipe image-plane GT (frame alignment verified by shift probe). Baseline + protocol + improvement levers frozen: `tests/parity/baselines/wildtrack_person_p4_20260721.json`. The 3-5x gap to the P3 multiview chain (0.9456 on the same footage) is the quantified case for MVDeTr as a MultiViewDetector component. Levers: CLAHE plugin, tracker tuned for 2 fps (or track at native 60 fps), association tuning, CC-ReID/DINOv2 streams
- [ ] Spatial-model plugins: GPS (port `geospatial.py`), learned transition-time topology; floor-plan graph interface

### Phase 4 — Serving, jobs, API
- [x] Unified model loader + refcounted LRU cache with VRAM budget + DeviceManager — **DONE 2026-07-22**: `athar/serving/devices.py` DeviceManager (per-device byte budgets: CUDA total×0.9 headroom or `ATHAR_VRAM_BUDGET_MB`; explicit `DeviceBudgetError` instead of CUDA OOM) + ReIDRuntime leases (`acquire()` → ModelLease context manager; leased models never evicted; single build per (model,device) — waiters block instead of re-deserializing; failed builds unwind reservations)
  - [x] **2026-07-21 first slice**: loader<->script cycle inverted — CLIP-SENet arch moved verbatim to `athar/components/embedders/clip_senet_v6.py` (the copy that made 91.36/93.3; eval script re-imports, serving no longer imports scripts). `reid_loaders` module globals killed (per-call `_loader_params`); fixed `lru_cache(2)` replaced by `athar/serving/runtime.py` ReIDRuntime (configurable LRU honoring REID_MODEL_CACHE_SIZE, thread-safe, build-outside-lock, stats; old API delegates). `clip_senet_model.py` (HF-hub-first air-gap bug) still to retire with the stage2 glue port
- [x] Model lifecycle registry impl (SQLite) + promote/rollback + eval-gate enforcement — **DONE 2026-07-22**: `athar/serving/lifecycle.py` ModelLifecycleDB (WAL; append-only `lifecycle_events` trail; promote-to-production demotes+records the superseded model, rollback restores it; YAML authoring imports CANDIDATES only). CLI `athar models list/show/import/promote/retire/rollback/events`
- [x] JobService: SQLite queue, worker subprocess, typed event pipe, cancel/resume, executor-agnostic (local | kaggle) (D13) — **DONE 2026-07-22**: `athar/jobs/` queue (own DB, WAL, atomic `UPDATE...RETURNING` claims, priority+FIFO, per-executor routing; stale-heartbeat requeue keeps `run_id` so the next worker RESUMES) + worker loop (LocalRunExecutor over shared `athar/pipeline/setup.py`; event-sink cancel bridge to CancellationToken; KaggleExecutor = Phase 6 seam) + JobService facade. CLI `athar worker`, `athar jobs submit/list/show/cancel`. Event pipe = the run's `events.jsonl` (no second store)
- [x] FastAPI app: thin routers → services → RunRepository — **DONE 2026-07-22**: `athar/api/` factory + ApiSettings (`ATHAR_*` env), routers /auth /runs /jobs /models /search /audit /health, SSE tails of run events (validate-then-stream), artifact downloads w/ escape guard, `athar serve`; live HTTP smoke green. OpenAPI served at /openapi.json; generated TS client shipped with the Phase 5 frontend (hey-api, `web/openapi.json` snapshot)
- [x] AuthN/AuthZ + tamper-evident append-only audit log — **DONE 2026-07-22**: server-side sessions (raw token only in the httponly cookie, sha256 in DB) + pwdlib argon2id + ordered dependency-RBAC (viewer<investigator<admin) on SQLAlchemy 2/SQLite WAL; hash-chained audit log + `/audit/verify` (finds first broken record; tamper test proves it). Case-ownership scoping shipped with the Case API 2026-07-22 (owner-or-admin, 404-not-403)
- [x] Search engine: explicit 409 + calibrated score→probability — **DONE 2026-07-22**: `IncompatibleStreams` (probe missing stream / dim / projection lineage) → HTTP 409, unknown run → 404, other refusals → 400 — never a silent fallback. `athar/search/calibration.py`: per-stream logistic (Platt) fit w/ provenance + StreamCalibrations artifact; /search returns `probability=null` for uncalibrated streams (no invented confidence). **Fitting real calibrations on a labeled benchmark = Phase 6 task**

### Phase 5 — Frontend (enterprise-grade)
- [x] Next.js App Router scaffold, design system (shadcn/ui + Tailwind), dark/light, **Arabic/RTL i18n scaffolding from day 1**
  - [x] **2026-07-22 scaffold LIVE in `web/`** (v1 leftovers stay gitignored in `/frontend/`): Next.js 16 + Turbopack + Tailwind 4 + pnpm; next-intl with **Arabic default + RTL day 1** (html dir/lang per request; IBM Plex Sans Arabic; verified live: /ar/jobs renders RTL with a real job row); hey-api TS client generated from the committed `web/openapi.json` snapshot (`pnpm generate:api`; generated code lint-excluded); login (session cookie) + auth-guarded runs/jobs pages against the real API; CORS added to the API for dev origins (explicit origins — cookie auth). Gotcha recorded: open via localhost, not 127.0.0.1 (SameSite treats them as different sites). Node 22 local vs D19's Node 24 pin — revisit at hardening
  - [x] **2026-07-22 design system**: `shadcn init -b base --rtl` (nova preset) — 14 components on **Base UI**; next-themes dark/light (class strategy, system default, toggle in shell); Base UI `DirectionProvider` for RTL popup positioning; per-direction font indirection (`--font-sans` → Geist LTR / IBM Plex Sans Arabic RTL); status labels localized (`statuses` namespace). Verified live in both themes, RTL
- [x] Case workspace: galleries, probes, search, results review (confirm/reject hypotheses with audit) — **DONE 2026-07-22**: **Case API** (SQLAlchemy tables cases/case_runs/targets/target_members/hypotheses in the app DB; owner-or-admin need-to-know scoping — non-owners get 404, never 403; hypotheses may only cite runs attached to the case; confirm/reject is final (409 on re-decide) and every mutation is audit-chained) + **workspace UI** (/cases list+create dialog; /cases/[id]: attach/detach evidence runs, targets w/ confirmed members, search panel over the case's runs, one-click attach-hit-as-hypothesis, attributed confirm/reject). Verified end-to-end in the live Arabic UI
- [ ] Cross-camera timeline + map view (GPS cameras; Shorouk coords) + evidence clips
- [x] Job monitoring (live typed events), run detail with resolved-config + provenance viewer — **DONE 2026-07-22**: shared `EventStream` (per-type SSE listeners — frames are named; self-closes on terminal events so the browser never reconnects to a finished stream); per-job live monitor on /jobs; /runs/[id] with chain-of-custody inputs (sha256), artifacts + download links, frozen config table w/ per-key layer provenance (deployment/case/run_override visually flagged)
- [x] Report export (PDF, chain-of-custody chain: video SHA → config hash → model SHA → results) — **DONE 2026-07-22**: `athar/reporting/` — self-contained RTL/LTR HTML (thumbnails as data URIs; model SHAs attested from the pinned weights manifest, "unrecorded" when unknown — never invented) printed by **Playwright chromium** (D19; Arabic shaping correct). GET /runs/{id}/report.pdf + .html preview (audited), CLI `athar report`, web download button. Live: 71KB Arabic PDF over HTTP in 1.4s

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
| P1 | VeRi-776 two-stream fusion | mAP | 93.3 ± 0.2 | ✅ **PASSED 2026-07-21** — ported v2 tree, full run (45:48, GTX 1050 Ti); v1-env baseline 93.268 same day |
| P2 | CityFlowV2 MTMC | IDF1 | 0.779 ± 0.002 | ✅ **PASSED 2026-07-21** — EXACT 0.77936 / id_sw 154 (local CPU stages 3-5, 33.7s; matches Kaggle 14v gate) |
| P3 | WILDTRACK (MVDeTr) | IDF1 / MODA | 0.946 / 0.903 | ✅ **PASSED 2026-07-21** — EXACT 0.94561 / 0.90336, IDSW 5 (local CPU ~12s; bit-for-bit vs Kaggle 14w) |
| P4 | Generic person tracking | IDF1 | establish baseline | ✅ **BASELINE 2026-07-21** — SCT 0.324 / MTMC 0.192 (WILDTRACK-as-plain-video, Kaggle T4; improvement target, see baseline JSON) |
| P5 | ATHAR-Bench v0 (Shorouk) | IDF1 + retrieval | establish baseline | ☐ needs annotation |

## 5. Port map (from council salvage audit)

- **Verbatim**: `stage4_association/{similarity,reranking,query_expansion,graph_solver}.py` + helpers; `stage2_features/{transreid_model,hsv_extractor,pca_whitening}.py`; `stage5_evaluation/{metrics,format_converter}.py`; `scripts/eval/eval_14t_fusion_veri776.py`; `configs/{model_registry,weights_manifest}.yaml`; v1 `tests/test_stage4/`, TransReID key-contract tests.
- **Refactor**: `serving/reid_loaders.py` (kill module globals + fixed lru_cache); stage helper configs → typed schema; MVDeTr txt-parsing + ground-plane tracking (extract functions first).
- **Rewrite**: all `pipeline.py` orchestrators; backend entirely; run identity/state; config layering; frontend.
- **Drop**: Kaggle-dispatch-as-default, Streamlit apps, dead config flags (v25–v49), `backend_api.py` shim.

## 6. Open items needing user input

- [x] ~~Gate P2 Kaggle access~~ — RESOLVED 2026-07-21: all four account tokens live in `~/.kaggle/` (`<name>_access_token` files); use `KAGGLE_API_TOKEN=$(cat ...)` to switch identity (CLI 2.0.1 ignores KAGGLE_USERNAME/KEY and KAGGLE_CONFIG_DIR)
- [ ] Remote branch pruning approval (`origin/backedn`, stale feature/fix/verify branches) — NOTE: `verify/14v-kaggle-b1` must be KEPT (Gate P2 pins commit `24e85f31` from it)
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
