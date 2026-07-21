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
- [ ] CLIP-SENet embedder adapter (needs Phase 4 loader refactor first — the working arch copy lives in the eval script) + DINOv2 generalist stream (D9)
- [ ] package stage (evidence clips/thumbnails, report inputs) — last DAG slot, thin for now
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
- [ ] **GATE P2: CityFlowV2 MTMC IDF1 = 0.779 ± 0.002** — infrastructure READY, blocked on Kaggle access:
  - [x] Harness: `tests/parity/test_cityflow_association.py` — runs v1 stages 3-5 locally on CPU (no GPU rule violation) against goldens, pinned to commit `24e85f31` (the SHA the public `14v-verify-b1-from-yaml` kernel drift-gated at 0.77936/154 on Kaggle); auto-creates worktree `../gp-v1-b1`
  - [x] Golden-packaging kernel: `scripts/kaggle/p2_cityflow_goldens/` (CPU kernel, adapted from 14v; drift-gates then tarballs stage1 tracklets + stage2 TTA features + trajectories + eval report + sha256 provenance)
  - [x] Fetch script: `scripts/kaggle/fetch_p2_goldens.py` (downloads + sha256-verifies into `data/goldens/`)
  - [ ] **BLOCKED (user)**: goldens live in PRIVATE kernel outputs under the `yahiaakhalafallah` account (`14c-tta-stage2`, `mtmc-10a-stages-0-2`); local token is `mrkdagods`. Any one of: (a) drop yahia's kaggle.json in `~/.kaggle/`, (b) make those two kernels public, (c) push+run `scripts/kaggle/p2_cityflow_goldens/` from that account. Then: fetch → `ATHAR_RUN_PARITY=1 pytest -m parity tests/parity/test_cityflow_association.py`
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
| P1 | VeRi-776 two-stream fusion | mAP | 93.3 ± 0.2 | ✅ **PASSED 2026-07-21** — ported v2 tree, full run (45:48, GTX 1050 Ti); v1-env baseline 93.268 same day |
| P2 | CityFlowV2 MTMC | IDF1 | 0.779 ± 0.002 | ⏸ harness+kernel+fetch READY; blocked on Kaggle access to yahia's private kernel outputs (see Phase 2) |
| P3 | WILDTRACK (MVDeTr) | IDF1 / MODA | 0.946 / 0.903 | ☐ not run |
| P4 | Generic person tracking | IDF1 | establish baseline | ☐ no number exists |
| P5 | ATHAR-Bench v0 (Shorouk) | IDF1 + retrieval | establish baseline | ☐ needs annotation |

## 5. Port map (from council salvage audit)

- **Verbatim**: `stage4_association/{similarity,reranking,query_expansion,graph_solver}.py` + helpers; `stage2_features/{transreid_model,hsv_extractor,pca_whitening}.py`; `stage5_evaluation/{metrics,format_converter}.py`; `scripts/eval/eval_14t_fusion_veri776.py`; `configs/{model_registry,weights_manifest}.yaml`; v1 `tests/test_stage4/`, TransReID key-contract tests.
- **Refactor**: `serving/reid_loaders.py` (kill module globals + fixed lru_cache); stage helper configs → typed schema; MVDeTr txt-parsing + ground-plane tracking (extract functions first).
- **Rewrite**: all `pipeline.py` orchestrators; backend entirely; run identity/state; config layering; frontend.
- **Drop**: Kaggle-dispatch-as-default, Streamlit apps, dead config flags (v25–v49), `backend_api.py` shim.

## 6. Open items needing user input

- [ ] **Gate P2 Kaggle access** (one-minute fix, unblocks the parity gate): the goldens are in private kernel outputs under `yahiaakhalafallah` (`14c-tta-stage2`, `mtmc-10a-stages-0-2`); local token is `mrkdagods`. Either drop yahia's `kaggle.json` into `~/.kaggle/`, make those two kernels public, or run `scripts/kaggle/p2_cityflow_goldens/` from that account — then `python scripts/kaggle/fetch_p2_goldens.py` + `ATHAR_RUN_PARITY=1 pytest -m parity tests/parity/test_cityflow_association.py`
- [ ] Remote branch pruning approval (`origin/backedn`, stale feature/fix/verify branches) — NOTE: `verify/14v-kaggle-b1` must be KEPT (Gate P2 kernel clones it)
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
