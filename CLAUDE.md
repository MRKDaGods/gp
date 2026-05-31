# CLAUDE.md — MTMC Tracker

Guidance for Claude Code working in this repository. Distilled from `.github/copilot-instructions.md`, `.github/agents/orchestrator.agent.md`, and the `docs/` tree (read on demand — see the table at the end).

## Project Overview
Multi-camera multi-target tracking (MTMC) of **vehicles** (CityFlowV2 / AI City Challenge 2022 Track 1) and **people** (WILDTRACK). A 7-stage offline pipeline: Ingestion → Tracking → Features → Indexing → Association → Evaluation → Visualization. Python 3.10–3.13, PyTorch, YOLO26m, TransReID ViT-B/16 CLIP, FAISS, SQLite. A FastAPI backend (`backend/`) and Next.js "ATHAR" dashboard (`frontend/`) wrap the pipeline. There is also an active **academic-paper writing effort** (two LaTeX documents) — see [Paper Work](#paper-work).

The project is **converged and in the paper-writing phase** (as of 2026-05-31). The core research result is locked; most work now is analysis, ablation documentation, and writing — not chasing new accuracy.

---

## Critical Rules (NEVER violate)

### 1. GPU work runs on Kaggle, never locally
- **NEVER** run GPU-intensive stages (0 ingestion, 1 detection/tracking, 2 feature extraction) or any ReID training locally — the local GPU is a GTX 1050 Ti and starves other work.
- ALL GPU work runs on **Kaggle** (T4/P100). Local machine is for: code editing, generating/pushing notebooks, monitoring kernels, and CPU-only stages (3 indexing, 4 association, 5 evaluation) on small data.
- Max **2 concurrent GPU kernels per Kaggle account** (accounts: `gumfreddy`, `mrkdagods`, `ali369`, `yahiaakhalafallah`). Check before pushing.

### 2. Always use `.venv` for local Python
- Local Python MUST use `.venv` (Python 3.11.9), NOT system Python 3.13. Activate: `.\.venv\Scripts\activate`. `start.py` auto-selects `.venv\Scripts\python.exe` if present.

### 3. Notebook (`.ipynb`) editing
- **NEVER** edit `.ipynb` via raw text find/replace — it may not round-trip to disk. Edit via `json.load() → modify → json.dump()` in a one-off Python script or the NotebookEdit tool.
- After any edit, verify on-disk with `python -c "import json; json.load(open(...))"`.
- Each line in a `source` array MUST end with `\n` EXCEPT the last line.
- On Windows use `ensure_ascii=True` in `json.dump` to avoid charmap codec errors.
- Do NOT commit `.ipynb` after editing in the Kaggle web UI — edit locally and push fresh.

### 4. Frame-ID convention
- Internal pipeline (Stages 0–4): **0-based**, matches YOLO output directly.
- MOTChallenge submission (Stage 5): **1-based** (converted in `src/stage5_evaluation/format_converter.py`).
- CityFlowV2 GT: **1-based** (standard MOT). Bboxes are `(x1,y1,x2,y2)` pixels internally; MOT export uses `(x,y,w,h)`.
- Never mix — check which context you're in.

### 5. Config override paths (OmegaConf)
- Loading order (lowest → highest priority): `configs/default.yaml` → `--dataset-config` (e.g. `configs/datasets/cityflowv2.yaml`) → `-o` CLI dotlist overrides.
- **Stage 4 lives under `cfg.stage4.association`**, deeply nested. Overrides MUST use the full path, e.g.:
  - `stage4.association.graph.similarity_threshold=X`
  - `stage4.association.fic.regularisation=X`
  - `stage4.association.gallery_expansion.threshold=X`
  - `stage4.association.query_expansion.k=X`
- **Don't guess override paths** — trace from `cfg.stage4.association.<...>` in `src/stage4_association/pipeline.py`. The public `solver` toggle (`cc` | `network_flow`) resolves to `graph.algorithm` internally.

### 6. Read the research docs BEFORE any experiment
- Read [docs/findings.md](docs/findings.md) and [docs/dead-ends.md](docs/dead-ends.md) **before proposing any experiment or parameter change.** Association tuning is **saturated** (225+ configs within ~0.3pp of the best) — the bottleneck is feature quality, not the association algorithm.
- Update [docs/findings.md](docs/findings.md) / [docs/what-worked.md](docs/what-worked.md) / [docs/dead-ends.md](docs/dead-ends.md) whenever experiments produce results or insights.

---

## How to Run

### Full system (backend + frontend)
```pwsh
python start.py        # backend on :8000, frontend on :3001 (kills stale ports, auto-uses .venv)
```
Or manually:
```pwsh
# Backend — launch backend_api:app (root backend_api.py re-exports backend.app:app
# AND sets the Windows proactor event-loop policy; this is the canonical entry).
python -m uvicorn backend_api:app --host 0.0.0.0 --port 8000
# Frontend (Next.js 14, App Router) — from frontend/
npm run dev            # http://127.0.0.1:3001
```
**Ports:** backend **:8000**, frontend **:3001**. The frontend reaches the backend on `:8000` via `frontend/.env.local` (`NEXT_PUBLIC_API_URL`, `NEXT_PUBLIC_WS_URL`), the `/api` rewrite + image host in `frontend/next.config.mjs`, and the `api.ts` fallback — all consistent with the backend port. Backend CORS allows `localhost`/`127.0.0.1` on ports **3000 and 3001** (the frontend runs on 3001). If you move the backend off 8000, update those four frontend references too (not just `NEXT_PUBLIC_API_URL`).

### Pipeline (CLI, `click`-based)
```pwsh
python scripts/run_pipeline.py -c configs/default.yaml                     # full pipeline (stages 0-6)
python scripts/run_pipeline.py -c configs/default.yaml -d configs/datasets/wildtrack.yaml
python scripts/run_pipeline.py -c configs/default.yaml -s 3,4,5            # subset of stages
python scripts/run_pipeline.py -c configs/default.yaml --smoke-test        # first 10 frames/camera
python scripts/run_pipeline.py -c configs/default.yaml --dry-run           # print resolved config, no run
python scripts/run_pipeline.py -c configs/default.yaml -o stage1.detector.device=cpu -o stage2.reid.half=false
```
Flags: `--config/-c` (required), `--dataset-config/-d`, `--stages/-s`, `--smoke-test`, `--dry-run`, `--override/-o` (repeatable dotlist).

### Tests
```pwsh
pytest tests/ -v
make test            # or make test-cov / make lint
```

---

## Architecture

```
configs/          OmegaConf YAML: default.yaml + datasets/ + experiments/ + models/ + model_registry.yaml
backend/          FastAPI: app.py (entry, backend.app:app), routers/, services/, repositories/, config.py
frontend/         Next.js 14 ATHAR dashboard (App Router, Radix UI, Tailwind, Zustand, React Query, Leaflet)
src/core/         data_models.py (schemas), config.py (loader), io_utils.py, constants.py
src/stage0_ingestion/      Frame extraction, preprocessing (CLAHE, resize)         → run_stage0
src/stage1_tracking/       YOLO26m detection + BoxMOT tracking; SSA, bidirectional → run_stage1
src/stage2_features/       ReID embeddings (TransReID ViT) + HSV + PCA whitening   → run_stage2
src/stage3_indexing/       FAISS IndexFlatIP + SQLite metadata                     → run_stage3
src/stage4_association/    Cross-camera association (similarity graph + CC solver) → run_stage4
src/stage5_evaluation/     TrackEval metrics (HOTA, IDF1, MOTA)                    → run_stage5
src/stage6_visualization/  Annotated video, BEV, timeline exports                 → run_stage6
src/stage_wildtrack_mvdetr/  MVDeTr ground-plane fast-path: replaces stages 1-4 for WILDTRACK
src/apps/         Streamlit dashboard, NL query, 3D sim
src/serving/, src/training/
scripts/          CLI entry points + Kaggle orchestration + analysis helpers
notebooks/kaggle/ Kaggle notebooks (10a/10b/10c chain, 09x/11x/12x/13x/14x, paper ablations)
tests/            pytest suite (tests/test_stageN/...)
docs/             Findings, dead-ends, performance state, paper drafts, etc. (see table below)
gp__Copy_/        MTMC thesis (LaTeX, chapters/). main.tex (root) = separate VeRi-776 ReID paper.
```

### Code patterns
- Each stage is `src/stageN_name/pipeline.py` exposing `run_stageN(cfg, ..., output_dir, smoke_test=False)`.
- Config sections mirror stage names (`stage0`…`stage4.association`). All config via OmegaConf — no env vars/resolvers; explicit `-o` overrides.
- Stages communicate through files under `outputs/<run_id>/stageN/...`. Backend reads run-scoped artifacts there.
- Tests: `tests/test_stageN/test_component.py`.

### Data schemas (`src/core/data_models.py`)
- **Tracklet** (stage 1): `track_id:int` (unique within camera), `camera_id:str`, `class_id:int` (COCO: 0=person,2=car,5=bus,7=truck), `class_name:str`, `frames:List[TrackletFrame]`.
- **TrackletFrame**: `frame_id:int` (0-based), `timestamp:float` (s), `bbox:(x1,y1,x2,y2)` px, `confidence:float`.
- **TrackletFeatures** (stage 2): `embedding:np.ndarray` (L2-normalized, **384-D after PCA**), `hsv_histogram` (L2-normalized), `raw_embedding` (768-D pre-PCA, optional), `multi_query_embeddings` (K×D, optional), + ids.
- **GlobalTrajectory** (stage 4): `global_id:int`, `tracklets:List[Tracklet]` (one per camera), `confidence:float`, `evidence:List[Dict]` (forensic merge audit trail), `timeline:List[Dict]`.
- **EvaluationResult** (stage 5): `mota, idf1, mtmc_idf1, hota, id_switches, mostly_tracked, mostly_lost, num_gt_ids, num_pred_ids, details`.

### Output file layout (`src/core/constants.py`)
```
outputs/<run_id>/
  config.yaml                         merged OmegaConf
  stage0/<camera>/frame_000123.jpg    extracted frames (png if stage0.lossless)
  stage1/tracklets_<camera>.json
  stage2/embeddings.npy (N,384)  +  embedding_index.json  +  hsv_features.npy
  stage3/faiss_index.bin  +  metadata.db
  stage4/global_trajectories.json     final cross-camera output
  stage5/evaluation_report.json
```

---

## Current Performance & Bottleneck

| Pipeline | Metric | Best (reproducible) | SOTA | Gap | Status |
|----------|--------|---------------------|------|-----|--------|
| Vehicle (CityFlowV2) | MTMC IDF1 | **0.77936** (14e B1) | 0.8486 (AIC22, 5-model) | 6.93pp | Feature-quality-limited; association tuning saturated |
| Person (WILDTRACK) | Ground-plane IDF1 | **0.947** (12b) | ~0.953 | 0.6pp | Tracker-limited (Kalman); fully converged |

- **Vehicle headline config** (14e B1): TTA Stage-2 features + Stage-4 fusion `similarity_threshold=0.48, aqe_k=2, fic_regularisation=0.5, w_tertiary=0.525`. Primary model: TransReID ViT-B/16 CLIP 256px (mAP 81.53 / R1 92.41). Tertiary: DINOv2 ViT-L/14. (`configs/default.yaml` ships its own v30 defaults, e.g. `similarity_threshold=0.60`, `pca.n_components=384` — the headline 14e B1 numbers come from a tuned sweep; trust [docs/performance-state.md](docs/performance-state.md) for the authoritative current config.)
- **Person**: MVDeTr ResNet18 detector (MODA≈0.92) + Kalman tracker.
- **Key paradox (do not forget):** stronger ReID mAP does NOT imply better MTMC IDF1. DINOv2 beats CLIP by +6.65pp mAP yet loses ~3pp MTMC IDF1 because it learns viewpoint-specific texture instead of cross-camera invariance. ReID benchmark mAP is a misleading proxy for cross-camera tracking.

### Locked tunings & closed levers (don't re-litigate — see findings.md)
- **AQE `k=2` is a discrete optimum on TTA features** — the 0.7703→0.77936 jump came from lowering AQE k 3→2, not from the TTA views themselves. k=1 is −0.88 to −1.00pp; k=3 regresses to 0.77149. Don't re-sweep.
- **The DINOv2 tertiary is contribution-saturated** — ID-switch count stays pinned at **154** across all view-count/pooling experiments. Further tertiary tweaks won't move IDF1; a genuinely different third stream or a learned edge classifier (GNN) would be required.
- **Cross-domain fusion always fails** — even a 91.54%-mAP VeRi-776 CLIP-SENet secondary cannot help CityFlowV2; the domain gap dominates. Don't try to borrow VeRi-776-trained models as a CityFlowV2 stream.
- **The gap to AIC22 is feature-quality, not association machinery** — confirmed by the CityTrack 7-component audit. Bidirectional tracking showed +5.10pp on the 3-camera S02 subset but tripled ID switches (43→149) and is unconfirmed on the full set, so it stays default-off. The only unambiguous lever is feature diversity (single vs 5-model ensemble).

---

## Dead Ends — DO NOT Repeat (read [docs/dead-ends.md](docs/dead-ends.md) for the full ledger)
1. **CSLS hubness reduction** — catastrophic (≈−34pp); penalizes genuine vehicle-type hubs.
2. **AFLink motion linking** — −4 to −13pp; motion is unreliable across non-overlapping cameras.
3. **CID_BIAS camera-distance bias** — −1 to −3pp; distorts FIC-calibrated similarities.
4. **384px ViT input** — −2.8pp; high-res overfits viewpoint texture.
5. **Hierarchical clustering** — −1 to −5pp (centroid averaging erases discriminative signal).
6. **SAM2 box-prompt masking** — −0.56pp; removes wheel/reflection cues TransReID uses.
7. **Feature concatenation / naive score fusion of a weak secondary** — adds noise; breaks calibration.
8. **AQE k=1 on TTA features** — re-introduces noise; k=2 is the discrete optimum.
9. **Network-flow solver over conflict_free_cc** — −0.24pp + more conflation.
10. **Robust tracklet pooling modes (median/medoid, 8 variants)** — all hurt vs softmax-quality-weighted mean.
- Also: don't add `mtmc_only=True` (drops single-cam tracks, −5pp IDF1); don't enable track smoothing/edge trim; don't compare ResNet101-IBN-a mAP to VeRi-776 baselines (our eval is CityFlowV2, different dataset/IDs).

---

## Kaggle Workflow
Read [docs/kaggle-workflow.md](docs/kaggle-workflow.md) before any `kaggle kernels push`.
- **Accounts & tokens (4):** `gumfreddy`, `mrkdagods`, `ali369`, `yahiaakhalafallah`. Each has a per-account token file in `~/.kaggle/` (NOT in the repo): `gumfreddy_access_token`, `mrkdagods_access_token` / `MRKDaGods__access_token`, `ali369_access_token` / `ali_369_access_token`, `yahiaakhalafallah_access_token`. The canonical account→token map is `scripts/dump_kaggle_kernel_summaries.py:24` (`ACCOUNT_TOKENS`). The active identity is whatever is in `~/.kaggle/kaggle.json`; tooling **hot-swaps** it by copying the chosen token file over `kaggle.json` before each CLI call (`copy_token_to_kaggle_json`). Default active account is `gumfreddy`. Never commit any token/key.
- **Why 4 accounts:** Kaggle allows **max 2 concurrent GPU sessions per account** — the multiple accounts exist to parallelize GPU work across slots. Weights are mirrored per-account (`<account>/mtmc-weights`); the 10a/10b/10c vehicle chain has run under `mrkdagods` and `yahiaakhalafallah`.
- **Notebook naming:** `10a` (stages 0–2, GPU) → `10b` (stage 3) → `10c` (stages 4–5) is the vehicle production chain. `11x`=WILDTRACK, `12x`=MVDeTr training, `13x`=CLIP-SENet training, `14x`=TTA/fusion/ablation sweeps, `09x`=ReID training, `paper_veri776_*` and `citytrack-s02-*`=paper ablations.
- **Push:** `kaggle kernels push -p notebooks/kaggle/10a_stages012/`. **Status:** `kaggle kernels status <owner/slug>`. **Logs:** `python scripts/kaggle_logs.py <owner/slug> --tail N`.
- **Orchestration scripts:** `scripts/kaggle_chain.py` (10a→10b→10c single pass), `scripts/kaggle_autopush.py` (multi-cycle auto-push with status polling), `scripts/dump_kaggle_kernel_summaries.py` (pull small artifacts → `docs/_data/kaggle_kernel_summaries.json`).
- **Safety:** NEVER push the same kernel twice before the prior version COMPLETES (creates duplicate GPU sessions). If push warns `not valid dataset sources`, cancel immediately. Disk hygiene: after `kaggle kernels output`, delete `last.pth` / failed `.pth` / 0-byte logs; keep `best_*.pth`, `eval_results.json`, `recipe.json`, summaries.
- `kernel-metadata.json` per notebook holds slug, `enable_gpu`, `machine_shape`, `dataset_sources`, `kernel_sources` — update sources before push if deps change.

---

## Paper Work
There are **two separate LaTeX documents**:
1. **VeRi-776 ReID paper** — root `main.tex` + `references.bib`. Targets IEEE venue. Headline: two-stream fusion **93.30% mAP / 98.45% R1** on VeRi-776 (locked). Stream 1 = TransReID ViT-B/16 CLIP; Stream 2 = CLIP-SENet ResNet101-IBN. Includes isolated-component ablation (A1–A5β) and seed-variance analysis. **Tie-band rule:** A5α (deployed) vs A5β (paper-described) differ ≤0.05pp mAP → within tie band → A5α retained; no Wave 4.
2. **MTMC thesis** — `gp__Copy_/main.tex` + `gp__Copy_/chapters/` (ch1 intro … ch6 conclusions + appendices). Documents the 7-stage system and the "single model reaches 91% of 5-model SOTA" story. `ch5_testing.tex` has the **CityTrack audit** section comparing to the AIC22 winner (Team28, 0.8486 IDF1): 7-component audit, 3 missing components re-implemented behind **default-off flags** (SSA `src/stage1_tracking/ssa.py`, bidirectional `src/stage1_tracking/bidirectional.py`, occlusion-aware `src/stage4_association/occlusion.py`), + S02 3-camera subset ablation. Conclusion: gap is feature-quality, not association machinery.
- **`veri776`** = the VeRi-776 ReID paper/ablation campaign. **`citytrack`** = the AIC22-winner completeness audit for the thesis. Spec docs live in `docs/subagent-specs/` (`veri776-wave3-paper-sync.md`, `citytrack-paper-section.md`). Collector: `scripts/collect_citytrack_ablation.py`. Aggregator: `scripts/paper/aggregate_paper_results.py`.
- LaTeX build artifacts (`gp__Copy_/main.pdf`, `.aux`, `.fls`, etc.) are generated — don't hand-edit; they're not the source.

---

## Multi-Agent Delegation
For complex/multi-stage/research work, delegate via the **Agent tool** (this mirrors `.github/agents/orchestrator.agent.md`, where the orchestrator delegates to a Planner + Coder).
- **`Explore`** — read-only codebase lookups, broad fan-out searches, Q&A. Keep the conclusion, not the file dumps.
- **`Plan`** — research, strategy, experiment design, implementation planning. Have it write a spec to `docs/subagent-specs/<task-name>.md` (problem analysis with file:line, exact changes, `stage4.association.X` overrides, expected IDF1/MOTA impact, rollback) before implementing.
- **`general-purpose`** — multi-step implementation, complex searches.
- Single obvious edit → just do it inline. Cross-stage change / unclear approach → plan first.
- **Experiment/optimization loops:** goal = maximize cross-camera IDF1 via `python scripts/run_pipeline.py -c configs/default.yaml`; in-scope `src/stage4_association/`, `src/stage2_features/`, `configs/`; out-of-scope `notebooks/`, `tests/`, `data/`. Always check [docs/dead-ends.md](docs/dead-ends.md) first.

---

## Reference Docs (load on demand)

| File | When to load |
|------|--------------|
| [docs/findings.md](docs/findings.md) | Before any experiment — strategic narrative & the feature-quality bottleneck |
| [docs/dead-ends.md](docs/dead-ends.md) | Before any parameter change — full sweep history of what failed |
| [docs/what-worked.md](docs/what-worked.md) | Quick reference for positive deltas |
| [docs/performance-state.md](docs/performance-state.md) | Authoritative current metrics, checkpoints, headline config |
| [docs/experiment-log.md](docs/experiment-log.md) | Full 225+ experiment ledger |
| [docs/architecture.md](docs/architecture.md), [docs/data_flow.md](docs/data_flow.md), [docs/data_contracts.md](docs/data_contracts.md) | Stage internals, association math, exact schemas |
| [docs/pipeline-vehicle.md](docs/pipeline-vehicle.md) / [docs/pipeline-person.md](docs/pipeline-person.md) | Per-pipeline deep dive (vehicle / WILDTRACK) |
| [docs/kaggle-workflow.md](docs/kaggle-workflow.md) | Before any `kaggle kernels push` — push safety, disk hygiene, lifecycle |
| [docs/models.md](docs/models.md) / [docs/models.generated.md](docs/models.generated.md) / [docs/model-cards.md](docs/model-cards.md) | Checkpoint provenance & verified metrics |
| [docs/integration-status.md](docs/integration-status.md) | Backend/registry integration state, PR list, canonical config values |
| [docs/paper-strategy.md](docs/paper-strategy.md) / [docs/paper-draft.md](docs/paper-draft.md) | Paper writing / venue decisions |
| [docs/SOTA_ANALYSIS.md](docs/SOTA_ANALYSIS.md) / [docs/BREAKTHROUGH_PLAN.md](docs/BREAKTHROUGH_PLAN.md) | Prior ROI estimates before proposing a new push |
| [docs/mvdetr-integration.md](docs/mvdetr-integration.md) | WILDTRACK MVDeTr fast-path details |

---

## What NOT to Do (quick list)
- Don't run GPU stages (0/1/2) or ReID training locally — Kaggle only.
- Don't use system Python — always `.venv`.
- Don't text-edit `.ipynb` or raw JSON Unicode — breaks structure; use json load/dump.
- Don't guess config override paths — trace from `cfg.stage4.association` in code.
- Don't repeat dead-end experiments — check [docs/dead-ends.md](docs/dead-ends.md) first.
- Don't add `mtmc_only=True`, track smoothing, or edge trim.
- Don't assume higher ReID mAP ⇒ better MTMC IDF1 (it doesn't — see the paradox above).
- Don't hand-edit generated LaTeX/PDF artifacts under `gp__Copy_/`.
- Don't push a Kaggle kernel twice before the prior run completes.
