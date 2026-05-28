# GitHub Copilot Instructions — MTMC Tracker

## Project Overview
Multi-camera multi-target tracking (vehicles/humans) on CityFlowV2 (AI City Challenge 2022 Track 1). 7-stage offline pipeline: Ingestion → Tracking → Features → Indexing → Association → Evaluation → Visualization. Python 3.10+, PyTorch, YOLO26m, TransReID ViT-Base/16 CLIP, FAISS, SQLite, Streamlit. Backend (FastAPI) under `backend/`, frontend (Next.js ATHAR) under `frontend/`.

---

## Critical Rules (NEVER violate)

### GPU Pipeline Execution
- **NEVER** run GPU-intensive pipeline stages (0, 1, 2) locally — local GPU is a GTX 1050 Ti and starves other work.
- ALL GPU work (detection, tracking, feature extraction, ReID training) MUST run on Kaggle.
- Local machine is ONLY for: code editing, pushing notebooks, monitoring Kaggle kernels, and CPU-only stages (3, 4, 5) on small datasets.
- Any local Python MUST use `.venv` (Python 3.11.9), NOT system Python 3.13. Activate with `.\.venv\Scripts\activate`.

### Notebook Editing (`.ipynb` files)
- **NEVER** use `replace_string_in_file` on `.ipynb` — it edits VS Code's in-memory buffer but may NOT save to disk.
- Edit via `json.load() → modify → json.dump()` in a Python script.
- After any edit, verify on-disk with `python -c "import json; ..."`.
- Each line in `source` arrays MUST end with `\n` EXCEPT the last line.
- On Windows, use `ensure_ascii=True` in `json.dump` to avoid charmap codec errors.

### Frame ID Convention
- Internal pipeline (Stages 0–4): **0-based**.
- MOT submission format: **1-based** (converted via `frame_id + 1` in `format_converter`).
- CityFlowV2 GT: **1-based** (standard MOT format).
- Never mix — check which context you're in.

### Config Override Paths (OmegaConf)
- Stage4 reads from `cfg.stage4.association`. Overrides MUST be `stage4.association.graph.similarity_threshold=X`, NOT `stage4.graph.X` or `stage4.X`.
- Other examples: `stage4.association.fic.regularisation=X`, `stage4.association.gallery_expansion.threshold=X`.
- Loading order: `default.yaml` → merge `cityflowv2.yaml` → CLI overrides.

### Research Findings — MUST READ Before Any Experiment
- Read [docs/findings.md](../docs/findings.md) and [docs/dead-ends.md](../docs/dead-ends.md) **before proposing any experiment or parameter change.**
- Update [docs/findings.md](../docs/findings.md) whenever experiments produce results, new dead ends are found, performance numbers change, or insights are gained.

---

## Architecture

```
configs/          YAML configuration (OmegaConf)
backend/          FastAPI service layer and API routers
frontend/         Next.js ATHAR dashboard and workflow UI
src/core/         Shared data models, config loader, utilities
src/stage0/       Frame extraction, preprocessing
src/stage1/       YOLO26m detection + BoxMOT tracking (BoT-SORT)
src/stage2/       ReID embeddings (TransReID ViT) + HSV + PCA whitening
src/stage3/       FAISS IndexFlatIP + SQLite metadata
src/stage4/       Cross-camera association (similarity graph + connected components)
src/stage5/       TrackEval metrics (HOTA, IDF1, MOTA)
src/stage6/       Visualization (annotated video, BEV, timeline)
src/apps/         Streamlit dashboard, NL query, 3D sim
scripts/          CLI entry points + helper scripts
notebooks/kaggle/ Kaggle training notebooks (10a/10b/10c pipeline chain)
tests/            pytest test suite
docs/             Findings, experiment log, performance state, dead ends, etc.
```

### Code Patterns
- Stages: `src/stageN_name/` with `pipeline.py` as entry point.
- Config sections mirror stage names (`stage0`, `stage1`, …, `stage4.association`).
- Tests: `tests/test_stageN/test_component.py`.
- All config via OmegaConf — no env vars or resolvers; all explicit `--override` args.
- Stages communicate through files in `data/outputs/`. Backend services read run-scoped artifacts under `data/outputs/<run_id>/...`. Tracklets are dicts of `camera_id`, `track_id`, `frames`, `boxes`, `embeddings`.

### Key Dependencies
Python ≥3.10, <3.14 · PyTorch ≥2.1 · timm ≥0.9 · ultralytics ≥8.1 · boxmot ≥10.0 · faiss-cpu ≥1.7 · omegaconf ≥2.3 · networkx ≥3.1 · streamlit ≥1.28. Kaggle P100 GPUs need PyTorch 2.4.1+cu124 (sm_60 compat).

---

## Current Performance (headline only)

| Pipeline | Metric | Best | SOTA | Gap | Status |
|----------|--------|------|------|-----|--------|
| Vehicle (CityFlowV2) | MTMC IDF1 | **0.77936** (14e B1) | 0.8486 (AIC22) | 6.93pp | Feature-quality-limited; 5-axis plateau confirmed |
| Person (WILDTRACK) | Ground-plane IDF1 | **0.947** (12b) | 0.953 | 0.6pp | Tracker-limited (Kalman); fully converged |

Vehicle config: TTA Stage-2 features (14c v2) + Stage-4 fusion `w_tertiary=0.525, similarity_threshold=0.48, aqe_k=2, fic_regularisation=0.5`. Person detector: MVDeTr ResNet18 (MODA=0.921). Full breakdown, model checkpoints, integration TODOs → [docs/performance-state.md](../docs/performance-state.md).

---

## Reference Docs (load on demand)

| File | When to load |
|------|--------------|
| [docs/findings.md](../docs/findings.md) | Before proposing any experiment — strategic narrative |
| [docs/dead-ends.md](../docs/dead-ends.md) | Before any parameter change — sweep history of what already failed |
| [docs/what-worked.md](../docs/what-worked.md) | Quick reference for positive deltas |
| [docs/performance-state.md](../docs/performance-state.md) | Detailed metrics, model checkpoints, integration status, PR list |
| [docs/experiment-log.md](../docs/experiment-log.md) | Full 225+ experiment ledger |
| [docs/kaggle-workflow.md](../docs/kaggle-workflow.md) | Before any `kaggle kernels push` — push safety, disk hygiene, session lifecycle |
| [docs/paper-strategy.md](../docs/paper-strategy.md) | Paper writing / venue decisions |
| [docs/models.md](../docs/models.md) | Model checkpoint provenance |

---

## Testing
```pwsh
pytest tests/ -v
python scripts/run_pipeline.py --config configs/default.yaml --smoke-test
```

---

## What NOT to Do
- Don't add `mtmc_only=True` for submission — drops single-cam tracks and hurts IDF1 ~5pp.
- Don't enable track smoothing or edge trim — neutral to harmful.
- Don't use text find/replace on raw JSON strings for Unicode — breaks JSON structure.
- Don't guess config override paths — trace from `cfg.stageN` in the pipeline code.
- Don't repeat dead-end experiments — check [docs/dead-ends.md](../docs/dead-ends.md) first.
- Don't compare ResNet101-IBN-a mAP to VeRi-776 baselines — our eval is CityFlowV2 (different dataset, 128 vs 576 IDs).
