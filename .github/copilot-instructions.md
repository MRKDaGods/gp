# GitHub Copilot Instructions — MTMC Tracker

## Project Overview
Multi-camera multi-target tracking system (vehicles/humans) on CityFlowV2 (AI City Challenge 2022 Track 1). 7-stage offline pipeline: Ingestion → Tracking → Features → Indexing → Association → Evaluation → Visualization. Python 3.10+, PyTorch, YOLO26m, TransReID ViT-Base/16 CLIP, FAISS, SQLite, Streamlit.

## Critical Rules (NEVER violate)

### Notebook Editing
- **NEVER** use `replace_string_in_file` on `.ipynb` files — it edits VS Code's in-memory buffer but may NOT save to disk
- For `.ipynb` edits: use `json.load() → modify → json.dump()` via a Python script
- After any `.ipynb` edit, verify on-disk state with `python -c "import json; ..."`
- Each line in notebook `source` arrays MUST end with `\n` EXCEPT the last line
- On Windows, use `ensure_ascii=True` in `json.dump` to avoid charmap codec errors

### Frame ID Convention
- Internal pipeline (Stages 0-4): **0-based** frame IDs
- MOT submission format: **1-based** (converted via `frame_id + 1` in format_converter)
- CityFlowV2 GT: **1-based** (standard MOT format)
- Never mix these — check which context you're in

### Config Override Paths
- Stage4 reads from `cfg.stage4.association` — override must be `stage4.association.graph.similarity_threshold=X`
- NOT `stage4.graph.similarity_threshold=X` (wrong level)
- NOT `stage4.similarity_threshold=X` (wrong level)
- Similarly: `stage4.association.fic.regularisation=X`, `stage4.association.gallery_expansion.threshold=X`
- Config loading: `default.yaml` → merge `cityflowv2.yaml` → CLI overrides

## Architecture

```
configs/          YAML configuration (OmegaConf)
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
```

## Key Dependencies & Versions
- Python >=3.10, <3.14
- PyTorch >=2.1, torchvision >=0.16
- timm >=0.9 (TransReID ViT-Base/16 CLIP backbone)
- ultralytics >=8.1 (YOLO26m detection)
- boxmot >=10.0 (BoT-SORT tracker)
- faiss-cpu >=1.7 (cosine similarity indexing)
- omegaconf >=2.3 (hierarchical config)
- networkx >=3.1 (graph algorithms)
- streamlit >=1.28 (web dashboard)
- P100 GPUs on Kaggle: need PyTorch 2.4.1+cu124 (sm_60 compat)

## Code Patterns

### Naming
- Stages: `src/stageN_name/` with `pipeline.py` as entry point
- Config sections mirror stage names: `stage0`, `stage1`, ..., `stage4.association`
- Test files: `tests/test_stageN/test_component.py`

### Configuration
- All config via OmegaConf from YAML + CLI overrides
- No env vars or OmegaConf resolvers — all explicit `--override` args
- Access: `cfg.stage4.association.graph.similarity_threshold`

### Data Flow Between Stages
- Stages communicate through files in `data/outputs/`
- Each stage reads predecessor's output, writes its own
- Tracklets: list of dicts with `camera_id`, `track_id`, `frames`, `boxes`, `embeddings`

## Current Performance State
- **Best Kaggle**: IDF1=0.784 (v80, 10c v44, ali369 account — min_hits=2)
- **Historical local claim**: IDF1=0.8297 (v47 — unverifiable, predates current experiment log)
- **SOTA target**: IDF1≈0.84 (AIC21 published)
- **Gap**: ~5.6pp from SOTA, caused by feature quality not association tuning (220+ configs exhausted)
- Key params (v80 best): sim_thresh=0.53, fic_reg=0.1, app_w=0.70, conflict_free_cc, min_hits=2, PCA=384D

## Kaggle Workflow
- Pipeline chain: 10a (stages 0-2, GPU) → 10b (stage 3, CPU) → 10c (stages 4-5, CPU)
- Push: `kaggle kernels push -p notebooks/kaggle/10X_stagesNN/`
- Logs: `python scripts/kaggle_logs.py <kernel_slug> --tail N`
- Auth tokens in `~/.kaggle/`: abdo (gumfreddy), mrk (mrkdagods), ali369 (lolo)
- Current account: ali369 (lolo_access_token)

## Testing
- Framework: pytest
- Run: `pytest tests/ -v`
- Smoke test: `python scripts/run_pipeline.py --config configs/default.yaml --smoke-test`

## What NOT to Do
- Don't add `mtmc_only=True` for submission — it drops single-cam tracks and hurts IDF1 by ~5pp
- Don't enable track smoothing or edge trim — neutral to harmful
- Don't use text find/replace on raw JSON strings for Unicode — breaks JSON structure
- Don't guess config override paths — always trace from `cfg.stageN` in the pipeline code
