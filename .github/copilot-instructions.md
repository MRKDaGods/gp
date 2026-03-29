# GitHub Copilot Instructions — MTMC Tracker

## Project Overview
Multi-camera multi-target tracking system (vehicles/humans) on CityFlowV2 (AI City Challenge 2022 Track 1). 7-stage offline pipeline: Ingestion → Tracking → Features → Indexing → Association → Evaluation → Visualization. Python 3.10+, PyTorch, YOLO26m, TransReID ViT-Base/16 CLIP, FAISS, SQLite, Streamlit.

## Critical Rules (NEVER violate)

### GPU Pipeline Execution
- **NEVER** run GPU-intensive pipeline stages (0, 1, 2) locally — the local machine has a GTX 1050 Ti which is too slow and starves other work
- ALL GPU-intensive work (detection, tracking, feature extraction, ReID training) MUST run on Kaggle
- Local machine is ONLY for: code editing, pushing notebooks, monitoring Kaggle kernels, and running CPU-only stages (3, 4, 5) on small datasets
- The virtual environment MUST be used for any local Python: `.venv` (Python 3.11.9), NOT system Python 3.13
- Activate with: `.\.venv\Scripts\activate`

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

### Research Findings — MUST READ
- **Always read `docs/findings.md` before proposing experiments or changes** — it contains all dead ends, current performance, and strategic analysis
- **Update `docs/findings.md`** whenever: new experiments produce results, new dead ends are discovered, performance numbers change, or new insights are gained
- The findings doc is the canonical source of truth for what has been tried and what to do next
- Before trying ANY approach, check the "Dead Ends" section to avoid repeating failed experiments

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
docs/findings.md  Research findings, dead ends, strategic analysis (KEEP UPDATED)
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
- **Best Kaggle MTMC IDF1**: 0.784 (v80/v44, ali369, min_hits=2) — official metric, comparable to SOTA
- **Best IDF1 (single-acc)**: 0.798 (v41) — NOT the same as MTMC IDF1
- **Best GLOBAL IDF1**: 0.805 (v85) — NOT comparable to SOTA (per-camera accumulators)
- **SOTA target**: IDF1≈0.8486 (AIC22 1st place)
- **Gap**: ~6.5pp MTMC IDF1 from SOTA — all three metrics are different, only MTMC IDF1 is comparable to AIC22
- **Association**: EXHAUSTED (225+ configs, all within 0.3pp of optimal)
- **Critical blocker**: 384px model (80.14% mAP) never properly deployed — wrong checkpoint on Kaggle
- **Secondary model**: ResNet101-IBN-a at 52.77% mAP — expected given no VeRi-776 pretraining
- See `docs/findings.md` for full analysis, dead ends, and action plan

## Experiment History
- **Full experiment log**: See `docs/experiment-log.md` for 225+ tracked experiments
- **Research findings**: See `docs/findings.md` for dead ends, strategic analysis, and what to do next
- Before trying ANY parameter change, check BOTH documents to avoid repeating failed experiments
- Key dead ends: CSLS (-34.7pp), hierarchical clustering (-1-5pp), FAC (-2.5pp), reranking (always worse with current features), camera-pair norm (zero effect), SGD for ResNet (catastrophic)
- Association parameters are EXHAUSTED. Future gains come from: deploying correct 384px model, VeRi-776 pretraining for ResNet101, CID_BIAS

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
- Don't repeat dead-end experiments — check `docs/findings.md` first
- Don't compare ResNet101-IBN-a mAP to VeRi-776 baselines — our eval is on CityFlowV2 (different dataset, 128 vs 576 IDs)
