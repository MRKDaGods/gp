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

### Vehicle Pipeline (CityFlowV2)
- **Best Reproducible MTMC IDF1**: 0.775 (10c v52, gumfreddy, v80-restored recipe)
- **Historical Best MTMC IDF1**: 0.784 (v80/v44, ali369) — ~1pp drift, not reproducible on current codebase
- **SOTA target**: IDF1≈0.8486 (AIC22 1st place, 5-model ensemble)
- **Gap to SOTA**: 7.36pp — caused by feature quality (single model), NOT association tuning
- **Primary model**: TransReID ViT-B/16 CLIP 256px — mAP=80.14%, R1=92.27% on CityFlowV2
- **Secondary model**: ResNet101-IBN-a — mAP=52.77% (too weak for ensemble, needs ≥65%)
- **Association**: EXHAUSTED (225+ configs, all within 0.3pp of optimal)
- See `docs/findings.md` for full analysis, dead ends, and action plan

### Person Pipeline (WILDTRACK)
- **Best Ground-plane IDF1**: 0.947 (confirmed across 12b v1, v2, v3; 59+ configs tested)
- **Best Ground-plane MODA**: 0.903 (12b v14)
- **Detector**: MVDeTr ResNet18, MODA=0.921 (12a v3, best achieved)
- **SOTA target**: IDF1≈0.953
- **Gap to SOTA**: 0.6pp — tracker-limited (Kalman), NOT detector-limited
- **Status**: FULLY CONVERGED — tracker-limited and exhaustively tested; Kalman, global optimal, and naive trackers all failed to beat 0.947

## Experiment History
- **Full experiment log**: See `docs/experiment-log.md` for 225+ tracked experiments
- **Research findings**: See `docs/findings.md` for dead ends, strategic analysis, and what to do next
- Before trying ANY parameter change, check BOTH documents to avoid repeating failed experiments

### Confirmed Dead Ends (DO NOT RETRY)
- **CSLS**: -34.7pp (catastrophic — penalizes genuine vehicle-type hubs)
- **384px ViT deployment**: -2.8pp (captures viewpoint-specific textures that hurt cross-camera matching)
- **AFLink motion linking**: confirmed harmful at **-3.8pp to -13.2pp MTMC IDF1** in clean retests; even `gap=100`, `dir_cos=0.90` loses **-3.82pp** because motion consistency is unreliable across non-overlapping CityFlowV2 cameras and AFLink creates false merges
- **CID_BIAS**: GT-learned version -3.3pp; topology CID_BIAS -1.0 to -1.2pp (additive bias distorts FIC-calibrated similarities)
- **DMT camera-aware training**: -1.4pp single-model (also 09g: 43.8% mAP, too weak)
- **Hierarchical clustering**: -1 to -5pp (centroid averaging loses discriminative signal)
- **FAC**: -2.5pp (cross-camera KNN consensus overwrites distinguishing details)
- **Reranking**: Always hurts (k-reciprocal sets contain false positives with current features)
- **Feature concatenation**: -1.6pp (mixes uncalibrated feature spaces)
- **Network flow solver**: -0.24pp MTMC IDF1, increased conflation from 27→30 instead of reducing it
- **VeRi-776→CityFlowV2 ResNet pretrain**: 42.7% mAP (worse than direct 52.77%)
- **Extended ResNet fine-tuning**: 50.61% mAP (degraded from 52.77%)
- **ArcFace on ResNet101-IBN-a**: 50.80% mAP (warm-start geometry mismatch, 6 variants exhausted at 52.77% ceiling)
- **ResNeXt101-IBN-a ArcFace**: 36.88% mAP (IBN-Net pretrained weights were for 32x32d while the model here used 32x8d; `strict=False` partial loading left many layers random and crippled training)
- **Score-level ensemble with 52.77% secondary**: -0.1pp (secondary too weak, adds noise)
- **Circle loss + triplet**: 16-30% mAP (conflicting gradients)
- **SGD for ResNet**: 30.27% mAP (catastrophic — AdamW essential for small datasets)
- **Global optimal tracker (person)**: -3.5pp IDF1 vs Kalman (assignment costs lose motion prediction advantage)
- **Extended Kalman sweeps (person)**: 59 configs within +-0.0004 IDF1 — fully exhausted
- **Person: improved detector→better tracking**: MODA 90.9→92.1% but IDF1 unchanged at 94.7%

### What Actually Worked
- Conflict-free CC (+0.21pp), intra-merge (+0.28pp), temporal overlap bonus (+0.9pp)
- FIC whitening (+1-2pp), power normalization (+0.5pp), PCA 384D, AQE K=3
- min_hits=2 (+0.2pp), Kalman tuning for person (+1.9pp IDF1)

### Remaining Untried Approaches
- GNN edge classification for association (not implemented)
- SAM2 foreground masking before ReID (not implemented)
- Graph-based multi-view tracking for person pipeline (not implemented)

## Kaggle Workflow
- Pipeline chain: 10a (stages 0-2, GPU) → 10b (stage 3, CPU) → 10c (stages 4-5, CPU)
- Push: `kaggle kernels push -p notebooks/kaggle/10X_stagesNN/`
- Logs: `python scripts/kaggle_logs.py <kernel_slug> --tail N`
- Auth tokens in `~/.kaggle/`: abdo (gumfreddy), mrk (mrkdagods), ali369 (lolo)
- Current active account: gumfreddy (gumfreddy_access_token) — ali369/mrkdagods tokens may be missing from ~/.kaggle/

### Kaggle Push Safety Rules (CRITICAL)
- **NEVER push a kernel more than once without confirming the previous version is fully running or complete** — rapid re-pushes create duplicate GPU sessions that consume both slots and block all other work
- After every push, check for warning lines like `The following are not valid dataset sources` — these indicate the run started but with missing inputs; **immediately cancel** the bad run via `kaggle kernels cancel <owner/slug>` before attempting a fix-and-repush
- If `kaggle kernels cancel` fails or is unavailable, **STOP immediately and tell the user** with the kernel URL so they can cancel manually from the Kaggle web UI
- Kaggle allows a maximum of 2 concurrent GPU sessions per account — always check active sessions before pushing a GPU-enabled notebook
- When iterating on kernel-metadata.json fixes, validate metadata locally first, then push **once**

### Kaggle Push Safety Rules (CRITICAL)
- **NEVER push a kernel more than once without confirming the previous version is fully running or complete** — rapid re-pushes create duplicate GPU sessions that consume both slots and block all other work
- After every push, check for warning lines like `The following are not valid dataset sources` — these indicate the run started but with missing inputs; **immediately cancel** the bad run via `kaggle kernels cancel <owner/slug>` before attempting a fix-and-repush
- If `kaggle kernels cancel` fails or is unavailable, **STOP immediately and tell the user** with the kernel URL so they can cancel manually from the Kaggle web UI
- Kaggle allows a maximum of 2 concurrent GPU sessions per account — always check active sessions before pushing a GPU-enabled notebook
- When iterating on kernel-metadata.json fixes, validate metadata locally first, then push **once**

### Kaggle Push Safety Rules (CRITICAL)
- **NEVER push a kernel more than once without confirming the previous version is fully running or complete** — rapid re-pushes create duplicate GPU sessions that consume both slots and block all other work
- After every push, check for warning lines like `The following are not valid dataset sources` — these indicate the run started but with missing inputs; **immediately cancel** the bad run via `kaggle kernels cancel <owner/slug>` before attempting a fix-and-repush
- If `kaggle kernels cancel` fails or is unavailable, **STOP immediately and tell the user** with the kernel URL so they can cancel manually from the Kaggle web UI
- Kaggle allows a maximum of 2 concurrent GPU sessions per account — always check active sessions before pushing a GPU-enabled notebook
- When iterating on kernel-metadata.json fixes, validate metadata locally first, then push **once**

## Paper Strategy
- A full publishability analysis and paper strategy is documented in `docs/paper-strategy.md`
- Best paper angle: "One Model, 91% of SOTA" — efficiency + exhaustive ablation study
- Target venues: IEEE Access, Multimedia Tools & Applications, Scientific Reports
- Key contribution: 225+ experiments proving feature quality (not association) is the MTMC bottleneck
- Before any paper-related work, read `docs/paper-strategy.md` for the full analysis

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
