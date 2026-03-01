# MTMC Tracker — Multi-Camera City-Wide Tracking System

Multi-camera tracking system for vehicles and humans on a city-wide scale. Processes offline video from multiple stationary cameras, performs detection, single-camera tracking, re-identification, and cross-camera association to produce global trajectories.

## Architecture

```
Stage 0: Ingestion       → Frame extraction, preprocessing, format unification
Stage 1: Tracking        → YOLO26 detection + BoxMOT per-camera tracking
Stage 2: Features        → ReID embeddings (OSNet/ResNet50-IBN) + HSV histograms + PCA
Stage 3: Indexing        → FAISS vector index + SQLite metadata store
Stage 4: Association     → Cross-camera matching via similarity graph + connected components
Stage 5: Evaluation      → HOTA, IDF1, MOTA metrics via TrackEval
Stage 6: Visualization   → Annotated videos, BEV maps, timeline views
Apps:    Dashboard        → Streamlit web UI, NL query, 3D simulation
```

## Quick Start

```bash
# Install
pip install -e .

# Download pre-trained models (YOLO26, BoxMOT ReID weights)
python scripts/download_models.py

# Run full pipeline
python scripts/run_pipeline.py --config configs/default.yaml

# Run a single stage
python scripts/run_stage.py --config configs/default.yaml --stage 1

# Run smoke test (tiny data, fast)
python scripts/run_pipeline.py --config configs/default.yaml --smoke-test

# Launch web dashboard
streamlit run src/apps/web_dashboard.py
```

## Project Structure

```
configs/          YAML configuration files
src/core/         Shared data models, config loader, utilities
src/stage0-6/     Pipeline stages (each with pipeline.py entry point)
src/apps/         Applications (dashboard, NL query, 3D sim)
scripts/          CLI entry points
notebooks/kaggle/ Kaggle training notebooks (ReID models)
tests/            pytest test suite
docs/             Documentation for team and supervisor
```

## Training (Kaggle)

ReID models are trained on Kaggle due to GPU constraints. See `notebooks/kaggle/` for:
1. `01_dataset_preparation.ipynb` — Prepare Market-1501, VeRi-776
2. `02_person_reid_training.ipynb` — Train person ReID (OSNet / ResNet50-IBN)
3. `03_vehicle_reid_training.ipynb` — Train vehicle ReID
4. `04_advanced_reid_training.ipynb` — TransReID (stretch goal)

After training, download weights to `models/reid/`.

## Datasets

| Dataset | Purpose | Source |
|---|---|---|
| Market-1501 | Person ReID training | Kaggle |
| VeRi-776 | Vehicle ReID training | Kaggle |
| AI City Challenge 2023 | Vehicle MTMC evaluation | Kaggle |
| MOT17 | Single-camera tracking validation | Kaggle |

## Team

See `docs/team_workload.md` for work division across 4 developers.

## Documentation

- `docs/architecture.md` — System architecture with Mermaid diagrams
- `docs/data_contracts.md` — Inter-stage data formats
- `docs/dataset_guide.md` — Dataset preparation guide
- `docs/setup_guide.md` — Environment setup
- `docs/kaggle_training_guide.md` — Kaggle training instructions
- `docs/team_workload.md` — Work division
