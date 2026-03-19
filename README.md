# MTMC Tracker — Multi-Camera City-Wide Tracking System

Multi-camera tracking system for vehicles and humans on a city-wide scale. Processes offline video from multiple stationary cameras, performs detection, single-camera tracking, re-identification, and cross-camera association to produce global trajectories.

## Architecture

```
Stage 0: Ingestion       → Frame extraction, preprocessing, format unification
Stage 1: Tracking        → YOLO26 detection + BoxMOT per-camera tracking
Stage 2: Features        → ReID embeddings (TransReID/OSNet) + HSV histograms + PCA
Stage 3: Indexing        → FAISS vector index + SQLite metadata store
Stage 4: Association     → Cross-camera matching via similarity graph + connected components
Stage 5: Evaluation      → HOTA, IDF1, MOTA metrics via TrackEval
Stage 6: Visualization   → Annotated videos, BEV maps, timeline views
Apps:    Dashboard       → Next.js web UI, NL query, 3D simulation
```

## Quick Start

### Backend (Python Pipeline)

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

# Launch legacy Streamlit dashboard
streamlit run src/apps/web_dashboard.py
```

### Frontend (Next.js Dashboard)

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev

# Build for production
npm run build

# Start production server
npm start
```

The frontend will be available at `http://localhost:3000`.

## Frontend Features

Modern Next.js dashboard with UniFi-style dark theme:

1. **Splash Screen** → Animated logo → Main dashboard
2. **Stage 0: Upload** → Drag-drop upload, video gallery, preview
3. **Stage 1: Detection** → YOLO bounding boxes (red), auto-run on upload
4. **Stage 2: Selection** → Click boxes to select (green), multi-select mode
5. **Stage 3: Inference** → Location filters (Egypt hierarchy), DateTime picker
6. **Stage 4: Timeline** → Clipchamp-style tracklet editor, split-screen video
7. **Stage 5: Refinement** → Select reference frames, re-search, variable speed
8. **Stage 6: Output** → Summarized video, grid view (1x1 to 5x5), statistics

### Frontend Tech Stack

- Next.js 14 + TypeScript + Tailwind CSS
- shadcn/ui components
- Zustand (state management)
- TanStack Query (data fetching)
- Video.js + Leaflet (future maps)

## Project Structure

```
configs/          YAML configuration files
src/core/         Shared data models, config loader, utilities
src/stage0-6/     Pipeline stages (each with pipeline.py entry point)
src/apps/         Applications (Streamlit dashboard, NL query, 3D sim)
frontend/         Next.js web dashboard (NEW)
scripts/          CLI entry points
notebooks/kaggle/ Kaggle training notebooks (ReID models)
tests/            pytest test suite
docs/             Documentation for team and supervisor
```

## Future Steps

1. **Local Demo Finalization**: Fully contain the app with local videos (Scene 2 subset for testing), local models, and self-hosted inference to ensure complete end-to-end functionality.
2. **Full Dataset Processing**: Once verified with the Scene 2 subset, scale inference to support the full uncompressed dataset.
3. **Cloud Deployment (Production)**: Decouple the monolithic app into a frontend client and a heavy backend server. The backend (inference engine and database) will run on a virtual machine or cloud infrastructure, while the UI will serve as a lightweight frontend connected via API.

## Future: Map-Based Features

When GPS/coordinate data is acquired:

- **2D Interactive Map**: Vehicle paths visualization on Leaflet map
- **Heatmap**: Vehicle density and common routes
- **Spatiotemporal Constraints**:
  - Max speed slider (0-200 km/h) to filter impossible matches
  - Search radius slider for maximum travel distance
  - Time window configuration

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
