# MTMC Tracker

MTMC Tracker is a multi-camera, multi-target tracking system for vehicles and
people. It tracks objects across a network of non-overlapping cameras and
assigns each object a single consistent identity across the whole network. The
project was developed as a graduation project and evaluated on three public
benchmarks: CityFlowV2 (AI City Challenge 2022, Track 1) for vehicles,
WILDTRACK for people, and VeRi-776 for single-camera vehicle re-identification.

The repository contains two things:

1. An offline seven-stage tracking pipeline (`src/`), driven from the command
   line.
2. A live application stack: a FastAPI backend (`backend/`) and a Next.js
   dashboard called ATHAR (`frontend/`) for interactive re-identification,
   multi-model score fusion, and evaluation.

## What it does

- **Detection and single-camera tracking.** Detects vehicles/people with YOLO
  and links them into per-camera tracklets with a BoxMOT tracker.
- **Appearance features.** Extracts a re-identification (ReID) embedding per
  tracklet (TransReID ViT-B/16 CLIP), an HSV colour histogram, and applies PCA
  whitening.
- **Cross-camera association.** Builds a tracklet similarity graph and solves it
  with connected components plus spatial-temporal and feature-improvement
  constraints to produce global trajectories.
- **Evaluation.** Reports HOTA, IDF1, MOTA (and MODA for the ground-plane person
  pipeline) with TrackEval.
- **Application.** Serves ReID search, multi-model fusion, and standalone
  evaluation jobs through the backend API and the ATHAR dashboard.

## Repository layout

```text
configs/                OmegaConf YAML config: default.yaml, datasets/, models/, model_registry.yaml
backend/                FastAPI service: app, routers/, services/, repositories/, models/
frontend/               Next.js (App Router) ATHAR dashboard
src/core/               Shared data models, config loading, IO utilities, constants
src/stage0_ingestion/   Frame extraction and preprocessing (CLAHE, resize)
src/stage1_tracking/    YOLO detection + BoxMOT tracking, tracklet building
src/stage2_features/    ReID embeddings (TransReID), HSV histograms, PCA whitening
src/stage3_indexing/    FAISS index + SQLite metadata
src/stage4_association/  Cross-camera association (similarity graph + connected components)
src/stage5_evaluation/  TrackEval metrics and MOTChallenge format conversion
src/stage6_visualization/ Annotated video, bird's-eye view, timeline outputs
src/stage_wildtrack_mvdetr/ MVDeTr ground-plane fast path for WILDTRACK
src/serving/            ReID model loaders and an LRU model cache used by the app
src/training/           ReID training loops, losses, and dataset builders
src/apps/               Streamlit dashboard, natural-language query, 3D simulation
scripts/                CLI entry points, asset download/verify, evaluation helpers
notebooks/kaggle/       GPU training, pipeline, and verification notebooks (run on Kaggle)
tests/                  Pytest test suite
docs/                   Architecture, dataset, and model-card reference docs
data/                   Local datasets and generated outputs (gitignored except small GT)
models/                 Local model checkpoints (gitignored)
```

## Architecture

The offline system is a seven-stage, file-based pipeline. Each stage reads the
previous stage's artifacts from `data/outputs/<run_id>/` and writes its own.

```text
stage0  Ingestion       Frames, preprocessing, dataset normalization
stage1  Tracking        YOLO detection and BoxMOT single-camera tracking
stage2  Features        TransReID CLIP embeddings, HSV histograms, PCA whitening
stage3  Indexing        FAISS inner-product index and SQLite metadata
stage4  Association     Tracklet similarity graph and connected-component solve
stage5  Evaluation      HOTA, IDF1, MOTA, MODA
stage6  Visualization   Annotated video, bird's-eye view, timeline exports
```

For WILDTRACK, `src/stage_wildtrack_mvdetr/` replaces stages 1-4 with an MVDeTr
ground-plane detector and a Kalman tracker.

See [docs/architecture.md](docs/architecture.md) for the detailed design.

## Requirements

- Python 3.10-3.13. A local virtual environment named `.venv` is recommended.
- Node.js 18+ and npm (only needed for the frontend dashboard).
- A free Kaggle account with an API token at `~/.kaggle/kaggle.json` (the model
  weights are hosted as a public Kaggle dataset; the API needs a token even for
  public data). See <https://www.kaggle.com/docs/api>.
- A CUDA GPU is optional. The ReID search and fusion endpoints run on CPU.
  Detection, tracking, and feature extraction over a full dataset are heavy and
  are intended to run on a GPU (this project used Kaggle T4/P100 for them).

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows; on Linux/macOS use: source .venv/bin/activate
pip install -r requirements.txt

python scripts/download_weights.py     # interactive model-set picker (or --set all)
python scripts/verify_assets.py        # verify checkpoint sizes/checksums
```

For the dashboard, also install the frontend dependencies:

```bash
cd frontend
npm install
cd ..
```

`SETUP.md` has the full asset table and options; `LAUNCH.md` has the app launch
commands.

## Datasets (not included)

The datasets are large, public, and externally hosted. They are **not** part of
this submission. Download them and place them under `data/raw/`:

| Dataset | Used for | Source |
| --- | --- | --- |
| CityFlowV2 (AI City 2022, Track 1) | Vehicle detection, tracking, MTMC, eval | <https://www.aicitychallenge.org/2022-data-and-evaluation/> (manual request) |
| WILDTRACK | Person ground-plane tracking and MTMC | WILDTRACK dataset release |
| VeRi-776 | Single-camera vehicle ReID training/eval | Public mirrors; a Kaggle copy is used by the setup script |
| Market-1501 | Person ReID training | Public person ReID benchmark |

`scripts/download_assets.py --datasets` fetches the public VeRi-776 evaluation
copy. CityFlowV2 must be requested and downloaded manually from the official AI
City Challenge site. The small CityFlowV2 ground-truth files under
`data/raw/cityflowv2/*/gt/` are kept in the repo so evaluation runs out of the
box; they originate from the CityFlowV2 dataset.

## Model weights (not included)

Model checkpoints are large binaries and are **not** committed. There are two
ways to obtain them:

1. **Download the pre-trained weights** used for the reported results. They are
   consolidated in one public Kaggle dataset and SHA-256 pinned in
   `configs/weights_manifest.yaml`:

   ```bash
   python scripts/download_weights.py --list        # show sets and files
   python scripts/download_weights.py --set all      # download everything (~2.3 GB)
   python scripts/download_weights.py --set veri      # just the VeRi-776 fusion streams
   ```

2. **Regenerate them by training.** The GPU training notebooks under
   `notebooks/kaggle/` reproduce each checkpoint (for example
   `09_vehicle_reid_cityflowv2/` and `13_clip_senet_train/` for the ReID
   backbones, `12a_wildtrack_mvdetr/` for the person detector). They are
   designed to run on Kaggle and write their checkpoints as kernel outputs.

Place downloaded or trained checkpoints under `models/` (see
[models/reid/README.md](models/reid/README.md) for the expected filenames,
sizes, and checksums). `scripts/verify_assets.py` checks them.

## Running the pipeline

The main entry point is `scripts/run_pipeline.py`:

```bash
# Vehicle pipeline (CityFlowV2). The base config is paired with a dataset config.
python scripts/run_pipeline.py --config configs/default.yaml --dataset-config configs/datasets/cityflowv2.yaml

# Person pipeline (WILDTRACK)
python scripts/run_pipeline.py --config configs/default.yaml --dataset-config configs/datasets/wildtrack.yaml

# A subset of stages, a quick smoke run, or a config-only dry run:
python scripts/run_pipeline.py --config configs/default.yaml --dataset-config configs/datasets/cityflowv2.yaml --stages 3,4,5
python scripts/run_pipeline.py --config configs/default.yaml --dataset-config configs/datasets/cityflowv2.yaml --smoke-test
python scripts/run_pipeline.py --config configs/default.yaml --dataset-config configs/datasets/cityflowv2.yaml --dry-run
```

Outputs are written under `data/outputs/<run_id>/`.

## Running the live app

Two terminals (or use `python start.py` to launch both at once):

```bash
# Terminal 1: backend (FastAPI on port 8000)
python -m uvicorn backend_api:app --host 127.0.0.1 --port 8000

# Terminal 2: frontend (Next.js on port 3001)
cd frontend
npm run dev
```

Then open <http://127.0.0.1:3001>. The dashboard provides interactive ReID
search, multi-model score fusion, and standalone evaluation jobs. See
[LAUNCH.md](LAUNCH.md) for the available pages and troubleshooting.

## Tests

```bash
pytest tests/
```

An application end-to-end check that starts the backend, exercises the
endpoints, and tears down is available with:

```bash
python scripts/test_phase2_e2e.py
```

## Results

| Pipeline | Benchmark | Metric | Value |
| --- | --- | --- | --- |
| Vehicle MTMC | CityFlowV2 | MTMC IDF1 | 0.779 |
| Person MTMC | WILDTRACK | IDF1 / MODA | 0.946 / 0.903 |
| Vehicle ReID | VeRi-776 | mAP (TransReID) | 89.97 |
| Vehicle ReID | VeRi-776 | mAP (two-stream fusion) | 93.30 |

The registered checkpoints and their verified metrics are listed in
`configs/model_registry.yaml` and documented in
[docs/model-cards.md](docs/model-cards.md).

## Third-party components and attributions

This project builds on open-source libraries and published methods. The
external libraries are installed from `requirements.txt` / `frontend/package.json`
and are **not** vendored into this repository:

- Detection: Ultralytics YOLO. Tracking: BoxMOT. Indexing: FAISS.
  Metrics: TrackEval and py-motmetrics. Backbones: PyTorch, timm, OpenCLIP.
  Backend: FastAPI. Frontend: Next.js, React, Radix UI, Tailwind CSS.
- `frontend/src/components/ui/` contains UI primitives generated from the
  shadcn/ui component library (<https://ui.shadcn.com/>).
- Several modules are our own implementations of published algorithms and cite
  the source paper in their header, including TransReID (He et al., ICCV 2021),
  the ReID "bag of tricks" baseline (Luo et al., CVPRW 2019), average query
  expansion (Chum et al., ICCV 2007), k-reciprocal re-ranking (Zhong et al.,
  CVPR 2017), and feature-improvement camera whitening (Liu et al., CVPRW 2021).

## License

Released under the MIT License. See [LICENSE](LICENSE).
