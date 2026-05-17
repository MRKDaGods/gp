# MTMC Tracker

MTMC Tracker is a multi-camera multi-target tracking system for vehicles and
persons. It was developed as a graduation thesis project and evaluated on AI
City Challenge 2022 Track 1 (CityFlowV2), WILDTRACK, and VeRi-776.

## Status

The main reproducible metrics currently tracked by the project are:

- Vehicle MTMC on CityFlowV2: MTMC IDF1 0.77936. The AIC22 first-place
  ensemble reports about 0.8486 IDF1, leaving a gap of about 6.9 percentage
  points.
- Person MTMC on WILDTRACK: IDF1 0.946. The project reference target is about
  0.953 IDF1, leaving a gap of about 0.7 percentage points.
- Person MTMC on WILDTRACK: MODA 0.903 for the ground-plane tracking operating
  point.
- Vehicle ReID on VeRi-776: TransReID ViT-B/16 CLIP mAP 89.97.
- Vehicle ReID on VeRi-776: CLIP-SENet with rerank and AQE mAP 91.54.
- Vehicle ReID on VeRi-776: TransReID x CLIP-SENet score fusion mAP 93.30.

These numbers are recorded in `docs/findings.md` and
`configs/model_registry.yaml`. The main research conclusion is that feature
quality and cross-camera invariance are the limiting factors for MTMC IDF1.
Association tuning has been tested extensively and does not close the remaining
performance gap.

## Architecture

The offline system is organized as a seven-stage file-based pipeline. Each
stage reads artifacts from the previous stage and writes run-scoped outputs
under `data/outputs/`.

```text
src/stage0/  Ingestion       Frames, preprocessing, dataset normalization
src/stage1/  Tracking        YOLO26m detection and BoT-SORT tracking
src/stage2/  Features        TransReID CLIP, HSV, and PCA whitening
src/stage3/  Indexing        FAISS IndexFlatIP and SQLite metadata
src/stage4/  Association     Similarity graph and NetworkX components
src/stage5/  Evaluation      TrackEval IDF1, HOTA, MOTA, and MODA
src/stage6/  Visualization   Annotated video, BEV, and timeline outputs
```

The repository also contains a live application stack. `backend/` is a FastAPI
service layer for model registry access, ReID inference, fusion experiments,
evaluation jobs, and pipeline orchestration. `frontend/` is the Next.js ATHAR
dashboard used for interactive ReID search, model fusion, and evaluation
workflows. The app uses the same model registry and local artifact layout as
the offline pipeline.

## Repository Layout

```text
configs/          OmegaConf YAML configuration and model registry entries
backend/          FastAPI service, routers, schemas, and orchestration code
frontend/         Next.js ATHAR dashboard
src/core/         Shared data models, configuration loading, and utilities
src/stage0/       Frame ingestion and preprocessing
src/stage1/       Detection and single-camera tracking
src/stage2/       ReID feature extraction and feature preprocessing
src/stage3/       FAISS indexing and metadata storage
src/stage4/       Cross-camera association
src/stage5/       Evaluation metrics and format conversion
src/stage6/       Visualization outputs
src/serving/      ReID model loaders and LRU model cache used by the app
src/apps/         Streamlit dashboard, NL query tools, and 3D simulation
scripts/          CLI entry points, setup helpers, and verification scripts
notebooks/kaggle/ GPU training, pipeline, and verifier notebooks
tests/            Pytest test suite
docs/findings.md  Research log, experiment outcomes, and metric claims
data/             Local datasets and generated outputs, gitignored
models/           Local model checkpoints, gitignored
```

## Setup

Use Python 3.10 or newer. On this project, the local virtual environment is
`.venv`. Install dependencies, configure Kaggle credentials, download public
checkpoints and optional datasets, and verify the local asset layout with the
setup scripts. See `SETUP.md` for asset download details and `LAUNCH.md` for
backend and frontend launch commands. CityFlowV2 must be downloaded manually
from the AI City Challenge site because the complete dataset is not available
as a public Kaggle dataset.

## Datasets

- CityFlowV2 / AI City Challenge 2022 Track 1: used for vehicle detection,
  tracking, ReID fine-tuning, MTMC association, and evaluation. Download it
  from the AI City Challenge data portal:
  <https://www.aicitychallenge.org/2022-data-and-evaluation/>.
- WILDTRACK: used for person detection, ground-plane tracking, and
  multi-camera person MTMC evaluation. Use the WILDTRACK project dataset
  release.
- VeRi-776: used for single-camera vehicle ReID training and evaluation for
  TransReID, CLIP-SENet, and fusion experiments. Public mirrors are available;
  the setup script uses a Kaggle-hosted copy for optional local evaluation.

## Models

The deployed and research checkpoint entries are registered in
`configs/model_registry.yaml`. The registry contains checkpoint paths, hosted
artifact references, model status, and verification metadata.

| Registry entry | Training data | Headline metric |
| --- | --- | --- |
| `vehicle_mtmc_14e_b1` | CityFlowV2 | MTMC IDF1 0.77936 |
| `person_mtmc_12b` | WILDTRACK | IDF1 0.946; MODA 0.903 |
| `cityflow_transreid` | CityFlowV2 | single-camera mAP 81.53 |
| `veri776_09v_v17_transreid` | VeRi-776 | mAP 89.97; R1 98.33 |
| `veri776_clipsenet_v6` | VeRi-776 | rerank+AQE mAP 91.54 |
| `veri776_14t_fusion` | VeRi-776 | mAP 93.30; R1 98.45 |

## Running The Pipeline

The main entry point is `scripts/run_pipeline.py`:

```bash
python scripts/run_pipeline.py --config configs/default.yaml
```

Use `--stages` to run a subset of stages and `--run-id` to control the output
directory. A small smoke path is available with:

```bash
python scripts/run_pipeline.py --config configs/default.yaml --smoke-test
```

GPU-heavy stages, especially detection, tracking, feature extraction, and ReID
training, are intended to run on Kaggle for this project. Local development is
used for code editing, CPU-friendly stages, app work, and tests.

## Running The Live App

The live stack consists of a FastAPI backend and the Next.js ATHAR frontend. It
supports interactive ReID, multi-model score fusion, and standalone evaluation
jobs. Use `LAUNCH.md` for the current two-terminal launch commands, available
routes, and troubleshooting notes.

## Tests

Run the Python test suite with:

```bash
pytest tests/
```

Run the application end-to-end verifier with:

```bash
python scripts/test_phase2_e2e.py
```

Several Kaggle verifier kernels are used for metric-level reproduction and
regression checks: 14v for CityFlowV2 14e B1 reproduction, 14w for WILDTRACK
tracking, 14x for CityFlowV2 sibling variants, 14y for VeRi-776 ReID
checkpoints, 14z for WILDTRACK MVDeTr detector evaluation, and 14aa for the 14t
fusion path.

## Research Findings

The research record is maintained in `docs/findings.md`, with supporting detail
in `docs/experiment-log.md`. The project has run more than 225 ablation
experiments across feature extraction, score fusion, query expansion, FIC
whitening, graph thresholds, network flow, tracklet filtering, and
person-tracking variants. The recurring result is that MTMC IDF1 is limited
mainly by feature quality and cross-camera invariance, not by additional
association tuning.

The paper direction documented in the repository is an efficiency and ablation
study: one main model family reaches about 91 percent of the AIC22 first-place
ensemble score while exposing which pipeline components are already saturated.

## License

The package metadata in `pyproject.toml` declares the project license as MIT. No
standalone `LICENSE` file is currently present in the repository.
