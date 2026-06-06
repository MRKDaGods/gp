# Setup and How to Run

This is the long version of how to get the project running from a clean machine.
The README has the short version if that's all you need. I wrote this one assuming
you have nothing but Python installed and have never seen the project before, so I
tried not to skip steps. If something here doesn't match what you see, the README,
SETUP.md and LAUNCH.md are the shorter references.

A quick map of what's coming:

1. [What the project actually is](#1-what-the-project-actually-is)
2. [What you need installed first](#2-what-you-need-installed-first)
3. [Getting the code](#3-getting-the-code)
4. [Setting up Python](#4-setting-up-python)
5. [The Python libraries (and why each one is there)](#5-the-python-libraries-and-why-each-one-is-there)
6. [Setting up Kaggle](#6-setting-up-kaggle)
7. [Getting the model weights](#7-getting-the-model-weights)
8. [The datasets, their sizes, and where they go](#8-the-datasets-their-sizes-and-where-they-go)
9. [Checking everything is in place](#9-checking-everything-is-in-place)
10. [Running the pipeline](#10-running-the-pipeline)
11. [Using your own footage and cameras](#11-using-your-own-footage-and-cameras)
12. [Running the app](#12-running-the-app)
13. [Reproducing the VeRi-776 numbers](#13-reproducing-the-veri-776-numbers)
14. [Running the tests](#14-running-the-tests)
15. [What results to expect](#15-what-results-to-expect)
16. [Where everything lives](#16-where-everything-lives)
17. [When things go wrong](#17-when-things-go-wrong)

---

## 1. What the project actually is

It's a multi-camera tracking system for vehicles and people. The idea is that the
same car or person shows up on several cameras that don't overlap, and the system
figures out it's the same object and gives it one ID across all of them.

There are two parts to it:

- The offline pipeline in `src/`. You point it at some video, and it runs through
  seven stages: pull frames out of the video, detect and track objects per camera,
  build appearance features, match identities across cameras, then score the result
  (HOTA / IDF1 / MOTA).
- The live app: a FastAPI backend (`backend/`) and a Next.js dashboard we called
  ATHAR (`frontend/`). It's the interactive side: upload images, search by
  appearance, fuse a few models together, run evaluations.

We tested it on CityFlowV2 (AI City Challenge 2022, Track 1) for vehicles,
WILDTRACK for people, and VeRi-776 for single-camera vehicle re-id. The pipeline
isn't tied to those, though, so if you have your own cameras there's a section on
that further down (section 11).

---

## 2. What you need installed first

Get these on the machine before anything else:

| Tool | Version | What it's for | Link |
| --- | --- | --- | --- |
| Python | 3.10 to 3.13 (I used 3.13) | Pretty much everything | <https://www.python.org/downloads/> |
| pip | comes with Python | Installing the libraries | - |
| git | anything recent | Cloning the repo | <https://git-scm.com/downloads> |
| Node.js + npm | Node 18 or newer (I used 22) | Only the frontend dashboard | <https://nodejs.org/> |
| A Kaggle account | free | Pulling the weights and the VeRi-776 data | <https://www.kaggle.com/> |

On the hardware side: a plain CPU is fine for the app and the lighter pipeline
stages (indexing, matching, scoring). A GPU is optional but helps a lot for the
heavy stages (detection, tracking, feature extraction). We ran those on Kaggle
T4/P100 GPUs. The code checks for CUDA and quietly falls back to CPU if there
isn't one, so nothing breaks either way, it's just slower.

For disk space, here's roughly what each piece costs so you can plan:

| What | Size |
| --- | ---: |
| All the model weights together | ~2.3 GB |
| VeRi-776 | ~1 GB |
| CityFlowV2 - the 6 cameras we actually use | a few GB |
| CityFlowV2 - the full release, if you grab the whole thing | ~20 GB |
| WILDTRACK - full image release | ~6-10 GB |

So if you just want to poke at the app with VeRi-776, ~4 GB free is plenty. For the
full CityFlowV2 vehicle run, leave yourself ~10 GB. If you really want every dataset
in full, budget 35 GB or more.

Quick sanity check after installing:

```powershell
python --version      # should say 3.10.x to 3.13.x
pip --version
git --version
node --version        # only if you want the frontend
npm --version
```

One note on the commands below: I'm on Windows, so you'll see
`.\.venv\Scripts\activate`. On Linux or macOS that's `source .venv/bin/activate`
instead. Everything runs on all three, I just happened to build it on Windows 11.

---

## 3. Getting the code

```powershell
git clone <your-repo-url> gp
cd gp
```

If you got this as a zip instead, unzip it and `cd` into the folder. Either way,
run everything from the project root - the folder with `README.md`,
`requirements.txt` and `src/` in it.

---

## 4. Setting up Python

Use a virtual environment. It keeps the project's libraries from fighting with
whatever else is on your system Python.

Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Linux / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

That one `pip install` line gets you everything for the pipeline, backend and
scripts. One thing that confused me early on: `torch` and `torchvision` are *not*
listed in `requirements.txt`. That's on purpose, so the GPU build doesn't get
clobbered when running on Kaggle. But `ultralytics`, `timm` and `open-clip-torch`
all depend on torch, so pip pulls in the CPU build for you anyway. For the app and
the light stages that's totally fine.

If you've got an NVIDIA GPU and want it to actually be used, install the CUDA build
of torch first and then the rest:

```powershell
# CUDA 12.1 example - grab the right line for your CUDA version from pytorch.org
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

To check torch can see the GPU:

```powershell
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

If you want the dev tools too (linter, the optional download helper):

```powershell
pip install -e ".[dev,reid]"
```

---

## 5. The Python libraries (and why each one is there)

You don't install these by hand, `requirements.txt` handles it. I'm just listing
them so nothing looks mysterious when pip starts downloading a pile of packages.

| Library | What we use it for |
| --- | --- |
| `numpy`, `scipy` | The numeric math underneath everything |
| `pandas` | Tables, manifests, result summaries |
| `omegaconf` | Loads and merges the YAML configs in `configs/` |
| `jsonschema` | Sanity-checks the model registry against its schema |
| `opencv-python-headless` | Reading video, pulling frames, image ops, CLAHE |
| `Pillow` | Loading the crop images for re-id and the app |
| `ultralytics` | The YOLO detector in stage 1 (this is what drags in torch) |
| `boxmot` | The single-camera trackers (BoT-SORT / DeepOCSORT) |
| `torch`, `torchvision` | The deep-learning backbones (see section 4 for the GPU build) |
| `timm` | Builds the ViT backbones for TransReID and DINOv2 |
| `open-clip-torch` | The CLIP-initialised ViT weights for the re-id model |
| `faiss-cpu` | Fast nearest-neighbour search in stage 3 |
| `networkx` | The cross-camera similarity graph and the connected-component solve |
| `scikit-learn` | PCA whitening and a few metric helpers |
| `motmetrics` | MOT metrics, alongside TrackEval |
| `matplotlib`, `plotly`, `seaborn` | Plots, the bird's-eye view, the 3D sim, reports |
| `streamlit` | The optional local dashboard in `src/apps/` |
| `sentence-transformers` | The natural-language query feature in `src/apps/` |
| `tqdm` | Progress bars |
| `loguru` | Logging |
| `click` | Argument parsing for the runner scripts |
| `rich` | The nice console output you see when the pipeline runs |
| `gdown` | Backup downloads from Google Drive |
| `kaggle` | The Kaggle API client the download scripts use |
| `trackeval` | The HOTA / IDF1 / MOTA scorer in stage 5 |
| `torchreid`, `tensorboard` | OSNet / ResNet-IBN re-id backbones (`tensorboard` is a torchreid import dependency) |
| `pretrainedmodels` | The SENet appearance branch of the CLIP-SENet re-id model |
| `fastapi`, `uvicorn`, `python-multipart` | The backend web service (`uvicorn backend_api:app`) and its file uploads |
| `pydantic` | Request / response schemas in the backend |

The frontend's JavaScript packages (Next.js, React, Radix, Tailwind and so on)
aren't pip's job - those come from `npm install` later.

The YOLO and BoxMOT model files get pulled in by the libraries above and the weight
downloader, so they're not copied into the repo.

---

## 6. Setting up Kaggle

The weights live on a public Kaggle dataset, and VeRi-776 comes from Kaggle too.
Even though both are public, the Kaggle API still wants a token, so you have to do
this once.

1. Make a free account at <https://www.kaggle.com/> if you don't have one.
2. Go to <https://www.kaggle.com/settings>, hit **API**, then **Create New Token**.
   It downloads a file called `kaggle.json`.
3. Drop `kaggle.json` here:
   - Windows: `C:\Users\<your-username>\.kaggle\kaggle.json`
   - Linux / macOS: `~/.kaggle/kaggle.json`
4. On Linux/macOS, tighten the permissions or the CLI nags you:
   ```bash
   chmod 600 ~/.kaggle/kaggle.json
   ```

To confirm it's working:

```powershell
kaggle datasets list -s mtmc-veri776-pipeline-weights
```

If you get a dataset row back instead of an auth error, you're set.

---

## 7. Getting the model weights

The checkpoints are too big to commit, so they aren't in the repo. They all sit in
one public Kaggle dataset (`mrkdagods/mtmc-veri776-pipeline-weights`, CC BY 4.0),
and each file's SHA-256 is pinned in `configs/weights_manifest.yaml`. The
downloader drops each file exactly where the configs expect it and checks the
hash afterward, so you don't have to think about paths.

```powershell
python scripts/download_weights.py --list          # just list the sets and files
python scripts/download_weights.py --set all        # grab everything (~2.3 GB)
python scripts/download_weights.py --set veri        # only the VeRi-776 streams
python scripts/download_weights.py --set all --dry-run   # show what it'd do, no download
python scripts/download_weights.py --set veri --force    # re-pull even if it's already there
```

Here's what each set gives you and where the files end up (sizes are the real
pinned file sizes):

| Set | Size | Files |
| --- | ---: | --- |
| `vehicle-mtmc-14e` | ~1.6 GB | `models/detection/yolo26m.pt` (44 MB), `models/tracker/osnet_x0_25_msmt17.pt` (3 MB), `models/reid/transreid_cityflowv2_best.pth` (347 MB), `models/reid/vehicle_transreid_dinov2_large_cityflowv2_final.pth` (1.2 GB) |
| `vehicle-mtmc-14k` | ~1.7 GB | everything in `14e` plus `models/reid/fastreid_r50_ibn_cityflowv2_final.pth` (100 MB) |
| `person-mtmc` | ~50 MB | `models/person_detection/MultiviewDetector.pth` |
| `veri` | ~685 MB | `models/reid/vehicle_transreid_vit_base_veri776.pth` (347 MB), `models/reid/clipsenet_v6_veri776_best.pth` (371 MB) |
| `all` | ~2.3 GB | all eight files above |

Pick based on what you're doing. Just want to play with re-id and fusion in the
app? `--set veri` is enough. Vehicle MTMC on CityFlowV2? `--set vehicle-mtmc-14e`.
Person MTMC? `--set person-mtmc`. Want it all? `--set all`.

If you'd rather not download them, you can retrain them. The notebooks under
`notebooks/kaggle/` produce every checkpoint (`09_vehicle_reid_cityflowv2/` and
`13_clip_senet_train/` for the re-id models, `12a_wildtrack_mvdetr/` for the person
detector). They're built for Kaggle and save the checkpoints as kernel outputs;
download those and drop them in `models/` under the filenames in the table.
`models/reid/README.md` has the details.

---

## 8. The datasets, their sizes, and where they go

Three datasets to run things, plus one more only if you want to retrain the
weights yourself. They all go under `data/raw/`. None of them are in the repo
because they're big and externally hosted - the only exception is the small
CityFlowV2 ground-truth files, which I left in so evaluation works without you
having to track them down.

A heads-up on sizes, since this caught me out: these are approximate and depend on
exactly how the source packages things, so double-check on the official page. The
one to not get wrong is CityFlowV2. The *full* release is big, around 20 GB, but
this project only ever touches a **6-camera subset** (scenes S01 and S02), which is
just a few GB of video. So if you ever read "CityFlowV2 is ~5 GB" somewhere, that's
the subset, not the whole thing.

| Dataset | What it's for | Rough size | Where it's from |
| --- | --- | --- | --- |
| CityFlowV2 (AIC 2022, Track 1) | Vehicle detection / tracking / MTMC + eval (also the re-id crop training) | full release ~20 GB; we only use a 6-camera subset, a few GB | <https://www.aicitychallenge.org/2022-data-and-evaluation/> (you have to register) |
| WILDTRACK (EPFL CVLAB) | Person ground-plane MTMC + eval | ~6-10 GB (full HD image release) | <https://www.epfl.ch/labs/cvlab/data/data-wildtrack/> |
| VeRi-776 | Single-camera vehicle re-id, training + eval | ~1 GB (49,357 images, 776 IDs, 20 cameras) | Kaggle `abhyudaya12/veri-vehicle-re-identification-dataset` |
| Market-1501 (training only) | Person re-id pre-training, for the Kaggle notebooks | ~150 MB (32,668 images, 1,501 IDs, 6 cameras) | Kaggle (search "Market-1501") |

You only need Market-1501 if you're retraining the person re-id model. With the
downloaded weights you can ignore it.

### 8a. VeRi-776 - the easy one

This is the only one that's fully automatic:

```powershell
python scripts/download_assets.py --datasets
```

It lands at `data/raw/veri776/` and should look like this:

```text
data/raw/veri776/
+-- image_query/        <- query images
+-- image_test/         <- gallery images
+-- image_train/        <- training images
+-- name_query.txt
+-- name_test.txt
+-- ... (other VeRi metadata files)
```

### 8b. CityFlowV2 - the manual one

CityFlowV2 needs you to register on the AI City Challenge site; there's no clean
public Kaggle copy of the whole thing. The full dataset is the ~20 GB one, but you
only need the 6 evaluation cameras.

1. Register and download AI City Challenge 2022 Track 1 from
   <https://www.aicitychallenge.org/2022-data-and-evaluation/>.
2. Put the per-camera folders under `data/raw/cityflowv2/`.

I already left `seqinfo.ini` and `gt/gt.txt` for the six cameras in the repo, so
the scoring works straight away. What you add from the official download is the
`vdo.avi` (and `roi.jpg`) for each one:

```text
data/raw/cityflowv2/
+-- S01_c001/
|   +-- vdo.avi          <- you add this (from the AIC22 download)
|   +-- roi.jpg          <- you add this (the road mask; stage 1 uses it to ignore off-road stuff)
|   +-- seqinfo.ini      <- already in the repo
|   +-- gt/
|       +-- gt.txt       <- already in the repo
+-- S01_c002/  (same four files)
+-- S01_c003/  (same four files)
+-- S02_c006/  (same four files)
+-- S02_c007/  (same four files)
+-- S02_c008/  (same four files)
```

Cameras get picked up automatically from the folder names (`S01_c001` and so on),
so you don't have to keep the whole download around - just these six folders.

### 8c. WILDTRACK - manual and optional

Only bother with this if you want the person pipeline. Grab the WILDTRACK release
from EPFL CVLAB and put it at `data/raw/wildtrack/`:

```text
data/raw/wildtrack/
+-- videos/                  <- the per-camera video (C1..C7)
+-- annotations_positions/   <- WILDTRACK's JSON annotations
+-- calibrations/            <- camera calibration files
+-- manifests/
    +-- ground_truth/        <- the eval ground truth
```

One catch: the WILDTRACK route uses the MVDeTr ground-plane fast path, which reads
a pre-computed detections file at `data/outputs/wildtrack_mvdetr/test.txt`. That
file comes out of the `12a_wildtrack_mvdetr` / `12b` notebooks in
`notebooks/kaggle/`. If you don't have it yet, run those notebooks first, or just
stick to the vehicle side and the app.

---

## 9. Checking everything is in place

```powershell
python scripts/verify_assets.py
```

This prints a table and checks each checkpoint (size and MD5) plus the optional
datasets. `OK` means it's there and the hash matches. The optional datasets only
fail the run if you pass `--strict`; by default only a missing required model file
makes it exit non-zero.

If you'd rather grab everything (weights + public datasets) in one go:

```powershell
python scripts/download_assets.py --all
```

---

## 10. Running the pipeline

The runner is `scripts/run_pipeline.py`. You always pair the base config
(`configs/default.yaml`) with a dataset config from `configs/datasets/`.

```powershell
# Vehicles (CityFlowV2), full seven stages
python scripts/run_pipeline.py --config configs/default.yaml --dataset-config configs/datasets/cityflowv2.yaml

# People (WILDTRACK), via the MVDeTr ground-plane path
python scripts/run_pipeline.py --config configs/default.yaml --dataset-config configs/datasets/wildtrack.yaml
```

A few variants I use a lot:

```powershell
# Dry run - prints the plan and runs nothing. Good first check.
python scripts/run_pipeline.py --config configs/default.yaml --dataset-config configs/datasets/cityflowv2.yaml --dry-run

# Smoke run - first ~10 frames per camera, just to confirm the wiring works.
python scripts/run_pipeline.py --config configs/default.yaml --dataset-config configs/datasets/cityflowv2.yaml --smoke-test

# Only some stages - e.g. redo indexing, matching and scoring.
python scripts/run_pipeline.py --config configs/default.yaml --dataset-config configs/datasets/cityflowv2.yaml --stages 3,4,5

# Override any config value on the command line.
python scripts/run_pipeline.py --config configs/default.yaml --dataset-config configs/datasets/cityflowv2.yaml -o stage1.detector.confidence_threshold=0.3
```

What each stage does, in plain terms:

```text
stage0  Ingestion       Decode the video into frames at a target FPS, run CLAHE/preprocessing
stage1  Tracking        YOLO detection + BoxMOT tracking per camera -> tracklets
stage2  Features        TransReID (+ DINOv2) embeddings, HSV colour histograms, PCA whitening
stage3  Indexing        FAISS index + a little SQLite metadata store
stage4  Association     The cross-camera graph -> connected components -> global IDs
stage5  Evaluation      HOTA / IDF1 / MOTA via TrackEval
stage6  Visualization   Annotated video, bird's-eye view, timeline exports
```

Everything lands in `data/outputs/<run_id>/`, one folder per stage, plus a
`pipeline.log` and the exact config it used.

Honest warning: stages 0 to 2 over a full dataset are heavy and we ran them on
Kaggle. On a laptop CPU they'll work but crawl, so lean on `--smoke-test`, a stage
subset, or the app. If there's no CUDA, the runner prints a warning and keeps going
on CPU.

---

## 11. Using your own footage and cameras

The pipeline isn't hard-coded to CityFlowV2 or WILDTRACK. It's meant to take any
set of cameras, and how much you tell it about the camera layout is up to you.

### Any dataset

`configs/datasets/` already has configs for CityFlowV2, WILDTRACK, VeRi-776,
Market-1501 and a couple more. To run on your own video:

1. Drop your videos under a new root, one folder per camera:
   ```text
   data/raw/my_dataset/
   +-- cam01/video.mp4
   +-- cam02/video.mp4
   +-- cam03/video.mp4
   ```
2. Copy one of the existing dataset configs (say
   `configs/datasets/cityflowv2.yaml`) to
   `configs/datasets/my_dataset.yaml`, then point `dataset.root_dir` and
   `stage0.input_dir` at `data/raw/my_dataset`. Set `type` to `mtmc_vehicle` or
   `mtmc_person` and the detector `classes` to match.
3. Run it:
   ```powershell
   python scripts/run_pipeline.py --config configs/default.yaml --dataset-config configs/datasets/my_dataset.yaml
   ```

### Adding cameras is just adding folders

Every subfolder under the dataset root that has a video in it becomes a camera, and
the folder name becomes that camera's ID. The `cameras: null` line in the config is
what says "find them automatically." So:

- To add a camera, drop in another folder with a video. Next run picks it up, no
  code changes.
- To run only some of them, give an explicit list like `cameras: [cam01, cam03]`
  and it'll skip the rest.

### Camera locations are optional - your call

You don't have to tell the system where the cameras are. With
`camera_transitions: null` (the default in a few of the configs) it just matches on
appearance and uses sensible defaults. That's it, it'll run.

If you *do* know the layout, you can feed it in to get better cross-camera matching:
per-pair transit times under
`stage4.association.spatiotemporal.camera_transitions` (min / max / mean / std in
seconds), and optionally a zone model under `stage4.association.zone_model`. The
CityFlowV2 config has a full worked example of both if you want to copy the shape.
So it's genuinely optional: add the geometry for a better score, or leave it out
and it still works.

---

## 12. Running the app

The app is the FastAPI backend (port 8000) and the Next.js dashboard. There are
ready-made start scripts in the repo, so honestly the easiest thing is to just use
one of those instead of managing two terminals yourself.

First time only, install the frontend packages (the `.bat`/`.sh` scripts will do
this for you if `node_modules` isn't there yet):

```powershell
cd frontend
npm install
cd ..
```

Then pick the launcher for your machine:

| Launcher | OS | What it does |
| --- | --- | --- |
| `python start.py` | any | The one I'd use. Frees ports 8000/3001, starts the backend, waits until it's healthy, starts the frontend on port 3001, streams the logs, and shuts both down cleanly on Ctrl+C. |
| `start.bat` | Windows | Double-click it (or `.\start.bat`). Checks Python and Node, installs deps, warns if `models/` is missing, opens backend + frontend in their own windows. Frontend on port 3000. |
| `./start.sh` | Linux / macOS | Same idea for bash. Backend + frontend on port 3000, Ctrl+C stops both. |

```powershell
# What I usually run, any OS:
python start.py
# then open http://localhost:3001
```

```powershell
# Windows, if you'd rather double-click:
.\start.bat
# then open http://localhost:3000
```

```bash
# Linux / macOS:
chmod +x start.sh   # first time only
./start.sh
# then open http://localhost:3000
```

If you'd rather drive it by hand, it's two terminals.

Terminal 1, the backend (port 8000):

```powershell
.\.venv\Scripts\activate
python -m uvicorn backend_api:app --host 127.0.0.1 --port 8000 --reload
```

Terminal 2, the frontend:

```powershell
cd frontend
npm run dev -- --port 3001     # plain `npm run dev` goes to port 3000
```

The backend accepts both 3000 and 3001, and the frontend already knows to talk to
the backend at `http://localhost:8000/api`, so there's nothing extra to configure.
If you ever need to point it somewhere else, copy `frontend/.env.example` to
`frontend/.env.local` and set `NEXT_PUBLIC_API_URL`.

What's in the dashboard:

| Page | What you do there |
| --- | --- |
| `/` | The landing page with links |
| `/reid` | Upload a query and some gallery images, pick a re-id model, see the ranked matches with scores. Rerank and AQE are supported. |
| `/fusion` | Pick two or more models, set weights that sum to 1, upload images, and compare the fused ranking against each model on its own. |
| `/eval` | Kick off an evaluation job, watch its status, read the results JSON. |

Models you can pick: `veri776_09v_v17_transreid`, `veri776_clipsenet_v6`,
`cityflow_transreid`. Evals: `veri776_transreid`, `veri776_clipsenet`,
`cityflow_transreid`, `veri776_14t_fusion`. Those need the `veri` and
`vehicle-mtmc-14e` weight sets from section 7.

---

## 13. Reproducing the VeRi-776 numbers

These rerun the single-camera re-id results from the report. They need the `veri`
weights (section 7) and the VeRi-776 data (section 8a). Swap in `--device cuda` if
you set up the GPU torch build, otherwise `--device cpu`.

```powershell
# Stream 1: TransReID ViT-B/16 CLIP - should land around 89.97% mAP on its own
python scripts/eval/eval_09v_transreid_veri776.py `
  --checkpoint models/reid/vehicle_transreid_vit_base_veri776.pth `
  --veri-root data/raw/veri776 `
  --device cpu `
  --output-json _repro_out/eval_09v.json

# The two-stream fusion: CLIP-SENet v6 x TransReID - target mAP 0.9330, R1 0.9845
python scripts/eval/eval_14t_fusion_veri776.py `
  --transreid-checkpoint models/reid/vehicle_transreid_vit_base_veri776.pth `
  --clipsenet-checkpoint models/reid/clipsenet_v6_veri776_best.pth `
  --veri-root data/raw/veri776 `
  --device cpu `
  --output-json _repro_out/eval_14t_fusion.json
```

Throw `--smoke` on either one for a quick 50-query x 200-gallery check if you just
want to know it runs.

---

## 14. Running the tests

```powershell
pytest tests/
```

There's also a full app end-to-end check that spins the backend up, hits the
endpoints, and tears it all down:

```powershell
python scripts/test_phase2_e2e.py
```

---

## 15. What results to expect

| Pipeline | Benchmark | Metric | Value |
| --- | --- | --- | --- |
| Vehicle MTMC | CityFlowV2 | MTMC IDF1 | 0.779 |
| Person MTMC | WILDTRACK | IDF1 / MODA | 0.946 / 0.903 |
| Vehicle re-id | VeRi-776 | mAP (TransReID) | 89.97 |
| Vehicle re-id | VeRi-776 | mAP (two-stream fusion) | 93.30 |

The checkpoints behind these and their verified metrics are listed in
`configs/model_registry.yaml` and written up in
[docs/model-cards.md](docs/model-cards.md).

---

## 16. Where everything lives

```text
gp/
+-- configs/                  The YAML configs: default.yaml, datasets/, models/, weights_manifest.yaml
+-- src/                      The seven-stage pipeline, plus serving/training/apps (our code)
+-- backend/                  The FastAPI service (app, routers, services, repositories)
+-- frontend/                 The Next.js ATHAR dashboard (npm install goes here)
+-- scripts/                  The CLI entry points: run_pipeline.py, download_*.py, verify_assets.py, eval/
+-- notebooks/kaggle/         The GPU training / pipeline / verification notebooks (run on Kaggle)
+-- tests/                    The pytest suite
+-- docs/                     architecture.md, dataset_guide.md, model-cards.md
+-- requirements.txt          Python deps (pip install -r)
+-- start.py / start.bat / start.sh   The launchers for backend + frontend
+-- backend_api.py            The ASGI entry point (uvicorn backend_api:app)
|
+-- data/                     (gitignored, you make it) datasets + run outputs
|   +-- raw/
|   |   +-- cityflowv2/       the per-camera S01_c0xx / S02_c0xx folders (you add vdo.avi + roi.jpg)
|   |   +-- veri776/          image_query/, image_test/, image_train/, name_*.txt
|   |   +-- wildtrack/        videos/, annotations_positions/, calibrations/, manifests/
|   +-- outputs/<run_id>/     pipeline results, one folder per stage
|
+-- models/                   (gitignored, the downloader fills it)
    +-- detection/            yolo26m.pt
    +-- tracker/              osnet_x0_25_msmt17.pt
    +-- person_detection/     MultiviewDetector.pth
    +-- reid/                 transreid_*.pth, clipsenet_*.pth, dinov2_*.pth, fastreid_*.pth
```

`data/` and `models/` are in `.gitignore` because they're big and external, which
is exactly why you fetch their contents yourself back in sections 7 and 8.

---

## 17. When things go wrong

A grab-bag of the errors I actually hit and what fixed them:

| What you see | What it usually means |
| --- | --- |
| `503 reid_dependency_missing` in the app | torch/timm aren't installed - activate the venv and `pip install -r requirements.txt` |
| `503 checkpoint_missing` in the app | weights aren't downloaded - `python scripts/download_weights.py --set all` |
| `503 dataset_missing` on an eval | `data/raw/veri776/` isn't there - `python scripts/download_assets.py --datasets` |
| `kaggle: command not found` or a 401 | `kaggle.json` isn't in `~/.kaggle/` or the token's bad - redo section 6 |
| `FileNotFoundError: Input directory does not exist` | the dataset isn't placed - add the per-camera folders under `data/raw/<dataset>/` (sections 8 and 11) |
| `No videos found` in stage 0 | the `vdo.avi` / video files aren't in the camera folders - add them |
| WILDTRACK can't find the detections file | `data/outputs/wildtrack_mvdetr/test.txt` is missing - make it with the `12a/12b` notebook (section 8c) |
| pipeline crawling, "No CUDA device found" | it's on CPU. install the CUDA torch build (section 4) or use `--smoke-test` |
| port 8000 / 3000 / 3001 already in use | `python start.py` frees them for you, or change `--port` in the manual command |
| `npm run dev` won't start | run `npm install` in `frontend/` first; needs Node 18+ |
| a weight fails its checksum | re-pull just that set: `python scripts/download_weights.py --set <name> --force` |

And if you just want the fastest possible "is this thing alive" run - CPU only,
VeRi-776, the app - here's the whole thing start to finish:

```powershell
git clone <your-repo-url> gp
cd gp
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
# drop kaggle.json in C:\Users\<you>\.kaggle\  (section 6)
python scripts/download_weights.py --set veri
python scripts/download_assets.py --datasets
python scripts/verify_assets.py
cd frontend; npm install; cd ..
python start.py
# open http://localhost:3001
```

---

If you want to go deeper, the README has the overview, `docs/architecture.md`
explains how the system is put together, `docs/model-cards.md` covers where the
models came from and their numbers, and `docs/dataset_guide.md` has more on the
datasets.
