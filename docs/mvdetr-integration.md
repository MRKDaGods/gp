# MVDeTr Integration for WILDTRACK

## Architecture Overview

MVDeTr replaces the repo's person pipeline for overlapping-camera datasets such as WILDTRACK.

- Input: 7 synchronized camera views per timestep.
- Backbone: ResNet-18 image encoder.
- Fusion: per-view image features are projected into a shared ground-plane feature map.
- World model: a deformable transformer (`--world_feat deform_trans`) aggregates multi-view evidence on the ground plane.
- Output: per-frame ground-plane detections that can be evaluated directly with ground-plane MODA / IDF1.

The upstream implementation uses WILDTRACK's native calibration files and native world-coordinate mapping:

- Dataset root: `~/Data/Wildtrack`
- Calibration files: `calibrations/intrinsic_zero/*.xml` and `calibrations/extrinsic/*.xml`
- Native grid: 480 x 1440 cells with 2.5 cm spacing
- World-coordinate origin: `(-300 cm, -900 cm)`

## How It Differs From the Current Pipeline

Current repo flow for people:

1. Detect independently in each camera.
2. Track independently in each camera.
3. Associate tracklets across cameras.

MVDeTr flow:

1. Read all seven camera images at the same timestamp.
2. Project per-camera features into a shared world grid using calibration.
3. Detect once on the ground plane.
4. Track in world coordinates.
5. Optionally project tracked ground-plane positions back into each camera view.

This avoids the WILDTRACK failure mode where one real person becomes up to seven independent single-camera tracks.

## Local Data and Calibration Findings

The workspace already contains the required WILDTRACK assets under `data/raw/wildtrack`:

- `videos/C1.mp4` through `videos/C7.mp4`
- `annotations_positions/*.json`
- `calibrations/intrinsic_zero/intr_CVLab1.xml` ... `intr_IDIAP3.xml`
- `calibrations/extrinsic/extr_CVLab1.xml` ... `extr_IDIAP3.xml`
- `manifests/ground_truth/`
- `manifests/roi_polygons.json`

The calibration format already matches both our existing helpers and the upstream MVDeTr loader:

- Intrinsics: OpenCV XML with `camera_matrix`
- Extrinsics: OpenCV XML with plain-text `rvec` and `tvec`
- Camera mapping: `CVLab1..4 -> C1..C4`, `IDIAP1..3 -> C5..C7`

The repo already had reusable WILDTRACK-specific geometry and evaluation code:

- `src/core/wildtrack_calibration.py`
- `src/stage5_evaluation/ground_plane_eval.py`

## Kaggle Training Notebook

New notebook folder:

- `notebooks/kaggle/12a_wildtrack_mvdetr/`

Files:

- `12a_wildtrack_mvdetr.ipynb`
- `kernel-metadata.json`

Notebook behavior:

1. Clones this repo.
2. Clones `hou-yz/MVDeTr`.
3. Builds the deformable-attention CUDA op.
4. Locates a mounted WILDTRACK Kaggle dataset, falling back to an official-source download attempt.
5. Creates the upstream-required dataset layout at `~/Data/Wildtrack`.
6. Trains MVDeTr for 10 epochs with ResNet-18 and `deform_trans` world fusion.
7. Reuses the produced `test.txt` ground-plane detections.
8. Converts detections into:
   - `ground_plane_tracks.csv`
   - `ground_plane_tracks.json`
   - projected per-camera tracklets
   - `global_trajectories.json`
9. Runs the repo's ground-plane evaluation.

Referenced Kaggle datasets:

- `mrkdagods/mtmc-weights`
- `aryashah2k/large-scale-multicamera-detection-dataset`

I could not verify Kaggle availability live because the local Kaggle CLI is unauthenticated, so the WILDTRACK dataset slug is taken from the repo's existing working notebooks.

## Integration Module

New module:

- `src/stage_wildtrack_mvdetr/__init__.py`
- `src/stage_wildtrack_mvdetr/pipeline.py`

Main entry point:

```python
from src.stage_wildtrack_mvdetr.pipeline import run_stage_wildtrack_mvdetr

tracklets_by_camera, trajectories = run_stage_wildtrack_mvdetr(
    detections_path="/path/to/test.txt",
    calibration_dir="data/raw/wildtrack/calibrations",
    output_dir="data/outputs/wildtrack_mvdetr",
)
```

What it does:

- Loads MVDeTr `test.txt` detections or equivalent CSV / JSON exports.
- Normalizes upstream WILDTRACK frame numbers from `0, 5, 10, ...` to repo frame IDs `0, 1, 2, ...`.
- Converts ground-grid coordinates into world coordinates in centimeters.
- Tracks detections with Hungarian matching on the ground plane.
- Projects each tracked world position back into every camera using the local calibration XMLs.
- Synthesizes per-camera person boxes from projected foot/head points.
- Saves repo-compatible `Tracklet` JSON files plus `global_trajectories.json` for Stage 5.

## Training Instructions

On Kaggle:

1. Attach `mrkdagods/mtmc-weights`.
2. Attach `aryashah2k/large-scale-multicamera-detection-dataset`.
3. Push `notebooks/kaggle/12a_wildtrack_mvdetr/`.
4. Run the notebook.

The notebook will create:

- an upstream MVDeTr log directory under `MVDeTr/logs/wildtrack/`
- a converted export directory under `data/outputs/wildtrack_mvdetr/`

## Inference Instructions

If training is already complete, run only the later notebook cells or call the integration stage directly:

```python
from src.stage_wildtrack_mvdetr.pipeline import run_stage_wildtrack_mvdetr

run_stage_wildtrack_mvdetr(
    detections_path="MVDeTr/logs/wildtrack/<run_name>/test.txt",
    calibration_dir="data/raw/wildtrack/calibrations",
    output_dir="data/outputs/wildtrack_mvdetr/<run_name>",
)
```

## Evaluation and Expected Performance

Published MVDeTr baseline on WILDTRACK: about 91.5% MODA.

Inside this repo, the correct primary metric for this integration is the existing ground-plane evaluation in `src/stage5_evaluation/ground_plane_eval.py`, not the current per-camera person MTMC baseline. The new stage produces `GlobalTrajectory` objects so it can feed directly into Stage 5.

## Integration Points With the Existing Pipeline

- Reuses the local WILDTRACK calibration loader.
- Reuses `Tracklet` and `GlobalTrajectory` data contracts.
- Reuses Stage 5 ground-plane evaluation.
- Bypasses the current WILDTRACK per-camera Stage 1 to Stage 4 stack for overlapping-camera datasets.

## Known Constraints

- The upstream deformable-attention op must compile successfully on Kaggle GPU runtime.
- The local environment should not be used for training.
- The local Kaggle CLI is not authenticated, so dataset-slug verification was indirect.
- Projected per-camera boxes are synthesized from world-space foot and head projections; they are suitable for compatibility and visualization, while ground-plane evaluation remains the primary benchmark.