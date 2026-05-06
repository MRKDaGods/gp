# CLIP-SENet CityFlowV2 Score Fusion

## Problem

Integrate CLIP-SENet v6 features into Stage 2 of the CityFlowV2 pipeline alongside the existing TransReID 09v features, then evaluate whether score-level fusion improves MTMC IDF1 over the current best reproducible setup.

## Plan

- Build Kaggle kernel `13c_clip_senet_cityflow_features`, mirroring the existing 10a Stage 2 feature extraction kernel under `notebooks/kaggle/10a_stages012/`.
- Use the CLIP-SENet v6 `best.pth` checkpoint as the feature extractor, chained from `yahiaakhalafallah/13-clip-senet-train`.
- Output per-detection BNNeck-normalized embeddings for all CityFlowV2 detections.
- Run Stage 4 association with score-level fusion between TransReID 09v features and CLIP-SENet features.
- Sweep `w_clip_senet` over `[0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]`.

## Expected Outcome

- Current best fusion baseline: IDF1 0.7703 from 10c v15 using TransReID 09v alone.
- Hypothesis: pairing two competitive and more diverse vehicle ReID backbones, TransReID 89.97 mAP and CLIP-SENet 91.54 mAP, could add +1-3pp MTMC IDF1.

## Risks

- CLIP-SENet may produce features in a different geometry, so PCA whitening calibration may need re-tuning.
- The 320x320 input size is slower than the current 256x256 feature extraction path.

## Required Inputs

- CityFlowV2 dataset on Kaggle: `thanhnguyenle/data-aicity-2023-track-2`.
- Existing Stage 1 detections and tracking output: use the 10a chain source/output `yahiaakhalafallah/mtmc-10a-stages-0-2`; the 10a checkpoint dataset is `yahiaakhalafallah/mtmc-10a-checkpoints`.
- CLIP-SENet v6 `best.pth` checkpoint: chain from `yahiaakhalafallah/13-clip-senet-train`.

## Acceptance Criterion

Find a score-fusion configuration that beats 0.7703 MTMC IDF1.