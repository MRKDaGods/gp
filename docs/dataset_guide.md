# Dataset Guide

## Overview
Datasets for training ReID models (on Kaggle) and evaluating the full MTMC pipeline (locally).

## Person Re-Identification

### Market-1501 (Primary)
- **IDs**: 1,501 identities
- **Images**: 32,668 images
- **Cameras**: 6
- **Source**: Kaggle (search "Market-1501")
- **Structure**: bounding_box_train/, bounding_box_test/, query/
- **Filename format**: `XXXX_cYsZ_NNNNNN_NN.jpg` (XXXX=person ID, Y=camera)
- **Used for**: Training OSNet-x1.0 and ResNet50-IBN-a person ReID
- **Target metrics**: mAP >= 85%, Rank-1 >= 94%

### MSMT17 (Secondary, hardest)
- **IDs**: 4,101 identities
- **Images**: 126,441 images
- **Cameras**: 15
- **Source**: Apply for access (harder to get)
- **Used for**: Advanced evaluation / transfer learning
- **Note**: Much harder than Market-1501 due to more cameras and complex scenes

### DukeMTMC-reID — DO NOT USE
- Retracted for ethical concerns (privacy violations)
- Any use in papers will be flagged by reviewers

## Vehicle Re-Identification

### VeRi-776 (Primary)
- **IDs**: 776 vehicle identities
- **Images**: 49,357 images
- **Cameras**: 20
- **Source**: Kaggle (search "VeRi-776") or request from authors
- **Structure**: image_train/, image_test/, image_query/
- **Filename format**: `XXXX_cYYY_NNNNN.jpg` (XXXX=vehicle ID, YYY=camera)
- **Used for**: Training vehicle ReID models
- **Target metrics**: mAP >= 78%, Rank-1 >= 95%

### AI City Challenge 2023 Track 2
- **Task**: Cityflow-NL — vehicle retrieval with natural language
- **Source**: Available on Kaggle
- **Used for**: Full pipeline evaluation
- **Note**: Also useful for NL query application development

## Tracking Benchmarks

### MOT17 (Tracker Validation)
- **Task**: Pedestrian multi-object tracking
- **Source**: https://motchallenge.net/
- **Used for**: Validating BoT-SORT / Deep-OCSORT performance on single-camera
- **Metrics**: MOTA, IDF1, HOTA

## Multi-Camera End-to-End

### AI City Challenge 2023 (Vehicle MTMC)
- Full vehicle multi-camera dataset
- Multiple cameras covering city intersections
- Ground truth for global trajectory evaluation

### EPFL Multi-Camera (Pedestrian MTMC)
- 4 cameras, office/campus setting
- Small enough (~1GB) to upload to Kaggle
- Good for integration testing

## Dataset Preparation

### Using prepare_dataset.py
```bash
# Market-1501
python scripts/prepare_dataset.py --dataset market1501 --root data/raw/market1501

# VeRi-776
python scripts/prepare_dataset.py --dataset veri776 --root data/raw/veri776

# AIC2023 (reads videos directly)
python scripts/prepare_dataset.py --dataset aic2023 --root data/raw/aic2023
```

This creates CSV manifests in `{root}/manifests/` with columns: image_path, identity_id, camera_id.

### Directory Structure
```
data/
├── raw/
│   ├── market1501/
│   │   ├── bounding_box_train/
│   │   ├── bounding_box_test/
│   │   ├── query/
│   │   └── manifests/        ← generated
│   ├── veri776/
│   │   ├── image_train/
│   │   ├── image_test/
│   │   ├── image_query/
│   │   └── manifests/        ← generated
│   └── aic2023/
│       └── ...
├── processed/                 ← pipeline outputs
└── models/                    ← trained weights
```

## Kaggle Upload Strategy
1. Market-1501 and VeRi-776 are already on Kaggle as datasets — just add to notebook
2. EPFL Multi-Camera: download locally, zip, upload as Kaggle dataset (~1GB)
3. AIC2023: check existing Kaggle datasets, may already be available
4. Model weights after training: download from Kaggle notebook output
