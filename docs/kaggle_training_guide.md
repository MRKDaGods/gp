# Kaggle Training Guide

## Overview
All ReID model training and large dataset operations run on Kaggle notebooks with GPU acceleration. This guide walks through the process.

## Prerequisites
- Kaggle account with phone verification (for GPU access)
- GPU quota: ~30 hours/week (P100 or T4)
- Datasets added to Kaggle (Market-1501, VeRi-776)

## Notebook Overview

| Notebook | Purpose | GPU Time | Output |
|---|---|---|---|
| 01_dataset_preparation | Download & prepare datasets | ~30 min (no GPU needed) | CSV manifests, statistics |
| 02_person_reid_training | Train person ReID models | ~3-4 hours | OSNet + ResNet50-IBN weights |
| 03_vehicle_reid_training | Train vehicle ReID models | ~2-3 hours | OSNet + ResNet50-IBN weights |
| 04_advanced_reid_training | TransReID (stretch goal) | ~6-8 hours | ViT ReID weights |

Total GPU time: ~6-8 hours for core models (notebooks 1-3).

## Step-by-Step Instructions

### Step 1: Upload Notebooks
1. Go to kaggle.com → Your Work → New Notebook
2. Upload each `.ipynb` from `notebooks/kaggle/`
3. Or copy-paste cell contents into new notebooks

### Step 2: Add Datasets
For each training notebook:
1. Click "Add Data" in the right sidebar
2. Search for "Market-1501" and add it
3. Search for "VeRi-776" and add it
4. Data appears at `/kaggle/input/`

### Step 3: Configure GPU
1. Settings → Accelerator → GPU T4 x2 (or P100)
2. Settings → Internet → Enable (needed for pip installs)
3. Settings → Persistence → Files (for saving models)

### Step 4: Run Notebooks in Order
1. **Notebook 01** (CPU only): Prepares dataset manifests and statistics
2. **Notebook 02**: Trains person ReID models on Market-1501
3. **Notebook 03**: Trains vehicle ReID models on VeRi-776
4. **Notebook 04** (optional): TransReID advanced training

### Step 5: Download Trained Weights
After each training notebook completes:
1. Navigate to the Output tab
2. Download `.pth.tar` model files
3. Place in local `models/reid/` directory

## Training Configuration

### Person ReID (Notebook 02)
- **Model A**: ResNet50-IBN-a
  - Embedding: 2048-dim
  - Epochs: 120
  - Optimizer: Adam, lr=3.5e-4
  - Loss: Triplet + Cross-Entropy + Label Smoothing
  - Scheduler: Cosine annealing with warmup (10 epochs)
  - Batch: 32 (8 IDs × 4 instances)

- **Model B**: OSNet-x1.0
  - Embedding: 512-dim
  - Epochs: 150
  - Same loss and optimizer setup
  - Lighter, faster inference

### Vehicle ReID (Notebook 03)
- Same architectures on VeRi-776
- Input: 224×224 (vehicles are squarer than people)
- Augmentation: random flip, random erasing, color jitter

### Target Metrics
| Dataset | Model | mAP Target | Rank-1 Target |
|---|---|---|---|
| Market-1501 | ResNet50-IBN | ≥85% | ≥94% |
| Market-1501 | OSNet-x1.0 | ≥85% | ≥94% |
| VeRi-776 | ResNet50-IBN | ≥78% | ≥95% |
| VeRi-776 | OSNet-x1.0 | ≥78% | ≥95% |

## Tips for Kaggle Sessions
- Save checkpoints every 10 epochs in case session gets killed
- Use Kaggle's "Save & Run All" for unattended execution
- Monitor GPU utilization (should be >80%)
- If session dies, resume from last checkpoint
- Total training for all core models fits within 1 week's GPU quota

## Model File Naming Convention
```
models/reid/
├── osnet_x1_0_market1501.pth.tar        # Person ReID (primary)
├── resnet50_ibn_a_market1501.pth.tar     # Person ReID (secondary)
├── osnet_x1_0_veri776.pth.tar            # Vehicle ReID (primary)
└── resnet50_ibn_a_veri776.pth.tar        # Vehicle ReID (secondary)
```
