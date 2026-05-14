# Ensemble Deployment Spec — ResNet101-IBN-a + TransReID ViT

> **Status**: Waiting for 09d v14 training completion
> **Expected impact**: +1-2pp IDF1 (78.4% → 79.4-80.4%)
> **Baseline**: v80 best, IDF1=78.4% (single TransReID ViT, min_hits=2)

---

## 0. Prerequisites

- [ ] 09d v14 training complete on Kaggle (ResNet101-IBN-a, 384x384, 60 epochs)
- [ ] Download `resnet101ibn_cityflowv2_384px_best.pth` from 09d output
- [ ] Verify checkpoint structure: keys start with `conv1.`, `layer1.`, `pool.`, `bottleneck.`

---

## 1. Upload Weights to mtmc-weights Dataset

The 10a notebook copies weights from `mrkdagods/mtmc-weights` via `shutil.copytree`.
The weight must exist at: `reid/resnet101ibn_cityflowv2_384px_best.pth` inside the dataset.

### Steps:
1. Download trained weights from 09d v14 output
2. Copy to local `models/reid/` directory
3. Verify checkpoint integrity (keys, mAP)
4. Upload: `cd models && kaggle datasets version -p . -m "Add ResNet101-IBN-a weights" --dir-mode zip`

**Important**: The dataset source `mrkdagods/mtmc-weights` is already in
`notebooks/kaggle/10a_stages012/kernel-metadata.json` — no metadata change needed.

---

## 2. Changes to 10a Notebook

**File**: `notebooks/kaggle/10a_stages012/mtmc-10a-stages-0-2-tracking-reid-features.ipynb`

### Change 1: Enable vehicle2 (Line 403)

**Old**:
```python
    "--override", "stage2.reid.vehicle2.enabled=false",
```

**New**:
```python
    "--override", "stage2.reid.vehicle2.enabled=true",
```

That's the ONLY change needed for 10a. Everything else is already configured:
- `cityflowv2.yaml` has vehicle2 config
- Weights are copied from mtmc-weights dataset
- Secondary PCA is auto-fitted
- `embeddings_secondary.npy` is auto-saved

---

## 3. Changes to 10c Notebook

**File**: `notebooks/kaggle/10c_stages45/mtmc-10c-stages-4-5-association-eval.ipynb`

### CRITICAL: 10c params are severely out-of-date vs v80 best

| Parameter | 10c Main Run | v80 Best |
|-----------|:------------:|:--------:|
| algorithm | connected_components | conflict_free_cc |
| appearance_weight | 0.75 | 0.70 |
| fic_reg | 3.0 | 0.1 |
| camera_bias | true | false |
| zone_model | true | false |
| gallery_expansion | not set | 0.50 |
| temporal_overlap | disabled | 0.05 |
| length_weight_power | not set | 0.3 |
| intra_merge_thresh | 0.75 | 0.80 |
| intra_merge_gap | 60 | 30 |
| AQE_K | 2 | 3 |

### Change 2: Update params cell (approx lines 183-228)

Replace params with v80-aligned values + ensemble enabled:
```python
AQE_K             = 3
SIM_THRESH        = 0.53
ALGORITHM         = "conflict_free_cc"
APPEARANCE_WEIGHT = 0.70
HSV_WEIGHT        = 0.0
ST_WEIGHT         = round(1.0 - APPEARANCE_WEIGHT - HSV_WEIGHT, 4)
BRIDGE_PRUNE      = 0.0
MAX_COMP_SIZE     = 12
INTRA_MERGE       = True
INTRA_MERGE_THRESH = 0.80
INTRA_MERGE_GAP   = 30
FUSION_WEIGHT     = 0.3   # Experiment values: 0.0, 0.2, 0.3, 0.4
CAMERA_BIAS       = False
ZONE_MODEL        = False
HIERARCHICAL      = False
MTMC_ONLY         = False
```

### Change 3: Fix FIC reg in main run cmd (line 258)

**Old**: `"stage4.association.fic.regularisation=3.0"`
**New**: `"stage4.association.fic.regularisation=0.1"`

### Change 4: Add missing v80 overrides to main run cmd

Add after the intra_camera_merge overrides:
```python
    "--override", "stage4.association.gallery_expansion.enabled=true",
    "--override", "stage4.association.gallery_expansion.threshold=0.50",
    "--override", "stage4.association.weights.length_weight_power=0.3",
    "--override", "stage4.association.temporal_overlap.enabled=true",
    "--override", "stage4.association.temporal_overlap.bonus=0.05",
    "--override", "stage4.association.temporal_overlap.max_mean_time=5.0",
```

---

## 4. Experiment Plan

### Phase 1: Baseline Confirmation
10a (vehicle2=true) → 10b → 10c with v80 params, FUSION_WEIGHT=0.0
Expected: IDF1 ≈ 78.4%

### Phase 2: Fusion Weight Sweep (10c only, 3 runs)
| Run | FUSION_WEIGHT | Rationale |
|-----|:------------:|-----------|
| A | 0.2 | Conservative |
| B | 0.3 | Moderate |
| C | 0.4 | Aggressive |

### Phase 3: Combine with v31 gallery_thresh=0.48

### Phase 4: Fine-tune sim_thresh, fic_reg if ensemble improves

---

## 5. Risk Assessment

- **Secondary model bad (mAP<30%)**: Set FUSION_WEIGHT=0.0 → safe fallback
- **Secondary hurts IDF1**: Keep single model, no 10a/10b re-run needed
- **GPU OOM**: Sequential processing, ~6GB per model on P100 (16GB)
- **Key remapping fails**: Check 09d checkpoint structure first
- **10a time limit**: ~55min with ensemble (well within 9h limit)

---

## 6. Kaggle Push Sequence

```bash
# 1. Upload weights
cd models && kaggle datasets version -p . -m "Add ResNet101-IBN-a weights" --dir-mode zip

# 2. Push 10a (GPU, ~55 min)
kaggle kernels push -p notebooks/kaggle/10a_stages012/

# 3. Push 10b (CPU, ~5 min) after 10a completes
kaggle kernels push -p notebooks/kaggle/10b_stage3/

# 4. Push 10c (CPU, ~6 min) after 10b completes
kaggle kernels push -p notebooks/kaggle/10c_stages45/
```
