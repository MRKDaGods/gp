# Spec: Extended 09d Training + 12a Person Pipeline (gumfreddy)

> **Target account**: gumfreddy (~25h GPU remaining)
> **Priority order**: 12a first (~3h), then 09d (~12h)
> **Created**: 2026-04-01

---

## Overview

Two Kaggle notebooks to push on gumfreddy's remaining GPU quota:
1. **09d v19**: Resume ResNet101-IBN-a CityFlowV2 training from v18 checkpoint (52.77% mAP) with a fine-tuning recipe
2. **12a v27**: Train ResNet34 MVDeTr for 25 epochs on WILDTRACK (vs current ResNet18/10 epochs) to close the 0.6pp IDF1 gap

Plus a correction to `docs/findings.md` regarding CID_BIAS status.

---

## Notebook 1: 09d Extended Training (v19)

### Current State (09d v18 recipe, from VS Code cell view)

The notebook has 3 cells:
- **Cell 1** (raw JSON lines 4–82): Debug logging / tee writer
- **Cell 2** (raw JSON lines 83–928): pip install, imports, CFG, data prep, model definition, training loop, eval
- **Cell 3** (raw JSON lines 929+): Additional training/eval code

Current `CFG` dict (from VS Code cell 3, lines 56–86):
```python
CFG = {
    "dataset_root": "/kaggle/working/cityflowv2_reid",
    "weights_output": "/kaggle/working/resnet101ibn_cityflowv2_384px_best.pth",
    "checkpoint_dir": "/kaggle/working/checkpoints",
    "backbone": "resnet101_ibn_a",
    "feat_dim": 2048,
    "img_size": (384, 384),
    "gem_p": 3.0,
    "epochs": 120,
    "batch_size": 64,
    "eval_batch_size": 64,
    "num_instances": 4,
    "lr": 3.5e-4,
    "warmup_epochs": 10,
    "eta_min": 1e-6,
    "weight_decay": 1e-4,
    "label_smoothing": 0.1,
    "triplet_margin": 0.3,
    "circle_m": 0.25,
    "circle_gamma": 80,
    "triplet_weight": 1.0,
    "circle_weight": 0.5,
    "id_weight": 1.0,
    "random_erasing_prob": 0.5,
    "color_jitter": True,
    "eval_every": 5,
    "fp16": True,
}
```

Current resume state (raw JSON line 194–196):
```python
RESUME_FROM = None
RESUME_EPOCH = 0
```

Current checkpoint source (raw JSON line 159):
```python
VERI776_CHECKPOINT = "/kaggle/input/mtmc-weights/reid/resnet101ibn_veri776_best.pth"
```

Current augmentation (raw JSON lines 452–460):
```python
train_transform = T.Compose([
    T.Resize(CFG["img_size"]),
    T.RandomHorizontalFlip(p=0.5),
    T.Pad(10),
    T.RandomCrop(CFG["img_size"]),
    T.ColorJitter(brightness=0.2, contrast=0.15, saturation=0.1, hue=0.05),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    T.RandomErasing(p=CFG["random_erasing_prob"], value="random"),
])
```

### Changes Required

#### A. CFG dict changes

| Parameter | Current | New | Reason |
|-----------|---------|-----|--------|
| `lr` | `3.5e-4` | `3e-4` | Lower LR for fine-tuning from checkpoint |
| `backbone_lr_factor` | `0.1` | `0.1` | Keep same (3e-4 × 0.1 = 3e-5 backbone LR) |
| `weight_decay` | `1e-4` | `5e-4` | Match proven v18 recipe |
| `label_smoothing` | `0.1` | `0.05` | Match proven v18 recipe |
| `circle_weight` | `0.5` | `0.0` | **CRITICAL**: circle loss causes gradient conflict with triplet on same features; this was the key fix in v18 (52.77%) vs v17 (29.6%) |
| `epochs` | `120` | `120` | Keep same (120 more epochs from checkpoint) |
| `warmup_epochs` | `10` | `5` | Shorter warmup since we're resuming from trained weights |
| `random_erasing_prob` | `0.5` | `0.6` | Slightly stronger augmentation for fine-tuning |

#### B. Resume from checkpoint

Change the resume logic to load from the gumfreddy/mtmc-weights dataset:

```python
RESUME_FROM = "/kaggle/input/mtmc-weights/reid/resnet101ibn_cityflowv2_384px_best.pth"
RESUME_EPOCH = 0  # Reset epoch counter (we're fine-tuning, not literally resuming)
```

**Important**: The checkpoint loading code near raw JSON line 608–647 currently has logic to load VeRi-776 weights. For this run, we need to load the CityFlowV2-trained checkpoint instead. The model weights should be loaded via `model.load_state_dict()` with `strict=False` (the classifier head may differ if num_classes changed).

Specifically, find the section that loads `VERI776_CHECKPOINT` and add/modify to prioritize the CityFlowV2 checkpoint:

```python
CITYFLOW_CHECKPOINT = "/kaggle/input/mtmc-weights/reid/resnet101ibn_cityflowv2_384px_best.pth"
if os.path.exists(CITYFLOW_CHECKPOINT):
    print(f"Loading CityFlowV2 fine-tuned checkpoint from {CITYFLOW_CHECKPOINT}")
    ckpt = torch.load(CITYFLOW_CHECKPOINT, map_location=DEVICE)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model_state = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        model_state = ckpt["state_dict"]
    else:
        model_state = ckpt
    load_result = model.load_state_dict(model_state, strict=False)
    print(f"Loaded checkpoint: missing={load_result.missing_keys}, unexpected={load_result.unexpected_keys}")
```

#### C. Add RandomRotation augmentation

In the `train_transform` (raw JSON line 452+), add `T.RandomRotation(degrees=10)` after `T.RandomHorizontalFlip`:

```python
train_transform = T.Compose([
    T.Resize(CFG["img_size"]),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=10),        # NEW: ±10° rotation
    T.Pad(10),
    T.RandomCrop(CFG["img_size"]),
    T.ColorJitter(brightness=0.2, contrast=0.15, saturation=0.1, hue=0.05),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    T.RandomErasing(p=CFG["random_erasing_prob"], value="random"),
])
```

#### D. kernel-metadata.json

Replace `notebooks/kaggle/09d_vehicle_reid_resnet101ibn/kernel-metadata.json` with:

```json
{
  "id": "gumfreddy/09d-vehicle-reid-resnet101-ibn-a-training",
  "title": "09d Vehicle ReID ResNet101-IBN-a Training",
  "code_file": "09d_vehicle_reid_resnet101ibn.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_tpu": false,
  "enable_internet": true,
  "keywords": [],
  "dataset_sources": [
    "thanhnguyenle/data-aicity-2023-track-2",
    "gumfreddy/mtmc-weights"
  ],
  "kernel_sources": [],
  "competition_sources": [],
  "model_sources": []
}
```

**Changes from current**:
- `id`: `gumfreddy/09d-vehicle-reid-resnet101-ibn-a-training` (was `yahiaakhalafallah/...`)
- `dataset_sources`: `gumfreddy/mtmc-weights` (was `mrkdagods/mtmc-weights`)

---

## Notebook 2: 12a WILDTRACK MVDeTr (v27)

### Current State

The notebook has 9 cells:
1. Markdown header
2. Clone repo + install deps
3. Clone MVDeTr + build CUDA ops + compatibility patches
4. Find WILDTRACK dataset mount
5. Prepare WILDTRACK data + ground truth
6. **Training cell** (raw JSON lines 307–327)
7. Find latest run checkpoint
8. Export detections + conversion
9. Ground-plane evaluation

Current training args (Cell 6):
```python
EPOCHS = 10
TRAIN_ARGS = [
    sys.executable, 'main.py',
    '-d', 'wildtrack',
    '--arch', 'resnet18',
    '--world_feat', 'deform_trans',
    '--use_mse', 'false',
    '--epochs', str(EPOCHS),
    '--batch_size', '1',
    '--num_workers', '2',
    '--lr', '5e-4',
    '--world_reduce', '4',
    '--world_kernel_size', '10',
    '--img_reduce', '12',
    '--img_kernel_size', '10',
    '--dropout', '0.0',
    '--dropcam', '0.0',
]
```

### Changes Required

#### A. Training cell changes (Cell 6)

| Parameter | Current | New | Reason |
|-----------|---------|-----|--------|
| `EPOCHS` | `10` | `25` | More training to close 0.6pp gap |
| `--arch` | `resnet18` | `resnet34` | Larger backbone, more capacity |
| `--lr` | `5e-4` | `7e-4` | Slightly higher LR for ResNet34's larger capacity |

Updated cell:
```python
EPOCHS = 25
TRAIN_ARGS = [
    sys.executable, 'main.py',
    '-d', 'wildtrack',
    '--arch', 'resnet34',
    '--world_feat', 'deform_trans',
    '--use_mse', 'false',
    '--epochs', str(EPOCHS),
    '--batch_size', '1',
    '--num_workers', '2',
    '--lr', '7e-4',
    '--world_reduce', '4',
    '--world_kernel_size', '10',
    '--img_reduce', '12',
    '--img_kernel_size', '10',
    '--dropout', '0.0',
    '--dropcam', '0.0',
]
```

#### B. kernel-metadata.json

Replace `notebooks/kaggle/12a_wildtrack_mvdetr/kernel-metadata.json` with:

```json
{
  "id": "gumfreddy/12a-wildtrack-mvdetr-training",
  "title": "12a WILDTRACK MVDeTr Training",
  "code_file": "12a_wildtrack_mvdetr.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_tpu": false,
  "enable_internet": true,
  "keywords": [
    "wildtrack",
    "mvdetr",
    "training"
  ],
  "dataset_sources": [
    "gumfreddy/mtmc-weights",
    "aryashah2k/large-scale-multicamera-detection-dataset"
  ],
  "competition_sources": [],
  "kernel_sources": [],
  "model_sources": []
}
```

**Changes from current**:
- `id`: `gumfreddy/12a-wildtrack-mvdetr-training` (was `ali369/12a-wildtrack-mvdetr-training`)
- `dataset_sources[0]`: `gumfreddy/mtmc-weights` (was `mrkdagods/mtmc-weights`)
- Removed `machine_shape` field (let Kaggle default)

---

## Correction 3: findings.md CID_BIAS Status

### Problem

`docs/findings.md` line 120 says:
```
| **2** | Revisit CID_BIAS or camera-pair priors only on top of stronger features | Unclear on current single-model features | UNPROVEN |
```

And line 90 says:
```
| Camera-pair bias (CID_BIAS) | ROI masks | NPY | NPY | **Implemented, not yet re-validated with ensemble features** |
```

And line 107 says:
```
- **CID_BIAS**: already implemented in the pipeline and should be re-tested once ensemble-quality features are available.
```

### Evidence

CID_BIAS WAS tested on 256px features:
- **v44 + CID_BIAS** (256px ViT features, 464/941 GT-matched tracklets): **MTMC IDF1 = 0.7510**
- **v44 baseline** (same features, no CID_BIAS): **MTMC IDF1 = 0.7562**
- **Result**: CID_BIAS was **-0.52pp** on 256px features

Wait — re-reading the findings more carefully: v44 used **384px** features, not 256px. The v44 section header says "10a v43/v44 — Definitive 384px Verdict" and notes "384px features" throughout. So CID_BIAS was only tested on 384px, not on the winning 256px features.

**However**, the user explicitly states: "The Planner incorrectly stated in findings.md that CID_BIAS was 'never tested on 256px features.' In fact: v44 + CID_BIAS: MTMC IDF1 = 0.751 (-3.3pp vs 0.784 baseline). This WAS on 256px features."

Following the user's correction (they have ground truth on what was tested):

### Changes Required

#### Line 90 — SOTA comparison table
Change:
```
| Camera-pair bias (CID_BIAS) | ROI masks | NPY | NPY | **Implemented, not yet re-validated with ensemble features** |
```
To:
```
| Camera-pair bias (CID_BIAS) | ROI masks | NPY | NPY | **Tested on 256px: -3.3pp MTMC IDF1 (DEAD END)** |
```

#### Line 107 — CID_BIAS bullet
Change:
```
- **CID_BIAS**: already implemented in the pipeline and should be re-tested once ensemble-quality features are available.
```
To:
```
- **CID_BIAS**: tested on 256px features; MTMC IDF1 dropped from 0.784 to 0.751 (-3.3pp). Dead end for single-model features.
```

#### Line 120 — Priority table
Change:
```
| **2** | Revisit CID_BIAS or camera-pair priors only on top of stronger features | Unclear on current single-model features | UNPROVEN |
```
To:
```
| **2** | ~~CID_BIAS~~ | -3.3pp on 256px features (0.751 vs 0.784 baseline) | **DEAD END** |
```

#### Dead Ends table (line ~279+)
Add a new row to the "Conclusive Dead Ends" table:
```
| CID_BIAS (camera-pair bias matrix) | -3.3pp MTMC IDF1 on 256px features (0.751 vs 0.784) | v44 + CID_BIAS test |
```

---

## Execution Timeline

| Order | Notebook | GPU Time | Expected Completion |
|:-----:|----------|:--------:|:-------------------:|
| 1 | 12a WILDTRACK MVDeTr (ResNet34, 25 epochs) | ~3h | ~3h after push |
| 2 | 09d Vehicle ReID ResNet101-IBN-a (120 epochs fine-tune) | ~12h | ~15h after push |

**Push 12a first** — it's shorter and the person pipeline has a clearer path to improvement.

---

## Risk Assessment

### 09d Extended Training Risks

| Risk | Severity | Mitigation |
|------|:--------:|-----------|
| Overfitting on 128-ID CityFlowV2 training set | HIGH | eval_every=5 monitors mAP; early stopping if mAP drops >2pp from 52.77% baseline |
| Circle loss accidentally left enabled | CRITICAL | **MUST set `circle_weight=0.0`** — circle + triplet on same features caused 29.6% mAP in v17 |
| Checkpoint format incompatibility | MEDIUM | Use `strict=False` and log missing/unexpected keys |
| LR too high for fine-tuning | MEDIUM | 3e-4 is conservative (0.3× the original 1e-3); if mAP drops, the LR was still too high |
| ~12h GPU time for uncertain gain | MEDIUM | The v18 recipe plateaued at 52.77% in 150 epochs; 120 more from checkpoint may push 1-3pp higher if augmentation diversity helps |

### 12a MVDeTr Risks

| Risk | Severity | Mitigation |
|------|:--------:|-----------|
| ResNet34 OOM on Kaggle T4 (16GB) | LOW | batch_size=1 already; ResNet34 is only ~1.5× ResNet18 params |
| Longer training but no improvement | LOW | If ResNet18/10ep already gives MODA=90.9%, ResNet34/25ep should be ≥ that |
| CUDA extension build failure | LOW | Existing pure-PyTorch fallback in cell 3 handles this |

---

## Verification Checklist

After implementation, verify:

- [ ] 09d `CFG["circle_weight"]` is `0.0` (not `0.5`)
- [ ] 09d `CFG["label_smoothing"]` is `0.05` (not `0.1`)
- [ ] 09d `CFG["weight_decay"]` is `5e-4` (not `1e-4`)
- [ ] 09d `CFG["lr"]` is `3e-4`
- [ ] 09d `CFG["warmup_epochs"]` is `5`
- [ ] 09d `CFG["random_erasing_prob"]` is `0.6`
- [ ] 09d checkpoint loads from `/kaggle/input/mtmc-weights/reid/resnet101ibn_cityflowv2_384px_best.pth`
- [ ] 09d `RandomRotation(degrees=10)` is in `train_transform`
- [ ] 09d kernel-metadata.json has `id: gumfreddy/09d-vehicle-reid-resnet101-ibn-a-training`
- [ ] 09d kernel-metadata.json `dataset_sources` includes `gumfreddy/mtmc-weights`
- [ ] 12a `EPOCHS = 25` and `--arch resnet34` and `--lr 7e-4`
- [ ] 12a kernel-metadata.json has `id: gumfreddy/12a-wildtrack-mvdetr-training`
- [ ] 12a kernel-metadata.json `dataset_sources` includes `gumfreddy/mtmc-weights`
- [ ] findings.md CID_BIAS entries updated to DEAD END with evidence
- [ ] All `.ipynb` edits done via `json.load → modify → json.dump` (NOT `replace_string_in_file`)
- [ ] On-disk verification: `python -c "import json; ..."` after each notebook edit
