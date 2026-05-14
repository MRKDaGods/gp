# 10a Notebook — 3-Model Ensemble Modifications

## Current State

10a (`notebooks/kaggle/10a_stages012/mtmc-10a-stages-0-2-tracking-reid-features.ipynb`) runs stages 0-2 on Kaggle GPU. Currently:
- **Vehicle1 (primary)**: TransReID ViT-B/16 CLIP, 384×384, 768D → PCA 384D
- **Vehicle2 (secondary)**: ResNet101-IBN-a — **DISABLED** in 10a via `stage2.reid.vehicle2.enabled=false`
- **Vehicle3 (tertiary)**: Not configured

The stage2 pipeline code (`src/stage2_features/pipeline.py`) already fully supports vehicle3 extraction. No pipeline code changes needed.

## Goal

Enable 3-model feature extraction: primary (TransReID ViT) + secondary (ResNet101-IBN-a DMT) + tertiary (ResNeXt101-IBN-a DMT).

## Model Weights Source

| Model | Training Notebook | Output Checkpoint | Kaggle Account |
|-------|------------------|-------------------|----------------|
| Vehicle2 (ResNet101-IBN-a DMT) | `09g_resnet101ibn_dmt` | `resnet101ibn_dmt_best.pth` | mrkdagods |
| Vehicle3 (ResNeXt101-IBN-a DMT) | `09h_resnext101ibn_dmt` | `resnext101ibn_dmt_best.pth` | gumfreddy |

### Weight Delivery Options

**Option A (Recommended): Upload to mtmc-weights dataset**
- Download 09g/09h outputs, rename to standard names, upload to `mrkdagods/mtmc-weights`
- Standard names: `resnet101ibn_dmt_best.pth`, `resnext101ibn_dmt_best.pth`
- No metadata changes to 10a needed for data sources

**Option B: Add as kernel_sources**
- Add `mrkdagods/09g-resnet101-ibn-a-dmt-cityflowv2` and `gumfreddy/09h-resnext101-ibn-a-dmt-cityflowv2` to 10a's `kernel_sources`
- Outputs mount at `/kaggle/input/09g-resnet101-ibn-a-dmt-cityflowv2/` etc.
- Requires extra copy logic in the weights cell

## Changes Required

### 1. `kernel-metadata.json` — Dataset Sources

If Option B: add kernel_sources entries. If Option A: no changes.

### 2. Cell 8 (Model Weights Copy — lines 129-188)

**Add to ESSENTIAL list** (line ~176):
```python
ESSENTIAL = [
    "models/detection/yolo26m.pt",
    "models/reid/transreid_cityflowv2_best.pth",
    "models/reid/resnet101ibn_veri776_best.pth",
    "models/reid/resnet101ibn_dmt_best.pth",      # NEW: 09g DMT vehicle2
    "models/reid/resnext101ibn_dmt_best.pth",      # NEW: 09h DMT vehicle3
    "models/tracker/osnet_x0_25_msmt17.pt",
]
```

**If Option B**, add copy logic before ESSENTIAL check:
```python
# Copy 09g/09h kernel outputs into models/reid/
for src_name, dst_name in [
    ("/kaggle/input/09g-resnet101-ibn-a-dmt-cityflowv2/resnet101ibn_dmt_best.pth",
     "models/reid/resnet101ibn_dmt_best.pth"),
    ("/kaggle/input/09h-resnext101-ibn-a-dmt-cityflowv2/resnext101ibn_dmt_best.pth",
     "models/reid/resnext101ibn_dmt_best.pth"),
]:
    if Path(src_name).exists() and not (PROJECT / dst_name).exists():
        shutil.copy2(src_name, str(PROJECT / dst_name))
        print(f"  Copied {Path(src_name).name} -> {dst_name}")
```

### 3. Cell 10 (Model Info Print — lines 197-201)

Add tertiary model path display:
```python
_tertiary_path = PROJECT / "models" / "reid" / "resnext101ibn_dmt_best.pth"
print(f"✓ Tertiary ResNeXt101-IBN DMT: {_tertiary_path.name} ({_tertiary_path.stat().st_size/1024**2:.0f} MB)")
```

### 4. Cell 17 (Pipeline Command — lines 311-348) — CRITICAL

**Remove**:
```python
"--override", "stage2.reid.vehicle2.enabled=false",
```

**Add these overrides**:
```python
# --- Vehicle2: ResNet101-IBN-a DMT (secondary ensemble) ---
"--override", "stage2.reid.vehicle2.enabled=true",
"--override", "stage2.reid.vehicle2.weights_path=models/reid/resnet101ibn_dmt_best.pth",
"--override", "stage2.reid.vehicle2.model_name=resnet101_ibn_a",
"--override", "stage2.reid.vehicle2.embedding_dim=2048",
"--override", "stage2.reid.vehicle2.input_size=[384,384]",
# --- Vehicle3: ResNeXt101-IBN-a DMT (tertiary ensemble) ---
"--override", "stage2.reid.vehicle3.enabled=true",
"--override", "stage2.reid.vehicle3.weights_path=models/reid/resnext101ibn_dmt_best.pth",
"--override", "stage2.reid.vehicle3.model_name=resnext101_ibn_a",
"--override", "stage2.reid.vehicle3.embedding_dim=2048",
"--override", "stage2.reid.vehicle3.input_size=[384,384]",
```

**Note**: `cityflowv2.yaml` already has vehicle2 config with `resnet101ibn_cityflowv2_384px_best.pth`. The CLI overrides will override the YAML values, pointing to the new DMT checkpoint instead.

### 5. Cell 19 (Checkpoint Save — lines 358-398)

**No changes needed**. The tar already includes all files under `stage2/` via `stage_dir.rglob("*")`, so `embeddings_secondary.npy` and `embeddings_tertiary.npy` will be automatically included.

### 6. Markdown Cells (Cosmetic)

Update Cell 1 header table (line 2-13) to reflect 3-model extraction:
```
| 2 | TransReID 768D + ResNet101-IBN 2048D + ResNeXt101-IBN 2048D -> PCA 384D features | ~30 min |
```

## Config Flow Summary

```
default.yaml          → vehicle2.enabled=false, vehicle3.enabled=false
  ↓ merge
cityflowv2.yaml       → vehicle2.enabled=true (weights: resnet101ibn_cityflowv2_384px_best.pth)
                         vehicle3.enabled=false (weights: "")
  ↓ CLI overrides from 10a
10a notebook overrides → vehicle2.enabled=true, weights_path=resnet101ibn_dmt_best.pth
                         vehicle3.enabled=true, weights_path=resnext101ibn_dmt_best.pth
```

## Stage2 Output Structure (After Changes)

```
stage2/
  embeddings.npy              # Primary TransReID (N, 384) after PCA
  embeddings_secondary.npy    # Vehicle2 ResNet101-IBN DMT (N, 384) after PCA
  embeddings_tertiary.npy     # Vehicle3 ResNeXt101-IBN DMT (N, 384) after PCA
  embeddings_hsv.npy          # HSV color histograms
  embeddings_index.json       # Track metadata
```

## Downstream Impact

- **10b** (stage 3 — FAISS indexing): Needs to handle secondary/tertiary embedding files. Check if `src/stage3_indexing/pipeline.py` indexes all three.
- **10c** (stage 4 — association): Needs `stage4.association` config for multi-model fusion weights. Check how secondary/tertiary embeddings are fused in similarity scoring.
- **Stage 4 fusion**: Currently uses `stage4.association.ensemble.secondary_weight` for vehicle2. Need to verify vehicle3 weight config exists.

## Runtime Estimate

Adding 2 extra models increases stage2 time:
- Current (1 model): ~20 min
- With 3 models: ~45-55 min (each model does forward passes on all crops)
- Total 10a runtime: ~110-130 min (within Kaggle 12h limit)

## Prerequisites

1. 09g kernel must have completed successfully → `resnet101ibn_dmt_best.pth` available
2. 09h kernel must have completed successfully → `resnext101ibn_dmt_best.pth` available
3. Weights uploaded to mtmc-weights (Option A) or kernel_sources configured (Option B)
4. Verify stage4 can handle 3-model fusion before running full pipeline