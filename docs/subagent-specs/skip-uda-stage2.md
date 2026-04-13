# Skip UDA Stage 2 in 09g/09h ReID Training

## Problem

Both 09g (ResNet101-IBN-a) and 09h (ResNeXt101-IBN-a) notebooks implement a DMT-inspired 2-stage training pipeline:
- **Stage 1**: Supervised ReID training on CityFlowV2 identity labels (120 epochs)
- **Stage 2**: Unsupervised Domain Adaptation (UDA) via DBSCAN pseudo-labeling (40 epochs)

**Results from cancelled runs (12-hour Kaggle limit):**

| Notebook | Stage 1 best mAP | Stage 1 time | Stage 2 result |
|----------|------------------|--------------|----------------|
| 09g (mrk, T4) | 0.3707 | ~4.0hr (t=14177s) | NaN loss at epoch 38, mAP=0.030 |
| 09h (abdo, T4) | 0.3740 | ~5.4hr (t=19571s) | mAP degraded 0.31→0.16, 93.5% DBSCAN noise |

**Root cause**: UDA is fundamentally inappropriate. We're training and evaluating ON CityFlowV2 — there's no domain gap to bridge. DBSCAN can't form meaningful clusters because the same-domain embeddings don't have the distributional shift that UDA expects.

**Impact**: Stage 2 wastes 4-7 hours of the 12-hour Kaggle budget and produces worse models than Stage 1 alone.

## Solution

Remove Stage 2 UDA entirely. Use Stage 1's best checkpoint as the final output. Reinvest freed GPU time into longer Stage 1 training.

## Files to Modify

- `notebooks/kaggle/09g_resnet101ibn_dmt/09g_resnet101ibn_dmt.ipynb`
- `notebooks/kaggle/09h_resnext101ibn_dmt/09h_resnext101ibn_dmt.ipynb`

No generator scripts exist — notebooks are edited directly.

## 10a Expected Weight Filenames

10a (`notebooks/kaggle/10a_stages012/mtmc-10a-stages-0-2-tracking-reid-features.ipynb`) expects these files in the `mtmc-weights` Kaggle dataset:

| 10a variable | Path in mtmc-weights | Source notebook |
|---|---|---|
| `SECONDARY_WEIGHTS` | `models/reid/resnet101ibn_dmt_best.pth` | 09g |
| `TERTIARY_WEIGHTS` | `models/reid/resnext101ibn_dmt_best.pth` | 09h |

Both must contain a dict with key `"state_dict"` mapping to model `state_dict()`.

**Current output filenames** (already correct):
- 09g Cell 8 saves to `/kaggle/working/resnet101ibn_dmt_best.pth` ✓
- 09h Cell 8 saves to `/kaggle/working/resnext101ibn_dmt_best.pth` ✓

## Exact Changes

### Change 1: Config — Increase Stage 1 Epochs (Cell 1, both notebooks)

In the `CFG` dict:

```python
# BEFORE
"train_epochs": 120,
"stage2_epochs": 40,
"stage2_recluster_every": 3,
"stage2_iters_per_epoch": 300,
"eval_every": 10,
"stage2_eval_every": 3,
...
"stage2_lr": 1.0e-4,
...
"dbscan_eps": 0.58,
"dbscan_min_samples": 4,
"fic_lambda": 5.0e-4,
...
"uda_include_eval": False,

# AFTER
"train_epochs": 240,           # doubled: ~8hr on T4, well within 12hr limit
"eval_every": 10,              # keep as-is (24 evals total)
...
```

Remove these keys from CFG (they're only used by Stage 2):
- `"stage2_epochs"`
- `"stage2_recluster_every"`
- `"stage2_iters_per_epoch"`
- `"stage2_eval_every"`
- `"stage2_lr"`
- `"dbscan_eps"`
- `"dbscan_min_samples"`
- `"fic_lambda"`
- `"uda_include_eval"`

Also remove the `"target_map"` key — it was only used as a baseline comparison reference but not functionally important.

**Epoch budget rationale**:
- 09g: 120 epochs ≈ 4hr → each epoch ≈ 2min → 240 epochs ≈ 8hr (safe within 12hr with 2hr setup/eval buffer)
- 09h: 120 epochs ≈ 5.4hr → each epoch ≈ 2.7min → 240 epochs ≈ 10.8hr (tight but doable; if 09h proves too slow, fall back to 200)

For 09h specifically, consider using `"train_epochs": 200` instead of 240 given the higher per-epoch cost (2.7min vs 2min). The coder should time the first few epochs and adjust if needed.

### Change 2: Remove UDA Data Prep (Cell 2, both notebooks)

Remove the `UDA_SOURCE_RECORDS` block and `uda_eval_loader`:

```python
# REMOVE these lines from Cell 2:
UDA_SOURCE_RECORDS = list(train_records)
if CFG["uda_include_eval"]:
    for record in query_records + gallery_records:
        staged = dict(record)
        staged["pid"] = -1
        UDA_SOURCE_RECORDS.append(staged)

# REMOVE:
uda_eval_loader = DataLoader(
    ReIDImageDataset(UDA_SOURCE_RECORDS, eval_transform),
    batch_size=128,
    shuffle=False,
    num_workers=CFG["num_workers"],
    pin_memory=True,
)

# REMOVE "uda_images" from SPLIT_STATS dict
```

### Change 3: Remove DBSCAN import (Cell 1, both notebooks)

Remove `from sklearn.cluster import DBSCAN` from the imports (no longer needed).

### Change 4: Replace Stage 2 Cell Entirely (Cell 6, both notebooks)

**09g Cell 6** (lines 684-822) — replace entire cell with:

```python
# Stage 2 UDA: SKIPPED
# UDA is inappropriate for same-domain training (CityFlowV2→CityFlowV2).
# DBSCAN classified 93.5% as noise, causing mAP collapse.
# Using Stage 1 best checkpoint directly as final model.

print("Stage 2 UDA skipped — using Stage 1 best model directly")
print(json.dumps(STAGE1_BEST, indent=2))
```

**09h Cell 6** (lines 817-955) — same replacement.

### Change 5: Simplify Final Evaluation Cell (Cell 7, both notebooks)

**09g Cell 7** (lines 825-849) — replace with:

```python
FINAL_MODEL = model  # Stage 1 best (already loaded from stage1_best.pth)
FINAL_METRICS = evaluate_model(FINAL_MODEL)

print(json.dumps({"final_metrics": FINAL_METRICS, "stage1_best": STAGE1_BEST}, indent=2))
```

**09h Cell 7** (lines 958-982) — same replacement.

### Change 6: Simplify Save Cell (Cell 8, both notebooks)

**09g Cell 8** (lines 852-900) — replace with:

```python
BEST_MODEL_PATH = Path("/kaggle/working/resnet101ibn_dmt_best.pth")
METADATA_PATH = Path("/kaggle/working/resnet101ibn_dmt_metadata.json")
HISTORY_PATH = Path("/kaggle/working/resnet101ibn_dmt_history.json")

torch.save({"state_dict": FINAL_MODEL.state_dict()}, BEST_MODEL_PATH)

with HISTORY_PATH.open("w", encoding="utf-8") as handle:
    json.dump({"stage1": STAGE1_HISTORY}, handle, indent=2)

metadata = {
    "model": "resnet101_ibn_a",
    "recipe": "Supervised ReID (Stage 2 UDA removed — same-domain makes it counterproductive)",
    "pooling": "GeM",
    "neck": "BNNeck",
    "pretraining": "ImageNet IBN-Net official weights",
    "dataset": "CityFlowV2 ReID",
    "img_size": list(CFG["img_size"]),
    "embedding_dim": model.feat_dim,
    "num_train_ids": len(train_pid_map),
    "num_cameras": len(camname_to_id),
    "stage1_best": STAGE1_BEST,
    "final_metrics": FINAL_METRICS,
    "split_stats": SPLIT_STATS,
    "config": CFG,
}
with METADATA_PATH.open("w", encoding="utf-8") as handle:
    json.dump(metadata, handle, indent=2)

print(json.dumps({
    "best_model": str(BEST_MODEL_PATH),
    "metadata": str(METADATA_PATH),
    "history": str(HISTORY_PATH),
    "checkpoint_dir": str(CHECKPOINT_DIR),
}, indent=2))
```

**09h Cell 8** (lines 985-1034) — same pattern but with:
- `BEST_MODEL_PATH = Path("/kaggle/working/resnext101ibn_dmt_best.pth")`
- `METADATA_PATH = Path("/kaggle/working/resnext101ibn_dmt_metadata.json")`
- `HISTORY_PATH = Path("/kaggle/working/resnext101ibn_dmt_history.json")`
- `"model": "resnext101_ibn_a"`
- `"backbone_variant": "resnext101_32x8d"` (keep 09h's extra field)

### Change 7: Remove `fic_whiten` and `build_pseudo_records` functions (Cell 6)

These functions are defined at the top of the old Stage 2 cell and are only used by UDA. They will be entirely removed as part of the cell replacement in Change 4.

## Additional Optimizations (Optional, Lower Priority)

Since we're doubling epochs, a few training tweaks could help:

1. **More warmup**: `"warmup_epochs": 20` (currently 10) — longer warmup helps with longer training
2. **Learning rate reduction after plateau**: Add ReduceLROnPlateau or just rely on the cosine schedule (which already adapts to `train_epochs`)
3. **Checkpoint last N**: Save the last model as well (not just best mAP), in case mAP metric is noisy at later epochs

These are optional and should NOT block the main change.

## Post-Deployment

1. Push 09g: `kaggle kernels push -p notebooks/kaggle/09g_resnet101ibn_dmt/`
2. Push 09h: `kaggle kernels push -p notebooks/kaggle/09h_resnext101ibn_dmt/`
3. Monitor with `python scripts/kaggle_logs.py <slug> --tail 50`
4. After success, download output `.pth` files and upload to `mtmc-weights` dataset
5. Then push 10a which picks up the new weights automatically

## Risks

- **240 epochs may timeout for 09h** (2.7min/epoch × 240 = 10.8hr + setup/eval). Mitigation: use 200 epochs for 09h, or time first 10 epochs and extrapolate.
- **Diminishing returns beyond 120 epochs**: The best mAP may already plateau by epoch 120. But since cosine LR keeps learning rate warm through the middle of training, additional epochs give the model more chances to find better minima.
- **No fallback**: Without Stage 2, there's no secondary model to fall back to. But Stage 2 was strictly worse, so this is a non-issue.
