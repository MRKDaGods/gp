# Ensemble 10a Deployment — LAION-2B CLIP ViT-B/16 Secondary Model

**Date**: 2026-04-18  
**Status**: READY TO DEPLOY — 09l v3 complete (78.61% mAP), pipeline.py constructor fixed  
**Goal**: Enable score-level ensemble feature extraction in 10a with LAION-2B CLIP ViT-B/16 secondary model

## Background

Single-model TransReID ViT-B/16 CLIP (OpenAI) achieves 80.14% mAP / IDF1=0.775. The gap to SOTA (0.8486) requires ensemble diversity. Previous attempts with ResNet101-IBN-a (52.77% mAP) failed at -0.1pp because the secondary was too weak.

**09l v3 result**: LAION-2B CLIP ViT-B/16 reached **78.61% mAP / 90.43% R1** — only 1.53pp behind primary, well above the 65% threshold. Ready for ensemble deployment.

## Approach: Kernel Source (NOT Dataset Upload)

Instead of uploading ~350MB to the `gumfreddy/mtmc-weights` dataset (slow), reference the 09l kernel output directly as a data source in 10a.

**Kernel**: `gumfreddy/09l-transreid-laion-2b-training` (v3, COMPLETE)  
**Model output path**: `exported_models/vehicle_transreid_vit_base_cityflowv2.pth`  
**Kaggle input path**: `/kaggle/input/09l-transreid-laion-2b-training/exported_models/vehicle_transreid_vit_base_cityflowv2.pth`

## Current State of Ensemble Infrastructure

### Stage 2 (Feature Extraction) — READY ✓

- `vehicle2` slot exists in `src/stage2_features/pipeline.py` (lines 222-245)
- Constructor passes `vit_model`, `concat_patch`, `num_cameras` — **bug already fixed**
- `save_separate=true` saves to `embeddings_secondary.npy` independently
- PCA whitening runs independently per model (`secondary_pca_model_path`)
- L2 normalization applied independently
- **No pipeline.py changes needed.**

### Stage 4 (Association) — READY ✓

- Secondary embeddings loading: `src/stage4_association/pipeline.py` lines 162-209
- Score-level blending: weighted average of cosine similarities
- Config: `stage4.association.secondary_embeddings.path` and `.weight`
- FIC whitening applied independently to secondary embeddings
- **No stage4 changes needed.**

### 10c Notebook — READY ✓

- `10c_stages45/mtmc-10c-stages-4-5-association-eval.ipynb` already has:
  - `SECONDARY_EMBEDDINGS_PATH = RUN_DIR / "stage2" / "embeddings_secondary.npy"` (auto-detected)
  - `secondary_embedding_overrides(weight)` function
  - `FUSION_WEIGHT = 0.30 if SECONDARY_EMBEDDINGS_PATH is not None else 0.00`
- **No 10c changes needed** — it auto-detects secondary embeddings if present.

## Required Changes (3 files only)

### Change 1: kernel-metadata.json — Add 09l as kernel source

**File**: `notebooks/kaggle/10a_stages012/kernel-metadata.json`

Add `gumfreddy/09l-transreid-laion-2b-training` to `kernel_sources`:

```json
{
  "id": "gumfreddy/mtmc-10a-stages-0-2-tracking-reid-features",
  "title": "MTMC 10a - Stages 0-2 (Tracking + ReID Features)",
  "code_file": "mtmc-10a-stages-0-2-tracking-reid-features.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_tpu": false,
  "enable_internet": true,
  "machine_shape": "NvidiaTeslaT4",
  "dataset_sources": [
    "gumfreddy/mtmc-weights",
    "thanhnguyenle/data-aicity-2023-track-2"
  ],
  "competition_sources": [],
  "kernel_sources": [
    "gumfreddy/09-vehicle-reid-cityflowv2",
    "gumfreddy/09l-transreid-laion-2b-training"
  ]
}
```

This makes the 09l output available at `/kaggle/input/09l-transreid-laion-2b-training/`.

### Change 2: 10a notebook — Copy LAION-2B weights + add vehicle2 overrides

**File**: `notebooks/kaggle/10a_stages012/mtmc-10a-stages-0-2-tracking-reid-features.ipynb`

#### 2a: Model copy cell (after the existing mtmc-weights copy block)

Add after the existing model copy logic (after the "All essential baseline weights present" line):

```python
# --- Copy LAION-2B CLIP secondary model from 09l kernel output ---
LAION2B_SRC = Path("/kaggle/input/09l-transreid-laion-2b-training/exported_models/vehicle_transreid_vit_base_cityflowv2.pth")
LAION2B_DST = PROJECT / "models" / "reid" / "transreid_cityflowv2_laion2b.pth"
if LAION2B_SRC.exists():
    shutil.copy2(str(LAION2B_SRC), str(LAION2B_DST))
    print(f"✓ LAION-2B secondary: {LAION2B_DST.name} ({LAION2B_DST.stat().st_size/1024**2:.0f} MB)")
else:
    print(f"⚠ LAION-2B weights not found at {LAION2B_SRC} — ensemble disabled")
    LAION2B_DST = None
```

Note: We rename to `transreid_cityflowv2_laion2b.pth` to distinguish from the primary `transreid_cityflowv2_best.pth`.

#### 2b: Vehicle ensemble status line

Update the existing status line from:
```python
print("✓ Vehicle ensemble overrides: disabled")
```
to:
```python
if LAION2B_DST and LAION2B_DST.exists():
    print(f"✓ Vehicle ensemble: ENABLED (LAION-2B secondary)")
else:
    print("✓ Vehicle ensemble overrides: disabled")
```

#### 2c: Pipeline execution cell — add vehicle2 overrides

Add these overrides to the `cmd` list, after the existing `stage2` overrides and before `stage1` overrides:

```python
    # --- Secondary LAION-2B CLIP model for ensemble ---
    "--override", "stage2.reid.vehicle2.enabled=true",
    "--override", "stage2.reid.vehicle2.model_name=transreid",
    "--override", "stage2.reid.vehicle2.weights_path=models/reid/transreid_cityflowv2_laion2b.pth",
    "--override", "stage2.reid.vehicle2.embedding_dim=768",
    "--override", "stage2.reid.vehicle2.input_size=[256,256]",
    "--override", "stage2.reid.vehicle2.vit_model=vit_base_patch16_clip_224.laion2b",
    "--override", "stage2.reid.vehicle2.num_cameras=59",
    "--override", "stage2.reid.vehicle2.clip_normalization=true",
    "--override", "stage2.reid.vehicle2.concat_patch=true",
    "--override", "stage2.reid.vehicle2.save_separate=true",
```

These should be wrapped in a conditional if we want a safe fallback:
```python
# Add vehicle2 overrides only if LAION-2B weights were copied successfully
if LAION2B_DST and LAION2B_DST.exists():
    cmd.extend([
        "--override", "stage2.reid.vehicle2.enabled=true",
        "--override", "stage2.reid.vehicle2.model_name=transreid",
        "--override", f"stage2.reid.vehicle2.weights_path={LAION2B_DST}",
        "--override", "stage2.reid.vehicle2.embedding_dim=768",
        "--override", "stage2.reid.vehicle2.input_size=[256,256]",
        "--override", "stage2.reid.vehicle2.vit_model=vit_base_patch16_clip_224.laion2b",
        "--override", "stage2.reid.vehicle2.num_cameras=59",
        "--override", "stage2.reid.vehicle2.clip_normalization=true",
        "--override", "stage2.reid.vehicle2.concat_patch=true",
        "--override", "stage2.reid.vehicle2.save_separate=true",
    ])
    print("✓ Vehicle2 ensemble overrides added to pipeline command")
```

### Change 3: default.yaml — Update vehicle2 defaults to TransReID CLIP

**File**: `configs/default.yaml` lines 83-90

Replace the current vehicle2 block:
```yaml
    vehicle2:
      enabled: false
      save_separate: true
      model_name: "resnet101_ibn_a"
      weights_path: "models/reid/resnet101ibn_cityflowv2_384px_best.pth"
      embedding_dim: 2048
      input_size: [384, 384]
      clip_normalization: false
```

With:
```yaml
    vehicle2:
      enabled: false  # enabled per-dataset for ensemble runs
      save_separate: true  # Save to embeddings_secondary.npy for stage4 fusion
      model_name: "transreid"
      weights_path: ""  # set via override on Kaggle
      embedding_dim: 768
      input_size: [256, 256]
      vit_model: "vit_base_patch16_clip_224.laion2b"
      num_cameras: 0  # set to 59 for CityFlowV2 via override
      clip_normalization: true
      concat_patch: false
```

This changes defaults from the dead ResNet101-IBN-a to TransReID CLIP LAION-2B.

## Weight Paths on Kaggle

| Weight File | Source | Kaggle Input Path | Local Copy Path |
|-------------|--------|-------------------|-----------------|
| `vehicle_transreid_vit_base_cityflowv2.pth` | 09l v3 output | `/kaggle/input/09l-transreid-laion-2b-training/exported_models/vehicle_transreid_vit_base_cityflowv2.pth` | `models/reid/transreid_cityflowv2_laion2b.pth` |

**No dataset upload needed** — the 09l kernel output is referenced directly via `kernel_sources`.

## PCA Considerations

The secondary CLIP model produces 768D embeddings (same as primary). PCA reduces to 384D.

Current code supports separate PCA via `stage2.pca.secondary_pca_model_path`. Since we're fitting PCA on the fly (not loading a pre-fitted model), the default behavior will auto-fit a new PCA for secondary embeddings. **No changes needed.**

## Runtime Impact

Stage 2 runtime on T4 (current ~30 min):
- Second CLIP ViT-B/16 adds ~25-30 min (same architecture, same batch size)
- Total stage 2: ~55-60 min
- 10a total (stages 0+1+2): ~20 + 45 + 60 = **~125 min** (within Kaggle 12h limit)

## Deployment Plan

### Phase 1: Apply changes (this session)
1. Update `kernel-metadata.json` — add 09l kernel source
2. Update 10a notebook — model copy + vehicle2 overrides
3. Update `default.yaml` — vehicle2 defaults to CLIP LAION-2B

### Phase 2: Push and run
1. Push 10a: `kaggle kernels push -p notebooks/kaggle/10a_stages012/`
2. Monitor: confirm "Ensemble ReID enabled" in logs
3. Verify `embeddings_secondary.npy` in 10a output

### Phase 3: Evaluate via 10c
1. 10c auto-detects secondary embeddings — just push 10b then 10c as normal
2. Default fusion weight is 0.30 (already configured in 10c)
3. If IDF1 improves: sweep weight ∈ {0.10, 0.20, 0.30, 0.40, 0.50}
4. Target: IDF1 > 0.775 (baseline) — need ≥+0.5pp to justify complexity

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| CLIP variants too correlated | No diversity gain | Different pretraining data (OpenAI vs LAION-2B) = different failure modes |
| OOM on T4 (16GB) | Stage 2 crash | Sequential extraction (not parallel); batch_size=64 should fit |
| PCA underfitting | Noisy secondary embeddings | N tracklets >> 384 (typically N~2000+) |
| Runtime exceeds Kaggle limit | Incomplete run | 125 min << 720 min limit; safe margin |
| 09l kernel output expires | Weights unavailable | Kaggle kernel outputs persist; if needed, re-run 09l or upload to dataset |

## Summary of What's Already Done vs. What Needs Doing

| Component | Status | Action |
|-----------|--------|--------|
| `pipeline.py` vehicle2 constructor | ✅ Fixed | None |
| `pipeline.py` vehicle3 constructor | ✅ Fixed | None |
| Stage 4 score-level fusion | ✅ Ready | None |
| 10c secondary embeddings auto-detection | ✅ Ready | None |
| `kernel-metadata.json` 09l source | ❌ Not done | Add kernel_source |
| 10a model copy cell | ❌ Not done | Add LAION-2B copy logic |
| 10a pipeline overrides | ❌ Not done | Add vehicle2 overrides |
| `default.yaml` vehicle2 defaults | ❌ Not done | Update to CLIP LAION-2B |