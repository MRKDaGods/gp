# 10a: Deploy Fine-Tuned R50-IBN Weights from 09n

> **Status**: Ready for implementation
> **Scope**: 10a notebook + reid_model.py key-format detection
> **Prerequisite**: 09n kernel must have completed successfully and produced output

## Motivation

The current 10a pipeline uses the VeRi-776 pretrained R50-IBN weights (52.77% mAP on CityFlowV2) as the secondary model for score-level fusion. Previous experiments confirmed that this secondary is too weak for meaningful ensemble gain (10c v56: -0.1pp with VeRi pretrained weights at fusion_weight=0.30).

The 09n kernel fine-tunes R50-IBN-a on CityFlowV2 using SBS pretrained initialization, which should reach 60–72% mAP. If mAP ≥ 65%, the secondary becomes strong enough for score-level fusion to provide meaningful improvement over the primary-only baseline (IDF1=0.775).

## Current State (10a)

- **kernel-metadata.json**: `kernel_sources: ["gumfreddy/09l-transreid-laion-2b-training"]`
- **Cell 9** (lines 212–224): Downloads VeRi-776 pretrained weights from GitHub
  - URL: `https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/veri_sbs_R50-ibn.pth`
  - Saves to: `models/reid/fastreid_sbs_r50_ibn_veri.pth`
- **Pipeline overrides** (Cell 12, line ~365): `stage2.reid.vehicle2.weights_path=models/reid/fastreid_sbs_r50_ibn_veri.pth`
- **Model name**: `fastreid_sbs_r50_ibn` (dispatches to `_build_fastreid_sbs_r50_ibn` in `reid_model.py`)

## Changes Required

### 1. kernel-metadata.json (`notebooks/kaggle/10a_stages012/kernel-metadata.json`)

Add the 09n kernel output as a data source:

```json
{
  "kernel_sources": [
    "gumfreddy/09l-transreid-laion-2b-training",
    "gumfreddy/09n-fastreid-r50-finetune-cityflowv2"
  ]
}
```

### 2. Cell 9: Replace GitHub Download with Kaggle Input Copy

**Remove** the `urllib.request` download block. **Replace** with:

```python
import shutil

FASTREID_FINETUNED_SRC = Path("/kaggle/input/09n-fastreid-r50-finetune-cityflowv2/fastreid_r50_ibn_cityflowv2_best.pth")
FASTREID_SECONDARY_PATH = PROJECT / "models" / "reid" / "fastreid_r50_ibn_cityflowv2.pth"
PRIMARY_WEIGHTS_PATH = PROJECT / "models" / "reid" / "transreid_cityflowv2_best.pth"

print(f"✓ Primary 256px ViT: {PRIMARY_WEIGHTS_PATH.name} ({PRIMARY_WEIGHTS_PATH.stat().st_size/1024**2:.0f} MB)")
assert FASTREID_FINETUNED_SRC.exists(), f"09n output not found: {FASTREID_FINETUNED_SRC}"
FASTREID_SECONDARY_PATH.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(str(FASTREID_FINETUNED_SRC), str(FASTREID_SECONDARY_PATH))
print(f"✓ Secondary R50-IBN (CityFlowV2 fine-tuned): {FASTREID_SECONDARY_PATH.name} ({FASTREID_SECONDARY_PATH.stat().st_size/1024**2:.0f} MB)")
```

### 3. Pipeline Override Update (Cell 12)

Change the weights path override from:
```
stage2.reid.vehicle2.weights_path=models/reid/fastreid_sbs_r50_ibn_veri.pth
```
To:
```
stage2.reid.vehicle2.weights_path=models/reid/fastreid_r50_ibn_cityflowv2.pth
```

**Keep unchanged:**
- `stage2.reid.vehicle2.model_name=fastreid_sbs_r50_ibn` (architecture is the same)
- `stage2.reid.vehicle2.embedding_dim=2048`
- All other vehicle2 overrides

### 4. CRITICAL: Fix Key-Format Detection in `reid_model.py`

**File**: `src/stage2_features/reid_model.py`, method `_build_fastreid_sbs_r50_ibn` (line ~171)

**Problem**: The `_remap_fastreid_sbs_r50_ibn_state_dict` function is designed for FastReID's checkpoint key format (`heads.pool_layer.p` → `pool.p`, `heads.bottleneck.*` → `bottleneck.*`). The 09n training saves with `model.state_dict()` which already uses the native `ReIDModelResNet50IBN` key format:
- `backbone.*` — passes through remap ✓
- `pool.p` — **DROPPED** by remap (not recognized) ✗
- `bottleneck.*` — **DROPPED** by remap (not recognized) ✗
- `classifier.*` — dropped by remap (expected, not used at inference)

Without this fix, loading 09n weights will silently lose the GeM pooling parameter (`pool.p`) and the entire BNNeck (`bottleneck.*`), causing the model to use randomly initialized values and severely degrading secondary embeddings.

**Fix**: Add key-format detection before the remap call in `_build_fastreid_sbs_r50_ibn`:

```python
if weights_path is not None:
    try:
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
        state_dict = self._unwrap_checkpoint_state_dict(checkpoint)

        # Detect if checkpoint is already in native ReIDModelResNet50IBN format
        # vs FastReID format (needs remapping)
        needs_remap = "heads.pool_layer.p" in state_dict or any(
            k.startswith("heads.bottleneck.") for k in state_dict
        )
        if needs_remap:
            state_dict = self._remap_fastreid_sbs_r50_ibn_state_dict(state_dict)

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        # ... rest unchanged
```

**Heuristic**: If the state dict contains `heads.pool_layer.p` or any `heads.bottleneck.*` key, it's in FastReID format and needs remapping. If it contains `pool.p` directly, it's already in native format and should be loaded as-is.

## What NOT to Change

- Do NOT change `model_name` — `fastreid_sbs_r50_ibn` is the correct dispatch name for both VeRi-pretrained and CityFlowV2-finetuned R50-IBN-a weights
- Do NOT change `embedding_dim` — remains 2048
- Do NOT change `input_size` — remains [256, 256]
- Do NOT change `clip_normalization` — remains false
- Do NOT remove the 09l kernel source — other notebooks may depend on it

## 10c Fusion Sweep After Deployment

After 10a produces the new secondary embeddings, run 10c with a fusion weight sweep. The existing sweep infrastructure in 10c (Cell after "5. Results") already supports this — no 10c code changes needed.

### Expected Sweep Parameters

The 10c notebook already has `FUSION_SWEEP_ENABLED = True` and sweeps:
```python
fusion_weights = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
```

### Recommended Focused Sweep

If the 09n mAP ≥ 65%, expand the sweep to finer granularity in the promising range:
```python
fusion_weights = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
```

### Expected Outcomes by Secondary mAP

| Secondary mAP | Expected Best Fusion Weight | Expected IDF1 Gain |
|---|---|---|
| 55–60% | 0.05–0.10 | +0.0 to +0.2pp (marginal) |
| 60–65% | 0.10–0.20 | +0.2 to +0.5pp |
| 65–72% | 0.15–0.30 | +0.5 to +1.5pp |
| >72% | 0.20–0.35 | +1.0 to +2.0pp |

### Why This Should Work Now

Previous ensemble attempt (10c v56) failed because:
1. The VeRi-pretrained secondary had only 52.77% mAP — too weak, adding noise
2. Score-level fusion with 52.77% secondary = -0.1pp

With CityFlowV2-finetuned weights (target ≥65% mAP):
1. The secondary produces embeddings calibrated to CityFlowV2 vehicle identities
2. The CNN (ResNet50-IBN) and ViT (TransReID) have complementary failure modes
3. Score-level fusion can correct the ~35 conflation errors in the current pipeline

### Success Criteria

- **Minimum**: mAP ≥ 60% and fusion_weight > 0.0 provides IDF1 ≥ 0.775 (no regression)
- **Target**: mAP ≥ 65% and best fusion_weight provides IDF1 ≥ 0.780 (+0.5pp)
- **Stretch**: mAP ≥ 70% and best fusion_weight provides IDF1 ≥ 0.790 (+1.5pp)

## Architecture Compatibility Notes

- **09n model class**: `ReIDModelResNet50IBN` from `src/training/model.py`
- **09n state dict keys**: `backbone.*`, `pool.p`, `bottleneck.*`, `classifier.*`
- **Inference model class**: `ReIDModelResNet50IBN` from `src/training/model.py` (same class)
- **At inference**: only `backbone.*`, `pool.p`, `bottleneck.*` are used; `classifier.*` is discarded via `strict=False`
- **Embedding**: 2048-dim, L2-normalized, produced by `forward()` → backbone → GeM → bottleneck → L2-norm

## Implementation Order

1. Fix `reid_model.py` key-format detection (Change 4) — MUST be done first
2. Edit kernel-metadata.json (Change 1)
3. Edit Cell 9 in 10a notebook (Change 2)
4. Edit Cell 12 pipeline overrides in 10a notebook (Change 3)
5. Push 10a: `kaggle kernels push -p notebooks/kaggle/10a_stages012/`
6. After 10a completes → 10b auto-chain → 10c runs fusion sweep
7. Analyze fusion sweep results in 10c output

## Verification

After implementing changes 1–4, verify locally:
```bash
# Check kernel-metadata.json has both sources
python -c "import json; d=json.load(open('notebooks/kaggle/10a_stages012/kernel-metadata.json')); assert 'gumfreddy/09n-fastreid-r50-finetune-cityflowv2' in d['kernel_sources']; print('OK')"

# Check the weights path in the notebook refers to the new file
python -c "import json; nb=json.load(open('notebooks/kaggle/10a_stages012/mtmc-10a-stages-0-2-tracking-reid-features.ipynb')); cells=''.join(c['source'] if isinstance(c['source'],str) else ''.join(c['source']) for c in nb['cells']); assert 'fastreid_r50_ibn_cityflowv2' in cells; assert 'veri_sbs_R50-ibn.pth' not in cells; print('OK')"
```
