# Test-Time Augmentation (TTA) for ReID Feature Extraction

## Status: DESIGN SPEC — Ready for Implementation

## 1. Current State (What Already Exists)

The codebase already has partial TTA infrastructure in `src/stage2_features/reid_model.py`:

| Augmentation | Implemented? | Active on Kaggle? | Notes |
|---|---|---|---|
| **Horizontal flip** | ✅ `flip_augment` | ✅ ON (default=True, no 10a override) | Standard ReID flip-average |
| **Color augment** | ✅ `color_augment` | ❌ OFF (10a sets `false`) | Tested v27: neutral-to-harmful |
| **Multiscale resize** | ✅ `multiscale_sizes` | ❌ OFF (always `[]`) | Never tested on Kaggle |
| **CamTTA (BN warmup)** | ✅ `camera_tta` | ❌ OFF | Tested v49: no gain |

**Current Kaggle TTA**: 2 views only (original + flip). Features are summed then divided by `n_views`.

## 2. What's New in This Proposal

### 2a. Center-Crop Scale TTA (NEW — not yet implemented)

Instead of the existing "resize → resize back" multiscale approach (which interpolates the entire image at a different resolution), **center-crop TTA** selects a sub-region of the original crop before resizing to model input size. This preserves local pixel detail within the crop region.

- **0.9x center-crop**: Crop the center 90% of the original crop → resize to 256×256. Zooms in, loses edges. Emphasizes vehicle body, reduces background noise.
- **1.1x padded crop**: Pad the crop with border reflection by 10% on each side → resize to 256×256. Zooms out, adds context. Helps with partial occlusions.

This is **different from** the existing `multiscale_sizes` approach because:
- Multiscale: resizes the full crop to intermediate resolution then back (blurs/sharpens all pixels uniformly)
- Center-crop: selects a spatial sub-region of the original pixels (preserves native resolution within the crop)

### 2b. Per-View L2 Normalization Before Averaging (NEW — not yet implemented)

The current `_extract_batch` sums raw (un-normalized) features across views, then divides by `n_views`. This means views producing larger-magnitude features contribute more to the average. The fix:

1. L2-normalize each view's features independently
2. Sum the normalized features
3. L2-normalize the final sum (done downstream in the pipeline)

This is standard practice in ReID competitions (e.g., BoT, TransReID, SOLIDER papers) and ensures equal contribution from each augmented view regardless of magnitude differences.

### 2c. Enable Existing Multiscale TTA (EXISTING — just needs config)

The `multiscale_sizes` parameter already works but was never tested. For TransReID ViT-Base/16 with 256×256 input (16×16 patches), good candidates are:
- `[224, 224]` — 14×14 patches, zooms in by ~1.14×
- `[288, 288]` — 18×18 patches, zooms out by ~0.89×

Note: ViT positional embeddings are interpolated at inference for non-training resolutions. The existing implementation handles this correctly (resize to intermediate → resize back to 256×256).

## 3. Design Decisions

### Q1: Where to add TTA — model level vs pipeline level?

**Answer: Model level (`ReIDModel._extract_batch`).**

Rationale:
- All TTA is per-crop, not per-tracklet — it belongs with the batch inference code
- The existing flip/color/multiscale TTA already lives here
- Pipeline level (`run_stage2`) handles tracklet-level operations (quality pooling, PCA, etc.)
- Adding center-crop TTA to `_extract_batch` is consistent with existing patterns

### Q2: What augmentations to use?

**Answer: Flip + center-crop scale (0.9x, 1.1x). NOT color augment or multiscale resize.**

Rationale:
- Flip is already proven (on by default, included in all best results)
- Color augment was tested and found harmful (v27) — do not re-enable
- Multiscale resize (existing) is worth testing but is a separate experiment
- Center-crop (0.9x, 1.1x) is the genuinely new approach that provides spatial diversity
- With flip + 2 center-crop scales: **6 views** total (original, flip, 0.9x, 0.9x-flip, 1.1x, 1.1x-flip)
- Cost: ~3× compute (6 views vs current 2 views with flip only)

### Q3: How to make it configurable?

**Answer: Add `center_crop_scales` list to the existing `stage2.reid` config section.**

```yaml
stage2:
  reid:
    flip_augment: true
    color_augment: false
    center_crop_scales: [0.9, 1.1]  # NEW: empty list = disabled
    multiscale_sizes: []  # existing, orthogonal
    normalize_views: true  # NEW: L2-norm per view before averaging
```

### Q4: L2-normalize before or after averaging?

**Answer: Before (per-view normalization), controlled by `normalize_views` flag.**

Rationale:
- Different augmentations produce features with different magnitudes (flip is symmetric but scale is not)
- Without per-view normalization, larger-magnitude views dominate the average
- Standard practice in competition ReID systems
- The flag allows A/B testing: `normalize_views=true` vs `normalize_views=false`
- Final L2-normalization still happens downstream (after PCA whitening)

## 4. Exact Code Changes

### File 1: `src/stage2_features/reid_model.py`

#### Change 1a: Add `center_crop_scales` and `normalize_views` to `__init__`

In `ReIDModel.__init__()`, after the `self.multiscale_sizes` assignment (~line 68):

```python
# Add parameters:
self.center_crop_scales = center_crop_scales or []  # e.g., [0.9, 1.1]
self.normalize_views = normalize_views  # L2-norm per view before averaging
```

Add corresponding `__init__` parameters:
```python
def __init__(
    self,
    ...
    center_crop_scales: Optional[List[float]] = None,  # NEW
    normalize_views: bool = False,  # NEW
):
```

#### Change 1b: Add `_center_crop` helper method

After `_preprocess` method (~line 440):

```python
def _center_crop_at_scale(self, crops: List[np.ndarray], scale: float) -> List[np.ndarray]:
    """Apply center-crop or center-pad at a given scale factor.

    Args:
        crops: List of BGR uint8 crops (original size, before preprocessing).
        scale: Scale factor. <1.0 = crop center region, >1.0 = pad with border reflection.

    Returns:
        List of cropped/padded BGR uint8 crops.
    """
    result = []
    for crop in crops:
        h, w = crop.shape[:2]
        if scale < 1.0:
            # Center-crop: take center `scale` fraction
            new_h, new_w = int(h * scale), int(w * scale)
            y_start = (h - new_h) // 2
            x_start = (w - new_w) // 2
            cropped = crop[y_start:y_start + new_h, x_start:x_start + new_w]
            result.append(cropped)
        elif scale > 1.0:
            # Pad with border reflection
            pad_h = int(h * (scale - 1.0) / 2)
            pad_w = int(w * (scale - 1.0) / 2)
            padded = cv2.copyMakeBorder(
                crop, pad_h, pad_h, pad_w, pad_w,
                borderType=cv2.BORDER_REFLECT_101,
            )
            result.append(padded)
        else:
            result.append(crop)
    return result
```

#### Change 1c: Modify `_extract_batch` to support center-crop TTA and per-view normalization

In `_extract_batch`, restructure the feature accumulation to optionally L2-normalize each view. Replace the current accumulation pattern with:

```python
@torch.no_grad()
def _extract_batch(self, batch_crops: List[np.ndarray], cam_id: Optional[int] = None) -> np.ndarray:
    # ... existing setup ...

    views = []  # collect all view features

    # View 1: Original
    batch_tensor = self._preprocess(batch_crops).to(self.device)
    if self.half:
        batch_tensor = batch_tensor.half()
    cam_tensor = self._make_cam_tensor(len(batch_crops), cam_id)
    if cam_tensor is not None:
        features = self.model(batch_tensor, cam_ids=cam_tensor)
    else:
        features = self.model(batch_tensor)
    if isinstance(features, (tuple, list)):
        features = features[0]
    views.append(features.float().cpu().numpy())

    # View 2: Horizontal flip (if enabled)
    if self.flip_augment:
        flipped_crops = [cv2.flip(c, 1) for c in batch_crops]
        # ... (existing flip code, append to views)
        views.append(flip_features)

    # View 3+: Color augment (if enabled) — existing code
    if self.color_augment:
        for alpha, beta in [(1.2, 15), (0.8, -10)]:
            # ... existing code, append to views
            views.append(aug_features)

    # View N+: Center-crop scale TTA (NEW)
    for scale in self.center_crop_scales:
        scaled_crops = self._center_crop_at_scale(batch_crops, scale)
        sc_tensor = self._preprocess(scaled_crops).to(self.device)
        if self.half:
            sc_tensor = sc_tensor.half()
        if cam_tensor is not None:
            sc_features = self.model(sc_tensor, cam_ids=cam_tensor)
        else:
            sc_features = self.model(sc_tensor)
        if isinstance(sc_features, (tuple, list)):
            sc_features = sc_features[0]
        views.append(sc_features.float().cpu().numpy())

        # Flip on center-cropped views too
        if self.flip_augment:
            sc_flipped = [cv2.flip(c, 1) for c in scaled_crops]
            sc_flip_tensor = self._preprocess(sc_flipped).to(self.device)
            if self.half:
                sc_flip_tensor = sc_flip_tensor.half()
            if cam_tensor is not None:
                sc_flip_feat = self.model(sc_flip_tensor, cam_ids=cam_tensor)
            else:
                sc_flip_feat = self.model(sc_flip_tensor)
            if isinstance(sc_flip_feat, (tuple, list)):
                sc_flip_feat = sc_flip_feat[0]
            views.append(sc_flip_feat.float().cpu().numpy())

    # View N+: Multiscale TTA — existing code, adapted to views list
    # ... (keep existing multiscale logic, but append to views list)

    # Aggregate views
    if self.normalize_views and len(views) > 1:
        # L2-normalize each view before averaging
        normalized = []
        for v in views:
            norms = np.linalg.norm(v, axis=1, keepdims=True)
            normalized.append(v / np.maximum(norms, 1e-8))
        return np.mean(normalized, axis=0)
    else:
        return np.sum(views, axis=0) / len(views)
```

### File 2: `src/stage2_features/pipeline.py`

#### Change 2a: Pass new config params to ReIDModel constructors

In `run_stage2()`, where `vehicle_reid` and `person_reid` are constructed, add the new parameters:

```python
center_crop_raw = stage_cfg.reid.get("center_crop_scales", [])
center_crop_scales = [float(s) for s in center_crop_raw] if center_crop_raw else []
normalize_views = stage_cfg.reid.get("normalize_views", False)

vehicle_reid = ReIDModel(
    ...
    center_crop_scales=center_crop_scales,  # NEW
    normalize_views=normalize_views,  # NEW
)
```

Same for `person_reid`, `vehicle_reid2`, `vehicle_reid3` constructors.

### File 3: `configs/default.yaml`

#### Change 3a: Add new config keys under `stage2.reid`

After the existing `multiscale_sizes` line (~line 113):

```yaml
    # --- Center-crop scale TTA: crop/pad center region at different scales ---
    # Each entry is a float scale factor. <1.0 = center-crop, >1.0 = pad+zoom-out.
    # Empty list = disabled. Recommended: [0.9, 1.1] for ±10% spatial diversity.
    center_crop_scales: []  # disabled by default
    # --- Per-view L2 normalization before averaging ---
    # When true, each TTA view is L2-normalized before averaging, ensuring
    # equal contribution regardless of magnitude. Standard in ReID competitions.
    normalize_views: false  # disabled by default for backward compat
```

### File 4: `configs/datasets/cityflowv2.yaml`

#### Change 4a: Add new config keys (matching default, can override later)

After `color_augment` line (~line 109):

```yaml
    center_crop_scales: []  # TTA: test with [0.9, 1.1]
    normalize_views: false  # TTA: test with true
```

### File 5: 10a Kaggle notebook (config overrides)

In the pipeline run cell, add overrides for the new TTA parameters:

```python
# TTA configuration — center-crop scale diversity
"stage2.reid.center_crop_scales=[0.9,1.1]",
"stage2.reid.normalize_views=true",
```

## 5. Expected Impact

### Compute Cost
- Current: 2 views (original + flip) → **2 forward passes per crop**
- With center-crop [0.9, 1.1] + flip: 6 views → **6 forward passes per crop** (3× cost)
- With center-crop [0.9, 1.1] + flip + normalize_views: same 6 views, negligible norm overhead
- On Kaggle P100 with 48 crops/tracklet: ~3× stage2 time (from ~15min to ~45min, within 12h limit)

### Expected Quality Impact
- **Center-crop TTA**: +0.2–0.5pp MTMC IDF1 (conservative estimate based on ReID competition reports)
- **Per-view normalization**: +0.0–0.2pp (removes magnitude bias, minor but free)
- **Combined**: +0.2–0.7pp potential, with no training required

### Risk Assessment
- **Low risk**: Zero-training approach, fully configurable, easy to disable
- **Main risk**: Compute budget on Kaggle P100 (12h limit) — 3× feature extraction cost
- **Mitigation**: Can reduce `samples_per_tracklet` from 48 to 32 to offset compute, or use only `[0.9]` (4 views = 2× cost)

## 6. Experiment Plan

### Phase 1: Per-view normalization only (cheapest, zero compute overhead)

```
stage2.reid.normalize_views=true
stage2.reid.center_crop_scales=[]
```
This tests whether fixing the magnitude bias in flip averaging alone helps. Costs nothing extra.

### Phase 2: Center-crop [0.9] only (2× compute)

```
stage2.reid.normalize_views=true
stage2.reid.center_crop_scales=[0.9]
```
The 0.9x center-crop alone (zoom in on vehicle body) is the highest-value single augmentation. With flip, this gives 4 views.

### Phase 3: Full center-crop [0.9, 1.1] (3× compute)

```
stage2.reid.normalize_views=true
stage2.reid.center_crop_scales=[0.9,1.1]
```
Adds the 1.1x padded view for context. 6 total views.

### Phase 4 (optional): Compare with multiscale_sizes

```
stage2.reid.normalize_views=true
stage2.reid.center_crop_scales=[]
stage2.reid.multiscale_sizes=[[224,224],[288,288]]
```
Test existing multiscale implementation head-to-head with center-crop to see which spatial diversity method works better.

Each phase requires a full 10a→10b→10c pipeline chain. Evaluate at stage5 MTMC IDF1. **Baseline is current 10c v52 at 0.775.**

## 7. Integration with Kaggle Pipeline

The 10a notebook (`notebooks/kaggle/10a_stages012/mtmc-10a-stages-0-2-tracking-reid-features.ipynb`) passes all stage2 config via `--override` CLI args. Integration is straightforward:

1. Add `"stage2.reid.center_crop_scales=[0.9,1.1]"` to the overrides list
2. Add `"stage2.reid.normalize_views=true"` to the overrides list
3. Consider reducing `"stage2.crop.samples_per_tracklet=32"` if 12h time limit is tight

No changes needed to 10b or 10c — they only consume the embeddings produced by 10a/stage2.

## 8. Files to Modify (Summary)

| File | Change Type | Lines Affected |
|---|---|---|
| `src/stage2_features/reid_model.py` | Add `__init__` params, `_center_crop_at_scale` method, refactor `_extract_batch` | ~40 new lines, ~30 refactored |
| `src/stage2_features/pipeline.py` | Pass new config params to ReIDModel constructors | ~8 lines |
| `configs/default.yaml` | Add `center_crop_scales` and `normalize_views` keys | 6 lines |
| `configs/datasets/cityflowv2.yaml` | Add matching keys | 2 lines |
| 10a notebook | Add 2 `--override` flags | 2 lines |