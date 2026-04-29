# Timestamp-Based CID_BIAS for Stage 4 Association

> **Date**: 2026-04-17  
> **Baseline**: MTMC IDF1 = 0.775 (10c v52)  
> **Expected Impact**: +1-2pp MTMC IDF1  
> **Prerequisite**: None — pure config/data change, no pipeline code modifications needed

---

## Background

### Why the Previous CID_BIAS Failed (-3.3pp)

The previous implementation (`scripts/compute_cid_bias.py`) learned per-camera-pair biases by:
1. Matching predicted tracklets to GT global IDs via frame-level IoU
2. Computing mean cosine similarity for same-identity cross-camera pairs
3. Centering biases around the global mean similarity

This failed because:
- Only **464 GT-matched tracklets** — far too few for robust statistics
- Biases were **derived from model errors** (noisy embeddings → noisy biases)
- The centering approach **shifted all similarities** relative to a noisy estimate of the global mean
- Result: the bias correction introduced more noise than it removed

### What AIC22 Winners Do Instead

AIC22 winners (Li et al., Luo et al.) use **topology-based CID_BIAS** — a static matrix derived from physical camera layout, NOT learned from data:

| Source | Description |
|--------|-------------|
| **Camera adjacency** | Cameras viewing the same intersection get positive bias (vehicles routinely appear in multiple cameras within seconds) |
| **Physical separation** | Cameras at different locations get negative bias (vehicles cannot teleport between scenes) |
| **Travel time windows** | Known min/max transit times constrain which cross-camera matches are plausible |

This is a **prior** on camera-pair match likelihood, not a learned correction. It's robust because it encodes physical facts, not statistical estimates.

---

## CityFlowV2 Camera Topology

### Scene Layout

```
Scene S01 — Single Intersection
┌─────────────────────────┐
│  c001  c002  c003       │    3 cameras covering one intersection
│    ↕     ↕     ↕        │    Vehicles transit between cameras in 0-20 seconds
│  [Overlapping FOVs]     │    min_time=0 for all intra-S01 pairs
└─────────────────────────┘

Scene S02 — Separate Intersection
┌─────────────────────────┐
│  c006  c007  c008       │    3 cameras covering another intersection
│    ↕     ↕     ↕        │    Vehicles transit between cameras in 0-30 seconds
│  [Overlapping FOVs]     │    min_time=0 for all intra-S02 pairs
└─────────────────────────┘

S01 ↔ S02: Physically separated locations. No camera_transitions defined in config.
Cross-scene matches are extremely unlikely (different roads, no shared traffic flow).
```

### Known Camera Transitions (from `cityflowv2.yaml`)

| Pair | min_time | max_time | mean_time | std_time | N (GT) |
|------|:--------:|:--------:|:---------:|:--------:|:------:|
| c001↔c002 | 0s | 20s | 1.9s | 3.4s | ≥46 |
| c001↔c003 | 0s | 5s | 0.5s | 0.9s | ≥46 |
| c002↔c003 | 0s | 20s | 1.4s | 3.1s | ≥46 |
| c006↔c007 | 0s | 25s | 0.9s | 2.6s | ≥46 |
| c006↔c008 | 0s | 30s | 8.1s | 6.2s | ≥46 |
| c007↔c008 | 0s | 20s | 5.8s | 5.1s | ≥46 |
| S01↔S02 | — | — | — | — | 0 |

---

## CID_BIAS Matrix Design

### Matrix Definition

A 6×6 symmetric matrix indexed by `[S01_c001, S01_c002, S01_c003, S02_c006, S02_c007, S02_c008]`.

Values are **additive offsets** applied to the combined similarity score before graph thresholding.

```
              c001    c002    c003    c006    c007    c008
c001      [  0.00,  +0.04,  +0.04,  -0.15,  -0.15,  -0.15 ]
c002      [ +0.04,   0.00,  +0.04,  -0.15,  -0.15,  -0.15 ]
c003      [ +0.04,  +0.04,   0.00,  -0.15,  -0.15,  -0.15 ]
c006      [ -0.15,  -0.15,  -0.15,   0.00,  +0.04,  +0.04 ]
c007      [ -0.15,  -0.15,  -0.15,  +0.04,   0.00,  +0.04 ]
c008      [ -0.15,  -0.15,  -0.15,  +0.04,  +0.04,   0.00 ]
```

### Bias Value Rationale

| Region | Bias | Rationale |
|--------|:----:|-----------|
| **Intra-S01** (c001↔c002, c001↔c003, c002↔c003) | **+0.04** | Same intersection, overlapping FOVs, vehicles transit in 0-20s. Matches are common and should be slightly favoured. Conservative: +0.04 is well below the noise floor (pair-std ≈ 0.05-0.10). |
| **Intra-S02** (c006↔c007, c006↔c008, c007↔c008) | **+0.04** | Same as intra-S01. Same intersection geometry, similar transit times. |
| **Cross-scene** (any S01↔S02 pair) | **-0.15** | Physically separated intersections with no shared traffic flow. No GT matches observed across scenes. A −0.15 penalty effectively requires similarity ≥ 0.70 (vs 0.55 threshold) to create a cross-scene edge — this is intentionally harsh because cross-scene matches are near-impossible in CityFlowV2. |
| **Diagonal** (same camera) | **0.00** | Same-camera pairs are already filtered by the hard temporal constraint (Step 2a). Bias is irrelevant for these pairs. |

### Why These Specific Values

**+0.04 intra-scene**: 
- The graph threshold is 0.55. A +0.04 bias effectively lowers the threshold to 0.51 for intra-scene pairs.
- This recovers borderline true matches (similarity 0.51-0.55) that currently get fragmented.
- The value is conservative: with ~87 fragmented IDs in the error profile, even recovering 5-10 requires only a gentle nudge.
- Risk mitigation: +0.04 is small enough that it won't push false matches (similarity ~0.40-0.45) above threshold.

**-0.15 cross-scene**:
- No GT vehicles appear in both S01 and S02. Any cross-scene match is guaranteed false.
- A -0.15 penalty means a cross-scene pair needs raw similarity ≥ 0.70 to pass the 0.55 threshold.
- In practice, no genuine different-identity pair should have similarity 0.70+ after FIC whitening.
- If any cross-scene pair genuinely exceeds 0.70, it's almost certainly a same-model/same-color coincidence that we want to suppress anyway.
- Could be set even more aggressively (e.g., -0.50), but -0.15 is sufficient and leaves room for adjustment.

---

## Implementation Plan

### What Already Exists

The pipeline infrastructure is **already fully implemented**:

1. **Config**: `cityflowv2.yaml` already has:
   ```yaml
   camera_bias:
     enabled: true
     cid_bias_npy_path: "configs/datasets/cityflowv2_cid_bias.npy"
     iterations: 0
   ```

2. **Pipeline code**: `src/stage4_association/pipeline.py` Step 5b already:
   - Loads the `.npy` matrix
   - Loads camera name mapping from the `.json` sidecar
   - Applies additive bias to every pair in `combined_sim`
   - Handles missing camera mappings gracefully

3. **Data files**: `configs/datasets/cityflowv2_cid_bias.npy` and `.json` already exist (but contain the old GT-learned biases that caused -3.3pp).

### Changes Required

Only **two files** need to change — both are data files, no code changes:

#### Change 1: Replace `configs/datasets/cityflowv2_cid_bias.npy`

Create a new script `scripts/generate_topology_cid_bias.py` that:
1. Defines the 6×6 bias matrix with the values above
2. Saves it as `configs/datasets/cityflowv2_cid_bias.npy`
3. Saves the camera mapping as `configs/datasets/cityflowv2_cid_bias.json`

```python
"""Generate topology-based CID_BIAS matrix for CityFlowV2.

Unlike the GT-learned approach (compute_cid_bias.py), this encodes physical
camera layout priors: intra-scene cameras get a positive bias, cross-scene
cameras get a negative bias.
"""
import json
import numpy as np
from pathlib import Path

CAMERAS = ["S01_c001", "S01_c002", "S01_c003", "S02_c006", "S02_c007", "S02_c008"]

# Scene membership
S01 = {"S01_c001", "S01_c002", "S01_c003"}
S02 = {"S02_c006", "S02_c007", "S02_c008"}

INTRA_SCENE_BIAS = 0.04   # Positive: favour matches within same intersection
CROSS_SCENE_BIAS = -0.15  # Negative: suppress matches across separate intersections

def generate():
    n = len(CAMERAS)
    bias = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            ci, cj = CAMERAS[i], CAMERAS[j]
            same_scene = (ci in S01 and cj in S01) or (ci in S02 and cj in S02)
            bias[i, j] = INTRA_SCENE_BIAS if same_scene else CROSS_SCENE_BIAS

    out_dir = Path("configs/datasets")
    np.save(out_dir / "cityflowv2_cid_bias.npy", bias)
    with open(out_dir / "cityflowv2_cid_bias.json", "w") as f:
        json.dump({"cameras": CAMERAS}, f, indent=2)

    print(f"Saved CID_BIAS matrix ({n}x{n}) and camera mapping")
    print(f"Cameras: {CAMERAS}")
    print(f"Matrix:\n{bias}")

if __name__ == "__main__":
    generate()
```

#### Change 2: Verify config is correct (no changes expected)

The existing `cityflowv2.yaml` config is already correct:
```yaml
camera_bias:
  enabled: true
  cid_bias_npy_path: "configs/datasets/cityflowv2_cid_bias.npy"
  iterations: 0  # 0 = skip iterative learning, only use static .npy matrix
```

The `iterations: 0` is critical — it skips the iterative GT-learned bias approach that caused the original -3.3pp regression.

#### Change 3: Disable in v80-restored experiment config (keep baseline clean)

In `configs/experiments/vehicle_v80_restored_candidate.yaml`, `camera_bias.enabled` is already `false`. Leave it as-is so the baseline experiment config remains unchanged.

### No Pipeline Code Changes

The pipeline code in `src/stage4_association/pipeline.py` (lines ~540-575) already handles everything:

```python
# Step 5b: Camera distance bias adjustment (iterative)
camera_bias_cfg = stage_cfg.get("camera_bias", {})
if camera_bias_cfg.get("enabled", False):
    cid_bias_path = camera_bias_cfg.get("cid_bias_npy_path", "")
    if cid_bias_path and Path(cid_bias_path).exists():
        cid_bias_matrix = np.load(cid_bias_path).astype(np.float32)
        # ... loads .json mapping, applies additive bias to combined_sim
```

This code:
- ✅ Loads the `.npy` matrix
- ✅ Loads camera names from `.json` sidecar
- ✅ Maps camera IDs to matrix indices
- ✅ Applies `combined_sim[(i, j)] = sim + float(cid_bias_matrix[ci, cj])`
- ✅ Logs adjustment counts and unmapped cameras
- ✅ Falls back gracefully if cameras are missing from the mapping

---

## Camera ID Mapping

### Internal vs CityFlowV2 Labels

Our pipeline uses camera IDs in the format `S01_c001`, `S01_c002`, etc. — these match the CityFlowV2 directory structure exactly. The `.json` sidecar file maps these names to matrix indices:

```json
{
  "cameras": [
    "S01_c001",  // index 0
    "S01_c002",  // index 1
    "S01_c003",  // index 2
    "S02_c006",  // index 3
    "S02_c007",  // index 4
    "S02_c008"   // index 5
  ]
}
```

If camera IDs from Stage 1 don't match this mapping (e.g., `c001` instead of `S01_c001`), the pipeline's `cam2idx.get()` lookup returns `None` and the pair is skipped with a logged warning. The existing code handles this — **no camera ID remapping needed**.

---

## Execution Plan

### Step 1: Generate new topology-based `.npy` file
```bash
python scripts/generate_topology_cid_bias.py
```

### Step 2: Verify the matrix
```bash
python -c "
import numpy as np, json
bias = np.load('configs/datasets/cityflowv2_cid_bias.npy')
with open('configs/datasets/cityflowv2_cid_bias.json') as f:
    cams = json.load(f)['cameras']
print('Cameras:', cams)
print('Matrix:')
for i, ci in enumerate(cams):
    row = ' '.join(f'{bias[i,j]:+.2f}' for j in range(len(cams)))
    print(f'  {ci}: [{row}]')
"
```

Expected output:
```
Cameras: ['S01_c001', 'S01_c002', 'S01_c003', 'S02_c006', 'S02_c007', 'S02_c008']
Matrix:
  S01_c001: [+0.00 +0.04 +0.04 -0.15 -0.15 -0.15]
  S01_c002: [+0.04 +0.00 +0.04 -0.15 -0.15 -0.15]
  S01_c003: [+0.04 +0.04 +0.00 -0.15 -0.15 -0.15]
  S02_c006: [-0.15 -0.15 -0.15 +0.00 +0.04 +0.04]
  S02_c007: [-0.15 -0.15 -0.15 +0.04 +0.00 +0.04]
  S02_c008: [-0.15 -0.15 -0.15 +0.04 +0.04 +0.00]
```

### Step 3: Run local evaluation (CPU-only, stages 4-5)
```bash
python scripts/run_pipeline.py \
    --config configs/default.yaml configs/datasets/cityflowv2.yaml \
    --stages 4 5 \
    --override stage4.association.camera_bias.enabled=true
```

### Step 4: Compare against baseline
```bash
# Baseline (camera_bias disabled)
python scripts/run_pipeline.py \
    --config configs/default.yaml configs/datasets/cityflowv2.yaml \
    --stages 4 5 \
    --override stage4.association.camera_bias.enabled=false
```

### Step 5: If positive, deploy to Kaggle 10c notebook
Add the `.npy` and `.json` files to the Kaggle dataset, ensure the notebook config enables `camera_bias`.

---

## Sensitivity Analysis Plan

If the initial +0.04 / -0.15 values don't produce expected gains, sweep:

| Parameter | Range | Step |
|-----------|-------|------|
| `INTRA_SCENE_BIAS` | +0.02 to +0.08 | 0.02 |
| `CROSS_SCENE_BIAS` | -0.05 to -0.30 | 0.05 |

This is a 4×6 = 24 config sweep, easily run locally since only stages 4-5 are involved.

Key interactions to watch:
- **Graph threshold**: The effective per-pair threshold becomes `threshold - bias`. If `INTRA_SCENE_BIAS` is too high, it effectively lowers the threshold below the noise floor.
- **Bridge pruning**: The bridge_prune_margin (0.05) interacts with the bias. A +0.04 intra-scene bias means bridge edges in intra-scene need only 0.56 raw similarity (instead of 0.60) to survive pruning.
- **Gallery expansion**: The orphan_match_threshold (0.40) is applied to `combined_sim` which already includes the bias. No interaction issue — orphans at cross-scene boundaries will have their similarity reduced, making it harder to absorb them (desired behaviour).

---

## Risk Assessment

| Risk | Severity | Mitigation |
|------|:--------:|------------|
| Intra-scene bias too aggressive | Medium | Start conservative (+0.04), sweep upward only if fragmentation persists |
| Cross-scene bias kills rare true cross-scene matches | Low | CityFlowV2 GT has 0 cross-scene matches. Even if one existed, -0.15 still allows similarity ≥ 0.70 to pass |
| Camera ID mismatch | Low | Pipeline already handles unmapped cameras gracefully (skip + warn). Verify with Step 2 verification command |
| Interaction with FIC whitening | Low | FIC runs before CID_BIAS in the pipeline. The bias corrects for residual camera-pair effects that FIC doesn't fully remove. These are complementary, not conflicting |
| Regression if bias values are wrong | Medium | Easy to disable (`camera_bias.enabled=false`) and revert to baseline. No code changes to undo |

---

## Why This Will Work When the Previous Attempt Failed

| Previous (GT-learned, -3.3pp) | New (topology-based) |
|-------------------------------|---------------------|
| Learned from 464 GT-matched tracklets | Hard-coded from physical camera layout |
| Biases reflected model errors + noise | Biases reflect physical facts (adjacency, separation) |
| Centering shifted all similarities relative to noisy global mean | No centering — direct additive offsets |
| Complex: cluster → learn → re-adjust → re-cluster | Simple: load matrix, add to similarities, done |
| Overfitted to specific embeddings | Embedding-independent (works with any model) |
| Required GT labels at test time | Zero GT dependency |

The key insight is that CID_BIAS should encode **what we know about the world** (camera topology), not **what we estimate from data** (noisy similarity statistics). AIC22 winners hard-code their bias matrices based on camera metadata files, not learned bias.

---

## Files Modified

| File | Action | Description |
|------|--------|-------------|
| `scripts/generate_topology_cid_bias.py` | **CREATE** | Script to generate the topology-based `.npy` and `.json` files |
| `configs/datasets/cityflowv2_cid_bias.npy` | **REPLACE** | New 6×6 matrix with topology-based values |
| `configs/datasets/cityflowv2_cid_bias.json` | **NO CHANGE** | Camera list remains the same |
| `configs/datasets/cityflowv2.yaml` | **NO CHANGE** | Config already correct (`enabled: true`, `iterations: 0`) |
| `src/stage4_association/pipeline.py` | **NO CHANGE** | Existing Step 5b code handles everything |