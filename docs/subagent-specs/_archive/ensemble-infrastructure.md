# Ensemble Infrastructure Spec: 3-Model Score-Level Fusion

**Date**: 2026-03-31  
**Status**: Design  
**Goal**: Support ViT-B/16 CLIP + ResNet101-IBN-a (09g) + ResNeXt101-IBN-a (09h) ensemble with score-level fusion

---

## 1. Architecture Decision: Score-Level Fusion

**Decision**: Score-level fusion (3 separate similarity matrices blended with learned weights).

**Rationale**:
- Stage 4 already implements score-level fusion for the secondary model — extending to a tertiary model is natural
- Feature-level fusion (concat → PCA) was the legacy approach (`save_separate=false`) and mixes uncalibrated embedding spaces, requiring careful normalization
- Score-level fusion lets each model independently benefit from its own PCA whitening, FIC normalization, and query expansion
- AIC22 2nd place (BOE) and AIC21 1st place (DAMO) both use score-level fusion across their 3-model ensemble
- Each model's contribution weight can be tuned independently without retraining PCA

**Formula**:
```
final_appearance_sim(i, j) = w1 * sim_vit(i,j) + w2 * sim_resnet(i,j) + w3 * sim_resnext(i,j)
where w1 + w2 + w3 = 1.0
```

Initial weights based on model quality: w1=0.5 (ViT, mAP=80.14%), w2=0.25 (ResNet101), w3=0.25 (ResNeXt101).

---

## 2. Current State: What Already Exists

### Stage 2 — Feature Extraction (`src/stage2_features/pipeline.py`)
- **Primary model**: `vehicle_reid` — loaded from `stage2.reid.vehicle` config
- **Secondary model**: `vehicle_reid2` — loaded from `stage2.reid.vehicle2` config when `enabled=true`
- **Secondary saving**: When `save_separate=true`, secondary embeddings get independent camera-BN + PCA whitening and are saved to `embeddings_secondary.npy`
- **Limitation**: Only supports 2 models (primary + one secondary)

### Stage 4 — Association (`src/stage4_association/pipeline.py`)
- **Secondary fusion**: `stage4.association.secondary_embeddings` — loads one `.npy` file, applies FIC whitening, blends at score level with configurable weight
- **Limitation**: Only supports 1 secondary embedding file and 1 weight

### Config (`configs/default.yaml`, `configs/datasets/cityflowv2.yaml`)
- `stage2.reid.vehicle2` exists but is hardcoded to a single secondary model
- `stage4.association.secondary_embeddings` supports a single path + weight

---

## 3. Required Changes

### 3.1 Config Schema Changes

#### `configs/default.yaml` — Stage 2 Section

**Current** (`stage2.reid`):
```yaml
vehicle2:
  enabled: false
  save_separate: true
  model_name: "resnet101_ibn_a"
  weights_path: "..."
  embedding_dim: 2048
  input_size: [384, 384]
  clip_normalization: false
```

**New** — Add `vehicle3` alongside `vehicle2`:
```yaml
vehicle2:
  enabled: false
  save_separate: true
  model_name: "resnet101_ibn_a"
  weights_path: "models/reid/resnet101ibn_cityflowv2_dmt_best.pth"
  embedding_dim: 2048
  input_size: [384, 384]
  clip_normalization: false
vehicle3:
  enabled: false
  save_separate: true
  model_name: "resnext101_ibn_a"
  weights_path: "models/reid/resnext101ibn_cityflowv2_dmt_best.pth"
  embedding_dim: 2048
  input_size: [384, 384]
  clip_normalization: false
```

#### `configs/default.yaml` — Stage 2 PCA Section

**Add** a tertiary PCA model path:
```yaml
pca:
  enabled: true
  n_components: 384
  pca_model_path: "models/reid/pca_transform.pkl"
  secondary_pca_model_path: "models/reid/pca_transform_secondary.pkl"
  tertiary_pca_model_path: "models/reid/pca_transform_tertiary.pkl"  # NEW
```

#### `configs/default.yaml` — Stage 4 Section

**Current** (`stage4.association`):
```yaml
secondary_embeddings:
  path: ""
  weight: 0.3
```

**New** — Replace with a list-based approach:
```yaml
secondary_embeddings:
  path: ""
  weight: 0.3
tertiary_embeddings:        # NEW
  path: ""
  weight: 0.0              # 0 = disabled
```

#### `configs/datasets/cityflowv2.yaml` — Overrides

```yaml
stage2:
  reid:
    vehicle2:
      enabled: true
      save_separate: true
      model_name: "resnet101_ibn_a"
      weights_path: "models/reid/resnet101ibn_cityflowv2_dmt_best.pth"
      embedding_dim: 2048
      input_size: [384, 384]
      clip_normalization: false
    vehicle3:
      enabled: true
      save_separate: true
      model_name: "resnext101_ibn_a"
      weights_path: "models/reid/resnext101ibn_cityflowv2_dmt_best.pth"
      embedding_dim: 2048
      input_size: [384, 384]
      clip_normalization: false
  pca:
    tertiary_pca_model_path: "models/reid/pca_transform_tertiary.pkl"

stage4:
  association:
    secondary_embeddings:
      path: "data/outputs/run_latest/stage2/embeddings_secondary.npy"
      weight: 0.25
    tertiary_embeddings:
      path: "data/outputs/run_latest/stage2/embeddings_tertiary.npy"
      weight: 0.25
```

---

### 3.2 Stage 2 Changes (`src/stage2_features/pipeline.py`)

**File**: `src/stage2_features/pipeline.py`

#### Change 1: Load tertiary model (around line 185-210)

After the `vehicle_reid2` loading block (~L185-210), add an analogous block for `vehicle3`:

```python
# --- Optional third vehicle ReID model for ensemble ---
vehicle_reid3: Optional[ReIDModel] = None
vehicle3_cfg = stage_cfg.reid.get("vehicle3", {})
if vehicle3_cfg.get("enabled", False):
    weights_path3 = vehicle3_cfg.get("weights_path")
    if weights_path3 and Path(weights_path3).exists():
        vehicle_reid3 = ReIDModel(
            model_name=vehicle3_cfg.get("model_name", "resnext101_ibn_a"),
            weights_path=weights_path3,
            embedding_dim=vehicle3_cfg.get("embedding_dim", 2048),
            input_size=tuple(vehicle3_cfg.get("input_size", [384, 384])),
            device=stage_cfg.reid.device,
            half=stage_cfg.reid.half,
            flip_augment=flip_augment,
            color_augment=color_augment,
            num_cameras=vehicle3_cfg.get("num_cameras", 0),
            clip_normalization=vehicle3_cfg.get("clip_normalization", False),
        )
        logger.info(
            f"Tertiary ensemble ReID: {vehicle3_cfg.get('model_name')}"
        )
```

#### Change 2: Track tertiary embeddings (around line 255)

Add `all_tertiary_embeddings: List[Optional[np.ndarray]] = []` alongside `all_secondary_embeddings`.

Add `vehicle3_separate = vehicle3_cfg.get("save_separate", False) and vehicle_reid3 is not None`.

#### Change 3: Extract tertiary features per tracklet (around line 350-365)

After the `reid2` extraction block, add:

```python
# Ensemble: extract from third model
raw_embedding3 = None
if reid3 is not None:
    raw_embedding3 = reid3.get_tracklet_embedding_from_scored_crops(
        scored_crops, cam_id=sie_cam_id, quality_temperature=quality_temperature
    )
```

Where `reid3` is assigned alongside `reid2`:
```python
reid3 = vehicle_reid3 if tracklet.class_id not in PERSON_CLASSES else None
```

Store: `all_tertiary_embeddings.append(raw_embedding3)` alongside secondary.

#### Change 4: Tertiary post-processing — camera-BN + PCA + L2 norm (around line 500-560)

Duplicate the `sec_matrix` handling block for `tert_matrix`:
- Camera-aware batch normalize separately
- PCA whiten with its own `tertiary_pca_model_path`
- L2 normalize
- Save to `embeddings_tertiary.npy`

#### Change 5: Save tertiary embeddings (around line 590)

```python
if vehicle3_separate and tert_matrix is not None:
    tert_matrix = l2_normalize(tert_matrix)
    tert_path = output_dir / "embeddings_tertiary.npy"
    np.save(tert_path, tert_matrix.astype(np.float32))
    logger.info(f"Tertiary embeddings saved: {tert_matrix.shape} -> {tert_path}")
```

---

### 3.3 Stage 2 Model Support (`src/stage2_features/reid_model.py`)

**File**: `src/stage2_features/reid_model.py`

#### Change: Add ResNeXt101-IBN-a routing (around line 105-110)

In `_build_model()`, add a case for `resnext101_ibn_a`:

```python
def _build_model(self, model_name: str, weights_path: Optional[str]):
    if self.is_transreid:
        return self._build_transreid(weights_path)
    if model_name.lower() == "resnet101_ibn_a":
        return self._build_resnet101_ibn(weights_path)
    if model_name.lower() == "resnext101_ibn_a":          # NEW
        return self._build_resnext101_ibn(weights_path)    # NEW
    return self._build_torchreid(model_name, weights_path)
```

#### Change: Add `_build_resnext101_ibn()` method

New method modeled on `_build_resnet101_ibn()`. Two options:
- **Option A (recommended)**: Add a `ReIDModelResNeXt101IBN` class to `src/training/model.py` that mirrors `ReIDModelResNet101IBN` but uses `resnext101_32x8d` as the base. The 09h notebook already defines this architecture.
- **Option B**: If the 09h checkpoint is compatible with `_build_resnet101_ibn()` (same IBN-a + GeM + BNNeck architecture but different backbone), we can parameterize the backbone construction.

Recommended: **Option A** — create a dedicated class since ResNeXt has grouped convolutions (32×8d) and different layer structure.

---

### 3.4 Training Model (`src/training/model.py`)

**File**: `src/training/model.py`

Add `ReIDModelResNeXt101IBN` class (approximately 50 lines), mirroring `ReIDModelResNet101IBN` but using:
- `torchvision.models.resnext101_32x8d` as base
- IBN-a layers on layer1, layer2, layer3  
- GeM pooling + BNNeck (same as ResNet101-IBN)
- Same forward() / extract_features() interface

The 09h Kaggle notebook (`notebooks/kaggle/09h_resnext101ibn_dmt/generate_09h_notebook.py`) already contains the architecture definition — extract and formalize into the training module.

---

### 3.5 Stage 4 Changes (`src/stage4_association/pipeline.py`)

**File**: `src/stage4_association/pipeline.py`

#### Change 1: Load tertiary embeddings (around line 175-200)

After the `sec_embeddings` loading block, add an analogous block:

```python
# Optional: load tertiary embeddings for 3-model score-level fusion
tert_cfg = stage_cfg.get("tertiary_embeddings", {})
tert_path = tert_cfg.get("path", "")
tert_embeddings: Optional[np.ndarray] = None
tert_weight = 0.0
if tert_path and Path(tert_path).exists():
    tert_raw = np.load(tert_path).astype(np.float32)
    if tert_raw.shape[0] == n:
        tert_weight = float(tert_cfg.get("weight", 0.25))
        tert_norms = np.linalg.norm(tert_raw, axis=1, keepdims=True)
        tert_embeddings = tert_raw / np.maximum(tert_norms, 1e-8)
        logger.info(
            f"Tertiary embeddings loaded: {tert_embeddings.shape[1]}D, "
            f"weight={tert_weight:.2f} (score-level fusion)"
        )
        if fic_cfg.get("enabled", False):
            tert_embeddings = per_camera_whiten(
                tert_embeddings, camera_ids,
                regularisation=float(fic_cfg.get("regularisation", 3.0)),
                min_samples=int(fic_cfg.get("min_samples", 5)),
            )
            logger.info("Applied FIC whitening to tertiary embeddings")
    else:
        logger.warning(f"Tertiary embeddings shape mismatch: {tert_raw.shape[0]} vs {n}")
```

#### Change 2: Adjust weight normalization

When all 3 models are active, the primary weight must be derived:
```python
primary_weight = 1.0 - sec_weight - tert_weight
```

Add a sanity check:
```python
if primary_weight < 0.1:
    logger.warning(
        f"Primary embedding weight too low ({primary_weight:.2f}). "
        f"Clamping to ensure primary model contributes."
    )
    # Re-normalize weights
    total = 1.0 + sec_weight + tert_weight
    sec_weight /= total
    tert_weight /= total
    primary_weight = 1.0 / total
```

#### Change 3: Blend tertiary scores (around line 420, after Step 3b)

After the existing secondary blending block:

```python
# Step 3c: Score-level fusion with tertiary embeddings
if tert_embeddings is not None and tert_weight > 0:
    logger.info(f"Blending tertiary appearance sim (weight={tert_weight:.2f})...")
    primary_w = 1.0 - sec_weight - tert_weight
    for (i, j) in appearance_sim:
        pri_sim = appearance_sim[(i, j)]
        tert_sim = float(np.dot(tert_embeddings[i], tert_embeddings[j]))
        # Recompute with 3-way blend
        if sec_embeddings is not None and sec_weight > 0:
            # Already blended primary+secondary; need to re-derive
            # Original: blended = (1-sw)*pri + sw*sec  →  pri = (blended - sw*sec) / (1-sw)
            sec_sim = float(np.dot(sec_embeddings[i], sec_embeddings[j]))
            original_pri = (pri_sim - sec_weight * sec_sim) / max(1 - sec_weight, 1e-8)
            appearance_sim[(i, j)] = primary_w * original_pri + sec_weight * sec_sim + tert_weight * tert_sim
        else:
            appearance_sim[(i, j)] = (1.0 - tert_weight) * pri_sim + tert_weight * tert_sim
    logger.info(f"3-model ensemble blending complete (w_pri={primary_w:.2f}, w_sec={sec_weight:.2f}, w_tert={tert_weight:.2f})")
```

**Important note**: The current secondary blending modifies `appearance_sim` in-place, making it harder to add a third model cleanly. A cleaner refactor would compute all three similarity values independently before blending:

```python
# CLEANER APPROACH (recommended refactor):
if sec_embeddings is not None or tert_embeddings is not None:
    primary_w = 1.0 - sec_weight - tert_weight
    blended = 0
    for (i, j) in list(appearance_sim.keys()):
        sim = primary_w * appearance_sim[(i, j)]
        if sec_embeddings is not None and sec_weight > 0:
            sim += sec_weight * float(np.dot(sec_embeddings[i], sec_embeddings[j]))
        if tert_embeddings is not None and tert_weight > 0:
            sim += tert_weight * float(np.dot(tert_embeddings[i], tert_embeddings[j]))
        appearance_sim[(i, j)] = sim
        blended += 1
    logger.info(
        f"Ensemble fusion: {blended} pairs blended "
        f"(w_pri={primary_w:.2f}, w_sec={sec_weight:.2f}, w_tert={tert_weight:.2f})"
    )
```

This replaces both the current Step 3b secondary blending AND the new tertiary blending with a single unified block. **Recommended approach.**

---

### 3.6 Kaggle 10a Notebook Changes

**File**: `notebooks/kaggle/10a_stages012/mtmc-10a-stages-0-2-tracking-reid-features.ipynb`

#### Current State
- Cell a14 runs the pipeline with `--override stage2.reid.vehicle2.enabled=false`
- This disables the secondary model entirely on Kaggle

#### Required Changes

**Option A (Sequential extraction — simpler, safer)**:
Run the pipeline 3 times, each extracting one model's features:
1. Run 1: Primary ViT only (existing), save to `stage2/`
2. Run 2: Load vehicle2, extract, save to `stage2/embeddings_secondary.npy`
3. Run 3: Load vehicle3, extract, save to `stage2/embeddings_tertiary.npy`

This avoids GPU OOM on P100 (16GB) which would struggle with 3 models loaded simultaneously.

**Option B (Integrated — requires GPU memory management)**:
Enable all 3 models in one pipeline run. The pipeline already loads models sequentially per tracklet, so memory impact is 3× model weights (~500MB each) but only 1 batch of activations at a time. P100 should handle this.

Change the override to:
```python
"--override", "stage2.reid.vehicle2.enabled=true",
"--override", "stage2.reid.vehicle3.enabled=true",
```

And add model weight paths for the Kaggle environment:
```python
"--override", "stage2.reid.vehicle2.weights_path=/kaggle/input/09g-weights/best_model.pth",
"--override", "stage2.reid.vehicle3.weights_path=/kaggle/input/09h-weights/best_model.pth",
```

**Recommendation**: Option B (integrated) is preferred. The models are loaded once and share the same crop extraction pass which avoids redundant I/O. P100 has 16GB VRAM which can hold all 3 models (~1.5GB total) plus a batch of activations. If OOM occurs, we can add a `torch.cuda.empty_cache()` between model forward passes.

#### Kaggle Datasets
- Need to create Kaggle datasets for 09g and 09h model weights
- These become input datasets for 10a notebook's `kernel-metadata.json`

---

## 4. File Change Summary

| File | Change | Scope |
|------|--------|-------|
| `configs/default.yaml` | Add `vehicle3` config block, `tertiary_pca_model_path`, `tertiary_embeddings` | ~15 lines added |
| `configs/datasets/cityflowv2.yaml` | Add `vehicle3` override, `tertiary_embeddings` override | ~15 lines added |
| `src/stage2_features/reid_model.py` | Add `resnext101_ibn_a` routing in `_build_model()` + `_build_resnext101_ibn()` method | ~40 lines added |
| `src/training/model.py` | Add `ReIDModelResNeXt101IBN` class | ~60 lines added |
| `src/stage2_features/pipeline.py` | Load vehicle3, extract tertiary features, post-process + save | ~80 lines added |
| `src/stage4_association/pipeline.py` | Load tertiary embeddings, unified 3-way score blending | ~40 lines added (replaces existing ~15 lines) |
| `notebooks/kaggle/10a_stages012/` | Enable vehicle2+vehicle3, add weight paths, add datasets | Cell edits + metadata |

**Total**: ~250 lines of new code, ~15 lines modified

---

## 5. Integration Test Plan

### Unit Tests

1. **`tests/test_stage2/test_ensemble_extraction.py`** (new)
   - Test that `vehicle_reid3` is loaded when `vehicle3.enabled=true`
   - Test that tertiary embeddings are saved to `embeddings_tertiary.npy`
   - Test shape consistency: primary, secondary, tertiary all have same N (tracklet count)
   - Test that tertiary PCA uses its own model (not shared with primary/secondary)

2. **`tests/test_stage4/test_ensemble_fusion.py`** (new)
   - Test 3-way score blending with known similarity values
   - Test weight normalization (weights sum to 1.0)
   - Test graceful degradation when tertiary embeddings are missing (falls back to 2-model)
   - Test graceful degradation when both secondary + tertiary are missing (primary only)

3. **`tests/test_stage2/test_reid_model.py`** (extend)
   - Test `_build_model("resnext101_ibn_a", None)` constructs without error
   - Test output shape matches `embedding_dim=2048`

### Smoke Tests

4. **Local smoke test** (CPU, 3 tracklets):
   ```bash
   python scripts/run_pipeline.py --config configs/default.yaml \
     --dataset configs/datasets/cityflowv2.yaml \
     --smoke-test \
     --override stage2.reid.vehicle2.enabled=true \
     --override stage2.reid.vehicle3.enabled=true \
     --override stage2.reid.device=cpu
   ```
   Verify: `embeddings.npy`, `embeddings_secondary.npy`, `embeddings_tertiary.npy` all created with matching row counts.

5. **Kaggle integration test** (P100):
   Run updated 10a notebook, verify all 3 embedding files are in the checkpoint archive.

### Ablation Validation

6. **Weight sweep** (on Kaggle via 10c):
   - (0.5, 0.25, 0.25) — baseline equal secondary/tertiary
   - (0.5, 0.3, 0.2) — bias toward ResNet101
   - (0.4, 0.3, 0.3) — reduced ViT dominance
   - (0.6, 0.2, 0.2) — higher ViT trust
   - (1.0, 0.0, 0.0) — primary only (regression check against current best)

---

## 6. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| P100 OOM with 3 models loaded | Pipeline crash | Load/unload models sequentially; `torch.cuda.empty_cache()` between models |
| Weak ResNet101/ResNeXt101 features hurt ensemble | IDF1 regression | Weight sweep with w=0.0 fallback; always test primary-only baseline |
| PCA dimension mismatch across models | Shape errors in stage4 | Each model gets its own PCA; verify all produce same n_components |
| 10a runtime increase | Kaggle 12hr limit | ~2.5× feature extraction time; within budget (currently ~3hr) |
| N mismatch between embedding files | Stage4 crash | Hard assert at load time: `assert sec_raw.shape[0] == tert_raw.shape[0] == n` |

---

## 7. Implementation Order

1. **`src/training/model.py`** — Add `ReIDModelResNeXt101IBN` class
2. **`src/stage2_features/reid_model.py`** — Add `resnext101_ibn_a` routing
3. **`configs/default.yaml`** — Add `vehicle3`, `tertiary_pca_model_path`, `tertiary_embeddings`
4. **`configs/datasets/cityflowv2.yaml`** — Add overrides
5. **`src/stage2_features/pipeline.py`** — Load + extract + post-process + save tertiary
6. **`src/stage4_association/pipeline.py`** — Unified 3-way score fusion
7. **Unit tests** — Verify shapes, weights, fallbacks
8. **Local smoke test** — CPU end-to-end with 3 models
9. **10a notebook** — Enable 3-model extraction on Kaggle
10. **10c notebook** — Weight sweep experiment

---

## 8. Performance Expectations

Based on SOTA analysis:
- AIC22 2nd (BOE) with 3 IBN models: IDF1=0.8437
- AIC21 1st (DAMO) with 3 IBN models: IDF1=0.8095
- Our current 1-model: MTMC IDF1=0.775

**Expected gain from 3-model ensemble**: +2-5pp IDF1, depending on model quality. The ResNet101 and ResNeXt101 models need to reach >55% mAP on CityFlowV2 eval to contribute meaningfully. Below 45% mAP, they may add noise rather than signal.

**Key dependency**: 09g (ResNet101-IBN-a DMT) and 09h (ResNeXt101-IBN-a DMT) training must complete with adequate mAP before ensemble benefits can materialize.