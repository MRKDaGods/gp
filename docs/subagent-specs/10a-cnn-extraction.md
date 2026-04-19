# 10a CNN Secondary Model Extraction — CLIP RN50x4

> **Status**: Ready for implementation  
> **Approach**: Standalone CNN extraction cell in 10a (Approach A — no pipeline changes)  
> **Replaces**: LAION-2B CLIP vehicle2 slot (confirmed dead end: 10c v56 = -0.5pp)

## Motivation

10c v56 proved that two CLIP ViT-B/16 variants (OpenAI + LAION-2B) are too correlated for score-level fusion (-0.5pp). The ensemble needs **architecturally diverse** models — specifically a CNN backbone — to provide complementary identity signal. CLIP RN50x4 is the chosen CNN candidate (see `09m-clip-cnn-training.md` for training spec).

## Design: Approach A — Standalone Cell in 10a

After the main `run_pipeline.py --stages 0,1,2` call completes, a **new code cell** performs CNN feature extraction independently. This avoids modifying the stage2 pipeline code and keeps the change self-contained.

### Why Not Approach B (Extend Stage2 Pipeline)

The stage2 `ReIDModel` class routes models through `_build_model()` which dispatches to either `_build_transreid()` or `_build_torchreid()`. Adding CLIP RN50x4 would require:
- Installing `open_clip_torch` as a project dependency
- Adding a new `_build_clip_rn50x4()` method in `reid_model.py`
- Handling the different forward() signature (no `cam_ids` parameter)
- Handling different input size (288×288 vs 256×256)
- Risk of breaking existing vehicle1/vehicle2 extraction

Approach A avoids all of this. The CNN extraction lives entirely in the notebook cell, reusing project utilities for crop extraction and post-processing.

## Changes Required

### 1. Kernel Metadata (`notebooks/kaggle/10a_stages012/kernel-metadata.json`)

Add the 09m kernel output as a data source:

```json
{
  "kernel_sources": [
    "gumfreddy/09l-transreid-laion-2b-training",
    "gumfreddy/09m-clip-rn50x4-vehicle-reid"
  ]
}
```

> **Note**: Keep the 09l source for now (other notebooks may reference it). It does no harm to mount it.

### 2. Pip Install Cell (Cell 5, lines 72–102)

Add `open_clip_torch` to the pip install list. Insert after the `timm` install:

```python
pip("open_clip_torch")
```

Also add to the dependency check cell (Cell 6, lines 105–132):

```python
("open_clip", "open_clip"),
```

### 3. Mount Weights Cell (Cell 8, lines 139–207)

Add CNN weight copy logic after the existing LAION-2B copy block (around line 200):

```python
# --- Copy CLIP RN50x4 CNN model from 09m kernel output ---
CNN_RN50X4_SRC = Path("/kaggle/input/09m-clip-rn50x4-vehicle-reid/exported_models/clip_rn50x4_cityflowv2.pth")
CNN_RN50X4_DST = PROJECT / "models" / "reid" / "clip_rn50x4_cityflowv2.pth"
if CNN_RN50X4_SRC.exists():
    shutil.copy2(str(CNN_RN50X4_SRC), str(CNN_RN50X4_DST))
    print(f"✓ CLIP RN50x4 CNN: {CNN_RN50X4_DST.name} ({CNN_RN50X4_DST.stat().st_size/1024**2:.0f} MB)")
else:
    print(f"⚠ CLIP RN50x4 weights not found at {CNN_RN50X4_SRC} — CNN ensemble disabled")
    CNN_RN50X4_DST = None
```

### 4. Remove LAION-2B Vehicle2 Overrides (Cell 17, lines 332–385)

Remove the `vehicle2` config overrides from the `run_pipeline.py` command. The LAION-2B secondary extraction was proven to hurt (-0.5pp in 10c v56). Delete these lines:

```python
# DELETE THIS BLOCK:
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
```

### 5. New Cell: CNN Secondary Feature Extraction (Insert Between Cell 17 and Cell 18)

This is the core new cell. Insert a **Markdown cell** + **Code cell** after the "Run Stages 0-2" cell and before the "Save Checkpoint" section.

#### 5a. Markdown Cell

```markdown
## 4b. CNN Secondary Feature Extraction (CLIP RN50x4)

Extract 640D CNN embeddings from the same tracklet crops using the CLIP RN50x4
model trained in 09m. These embeddings are saved as `embeddings_secondary.npy`
for score-level fusion in 10c.

The CNN's local receptive fields (convolutions) complement the primary ViT's
global attention, providing the architectural diversity needed for effective ensemble.
```

#### 5b. Code Cell — Full Implementation

```python
import time as _cnn_time
_cnn_t0 = _cnn_time.time()

# --- Guard: skip if CNN weights not available ---
if CNN_RN50X4_DST is None or not CNN_RN50X4_DST.exists():
    print("⚠ CLIP RN50x4 weights not available — skipping CNN extraction")
else:
    import open_clip
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    import json as _json
    import cv2
    from pathlib import Path
    from loguru import logger

    from src.core.io_utils import load_tracklets_by_camera
    from src.stage2_features.crop_extractor import CropExtractor
    from src.stage2_features.pipeline import _load_frames_for_camera
    from src.stage2_features.embeddings import camera_aware_batch_normalize, l2_normalize
    from src.stage2_features.pca_whitening import PCAWhitener

    # ---- 1. Define model class (inline — avoids modifying pipeline code) ----
    class CLIPResNetReID(nn.Module):
        """CLIP RN50x4 visual encoder + BNNeck for ReID inference."""

        def __init__(self, embed_dim=640):
            super().__init__()
            clip_model = open_clip.create_model('RN50x4', pretrained=None)
            self.backbone = clip_model.visual  # ModifiedResNet + AttentionPool2d
            self.feat_dim = embed_dim
            self.bn = nn.BatchNorm1d(self.feat_dim)
            self.bn.bias.requires_grad_(False)
            self.cls_head = nn.Linear(self.feat_dim, 1, bias=False)  # dummy, not used

        def forward(self, x):
            f = self.backbone(x)       # (B, 640)
            bn = self.bn(f)            # BNNeck
            return F.normalize(bn, p=2, dim=1)  # L2-normalized for retrieval

    # ---- 2. Load trained weights ----
    cnn_model = CLIPResNetReID(embed_dim=640)
    ckpt = torch.load(str(CNN_RN50X4_DST), map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    missing, unexpected = cnn_model.load_state_dict(state, strict=False)
    # Only cls_head weight mismatch is expected (num_classes differs)
    critical_missing = [k for k in missing if not k.startswith("cls_head")]
    if critical_missing:
        print(f"⚠ CNN model missing critical keys: {critical_missing}")
    cnn_model.eval().to(DEVICE)
    if DEVICE == "cuda":
        cnn_model.half()
    print(f"✓ CLIP RN50x4 loaded: 640D, 288×288, device={DEVICE}")

    # ---- 3. CLIP normalization constants ----
    _CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    _CLIP_STD  = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    CNN_H, CNN_W = 288, 288

    def preprocess_cnn(crops):
        """Preprocess BGR crops for CLIP RN50x4: resize 288×288, CLIP normalize."""
        processed = []
        for img in crops:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (CNN_W, CNN_H), interpolation=cv2.INTER_CUBIC)
            img = img.astype(np.float32) / 255.0
            img = (img - _CLIP_MEAN) / _CLIP_STD
            img = np.transpose(img, (2, 0, 1))  # HWC → CHW
            processed.append(img)
        return torch.from_numpy(np.stack(processed, axis=0))

    @torch.no_grad()
    def extract_cnn_batch(batch_crops, flip_augment=True):
        """Extract CNN embeddings for a batch with optional flip augmentation."""
        tensor = preprocess_cnn(batch_crops).to(DEVICE)
        if DEVICE == "cuda":
            tensor = tensor.half()
        features = cnn_model(tensor).float().cpu().numpy()
        n_views = 1

        if flip_augment:
            flipped = [cv2.flip(c, 1) for c in batch_crops]
            flip_tensor = preprocess_cnn(flipped).to(DEVICE)
            if DEVICE == "cuda":
                flip_tensor = flip_tensor.half()
            features = features + cnn_model(flip_tensor).float().cpu().numpy()
            n_views += 1

        return features / n_views

    def extract_cnn_features(crops, batch_size=64):
        """Extract CNN features for all crops with batching."""
        if not crops:
            return np.empty((0, 640), dtype=np.float32)
        all_embs = []
        for i in range(0, len(crops), batch_size):
            batch = crops[i:i + batch_size]
            try:
                all_embs.append(extract_cnn_batch(batch))
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    half_bs = max(1, len(batch) // 2)
                    for j in range(0, len(batch), half_bs):
                        all_embs.append(extract_cnn_batch(batch[j:j + half_bs]))
                else:
                    raise
        return np.concatenate(all_embs, axis=0)

    # ---- 4. Load tracklets and extract CNN embeddings in index_map order ----
    stage1_dir = RUN_DIR / "stage1"
    stage2_dir = RUN_DIR / "stage2"
    s0_dir = RUN_DIR / "stage0"

    # Load the index_map to guarantee identical ordering with primary embeddings
    with open(stage2_dir / "embedding_index.json") as f:
        index_map = _json.load(f)
    print(f"Index map: {len(index_map)} tracklets")

    # Load all tracklets from stage1
    tracklets_by_camera = load_tracklets_by_camera(stage1_dir)

    # Build lookup: (camera_id, track_id) → Tracklet
    tracklet_lookup = {}
    for cam_id, tracklets in tracklets_by_camera.items():
        for t in tracklets:
            tracklet_lookup[(cam_id, t.track_id)] = t

    # Initialize crop extractor with same params as primary extraction
    crop_extractor = CropExtractor(
        min_area=256,
        padding_ratio=0.05,
        samples_per_tracklet=48,
        min_quality=0.05,
        laplacian_min_var=0.0,
    )

    # Pre-load frames by camera (same strategy as stage2 pipeline)
    frames_cache = {}
    for cam_id in tracklets_by_camera:
        needed_ids = set()
        for t in tracklets_by_camera[cam_id]:
            for tf in t.frames:
                needed_ids.add(tf.frame_id)
        frames_cache[cam_id] = _load_frames_for_camera(s0_dir, cam_id, needed_ids)
        print(f"  {cam_id}: {len(frames_cache[cam_id])} frames loaded")

    # Extract CNN embeddings in exact index_map order
    cnn_embeddings = []
    camera_ids = []
    failed = 0

    for entry in index_map:
        cam_id = entry["camera_id"]
        track_id = entry["track_id"]
        tracklet = tracklet_lookup.get((cam_id, track_id))

        if tracklet is None:
            # Tracklet was in index_map but not found — fill with zeros
            cnn_embeddings.append(np.zeros(640, dtype=np.float32))
            camera_ids.append(cam_id)
            failed += 1
            continue

        # Extract quality-scored crops using same strategy as primary
        cam_frames = frames_cache.get(cam_id, {})
        scored_crops = crop_extractor.extract_crops_from_frames(tracklet, cam_frames)

        if not scored_crops:
            cnn_embeddings.append(np.zeros(640, dtype=np.float32))
            camera_ids.append(cam_id)
            failed += 1
            continue

        # Extract per-crop CNN embeddings
        crops = [sc.image for sc in scored_crops]
        qualities = np.array([sc.quality for sc in scored_crops], dtype=np.float32)
        crop_embeddings = extract_cnn_features(crops)

        # Quality-weighted temporal attention pooling (same as primary)
        quality_temperature = 3.0
        weights = np.exp(qualities * quality_temperature)
        weights = weights / weights.sum()
        pooled = (crop_embeddings * weights[:, np.newaxis]).sum(axis=0)

        cnn_embeddings.append(pooled)
        camera_ids.append(cam_id)

    cnn_matrix = np.stack(cnn_embeddings, axis=0)  # (N, 640)
    print(f"\nCNN raw embeddings: {cnn_matrix.shape}, failed={failed}/{len(index_map)}")

    # ---- 5. Post-processing: Camera-BN → PCA whitening → L2 normalize ----
    # Camera-aware batch normalization (same as primary pipeline uses)
    cnn_matrix = camera_aware_batch_normalize(cnn_matrix, camera_ids)
    print(f"✓ Camera-BN applied")

    # PCA whitening: 640D → 384D (match primary dimensionality)
    pca_components = 384
    cnn_pca = PCAWhitener(n_components=pca_components)
    cnn_pca.fit(cnn_matrix)
    cnn_pca_path = str(stage2_dir / "pca_transform_cnn_secondary.pkl")
    cnn_pca.save(cnn_pca_path)
    cnn_matrix = cnn_pca.transform(cnn_matrix)
    print(f"✓ PCA: 640D → {pca_components}D")

    # L2 normalize
    cnn_matrix = l2_normalize(cnn_matrix).astype(np.float32)
    print(f"✓ L2-normalized: {cnn_matrix.shape}, dtype={cnn_matrix.dtype}")

    # ---- 6. Save as embeddings_secondary.npy ----
    sec_path = stage2_dir / "embeddings_secondary.npy"
    np.save(sec_path, cnn_matrix)
    print(f"\n✓ CNN secondary embeddings saved: {sec_path}")
    print(f"  Shape: {cnn_matrix.shape}")
    print(f"  Norm check (row 0): {np.linalg.norm(cnn_matrix[0]):.4f}")

    # Cleanup: free GPU memory before checkpoint packaging
    del cnn_model, cnn_matrix
    torch.cuda.empty_cache()

    _cnn_elapsed = _cnn_time.time() - _cnn_t0
    print(f"\n✓ CNN extraction done in {_cnn_elapsed/60:.1f} min")
```

### 6. No Changes Needed in 10c

10c already:
- Auto-detects `embeddings_secondary.npy` at `RUN_DIR / "stage2" / "embeddings_secondary.npy"`
- Sets `FUSION_WEIGHT = 0.30` when secondary embeddings exist
- Passes overrides: `stage4.association.secondary_embeddings.path={path}` and `stage4.association.secondary_embeddings.weight={weight}`
- Stage4 applies FIC whitening to secondary embeddings independently

The **only change** is that secondary embeddings now come from a **CNN** (architecturally diverse) instead of another ViT (correlated). No config changes needed.

### Suggested 10c Sweep Weights

Once deployed, 10c should sweep fusion weights to find the optimal blend:

```python
# In 10c sweep cell, test these CNN fusion weights:
FUSION_WEIGHT_VALUES = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
```

Start with `0.20` as the default (lower than ViT-ViT since CNN features may be less calibrated). The primary model at 80.14% mAP is significantly stronger than the CNN secondary, so primary-heavy weighting is expected to be optimal.

## Technical Details

### Embedding Format

| Property | Primary (ViT-B/16) | CNN Secondary (RN50x4) |
|----------|:-:|:-:|
| Raw dim | 768 (1536 with concat_patch) | **640** |
| Post-PCA dim | 384 | **384** |
| Input size | 256×256 | **288×288** |
| Normalization | CLIP (OpenAI) | **CLIP (OpenAI)** |
| Interpolation | BICUBIC | **BICUBIC** |
| Flip augment | Yes | **Yes** |
| Quality pooling | Softmax τ=3.0 | **Softmax τ=3.0** |
| Camera-BN | Yes | **Yes** |
| L2 normalized | Yes | **Yes** |
| Saved as | `embeddings.npy` | **`embeddings_secondary.npy`** |

### Ordering Guarantee

The CNN cell reads `embedding_index.json` from stage2 output and iterates in the **exact same order** as the primary embeddings. This ensures row `i` in `embeddings_secondary.npy` corresponds to the same tracklet as row `i` in `embeddings.npy`.

### PCA Whitening Decision

**Yes, we apply PCA** — 640D → 384D. Rationale:
1. Stage4's FIC whitening works better on lower-dimensional features (avoids singular covariance)
2. Matching primary dimensionality (384D) keeps the fusion balanced
3. PCA decorrelates the CNN features and removes noise dimensions
4. The stage2 pipeline already does independent PCA on secondary embeddings (we replicate this)

The PCA model is saved as `pca_transform_cnn_secondary.pkl` in the stage2 output directory. It is **fit on the current run's data** (not a cached model), since the CNN features are new.

### GPU Memory Considerations

- CLIP RN50x4 visual encoder: ~350 MB in fp16
- Batch size 64 at 288×288: ~200 MB activations
- Total CNN overhead: ~550 MB on top of existing stage2 memory
- After CNN extraction, model is deleted and `torch.cuda.empty_cache()` frees memory before checkpoint packaging
- Kaggle T4 has 16 GB VRAM — no memory pressure expected

### Time Estimate

CNN extraction processes the same ~200-400 tracklets across 6 cameras:
- Crop extraction from disk: reuses stage0 frames (already extracted)
- CNN inference with flip augment: ~10-15 min on T4 (288×288, batch_size=64)
- PCA + L2 norm: <1 sec
- Total: ~10-15 min additional runtime on top of stages 0-2

### Error Handling

- If 09m weights are not found (kernel not attached), CNN extraction is skipped entirely
- If a tracklet has no usable crops, a zero vector is inserted (same strategy as failed tracklets in primary)
- CUDA OOM triggers automatic batch size halving
- CNN extraction failure does NOT block the checkpoint — stages 0-2 outputs are already saved

## Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| CLIP RN50x4 mAP too low for useful fusion | Medium | 09m training spec targets ≥65% mAP; if below, fusion weight sweep in 10c will find this |
| PCA 384D loses too much CNN information (640D → 384D keeps 60% of dims) | Low | 384D captures >99% variance for well-trained features |
| CNN and ViT features still too correlated (both CLIP pretrained) | Low | Architecture difference (conv vs attention) is the diversity axis; pretraining similarity is less important |
| Crop re-extraction adds runtime | Low | ~10-15 min acceptable; frames already on disk |
| index_map ordering mismatch | Low | Code explicitly iterates index_map entries |

## Verification

After the first 10a run with CNN extraction:

1. **Shape check**: `embeddings_secondary.npy` should be `(N, 384)` where N matches `embeddings.npy`
2. **Norm check**: `np.linalg.norm(emb[0])` should be ≈ 1.0
3. **Non-zero check**: Verify that failed tracklet count is small (<5% of N)
4. **10c fusion check**: Run 10c with `FUSION_WEIGHT=0.20` and compare against `FUSION_WEIGHT=0.0` baseline

## Implementation Checklist

- [ ] Train 09m CLIP RN50x4 on CityFlowV2 (prerequisite — separate task)
- [ ] Upload 09m kernel output as Kaggle dataset/kernel output
- [ ] Edit 10a `kernel-metadata.json`: add `gumfreddy/09m-clip-rn50x4-vehicle-reid` to `kernel_sources`
- [ ] Edit 10a pip cell: add `open_clip_torch`
- [ ] Edit 10a dep check cell: add `open_clip` check
- [ ] Edit 10a mount cell: add CNN weight copy logic
- [ ] Remove LAION-2B vehicle2 overrides from stages 0-2 command
- [ ] Insert CNN extraction cell (markdown + code) after stages 0-2 cell
- [ ] Push 10a and run on Kaggle
- [ ] Verify `embeddings_secondary.npy` in checkpoint
- [ ] Run 10c with CNN fusion weight sweep