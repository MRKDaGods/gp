# 09j ResNeXt101-IBN-a ArcFace Training — Spec

> **Date**: 2026-04-17
> **Parent**: Architecture Overhaul Phase 2
> **Template**: 09i ResNet101-IBN-a ArcFace notebook
> **Purpose**: Train the 3rd ensemble model (architectural diversity via grouped convolutions)

---

## Architecture Decision

### Backbone: ResNeXt101-32x8d-IBN-a

| Property | Value |
|----------|-------|
| **timm model name** | N/A — uses `torchvision.models.resnext101_32x8d` + custom IBN-a injection |
| **Class** | `ReIDModelResNeXt101IBN` (already in `src/training/model.py` L215-260) |
| **Builder** | `_build_resnext101_ibn_a()` (L133-155 in model.py) |
| **Parameters** | ~88M (vs ResNet101's ~44M) |
| **Feature dim** | 2048 (same as ResNet101-IBN-a) |
| **Groups** | 32 groups × 8 width per group |
| **IBN-a layers** | layer1, layer2, layer3 (InstanceNorm on half channels) |
| **last_stride** | 1 (high-resolution feature maps) |
| **Pooling** | GeM (p=3.0, learnable) |
| **Neck** | BNNeck (BatchNorm1d, bias frozen) |
| **Head** | ArcFace (s=30.0, m=0.35) on BNNeck features |

### Why Not timm?

timm does NOT provide IBN-a variants. The codebase uses torchvision's standard ResNeXt101 and injects `IBN_a` modules by replacing `bn1` in layers 1-3. This is the same approach used for ResNet101-IBN-a.

### Pretrained Weights Initialization

**Use IBN-Net GitHub pretrained weights** (NOT torchvision ImageNet):
- URL: `https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnext101_ibn_a-6ace051d.pth`
- These weights were trained WITH IBN-a active — gives better initialization than torchvision weights (which lack IBN-a parameters)
- The 09h notebook already downloads these weights; replicate that pattern
- Load with `strict=False` to skip FC layer mismatch

**Critical**: The current `_build_resnext101_ibn_a()` in model.py loads torchvision weights then injects IBN-a (meaning IBN-a params start random). For the notebook, override this: download IBN-Net weights and load them AFTER building the model with `pretrained=False`.

---

## Training Hyperparameters

Follow 09i's proven recipe exactly, with batch size adjusted for the larger model:

| Parameter | 09i (ResNet101) | 09j (ResNeXt101) | Rationale |
|-----------|:---:|:---:|---|
| **Epochs** | 160 | 160 | Same schedule |
| **Batch size** | 64 (P=16, K=4) | **48 (P=12, K=4)** | ResNeXt is ~2x params; 48 is safe on T4 16GB with fp16 |
| **Image size** | 384×384 | 384×384 | Square crops, same as 09i |
| **Optimizer** | AdamW | AdamW | Same |
| **LR** | 3.5e-4 | 3.5e-4 | Same |
| **Weight decay** | 5e-4 | 5e-4 | Same |
| **Warmup** | 10 epochs, factor=0.01 | 10 epochs, factor=0.01 | Same |
| **LR schedule** | MultiStep [80, 120], γ=0.1 | MultiStep [80, 120], γ=0.1 | Same |
| **ArcFace** | s=30.0, m=0.35 | s=30.0, m=0.35 | Same |
| **Triplet** | Hard mining, margin=0.3 | Hard mining, margin=0.3 | Same |
| **Center loss** | weight=5e-4, starts epoch 20, SGD lr=0.5 | weight=5e-4, starts epoch 20, SGD lr=0.5 | Same |
| **Label smoothing** | 0.1 | 0.1 | Same |
| **Mixed precision** | fp16 | fp16 | Essential for T4 |
| **Eval frequency** | Every 10 epochs | Every 10 epochs | Same |
| **Flip test** | Forward + horizontal flip, average | Forward + horizontal flip, average | Same |

### Why Keep Same Hyperparameters?

ResNeXt101-32x8d is architecturally different (grouped convolutions) but:
- Same feature dimension (2048D)
- Same BNNeck + ArcFace head structure
- Similar optimization landscape for vehicle ReID
- The 09i recipe is already tuned for CityFlowV2's small dataset (128 IDs)
- Only batch size needs reduction due to memory constraints

### Batch Size Memory Estimate (T4 16GB, fp16)

| Model | Params | Batch 64 @ 384² | Batch 48 @ 384² |
|-------|:---:|:---:|:---:|
| ResNet101-IBN-a | ~44M | ✅ ~12GB | ✅ ~9GB |
| ResNeXt101-32x8d-IBN-a | ~88M | ⚠️ ~15.5GB (tight) | ✅ ~12GB |

**Recommendation**: Start with batch 48. If training succeeds without OOM, can try 64 in v2.

---

## Notebook Structure (09j)

Clone 09i and make these targeted changes:

### Cell 1: Markdown Header
```markdown
# 09j ResNeXt101-IBN-a ArcFace Training

Third ensemble model for CityFlowV2 vehicle ReID.
Architecture: ResNeXt101-32x8d-IBN-a with ArcFace + Triplet + Center loss.
Based on 09i training recipe.
```

### Cell 2: Bootstrap (No Changes)
Same GPU detection, pip installs, CUDA/Volta detection as 09i.

### Cell 3: IO Setup
Same Kaggle input/output paths. No changes needed.

### Cell 4: Training Script — Changes Required

**4A. Model Class Replacement**

Replace `ReIDModelResNet101IBNArcFace` with `ReIDModelResNeXt101IBNArcFace`:

```python
class ReIDModelResNeXt101IBNArcFace(nn.Module):
    """ResNeXt101-32x8d-IBN-a with GeM + BNNeck + ArcFace."""
    def __init__(self, num_classes, last_stride=1, gem_p=3.0):
        super().__init__()
        self.backbone = _build_resnext101_ibn_a_from_ibn_weights(last_stride)
        self.feat_dim = 2048
        self.pool = GeM(p=gem_p)
        self.bottleneck = nn.BatchNorm1d(self.feat_dim)
        self.bottleneck.bias.requires_grad_(False)
        nn.init.constant_(self.bottleneck.weight, 1)
        nn.init.constant_(self.bottleneck.bias, 0)
        self.arcface = ArcFaceHead(self.feat_dim, num_classes, s=30.0, m=0.35)
```

**4B. Backbone Builder — IBN-Net Pretrained Weights**

Add a new builder that downloads IBN-Net weights instead of using torchvision:

```python
IBN_RESNEXT_URL = "https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnext101_ibn_a-6ace051d.pth"

def _build_resnext101_ibn_a_from_ibn_weights(last_stride=1):
    """Build ResNeXt101-IBN-a using IBN-Net pretrained weights."""
    import torchvision.models as tv_models

    # Build architecture without pretrained weights
    base = tv_models.resnext101_32x8d(weights=None)

    # Inject IBN-a layers
    for layer in [base.layer1, base.layer2, base.layer3]:
        for block in layer:
            if hasattr(block, "bn1"):
                block.bn1 = IBN_a(block.bn1.num_features)

    # Download and load IBN-Net pretrained weights
    cache_dir = "/kaggle/working/pretrained"
    os.makedirs(cache_dir, exist_ok=True)
    weight_path = os.path.join(cache_dir, "resnext101_ibn_a-6ace051d.pth")
    if not os.path.exists(weight_path):
        print("Downloading IBN-Net ResNeXt101-IBN-a pretrained weights...")
        import urllib.request
        urllib.request.urlretrieve(IBN_RESNEXT_URL, weight_path)

    state_dict = torch.load(weight_path, map_location="cpu")
    # Load with strict=False — skip fc.weight, fc.bias
    missing, unexpected = base.load_state_dict(state_dict, strict=False)
    print(f"Loaded IBN-Net weights: {len(missing)} missing, {len(unexpected)} unexpected")

    # Set last stride
    if last_stride == 1:
        for module in base.layer4.modules():
            if isinstance(module, nn.Conv2d) and module.stride == (2, 2):
                module.stride = (1, 1)

    base.fc = nn.Identity()
    base.avgpool = nn.Identity()
    return base
```

**4C. Batch Size & P/K Configuration**

```python
# Changed from 09i:
BATCH_SIZE = 48    # was 64 — ResNeXt101 is larger
P = 12             # was 16 — identities per batch
K = 4              # same — samples per identity
```

**4D. Warm-Start Logic**

Remove 09i's warm-start from 09d checkpoint (wrong architecture). Instead:
- Start from IBN-Net pretrained ImageNet weights only
- No warm-start from any prior CityFlowV2 checkpoint

```python
# REMOVED: load_matching_state_dict from 09d checkpoint
# ResNeXt101 != ResNet101, cannot transfer weights
# IBN-Net ImageNet pretrained is our starting point
```

**4E. Output Filenames**

```python
BEST_WEIGHTS_PATH = "/kaggle/working/resnext101ibn_cityflowv2_384px_best.pth"
TRAINING_LOG_PATH = "/kaggle/working/resnext101ibn_training_history.json"
CURVES_PATH = "/kaggle/working/resnext101ibn_training_curves.png"
```

### Cell 5: Execution (No Changes)
Same subprocess run pattern as 09i.

---

## kernel-metadata.json

```json
{
  "id": "mrkdagods/09j-resnext101ibn-arcface",
  "title": "09j ResNeXt101-IBN ArcFace Training",
  "code_file": "09j_resnext101ibn_arcface.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "machine_shape": "NvidiaTeslaT4",
  "dataset_sources": [
    "thanhnguyenle/data-aicity-2023-track-2",
    "gumfreddy/mtmc-weights"
  ],
  "competition_sources": [],
  "kernel_sources": [],
  "model_sources": []
}
```

**Account**: `mrkdagods` — runs in parallel with 09i on `gumfreddy`.
**Alternative**: `ali369` if mrkdagods tokens are unavailable (check `~/.kaggle/`).

---

## Expected Performance Targets

| Metric | ResNet101-IBN-a (09d, old recipe) | ResNet101-IBN-a (09i, ArcFace) | ResNeXt101-IBN-a (09j, target) |
|--------|:---:|:---:|:---:|
| CityFlowV2 mAP | 52.77% | 60-68% (expected) | **55-65%** |
| CityFlowV2 R1 | ~70% | 75-85% (expected) | **72-82%** |

### Why 55-65% Is Realistic

- ResNeXt101-32x8d has ~2x parameters vs ResNet101 but the same 2048D output
- CityFlowV2 has only 128 training IDs — larger models don't automatically win
- IBN-Net pretrained weights give better initialization than random IBN-a injection
- ArcFace + Triplet + Center is a much stronger recipe than 09d's old CE + Triplet
- The ensemble value comes from **architectural diversity**, not raw mAP superiority

### Minimum Viable mAP for Ensemble Utility

**≥50% mAP** is needed for the tertiary model to add value to the ensemble. Below 50%, it adds more noise than signal (confirmed by our 52.77% ResNet101 ensemble test which was neutral at 0.30 weight).

---

## Deployment After Training

### 1. Download Weights from Kaggle

```bash
# After 09j completes:
kaggle kernels output mrkdagods/09j-resnext101ibn-arcface -p models/reid/
# Expected: resnext101ibn_cityflowv2_384px_best.pth
```

### 2. Upload as Kaggle Model (for 10a consumption)

```bash
# Upload to Kaggle Models for the pipeline to consume
kaggle models create -t "resnext101ibn-cityflowv2-384px"
kaggle model-instances create mrkdagods/resnext101ibn-cityflowv2-384px/pyTorch/default \
  -p models/reid/ \
  --license-name "Apache 2.0"
```

### 3. Stage 2 Config Update

```yaml
# In configs/cityflowv2.yaml:
stage2:
  reid:
    vehicle3:
      enabled: true
      model_name: "resnext101_ibn_a"
      weights_path: "models/reid/resnext101ibn_cityflowv2_384px_best.pth"
      embedding_dim: 2048
      input_size: [384, 384]
```

The deployment path already exists — `src/stage2_features/reid_model.py` has `_build_resnext101_ibn()` that routes `model_name="resnext101_ibn_a"` to `ReIDModelResNeXt101IBN`.

---

## Risks & Mitigations

| Risk | Likelihood | Mitigation |
|------|:---:|---|
| T4 OOM at batch 48 | Low | Drop to batch 32 (P=8, K=4). Still viable with longer training. |
| mAP < 50% (useless for ensemble) | Low-Med | ArcFace recipe is proven on 09i. If low, try batch 64 with gradient accumulation. |
| IBN-Net weight download fails | Low | Fallback: use torchvision ImageNet weights (`pretrained=True`). |
| Training > 12h (Kaggle timeout) | Medium | ResNeXt is 2x slower per step than ResNet101. 160 epochs × batch 48 may take ~10-11h. If >12h, reduce to 120 epochs with milestones [60, 90]. |
| mrkdagods tokens missing | Medium | Use ali369 account instead. Update kernel-metadata.json id field. |

---

## Relationship to Other Notebooks

```
09d (ResNet101-IBN-a, old recipe)     → 52.77% mAP, trained, deployed
09g (ResNet101-IBN-a, DMT)            → Dead end, -1.4pp MTMC IDF1
09h (ResNeXt101-IBN-a, DMT)           → Dead end (DMT harmful)
09i (ResNet101-IBN-a, ArcFace) ← NEW  → Training on gumfreddy
09j (ResNeXt101-IBN-a, ArcFace) ← NEW → This spec, trains on mrkdagods/ali369
```

**09j is NOT a fork of 09h**. It's a fork of **09i** with the backbone swapped. DMT is explicitly excluded per confirmed dead-end findings.

---

## Checklist for Implementation

- [ ] Create directory: `notebooks/kaggle/09j_resnext101ibn_arcface/`
- [ ] Clone 09i notebook → `09j_resnext101ibn_arcface.ipynb`
- [ ] Replace model class: `ReIDModelResNet101IBNArcFace` → `ReIDModelResNeXt101IBNArcFace`
- [ ] Add IBN-Net weight download function (`_build_resnext101_ibn_a_from_ibn_weights`)
- [ ] Change batch size: 64→48, P: 16→12
- [ ] Remove 09d warm-start logic (architecture mismatch)
- [ ] Update output filenames (resnext101ibn_*)
- [ ] Update markdown header (Cell 1)
- [ ] Create kernel-metadata.json for mrkdagods account
- [ ] Verify notebook JSON is valid (`python -c "import json; json.load(open(...))"`)
- [ ] Push: `kaggle kernels push -p notebooks/kaggle/09j_resnext101ibn_arcface/`
- [ ] Monitor: `python scripts/kaggle_logs.py mrkdagods/09j-resnext101ibn-arcface --tail 20`
```