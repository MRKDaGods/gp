# EVA02 ViT-B/16 Inference Support — Deployment Spec

## Status
- **09o training**: Running on Kaggle
- **Checkpoint**: `eva02_vit_cityflowv2_final.pth` (EMA weights)
- **Target slot**: `vehicle3` (tertiary ensemble model)

## Key Finding: Minimal Code Changes Required

**The existing TransReID class already supports EVA02 architecture.** The `TransReID.forward()` method in `src/stage2_features/transreid_model.py` already handles:
1. RoPE (rotary position embedding) via `_pos_embed()` tuple detection (line ~115)
2. Passing `rope=rot_pos_embed` to each transformer block (line ~131)
3. SwiGLU FFN and sub-layer normalization (encapsulated within timm blocks)

The model name `"eva02_vit"` is already in `ReIDModel._TRANSREID_NAMES`, so it routes to `_build_transreid()` which calls `build_transreid()`.

## Architecture Compatibility Analysis

### 09o Training Model (EVA02ReID)
```python
class EVA02ReID(nn.Module):
    # Simplified for training on Kaggle
    # Keys: vit.*, bn.*, proj.* (Identity), cls_head.*
    # No SIE, no JPM
    # Forward: vit.forward_features() + forward_head() → bn → proj → normalize
```

### Pipeline Inference Model (TransReID)
```python
class TransReID(nn.Module):
    # Full-featured for inference
    # Keys: vit.*, bn.*, proj.*, cls_head.*, sie_embed (optional), bn_jpm.* (optional)
    # Forward: manual patch_embed → _pos_embed → norm_pre → blocks(rope) → norm → bn → proj → normalize
```

### Checkpoint Key Mapping

| 09o Checkpoint Key | TransReID Model Key | Status |
|---|---|---|
| `vit.patch_embed.*` | `vit.patch_embed.*` | ✅ Direct match |
| `vit.cls_token` | `vit.cls_token` | ✅ Direct match |
| `vit.pos_embed` | `vit.pos_embed` | ✅ Direct match (256×256 → 16×16 grid) |
| `vit.norm_pre.*` | `vit.norm_pre.*` | ✅ Direct match |
| `vit.blocks.N.*` | `vit.blocks.N.*` | ✅ Direct match |
| `vit.norm.*` | `vit.norm.*` | ✅ Direct match |
| `bn.weight/bias/running_*` | `bn.weight/bias/running_*` | ✅ Direct match |
| `cls_head.weight` (576 classes) | `cls_head.weight` (1 class) | ⚠️ Shape mismatch → dropped (expected) |
| — | `sie_embed` | ⚠️ Missing (SIE disabled with num_cameras=0) |
| — | `bn_jpm.*`, `jpm_cls.*` | ⚠️ Missing (JPM unused at inference, non-critical) |

**Result**: `load_state_dict(strict=False)` handles all mismatches correctly. The existing `build_transreid()` key remapping and shape mismatch handling covers this.

### Checkpoint Format (from 09o)
```python
{
    "model": <EMA state_dict>,  # Key: "model" → unwrapped by build_transreid()
    "epoch": int,
    "mAP": float,
}
```

## Changes Required

### 1. `configs/default.yaml` — Update vehicle3 defaults

**Current** (resnext101_ibn_a placeholder):
```yaml
vehicle3:
  enabled: false
  save_separate: true
  model_name: "resnext101_ibn_a"
  weights_path: ""
  embedding_dim: 2048
  input_size: [384, 384]
  clip_normalization: false
```

**New** (EVA02 defaults):
```yaml
vehicle3:
  enabled: false  # enabled when 09o completes and weights are available
  save_separate: true
  model_name: "eva02_vit"
  weights_path: "models/reid/eva02_vit_cityflowv2.pth"
  embedding_dim: 768
  input_size: [256, 256]
  vit_model: "eva02_base_patch16_clip_224.merged2b"
  clip_normalization: true  # CRITICAL: EVA02 trained with CLIP normalization
  num_cameras: 0  # No SIE for EVA02 (RoPE handles position encoding)
  concat_patch: false
```

**Why these values:**
- `model_name: "eva02_vit"` — already in `_TRANSREID_NAMES`, routes to TransReID builder
- `embedding_dim: 768` — EVA02 ViT-B output dimension
- `input_size: [256, 256]` — matches 09o training resolution
- `vit_model: "eva02_base_patch16_clip_224.merged2b"` — exact timm model from training
- `clip_normalization: true` — **CRITICAL BUG FIX**: 09o trains with CLIP mean/std; inference MUST match
- `num_cameras: 0` — EVA02 doesn't use SIE; position info comes from RoPE

### 2. `src/stage2_features/reid_model.py` — Add EVA02 vit_model fallback

The timm model name `eva02_base_patch16_clip_224.merged2b` may not be available in all timm versions (the `.merged2b` suffix is a pretrained variant tag). Add a fallback in `_build_transreid()`:

**Current code** (line ~365):
```python
def _build_transreid(self, weights_path: Optional[str]):
    """Build TransReID ViT model."""
    from src.stage2_features.transreid_model import build_transreid

    model = build_transreid(
        num_classes=1,
        num_cameras=self.num_cameras,
        embed_dim=self.embedding_dim,
        vit_model=self.vit_model,
        pretrained=weights_path is None,
        weights_path=weights_path,
        img_size=self.input_size,
    )
    return model
```

**New code** — add EVA02 vit_model fallback:
```python
def _build_transreid(self, weights_path: Optional[str]):
    """Build TransReID ViT model."""
    from src.stage2_features.transreid_model import build_transreid

    vit_model = self.vit_model

    # EVA02 fallback: merged2b variant may not exist in all timm versions
    if "eva02" in vit_model.lower() and "." in vit_model:
        import timm
        if not timm.is_model(vit_model):
            fallback = vit_model.rsplit(".", 1)[0]  # strip variant tag
            logger.warning(
                f"timm model {vit_model!r} not available, falling back to {fallback!r}"
            )
            vit_model = fallback

    model = build_transreid(
        num_classes=1,
        num_cameras=self.num_cameras,
        embed_dim=self.embedding_dim,
        vit_model=vit_model,
        pretrained=weights_path is None,
        weights_path=weights_path,
        img_size=self.input_size,
    )
    return model
```

**Why**: The 09o training notebook has the same fallback pattern (`REQUESTED_VIT_MODEL → FALLBACK_VIT_MODEL`). Both model names have identical architecture — the `.merged2b` suffix only specifies the pretrained weight variant. Since we're loading our own trained weights (not timm pretrained), the fallback is safe.

### 3. `src/stage2_features/transreid_model.py` — No changes needed

The existing code already handles EVA02:
- `_pos_embed()` tuple detection (lines 113-119)
- `rope=rot_pos_embed` in block loop (lines 130-133)
- `norm_pre` call (lines 126-128)
- Checkpoint key remapping handles `"model"` key (line 218)
- Shape mismatch handling drops `cls_head` (lines 253-291)
- `strict=False` loading tolerates missing JPM/SIE keys (line 299)

### 4. `src/stage2_features/pipeline.py` — No changes needed

The existing vehicle3 loading block (lines ~268-285) already:
- Reads `vehicle3_cfg` from `stage_cfg.reid.get("vehicle3", {})`
- Creates `ReIDModel(model_name=..., vit_model=..., clip_normalization=..., ...)`
- Passes all relevant parameters
- Handles weights_path existence check
- Saves tertiary embeddings separately when `save_separate=true`

The tertiary PCA pipeline (lines ~655-700) also already works:
- Applies camera-BN to tertiary embeddings
- Fits/loads tertiary PCA from `tertiary_pca_model_path`
- L2-normalizes and saves to `embeddings_tertiary.npy`

### 5. `notebooks/kaggle/10a_stages012/10a_stages012.ipynb` — Fix clip_normalization

**Current 10a notebook** (conditional vehicle3 overrides):
```python
if EVA02_AVAILABLE:
    cmd += [
        "--override", "stage2.reid.vehicle3.enabled=true",
        "--override", "stage2.reid.vehicle3.model_name=eva02_vit",
        "--override", "stage2.reid.vehicle3.weights_path=models/reid/eva02_vit_cityflowv2.pth",
        "--override", "stage2.reid.vehicle3.embedding_dim=768",
        "--override", "stage2.reid.vehicle3.input_size=[256,256]",
        "--override", "stage2.reid.vehicle3.clip_normalization=false",  # ← BUG
        "--override", "stage2.reid.vehicle3.save_separate=true",
    ]
```

**Fix** — change `clip_normalization=false` to `true` and add `vit_model` + `num_cameras`:
```python
if EVA02_AVAILABLE:
    cmd += [
        "--override", "stage2.reid.vehicle3.enabled=true",
        "--override", "stage2.reid.vehicle3.model_name=eva02_vit",
        "--override", "stage2.reid.vehicle3.weights_path=models/reid/eva02_vit_cityflowv2.pth",
        "--override", "stage2.reid.vehicle3.embedding_dim=768",
        "--override", "stage2.reid.vehicle3.input_size=[256,256]",
        "--override", "stage2.reid.vehicle3.vit_model=eva02_base_patch16_clip_224.merged2b",
        "--override", "stage2.reid.vehicle3.clip_normalization=true",  # FIXED: EVA02 uses CLIP norm
        "--override", "stage2.reid.vehicle3.num_cameras=0",
        "--override", "stage2.reid.vehicle3.save_separate=true",
    ]
```

**Why this matters**: EVA02 was pretrained on CLIP data and 09o trained with `CLIP_MEAN/CLIP_STD` normalization. Using ImageNet normalization at inference would create a train/test distribution mismatch and significantly degrade features.

### 6. Stage 4 Tertiary Fusion — Already supported

Config in `default.yaml` → `stage4.association.tertiary_embeddings`:
```yaml
tertiary_embeddings:
  path: ""       # auto-populated or set to embeddings_tertiary.npy path
  weight: 0.0    # tune after EVA02 mAP is known
```

**Initial weight sweep plan** (after EVA02 mAP is known):
- If EVA02 mAP ≥ 75%: sweep weight in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
- If EVA02 mAP 65-75%: sweep weight in [0.05, 0.10, 0.15]
- If EVA02 mAP < 65%: likely too weak for ensemble (see R50-IBN at 52.77% precedent)

## Embedding Dimensions & PCA Configuration

| Model | Raw Dim | PCA Target | PCA Path |
|---|---|---|---|
| Primary (TransReID CLIP) | 768 (or 1536 with concat_patch) | 384 | `models/reid/pca_transform.pkl` |
| Secondary (R50-IBN) | 2048 | 384 | `models/reid/pca_transform_secondary.pkl` |
| **Tertiary (EVA02)** | **768** | **384** | `models/reid/pca_transform_tertiary.pkl` |

All three models output to the same 384D PCA space after whitening. The tertiary PCA path is already configured in `default.yaml` at `stage2.pca.tertiary_pca_model_path`.

## RoPE Handling — Already Solved

EVA02 uses Rotary Position Embeddings (RoPE) inside attention blocks instead of standard absolute position embeddings. The current `TransReID.forward()` already handles this:

```python
# Line 113-119 of transreid_model.py
if hasattr(self.vit, "_pos_embed"):
    result = self.vit._pos_embed(x)
    if isinstance(result, tuple):
        x, rot_pos_embed = result  # EVA02 returns (x, rope)
    else:
        x = result  # Standard ViT returns x

# Line 130-133 of transreid_model.py
for blk in self.vit.blocks:
    if rot_pos_embed is not None:
        x = blk(x, rope=rot_pos_embed)  # EVA02 blocks need rope
    else:
        x = blk(x)  # Standard ViT blocks
```

**No SIE needed**: RoPE encodes position information directly in attention. Setting `num_cameras=0` disables SIE, which is correct for EVA02.

## Weight File Resolution Chain (10a)

```
1. /kaggle/input/09o-eva02-vit-reid-cityflowv2/eva02_vit_cityflowv2_final.pth
   → Copy to models/reid/eva02_vit_cityflowv2.pth
   → Set EVA02_AVAILABLE = True
   
2. If not found: EVA02_AVAILABLE = False
   → vehicle3 overrides not added to command
   → Pipeline runs without tertiary model (graceful fallback)
```

## Testing Plan

### Unit Tests

1. **Architecture test**: Verify TransReID can instantiate with EVA02 backbone
   ```python
   def test_transreid_eva02_instantiation():
       model = TransReID(
           num_classes=1,
           num_cameras=0,
           embed_dim=768,
           vit_model="eva02_base_patch16_clip_224",
           pretrained=False,
           sie_camera=False,
           jpm=True,
           img_size=(256, 256),
       )
       x = torch.randn(2, 3, 256, 256)
       with torch.no_grad():
           model.eval()
           out = model(x)
       assert out.shape == (2, 768)
       # Verify L2-normalized
       norms = torch.norm(out, dim=1)
       assert torch.allclose(norms, torch.ones(2), atol=1e-5)
   ```

2. **CLIP normalization test**: Verify auto-detection works
   ```python
   def test_eva02_clip_normalization_autodetect():
       model = ReIDModel(
           model_name="eva02_vit",
           weights_path=None,
           embedding_dim=768,
           input_size=(256, 256),
           device="cpu",
           half=False,
           vit_model="eva02_base_patch16_clip_224.merged2b",
           clip_normalization=None,  # auto-detect
       )
       assert model.clip_normalization is True  # "clip" in vit_model
   ```

3. **Checkpoint loading test**: Verify 09o checkpoint format loads correctly
   ```python
   def test_eva02_checkpoint_loading():
       # Create a mock 09o-format checkpoint
       model = TransReID(num_classes=576, num_cameras=0, embed_dim=768,
                         vit_model="eva02_base_patch16_clip_224",
                         pretrained=False, sie_camera=False, img_size=(256, 256))
       mock_ckpt = {"model": model.state_dict(), "epoch": 120, "mAP": 0.75}
       torch.save(mock_ckpt, "/tmp/test_eva02_ckpt.pth")
       
       # Load into inference model (num_classes=1)
       loaded = build_transreid(
           num_classes=1, num_cameras=0, embed_dim=768,
           vit_model="eva02_base_patch16_clip_224",
           weights_path="/tmp/test_eva02_ckpt.pth",
           img_size=(256, 256),
       )
       x = torch.randn(2, 3, 256, 256)
       with torch.no_grad():
           loaded.eval()
           out = loaded(x)
       assert out.shape == (2, 768)
   ```

4. **timm fallback test**: Verify merged2b → base fallback
   ```python
   def test_eva02_vit_model_fallback():
       # This test verifies the fallback works if merged2b isn't available
       # Implementation in reid_model.py _build_transreid
       import timm
       base_name = "eva02_base_patch16_clip_224"
       assert timm.is_model(base_name)  # Base should always exist
   ```

### Integration Tests (Kaggle, after 09o completes)

1. **10a smoke test**: Run 10a with `EVA02_AVAILABLE=True` on a single camera
   - Verify tertiary embeddings shape: `(N, 768)` raw → `(N, 384)` after PCA
   - Verify `embeddings_tertiary.npy` is saved
   - Verify no CUDA errors or NaN embeddings

2. **10c fusion test**: Run 10c with tertiary embeddings
   - Sweep `stage4.association.tertiary_embeddings.weight` in [0.0, 0.05, 0.10, 0.15, 0.20]
   - Compare MTMC IDF1 against baseline (no tertiary)
   - Watch for conflation changes (should decrease if EVA02 is decorrelated)

3. **Memory budget**: Verify P100 16GB can handle 3 models simultaneously
   - Primary ViT: ~350MB (FP16)
   - Secondary R50-IBN: ~200MB (FP16)
   - Tertiary EVA02: ~350MB (FP16)
   - Total: ~900MB model weights + ~4GB activations = well within 16GB

## Summary of Changes

| File | Change | Effort |
|---|---|---|
| `configs/default.yaml` | Update vehicle3 defaults to EVA02 | Trivial |
| `src/stage2_features/reid_model.py` | Add timm model name fallback in `_build_transreid()` | Small (~10 lines) |
| `src/stage2_features/transreid_model.py` | **No changes** | None |
| `src/stage2_features/pipeline.py` | **No changes** | None |
| `10a notebook` | Fix `clip_normalization=true`, add `vit_model` + `num_cameras` overrides | Small (~3 line edits) |

**Total implementation effort**: ~15 minutes. The heavy lifting was already done when TransReID's forward pass was generalized to handle timm's `_pos_embed()` tuple return and `rope` block arguments.

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| timm version incompatibility on Kaggle | Low | Medium | Fallback from merged2b to base variant |
| EVA02 too correlated with primary CLIP ViT | Medium | High | Different pretraining (MIM+CLIP vs CLIP-only) should help; if r<0.85, useful |
| EVA02 mAP too low (<65%) | Low-Medium | High | Won't ensemble; use as diagnostic only |
| Memory OOM with 3 models on P100 | Very Low | Medium | ~900MB total, well within 16GB budget |
| CLIP norm bug (false → true) already deployed | N/A | N/A | Current vehicle3 is disabled; fix before enabling |

## Dependencies

- **09o completion**: Checkpoint must be saved as `eva02_vit_cityflowv2_final.pth`
- **09o Kaggle dataset**: Output must be published as dataset for 10a to consume
- **10a kernel metadata**: Must add `09o-eva02-vit-reid-cityflowv2` as input dataset
- **timm >= 0.9**: Required for EVA02 model support (already satisfied on Kaggle)