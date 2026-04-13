# Fix 09g/09h Pretrained Weight Loading Errors

## Summary
Both notebooks error in Cell 3 at `model = build_model(len(train_pid_map))`. Two distinct root causes.

---

## Error 1: 09g (ResNet101-IBN-a) — IBN Attribute Name Mismatch

### Root Cause
The `IBN_a` class in Cell 3 uses `self.instance_norm` / `self.batch_norm` as attribute names:
```python
class IBN_a(nn.Module):
    def __init__(self, planes):
        super().__init__()
        half = planes // 2
        self.instance_norm = nn.InstanceNorm2d(half, affine=True)   # ← wrong name
        self.batch_norm = nn.BatchNorm2d(planes - half)             # ← wrong name
```

The IBN-Net pretrained checkpoint (`resnet101_ibn_a-59ea0ac6.pth`) uses `IN` and `BN` as sub-module names. After monkey-patching `base.layer1/2/3` blocks with `IBN_a`, the model's state_dict has keys like `layer1.0.bn1.instance_norm.weight`, but the checkpoint has `layer1.0.bn1.IN.weight`.

With `strict=False`, the checkpoint keys become "unexpected" (silently ignored), and the model's IBN keys become "missing". The `allowed_missing` guard then raises `RuntimeError` because the missing set is far larger than `{fc.weight, fc.bias}`.

### Fix
Rename the `IBN_a` attributes to match the IBN-Net checkpoint convention (this already matches what 09h does):

**Cell 3 (notebook lines 327–424), IBN_a class — 3 changes:**

1. `self.instance_norm = nn.InstanceNorm2d(half, affine=True)` → `self.IN = nn.InstanceNorm2d(half, affine=True)`
2. `self.batch_norm = nn.BatchNorm2d(planes - half)` → `self.BN = nn.BatchNorm2d(planes - half)`
3. `[self.instance_norm(x[:, :split]), self.batch_norm(x[:, split:])]` → `[self.IN(x[:, :split]), self.BN(x[:, split:])]`

**Also update `generate_09g_notebook.py`** (if it exists and is used to regenerate): apply the same 3 renames so future regenerations produce the correct code.

No other cells reference these attribute names, so no downstream changes needed.

---

## Error 2: 09h (ResNeXt101-IBN-a) — Architecture Width Mismatch

### Root Cause
The notebook creates a **ResNeXt101_32x8d** base model:
```python
base = tv_models.resnext101_32x8d(weights=None)
```

But downloads the IBN-Net pretrained checkpoint `resnext101_ibn_a-6ace051d.pth`, which was trained on **ResNeXt101_32x4d** (32 groups × 4 width_per_group = 128 channels in layer1 bottleneck).

ResNeXt101_32x8d has 32 groups × 8 width_per_group = 256 channels in layer1 bottleneck. This causes shape mismatches:
- `layer1.0.conv1.weight`: checkpoint [128, 64, 1, 1] vs model [256, 64, 1, 1]
- `layer1.0.bn1.IN.weight`: checkpoint [64] vs model [128]
- (and throughout all layers)

### Fix
Replace the base model with the correct 32x4d architecture. `torchvision` does NOT expose `resnext101_32x4d` as a named model (only `resnext50_32x4d`, `resnext101_32x8d`, `resnext101_64x4d`), so construct it manually:

**Cell 3 (notebook lines 459–556), ResNeXt101IBNNeck.__init__ — 1 change:**

Replace:
```python
base = tv_models.resnext101_32x8d(weights=None)
```
With:
```python
from torchvision.models.resnet import ResNet, Bottleneck
base = ResNet(Bottleneck, [3, 4, 23, 3], groups=32, width_per_group=4)
```

**Also update `generate_09h_notebook.py`** transform:
```python
# Current:
('base = tv_models.resnet101(weights=None)', 'base = tv_models.resnext101_32x8d(weights=None)'),

# Change to:
('base = tv_models.resnet101(weights=None)',
 'from torchvision.models.resnet import ResNet, Bottleneck
        base = ResNet(Bottleneck, [3, 4, 23, 3], groups=32, width_per_group=4)'),
```

Also update the timm check string for consistency:
```python
# Current:
'if hasattr(timm, "list_models") and "resnext101_32x8d" in timm.list_models(pretrained=False):'
# Change to:
'if hasattr(timm, "list_models") and "resnext101_32x4d" in timm.list_models(pretrained=False):'
```

### Channel Width Reference
| Architecture       | groups | width_per_group | layer1 bottleneck channels |
|-------------------|--------|-----------------|---------------------------|
| ResNet101          | 1      | 64              | 64                        |
| ResNeXt101_32x4d   | 32     | 4               | 128                       |
| ResNeXt101_32x8d   | 32     | 8               | 256                       |
| ResNeXt101_64x4d   | 64     | 4               | 256                       |

The IBN-Net `resnext101_ibn_a-6ace051d.pth` matches **32x4d** (128 channels).

### Model Parameter Counts
- ResNeXt101_32x8d: ~88.8M params (larger)
- ResNeXt101_32x4d: ~44.2M params (correct for this checkpoint)
- ResNet101: ~44.5M params (similar to 32x4d)

---

## Additional Issues Found

### 1. Generator Script Drift (09h)
`generate_09h_notebook.py` uses 09g as a template. If 09g's `IBN_a` is fixed (renamed to `IN`/`BN`), the generator's find/replace transforms for `instance_norm`→`IN` and `batch_norm`→`BN` will FAIL because the source strings no longer exist. Two options:
- **Option A**: Fix 09g first, then update 09h's generator to remove the now-unnecessary IBN renames (the transforms would be no-ops)
- **Option B**: Fix both notebooks independently (don't regenerate 09h from 09g)

**Recommendation**: Option B — fix both notebooks directly, then update both generator scripts to match, since the generators may not be run again.

### 2. feat_dim Assumption (09h)
`ResNeXt101IBNNeck` hardcodes `feat_dim=2048`. For ResNeXt101_32x4d, the final layer output is indeed 2048 channels (same as ResNet101 and ResNeXt101_32x8d — only the bottleneck widths differ, not the output). So this is correct and needs no change.

### 3. No Other Cell Dependencies
Neither notebook references `IBN_a`, `instance_norm`, `batch_norm`, `IN`, or `BN` outside of Cell 3. The model is used as a black box in training/eval cells, so no downstream changes are needed.

---

## Implementation Order
1. Fix 09g Cell 3: rename `instance_norm`→`IN`, `batch_norm`→`BN` (3 string replacements)
2. Fix 09h Cell 3: replace `tv_models.resnext101_32x8d(weights=None)` with `ResNet(Bottleneck, [3, 4, 23, 3], groups=32, width_per_group=4)` (1 replacement + 1 import)
3. Update generator scripts to match
4. Push both notebooks to Kaggle and verify Cell 3 passes

## Verification
After fix, Cell 3 should print `Model params: N` without error:
- 09g: ~44.5M params (ResNet101-IBN-a)
- 09h: ~44.2M params (ResNeXt101_32x4d-IBN-a)
