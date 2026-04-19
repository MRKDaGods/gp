# 09m: TransReID DFN-2B CLIP Fine-Tuning on CityFlowV2

## Context

Fallback secondary model training in case 09l v2 (LAION-2B CLIP) doesn't reach ≥65% mAP.
DFN-2B (Data Filtering Networks, Apple) uses quality-filtered LAION-2B data, which may
produce better transfer features than raw LAION-2B pretraining.

**Source notebook**: `notebooks/kaggle/09l_transreid_laion2b/09l_transreid_laion2b.ipynb`
**Target notebook**: `notebooks/kaggle/09m_transreid_dfn2b/09m_transreid_dfn2b.ipynb`

## Trigger Condition

Create and push 09m **only if** 09l v2 finishes with mAP < 65% on CityFlowV2 val.

## Key Differences from 09l (LAION-2B)

| Parameter | 09l (LAION-2B) | 09m (DFN-2B) |
|-----------|----------------|--------------|
| timm model name | `vit_base_patch16_clip_224.laion2b` | `vit_base_patch16_clip_224.dfn2b` |
| Input resolution | 256×256 | 224×224 (native DFN-2B) |
| Train resize | 272×272 (`H + 16`) | 240×240 (`H + 16`) |
| Train crop | 256×256 | 224×224 |
| Test resize | 256×256 | 224×224 |
| pos_embed in VeRi skip_keys | Yes (shape mismatch: 224→256) | **No** (shapes match at 224) |
| Output dir | `working/09l_laion2b/` | `working/09m_dfn2b/` |
| Best checkpoint name | `transreid_cityflowv2_laion2b_best_ema.pth` | `transreid_cityflowv2_dfn2b_best_ema.pth` |

Everything else is **identical**: Triplet Loss, EMA (0.9999), 160 epochs, AdamW,
backbone_lr=1e-4, head_lr=1e-3, LLRD=0.75, CE(eps=0.05), Center loss (delayed@ep15),
batch_size=64, PKSampler(p=16,k=4), SIE, JPM, BNNeck, CLIP normalization, same augmentation
stack (flip, pad+crop, color jitter, random erasing p=0.5).

## Exact Changes Required (09l → 09m)

### Cell 1 (Markdown header)
- Title: `# 09m: TransReID DFN-2B CLIP Fine-Tuning on CityFlowV2`
- Replace all "LAION-2B" references with "DFN-2B"
- Model line: `vit_base_patch16_clip_224.dfn2b`
- Resolution line: `Resize input to **224x224** (native DFN-2B resolution)`
- Output paths: `working/09m_dfn2b/...`
- Checkpoint name: `transreid_cityflowv2_dfn2b_best_ema.pth`

### Cell 4 (Setup / paths)
- `OUTPUT_DIR = Path("/kaggle/working/09m_dfn2b/output")`
- No other changes needed

### Cell 10 (Augmentation / data transforms)
```python
H, W = 224, 224  # native DFN-2B CLIP resolution

train_tf = T.Compose([
    T.Resize((H + 16, W + 16), interpolation=T.InterpolationMode.BICUBIC),
    T.Pad(4, padding_mode='reflect'),
    T.RandomCrop((H, W)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.2, contrast=0.15, saturation=0.1),
    T.ToTensor(),
    T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    T.RandomErasing(p=0.5, scale=(0.02, 0.33), value='random'),
])

test_tf = T.Compose([
    T.Resize((H, W), interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
])
```

### Cell 12 (TransReID model definition)
```python
VIT_MODEL = "vit_base_patch16_clip_224.dfn2b"
```

### Cell 13 (VeRi-776 weight loading)
**Critical change**: Remove `"pos_embed"` from skip_keys since at 224×224 the positional
embedding shapes match between VeRi-776 pretrained and DFN-2B CLIP.

```python
# Skip classifier head and SIE (re-initialized for CityFlowV2)
# pos_embed NOT skipped: shapes match at 224x224 (VeRi was also 224)
skip_keys = ["cls_head", "jpm_cls", "sie_embed"]
```

Update the comment above:
```python
# pos_embed is included because DFN-2B uses native 224 -- same as VeRi-776 pretrained.
```

### Cell 14 (Training markdown)
- `256x256 resolution` → `224x224 resolution (native DFN-2B)`
- `LAION-2B Triplet recipe` → `DFN-2B Triplet recipe`

### Cell 15 (Training loop)
- `best_model_path` → uses `transreid_cityflowv2_dfn2b_best_ema.pth`
- Print strings: replace "LAION-2B" with "DFN-2B"

### Cell 16 (Export)
- `variant`: `"v1_triplet_recipe_dfn2b_ema"`
- `experiment`: `"Experiment A: DFN-2B Triplet recipe (EMA)"`
- `training` string: replace "LAION-2B" with "DFN-2B"
- `architecture`: `VIT_MODEL` (already dynamic, will pick up DFN-2B)
- `input_size`: `[224, 224]` (from `H, W` variables)

### Cell 17 (Integration instructions markdown)
```yaml
stage2:
  reid:
    vehicle:
      model_name: "transreid"
      weights_path: "models/reid/vehicle_transreid_vit_base_cityflowv2.pth"
      embedding_dim: 768
      input_size: [224, 224]
      vit_model: "vit_base_patch16_clip_224.dfn2b"
      num_cameras: 6
      clip_normalization: true
```

## kernel-metadata.json

```json
{
  "id": "gumfreddy/09m-transreid-dfn-2b-training",
  "title": "09m TransReID DFN-2B Training",
  "code_file": "09m_transreid_dfn2b.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "machine_shape": "NvidiaTeslaT4",
  "dataset_sources": [],
  "competition_sources": [],
  "kernel_sources": [],
  "model_sources": [
    "mrkdagods/transreid-veri/other/default/1"
  ]
}
```

## Expected File Structure

```
notebooks/kaggle/09m_transreid_dfn2b/
├── 09m_transreid_dfn2b.ipynb
└── kernel-metadata.json
```

## Expected Outputs (on Kaggle)

```
/kaggle/working/09m_dfn2b/output/
├── transreid_cityflowv2_dfn2b_best_ema.pth
├── training_curves.png
└── ...
/kaggle/working/exported_models/
├── vehicle_transreid_vit_base_cityflowv2.pth
└── vehicle_reid_cityflowv2_metadata.json
```

## Success Criteria

- Stable finite training loss throughout all 160 epochs (no inf/NaN)
- CityFlowV2 validation **mAP ≥ 65%** (minimum for ensemble viability)
- Ideally mAP ≥ 70% for meaningful ensemble contribution
- Exported checkpoint ready for Stage 2 integration

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| DFN-2B timm weights not available | Low | `timm.list_pretrained('*dfn2b*')` to verify before building |
| mAP below 65% | Medium | Both LAION-2B and DFN-2B failing would confirm the gap is dataset-size dependent, not filtration-dependent |
| pos_embed transfer hurts | Low | VeRi-776 pos_embed was learned at 224; should help not hurt. If mAP is unusually low, retry with pos_embed skipped |

## Verification Before Building

Run this locally to confirm the timm model name resolves:
```python
import timm
models = timm.list_pretrained('*clip*dfn*')
print([m for m in models if 'base' in m and 'patch16' in m])
# Should include: vit_base_patch16_clip_224.dfn2b
```

## Implementation Command

When triggered, the @coder agent should:
1. Copy 09l notebook to `notebooks/kaggle/09m_transreid_dfn2b/09m_transreid_dfn2b.ipynb`
2. Apply all changes listed above via `json.load() → modify → json.dump()`
3. Write `kernel-metadata.json`
4. Verify with `python -c "import json; ..."`
5. Push: `kaggle kernels push -p notebooks/kaggle/09m_transreid_dfn2b/`