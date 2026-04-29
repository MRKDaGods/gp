# 09m: CLIP RN50x4 Vehicle ReID Training on CityFlowV2

> **Replaces**: Previous 09m DFN-2B spec (contingency no longer needed after 09l v3 reached 78.61% mAP)

## Motivation

**10c v56 proved that two CLIP ViT-B/16 variants (OpenAI + LAION-2B) are too correlated for score-level fusion (-0.5pp)**. The ensemble needs **architecturally diverse** models — specifically a CNN backbone — to provide complementary identity signal. CLIP RN50x4 is a natural choice:

| Property | Primary (ViT-B/16 CLIP) | Secondary (LAION-2B ViT-B/16) | **Proposed (RN50x4 CLIP)** |
|----------|:-:|:-:|:-:|
| Architecture | Vision Transformer | Vision Transformer | **Modified ResNet** |
| Receptive field | Global (self-attention) | Global (self-attention) | **Local (convolutions)** |
| CLIP pretraining | OpenAI | LAION-2B | **OpenAI** |
| Feature dim | 768 | 768 | **640** |
| Params (visual) | ~86M | ~86M | **~87M** |
| Native resolution | 224 → 256 | 224 → 256 | **288** |

The CNN's local receptive fields capture different visual patterns (edges, textures, local shapes) than the ViT's global attention (holistic structure, long-range dependencies). This architectural diversity is exactly what AIC22 winning methods exploit with 3-5 model ensembles.

## Architecture

### CLIP RN50x4 Visual Encoder (from `open_clip`)

```
Input (B, 3, 288, 288)
  │
  ├─ 3-layer stem: conv1(3→40) → conv2(40→40) → conv3(40→80) → avgpool
  │   Spatial: 288 → 144
  │
  ├─ layer1: 4× Bottleneck (80 → 320)     Spatial: 144 → 72
  ├─ layer2: 6× Bottleneck (320 → 640)    Spatial: 72 → 36
  ├─ layer3: 10× Bottleneck (640 → 1280)  Spatial: 36 → 18
  ├─ layer4: 6× Bottleneck (1280 → 2560)  Spatial: 18 → 9
  │
  └─ AttentionPool2d(spatial=9, embed=2560, heads=40, output=640)
      → (B, 640)
```

### ReID Head

```
backbone output (B, 640)   ← raw features for Triplet + Center loss
  │
  ├─ BNNeck: BatchNorm1d(640, affine=True, bias frozen)
  │   → (B, 640)            ← normalized features for CE classifier + inference
  │
  └─ Classifier: Linear(640 → num_classes, bias=False)
      → (B, num_classes)     ← logits for CE loss
```

**No SIE, no JPM** — these are ViT-specific components (patch-token camera embedding and random patch shuffling). CNNs don't have discrete patch tokens.

### Model Code Snippet

```python
import open_clip
import torch.nn as nn
import torch.nn.functional as F

class CLIPResNetReID(nn.Module):
    """CLIP RN50x4 visual encoder + BNNeck ReID head."""
    
    def __init__(self, num_classes, embed_dim=640):
        super().__init__()
        clip_model = open_clip.create_model('RN50x4', pretrained='openai')
        self.backbone = clip_model.visual  # ModifiedResNet with AttentionPool2d
        self.feat_dim = embed_dim  # 640 from attention pool
        
        # BNNeck (standard ReID practice)
        self.bn = nn.BatchNorm1d(self.feat_dim)
        self.bn.bias.requires_grad_(False)
        
        # ID classifier
        self.cls_head = nn.Linear(self.feat_dim, num_classes, bias=False)
        nn.init.normal_(self.cls_head.weight, std=0.001)
    
    def forward(self, x):
        f = self.backbone(x)          # (B, 640) — pre-BN for triplet/center
        bn = self.bn(f)               # BNNeck
        if self.training:
            cls = self.cls_head(bn)
            return cls, f             # cls for CE, f for triplet+center
        return F.normalize(bn, p=2, dim=1)  # L2-normalized for retrieval
    
    def get_param_groups(self, backbone_lr, head_lr):
        """Two-group LR: backbone (lower) + head (higher)."""
        backbone_params = list(self.backbone.parameters())
        head_params = list(self.bn.parameters()) + list(self.cls_head.parameters())
        return [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params, "lr": head_lr},
        ]
```

### Why Attention Pool (not GAP)

| Option | Dim | Pros | Cons |
|--------|:---:|------|------|
| **Attention Pool (keep)** | 640 | Preserves CLIP pretraining; learned spatial weighting; compact dim | Fixed spatial size (9×9); requires 288×288 input |
| GAP + projection | 2560→D | Flexible input size; simpler | Discards pretrained attention weights; 2560D too large for 128 IDs |

**Decision**: Keep attention pool for v1. The 640D output is well-sized, and discarding the pretrained attention layer risks losing cross-domain transfer quality.

## Training Recipe

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Input resolution | **288×288** | CLIP RN50x4 native; attention pool expects 9×9 spatial |
| Backbone LR | **1e-4** | Same as 09l; proven for CLIP fine-tuning |
| Head LR | **1e-3** | 10× backbone, standard practice |
| Weight decay | **5e-4** | Same as 09l |
| Optimizer | **AdamW** | SGD is a dead end (09d: 30.27% mAP catastrophic) |
| Batch size | **64** | PKSampler(p=16, k=4); fits in 16GB |
| Epochs | **200** | Longer than 09l v2 (160) because no VeRi-776 warmup |
| Warmup | **10 epochs** | Linear LR warmup |
| LR schedule | **Cosine decay** after warmup | Same as 09l |
| EMA | **decay=0.9999** | Proven beneficial in 09l v3 |
| Label smoothing | **0.05** | Same as 09l |
| **No LLRD** | — | LLRD is ViT-specific; CNN uses 2-group LR |
| **No freezing** | — | Start fine-tuning all layers from epoch 0 |
| Mixed precision | **fp16 autocast** | Standard for Kaggle GPU |
| Grad clipping | **max_norm=5.0** | Same as 09l |

### Loss Functions

| Loss | Config | When | Rationale |
|------|--------|------|-----------|
| CE + Label Smoothing | eps=0.05 | All epochs | ID classification; LS prevents overconfident predictions |
| Triplet (hard mining) | margin=0.3 | All epochs | Metric learning; proven stable (unlike CircleLoss which is a dead end) |
| Center Loss | weight=5e-4, SGD lr=0.5 | Epoch ≥15 | Compactness; delayed to let features stabilize first |

**DO NOT USE CircleLoss** — confirmed catastrophic on all CLIP backbones (09 v4: 18.45% mAP, 09l v1: 20.36% mAP, training loss `inf` throughout).

### Augmentations (Baseline Stack)

```python
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]
H, W = 288, 288

train_tf = T.Compose([
    T.Resize((H + 16, W + 16), interpolation=T.InterpolationMode.BICUBIC),  # 304×304
    T.RandomHorizontalFlip(p=0.5),
    T.Pad(10),
    T.RandomCrop((H, W)),
    T.ColorJitter(brightness=0.2, contrast=0.15, saturation=0.1, hue=0.0),
    T.ToTensor(),
    T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    T.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value="random"),
])

test_tf = T.Compose([
    T.Resize((H, W), interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
])
```

Use the **baseline augmentation stack only**. The augmentation overhaul (RandomGrayscale, GaussianBlur, RandomPerspective) is a confirmed dead end for MTMC (-5.3pp in 10c v48/v49).

### Data Loading

- **Source**: CityFlowV2 downloaded from Google Drive (same as 09l)
- **No VeRi-776 pretraining**: Unlike the ViT path, no RN50x4 VeRi-776 checkpoint exists. Train CLIP → CityFlowV2 directly.
- **Crop extraction**: GT bounding boxes from video, same protocol as 09l
- **Split**: 70% train / 30% eval IDs (same as 09l), MIN_CAMS_FOR_EVAL=2
- **PKSampler**: p=16 IDs per batch, k=4 images per ID
- **Workers**: num_workers=4

### Evaluation Protocol

Same as 09l:
- **mAP + Rank-1** on CityFlowV2 eval split
- **Reranking** (k-reciprocal): computed but not used for model selection
- **Flip augmentation**: average features from original + horizontally flipped
- **Eval frequency**: every 20 epochs
- **Best model**: selected by highest mAP (not reranked mAP)

## Notebook Cell Structure

| Cell # | Type | Content | Notes |
|:------:|:----:|---------|-------|
| 1 | Markdown | Title, goals, recipe summary | |
| 2 | Code | `!pip install -q open_clip_torch timm matplotlib pandas loguru gdown` | Adds `open_clip_torch` |
| 3 | Markdown | "## 1. Setup" | |
| 4 | Code | Imports, device detection, output dirs | OUTPUT_DIR = `/kaggle/working/09m_clip_rn50x4/output` |
| 5 | Markdown | "## 1.5 Download CityFlowV2 from Google Drive" | |
| 6 | Code | CityFlowV2 download + extraction | **Identical to 09l** |
| 7 | Markdown | "## 2. CityFlowV2 Dataset — Crop Extraction" | |
| 8 | Code | Locate cameras, extract GT crops | **Identical to 09l** |
| 9 | Code | Build train/query/gallery splits | **Identical to 09l** |
| 10 | Markdown | "## 3. Data Loading + Losses" | |
| 11 | Code | CLIP normalization, transforms (288×288), ReIDDataset, PKSampler, DataLoaders | Resolution changed to 288 |
| 12 | Code | Loss functions: CE+LS, TripletLoss, CenterLoss | **Identical to 09l** (exclude CircleLoss) |
| 13 | Markdown | "## 4. Evaluation Functions" | |
| 14 | Code | extract_features, eval_market1501, compute_reranking, full_eval | Modified: `pass_cams=False` (no SIE), default feat dim 640 |
| 15 | Markdown | "## 5. CLIP RN50x4 ReID Model" | |
| 16 | Code | CLIPResNetReID class definition + instantiation | **New**: open_clip model, BNNeck, no SIE/JPM |
| 17 | Markdown | "## 6. Training" | |
| 18 | Code | Training loop: 200 epochs, cosine LR, EMA | Adapted: 2-group LR (no LLRD), no cam_ids |
| 19 | Code | Training curves plot | |
| 20 | Markdown | "## 7. Export Model" | |
| 21 | Code | Export best EMA checkpoint + metadata JSON | |
| 22 | Markdown | "## 8. Inference Integration" | Stage 2 config snippet |

### Key Differences from 09l

| Aspect | 09l (ViT LAION-2B) | **09m (RN50x4 CLIP)** |
|--------|---------------------|----------------------|
| Install | `timm` | `open_clip_torch` + `timm` |
| Resolution | 256×256 | **288×288** |
| Model class | TransReID (ViT + SIE + JPM) | **CLIPResNetReID (CNN + BNNeck only)** |
| Backbone | `timm.create_model('vit_base_patch16_clip_224.laion2b')` | **`open_clip.create_model('RN50x4', pretrained='openai').visual`** |
| Feature dim | 768 | **640** |
| VeRi-776 init | Yes (attached model source) | **No** (CLIP-only init) |
| LR groups | LLRD with per-block scaling | **2 groups: backbone + head** |
| SIE | Yes (camera embedding) | **No** |
| JPM | Yes (patch shuffling) | **No** |
| cam_ids in forward | Yes | **No** |
| Epochs | 300 (resumed from 160) | **200** (from scratch) |
| Resume | From 09l v2 checkpoint | **No resume** (fresh training) |

## Dataset Access on Kaggle

### CityFlowV2 (Required)

Downloaded from Google Drive at runtime (same as 09l):
- **Google Drive ID**: `13wNJpS_Oaoe-7y5Dzexg_Ol7bKu1OWuC`
- **Archive**: `AIC22_Track1_MTMC_Tracking.zip` (~20GB)
- **Extracted to**: `/tmp/cityflowv2/` (not /kaggle/working/ to save space)
- **Splits used**: train + validation (have GT), test excluded (no GT)
- **Cameras**: 46 unique cameras across 5 scenes (S01-S05)

### VeRi-776 (NOT needed)

No VeRi-776 pretrained weights exist for CLIP RN50x4. Training starts from CLIP-only init. If v1 underperforms, a follow-up 09m-veri notebook could train CLIP RN50x4 → VeRi-776 → CityFlowV2 (3-stage progressive specialization).

## Export

### Output Files

```
/kaggle/working/09m_clip_rn50x4/output/
  ├── clip_rn50x4_cityflowv2_best_ema.pth      # Best EMA checkpoint by mAP
  ├── clip_rn50x4_cityflowv2_last_ema.pth       # Final EMA checkpoint
  └── training_curves.png                         # Loss + metrics plot

/kaggle/working/exported_models/
  ├── clip_rn50x4_cityflowv2.pth                 # Exported for stage2
  └── clip_rn50x4_cityflowv2_metadata.json        # Training metadata
```

### Export Format

```python
torch.save({"state_dict": ema_state_dict}, export_path)
```

The exported checkpoint contains the full `CLIPResNetReID` state dict (backbone + BNNeck + classifier). Stage 2 loads it, instantiates `CLIPResNetReID`, calls `load_state_dict`, and uses the model in eval mode.

### Metadata JSON Schema

```json
{
  "task": "vehicle_reid",
  "dataset": "cityflowv2",
  "model": {
    "architecture": "clip_rn50x4",
    "type": "clip_resnet_reid",
    "embedding_dim": 640,
    "input_size": [288, 288],
    "normalization": {"mean": [0.48145466, 0.4578275, 0.40821073],
                      "std": [0.26862954, 0.26130258, 0.27577711]},
    "tricks": ["BNNeck", "CE+LS(0.05)", "TripletLoss(m=0.3)", "CenterLoss(delayed@ep15)",
               "CosLR", "RE", "CLIP-norm", "AugBaseline", "EMA"],
    "best_mAP": 0.0,
    "epochs": 200,
    "backbone_lr": 1e-4,
    "head_lr": 1e-3,
    "ema_decay": 0.9999
  }
}
```

## Stage 2 Integration

### Config Addition (`default.yaml`)

The trained model would be deployed as `vehicle3` (tertiary) or replace `vehicle2`:

```yaml
stage2:
  reid:
    vehicle3:
      enabled: true
      save_separate: true
      model_name: "clip_rn50x4"
      weights_path: "models/reid/clip_rn50x4_cityflowv2.pth"
      embedding_dim: 640
      input_size: [288, 288]
      clip_normalization: true
```

### Code Changes Required in `src/stage2_features/reid_model.py`

Add to `_build_model()`:
```python
elif model_name.lower() == "clip_rn50x4":
    return self._build_clip_rn50x4(weights_path)
```

Implement `_build_clip_rn50x4()`:
```python
def _build_clip_rn50x4(self, weights_path):
    import open_clip
    clip_model = open_clip.create_model('RN50x4', pretrained=None)
    backbone = clip_model.visual
    
    # Build full ReID model
    model = CLIPResNetReID.__new__(CLIPResNetReID)
    nn.Module.__init__(model)
    model.backbone = backbone
    model.feat_dim = 640
    model.bn = nn.BatchNorm1d(640)
    model.bn.bias.requires_grad_(False)
    model.cls_head = nn.Linear(640, 1, bias=False)  # dummy, not used at inference
    
    # Load trained weights
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    return model
```

### PCA Integration

The 640D embeddings go through the existing tertiary PCA path:
- `pca.tertiary_pca_model_path: "models/reid/pca_transform_tertiary.pkl"`
- PCA is fitted on the first run and cached

### Stage 4 Fusion

```yaml
stage4:
  association:
    tertiary_embeddings:
      path: ""  # auto-detected from stage2 output
      weight: 0.2  # tune: start at 0.2, sweep 0.1-0.4
```

## kernel-metadata.json

```json
{
  "id": "gumfreddy/09m-clip-rn50x4-vehicle-reid",
  "title": "09m CLIP RN50x4 Vehicle ReID CityFlowV2",
  "code_file": "09m_clip_rn50x4.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "machine_shape": "NvidiaTeslaT4",
  "dataset_sources": [],
  "competition_sources": [],
  "kernel_sources": [],
  "model_sources": []
}
```

**Notes**:
- `enable_internet: true` — needed for Google Drive download + `open_clip` model download
- No `kernel_sources` or `model_sources` — no VeRi-776 weights to attach
- `machine_shape: "NvidiaTeslaT4"` — T4 (16GB) is sufficient; P100 also works

## Expected Training Time

| Phase | Estimated Time | Notes |
|-------|:-:|------|
| CityFlowV2 download + extraction | ~15-20 min | ~20GB from Google Drive |
| Crop extraction from video | ~10-15 min | 46 cameras, GT bboxes |
| Training (200 epochs) | ~4-5 hours | ~87M params at 288×288, batch 64 |
| Evaluation (10 checkpoints) | ~15-20 min | query + gallery forward passes |
| **Total** | **~5-6 hours** | Within Kaggle 12-hour limit |

## VRAM Budget (16GB)

| Component | Estimated | Notes |
|-----------|:-:|------|
| Model parameters | ~350MB | 87M × 4B |
| Gradients | ~350MB | Same as params |
| Optimizer (AdamW) | ~700MB | 2 momentum buffers |
| EMA copy | ~350MB | Copy of model params |
| Activations (batch 64, 288px) | ~4-6GB | Peak at layer3-4 |
| PyTorch overhead | ~2GB | CUDA context, allocator |
| **Total** | **~8-10GB** | Comfortable within 16GB |

If VRAM is tight, reduce batch_size to 48 (still fits PKSampler with p=16, k=3).

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|:----------:|:------:|------------|
| mAP < 65% without VeRi-776 warmup | **Medium** | High — model not useful for ensemble | Train VeRi-776 intermediate step (09m-veri follow-up) |
| Attention pool spatial mismatch at non-288px | **Low** (using 288) | High — broken features | Fixed by using native 288×288 |
| VRAM OOM at batch 64 | **Low** | Medium — slower training | Reduce to batch 48; add gradient checkpointing |
| CircleLoss accidentally included | **Low** | Critical — destroys training | **NEVER include CircleLoss**; use Triplet only |
| open_clip version incompatibility on Kaggle | **Low** | Medium — model won't load | Pin version: `pip install open_clip_torch==2.24.0` |
| Training time exceeds 12h Kaggle limit | **Low** | Medium — incomplete run | 200 epochs estimated at 5-6h; safe margin |
| Features too correlated with primary ViT | **Low** | High — ensemble doesn't help | CNN features should be architecturally diverse; verify with cosine similarity analysis |
| CityFlowV2 Google Drive link expires | **Medium** | Critical — no data | Document backup links; consider Kaggle Dataset upload |

### Highest Risk: mAP < 65%

The non-CLIP ceiling (~50%) is established, and CLIP pretraining is the key factor. However, all models that exceeded 65% so far used the 3-stage pipeline: CLIP → VeRi-776 → CityFlowV2. Going directly CLIP → CityFlowV2 with a CNN has not been tested.

**Evidence for optimism**: The CLIP RN50x4 pretrained features are strong general-purpose visual features. Even without vehicle-specific VeRi-776 warmup, the CLIP pretraining should provide much better initialization than ImageNet (which gave ResNet101-IBN-a only 52.77%).

**Contingency**: If v1 yields < 65% mAP:
1. **v2**: Add VeRi-776 intermediate training (requires a separate 09m-veri notebook)
2. **v3**: Try GAP instead of attention pool (2560D → projection to 768D, then standard BNNeck)
3. **v4**: Try CLIP RN50x16 (larger, 936D) if RN50x4 is too weak

## Success Criteria

| Criterion | Threshold | Priority |
|-----------|:---------:|:--------:|
| Training completes without errors | — | P0 |
| Training loss is finite throughout | — | P0 |
| CityFlowV2 val **mAP ≥ 65%** | 65% | **P1** |
| CityFlowV2 val **R1 ≥ 80%** | 80% | P2 |
| Exported model loads in stage2 | — | P2 |
| Feature cosine similarity with primary < 0.7 mean | < 0.7 | P3 |

## Relation to Pipeline Slot Naming

| Notebook | Model | Status |
|----------|-------|--------|
| 09b | TransReID ViT-B/16 CLIP (OpenAI) 256px | **Primary** — deployed, mAP=80.14% |
| 09l | TransReID ViT-B/16 CLIP (LAION-2B) 256px | **Dead end for ensemble** — mAP=78.61% but too correlated with primary |
| **09m** | **CLIP RN50x4 (OpenAI) 288px** | **Proposed** — architecturally diverse CNN secondary |
| 09d/09i/09j | ResNet/ResNeXt-IBN-a variants | **Dead end** — ceiling ~52.77% mAP, all 6 variants exhausted |
| 09k | ViT-Small/16 IN-21k | **Dead end** — 48.66% mAP, confirms non-CLIP ceiling |

---

## Appendix: open_clip RN50x4 Details

- **Model string**: `open_clip.create_model('RN50x4', pretrained='openai')`
- **Visual encoder**: `model.visual` → `ModifiedResNet`
- **Layers**: [4, 6, 10, 6] (layer1-4 block counts)
- **Width**: 80 (base channel width)
- **Output dim (attention pool)**: 640
- **Output dim (GAP, no attention pool)**: 2560
- **Native resolution**: 288×288
- **Spatial at layer4 output**: 9×9 (288 / 32)
- **Available pretrained**: `'openai'` only (no LAION variants for RN50x4)
- **Total params (visual only)**: ~87M

---

END OF SPEC