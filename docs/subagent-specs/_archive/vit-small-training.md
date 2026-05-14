# 09k: ViT-Small/16 Secondary Model Training

## Goal
Train a ViT-Small/16 as secondary ReID model for score-level ensemble with ViT-B/16 primary. Target: ≥65% mAP on CityFlowV2 validation.

## Architecture Decision

**Model**: `vit_small_patch16_224.augreg_in21k_ft_in1k` (timm)
- 22M params, embed_dim=384, 6 attention heads, 12 blocks
- ImageNet-21k pretrained → IN-1k fine-tuned (81.4% IN-1k top-1)
- OpenAI never released ViT-Small CLIP, so IN-21k is the strongest available pretraining

**Fallback**: `deit_small_patch16_224` (DeiT-Small, distillation-trained, 79.9% IN-1k top-1)

**Recipe**: TransReID (SIE + JPM + overlap patch embed) — same as primary
- Rationale: maximizing secondary mAP is more important than feature decorrelation
- Diversity already comes from: 4× fewer params, 6 vs 12 heads, 384D vs 768D, IN-21k vs CLIP pretraining, 224 vs 256 resolution

## Key Differences from Primary (Ensemble Diversity Sources)

| Dimension | Primary (ViT-B/16) | Secondary (ViT-S/16) |
|-----------|--------------------|-----------------------|
| Params | 86M | 22M |
| Heads | 12 | 6 |
| Embed dim | 768 | 384 |
| Pretraining | CLIP (400M image-text) | IN-21k (14M images) |
| Resolution | 256×256 | 224×224 |
| Normalization | CLIP stats | ImageNet stats |
| PCA output | 384D | 256D |

## Training Hyperparameters

```yaml
# Model
vit_model: "vit_small_patch16_224.augreg_in21k_ft_in1k"
img_size: 224
embed_dim: 384
num_cameras: 59
sie_camera: true
jpm: true

# Data
batch_size: 64            # PK sampler: P=16, K=4
num_workers: 4
normalize_mean: [0.485, 0.456, 0.406]  # ImageNet
normalize_std: [0.229, 0.224, 0.225]

# Augmentation (same as primary baseline)
resize: [240, 240]        # 224 + 16
random_hflip: 0.5
pad: 10
random_crop: [224, 224]
color_jitter: [0.2, 0.15, 0.1, 0.0]
random_erasing: {p: 0.5, scale: [0.02, 0.33], ratio: [0.3, 3.3], value: random}

# Losses (proven recipe — DO NOT use CircleLoss)
ce_label_smooth: 0.1      # Standard epsilon (primary used 0.05)
triplet_margin: 0.3        # Hard mining
center_weight: 5e-4
center_start_epoch: 15
jpm_loss_weight: 0.5       # Auxiliary JPM branch

# Optimizer
optimizer: AdamW
backbone_lr: 1e-4
head_lr: 1e-3
weight_decay: 5e-4
llrd_decay: 0.75           # Layer-wise LR decay
grad_clip_norm: 5.0

# Schedule
epochs: 120
warmup_epochs: 10          # Linear warmup
scheduler: CosineAnnealingLR
```

## Inference & Integration

### Stage 2 Config Additions
```yaml
stage2:
  reid:
    vehicle2:
      enabled: true
      model_name: "transreid"
      weights_path: "models/reid/vit_small_cityflowv2_best.pth"
      embedding_dim: 384
      input_size: [224, 224]
      vit_model: "vit_small_patch16_224.augreg_in21k_ft_in1k"
      num_cameras: 59
      clip_normalization: false
      save_separate: true
  pca:
    secondary:
      enabled: true
      n_components: 256
```

### Stage 4 Config
```yaml
stage4:
  association:
    secondary_embeddings:
      path: "data/outputs/embeddings_secondary.npy"
      weight: 0.25    # Start conservative; sweep 0.15-0.35
```

## Expected Performance

- **mAP estimate**: 66-72% (ViT-B/16 CLIP achieves 80.14%; ~4× fewer params + weaker pretraining → ~10-14pp drop)
- **Ensemble IDF1 impact**: +0.3 to +1.0pp MTMC IDF1 (if mAP ≥65%, unlike ResNet101-IBN-a's 52.77% which gave -0.1pp)
- **Ensemble weight sweet spot**: 0.20-0.30 for secondary (primary dominates)

## Notebook Structure (09k)

Same structure as `09_vehicle_reid_cityflowv2/` with these changes:
1. Model init: `vit_small_patch16_224.augreg_in21k_ft_in1k`, `embed_dim=384`, `img_size=224`
2. Normalization: ImageNet stats (not CLIP)
3. Input resolution: 224×224 (not 256×256)
4. Label smoothing: ε=0.1 (slightly more regularization for smaller model)
5. Save weights as `vit_small_cityflowv2_best.pth`

## Risks

1. **mAP < 65%** → ensemble still hurts (same failure mode as ResNet). Mitigation: IN-21k pretraining is strong; ViT architecture has proven CityFlowV2 advantage
2. **Feature correlation too high** → ensemble gain < expected. Mitigation: different pretraining (IN-21k vs CLIP) + different resolution (224 vs 256) provide diversity
3. **Overfitting** → 22M params on 128 IDs. Mitigation: standard augmentation + label smoothing ε=0.1 (higher than primary's 0.05)

## NOT Doing
- Different loss recipe (Circle loss is a dead end; CE+Triplet+Center is proven)
- 256×256 resolution (reduces diversity from primary)
- Feature concatenation (dead end: -1.6pp, mixes uncalibrated spaces)
- CLIP normalization (no CLIP weights → ImageNet stats appropriate)