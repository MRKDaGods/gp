# Pretrained Vehicle ReID Ensemble — Spec

## Status: READY FOR IMPLEMENTATION

## Motivation

All prior ensemble attempts trained secondary models from scratch on CityFlowV2's tiny 128-ID training set:
- ResNet101-IBN-a → 52.77% mAP (too weak, -0.1pp fusion)
- ResNeXt101-IBN-a → 36.88% mAP (partial weight loading, dead end)
- ViT-Small/16 IN-21k → 48.66% mAP (non-CLIP ceiling)
- CLIP RN50x4 → 1.55% mAP (catastrophic)
- LAION-2B CLIP ViT → 78.61% mAP (strong but too correlated, -0.5pp)

**Key insight**: We never tried using a community-pretrained vehicle ReID model that was already trained on VeRi-776 (576 IDs, 37K images) or VehicleNet (multi-dataset blend). These models:
1. Are already strong on vehicle ReID (80-83% mAP on VeRi-776)
2. Use CNN architectures (maximally diverse from our CLIP ViT-B/16)
3. Were trained with different methodologies (BoT/SBS/VehicleNet recipes)
4. Need NO CityFlowV2 fine-tuning — just extract features from existing crops

## Question 1: Why Ensemble Has Failed So Far

### LAION-2B CLIP ViT (78.61% mAP) → -0.5pp
- **Root cause**: Feature correlation. Both primary (OpenAI CLIP ViT-B/16) and secondary (LAION-2B CLIP ViT-B/16) share the same architecture family, same patch-level attention processing, same 768D feature space geometry
- Both models learned similar discriminative patterns because CLIP pretraining encourages the same visual-semantic alignment regardless of dataset
- Fusion at 30% weight adds noise because the secondary disagrees only on ambiguous cases where neither model is reliable

### ResNet101-IBN-a (52.77% mAP) → -0.1pp
- **Root cause**: Quality gap. The 28pp mAP deficit means the secondary model is near-random for cross-camera matching
- At 30% fusion weight, 30% of similarity scores come from a model that cannot distinguish same-model vehicles across viewpoints
- The model was architecturally diverse but starved of training data (128 IDs)

### The Sweet Spot for a Viable Secondary
- **Minimum quality**: ≥70% mAP on VeRi-776 (proxy for cross-camera vehicle discrimination)
- **Ideal quality**: 80%+ mAP on VeRi-776 (comparable to primary on a standard benchmark)
- **Architecture**: Must be a CNN (ResNet/ResNeXt) to ensure orthogonal feature representations vs our ViT
- **Training data**: VeRi-776 or larger (VehicleNet) — NOT just CityFlowV2's 128 IDs
- **Training recipe**: Different from our CE+Triplet+CLIP — ideally BoT/SBS with circle loss, GeM pooling, non-local blocks

## Question 2: Available Pretrained Vehicle ReID Models

### Candidate #1: fast-reid SBS(R50-ibn) on VeRi-776 ⭐⭐⭐
| Property | Value |
|----------|-------|
| **Architecture** | ResNet50-IBN-a + Non-local blocks + GeM pooling + BNNeck |
| **Training data** | VeRi-776 (576 IDs, 37K images) |
| **VeRi-776 metrics** | R1=97.0%, mAP=81.9%, mINP=46.3% |
| **Feature dim** | 2048 (before BNNeck) |
| **Input size** | 256×256 (standard fastreid vehicle config) |
| **Download URL** | `https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/veri_sbs_R50-ibn.pth` |
| **Source** | [JDAI-CV/fast-reid](https://github.com/JDAI-CV/fast-reid) MODEL_ZOO.md |
| **License** | Apache 2.0 |
| **Downloadable** | ✅ Direct GitHub releases (no Google Drive) |
| **Diversity from primary** | ✅ Maximum (CNN vs ViT, GeM vs CLS token, circle loss vs triplet, ImageNet vs CLIP init) |

**Why #1**: Direct download from GitHub releases (no Google Drive auth), 81.9% mAP exceeds our 70% threshold, and ResNet50-IBN with non-local+GeM is maximally diverse from CLIP ViT-B/16. The SBS recipe (circle loss, freeze backbone, cutout, auto-aug, cosine LR, soft-margin triplet, non-local, GeM) produces features with fundamentally different discriminative biases.

### Candidate #2: layumi VehicleNet+VeRi ResNet50-IBN-a ⭐⭐
| Property | Value |
|----------|-------|
| **Architecture** | ResNet50 with IBN-a, label smoothing, random erasing (Person_reID_baseline_pytorch format) |
| **Training data** | VehicleNet (CompCars+VehicleID+VeRi blend) → fine-tuned on VeRi-776 |
| **VeRi-776 metrics** | mAP=83.41% (strongest published ResNet50 vehicle ReID result) |
| **Feature dim** | 512 (projection) or 2048 (backbone) |
| **Input size** | 256×256 |
| **Download URL** | `https://drive.google.com/file/d/1Sor7Grh_1Kot6CBLaw2alDT4Nr3JuH3C/view?usp=sharing` |
| **Source** | [layumi/AICIty-reID-2020](https://github.com/layumi/AICIty-reID-2020) |
| **License** | MIT |
| **Downloadable** | ⚠️ Google Drive (may need gdown on Kaggle) |
| **Diversity from primary** | ✅ High (CNN vs ViT, VehicleNet pretraining vs CLIP, simpler recipe) |

**Why #2**: Highest published mAP (83.41%) for a ResNet50 on VeRi-776. VehicleNet pretraining exposes the model to ~220K vehicle images across multiple datasets, giving broader vehicle appearance coverage. Risk: Google Drive download may be flaky on Kaggle; model format from Person_reID_baseline_pytorch needs adaptation.

### Candidate #3: fast-reid BoT(R50-ibn) on VehicleID ⭐
| Property | Value |
|----------|-------|
| **Architecture** | ResNet50-IBN-a + BNNeck (standard BoT recipe, no non-local or GeM) |
| **Training data** | VehicleID (26K IDs, 221K images) |
| **VehicleID metrics** | R1=86.6% (small), 82.9% (medium), 80.6% (large) |
| **Feature dim** | 2048 |
| **Input size** | 256×256 |
| **Download URL** | `https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/vehicleid_bot_R50-ibn.pth` |
| **Source** | [JDAI-CV/fast-reid](https://github.com/JDAI-CV/fast-reid) MODEL_ZOO.md |
| **License** | Apache 2.0 |
| **Downloadable** | ✅ Direct GitHub releases |
| **Diversity from primary** | ✅ High (CNN vs ViT, simpler BoT recipe, VehicleID distribution) |

**Why #3**: Trained on VehicleID (26K IDs) — much larger identity space than VeRi-776. Simpler BoT architecture is easier to integrate (no non-local blocks). Risk: VehicleID has a different viewpoint distribution from CityFlowV2, so features may transfer less well than VeRi-trained models.

### Rejected Candidates
| Model | Reason for Rejection |
|-------|---------------------|
| **CLIP-ReID ViT on VeRi** | Same CLIP ViT architecture family as primary → would be correlated like 09l v3 |
| **TransReID official VeRi** | No pretrained weights available for download (code-only repo) |
| **SOLIDER** | Human-centric only, no vehicle models |
| **AIC22 1st place (heshuting555)** | Repository returns 404, weights not publicly available |
| **fast-reid VERI-Wild BoT(R50-ibn)** | VERI-Wild is a different benchmark; VeRi-776 models are closer to CityFlowV2 distribution |

## Question 3: Strategy for Pretrained Ensemble

### Core Approach
1. **NO CityFlowV2 fine-tuning** — use the pretrained model as-is for feature extraction
2. Extract features from the same Stage-1 crops that the primary model uses
3. PCA whiten to 384D (same as primary) with a SEPARATE PCA model fitted on the pretrained features
4. FIC whitening applied separately per model in Stage 4
5. Score-level fusion in 10c with weight sweep 0.0–0.5

### Why This Should Work (Unlike Previous Attempts)
- **Quality**: fast-reid SBS at 81.9% mAP on VeRi is comparable to our primary's 80.14% mAP on CityFlowV2
- **Diversity**: ResNet50-IBN + non-local + GeM sees completely different low-level patterns than CLIP ViT-B/16
  - CNN: local receptive fields, texture-biased, IBN handles style variation
  - ViT: global self-attention, shape-biased, CLIP alignment
  - Non-local blocks: explicit long-range dependency modeling vs ViT's attention
  - GeM pooling: emphasizes high-activation regions vs CLS token aggregation
- **Data coverage**: VeRi-776 has 576 IDs in traffic camera settings — similar domain to CityFlowV2 but 4.5× more identities
- **No overfitting risk**: Model has never seen CityFlowV2, so no risk of memorizing the tiny 128-ID training set

### Risk Assessment
| Risk | Mitigation |
|------|-----------|
| VeRi→CityFlowV2 domain gap | FIC whitening calibrates per-camera, PCA aligns feature distributions |
| Feature dim mismatch (2048 vs 768) | Independent PCA to 384D for each model |
| SBS non-local/GeM architecture complexity | Can strip non-local blocks if needed (still functional, just fewer features) |
| Fast-reid state dict key mismatch | Map keys from fastreid format to our ResNet50-IBN-a model |
| Pretrained features may not FIC-calibrate well | FIC regularization sweep in 10c should handle this |

## Question 4: Implementation Plan

### Phase 1: Model Integration (10a notebook changes)

#### Step 1: Download weights
```python
# In 10a notebook, early cell
!wget -q https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/veri_sbs_R50-ibn.pth     -O models/reid/fastreid_veri_sbs_r50ibn.pth
```

#### Step 2: Build fast-reid compatible model
Need to implement a `FastReIDResNet50IBN` class that:
- Builds ResNet50-IBN-a backbone (our codebase already has IBN-a support via `ReIDModelResNet101IBN`)
- Adds Non-local block after layer3 (or load with strict=False and skip)
- Adds GeM pooling (replace global avg pool)
- Adds BNNeck
- Maps fastreid state dict keys to our naming convention

Key fastreid state dict mappings:
```
fastreid "backbone.conv1.weight"    → "backbone.conv1.weight"
fastreid "backbone.layer1.0.*"      → "backbone.layer1.0.*"
fastreid "heads.bnneck.0.weight"    → "bn_neck.weight" (or similar)
fastreid "heads.bnneck.0.bias"      → "bn_neck.bias"
```

**Simpler alternative**: Use our existing ResNet101-IBN-a loader but adapted for ResNet50-IBN-a (4 fewer layers per block). Load with `strict=False`, accepting that non-local block weights will be missing. The backbone features will still be meaningful without non-local — it's an additive module.

#### Step 3: Feature extraction (same crops, same quality weighting)
```python
# Extract from same crops as primary
secondary_model = build_fastreid_r50ibn("models/reid/fastreid_veri_sbs_r50ibn.pth")
# Same preprocessing but with ImageNet normalization (not CLIP)
# Same flip augmentation, quality-weighted pooling
# Output: 2048D features per tracklet
raw_secondary = extract_features(secondary_model, crops, normalize="imagenet")
```

#### Step 4: PCA whiten to 384D
```python
# Fit SEPARATE PCA on secondary features
from sklearn.decomposition import PCA
pca_secondary = PCA(n_components=384, whiten=True)
secondary_384d = pca_secondary.fit_transform(raw_secondary)
secondary_384d = l2_normalize(secondary_384d)
np.save(f"{run_dir}/stage2/embeddings_secondary.npy", secondary_384d)
```

#### Step 5: Save for 10c
The secondary embeddings are saved to `embeddings_secondary.npy` — the 10c notebook already knows how to load and fuse these.

### Phase 2: Fusion Sweep (10c notebook)

The 10c notebook already has score-level fusion infrastructure:
```python
FUSION_WEIGHT = 0.30  # sweep 0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5
"--override", f"stage4.association.secondary_embeddings.path={path}",
"--override", f"stage4.association.secondary_embeddings.weight={weight}",
```

Sweep config matrix for 10c:
- `secondary_weight`: [0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
- `fic_reg`: [0.10, 0.30, 0.50, 1.00] (secondary may need different FIC)
- Keep primary association params at v52 baseline (`sim_thresh=0.50`, `appearance_weight=0.70`, etc.)

### Phase 3: Evaluate (expected from 10c output)
- Compare MTMC IDF1 with and without pretrained secondary
- Check per-camera breakdown for systematic improvements
- Analyze conflated vs fragmented ID changes

### Architecture Details for Each Candidate

#### fast-reid SBS(R50-ibn) — Build Instructions
```python
import torch
import torch.nn as nn
import torchvision.models as models

class FastReIDR50IBN(nn.Module):
    """Minimal fast-reid SBS ResNet50-IBN-a for inference."""
    def __init__(self):
        super().__init__()
        # ResNet50-IBN-a backbone
        # fastreid uses resnet_ibna from their own implementation
        # Key: last_stride=1 (same as our ResNet101-IBN setup)
        self.backbone = build_resnet50_ibn_a(last_stride=1)
        # GeM pooling
        self.pool = GeneralizedMeanPooling(p=3.0)
        # BNNeck
        self.bn_neck = nn.BatchNorm1d(2048)
        self.bn_neck.bias.requires_grad_(False)

    def forward(self, x):
        feat = self.backbone(x)  # (B, 2048, H, W)
        pooled = self.pool(feat)  # (B, 2048, 1, 1)
        pooled = pooled.flatten(1)  # (B, 2048)
        bn_feat = self.bn_neck(pooled)  # (B, 2048)
        return bn_feat  # Use BN features for inference
```

#### layumi VehicleNet — Build Instructions
```python
# Simpler architecture: standard ResNet50 with last_stride=1
# Model from Person_reID_baseline_pytorch
class VehicleNetR50(nn.Module):
    def __init__(self):
        super().__init__()
        model = models.resnet50(pretrained=False)
        model.layer4[0].downsample[0].stride = (1, 1)
        model.layer4[0].conv2.stride = (1, 1)
        self.backbone = nn.Sequential(*list(model.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn = nn.BatchNorm1d(2048)

    def forward(self, x):
        feat = self.backbone(x)
        pooled = self.pool(feat).flatten(1)
        return self.bn(pooled)
```

### Execution Order
1. **Implement in 10a**: Add secondary model extraction cell for fast-reid SBS R50-ibn
2. **Push 10a**: Run on Kaggle (10a extracts both primary + secondary features)
3. **Chain to 10b → 10c**: Standard pipeline chain
4. **Sweep in 10c**: Fusion weight sweep over [0.0, 0.1, 0.2, 0.3, 0.4]
5. **Evaluate**: Compare MTMC IDF1 with baseline

### Expected Impact
- **Optimistic**: +1-3pp MTMC IDF1 (closing gap to ~79-80%)
- **Realistic**: +0.5-1.5pp (meaningful improvement from diversity)
- **Pessimistic**: +0pp or negative (VeRi→CityFlowV2 domain gap too large)

The optimistic case is plausible because:
1. AIC22 winners got their biggest gains from ensemble diversity, not single-model quality
2. Our primary model's failure modes (same-model vehicles, viewpoint changes) are exactly what a differently-trained CNN would complement
3. The 81.9% mAP quality threshold is far above the 70% minimum we identified

### What Could Go Wrong
1. **Domain gap**: VeRi-776 cameras are Chinese traffic surveillance; CityFlowV2 is US intersection cameras. Different vehicle distributions, license plates, road markings.
2. **Feature calibration**: The pretrained model's feature distribution may be so different from our primary that PCA+FIC can't align them well enough for score-level fusion.
3. **Non-local block loading**: If the fastreid state dict requires exact non-local block architecture, we may need to implement it rather than skip with strict=False.

### Why This Is Different From All Previous Attempts
| Previous Attempt | Problem | This Approach |
|-----------------|---------|---------------|
| Train R101-IBN on CityFlowV2 | Only 128 IDs → 52.77% mAP ceiling | Pretrained on 576 IDs → 81.9% mAP |
| Train ResNeXt on CityFlowV2 | Weight mismatch → 36.88% mAP | Using official pretrained weights |
| LAION-2B CLIP ViT fusion | Too correlated with primary | CNN architecture = orthogonal features |
| CLIP RN50x4 from scratch | Recipe broken → 1.55% mAP | No training needed, just inference |
| VeRi→CityFlowV2 fine-tune | Fine-tuning hurt (42.7% < 52.77%) | No fine-tuning at all |
