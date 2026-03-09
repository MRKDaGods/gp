# SOTA Analysis & Implementation Plan

## Target Datasets & SOTA Benchmarks

### 1. Market-1501 (Person ReID)
| Method | mAP | Rank-1 | Year | Key Technique |
|--------|-----|--------|------|---------------|
| SOLIDER (Swin-B) + Re-ranking | **95.6** | **96.7** | 2023 | Self-supervised human pre-training |
| SOLIDER (Swin-B) | 93.9 | 96.9 | 2023 | Semantic controllable SSL |
| Swin (all tricks + Circle + DG) | 94.0 | 85.36 | 2022 | Swin Transformer + losses |
| TransReID* (ViT-B/16, stride 12) | 89.0 | 95.1 | 2021 | ViT + JPM + SIE |
| BoT (IBN-Net50-a) | 88.2 | 95.0 | 2019 | ResNet50-IBN + tricks |
| BoT (IBN-Net50-a) + Re-ranking | **94.2** | **95.4** | 2019 | + k-reciprocal re-ranking |
| **Our system (ResNet50-IBN-a)** | ~85-88 | ~94-95 | - | torchreid + PCA whitening |

### 2. VeRi-776 (Vehicle ReID)
| Method | mAP | Rank-1 | Year | Key Technique |
|--------|-----|--------|------|---------------|
| TransReID* (ViT-B/16) | **82.1** | **97.4** | 2021 | ViT + JPM + SIE + viewpoint |
| AICITY2020 DMT | ~80 | ~96 | 2020 | Multi-domain training |
| VehicleNet (ResNet50-IBN) | ~77 | ~96 | 2020 | Cross-domain pre-training |
| **Our system (ResNet50-IBN-a)** | ~75 | ~95 | - | torchreid + PCA whitening |

### 3. WILDTRACK (Multi-View Detection)
| Method | MODA | MODP | Year | Key Technique |
|--------|------|------|------|---------------|
| MVDeTr (Deformable Transformer) | **91.5** | 82.1 | 2021 | Shadow transformer + focal loss |
| MVDet (Conv) | 88.2 | 75.7 | 2020 | Multi-view feature projection |
| 3DROM | 88.1 | - | 2020 | 3D reconstruction |
| POM + KSP | 80.3 | 75.4 | 2008 | Probabilistic occupancy map |
| **Our pipeline (detect→track→project)** | ~0-5 | - | - | Per-camera YOLO → back-project |

### 4. CityFlow / AI City Challenge (MTMC Vehicle Tracking)
| Method | IDF1 | MOTA | Year | Key Technique |
|--------|------|------|------|---------------|
| AIC21 1st Place (Liu et al.) | **84.1** | ~80 | 2021 | Zone-based ST + Re-ranking |
| BoTrack | ~80 | ~75 | 2022 | Strong ReID + IoU matching |
| **Our pipeline** | ~30 | ~-60 | - | FAISS + Louvain (on WILDTRACK) |

---

## Gap Analysis: Our Pipeline vs SOTA

### GAP 1: ReID Backbone (HIGH IMPACT)
**Current**: ResNet50-IBN-a via torchreid (2048-dim) → PCA to 256/512
**SOTA**: TransReID ViT-B/16 with JPM + SIE, or SOLIDER Swin-B
**Impact**: +3-5% mAP on Market-1501, +5-7% mAP on VeRi-776

**Problem**: TransReID/SOLIDER require 7-12GB GPU VRAM for inference. Our GTX 1050 Ti has 4GB.

**Solution**: Use **CLIP-ReID** or **SOLIDER** pre-trained weights fine-tuned to smaller models that our hardware can handle, OR use ViT-Small/DeiT-Small (fits in 4GB). Alternatively, load pre-extracted features for evaluation, and use ResNet50-IBN-a with ALL BoT tricks for deployment:
- **BoT Recipe** (achievable on our hardware):
  - ResNet50-IBN-a backbone ✅ (already have)
  - BNNeck (batch norm before classifier) → need to verify
  - Triplet loss + ID loss + Center loss → need training script
  - Warm-up LR schedule
  - Random erasing augmentation (p=0.5)
  - Label smoothing (0.1)
  - Last stride = 1
  - **Re-ranking** at test time ✅ (already have k-reciprocal)
  - This alone achieves: Market mAP=94.2%, R1=95.4% (with re-ranking)

**Action**: The most practical approach is to train ResNet50-IBN-a with the full BoT recipe on Market-1501 and VeRi-776, then use those weights. The pre-trained weights we have may already include some of these tricks.

### GAP 2: Association / Clustering (MEDIUM IMPACT)
**Current**: FAISS top-K → mutual NN → k-reciprocal re-ranking → weighted fusion → Louvain community detection → gallery expansion
**SOTA** (AIC21 1st Place):
  - Zone-based spatio-temporal constraints (crossroad zone transition model)
  - Camera-aware distance bias (CID_BIAS)
  - Hierarchical clustering with timestamp ordering
  - Multi-model ReID ensemble (2× ResNet101-IBN + 1× ResNeXt101-IBN)

**Our advantages**: We already have k-reciprocal re-ranking, mutual NN, and Louvain. 
**Key missing pieces**:
  1. **Zone-based spatio-temporal model** — AIC21 defines entry/exit zones per camera and computes transition time distributions between zone pairs. This is critical for CityFlow.
  2. **Camera distance bias** — Per camera-pair bias terms to calibrate cross-camera distances
  3. **Hierarchical association** — First associate within scene, then across scenes
  4. **Ensemble ReID** — Average features from multiple models

### GAP 3: WILDTRACK-Specific (HIGH IMPACT for WILDTRACK metric)
**Current**: Detect per-camera → track → project to ground plane → NMS → evaluate  
**SOTA**: MVDeTr / MVDet use end-to-end learned multi-view feature projection to a  ground-plane occupancy grid. This is a fundamentally different architecture.

**Reality check**: Achieving 88-91% MODA on WILDTRACK requires training MVDeTr (needs ground-plane supervision). Our architecture can't match this without a dedicated multi-view fusion module. However, our pipeline is correct for city-wide MTMC (CityFlow, VeRi, Market).

**Practical approach for WILDTRACK+our architecture**:
  1. Per-camera tracking (Stage 1) ✅
  2. Multi-view triangulation / consensus filtering for ground-plane eval
  3. Hungarian matching on ground plane for identity assignment
  4. This gives reasonable MOTA/IDF1 on the 2D per-camera protocol, and a separate GP MODA for the ground-plane protocol

### GAP 4: Training Pipeline (NEEDED for SOTA)
**Current**: We load pre-trained weights, no training code
**Needed**: Training script for ReID to:
  - Fine-tune on target domain (Market/VeRi/CityFlow training splits)
  - Apply BoT tricks (BNNeck, triplet+ID+center loss, warmup, label smoothing, RE)
  - This is the single biggest improvement we can make

---

## Implementation Roadmap (Priority Order)

### Phase 1: ReID Training Pipeline (Biggest bang for the buck)
1. Create `src/training/reid_trainer.py` — BoT-style training
2. Create `src/training/datasets/` — Market-1501, VeRi-776, MSMT17 dataset loaders
3. Create `src/training/losses/` — Triplet, Center, Circle losses
4. Support: BNNeck, label smoothing, warmup, random erasing, last stride=1
5. Train on Market-1501 → expect mAP ~85-88% without re-ranking, ~93-94% with
6. Train on VeRi-776 → expect mAP ~77-80%

### Phase 2: Association Improvements
1. **Camera distance bias** — Learn per camera-pair distance offsets from training data
2. **Zone-based ST model** — For CityFlow: define camera entry/exit zones  
3. **Hierarchical clustering** — Scene-level → cross-scene association
4. **Temporal consistency constraints** — Enforce timestamp ordering in trajectories

### Phase 3: Evaluation Protocols
1. **Market-1501 / VeRi-776 pure ReID eval** — Standard mAP/CMC evaluation script
2. **WILDTRACK ground-plane eval** — Already implemented ✅
3. **CityFlow MTMC eval** — IDF1/MOTA with the AIC metric protocol
4. **Cross-dataset generalization** — Train on Market, test on WILDTRACK etc.

### Phase 4: Advanced (if hardware allows)
1. **ViT-Small ReID** — If we can fit ViT-Small (4GB), big accuracy boost
2. **Multi-model ensemble** — Average features from 2+ models
3. **Domain adaptation** — Use DG-Market synthetic data for training
4. **SOLIDER weights** — Use SOLIDER pre-trained Swin weights for feature init

---

## Immediate Actionable Steps

### Step 1: Create ReID Evaluation Script
Test our current models on Market-1501 and VeRi-776 to establish baseline numbers.

### Step 2: Verify Model Quality
Our `person_resnet50ibn_market1501.pth` — what training recipe was used?
If it's a vanilla supervised model, gains from BoT tricks could be +5-8% mAP.

### Step 3: Implement BoT Training
The `reid-strong-baseline` codebase (2.3k stars) provides the exact recipe.
Port key components into our training module.

### Step 4: Zone Model for CityFlow
Define camera zones and transition time distributions for CityFlow dataset.
This is dataset-specific but critical for MTMC accuracy.

---

## Hardware Constraints (GTX 1050 Ti, 4GB VRAM)
- ResNet50-IBN-a: **fits** (inference ~1.5GB, training ~3.5GB with small batch)
- ViT-Small: **fits** for inference (~2GB), tight for training
- ViT-Base / Swin-B: **does NOT fit** for training; inference possible with fp16
- SOLIDER Swin-B: **does NOT fit** for training; can use pre-trained weights for inference only
- TransReID ViT-Base: 7GB training → **does NOT fit**

**Recommendation**: ResNet50-IBN-a with full BoT tricks is the best cost/performance tradeoff on our hardware. This achieves mAP=94.2% on Market-1501 with re-ranking, competitive with TransReID.
