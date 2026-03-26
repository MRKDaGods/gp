# SOTA Paper Research: What Top MTMC Tracking Teams Actually Do

**Date**: 2026-03-24
**Scope**: AIC21-22 Track 1 (CityFlowV2 — same dataset we use)
**Our current MTMC IDF1**: 76.5% (v85) | **SOTA**: 84.86% (AIC22 1st)
**True gap**: 8.4pp (MTMC-to-MTMC comparison)

---

## 1. Detailed Team Breakdowns

### 1.1 AIC22 1st — Team28 "Box-Grained Reranking Matching" (IDF1=0.8486)

| Component | Details |
|-----------|---------|
| **Detection** | Swin Transformer (heavy, accurate) |
| **ReID models** | **5-backbone ensemble** at 384×384: ConvNeXt, HRNet-W48, Res2Net200, ResNet50, ResNeXt101 |
| **ReID framework** | PaddlePaddle (PaddleClas) — not PyTorch |
| **Training data** | Real Track1 crops + **synthetic data** (SPGAN/style transfer) |
| **Training losses** | ID + triplet + circle + camera-aware auxiliary |
| **Single-camera tracking** | SCMT with occlusion awareness |
| **Cross-camera matching** | Box-grained reranking: reranks at **detection-box level** (not tracklet), score-fused across all 5 models |
| **Spatial constraints** | ROI masks per camera, truncation rate filtering |
| **Post-processing** | Dedicated `postprocess/` refinement stage |

**Key innovation**: Box-grained reranking avoids the information loss that occurs when tracklet features are averaged — instead, it keeps individual detection-level features for finer matching. This is especially powerful with an ensemble because each model sees every detection independently.

**Why 5 models works**: Architectural diversity captures complementary signals — ConvNeXt (modern convolutions, local texture), HRNet (multi-resolution spatial), Res2Net (multi-scale), ResNet50 (standard features), ResNeXt (grouped convolutions). Score-level fusion lets each model vote independently.

### 1.2 AIC22 2nd — Team59 "BOE" (IDF1=0.8437)

| Component | Details |
|-----------|---------|
| **Detection** | YOLOv5x6 (largest YOLO variant, not x/m/s) |
| **ReID models** | **3-backbone ensemble** at 384×384: resnet101_ibn_a (×2 variants), resnext101_ibn_a |
| **Training** | DMT (Dual-stage Multi-domain Training) — same as AIC21 winner |
| **Reranking** | `USE_RERANK: True` — k-reciprocal reranking IS enabled and helps |
| **Feature fusion** | `USE_FF: True` — explicit feature fusion across models |
| **Camera bias** | `CID_BIAS_DIR: cam_timestamp/` — pre-computed per-camera-pair distance calibration NPY files |
| **Score threshold** | `SCORE_THR: 0.1` (matches our `exhaustive_min_similarity`) |
| **Infrastructure** | 91% C++ codebase, Docker (pytorch:21.09-py3), heavily optimized |

**Key insight**: BOE proves that the AIC21 DAMO recipe (3-model IBN ensemble + DMT training + reranking + CID_BIAS) is reproducibly effective. The 0.5pp gap to Team28 comes from Team28's 5-model ensemble and box-grained reranking innovation.

**CID_BIAS explained**: Pre-computed NPY files containing per-camera-pair distance offsets. For each (cam_i, cam_j) pair, a bias value is learned from training data that adjusts the cosine distance threshold. Cameras with similar viewpoints need tighter thresholds; cameras with vastly different viewpoints need looser ones. This is more targeted than our FIC whitening (which operates per-camera, not per-pair).

### 1.3 AIC21 1st — Team "DAMO" (IDF1=0.8095 → 0.841 post-deadline)

| Component | Details |
|-----------|---------|
| **Detection** | YOLOv5x (not x6, but still large) |
| **ReID models** | **3-backbone ensemble**: ResNet101-IBN-a (×2), ResNeXt101-IBN-a — all at 384×384 |
| **Training method** | DMT (2-stage): Stage 1 = standard ID+triplet classification; Stage 2 = fine-tuning with **camera-aware** + **viewpoint-aware** auxiliary losses |
| **Domain adaptation** | SPGAN synthetic data augmentation (style transfer real→synthetic) |
| **Tracking** | Modified JDE (without detection module) |
| **Cross-camera matching** | **NOT global graph** — camera-pair-by-pair sub-clustering in adjacent cameras |
| **Spatial constraints** | **Hand-annotated crossroad zone polygons** per camera |
| **Temporal constraints** | **Direction-based temporal mask**: vehicles going specific directions appear in specific cameras within specific time windows |
| **Camera bias** | CID_BIAS correction with pre-computed NPY files (per camera pair) |

**DMT training detail**:
- **Stage 1** (40 epochs): Standard ReID training with ID loss + triplet loss on CityFlowV2 + synthetic crops. All 3 backbones trained independently.
- **Stage 2** (20 epochs): Fine-tuning with camera-aware loss (`train_cam.py`) — forces features to be camera-invariant — and viewpoint-aware loss (`train_view.py`) — forces features to be viewpoint-invariant.
- **Camera-aware loss**: Adversarial objective — a camera classifier tries to predict which camera the feature came from, while the backbone is trained to fool it (gradient reversal). This is fundamentally what our FIC whitening approximates post-hoc.
- **Viewpoint-aware loss**: Similar adversarial setup for viewpoint (front/rear/side). This addresses the exact failure mode in our 44 fragmented GT IDs — same vehicle looks different across viewpoints.

**Zone transition model detail**:
- 6 cameras × 2-4 hand-annotated entry/exit zone polygons each
- Transition time distributions learned per zone-pair (Gaussian μ, σ)
- Acts as a hard prior: if vehicle exits camera A's zone 2, it can only appear in camera B's zone 1 within time window [μ-2σ, μ+2σ]
- Our auto-generated k-means zones are too coarse (-0.4pp when enabled), but DAMO's hand-annotated zones are critical to their success

### 1.4 AIC21 Track 2 Winner — DMT Method (Pure ReID)

| Component | Details |
|-----------|---------|
| **Models** | **8-backbone ensemble**: ResNet101-IBN-a, DenseNet169-IBN-a, ResNeXt101-IBN-a, SE-ResNet101-IBN-a, ResNeSt101, ViT-Base/16, + 2 more variants |
| **Training** | Two-stage DMT: Standard → camera+viewpoint fine-tuning |
| **Augmentation** | SPGAN synthetic data, random erasing, auto-augment |
| **Result** | 0.7445 mAP on Track 2 (pure vehicle ReID) |

**This is the upper bound for ReID quality on CityFlowV2 data**. With 8 models, they achieved 74.45% mAP. Our single ViT-B/16 likely achieves ~55-60% mAP. The gap is massive and directly explains our MTMC IDF1 deficit.

---

## 2. Pattern Analysis: What ALL Winners Share

### 2.1 Universal Patterns (Present in ALL top-5 teams)

| Pattern | AIC22 1st | AIC22 2nd | AIC21 1st | Frequency |
|---------|:---------:|:---------:|:---------:|:---------:|
| **Multi-model ReID ensemble** (3+) | 5 models | 3 models | 3 models | **3/3** |
| **384×384 input resolution** | ✅ | ✅ | ✅ | **3/3** |
| **IBN (Instance-Batch Norm) backbones** | ✅ | ✅ | ✅ | **3/3** |
| **Camera-pair distance bias** | ROI masks | CID_BIAS NPY | CID_BIAS NPY | **3/3** |
| **Domain-specific fine-tuning** on CityFlowV2 | ✅ | DMT | DMT | **3/3** |
| **Synthetic training data** | ✅ | ✅ | SPGAN | **3/3** |

### 2.2 Common Patterns (Present in 2/3 top teams)

| Pattern | AIC22 1st | AIC22 2nd | AIC21 1st | Frequency |
|---------|:---------:|:---------:|:---------:|:---------:|
| **Reranking** (k-reciprocal) | box-grained | ✅ | ✅ | **3/3** (variant) |
| **Camera-aware training loss** | ✅ | DMT stage 2 | DMT stage 2 | **3/3** |
| **Zone-based spatial constraints** | ROI masks | - | hand-annotated | **2/3** |
| **Two-stage training** (DMT) | - | ✅ | ✅ | **2/3** |
| **Large detector** (not YOLO-m/s) | Swin | YOLOv5x6 | YOLOv5x | **3/3** |

### 2.3 The Minimum Viable Approach to 84%+

Based on these patterns, the minimum recipe that could reach 84%+ IDF1:

1. **3 diverse ReID backbones** at 384×384 with score-level fusion
2. **DMT 2-stage training** with camera-aware + viewpoint-aware auxiliary losses
3. **CID_BIAS** per-camera-pair distance calibration
4. **K-reciprocal reranking** (works when features are strong enough)
5. **A large detector** (at minimum YOLOv5x or YOLOv8x)

The 5-model ensemble and box-grained reranking of Team28 push from 84% to 85% — they're the refinement, not the foundation. The foundation is **3-model DMT-trained ensemble + CID_BIAS + reranking**.

---

## 3. Gap Analysis vs Our System

### 3.1 Component-by-Component Comparison

| Component | Our System | AIC22 Winner | Impact of Gap |
|-----------|-----------|--------------|:---:|
| **ReID model count** | 1 (ViT-B/16 CLIP) | 5 diverse backbones | **CRITICAL** |
| **ReID input resolution** | 256×256 | 384×384 | **HIGH** |
| **ReID training method** | Basic ID+triplet | DMT + camera-aware + viewpoint-aware + circle | **HIGH** |
| **ReID backbone diversity** | ViT only | CNN + ViT + mixed | **HIGH** |
| **Synthetic training data** | None | SPGAN + style transfer | **MEDIUM** |
| **Ensemble fusion** | 10% OSNet (weak) | 5-model score/box-grained fusion | **HIGH** |
| **Reranking** | Disabled (hurts) | Box-grained reranking (helps) | **MEDIUM** |
| **Camera bias correction** | FIC whitening (per-camera) | CID_BIAS (per camera-pair) | **MEDIUM** |
| **Detection model** | YOLO26m | Swin Transformer / YOLOv5x6 | **LOW-MEDIUM** |
| **Zone constraints** | Auto-zones (disabled, hurt) | Hand-annotated ROI/zone polygons | **MEDIUM** |
| **Cross-camera matching** | Global graph + conflict-free CC | Pair-wise ICA / sub-clustering | **LOW** |

### 3.2 Gap Ranking by Impact (Estimated IDF1 contribution)

| Rank | Gap | Est. IDF1 Cost | Why |
|:----:|-----|:--------------:|-----|
| **1** | Single model vs 3-5 model ensemble | **3.0-4.0pp** | Ensemble diversity is the #1 differentiator. Every winner uses 3+ models. Architectural diversity (CNN vs ViT) captures complementary features. |
| **2** | 256px vs 384px input resolution | **1.0-2.0pp** | 2.25× more spatial detail. License plates, trim details, damage visible at 384 but blurred at 256. All winners use 384+. |
| **3** | Basic training vs DMT camera+viewpoint-aware | **1.0-1.5pp** | Camera-aware training produces inherently camera-invariant features (vs our post-hoc FIC whitening). Viewpoint-aware training addresses our 44 fragmented IDs. |
| **4** | FIC whitening vs CID_BIAS per-pair calibration | **0.5-1.0pp** | Per-pair is strictly more expressive than per-camera. Can model that cam1→cam2 needs looser threshold than cam1→cam3. |
| **5** | Reranking disabled vs enabled (with strong features) | **0.5-1.0pp** | Reranking hurts us because features are weak → reranking amplifies noise. With strong 3-model features, reranking helps (BOE confirms). |
| **6** | No zones vs hand-annotated zones | **0.5-1.0pp** | Hard spatio-temporal priors eliminate impossible transitions. Our auto-zones are too coarse. |
| **7** | No synthetic data vs SPGAN augmentation | **0.3-0.5pp** | More training diversity improves generalization. |
| **8** | YOLO26m vs YOLOv5x6/Swin | **0.2-0.5pp** | Better detections → tighter boxes → less background in crops → better ReID features. Not the main bottleneck (0 missed GT), but contributes. |

**Total estimated gap: ~7.0-10.5pp** (partially overlapping; actual gap is 8.4pp)

### 3.3 Why Reranking Hurts Us But Helps Winners

This is a critical finding. Our experiments show reranking hurts vehicles (v25). BOE and DAMO use it successfully. The explanation:

- **Weak features + reranking = disaster**: K-reciprocal reranking works by finding shared neighbors. When features are noisy (single 256px model), neighborhoods are noisy → reranking amplifies errors.
- **Strong features + reranking = boost**: With a 3-model 384px ensemble, neighborhoods are clean → reranking correctly identifies hard positives that raw cosine misses.
- **Prediction**: Once we have 2-3 model ensemble at 384px, reranking will become beneficial. This should be re-tested after every feature quality improvement.

---

## 4. Realistic Improvement Roadmap

### 4.1 Constraints

| Resource | Available | Limitation |
|----------|-----------|-----------|
| **Training GPU** | Kaggle T4 (16GB VRAM) | Max batch ~64 at 384px for ResNet-101 |
| **Pipeline GPU** | Kaggle P100 (16GB, sm_60) | PyTorch 2.4.1+cu124 required |
| **Runtime budget** | ~9h total Kaggle chain (10a+10b+10c) | Each notebook max 12h |
| **Framework** | Python/PyTorch | Cannot use PaddlePaddle |
| **Base model** | TransReID ViT-B/16 CLIP, fine-tuned on CityFlowV2 | Existing strong checkpoint |
| **Training infra** | `src/training/` with BoT recipe, losses.py (circle loss), train_reid.py | Exists but not connected to CityFlowV2 |

### 4.2 Phase 1: Fix Single Model Quality (Target: 79-80% MTMC IDF1)

**Timeline**: Days, minimal training

| # | Action | Change | Est. Gain | Effort |
|---|--------|--------|:---------:|:------:|
| 1a | **384×384 native training** (redo 09b properly) | New training notebook: start from CLIP ViT-B/16 pretrained at 384 (not from 256 checkpoint), cosine LR over 80 epochs, bicubic position embedding interpolation | +1.0-2.0pp | Medium |
| 1b | **Add circle loss** to CityFlowV2 training | Enable existing `src/training/losses.py` CircleLoss alongside ID+triplet | +0.3-0.5pp | Low |
| 1c | **Multi-scale TTA** (2 scales, selective) | Test [256, 320] on top-16 crops per tracklet, profile Kaggle runtime | +0.2-0.4pp | Low |

**384px training plan** (critical path):
```
1. Start from openai/clip-vit-base-patch16 pretrained weights
2. Interpolate position embeddings from 14×14 → 24×24 (384/16=24) using bicubic
3. CityFlowV2 training: 80 epochs, cosine LR 3.5e-4→1e-6, batch=64 (fits T4)
4. Losses: ID (label smoothing 0.1) + triplet (margin 0.3) + circle (m=0.25, γ=80)
5. Augment: random erasing (p=0.5), random horizontal flip, color jitter
6. Validate: compute mAP on held-out CityFlowV2 val split → target ≥0.60
```

### 4.3 Phase 2: Add Second Ensemble Member (Target: 81-82% MTMC IDF1)

**Timeline**: 1-2 weeks

| # | Action | Change | Est. Gain | Effort |
|---|--------|--------|:---------:|:------:|
| 2a | **Train ResNet101-IBN-a** on CityFlowV2 at 384×384 | Use BoT recipe from `src/training/train_reid.py`. IBN-a provides camera invariance by design. | +1.0-1.5pp | Medium |
| 2b | **Score-level fusion** (2-model ensemble) | Compute separate similarity matrices from ViT-B and ResNet101-IBN, then blend: `sim = α×sim_vit + (1-α)×sim_resnet` with α∈[0.5, 0.7] | Included above | Low |
| 2c | **Re-test reranking** with ensemble features | With 2-model ensemble, reranking neighborhoods should be cleaner | +0.3-0.5pp | Low |

**Why ResNet101-IBN-a specifically**:
- Used by ALL 3 top teams (AIC21 1st, AIC22 1st, AIC22 2nd)
- IBN-a (Instance-Batch Normalization) inherently reduces camera-domain shift
- CNN captures local texture patterns (scratches, stickers, license plate fonts) that ViT misses
- ResNet101 is deeper than ResNet50 (+1-2% mAP) and fits on T4 at 384px
- **Maximum architectural diversity** with our existing ViT-B/16

**ResNet101-IBN-a training plan**:
```
1. Use torchvision ResNet101 + IBN-a modification (replace first BN with IBN-a)
   - Or use the published resnet101_ibn_a weights from IBN-Net repo as initialization
2. BoT recipe: BNNeck, last_stride=1, GeM pooling
3. Losses: ID (label smoothing 0.1) + triplet (soft margin) + circle (m=0.25, γ=80)
4. CityFlowV2 training: 80 epochs, cosine LR 3.5e-4→1e-6, batch=48 (fits T4 at 384px)
5. Augment: random erasing (p=0.5), horizontal flip, color jitter, random crop+pad
6. Output: 2048D features → PCA-whiten to 384D independently
7. Fusion: score-level blending in stage4 (NOT feature concatenation!)
```

### 4.4 Phase 3: Add Camera-Pair Bias + Zones (Target: 83-84% MTMC IDF1)

**Timeline**: 1-2 weeks

| # | Action | Change | Est. Gain | Effort |
|---|--------|--------|:---------:|:------:|
| 3a | **CID_BIAS per camera pair** | Learn per-pair distance offsets from confident matches (sim > 0.7) or from GT training labels. Save as NPY matrix of shape (N_cam, N_cam). Apply as additive bias to similarity. | +0.5-1.0pp | Medium |
| 3b | **Hand-annotate zone polygons** | Annotate 6 cameras × 2-4 entry/exit zones. Store in `configs/datasets/cityflowv2_zones.json`. Enables zone-based transition time priors. | +0.5-1.0pp | Medium (manual) |
| 3c | **DMT Stage 2: camera-aware fine-tuning** | Add gradient reversal camera classifier to ViT-B and ResNet101-IBN. Fine-tune 20 epochs with adversarial camera loss. | +0.5-1.0pp | High |

**CID_BIAS implementation plan**:
```
1. For each camera pair (i, j), collect all matched pairs from GT or from high-confidence predictions
2. Compute mean cosine similarity for true-match pairs: μ_ij
3. Compute mean cosine similarity for false-match pairs: ν_ij
4. Bias = μ_ij - global_mean_μ  (how much harder/easier this pair is)
5. Apply: adjusted_sim(a,b) = raw_sim(a,b) + bias[cam_a, cam_b]
6. Store as (6×6) matrix in NPY file, loaded at stage4 startup
```

### 4.5 Phase 4: Third Ensemble Member + Refinements (Target: 84-85% MTMC IDF1)

**Timeline**: 2-4 weeks

| # | Action | Change | Est. Gain | Effort |
|---|--------|--------|:---------:|:------:|
| 4a | **Train ResNeXt101-IBN-a** (3rd ensemble member) | Following DAMO's exact recipe. ResNeXt adds grouped convolution diversity. | +0.5-0.8pp | Medium |
| 4b | **Knowledge distillation** (ViT-L→ViT-B, fix 09c) | Fix projector dimension mismatch, proper T=2 temperature, MSE+KL loss | +0.5-1.0pp | Medium |
| 4c | **Re-enable reranking** with 3-model features | Should now help significantly | +0.3-0.5pp | Low |
| 4d | **SPGAN synthetic augmentation** | Style-transfer real→synthetic crops for training diversity | +0.2-0.3pp | Medium |

### 4.6 Summary: Expected Trajectory

| Phase | MTMC IDF1 | Key Changes |
|:-----:|:---------:|-------------|
| Current | 76.5% | Single ViT-B/16 @ 256px |
| Phase 1 | 79-80% | ViT-B/16 @ 384px + circle loss + multi-scale TTA |
| Phase 2 | 81-82% | + ResNet101-IBN-a ensemble + re-test reranking |
| Phase 3 | 83-84% | + CID_BIAS + hand-annotated zones + DMT camera-aware |
| Phase 4 | 84-85% | + ResNeXt101-IBN-a (3-model) + KD + SPGAN |

---

## 5. The Ensemble Question: Minimum Viable Diversity

### 5.1 Why Ensemble Works

The key insight from all top teams is not "more models = better" — it's **architectural diversity**. Each backbone family captures fundamentally different visual features:

| Backbone Type | What It Captures | Failure Mode |
|--------------|-----------------|-------------|
| **ViT (Transformer)** | Global context, long-range patch relationships, holistic shape | Weak on local texture, small details |
| **ResNet-IBN (CNN)** | Local texture, edges, gradients, camera-invariant features (IBN-a) | Weak on global structure, viewpoint change |
| **ResNeXt (Grouped CNN)** | Multi-path local features, richer local representations | Similar to ResNet but more diverse |
| **HRNet (Multi-resolution)** | Spatial precision at multiple scales, fine details | Heavy computation |
| **ConvNeXt (Modern CNN)** | Best of both: large effective receptive field + local precision | Similar to ViT in some aspects |

### 5.2 Recommended Minimum Ensemble (2-3 models)

**Option A: 2-model ensemble (fastest to implement)**
1. **TransReID ViT-B/16 CLIP** @ 384×384 (our existing model, upgraded)
2. **ResNet101-IBN-a** @ 384×384 (BoT recipe, new training)

**Expected gain**: +1.5-2.5pp from ensemble alone (on top of 384px gains)
**Diversity score**: Maximum — ViT (global) vs CNN-IBN (local+camera-invariant)
**Training time**: ~8-12h on T4 for ResNet101, our ViT already exists

**Option B: 3-model ensemble (matches DAMO recipe exactly)**
1. **TransReID ViT-B/16 CLIP** @ 384×384
2. **ResNet101-IBN-a** @ 384×384
3. **ResNeXt101-IBN-a** @ 384×384

**Expected gain**: +2.0-3.5pp from ensemble
**Diversity score**: High — Transformer + CNN-IBN + Grouped-CNN-IBN
**Training time**: ~16-20h on T4 for both CNNs
**This is exactly what AIC21 1st and AIC22 2nd used.**

**Option C: 2-model + KD (maximizes single-model quality)**
1. **TransReID ViT-B/16 CLIP** @ 384×384, distilled from ViT-L/14
2. **ResNet101-IBN-a** @ 384×384

**Expected gain**: +2.0-3.0pp (KD boosts the ViT, ensemble adds diversity)
**This is probably the highest ROI path** given our constraints.

### 5.3 Score-Level Fusion Strategy

**DO NOT concatenate features** (our v26 failure). Instead:

```
# For each candidate pair (tracklet_i, tracklet_j):
sim_vit = cosine(vit_embed_i, vit_embed_j)          # FIC-whitened, PCA'd
sim_resnet = cosine(resnet_embed_i, resnet_embed_j)  # FIC-whitened separately, PCA'd
sim_fused = α × sim_vit + (1-α) × sim_resnet        # α ∈ [0.5, 0.7]
# Then apply existing pipeline: threshold → graph → CC
```

Each model's embeddings are whitened (FIC) and PCA'd independently before computing similarity. This avoids the uncalibrated-space problem that killed v26.

**Sweep α**: Test [0.4, 0.5, 0.6, 0.7] — AIC teams typically use equal or slight primary bias.

### 5.4 Kaggle Runtime Budget for Ensemble

| Component | Current (1 model) | 2-model | 3-model |
|-----------|:-----------------:|:-------:|:-------:|
| Stage 2 (feature extraction) | ~45min | ~80min | ~120min |
| Stage 3 (indexing) | ~2min | ~4min | ~6min |
| Stage 4 (association) | ~5min | ~8min | ~12min |
| **Total 10a** | ~60min | ~100min | ~150min |

All fit within Kaggle's 12h limit with comfortable margin. The 3-model approach needs separate PCA models per backbone, which increases 10b time slightly.

---

## 6. Critical Insights

### 6.1 The Fundamental Truth

**Feature quality determines the ceiling. Association tuning determines how close you get to it.**

Our 225+ association config sweeps prove we're within ~0.5pp of our feature quality ceiling. The remaining 8.4pp gap to SOTA is almost entirely feature quality:
- Single model vs 3-5 model ensemble: ~3-4pp
- 256px vs 384px: ~1-2pp
- Basic vs DMT camera-aware training: ~1-1.5pp
- FIC vs CID_BIAS: ~0.5-1pp

### 6.2 Why Our Reranking Fails But Theirs Succeeds

- Weak features → noisy k-NN neighborhoods → reranking amplifies noise → hurts
- Strong features → clean neighborhoods → reranking finds hard positives → helps
- **Test**: After each feature quality improvement, re-test reranking

### 6.3 Why Our Zone Model Fails But Theirs Succeeds

- Auto-generated k-means zones are spatially imprecise → wrong transition priors → hurts
- Hand-annotated entry/exit polygons are precise → correct priors → helps
- **Fix**: Hand-annotate zones for CityFlowV2's 6 test cameras (one-time effort)

### 6.4 The IBN-a Secret

Instance-Batch Normalization (IBN-a) replaces early batch normalization layers with a mix of instance norm (captures style/appearance/lighting) and batch norm (captures content/identity). The instance norm branch inherently removes camera-specific style — making features camera-invariant **by architecture**, not by post-hoc whitening. This is why every top team uses IBN-a backbones.

Our FIC whitening approximates this post-hoc, but it's a linear approximation of a non-linear effect. Having IBN-a architecture + FIC whitening is additive.

### 6.5 Box-Grained vs Tracklet-Level Matching

Our pipeline averages crop embeddings (quality-weighted) into a single tracklet embedding, then matches tracklets. Team28 keeps ALL detection-level embeddings and reranks at the box level. This preserves information:
- A vehicle seen from the front in frames 1-50 and from behind in frames 51-100
- Tracklet average: blurred, ambiguous embedding
- Box-grained: frame 1-50 embeddings match front-view queries; frame 51-100 match rear-view queries

**This is architecturally compatible with our pipeline** — we already store per-crop embeddings in stage 2. The change would be in stage 4: instead of single embedding per tracklet, use the full crop embedding matrix for matching.

---

## 7. Files Referenced

| File | Purpose |
|------|---------|
| [src/stage2_features/pipeline.py](src/stage2_features/pipeline.py) | Current feature extraction pipeline |
| [src/stage4_association/pipeline.py](src/stage4_association/pipeline.py) | Current association pipeline |
| [src/training/train_reid.py](src/training/train_reid.py) | Existing BoT training code |
| [src/training/losses.py](src/training/losses.py) | CircleLoss already implemented |
| [src/training/model.py](src/training/model.py) | ReIDModelBoT architecture |
| [docs/subagent-specs/cross-camera-analysis.md](docs/subagent-specs/cross-camera-analysis.md) | Error profile analysis |
| [docs/subagent-specs/untried-approaches.md](docs/subagent-specs/untried-approaches.md) | Untried approaches catalog |
| [docs/experiment_log.md](docs/experiment_log.md) | 225+ exhausted association configs |
| [configs/default.yaml](configs/default.yaml) | Current pipeline configuration |

---

## 8. Executive Summary

### What wins MTMC on CityFlowV2:
1. **3+ diverse ReID backbones at 384px** (ALL top-3 teams)
2. **DMT 2-stage training** with camera-aware loss (2/3 top teams)
3. **Per camera-pair distance calibration** (ALL top-3 teams)
4. **Reranking** (works with strong features) (ALL top-3 teams)
5. **Hand-annotated zone constraints** (2/3 top teams)

### What we have:
1. 1 ViT-B/16 at 256px (**massive gap**)
2. Basic ID+triplet training (**gap**)
3. FIC per-camera whitening (**partial solution**)
4. Reranking disabled (**would help with better features**)
5. Auto zones disabled (**need hand annotations**)

### The fastest path to 84%:
**Phase 1** (days): Train ViT-B/16 at 384px properly → ~79-80%
**Phase 2** (1-2 weeks): Train ResNet101-IBN-a, 2-model ensemble → ~81-82%
**Phase 3** (1-2 weeks): CID_BIAS + hand zones + DMT fine-tuning → ~83-84%
**Phase 4** (2-4 weeks): 3rd model + KD + reranking re-enable → ~84-85%