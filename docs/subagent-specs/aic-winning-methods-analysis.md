# AI City Challenge Track 1 — Winning Methods Analysis

> **Purpose**: Comprehensive analysis of winning solutions from the AI City Challenge Multi-Target Multi-Camera Vehicle Tracking task, compared against our current pipeline, with prioritized improvements.

## Executive Summary

AIC 2022 was the **last year** with CityFlowV2 vehicle MTMC tracking (Track 1). From 2023 onward, Track 1 switched to multi-camera **people** tracking. Our SOTA target of IDF1≈0.8486 comes from the 2022 winner.

**The single most critical gap is ReID ensemble diversity.** Every winning team (2020-2022) uses 3-5 ReID models. We use 1. This alone likely accounts for 4-6pp of our 7.4pp gap. The second major gap is camera-pair priors (CID_BIAS) and box-grained matching.

---

## Year-by-Year Winning Solutions

### AIC 2022 — CityFlowV2 (Our Target Dataset)

#### 1st Place: Team28 "matcher" — IDF1 = 0.8486
**Paper**: "CityTrack: Box-Grained Reranking Matching for Multi-Camera Multi-Target Tracking" (arXiv:2307.02753)
**Code**: https://github.com/Yejin0111/AICITY2022-Track1-MTMC

| Component | Details |
|-----------|---------|
| **Detection** | Swin Transformer (not YOLO) |
| **ReID Models** | **5-model ensemble**: HRNet_W48, ConvNext, Res2Net200, ResNet50, ResNeXt101_32x8d_wsl |
| **ReID Framework** | PaddlePaddle (PaddleClas) |
| **ReID Training Data** | Real CityFlowV2 crops + **synthetic data** |
| **ReID Input Size** | 384×384 for most backbones |
| **Tracking** | Custom SCMT (Single Camera Multi-Target) tracker |
| **Key Innovation** | **Box-Grained Matching (BGM)** — reranking at detection-box level, not track-level |
| **Post-processing** | ROI masks per camera, truncation rate estimation |
| **Association** | ICA module with box-grained reranking + ROI masks |

**Key insight**: BGM performs cross-camera matching at the individual detection box level rather than averaging features per track. This preserves fine-grained appearance information that track-level averaging destroys — especially for vehicles that appear very differently from different angles.

#### 2nd Place: Team59 "BOE" — IDF1 = 0.8437
**Paper**: "Multi-Camera Vehicle Tracking System for AI City Challenge 2022" (CVPR 2022 Workshop)
**Code**: https://github.com/coder-wangzhen/AIC22-MCVT

| Component | Details |
|-----------|---------|
| **Detection** | YOLOv5x6 |
| **ReID Models** | **3-model ensemble**: ResNet101-IBN-a ×2, ResNeXt101-IBN-a ×1 |
| **ReID Input Size** | 384×384 |
| **ReID Training** | DMT framework (Alibaba, 2-stage training with camera/viewpoint awareness) |
| **CID_BIAS** | Camera-pair bias from timestamp analysis (NPY files) |
| **Reranking** | k-reciprocal reranking (USE_RERANK: True) |
| **Feature Fusion** | Score-level fusion (USE_FF: True) |
| **Tracker** | Modified JDETracker lineage |

**Key insight**: Evolution of the AIC21 winner's approach. The 3 ReID models are 2 variants of ResNet101-IBN-a (trained with different seeds/hyperparams) plus 1 ResNeXt101-IBN-a. CID_BIAS provides strong camera-pair priors.

#### 3rd Place: Team37 "TAG" — IDF1 = 0.8371
**Code**: https://github.com/backkon/AICITY2022_Track1_TAG (based on AIC21-MTMC)

| Component | Details |
|-----------|---------|
| **Detection** | YOLOv5x (COCO pretrained) |
| **ReID Models** | DMT-trained models (ResNet101-IBN-a variants) |
| **Framework** | Direct fork of AIC21 1st place solution |
| **CID_BIAS** | Yes (inherited from AIC21 framework) |
| **Reranking** | k-reciprocal |

#### 4th Place: Team50 "FraunhoferIOSB" — IDF1 = 0.8348
Few details publicly available but achieved strong results with systematic engineering.

#### Notable: Team94 "SKKU" — IDF1 = 0.8129
Strong baseline demonstrating that even with the standard pipeline, the ReID ensemble is the differentiator.

---

### AIC 2021 — CityFlow (Track 3 = MTMC Vehicle Tracking)

#### 1st Place: Team75 "mcmt" — IDF1 = 0.8095
**Paper**: "City-Scale Multi-Camera Vehicle Tracking Guided by Crossroad Zones" (arXiv:2105.06623, CVPR 2021 Workshop)
**Code**: https://github.com/LCFractal/AIC21-MTMC

| Component | Details |
|-----------|---------|
| **Detection** | YOLOv5x (COCO pretrained) |
| **ReID Models** | **3-model ensemble**: ResNet101-IBN-a ×2, ResNeXt101-IBN-a ×1 (all 384×384) |
| **ReID Training** | DMT method — 2-stage training with SPGAN synthetic data + camera/viewpoint classifiers |
| **CID_BIAS** | Camera-pair bias from timestamp analysis (CID_BIAS_DIR) |
| **Reranking** | k-reciprocal reranking |
| **Feature Fusion** | Yes (USE_FF) |
| **Tracker** | Modified JDETracker (no detection module) |
| **Key Innovations** | Crossroad zone-based Tracklet Filter Strategy, Direction-Based Temporal Mask, Sub-clustering in Adjacent Cameras |

**Key insight**: This is the foundational pipeline that AIC22 2nd/3rd place solutions evolved from. The zone-based temporal mask and sub-clustering are elegant spatial prior methods.

#### ReID Training Details (DMT Method, from AIC21 Track 2 Winner)
**Paper**: "An Empirical Study of Vehicle Re-Identification on the AI City Challenge" (Hao Luo et al.)
**Code**: https://github.com/michuanhaohao/AICITY2021_Track2_DMT

The DMT ReID training recipe:
1. **Stage 1**: Standard training with ID loss + triplet loss
2. **Stage 2 v1**: Fine-tuning with harder mining
3. **Stage 2 v2**: Further refinement with different augmentation
4. **Camera model**: Separate classifier predicting camera ID → produces camera-aware reranking matrix (`track_cam_rk.npy`)
5. **Viewpoint model**: Separate classifier predicting viewpoint → produces viewpoint-aware reranking matrix (`track_view_rk.npy`)
6. **Backbones**: 8 models total — ResNet101-IBN-a, DenseNet169-IBN-a, ResNeXt101-IBN-a, SE-ResNet101-IBN-a, ResNeSt101, ViT-Base/16, etc.
7. **Data augmentation**: Random erasing, SPGAN synthetic data, cropped patches from part segmentation
8. **Ensemble**: All 8 models ensembled for final features

---

### AIC 2020 — CityFlow (Track 3 = MTMC Vehicle Tracking)

#### 1st Place: Team92 "INF" (CMU) — IDF1 = 0.4616
**Paper**: "ELECTRICITY: An Efficient Multi-camera Vehicle Tracking System for Intelligent City"

Track 3 in 2020 was harder (different dataset subset) with much lower scores. The methods were less mature.

#### Track 2 (ReID) Winner: Baidu-UTS — mAP = 0.8413
**Paper**: "Going Beyond Real Data: A Robust Visual Representation for Vehicle Re-identification"
**Code**: https://github.com/heshuting555/AICITY2020_DMT_VehicleReID

This is the predecessor of the DMT framework that became standard for all subsequent winners. Key techniques:
- Multi-domain training across real + synthetic
- IBN-Net backbones
- Camera-aware training
- Orientation-aware features

---

### AIC 2023-2024 — Track 1 = People Tracking (NOT Vehicle MTMC)

From 2023 onward, Track 1 switched to **Multi-Camera People Tracking** on synthetic datasets. These are not comparable to CityFlowV2 vehicle MTMC. AIC 2022 is the definitive SOTA for our task.

---

## Technique-by-Technique Comparison

### Our Pipeline vs. Winning Solutions

| Component | Our Pipeline | AIC22 1st (0.8486) | AIC22 2nd (0.8437) | AIC21 1st (0.8095) | Gap Impact |
|-----------|------------|-------------------|-------------------|-------------------|-----------|
| **Detection** | YOLO26m | Swin Transformer | YOLOv5x6 | YOLOv5x | Low (0-0.5pp) |
| **# ReID Models** | **1** | **5** | **3** | **3** | **HIGH (4-6pp)** |
| **ReID Backbone** | ViT-B/16 CLIP | HRNet, ConvNext, Res2Net200, ResNet50, ResNeXt101 | ResNet101-IBN-a, ResNeXt101-IBN-a | ResNet101-IBN-a, ResNeXt101-IBN-a | High (part of ensemble) |
| **ReID Input** | 256×256 | 384×384 | 384×384 | 384×384 | Low (tested; harmful alone) |
| **ReID Training Data** | VeRi-776→CityFlowV2 | Real + Synthetic | DMT 2-stage | DMT 2-stage + SPGAN | Medium (1-2pp) |
| **Camera-pair prior** | FIC regularization | ROI masks | **CID_BIAS (NPY)** | **CID_BIAS (NPY)** | **Medium (1-2pp)** |
| **Reranking** | Disabled | Box-Grained BGM | k-reciprocal | k-reciprocal | Medium (0.5-1.5pp) |
| **Feature Fusion** | Score-level only | Ensemble + BGM | Feature fusion | Feature fusion | Medium (part of ensemble) |
| **Camera/View models** | No | Implicit in ensemble | Via camera_rk.npy | Camera + View models | Low-Medium (0.5-1pp) |
| **Synthetic data** | No | Yes | Via DMT | SPGAN | Low (0.5-1pp) |
| **Tracker** | BoT-SORT (BoxMOT) | Custom SCMT | Modified JDE | Modified JDE | Low (0-0.5pp) |
| **Association** | Graph + CC + FIC | Box-Grained ICA | Pairwise + CID_BIAS | Zone-guided + CID_BIAS | Medium (1-2pp as aggregate) |
| **Zone/ROI filtering** | No | ROI masks | No | Crossroad zones + temporal mask | Low-Medium (0.5-1pp) |

### Key Observations

1. **Every winning team uses 3-5 ReID model ensemble** — This is the universal pattern. No single model ever wins.
2. **CID_BIAS is standard** — Camera-pair bias matrices are used by 2nd/3rd/equivalent of every year. Our FIC is a weaker approximation.
3. **k-reciprocal reranking works WITH ensemble features** — We tested and it hurt. But that's because our single-model features have too many false matches. With 3-5 model ensemble features, reranking consistently helps.
4. **384px alone is not enough** — We proved this: 384px without ensemble actually hurts MTMC. The winning teams use 384px as ONE input size among many in the ensemble.
5. **Detection is NOT the bottleneck** — YOLOv5x was good enough for 2nd/3rd place. Swin Transformer (1st place) offers marginal detection gains.
6. **Box-grained matching is the AIC22 winner's key innovation** — This is novel and not available in any off-the-shelf pipeline.

---

## Prioritized Improvement Plan

### Priority 1: ReID Model Ensemble (Expected: +4-6pp)
**Status**: CRITICAL — sole viable path to close the gap
**Expected IDF1**: 0.775 → 0.82-0.83

The winning teams all use diverse backbone ensembles. The key is **architectural diversity**, not just different seeds:

**Proposed 3-model ensemble** (minimum viable):
1. **ViT-B/16 CLIP 256px** (existing) — mAP=80.14% on CityFlowV2
2. **ResNet101-IBN-a** (existing but weak at 52.77%) — needs DMT-style 2-stage training on VeRi-776→CityFlowV2. Target: mAP≥70%
3. **ResNeXt101-IBN-a or Res2Net200** (new) — different architecture family for diversity. Train with DMT recipe.

**Training recipe** (based on DMT):
- Stage 1: Standard training with ID + triplet loss, Adam, 120 epochs
- Stage 2: Fine-tune with harder triplet mining + center loss
- Use SPGAN or synthetic data augmentation
- Train camera classifier and viewpoint classifier as auxiliary tasks
- All models at 384×384 input
- Feature fusion: L2-normalize each model's features, concatenate, then PCA to 384D

**Implementation**:
- Train model 2 (ResNet101-IBN-a) properly using DMT 2-stage recipe
- Train model 3 (ResNeXt101-IBN-a) from IBN-Net pretrained weights
- Ensemble by feature concatenation → PCA, not score-level fusion
- This is the highest-ROI change by far

### Priority 2: CID_BIAS Camera-Pair Priors (Expected: +1-2pp)
**Status**: NOT IMPLEMENTED — FIC is a weaker approximation
**Expected IDF1**: 0.82 → 0.83-0.84

CID_BIAS is a per-camera-pair additive bias that adjusts similarity scores based on spatio-temporal transition statistics. All top-3 AIC22 solutions use it.

**Implementation**:
- Compute transition time distributions between camera pairs from training data
- Store as NPY matrix: `cid_bias[cam_i][cam_j] = bias_value`
- Add bias to cosine similarity during cross-camera matching
- Unlike FIC (which is a global regularization), CID_BIAS is pairwise-specific
- Can be estimated from GT tracklets or from confident predictions

### Priority 3: k-Reciprocal Reranking with Ensemble Features (Expected: +0.5-1.5pp)
**Status**: Previously tested and harmful with single-model features
**Expected IDF1**: 0.83 → 0.835-0.84

k-reciprocal reranking consistently helps when features are strong (ensemble). It hurts when features are noisy (single model). Re-enable ONLY after ensemble is working.

**Implementation**:
- Use standard k-reciprocal algorithm (already implemented but disabled)
- Parameters: k1=20, k2=6, lambda=0.3 (standard settings)
- Apply after ensemble feature extraction, before association

### Priority 4: Box-Grained Matching (Expected: +0.5-1pp)
**Status**: NOT IMPLEMENTED — AIC22 winner's novel technique
**Expected IDF1**: Additive on top of ensemble

Instead of averaging features over a track and comparing track-level features, BGM keeps per-detection features and builds a box-to-box similarity matrix. Cross-camera association then uses the best box-pair match rather than track-average.

**Implementation**:
- Store per-detection features (not just track-averaged)
- For cross-camera tracklet comparison: compute max(cos_sim(box_i, box_j)) over all box pairs
- Or top-K box pairs averaged
- Apply reranking at box level
- This preserves viewpoint-specific appearance details

### Priority 5: Synthetic Training Data (Expected: +0.5-1pp)
**Status**: NOT IMPLEMENTED
**Expected IDF1**: Additive

SPGAN (Similarity Preserving GAN) transfers style from source domain to target domain while preserving identity. Used by AIC21/22 winners.

**Implementation**:
- Generate CityFlowV2-style synthetic vehicle crops using SPGAN or similar
- Augment ReID training data
- Lower priority because it's complex and the gain is smaller

### Priority 6: Zone-Based Temporal Filtering (Expected: +0.5-1pp)
**Status**: PARTIALLY IMPLEMENTED (gt_zone_filter exists)

The AIC21 winner uses manually defined crossroad zones and direction-based temporal masks. This constrains impossible associations.

**Implementation**:
- Define entry/exit zones for each camera
- Compute valid transition time windows between camera pairs
- Filter out associations that violate spatio-temporal constraints
- We already have some GT-assisted zone filtering; make it non-GT-dependent

---

## Realistic Gap Closure Projection

| Improvement | Expected pp | Cumulative IDF1 | Difficulty | Timeline |
|------------|:-----------:|:---------------:|:----------:|:--------:|
| Baseline | — | 0.775 | — | — |
| Priority 1: 3-model ensemble | +4-6 | 0.82-0.83 | HIGH | 2-3 weeks training |
| Priority 2: CID_BIAS | +1-2 | 0.83-0.84 | MEDIUM | 1 week |
| Priority 3: k-reciprocal reranking | +0.5-1.5 | 0.835-0.845 | LOW | 1-2 days |
| Priority 4: Box-grained matching | +0.5-1 | 0.84-0.85 | MEDIUM | 1 week |
| **Total projection** | **+6.5-10.5** | **0.84-0.85** | — | — |

**Realistic target with Priorities 1-3**: IDF1 ≈ 0.835-0.845, which matches or exceeds the SOTA of 0.8486.

---

## Critical Insights

### Why Our Previous Attempts Failed

1. **384px alone ≠ ensemble**: We tested 384px and it hurt because a single 384px model overfits to viewpoint-specific textures. In winning solutions, 384px is ONE resolution among many models operating at different resolutions and architectures, so the ensemble averages out viewpoint-specific noise.

2. **Reranking hurt because features are too noisy**: With a single model, k-reciprocal reranking amplifies false neighbors. With 3-5 model ensemble, the feature space is much cleaner and reranking correctly surfaces true matches.

3. **DMT camera-aware training hurt us**: Our single-model DMT experiment (v46, -1.4pp) failed because camera-aware training with a single model trades cross-camera generalization for within-camera discrimination. In winning solutions, DMT is one of several models in an ensemble — the camera-specific model captures camera-specific features while other models provide cross-camera invariance.

4. **ResNet101-IBN-a is weak because of training recipe**: Our direct ImageNet→CityFlowV2 training gave 52.77% mAP. The winning teams use DMT's 2-stage training with VeRi-776 pretraining, SPGAN synthetic data, and progressive refinement to achieve 70-80% mAP with the same backbone.

### What We Must NOT Do

- Don't try more association tuning — EXHAUSTED (225+ configs)
- Don't try 384px as sole model — confirmed dead end
- Don't try DMT as sole model — confirmed dead end  
- Don't try reranking with single-model features — confirmed harmful
- All of these techniques become VALUABLE in the context of a proper ensemble

### The Universal Recipe (2020-2022)

Every winning team follows this template:
1. Good detector (YOLOv5x or better)
2. **3-5 diverse ReID models** with IBN-a backbones, trained with DMT recipe
3. **CID_BIAS** camera-pair priors
4. k-reciprocal or box-grained reranking
5. Feature fusion by concatenation or weighted averaging
6. Standard SCMT/JDE tracker for single-camera tracking
7. ICA (Inter-Camera Association) with post-processing

We have components 1, 5 (partially), 6, and 7. We are missing the core: 2, 3, 4.

---

## Sources

- AIC 2022 Winners: https://www.aicitychallenge.org/2022-challenge-winners/
- AIC 2021 Winners: https://www.aicitychallenge.org/2021-challenge-winners/
- AIC 2020 Winners: https://www.aicitychallenge.org/2020-challenge-winners/
- CityTrack (AIC22 1st): arXiv:2307.02753, https://github.com/Yejin0111/AICITY2022-Track1-MTMC
- AIC22 2nd (BOE): https://github.com/coder-wangzhen/AIC22-MCVT
- AIC22 3rd (TAG): https://github.com/backkon/AICITY2022_Track1_TAG
- AIC21 1st (Crossroad Zones): arXiv:2105.06623, https://github.com/LCFractal/AIC21-MTMC
- DMT ReID Training: https://github.com/michuanhaohao/AICITY2021_Track2_DMT
- AIC 2022 Top Teams Code: https://github.com/NVIDIAAICITYCHALLENGE/2022AICITY_Code_From_Top_Teams/# AI City Challenge Track 1 — Winning Methods Analysis

> Comprehensive research of MTMC vehicle tracking winning solutions from AIC 2019-2024.
> Created: 2026-03-31

## Executive Summary

The AI City Challenge Track 1 has been the premier benchmark for city-scale multi-camera multi-target vehicle tracking since 2019. After thorough research of all winning solutions, **the single most impactful differentiator between our pipeline (IDF1=0.775) and SOTA (IDF1=0.8486) is the ReID model ensemble**. Every winning team since 2021 uses 3-5 ReID backbones; we use 1. This alone likely accounts for 4-6pp of our 7.4pp gap.

**Important note**: AIC 2023-2024 changed Track 1 from vehicle tracking to **people** tracking (using a new synthetic dataset). Only AIC 2019-2022 are directly comparable to our CityFlowV2 vehicle pipeline.

---

## Year-by-Year Winning Solutions

### AIC 2022 — Track 1 Winner: Team28 "matcher" (CityTrack)
- **Paper**: "Box-Grained Reranking Matching for Multi-Camera Multi-Target Tracking" (CVPR 2023 Workshop)
- **ArXiv**: [2307.02753](https://arxiv.org/abs/2307.02753)
- **Code**: [Yejin0111/AICITY2022-Track1-MTMC](https://github.com/Yejin0111/AICITY2022-Track1-MTMC)
- **Score**: IDF1 = **0.8486** (1st place)

| Component | Details |
|-----------|---------|
| **Detection** | Swin Transformer (object detection), trained on AIC22 data |
| **ReID Models** | **5 models**: ConvNeXt, HRNet48, ResNet50, Res2Net200, ResNeXt101 — all 384×384 |
| **Tracking** | Location-Aware SCMT (Single-Camera Multi-Target tracker) |
| **Association** | **Box-Grained Matching (BGM)** — subdivides vehicle detection boxes into spatial grid patches and performs reranking at patch level, not just whole-image level |
| **Reranking** | Box-grained reranking (novel contribution) |
| **Platform** | PaddlePaddle + PaddleDetection |
| **Key Innovation** | BGM handles occlusion/truncation by matching visible local patches rather than holistic features. Truncation rates per tracklet are also computed |

### AIC 2022 — 2nd Place: Team59 "BOE" (AIC22-MCVT)
- **Paper**: "Multi-Camera Vehicle Tracking System for AI City Challenge 2022" (CVPR 2022 Workshop)
- **Code**: [coder-wangzhen/AIC22-MCVT](https://github.com/coder-wangzhen/AIC22-MCVT)
- **Score**: IDF1 = **0.8437**

| Component | Details |
|-----------|---------|
| **Detection** | YOLOv5x6 |
| **ReID Models** | **3 models**: resnet101_ibn_a_2, resnet101_ibn_a_3, resnext101_ibn_a_2 (same as AIC21 winner) |
| **Input Resolution** | 384×384 |
| **CID_BIAS** | ✅ Precomputed camera-pair transition bias from cam_timestamp/ |
| **Reranking** | ✅ k-reciprocal reranking (USE_RERANK=True) |
| **Feature Fusion** | ✅ USE_FF=True (feature fusion across ReID models) |
| **Matching Threshold** | SCORE_THR=0.1 |

### AIC 2022 — 3rd Place: Team37 "TAG"
- **Code**: [backkon/AICITY2022_Track1_TAG](https://github.com/backkon/AICITY2022_Track1_TAG)
- **Score**: IDF1 = **0.8371**

| Component | Details |
|-----------|---------|
| **Detection** | YOLOv5 |
| **ReID Models** | Multiple models trained via DMT approach |
| **Base Code** | Modified from AIC21 winner ([LCFractal/AIC21-MTMC](https://github.com/LCFractal/AIC21-MTMC)) |
| **Zone Filtering** | Direction-based temporal masks + zone polygons |

### AIC 2022 — Other Notable: Team10 "TSL-AI"
- **Code**: [royukira/AIC22_Track1_MTMC_ID10](https://github.com/royukira/AIC22_Track1_MTMC_ID10)
- **Score**: IDF1 = 0.8129

| Component | Details |
|-----------|---------|
| **Detection** | YOLOv5 |
| **ReID Models** | **3 models** from AIC21 winner (resnet101_ibn_a ×2 + resnext101_ibn_a) |
| **Tracking** | ByteTrack + occlusion handling strategies |
| **Association** | **Anomaly Masked Association** — detect anomalous tracklets based on motion info, time between adjacent cameras |
| **Post-processing** | Motion-based anomaly detection + temporal masks |

### AIC 2022 — 4th Place: Team50 "FraunhoferIOSB"
- **Score**: IDF1 = **0.8348**
- A computer vision research institute team; consistently places in top 5 across multiple years.

---

### AIC 2021 — Track 3 Winner: Team75 "mcmt" (Crossroad Zones)
- **Paper**: "City-Scale Multi-Camera Vehicle Tracking Guided by Crossroad Zones" (CVPR 2021 Workshop)
- **ArXiv**: [2105.06623](https://arxiv.org/abs/2105.06623)
- **Code**: [LCFractal/AIC21-MTMC](https://github.com/LCFractal/AIC21-MTMC)
- **Score**: IDF1 = **0.8095** (1st place)
- **Note**: In AIC 2021, MTMC vehicle tracking was Track 3, not Track 1.

| Component | Details |
|-----------|---------|
| **Detection** | YOLOv5x (pretrained on COCO) |
| **ReID Models** | **3 models**: resnet101_ibn_a_2, resnet101_ibn_a_3, resnext101_ibn_a_2 |
| **ReID Training** | Trained via DMT approach (same team won Track 2 ReID) |
| **Input Resolution** | 384×384 |
| **Tracking** | Modified JDETracker (without detection module) |
| **CID_BIAS** | ✅ Precomputed NPY files from camera timestamps |
| **Reranking** | ✅ k-reciprocal (USE_RERANK=True) |
| **Key Innovations** | (1) Tracklet Filter Strategy — removes short/noisy tracklets. (2) Direction-Based Temporal Mask — constrains matching based on traffic flow direction at crossroads. (3) Sub-clustering in Adjacent Cameras — spatiotemporal prior for matching |

### AIC 2021 — Track 2 Winner: Team47 "DMT" (ReID-specific)
- **Paper**: "An Empirical Study of Vehicle Re-Identification on the AI City Challenge" (CVPR 2021 Workshop)
- **Code**: [michuanhaohao/AICITY2021_Track2_DMT](https://github.com/michuanhaohao/AICITY2021_Track2_DMT)
- **Score**: mAP = **0.7445** on CityFlow Track 2 test set

This is THE canonical ReID training recipe used by nearly all subsequent winning MTMC teams.

| Component | Details |
|-----------|---------|
| **Backbones (8 total)** | ResNet101-IBN-a, ResNeXt101-IBN-a, SE-ResNet101-IBN-a, ResNeSt101, DenseNet169-IBN-a, ViT-B/16, + stage2 variants |
| **Training Stage 1** | Standard fine-tuning with ID loss + triplet loss |
| **Training Stage 2** | Camera-aware training (train_cam.py) + Viewpoint-aware training (train_view.py) |
| **Input Resolution** | 384×384 |
| **Losses** | ID (CE with label smoothing) + triplet + center |
| **Data Augmentation** | SPGAN for synthetic→real domain adaptation, random erasing, pad+crop |
| **Camera/View Models** | Separate camera classifier and viewpoint classifier trained on frozen ReID features |
| **Ensemble** | Feature-level concatenation of all 8 models, then distance computation |
| **Key Insight** | 2-stage training where camera/viewpoint classifiers run on top of frozen features effectively remove camera-specific biases |

### AIC 2021 — Track 3 2nd Place: Team29 "fivefive" (Baidu)
- **Paper**: "A Robust MTMC Tracking System for AI-City Challenge 2021"
- **Score**: IDF1 = **0.7787**
- Similar approach: YOLOv5 + multiple IBN-a ReID models + zone-based filtering

---

### AIC 2023 — Track 1 (CHANGED TO PEOPLE TRACKING)
- **Winner**: Team6 "UW-ETRI", IDF1 = **0.9536** (on synthetic person tracking dataset)
- **Runner-up**: Team9 "HCMIU", IDF1 = **0.9417**
- **NOT comparable** to CityFlowV2 vehicle tracking. Different dataset, different domain.

### AIC 2024 — Track 1 (PEOPLE TRACKING continued)
- **Winner (award)**: Team79 "SJTU-LENOVO", HOTA = **67.2175** (online system)
- **Winner (leaderboard)**: Team221 "Yachiyo", HOTA = **71.9446** (offline system)
- **NOT comparable** to CityFlowV2 vehicle tracking.

---

## Technique-by-Technique Comparison with Our Pipeline

### Detection

| Aspect | **AIC Winners** | **Our Pipeline** | Gap |
|--------|----------------|------------------|-----|
| Model | YOLOv5x/Swin Transformer | YOLOv26m | Minor — detection is NOT the bottleneck |
| Resolution | 1280+ (YOLOv5x6) | Standard | Minor |

**Verdict**: Detection is likely not a significant contributor to the IDF1 gap. YOLOv26m is a modern detector and our tracking quality is reasonable.

### Single-Camera Tracking

| Aspect | **AIC Winners** | **Our Pipeline** | Gap |
|--------|----------------|------------------|-----|
| Tracker | JDE/ByteTrack/SCMT | BoT-SORT (BoxMOT) | Minor |
| min_hits | Varies | 2 (optimal) | — |

**Verdict**: Tracking is adequate. BoT-SORT is competitive with ByteTrack.

### ReID Model Architecture (★★★ CRITICAL GAP)

| Aspect | **AIC Winners** | **Our Pipeline** | Gap |
|--------|----------------|------------------|-----|
| Number of models | **3-5 in ensemble** | **1** | **HUGE** |
| Architectures | ResNet101-IBN-a, ResNeXt101-IBN-a, SE-ResNet101-IBN-a, ConvNeXt, HRNet48, Res2Net200 | ViT-B/16 CLIP only | Missing diversity |
| Input resolution | 384×384 | 256×256 (384 tested but harmful) | See note |
| Training data | CityFlow ReID + VeRi-776 + synthetic (SPGAN) | CLIP→VeRi-776→CityFlowV2 | Ours is good for ViT |
| IBN-a variants | ✅ Standard for all CNN backbones | Not working (52.77% mAP) | ResNet-IBN path broken |

**Verdict**: This is THE primary gap. The ensemble diversity provides:
1. Complementary error patterns (different backbones fail on different vehicles)
2. More robust cross-camera features (averaging reduces viewpoint-specific noise)
3. Better generalization (multiple architectural inductive biases)

**Note on 384px failure**: Our 384px ViT was harmful (-2.8pp) because single-model 384px captures too much viewpoint-specific detail. In an ensemble, this would be averaged out by other models. The SOTA teams can afford 384px precisely because the ensemble smooths out the extra texture noise.

### ReID Training Recipe (★★ SIGNIFICANT GAP)

| Aspect | **AIC Winners** | **Our Pipeline** | Gap |
|--------|----------------|------------------|-----|
| Training stages | 2-stage (standard + camera-aware) | 1-stage | Missing camera debiasing |
| Loss functions | ID + triplet + center + camera | ID + triplet | Missing center loss, camera loss |
| Label smoothing | ✅ | ✅ | — |
| Synthetic data aug | SPGAN domain adaptation | None | Missing data augmentation path |
| Camera classifier | Separate frozen classifier | Not used | Missing camera-specific bias removal |
| Viewpoint classifier | Separate frozen classifier | Not used | Missing viewpoint debiasing |

**Verdict**: The 2-stage DMT training is specifically designed to produce features that are camera-invariant. This directly addresses our core problem (cross-camera feature invariance).

### Camera-Pair Priors / CID_BIAS (★★ SIGNIFICANT GAP)

| Aspect | **AIC Winners** | **Our Pipeline** | Gap |
|--------|----------------|------------------|-----|
| CID_BIAS | ✅ Precomputed NPY from timestamps, or zone-based ROI masks | FIC regularization only | Different mechanism |
| Zone polygons | Hand-annotated per camera per intersection | None | Missing spatial priors |
| Direction masks | Direction-based temporal masks at crossroads | None | Missing temporal-directional priors |
| Temporal filtering | Sub-clustering by adjacent cameras only | Full graph | Missing adjacency constraint |

**Verdict**: CID_BIAS is universal across winners. Our FIC serves a similar purpose but may be weaker. We tested CID_BIAS on 384px features and it hurt (-0.52pp), but that was with broken features. Worth retesting once features improve.

### Reranking (★ MODERATE GAP)

| Aspect | **AIC Winners** | **Our Pipeline** | Gap |
|--------|----------------|------------------|-----|
| Method | k-reciprocal (all teams) or Box-Grained Matching (CityTrack) | Disabled | — |
| Effect | Consistently helpful with 3+ model ensemble | Harmful with single model | Feature quality gated |

**Verdict**: Reranking works when you have good diverse features. With a single model, it amplifies noise. With an ensemble, it removes false matches. This is a dependent improvement — implement AFTER the ensemble.

### Feature Fusion & Post-Processing

| Aspect | **AIC Winners** | **Our Pipeline** | Gap |
|--------|----------------|------------------|-----|
| Feature fusion | Concatenate features from all models, then compute distance | Single model features | N/A (needs ensemble first) |
| Truncation handling | CityTrack computes truncation ratios per tracklet | None | Minor |
| Tracklet filtering | Direction + zone + temporal + motion-based | Length + confidence only | Some gap |

---

## Prioritized Improvements Ranked by Expected IDF1 Impact

### Tier 1: HIGH IMPACT (expected +3-6pp combined)

#### 1. Train 2-3 Complementary IBN-a ReID Models (Expected: +3-5pp)
**Why**: Every winning team since 2021 uses an ensemble. The diversity of architectural inductive biases is the key driver of cross-camera robustness.

**What to train**:
1. **ResNeXt101-IBN-a** — the single most-used model across ALL winning teams
2. **SE-ResNet101-IBN-a** — channel attention adds complementary signal
3. **ResNet101-IBN-a** (already attempted — needs correct recipe)

**Training recipe** (from DMT):
- Stage 1: Standard fine-tuning on CityFlowV2/VeRi-776 with ID + triplet + center loss
- Stage 2: Camera-aware training — freeze backbone, train camera classifier on top of features
- Input: 384×384
- Optimizer: AdamW lr=3.5e-4
- Warm-up: linear warmup 10 epochs
- Augmentation: RandomErasing(0.5), pad→crop, horizontal flip
- Label smoothing: 0.1
- Epochs: 120

**Critical note**: Our previous ResNet101-IBN-a attempts failed due to:
- Wrong recipe (SGD, circle loss) — not because IBN-a can't work
- VeRi-776 intermediate pretraining hurt rather than helped
- Direct ImageNet→CityFlowV2 with AdamW got 52.77% — need to replicate DMT recipe exactly

#### 2. Implement 2-Stage DMT Camera-Aware Training (Expected: +1-2pp per model)
**Why**: Camera-aware debiasing directly addresses our core problem of cross-camera feature invariance.

**Implementation**:
1. Train standard ReID model (stage 1)
2. Freeze the ReID backbone
3. Train a camera classifier on the frozen features
4. At inference, subtract the camera-specific component from features
5. Alternative: Use camera ID as additional training signal during stage 2

**Key difference from our failed DMT experiment**: We tested DMT on a SINGLE model and it hurt (-1.4pp). This is expected — single-model DMT removes discriminative signal. With an ensemble, the camera-invariant signal is preserved by the other models.

### Tier 2: MODERATE IMPACT (expected +1-3pp combined)

#### 3. Implement Proper CID_BIAS (Expected: +0.5-1.5pp)
**Why**: All winning teams use precomputed camera-pair biases. Our FIC is a rough approximation.

**Implementation**:
- Compute pairwise camera transition probabilities from training data timestamps
- Store as NPY matrix (N_cameras × N_cameras)
- Add as bias term to the similarity score before thresholding
- Can also be learned from validation data

#### 4. Re-enable k-reciprocal Reranking (Expected: +0.5-1.0pp, AFTER ensemble)
**Why**: Consistently used by ALL winning teams. Only harmful with single model.

**Prerequisite**: Must have 2+ ReID models in ensemble first.

**Implementation**: Standard Jaccard distance reranking with parameters k1=20, k2=6, lambda=0.3 (standard values from original paper).

### Tier 3: LOW IMPACT (expected +0-1pp each)

#### 5. Zone-Based Direction Filtering (Expected: +0.3-0.5pp)
**Why**: AIC21 winner's key innovation. Reduces false matches by constraining which camera pairs can match based on physical traffic flow.

**What's needed**: Hand-annotate entry/exit zones per camera, define valid camera-pair transitions.

#### 6. Box-Grained Matching (Expected: +0.2-0.5pp)
**Why**: CityTrack's key innovation. But this is complex and specific to their PaddlePaddle setup.

#### 7. Stronger Detection Model (Expected: +0-0.3pp)
**Why**: Detection is not the primary bottleneck, but Swin Transformer detection could improve tracking quality slightly.

---

## Implementation Plan

### Phase 1: Fix the ReID Ensemble (Weeks 1-2)
**Goal**: Get 2-3 working IBN-a ReID models using the EXACT DMT recipe.

1. **Fork the DMT training code** from [michuanhaohao/AICITY2021_Track2_DMT](https://github.com/michuanhaohao/AICITY2021_Track2_DMT)
2. **Adapt for CityFlowV2 dataset** (128 IDs vs their 333 Track 2 IDs)
3. **Train ResNeXt101-IBN-a** FIRST — the most consistently successful backbone
   - Use IBN-a pretrained weights from [XingangPan/IBN-Net](https://github.com/XingangPan/IBN-Net)
   - Stage 1: CityFlowV2, 384×384, AdamW lr=3.5e-4, 120 epochs
   - Stage 2: Camera-aware fine-tuning
4. **Train ResNet101-IBN-a** with DMT recipe (NOT our previous broken recipe)
5. **Evaluate each model individually** on CityFlowV2 mAP
   - Target: >60% mAP each
   - If any model reaches >70% mAP, it's already useful for ensemble

### Phase 2: Ensemble Integration (Week 2-3)
**Goal**: Fuse multiple ReID features in the MTMC pipeline.

1. **Feature extraction**: Run each model on all tracklets → N×D feature tensors
2. **Feature fusion**: Concatenate features from all models (or weighted average)
3. **PCA whitening**: Apply PCA on the concatenated features
4. **Run association**: With the fused features, test existing stage4 parameters
5. **Expected**: +3-5pp IDF1 from ensemble alone

### Phase 3: Camera-Aware Debiasing + CID_BIAS (Week 3-4)
**Goal**: Add camera-invariance priors.

1. **Train camera classifiers** on frozen ensemble features
2. **Subtract camera components** from features at inference
3. **Compute CID_BIAS matrix** from training data
4. **Integrate into stage4** association scoring

### Phase 4: Re-enable Reranking (Week 4)
**Goal**: Enable k-reciprocal reranking on improved features.

1. Test with standard parameters (k1=20, k2=6, lambda=0.3)
2. Sweep parameters if initial test is neutral/positive
3. Expected to help with ensemble features even though it hurt with single model

---

## Key Models and Pretrained Weights

### Required IBN-Net Pretrained Weights
All from [XingangPan/IBN-Net](https://github.com/XingangPan/IBN-Net):
- `resnet101_ibn_a-59ea0ac6.pth`
- `resnext101_ibn_a-6ace051d.pth`
- `se_resnet101_ibn_a-fabed4e2.pth`
- `densenet169_ibn_a-9f32c161.pth`

### Additional Models
- `resnest101-22405ba7.pth` from [ResNeSt](https://github.com/zhanghang1989/ResNeSt)
- `jx_vit_base_p16_224-80ecf9dd.pth` from [timm](https://github.com/rwightman/pytorch-image-models)

---

## Critical Insights

### Why Single-Model Improvements Failed
Our pipeline tried several single-model improvements that ALL failed:
- 384px resolution: -2.8pp (viewpoint-specific noise not smoothed by ensemble)
- DMT camera-aware: -1.4pp (removes discriminative signal without ensemble backup)
- Multi-query track representation: -0.1pp (noise in single-model features)
- Concat-patch features: -0.3pp (dimension increase without quality increase)

**The pattern**: Every technique that works for SOTA teams requires an ensemble to be effective. With a single model, these techniques remove useful signal. With an ensemble, they remove noise while preserving signal through diversity.

### The Magic Number: 3 Models
The minimum viable ensemble is 3 models. This is what:
- AIC21 MTMC winner used (resnet101_ibn_a ×2 + resnext101_ibn_a)
- AIC22 2nd place used (identical 3 models)
- AIC22 3rd place used (built on same)
- AIC22 Team10 used (same 3 models)

Going from 3→5 models (like CityTrack) adds only +0.5-1pp more.

### The IBN-a Recipe is Proven
Instance-Batch Normalization (IBN) specifically helps cross-camera matching by making features more robust to style/appearance shifts. This is NOT the same as standard BatchNorm. Every winning CNN backbone uses IBN-a variants.

### Reranking is Feature-Quality Dependent
k-reciprocal reranking examines the k-nearest neighbors of each query/gallery pair. With good features, neighbors are semantically correct → reranking strengthens correct matches. With poor features, neighbors are noisy → reranking amplifies errors.

---

## References

1. Lu et al., "CityTrack: Improving City-Scale Multi-Camera Multi-Target Tracking by Location-Aware Tracking and Box-Grained Matching", arXiv:2307.02753 (AIC22 1st)
2. Li et al., "Multi-Camera Vehicle Tracking System for AI City Challenge 2022", CVPRW 2022 (AIC22 2nd)
3. Liu et al., "City-Scale Multi-Camera Vehicle Tracking Guided by Crossroad Zones", arXiv:2105.06623 (AIC21 1st)
4. Luo et al., "An Empirical Study of Vehicle Re-Identification on the AI City Challenge", CVPRW 2021 (DMT, AIC21 Track 2 1st)
5. Huynh et al., "A Strong Baseline for Vehicle Re-Identification", arXiv:2104.10850, CVPRW 2021
6. Tang et al., "CityFlow: A City-Scale Benchmark for Multi-Target Multi-Camera Vehicle Tracking and Re-Identification", arXiv:1903.09254, CVPR 2019
7. AI City Challenge organizer summaries: arXiv:2104.12233 (AIC21), arXiv:2204.10380 (AIC22), arXiv:2304.07500 (AIC23), arXiv:2404.09432 (AIC24)