# Pretrained Vehicle ReID Ensemble v2 — Actionable Experiment Spec

## Status: READY FOR IMPLEMENTATION (Phase 1 is cheap; Phase 2 conditional)

## Executive Summary

Use community-pretrained vehicle ReID models (fast-reid SBS R50-IBN-a on VeRi-776) as a secondary feature extractor in the MTMC pipeline. This is the only remaining untried ensemble path: **architecturally diverse** (CNN vs ViT) AND **community-pretrained** on a vehicle dataset (not trained from scratch on CityFlowV2's tiny 128 IDs).

**Honest probability of improving MTMC IDF1: 15-25%.**

This is a long shot, but Phase 1 is cheap (~3 hours Kaggle time) and provides valuable diagnostic information regardless of outcome.

## Why Previous Ensembles Failed — And Why This Might Be Different

| Previous Attempt | mAP (CityFlowV2) | Fusion Result | Root Cause |
|---|---|---|---|
| LAION-2B CLIP ViT (09l v3) | 78.61% | -0.5pp | Same CLIP ViT architecture → correlated features |
| ResNet101-IBN-a (09d) | 52.77% | -0.1pp | Too weak (28pp deficit) → noise at 30% weight |
| ResNeXt101-IBN-a (09j) | 36.88% | N/A | Broken weight loading → dead on arrival |
| ViT-Small IN-21k (09k) | 48.66% | N/A | Non-CLIP ceiling |
| CLIP RN50x4 (09m) | 1.55% | N/A | QuickGELU mismatch → catastrophic |

**The pattern**: every secondary was either (a) architecturally similar to the primary (LAION ViT), or (b) trained from scratch on only 128 CityFlowV2 IDs and too weak.

**What's new**: A fast-reid SBS(R50-ibn) model pretrained on VeRi-776 (576 IDs) is:
1. **Architecturally diverse**: CNN with GeM+NL blocks vs ViT with CLS token
2. **NOT trained from scratch on CityFlowV2**: Uses a community-pretrained checkpoint that already understands vehicle identity at 81.9% mAP on VeRi-776
3. **Different training recipe**: Circle loss + GeM + Non-local + auto-aug vs our CE + Triplet + CLIP init

## Critical Honest Assessment

### The Elephant in the Room: VeRi→CityFlowV2 Domain Gap

**81.9% mAP on VeRi-776 ≠ anything on CityFlowV2.** These are different datasets:

| Property | VeRi-776 | CityFlowV2 |
|---|---|---|
| Location | Beijing, China | US intersections |
| Vehicle types | Chinese vehicle models | US vehicle models |
| Camera angles | 20 cameras, varied | 6 cameras, intersection |
| # Training IDs | 576 | 128 |
| # Training images | 37,778 | ~7,500 |

**Our own VeRi→CityFlowV2 transfer evidence**: 09e trained ResNet101-IBN-a on VeRi-776 to 62.52% mAP (on VeRi test), then 09f fine-tuned on CityFlowV2 and got only 42.7% mAP — **worse than direct ImageNet→CityFlowV2 (52.77%)**. VeRi pretraining actually hurt.

**Why this might still differ**: The SBS model starts from a much stronger VeRi baseline (81.9% vs our 62.52%), uses a fundamentally different architecture (NL blocks + GeM), and was trained with a different recipe (circle loss + cutout + auto-aug). The feature space geometry might transfer better. But we must be realistic: without CityFlowV2 fine-tuning, this model will likely perform at **30-50% mAP on CityFlowV2** — worse than the 52.77% ResNet that already failed in ensemble.

### What Has to Go Right for This to Work

1. The VeRi-trained features must capture **vehicle type/color/shape** signals that are universal enough to help cross-camera matching on CityFlowV2
2. Even if absolute CityFlowV2 mAP is low, the error patterns must be **uncorrelated** with the primary ViT's errors
3. PCA whitening + FIC calibration must be able to **normalize the distribution gap** between VeRi-trained and CityFlowV2-trained feature spaces
4. The fusion weight sweep must find a regime where the diverse-but-weak secondary helps more than it hurts

**Probability assessment**:
- Path A (direct zero-shot deployment): **10-15%** chance of any IDF1 improvement
- Path B (SBS init → CityFlowV2 fine-tuning): **20-30%** chance
- Combined: **~20-25%** that any path produces a net MTMC IDF1 gain

## Available Pretrained Models (Confirmed from fast-reid MODEL_ZOO)

### Candidate #1: fast-reid VeRi SBS(R50-ibn) ⭐⭐⭐ PRIMARY

| Property | Value |
|---|---|
| **Architecture** | ResNet50-IBN-a + Non-local [0,2,3,0] + GeM pooling + BNNeck |
| **Training data** | VeRi-776 (576 IDs, 37,778 train images) |
| **VeRi metrics** | R1=97.0%, mAP=81.9%, mINP=46.3% |
| **Feature dim** | 2048 (BNNeck output) |
| **Input size** | 256×256 (standard fastreid vehicle config) |
| **Download URL** | `https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/veri_sbs_R50-ibn.pth` |
| **License** | Apache 2.0 |
| **Download** | ✅ Direct GitHub releases (no Google Drive) |
| **Diversity** | ✅ Maximum (CNN vs ViT, GeM vs CLS token, circle loss vs triplet, ImageNet init vs CLIP init) |

### Candidate #2: fast-reid VERI-Wild BoT(R50-ibn) ⭐⭐ BACKUP

| Property | Value |
|---|---|
| **Architecture** | ResNet50-IBN-a + BNNeck (no NL blocks, standard GAP) |
| **Training data** | VERI-Wild (40,671 IDs, 174K images, 174 cameras) |
| **VERI-Wild metrics** | R1=96.4% (small), mAP=87.7% (small) |
| **Feature dim** | 2048 |
| **Download URL** | `https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/veriwild_bot_R50-ibn.pth` |
| **Why backup** | Much larger training set (40K IDs vs 576) may produce more generalizable features, but simpler architecture (no NL/GeM) and trained on Chinese highway cameras (larger domain gap than VeRi's city cameras) |

### Candidate #3: fast-reid VehicleID BoT(R50-ibn) ⭐ LOWEST PRIORITY

| Property | Value |
|---|---|
| **Architecture** | ResNet50-IBN-a + BNNeck (no NL blocks) |
| **Training data** | VehicleID (26,267 IDs, 221K images) |
| **VehicleID metrics** | R1=86.6% (small), 82.9% (medium), 80.6% (large) |
| **Feature dim** | 2048 |
| **Download URL** | `https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/vehicleid_bot_R50-ibn.pth` |
| **Why lowest** | VehicleID has front/rear viewpoints only — very different from CityFlowV2's multi-angle intersection cameras |

### Rejected Candidates

| Model | Reason |
|---|---|
| **layumi VehicleNet R50** | Google Drive download (flaky on Kaggle), non-trivial model format adaptation |
| **CLIP-ReID ViT on VeRi** | Same CLIP ViT architecture family → correlated with primary (same failure as 09l v3 / 10c v56) |
| **TransReID official VeRi** | No pretrained weights publicly available |
| **AIC22 1st place** | Repository returns 404 |

## Experiment Design

### Phase 1: Zero-Shot Diagnostic (ONE 10a→10c chain run)

**Goal**: Determine the CityFlowV2 transfer quality of VeRi-pretrained features and test score-level fusion.

**Cost**: ~3 hours total Kaggle time. One 10a run (GPU) + 10b (CPU) + 10c (CPU).

#### Step 1: Download weights in 10a

```python
# Early cell in 10a notebook
!wget -q https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/veri_sbs_R50-ibn.pth \
    -O models/reid/fastreid_veri_sbs_r50ibn.pth
```

#### Step 2: Build inference model in 10a

Need a new `ReIDModelResNet50IBN` class. Nearly identical to the existing `ReIDModelResNet101IBN` in `src/training/model.py`, but using torchvision's `resnet50` instead of `resnet101`.

Key implementation details:
- Use `torchvision.models.resnet50(weights=None)` as base
- Apply IBN-a patches to layer1, layer2, layer3 (same code as R101)
- Set `last_stride=1` in layer4 (same as R101)
- Add GeM pooling (same as R101, p=3.0)
- Add BNNeck (same as R101)
- Forward returns `F.normalize(bn_feat, p=2, dim=1)` in eval mode

Fast-reid state dict key mapping:
```
backbone.conv1.weight          → backbone.conv1.weight      (direct)
backbone.bn1.*                 → backbone.bn1.*             (direct, but IBN splits need care)
backbone.layer{1-4}.*          → backbone.layer{1-4}.*      (direct — R50 has [3,4,6,3] blocks)
backbone.NL_2.*, backbone.NL_3.* → SKIP (not in our model, loaded with strict=False)
heads.pool_layer.p             → pool.p                     (GeM parameter)
heads.bottleneck.0.*           → bottleneck.*               (BNNeck)
heads.classifier.*             → SKIP (not used for inference)
```

The IBN-a layer naming in fast-reid uses `IN` / `BN` split notation. Our IBN-a implementation uses the same pattern. Need to verify exact key names match or add remapping.

**Critical**: Load with `strict=False`. Non-local block weights will be "unexpected" and ignored. The backbone features are still meaningful without them — NL blocks are additive attention modules.

#### Step 3: Register in ReIDModel dispatcher

Add `"fastreid_r50_ibn"` as a new model_name in `ReIDModel._build_model()`, routing to a new `_build_fastreid_r50_ibn(self, weights_path)` method. This method:
1. Builds ReIDModelResNet50IBN(num_classes=1, last_stride=1, gem_p=3.0)
2. Loads fast-reid checkpoint with key remapping
3. Uses ImageNet normalization (NOT CLIP)
4. Uses BILINEAR interpolation (CNN, not ViT)

#### Step 4: Configure vehicle2 in 10a

```yaml
stage2.reid.vehicle2:
  enabled: true
  save_separate: true
  model_name: "fastreid_r50_ibn"
  weights_path: "models/reid/fastreid_veri_sbs_r50ibn.pth"
  embedding_dim: 2048
  input_size: [256, 256]
  clip_normalization: false
```

#### Step 5: PCA whitening

The 2048D features get PCA-whitened to 384D using a SEPARATE PCA model (saved as `pca_transform_secondary.pkl`). This is already handled by the existing Stage 2 pipeline when `save_separate: true` is set.

#### Step 6: Fusion sweep in 10c

```python
# 10c sweep matrix
FUSION_WEIGHTS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
# Keep primary association params at v52 baseline:
# sim_thresh=0.50, appearance_weight=0.70, fic_reg=0.50, aqe_k=3
# gallery_thresh=0.48, orphan_match=0.38
```

Lower fusion weights than previous attempts (0.05-0.15) because the secondary is expected to be weaker. Also test separate FIC regularization for secondary.

#### Step 7: Diagnostic outputs

Regardless of IDF1 result, capture:
- Secondary model's per-camera mean cosine similarity matrix (to assess feature quality)
- Distribution overlap between primary and secondary similarity scores
- Per-camera pair improvement/degradation analysis
- Number of pairs where primary and secondary disagree on top-1 match

### Phase 2: CityFlowV2 Fine-Tuning (conditional on Phase 1)

**Gate condition**: Phase 1 shows ANY of:
- Positive IDF1 delta at any fusion weight, OR
- Secondary features show meaningful cross-camera discrimination (mean inter-camera cosine < 0.5 for different IDs), OR
- Diversity analysis shows the secondary disagrees with primary on >20% of difficult pairs

If Phase 1 is completely negative (secondary features are random noise on CityFlowV2), skip Phase 2.

#### 09n Notebook Design

**Key difference from failed 09f**: The SBS R50-ibn starts from a MUCH stronger VeRi baseline (81.9% mAP vs our 62.52%) and has a different architecture. The fine-tuning recipe should preserve the VeRi-learned features while adapting to CityFlowV2.

```python
# 09n: Fine-tune fast-reid VeRi SBS R50-ibn on CityFlowV2
# Recipe: CE+LS(0.05) + Triplet(0.3) + Center(5e-4, delayed epoch 15)
# DO NOT use CircleLoss (confirmed dead end)
# DO NOT use ArcFace (confirmed dead end for ResNet)

# Key hyperparameters:
BACKBONE_LR = 5e-5     # Lower than ViT (1e-4) — preserve VeRi features
HEAD_LR = 5e-4          # Standard
EPOCHS = 120
EMA = False             # Confirmed dead end
OPTIMIZER = "AdamW"     # NEVER use SGD (confirmed dead end)
INPUT_SIZE = (256, 256)
BATCH_SIZE = 32         # P_K sampling: P=16 identities, K=2 per identity
```

**SBS→CityFlowV2 adaptation concerns**:
1. The SBS model was trained with circle loss; our recipe uses CE+triplet. The feature geometry may not adapt smoothly. Use a lower backbone LR to mitigate.
2. The 128-ID CityFlowV2 train set is much smaller than VeRi-776. Regularization is critical.
3. GeM pooling parameters should be frozen initially (use the learned p from VeRi training).

**Success threshold**: mAP ≥ 60% on CityFlowV2 eval. Below this, ensemble is likely to add noise.

### Phase 3: VERI-Wild Backup (conditional on Phase 1)

If Phase 1 with VeRi SBS is negative, try the VERI-Wild BoT model as a second data point. The VERI-Wild training set (40K IDs, 174K images) is 70× larger than VeRi and might produce more generalizable vehicle features despite the simpler BoT architecture.

Same 10a integration, just swap the weights URL:
```python
!wget -q https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/veriwild_bot_R50-ibn.pth \
    -O models/reid/fastreid_veriwild_bot_r50ibn.pth
```

The BoT model has NO non-local blocks and uses GAP instead of GeM, so the key mapping is simpler. The existing `ReIDModelResNet50IBN` class (without NL blocks) will work directly.

## Implementation Changes Required

### 1. `src/training/model.py` — Add `ReIDModelResNet50IBN`

```
class ReIDModelResNet50IBN(nn.Module):
    # Nearly identical to ReIDModelResNet101IBN
    # Differences:
    #   - Uses torchvision.models.resnet50 instead of resnet101
    #   - R50 has [3, 4, 6, 3] blocks vs R101's [3, 4, 23, 3]
    #   - Everything else (IBN-a patches, last_stride, GeM, BNNeck) is identical
```

### 2. `src/stage2_features/reid_model.py` — Add `fastreid_r50_ibn` dispatcher

```python
def _build_model(self, model_name, weights_path):
    if self.is_transreid:
        return self._build_transreid(weights_path)
    if model_name.lower() == "resnet101_ibn_a":
        return self._build_resnet101_ibn(weights_path)
    if model_name.lower() == "resnext101_ibn_a":
        return self._build_resnext101_ibn(weights_path)
    if model_name.lower() == "fastreid_r50_ibn":        # NEW
        return self._build_fastreid_r50_ibn(weights_path)
    return self._build_torchreid(model_name, weights_path)
```

New `_build_fastreid_r50_ibn()` method handles the fast-reid state dict key remapping.

### 3. `notebooks/kaggle/10a_stages012/` — Add secondary extraction cell

Add a cell after primary extraction that:
1. Downloads fast-reid VeRi weights
2. Builds the fastreid_r50_ibn model
3. Extracts features from the same crops
4. PCA-whitens to 384D with a separate PCA model
5. Saves as `embeddings_secondary.npy`

### 4. `configs/default.yaml` — vehicle2 config for fastreid_r50_ibn

Update the vehicle2 section defaults:
```yaml
vehicle2:
  enabled: false
  save_separate: true
  model_name: "fastreid_r50_ibn"
  weights_path: "models/reid/fastreid_veri_sbs_r50ibn.pth"
  embedding_dim: 2048
  input_size: [256, 256]
  clip_normalization: false
```

## Expected Outcomes

### Optimistic (15% probability)
- Secondary adds +1-2pp MTMC IDF1 at 10-20% fusion weight
- Architecturally diverse CNN features complement ViT on difficult same-model vehicle pairs
- CityFlowV2 fine-tuning (Phase 2) pushes secondary to >65% mAP and enables stronger fusion

### Realistic (60% probability)
- Zero-shot transfer gives <40% CityFlowV2 mAP
- Fusion is neutral to slightly negative (-0.1 to 0.0pp)
- BUT we learn that the VeRi→CityFlowV2 domain gap is fundamental, closing this path definitively

### Pessimistic (25% probability)
- Features are near-random on CityFlowV2 (like CLIP RN50x4)
- Fusion degrades IDF1 at every weight
- Phase 2 fine-tuning also fails (VeRi init hurts like 09f)

## What We Learn Either Way

Even if the experiment fails, it provides:
1. **First direct measurement of community VeRi→CityFlowV2 transfer quality** (never tested with a strong model)
2. **Confirmation/refutation of the "diversity matters more than quality" ensemble hypothesis**
3. **Data point on whether CNN features are truly orthogonal to ViT features on this task**
4. **Definitive closure of the pretrained-ensemble path** if it fails, freeing focus for GNN association or paper writing

## Execution Order

1. **Implement `ReIDModelResNet50IBN`** in `src/training/model.py` (reuse R101 code, change base to resnet50)
2. **Add `_build_fastreid_r50_ibn()`** in `src/stage2_features/reid_model.py` (key remapping)
3. **Update 10a notebook** with download + secondary extraction cell
4. **Push 10a** → Run on Kaggle (extracts primary + secondary features)
5. **Chain to 10b → 10c** with fusion weight sweep
6. **Evaluate results** → decide Phase 2 gate

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| fast-reid state dict key mismatch | Model loads garbage features | Print loaded/missing/unexpected keys; validate mean activation |
| IBN-a implementation differences | Subtle feature corruption | Compare our IBN-a with fast-reid's; test with a known input |
| Non-local block removal hurts features | Lower quality than advertised 81.9% | The backbone features are still valid; NL blocks are additive |
| PCA whitening doesn't calibrate well | Misaligned feature space in fusion | Test multiple PCA dims (256, 384, 512) |
| Kaggle download fails | Blocked experiment | fast-reid uses GitHub releases (not Google Drive) — reliable |
| 256×256 vs original fast-reid input size | Mismatched receptive field | Verify fast-reid VeRi config uses 256×256 for vehicles |

## Key Differences From v1 Spec

1. **Brutally honest probability assessment** (15-25% vs v1's implicit ~60%)
2. **Acknowledged the 09f precedent** (VeRi→CityFlowV2 transfer already failed once)
3. **Added VERI-Wild backup** as a second candidate with 70× more training IDs
4. **Phased design** with explicit gate conditions (don't invest in Phase 2 if Phase 1 is dead)
5. **Diagnostic outputs** specified so we learn something even if IDF1 doesn't improve
6. **No hand-waving about VeRi mAP = CityFlowV2 mAP** — these are different datasets