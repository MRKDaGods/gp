# Autonomous CityFlowV2 MTMC IDF1 Improvement Plan

## Status: READY FOR AUTONOMOUS EXECUTION

## Executive Summary

Multi-phase plan to push vehicle MTMC IDF1 from **0.775** toward SOTA (**0.8486**). The primary path is fine-tuning the already-deployed fast-reid SBS R50-IBN on CityFlowV2 to create a strong, architecturally diverse secondary model for score-level ensemble. Backup paths address primary model improvements and alternative secondary candidates.

**Honest probability of meaningful improvement (≥+1pp IDF1): 35-45%.**

The ensemble hypothesis is sound (every AIC winner used 3-5 models), and this time we have genuinely diverse architectures (CNN+GeM vs ViT+CLS). But CityFlowV2's tiny 128-ID train set is the fundamental constraint.

---

## Current State

| Component | Value | Notes |
|-----------|-------|-------|
| Primary model | TransReID ViT-B/16 CLIP 256px | mAP=80.14%, R1=92.27% on CityFlowV2 |
| Secondary model | fast-reid SBS R50-IBN (VeRi-776 pretrained) | 81.9% mAP on VeRi-776, zero-shot on CityFlowV2 gave +0.06pp (noise) |
| Best MTMC IDF1 | 0.775 | 10c v52, v80-restored recipe |
| Association | EXHAUSTED | 225+ configs, all within 0.3pp of optimal |
| SOTA target | 0.8486 | AIC22 1st place (5-model ensemble) |
| Gap | 7.36pp | Caused by feature quality, NOT association |

### Why Zero-Shot Failed and Fine-Tuning Should Work

The zero-shot R50-IBN at +0.06pp is expected, not discouraging:
- **VeRi-776 ≠ CityFlowV2**: Different cities (Beijing vs US), different vehicle models, different camera layouts
- **81.9% mAP on VeRi ≈ 30-50% mAP on CityFlowV2 zero-shot** — massive domain gap
- **09f precedent was with a WEAKER model**: Our previous VeRi→CityFlowV2 transfer (09f) started from only 62.52% mAP on VeRi and reached 42.7% mAP on CityFlowV2. The SBS model starts from 81.9% — a 19pp stronger baseline
- **Architecture diversity is genuine**: CNN (GeM pooling, Non-local blocks, circle-loss-trained) vs ViT (CLS token, CLIP init, triplet-trained). The 10c v56 LAION-2B fusion failed specifically because both models were CLIP ViT-B/16 — too correlated. R50-IBN is fundamentally different.

---

## Phase 1: Fine-Tune R50-IBN on CityFlowV2

### Goal
Train the fast-reid SBS R50-IBN (VeRi-776 pretrained) on CityFlowV2 to reach ≥65% mAP, creating an ensemble-worthy secondary model.

### Hypothesis
The SBS model's strong VeRi-776 features (81.9% mAP) should transfer well to CityFlowV2 with gentle fine-tuning, unlike the weaker 62.52% VeRi baseline that failed in 09f. The key is LOW backbone learning rate to preserve the VeRi-learned vehicle identity representations while adapting the head to CityFlowV2's 128 identities.

### Training Recipe (Detailed)

#### Notebook: `09n_finetune_sbs_r50ibn_cityflow`

```python
# ═══════════════════════════════════════════════════════════
# Architecture
# ═══════════════════════════════════════════════════════════
BACKBONE = "resnet50_ibn_a"        # From fast-reid SBS checkpoint
FEAT_DIM = 2048                     # R50 output dimension
POOLING = "gem"                     # GeM p=3.0 (preserve SBS learned p)
BNNECK = True                       # Standard BoT-style BNNeck
INPUT_SIZE = (256, 256)             # [H, W] — matches deployment config
NUM_CLASSES = 128                   # CityFlowV2 training identities

# ═══════════════════════════════════════════════════════════
# Losses — PROVEN STABLE RECIPE ONLY
# ═══════════════════════════════════════════════════════════
# CE + Label Smoothing: identity classification backbone
CE_LABEL_SMOOTH = 0.05

# Triplet Loss: metric learning for embedding space
TRIPLET_MARGIN = 0.3
TRIPLET_WEIGHT = 1.0

# Center Loss: compact intra-class clusters (delayed start)
CENTER_WEIGHT = 5e-4
CENTER_DELAY_EPOCHS = 15            # Let backbone stabilize before pulling clusters

# ⚠️ NEVER use CircleLoss (inf loss, catastrophic — 09 v4, 09l v1)
# ⚠️ NEVER use ArcFace (geometry mismatch from warm-start — 09i v1)

# ═══════════════════════════════════════════════════════════
# Optimizer — AdamW ONLY
# ═══════════════════════════════════════════════════════════
OPTIMIZER = "AdamW"                 # NEVER SGD (09d mrkdagods: 30.27% mAP catastrophic)
BACKBONE_LR = 3e-5                  # LOW — preserve VeRi features
HEAD_LR = 5e-4                      # Standard head LR
WEIGHT_DECAY = 1e-4
BIAS_DECAY = 0.0                    # No WD on biases

# ═══════════════════════════════════════════════════════════
# Schedule
# ═══════════════════════════════════════════════════════════
EPOCHS = 200                        # Extended: 09l showed gains up to 300 epochs
WARMUP_EPOCHS = 10                  # Linear warmup from 0.1× LR
SCHEDULER = "cosine"                # Cosine annealing to ~0
WARMUP_START_FACTOR = 0.1

# ═══════════════════════════════════════════════════════════
# Batch / Sampling
# ═══════════════════════════════════════════════════════════
BATCH_SIZE = 48                     # P×K: 16 IDs × 3 images per ID
P_IDS = 16                          # Identities per batch
K_INSTANCES = 3                     # Images per identity

# ═══════════════════════════════════════════════════════════
# Augmentation — BASELINE ONLY (augoverhaul confirmed harmful for MTMC)
# ═══════════════════════════════════════════════════════════
# RandomHorizontalFlip(p=0.5)
# Pad(10) + RandomCrop(256, 256)
# ColorJitter(brightness=0.2, contrast=0.15, saturation=0.1, hue=0.0)
# Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet
# RandomErasing(p=0.5, scale=(0.02, 0.33))

# ═══════════════════════════════════════════════════════════
# Other
# ═══════════════════════════════════════════════════════════
EMA = False                         # Confirmed dead end (09 v3)
FP16 = True                         # Mixed precision for P100 speed
FREEZE_GEM_P = True                 # Preserve learned GeM power from VeRi training
                                    # (unfreeze after epoch 30 if desired)
```

#### Why These Specific Values

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Backbone LR = 3e-5 | 3× lower than ViT (1e-4) | VeRi features are already strong; aggressive LR would destroy them. 09f used 3.5e-4 (too high) and degraded to 42.7% |
| Head LR = 5e-4 | Standard | Head needs to learn CityFlowV2's 128-class structure from scratch |
| Epochs = 200 | Extended | 09l v2 at 160 epochs was still improving (+5.53pp mAP in final 20 epochs); 200 gives headroom |
| Batch = 48 | P=16, K=3 | Provides enough positive pairs for triplet mining without exhausting the 128-ID pool |
| No CircleLoss | Dead end | Catastrophically unstable: inf loss in 09 v4 and 09l v1 |
| No ArcFace | Dead end | Geometry mismatch from warm-start: 09i peaked at 50.80% below the 52.77% baseline |
| No augoverhaul | Dead end for MTMC | 81.59% mAP → only 0.722 MTMC IDF1 (-5.3pp regression) |
| Freeze GeM p | Preserve | The SBS model learned an optimal p during VeRi training; freezing it prevents early-training destabilization |
| ImageNet norm | Not CLIP | R50-IBN uses ImageNet normalization (mean=[0.485,0.456,0.406]), NOT CLIP normalization |

#### Weight Initialization

```python
# Load VeRi SBS checkpoint
checkpoint = torch.load("fastreid_veri_sbs_r50ibn.pth", map_location="cpu")
# Remap fast-reid keys → our model keys (same remapping as reid_model.py)
# Skip: heads.classifier.*, backbone.NL_* (Non-local blocks not in our model)
# Map: heads.pool_layer.p → pool.p, heads.bottleneck.0.* → bottleneck.*
# Keep: backbone.* (direct)
model.load_state_dict(remapped_state_dict, strict=False)

# Reinitialize classifier head for 128 CityFlowV2 classes
nn.init.normal_(model.classifier.weight, std=0.001)
```

### Expected mAP Range

| Scenario | CityFlowV2 mAP | Probability | Reasoning |
|----------|:-:|:-:|-----------|
| Pessimistic | 50-60% | 25% | Domain gap too large, similar to 09f failure pattern despite stronger start |
| Realistic | 65-72% | 50% | SBS pretrain transfers well with gentle fine-tuning; clears ensemble threshold |
| Optimistic | 72-78% | 25% | SBS features are highly transferable; architecture captures complementary signals |

### Success Threshold
- **Minimum viable**: mAP ≥ 60% (worth testing in ensemble)
- **Ensemble-worthy**: mAP ≥ 65% (findings.md threshold for useful ensemble)
- **Strong secondary**: mAP ≥ 70% (comparable to primary, high-confidence ensemble benefit)

### Training Diagnostics to Capture
Every 20 epochs, log:
1. mAP, R1, mAP_rerank, R1_rerank on CityFlowV2 eval split
2. Per-camera mean cosine similarity matrix (diagnose camera-specific degradation)
3. Training loss components (CE, triplet, center) separately
4. Feature magnitude distribution (detect mode collapse early)

---

## Phase 2: Deploy Fine-Tuned Secondary + Ensemble Evaluation

### Gate Condition
Phase 1 produced a model with **mAP ≥ 60%** on CityFlowV2 eval split.

If mAP < 60%, skip to Phase 3 backup approaches.

### Deployment in 10a

```yaml
# In 10a notebook, enable vehicle2 with fine-tuned weights:
stage2.reid.vehicle2:
  enabled: true
  save_separate: true
  model_name: "fastreid_sbs_r50_ibn"
  weights_path: "models/reid/sbs_r50ibn_cityflow_best.pth"  # Fine-tuned checkpoint
  embedding_dim: 2048
  input_size: [256, 256]
  clip_normalization: false
```

### PCA Whitening
- Secondary features (2048D) → PCA whitened to **384D** using a SEPARATE PCA model
- Path: `models/reid/pca_transform_secondary.pkl` (already configured in the pipeline)
- The pipeline handles this automatically when `save_separate: true`

### Fusion Sweep in 10c

```python
# Sweep matrix: 8 fusion weights × 3 FIC regularization values = 24 configs
SECONDARY_WEIGHTS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
FIC_REG_SECONDARY = [0.10, 0.50, 1.00]  # May need different FIC for CNN vs ViT

# Hold primary association at v52 baseline:
# sim_thresh=0.50, appearance_weight=0.70, fic_reg=0.50, aqe_k=3
# gallery_thresh=0.48, orphan_match=0.38
```

Start with lower weights (0.10-0.20) since the secondary will likely be weaker than the primary.

### Expected Impact (If Phase 1 Succeeds)

| Secondary mAP | Expected Ensemble Gain | Confidence |
|:-:|:-:|:-:|
| 60-65% | +0.0 to +0.5pp | Low — marginal secondary, may still add noise |
| 65-70% | +0.5 to +1.5pp | Medium — clears the diversity threshold |
| 70-75% | +1.0 to +2.5pp | Medium-High — strong complementary signal |
| 75%+ | +1.5 to +3.0pp | High — approaches AIC-quality secondary |

### Diagnostic Outputs (Regardless of IDF1 Result)
1. Per-camera-pair improvement/degradation heatmap
2. Number of identity pairs where primary and secondary disagree on top-1 match
3. Distribution overlap between primary and secondary similarity scores
4. Conflation count at each fusion weight (does secondary reduce the 27 conflated IDs?)

---

## Phase 3: Backup Approaches (If Phase 1 Fails)

### 3A: Extended Primary Training (120→200 Epochs)

**Probability of helping: 20-30%**

The primary ViT was trained for only 120 epochs. The 09l sequence showed dramatic gains from extended training (61.51% at 160 epochs → 78.61% at 300 epochs). While the primary is already much stronger, it may still be schedule-limited.

**Recipe**: Same as the deployed 09b recipe (CE+LS+Triplet+Center, AdamW, cosine schedule), but:
- Resume from the 09b v2 checkpoint
- Extend to 200 total epochs
- Reset the cosine scheduler for a new 80-epoch annealing phase
- Backbone LR = 5e-5 (lower for fine-tuning phase)
- Head LR = 5e-4

**Expected outcome**: +0.5-1.5pp mAP on CityFlowV2 eval, which MAY or MAY NOT translate to better MTMC IDF1 (the augoverhaul lesson: higher mAP doesn't guarantee better MTMC).

**Key risk**: The augoverhaul regression showed that mAP improvements from training changes can hurt MTMC. Only deploy if MTMC IDF1 improves in the 10c evaluation, not just if mAP improves.

### 3B: VERI-Wild Secondary (Alternative Pretraining Data)

**Probability of helping: 15-25%**

If the VeRi-776 SBS model fails to transfer to CityFlowV2, try the VERI-Wild BoT(R50-IBN) which was trained on **40,671 IDs** (70× more than VeRi-776). The much larger training set may produce more generalizable features despite the simpler architecture.

```python
WEIGHTS_URL = "https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/veriwild_bot_R50-ibn.pth"
```

Key differences:
- No Non-local blocks, no GeM (uses standard GAP)
- Trained on Chinese highway cameras (larger domain gap than VeRi's city cameras)
- But 70× more training identities → potentially more robust features

Fine-tuning recipe: Same as Phase 1 (CE+LS+Triplet+Center, AdamW, low backbone LR).

### 3C: Cross-Camera Invariance Augmentation for Primary

**Probability of helping: 15-20%**

Instead of the failed augoverhaul (which added geometric distortions that hurt cross-camera matching), try targeted **photometric-only** augmentations that simulate cross-camera appearance changes:

```python
# Camera-transition simulation augmentations:
# - Strong brightness/contrast shifts (simulating different camera exposures)
# - Color temperature shifts (simulating different white balance)
# - NO geometric augmentations (these destroyed cross-camera features)
transforms = [
    RandomHorizontalFlip(p=0.5),
    Pad(10) + RandomCrop(256, 256),
    # Stronger photometric only:
    ColorJitter(brightness=0.4, contrast=0.3, saturation=0.2, hue=0.05),
    RandomAutoContrast(p=0.2),
    Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    RandomErasing(p=0.5, scale=(0.02, 0.33)),
]
```

This targets the actual MTMC bottleneck (cross-camera appearance shift) without the geometric distortions that confused the feature space.

### 3D: Rank-Level Fusion Instead of Score-Level

**Probability of helping: 10-15%**

If the fine-tuned secondary reaches ≥65% mAP but score-level fusion still hurts (like 10c v56), try **rank-level fusion**:

```python
# Instead of: sim = w1 * sim_primary + w2 * sim_secondary
# Try: rank_primary = argsort(sim_primary)
#      rank_secondary = argsort(sim_secondary)
#      fused_rank = rank_primary + rank_secondary
#      top_matches = argsort(fused_rank)[:top_k]
```

Rank fusion is less sensitive to feature calibration differences and may handle the CNN-vs-ViT distribution mismatch better than raw score averaging.

---

## Phase 4: Combined Deployment (If Multiple Phases Succeed)

If Phase 1 succeeds AND Phase 3A succeeds:
1. Deploy extended primary + fine-tuned R50-IBN secondary
2. Re-sweep association parameters (even though exhausted for single-model, the optimal point may shift with ensemble features)
3. Test fusion weights [0.15, 0.20, 0.25] with the improved primary

This combination addresses BOTH the feature quality gap AND the diversity gap.

---

## Questions Answered

### 1. Best training recipe for fine-tuning R50-IBN on CityFlowV2?
See Phase 1 detailed recipe above. Key: **CE+LS(0.05) + Triplet(0.3) + Center(5e-4, delayed 15 epochs)**, **AdamW**, **backbone LR=3e-5**, **head LR=5e-4**, **200 epochs**, **cosine schedule**, **baseline augmentations only**, **NO CircleLoss/ArcFace/SGD/EMA**.

### 2. What mAP do we need from the secondary?
- **Minimum viable**: 60% mAP (worth testing)
- **Ensemble-worthy**: ≥65% mAP (findings.md threshold)
- **High-confidence benefit**: ≥70% mAP
- Context: The failed 52.77% R101 secondary gave -0.1pp; the 78.61% LAION-2B secondary gave -0.5pp (but was too correlated). A 65%+ architecturally diverse secondary is unprecedented and untested.

### 3. Other high-probability approaches beyond fine-tuning?
- **Extended primary training (3A)**: 20-30% probability, +0.5-1.5pp potential
- **VERI-Wild secondary (3B)**: 15-25% probability, alternative diversity source
- **Photometric augmentation for primary (3C)**: 15-20% probability, targets cross-camera invariance
- **Rank-level fusion (3D)**: 10-15% probability, alternative fusion method

### 4. Should we attempt anything with the primary model?
Yes, but as a **backup** (Phase 3A/3C), not the primary effort. The primary is already at 80.14% mAP, and the augoverhaul lesson shows that mAP improvements don't automatically translate to MTMC IDF1. Extended training with the ORIGINAL recipe (not augoverhaul) is safer.

### 5. Execution sequence?
Phase 1 → Phase 2 → [If Phase 1 fails: Phase 3B/3C] → Phase 4 combined. See timeline below.

---

## Risk Assessment

| Risk | Impact | Probability | Mitigation |
|------|--------|:-:|-----------|
| VeRi→CityFlowV2 domain gap too large | Phase 1 fails (mAP < 60%) | 25% | Phase 3B (VERI-Wild), Phase 3A/3C (primary improvements) |
| Fine-tuned secondary still too correlated with primary | Ensemble neutral despite good secondary mAP | 15% | Architecture is genuinely diverse (CNN vs ViT); test rank fusion (3D) |
| CityFlowV2 128-ID overfitting | Secondary mAP peaks early then degrades | 30% | Save checkpoints every 20 epochs; evaluate all; use early stopping |
| Extended primary training hurts MTMC | Same as augoverhaul regression | 20% | Only deploy if MTMC IDF1 improves in 10c, not just if mAP improves |
| Kaggle GPU quota exhausted | Can't complete all phases | 10% | Prioritize Phase 1 (highest ROI); push smaller epoch runs if quota is tight |
| Fast-reid key remapping breaks | Model loads garbage | 5% | Already tested in zero-shot deployment; key mapping is verified |

### Expected Improvement Range

| Outcome | MTMC IDF1 | Probability |
|---------|:-:|:-:|
| No improvement (all phases fail) | 0.775 | 30% |
| Small gain from fine-tuned ensemble | 0.780-0.790 | 35% |
| Moderate gain from ensemble + primary improvement | 0.790-0.810 | 25% |
| Large gain approaching SOTA | 0.810-0.830 | 10% |

**Realistic expectation: 0.780-0.800 MTMC IDF1** if the fine-tuned R50-IBN reaches ≥65% mAP and ensemble fusion works.

---

## Execution Timeline

### Step 1: Build 09n Fine-Tuning Notebook (Implementation)
- Create `notebooks/kaggle/09n_finetune_sbs_r50ibn_cityflow/`
- Base on the 09l notebook structure (proven Kaggle training template)
- Load SBS R50-IBN weights with key remapping
- Configure training recipe per Phase 1 spec
- Add per-epoch evaluation on CityFlowV2 eval split
- Push to Kaggle

### Step 2: Run Phase 1 Training (~4-5 hours on Kaggle P100)
- `kaggle kernels push -p notebooks/kaggle/09n_finetune_sbs_r50ibn_cityflow/`
- Monitor with `python scripts/kaggle_logs.py <slug> --tail 50`
- Save best checkpoint as `sbs_r50ibn_cityflow_best.pth`

### Step 3: Evaluate Phase 1 Gate (mAP ≥ 60%)
- Read final mAP/R1 from Kaggle logs
- If mAP < 60%: proceed to Phase 3 backups
- If mAP ≥ 60%: proceed to Step 4

### Step 4: Update 10a for Fine-Tuned Secondary Deployment
- Enable vehicle2 with fine-tuned weights path
- Ensure correct input_size=[256,256] and clip_normalization=false
- Push 10a notebook

### Step 5: Run 10a→10b→10c Chain (~3 hours total)
- 10a: Extract primary + secondary features (GPU, ~90 min)
- 10b: Build FAISS index (CPU, ~10 min)
- 10c: Association + evaluation with fusion sweep (CPU, ~60 min)

### Step 6: Evaluate Ensemble Results
- Compare best fusion weight vs single-model baseline (0.775)
- If positive: optimize fusion weight in a focused follow-up sweep
- If negative: analyze diagnostic outputs and proceed to Phase 3D (rank fusion)

### Step 7: Phase 3 Backups (If Needed)
- 3A: Extended primary training (another 4-5 hour Kaggle run)
- 3B: VERI-Wild fine-tuning (another 4-5 hour Kaggle run)
- 3C: Photometric augmentation retrain (another 4-5 hour Kaggle run)

### Total Kaggle Time Budget
- Phase 1+2: ~8 hours (training + evaluation chain)
- Phase 3 backups: ~5 hours each
- **Total worst case**: ~23 hours across 4-5 Kaggle runs
- **Best case**: ~8 hours (Phase 1+2 succeeds)

---

## Files to Create/Modify

### New Files
1. `notebooks/kaggle/09n_finetune_sbs_r50ibn_cityflow/09n_finetune_sbs_r50ibn_cityflow.ipynb` — Training notebook
2. `notebooks/kaggle/09n_finetune_sbs_r50ibn_cityflow/kernel-metadata.json` — Kaggle metadata

### Files to Modify
3. `notebooks/kaggle/10a_stages012/10a_stages012.ipynb` — Add fine-tuned secondary weights download + extraction
4. `notebooks/kaggle/10c_stages45/10c_stages45.ipynb` — Add fusion weight sweep matrix
5. `configs/datasets/cityflowv2.yaml` — Update vehicle2 weights_path for fine-tuned model

### No Changes Needed
- `src/training/model.py` — ReIDModelResNet50IBN already exists
- `src/stage2_features/reid_model.py` — fastreid_sbs_r50_ibn dispatcher already exists
- `configs/default.yaml` — vehicle2 config section already exists

---

## Lessons from Previous Failures (DO NOT REPEAT)

| Failure | Lesson | How This Plan Avoids It |
|---------|--------|------------------------|
| 09f: VeRi→CityFlowV2 R101 gave only 42.7% mAP | Weak VeRi pretrain (62.52%) + aggressive LR (3.5e-4) destroyed features | SBS starts at 81.9% mAP; backbone LR is 3e-5 (10× lower) |
| 09i: ArcFace warm-start from CE = 50.80% | Geometry mismatch between CE and angular-margin objectives | Using CE+Triplet throughout — no objective switch |
| 10c v56: LAION-2B CLIP fusion = -0.5pp | Two CLIP ViT-B/16 models are too correlated | R50-IBN is a CNN with GeM+NL+circle-loss init — maximally diverse |
| Augoverhaul: +1.45pp mAP → -5.3pp MTMC IDF1 | Geometric augmentations hurt cross-camera features | Using baseline augmentations only |
| 09 v4 / 09l v1: CircleLoss = inf loss | CircleLoss(gamma=128) overflows in fp16 | Not using CircleLoss at all |
| 09d v3: Extended R101 fine-tuning = 50.61% | Over-training on small dataset with aggressive LR | Using cosine schedule with gentle decay and 200 epochs (not resuming with lower LR) |

---

## Reference Artifacts

- `docs/findings.md` — Full dead-end catalog and performance history
- `docs/subagent-specs/pretrained-ensemble-v2.md` — Original zero-shot ensemble spec
- `src/training/model.py` — ReIDModelResNet50IBN class (already implemented)
- `src/stage2_features/reid_model.py` — fastreid_sbs_r50_ibn dispatcher (already implemented)
- `configs/default.yaml` — vehicle2 config section
- `configs/datasets/cityflowv2.yaml` — CityFlowV2 pipeline overrides
- `notebooks/kaggle/09l_transreid_laion2b/` — Successful extended training reference
