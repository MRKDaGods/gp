# Next Steps Plan — Vehicle MTMC Toward SOTA

> **Created**: 2026-04-20 | **Status**: ACTIVE
> **Current Best MTMC IDF1**: 0.7736 (10c v60, w=0.10 fusion with R50-IBN)
> **Baseline (primary only)**: 0.7730 (10c v60, w=0.00)
> **SOTA Target**: 0.8486 (AIC22 1st place, 5-model ensemble)
> **Gap**: 7.5pp

---

## Situation Assessment

### What's Running on Kaggle
| Kernel | Task | Expected mAP | ETA |
|--------|------|:---:|-----|
| **09o** | EVA02 ViT-B/16 MIM+CLIP on CityFlowV2 | 75-82% | Running now |
| **09p** | Extended R50-IBN (200 more epochs from 63.64%) | 68-73% | Running now |
| **09q** | Extended primary ViT (120 more epochs from 80.14%) | 81-84% | NEEDS push (GPU quota full) |

### Current Model Inventory
| Model | Architecture | mAP | Role | Status |
|-------|-------------|:---:|------|--------|
| TransReID ViT-B/16 CLIP (OpenAI) | ViT + CLS | 80.14% | Primary | ✅ Deployed |
| FastReID SBS R50-IBN (fine-tuned) | CNN + GeM + NL | 63.64% | Secondary | ✅ Tested: +0.06pp at w=0.10 |
| EVA02 ViT-B/16 MIM+CLIP | ViT + RoPE | TBD | Tertiary candidate | 🔄 Training (09o) |

### Key Findings From 10c v60
- R50-IBN fusion gain is **negligible** (+0.06pp at w=0.10)
- Higher weights actively hurt: w=0.15 → -0.18pp, w=0.50 → -1.05pp
- A useful secondary needs **≥70% mAP** or **fundamentally different architecture** (e.g., EVA02)

---

## Phase 1: Immediate Actions (While Training Runs)

### 1A. TTA Feature Extraction Testing (HIGHEST PRIORITY)

**Why now**: TTA is already implemented in `reid_model.py`, default-off. Zero training cost. Can test on existing 10a features by re-running 10a with TTA enabled, then 10b→10c.

**Implementation**: Modify 10a notebook config overrides to enable TTA:
```yaml
# Center-crop scale TTA (NEW, never tested)
stage2.reid.center_crop_scales=[0.9, 1.1]
stage2.reid.normalize_views=true
```

**TTA Sweep Matrix** (run as separate 10a→10b→10c chains):

| Experiment | Config | Views | Expected Impact |
|------------|--------|:-----:|:---:|
| TTA-1: Center-crop only | `center_crop_scales=[0.9,1.1]`, `normalize_views=true` | 6 (orig+flip × 3 scales) | +0.0 to +0.5pp |
| TTA-2: Multiscale only | `multiscale_sizes=[[224,224],[288,288]]`, `normalize_views=true` | 6 | +0.0 to +0.5pp |
| TTA-3: Combined | Both center-crop + multiscale, `normalize_views=true` | 10 | +0.0 to +0.5pp |

**10a Config Overrides for TTA-1**:
```
stage2.reid.center_crop_scales=[0.9,1.1]
stage2.reid.normalize_views=true
```

**10c Config**: Keep association at baseline recipe:
```
stage4.association.graph.similarity_threshold=0.50
stage4.association.weights.vehicle.appearance=0.70
stage4.association.fic.regularisation=0.50
stage4.association.query_expansion.k=3
stage4.association.gallery_expansion.threshold=0.48
stage4.association.gallery_expansion.orphan_match_threshold=0.38
```

**Runtime Warning**: TTA-1 triples feature extraction time (~3× more forward passes). TTA-3 quintuples it. On Kaggle P100, 10a may approach the 12h time limit with TTA-3. Start with TTA-1.

**Success Criteria**: MTMC IDF1 > 0.7750 (any positive gain over 0.7730 baseline)

### 1B. Camera-Pair Specific Thresholds (LOW PRIORITY)

**Why**: Already implemented in `pipeline.py` Step 5e but never activated. FIC whitening likely already handles distribution alignment, so expected gain is marginal (+0.0 to +0.3pp). Still, zero implementation cost.

**Config** (add to 10c overrides):
```yaml
stage4.association.pair_thresholds.enabled=true
stage4.association.pair_thresholds.thresholds:
  # Same-scene pairs: higher threshold (easier match, same viewpoint domain)
  S01_c001-S01_c002: 0.52
  S01_c001-S01_c003: 0.52
  S01_c002-S01_c003: 0.52
  S02_c006-S02_c007: 0.52
  S02_c006-S02_c008: 0.52
  S02_c007-S02_c008: 0.52
  # Cross-scene pairs are already blocked by spatio-temporal constraints
```

**Alternative approach**: Run 10c with `camera_pair_norm.enabled=true` first (simpler — distribution centering). If that helps, then try per-pair thresholds.

**Test Order**:
1. `camera_pair_norm.enabled=true, min_pairs=10` (simplest)
2. Per-pair thresholds (if norm helps)

**Success Criteria**: MTMC IDF1 > 0.7750

### 1C. Push 09q to Kaggle (When GPU Quota Opens)

**Why**: Extended primary ViT training is the single highest-leverage action. The current 80.14% mAP was trained for only 120 epochs with cosine annealing, and the 09l sequence showed LAION-2B CLIP gaining +17pp mAP from extended training.

**Action**: Push `notebooks/kaggle/09q_transreid_extended/` as soon as a GPU slot opens.

**Critical Constraint**: Even if mAP increases, MTMC IDF1 may NOT improve (augoverhaul lesson: +1.45pp mAP → -5.3pp MTMC IDF1). Must validate via full 10a→10b→10c chain before declaring success.

---

## Phase 2: When 09o Completes (EVA02)

### Scenario A: EVA02 mAP ≥ 75%

This is the **breakthrough scenario**. EVA02 uses MIM+CLIP pretraining (decorrelated from OpenAI CLIP), which should produce genuinely diverse features.

**Deployment Steps**:

1. **Download 09o checkpoint** to Kaggle datasets
2. **Update 10a notebook** to enable `vehicle2` (or `vehicle3`) with EVA02:
   ```yaml
   stage2.reid.vehicle2.enabled=true
   stage2.reid.vehicle2.model_name=eva02_vit
   stage2.reid.vehicle2.weights_path=/kaggle/input/09o-model/eva02_cityflowv2_best.pth
   stage2.reid.vehicle2.embedding_dim=768
   stage2.reid.vehicle2.input_size=[256,256]
   stage2.reid.vehicle2.vit_model=eva02_base_patch16_clip_224.merged2b
   stage2.reid.vehicle2.clip_normalization=false
   stage2.reid.vehicle2.save_separate=true
   ```
3. **Run 10a→10b→10c** with score-level fusion sweep:
   ```
   # Sweep EVA02 fusion weight
   stage4.association.secondary_embeddings.weight=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
   ```
4. **If 2-model fusion works**: Test 3-model fusion (primary + R50-IBN + EVA02):
   ```yaml
   stage2.reid.vehicle2.enabled=true   # EVA02
   stage2.reid.vehicle3.enabled=true   # R50-IBN
   stage4.association.secondary_embeddings.weight=0.15  # EVA02
   stage4.association.tertiary_embeddings.weight=0.10   # R50-IBN
   ```

**Important**: The EVA02 model requires a custom `EVA02ReID` wrapper class (see 09o spec) that handles RoPE. The `reid_model.py` dispatcher must be updated to route `eva02_vit` to this new builder. The 09o notebook trains with this wrapper, but the inference code in `reid_model.py` needs a corresponding `_build_eva02_transreid()` method.

**Code Changes Required**:
- `src/stage2_features/reid_model.py`: Add `"eva02_vit"` to `_TRANSREID_NAMES`, add `_build_eva02_transreid()` method
- `src/stage2_features/transreid_model.py`: Add EVA02ReID inference class (or import from training code)

**Expected Impact**: +1.0 to +3.0pp MTMC IDF1 from decorrelated 2-model ensemble

### Scenario B: EVA02 mAP < 75%

EVA02 is too weak for meaningful fusion (same problem as R50-IBN at 63.64%).

**Actions**:
- Check if training was schedule-limited (still improving at final epoch?)
- If yes: extend training (09o v2 with resumed schedule, 200→400 epochs)
- If no: EVA02 is a dead end for CityFlowV2. Move to DFN-2B or other CLIP variants
- Re-prioritize: focus on 09p/09q results instead

---

## Phase 3: When 09p Completes (Extended R50-IBN)

### Scenario A: R50-IBN mAP ≥ 70%

**Actions**:
1. Retry fusion at higher weight than the 10c v60 sweep:
   ```
   # Higher weights may now be viable at 70%+ mAP
   stage4.association.secondary_embeddings.weight=[0.10, 0.15, 0.20, 0.25]
   ```
2. The 10c v60 showed w=0.15 hurt at 63.64% mAP. At 70%+ mAP, w=0.15-0.20 might be the new sweet spot.

**Expected Impact**: +0.3 to +1.0pp over primary-only baseline (up from the negligible +0.06pp at 63.64%)

### Scenario B: R50-IBN mAP < 68%

Marginal improvement. Not worth re-running the full pipeline chain.

**Actions**:
- Check if still improving at final epoch. If yes, extend further (09p v2).
- If plateaued: R50-IBN is ceiling-limited at ~65-68% on CityFlowV2. Accept this and focus on EVA02 and primary extension.
- R50-IBN at <68% is only useful in a 3+ model ensemble where even small diversity helps, not in 2-model score fusion.

---

## Phase 4: When 09q Completes (Extended Primary ViT)

### Deployment

1. **Download 09q checkpoint** to Kaggle datasets
2. **Update 10a** to use the new primary weights:
   ```yaml
   stage2.reid.vehicle.weights_path=/kaggle/input/09q-model/transreid_cityflowv2_extended_best.pth
   ```
3. **Run full 10a→10b→10c** with baseline association recipe (same as Phase 1A config)
4. **Compare MTMC IDF1** against the 0.7730 baseline

**Critical Warning**: Higher mAP does NOT guarantee better MTMC IDF1. The augoverhaul model had +1.45pp mAP but -5.3pp MTMC IDF1. Only declare success if MTMC IDF1 improves.

**If MTMC improves**: Re-run all fusion experiments (TTA, R50-IBN, EVA02) on top of the new primary. The improved primary may also make secondary fusion more effective.

**If MTMC regresses**: The extended training changed the feature geometry in a way that hurts cross-camera matching (like augoverhaul). Discard the extended checkpoint and keep the original 80.14% model.

**Expected Impact**: +0.0 to +1.0pp MTMC IDF1 (25% probability of positive gain — cautious given augoverhaul precedent)

---

## Experiment Priority Order

### Priority Matrix

| Priority | Experiment | Dependencies | Expected Gain | Risk |
|:--------:|-----------|-------------|:---:|:---:|
| **P0** | TTA-1 (center-crop scales) | None — run NOW | +0.0 to +0.5pp | LOW |
| **P1** | Push 09q when GPU opens | GPU quota | +0.0 to +1.0pp | MED |
| **P2** | Camera-pair norm test | None — run NOW | +0.0 to +0.3pp | LOW |
| **P3** | Deploy EVA02 (when 09o done) | 09o completion + code changes | +1.0 to +3.0pp | MED |
| **P4** | Retry R50-IBN fusion (when 09p done) | 09p completion | +0.0 to +1.0pp | LOW |
| **P5** | Deploy extended primary (when 09q done) | 09q completion | +0.0 to +1.0pp | MED |
| **P6** | 3-model ensemble (EVA02 + R50-IBN + primary) | P3 + P4 | +1.5 to +3.5pp | MED |
| **P7** | Embedding concat + re-PCA (if score fusion plateaus) | Multiple models | +0.0 to +0.5pp | LOW |

### Execution Timeline

```
NOW ─────────────────────────────────────────────────────────────
 │
 ├── [P0] Push TTA-1 10a notebook to Kaggle
 ├── [P2] Push camera_pair_norm 10c experiment
 │
 ├── Wait for 09o/09p completion...
 │
WHEN GPU OPENS ──────────────────────────────────────────────────
 │
 ├── [P1] Push 09q extended primary ViT training
 │
WHEN 09o COMPLETES ──────────────────────────────────────────────
 │
 ├── [P3] Add EVA02 builder to reid_model.py
 ├── [P3] Deploy EVA02 in 10a, run fusion sweep in 10c
 │
WHEN 09p COMPLETES ──────────────────────────────────────────────
 │
 ├── [P4] If mAP ≥ 70%: retry fusion at w=0.15-0.20
 │
WHEN 09q COMPLETES ──────────────────────────────────────────────
 │
 ├── [P5] Deploy extended primary, run baseline eval
 ├── If P5 improves: re-run P0/P3/P4 on new primary
 │
WHEN P3+P4+P5 ALL DONE ─────────────────────────────────────────
 │
 ├── [P6] 3-model ensemble: primary + EVA02 + R50-IBN
 ├── [P7] If score fusion plateaus: try embedding concat + re-PCA
```

---

## Projected Outcomes

| Scenario | MTMC IDF1 | Probability | Key Requirement |
|----------|:---------:|:-----------:|-----------------|
| No improvement from any action | 0.773 | 15% | All models too correlated or too weak |
| TTA + camera-pair gains only | 0.775-0.780 | 25% | Feature extraction improvements without model changes |
| EVA02 ensemble works | 0.790-0.810 | 35% | EVA02 ≥75% mAP AND decorrelated features |
| Extended primary + EVA02 ensemble | 0.800-0.820 | 20% | Both 09q and 09o produce MTMC gains |
| Near-SOTA breakthrough | 0.820-0.840 | 5% | 3+ model ensemble with strong individuals + optimal fusion |

**Realistic expectation**: 0.785-0.810 if EVA02 delivers decorrelated features above 75% mAP.

---

## Specific 10a/10c Config Changes Reference

### 10a Notebook Config Overrides

**For TTA-1 experiment**:
```python
overrides = [
    "stage2.reid.center_crop_scales=[0.9,1.1]",
    "stage2.reid.normalize_views=true",
]
```

**For EVA02 deployment** (after 09o):
```python
overrides = [
    "stage2.reid.vehicle2.enabled=true",
    "stage2.reid.vehicle2.model_name=eva02_vit",
    "stage2.reid.vehicle2.weights_path=/kaggle/input/09o-eva02-model/eva02_cityflowv2_best.pth",
    "stage2.reid.vehicle2.embedding_dim=768",
    "stage2.reid.vehicle2.input_size=[256,256]",
    "stage2.reid.vehicle2.vit_model=eva02_base_patch16_clip_224.merged2b",
    "stage2.reid.vehicle2.clip_normalization=false",
    "stage2.reid.vehicle2.save_separate=true",
]
```

**For extended primary** (after 09q):
```python
overrides = [
    "stage2.reid.vehicle.weights_path=/kaggle/input/09q-extended-model/transreid_cityflowv2_extended_best.pth",
]
```

### 10c Notebook Config Overrides

**Baseline association recipe** (use for all feature experiments):
```python
overrides = [
    "stage4.association.graph.similarity_threshold=0.50",
    "stage4.association.weights.vehicle.appearance=0.70",
    "stage4.association.fic.regularisation=0.50",
    "stage4.association.query_expansion.k=3",
    "stage4.association.gallery_expansion.threshold=0.48",
    "stage4.association.gallery_expansion.orphan_match_threshold=0.38",
]
```

**For camera-pair norm test** (add to baseline):
```python
overrides += [
    "stage4.association.camera_pair_norm.enabled=true",
    "stage4.association.camera_pair_norm.min_pairs=10",
]
```

**For EVA02 fusion sweep** (add to baseline):
```python
# Sweep secondary weight
for w in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
    overrides_w = overrides + [
        f"stage4.association.secondary_embeddings.weight={w}",
        "stage4.association.secondary_embeddings.path=data/outputs/stage2/embeddings_secondary.npy",
    ]
```

**For 3-model ensemble** (add to baseline):
```python
overrides += [
    "stage4.association.secondary_embeddings.path=data/outputs/stage2/embeddings_secondary.npy",
    "stage4.association.secondary_embeddings.weight=0.15",
    "stage4.association.tertiary_embeddings.path=data/outputs/stage2/embeddings_tertiary.npy",
    "stage4.association.tertiary_embeddings.weight=0.10",
]
```

---

## Dead Ends — DO NOT RETRY

These are confirmed from `docs/findings.md` and copied here for reference:
- **Score-level fusion of two CLIP ViTs** (-0.5pp, too correlated)
- **R50-IBN at 63.64% fusion** (+0.06pp, negligible)
- **CID_BIAS** (both GT-learned and topology variants: -1.0 to -3.3pp)
- **SAM2 foreground masking** (-8.7pp)
- **AFLink** (-3.8 to -13.2pp)
- **CircleLoss** (inf loss, catastrophic)
- **384px deployment** (-2.8pp)
- **Augoverhaul** (+1.45pp mAP → -5.3pp MTMC IDF1)
- **Network flow solver** (-0.24pp, increased conflation)
- See `docs/findings.md` for the complete list of 25+ confirmed dead ends
