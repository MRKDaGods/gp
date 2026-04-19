# SOTA Breakthrough Strategy — Vehicle MTMC IDF1

## Status: ACTIVE

## Executive Summary

Comprehensive strategy to close the **7.36pp gap** from our current best (**MTMC IDF1 = 0.775**) toward SOTA (**0.8486**). The primary path is **multi-model ensemble with diverse architectures**, backed by **primary model extension** and **alternative fusion methods**. Association tuning is exhausted (225+ configs); all gains must come from feature quality and ensemble diversity.

**Current state (2026-04-19):**
| Component | Value | Notes |
|-----------|-------|-------|
| Primary model | TransReID ViT-B/16 CLIP 256px | mAP=80.14%, R1=92.27% |
| Secondary model | fast-reid SBS R50-IBN (fine-tuned) | mAP=63.64%, R1=78.69% — **just completed** |
| Best MTMC IDF1 | 0.775 | 10c v52, v80-restored recipe |
| Association | EXHAUSTED | 225+ configs, all within 0.3pp |
| SOTA | 0.8486 | AIC22 1st place (5-model ensemble) |

**Honest probability assessment:**
| Outcome | MTMC IDF1 | Probability |
|---------|:---------:|:-----------:|
| No improvement | 0.775 | 20% |
| Small ensemble gain | 0.780-0.795 | 40% |
| Moderate multi-model gain | 0.795-0.815 | 30% |
| Near-SOTA breakthrough | 0.815-0.840 | 10% |

**Realistic expectation: 0.790-0.810** if fine-tuned R50-IBN ensemble works and we add 1-2 more diverse models.

---

## Part 1: Immediate Actions (Can Start Now)

### 1A. Deploy Fine-Tuned R50-IBN Ensemble (HIGHEST PRIORITY)

**Goal**: Test whether the 63.64% mAP R50-IBN improves MTMC via score-level fusion with the primary ViT.

**Why this is the #1 action:**
- R50-IBN at 63.64% is the **first non-CLIP secondary to approach the 65% ensemble threshold**
- It is **genuinely architecturally diverse**: CNN + GeM pooling + NL blocks + circle-loss-pretrained vs ViT + CLS token + CLIP init + triplet-trained
- The 10c v56 LAION-2B fusion failed because both models were CLIP ViT-B/16 (too correlated). R50-IBN is maximally different.
- Score-level fusion with the weak 52.77% R101 gave -0.1pp; with a strong 78.61% CLIP ViT gave -0.5pp (correlation). R50-IBN at 63.64% is in new territory: weaker individually but architecturally diverse.

**Fusion sweep config:**
```python
SECONDARY_WEIGHTS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
# Hold primary association at v52 baseline:
# sim_thresh=0.50, appearance_weight=0.70, fic_reg=0.50, aqe_k=3
# gallery_thresh=0.48, orphan_match=0.38
```

**Expected impact:** +0.0 to +1.5pp MTMC IDF1 (50% probability of positive gain)

**Execution:**
1. Update 10a to enable vehicle2 with fine-tuned R50-IBN weights
2. Run 10a → 10b → 10c chain on Kaggle
3. Sweep fusion weights from 0.05 to 0.30
4. Capture per-camera-pair improvement/degradation heatmap
5. Check if conflation count drops (currently 27 conflated predicted IDs)

### 1B. Push R50-IBN Training Higher (Parallel)

**Goal**: Push R50-IBN from 63.64% to 70%+ mAP with extended training.

**Key insight from 09l sequence**: LAION-2B CLIP went from 61.51% (160ep) → 78.61% (300ep), gaining +17pp from extended training alone. The R50-IBN at 63.64% may be similarly schedule-limited.

**Recipe adjustments to try:**
1. **Extended training (200→400 epochs)**: Resume from best checkpoint with a fresh cosine schedule
   - Backbone LR = 1e-5 (10× lower for fine-tuning phase)
   - Head LR = 2e-4
   - Cosine annealing over 200 new epochs
2. **Warmup restart**: Reset LR to warmup levels, re-anneal over 200 epochs
3. **Stronger photometric augmentation** (NOT geometric — augoverhaul lesson):
   ```python
   ColorJitter(brightness=0.3, contrast=0.25, saturation=0.15, hue=0.03)
   RandomAutoContrast(p=0.15)
   # NO RandomGrayscale, GaussianBlur, RandomPerspective (these killed MTMC IDF1)
   ```
4. **Increased batch diversity**: Try P=24 K=2 (more identities, fewer samples) vs current P=16 K=3

**DO NOT try:**
- CircleLoss (catastrophic in fp16, dead end)
- ArcFace (geometry mismatch on warm-start, dead end)
- SGD (catastrophic for this data regime, dead end)
- Heavy geometric augmentation (augoverhaul killed MTMC IDF1)
- EMA (neutral at best, dead end per 09 v3)

**Expected impact:** +3-7pp mAP on CityFlowV2 eval, translating to +0.5-1.0pp additional MTMC IDF1 over the 63.64% baseline ensemble

### 1C. Alternative Fusion Methods (After 1A Results)

If score-level fusion with R50-IBN is neutral or negative despite diverse architectures:

**Rank-level fusion:**
```python
# Instead of: sim = w1 * sim_primary + w2 * sim_secondary
# Try: rank_primary = argsort(argsort(sim_primary))  # rank matrix
#      rank_secondary = argsort(argsort(sim_secondary))
#      fused_rank = rank_primary + rank_secondary
#      Re-threshold on rank values
```
Rank fusion is less sensitive to calibration differences between CNN and ViT similarity distributions.

**Embedding-level fusion (concatenation + re-whitening):**
```python
# Concatenate PCA-whitened embeddings: 384D (ViT) + 384D (R50) = 768D
# Apply joint PCA whitening to 384D
# Then use single FIC + AQE pipeline
```
Previous `concat_patch` was raw 1536D without re-whitening; this is a fundamentally different approach.

**Expected impact:** +0.0 to +0.5pp over best score-level fusion

---

## Part 2: Primary Model Improvement (The Biggest Lever)

### 2A. Extended Primary ViT Training (120→240+ Epochs)

**Goal:** Push ViT-B/16 CLIP from 80.14% to 83%+ mAP while preserving MTMC quality.

**Evidence this could work:**
- 09l showed LAION-2B CLIP gaining +17pp from 160→300 epochs
- The primary was trained for only 120 epochs with cosine annealing
- mAP=80.14% is strong but the model may be schedule-limited, not capacity-limited

**Recipe:**
- Resume from 09b v2 checkpoint (80.14% mAP, the deployed model)
- Fresh cosine schedule over 120 new epochs (total: 240)
- Backbone LR = 5e-5 (half of original 1e-4)
- Head LR = 5e-4 (same as original)
- LLRD = 0.75 (same as original)
- **Baseline augmentations ONLY** (not augoverhaul — that killed MTMC)
- CE + LS(0.05) + Triplet(0.3) + Center(5e-4) (proven stable recipe)

**Critical constraint:** Higher mAP does NOT guarantee better MTMC IDF1 (augoverhaul lesson: +1.45pp mAP → -5.3pp MTMC IDF1). Only deploy the extended model if a full 10c evaluation shows MTMC improvement.

**Expected impact:** +0.5 to +2.0pp mAP, translating to +0.0 to +1.0pp MTMC IDF1 (25% probability of MTMC gain)

### 2B. What SOTA Actually Uses (and Can We Replicate)

**AIC22 1st place (IDF1=0.8486) — He et al.:**
- 5-model ReID ensemble:
  - TransReID ViT-B/16 (384px) × 2 variants
  - ResNet101-IBN-a
  - ResNeXt101-IBN-a
  - SE-ResNet101-IBN-a
- Training: CE + Triplet + Center + Circle (ID-specific)
- Camera-aware training (DMT) on each model
- Box-Grained Matching for spatial reasoning
- All models trained on VeRi-776 → fine-tuned on CityFlowV2

**What we can replicate:**
- ✅ TransReID ViT-B/16 (primary, already at 80.14%)
- ✅ R50-IBN CNN (secondary, 63.64% and improving)
- ⚠️ DMT camera-aware training (tested: -1.4pp in single-model; may work in ensemble)
- ❌ Box-Grained Matching (not implemented)
- ❌ SE-ResNet101-IBN-a (no pretrained vehicle weights available)

**What we cannot replicate:**
- 384px input (dead end: -2.8pp MTMC IDF1)
- Circle loss (catastrophic in our recipe)
- ResNeXt101-IBN-a (pretrained weight mismatch, dead end)

**Key insight:** SOTA's techniques (DMT, circle loss, 384px) work because they operate in a **5-model ensemble regime** where noise is averaged out. In our 1-2 model regime, these same techniques are harmful. We need to build up the model count first.

---

## Part 3: Multi-Model Ensemble (The SOTA Approach)

### The Diversity Problem

Two CLIP ViT-B/16 models (OpenAI + LAION-2B) are too correlated for useful fusion (10c v56: -0.5pp). We need models that are:
1. **Architecturally diverse** (CNN vs ViT)
2. **Pretrain-diverse** (different datasets, different objectives)
3. **Individually strong** (≥65% mAP on CityFlowV2)

### Candidate Models for 3-5 Model Ensemble

**Tier 1: Ready or near-ready (high confidence)**

| Model | Architecture | Pretrain | CityFlowV2 mAP | Status |
|-------|-------------|----------|:-:|--------|
| TransReID ViT-B/16 CLIP (OpenAI) | ViT + CLS | CLIP 400M → VeRi-776 → CityFlowV2 | 80.14% | ✅ Deployed |
| fast-reid SBS R50-IBN | CNN + GeM + NL | VeRi-776 (81.9%) → CityFlowV2 | 63.64% | ✅ Just completed; push to 70%+ |

**Tier 2: High-probability candidates (need training)**

| Model | Architecture | Pretrain Source | Expected mAP | Rationale |
|-------|-------------|----------------|:-:|-----------|
| fast-reid BoT R50-IBN (VERI-Wild) | CNN + GAP | VERI-Wild (40,671 IDs) | 55-70% | 70× more pretraining IDs than VeRi-776; different camera domain (Chinese highways). Different from SBS model (no GeM, no NL blocks). Available at `fast-reid` GitHub releases. |
| EVA02 ViT-B/16 CLIP | ViT + CLS | LAION-2B + EVA pretrain | 75-82% | Different vision pretraining (masked image modeling + CLIP), same inference architecture. Compatible with existing TransReID code — just change `vit_model` string. Gives ViT diversity without CNN. |
| DFN-2B ViT-B/16 CLIP | ViT + CLS | DFN-filtered LAION-2B | 70-80% | Apple's Data Filtering Network selects cleaner training data. Already spec'd in `09m-dfn2b-training.md`. Compatible with TransReID code. |

**Tier 3: Speculative candidates (lower confidence)**

| Model | Architecture | Pretrain Source | Expected mAP | Risk |
|-------|-------------|----------------|:-:|------|
| SigLIP ViT-B/16 | ViT | WebLI (10B images) | 70-80% | Different training objective (sigmoid vs softmax); may lack `cls_token` (check architecture). Google's dataset may give different feature biases. |
| fast-reid SBS R101-IBN | CNN + GeM + NL | VeRi-776 | 65-75% | If available in fast-reid releases. Deeper CNN = different error profile from R50. But may be too similar to R50-IBN. |
| ConvNeXt-V2 Base CLIP | CNN (modern) | LAION-2B CLIP | 65-75% | Modern CNN with CLIP init. But **requires code changes** — not a ViT, so SIE/JPM won't work. Needs separate ReID head design. HIGH EFFORT. |

### Recommended 5-Model Target

| Slot | Model | Diversity Type | Priority |
|------|-------|---------------|:--------:|
| 1 | TransReID ViT-B/16 CLIP (OpenAI) | ViT + CLIP | ✅ Done |
| 2 | fast-reid SBS R50-IBN (CityFlowV2 fine-tuned) | CNN + VeRi pretrain | ✅ Done (push higher) |
| 3 | EVA02 ViT-B/16 CLIP | ViT + MIM+CLIP pretrain | **Train next** |
| 4 | fast-reid BoT R50-IBN (VERI-Wild fine-tuned) | CNN + Wild pretrain | Train after slot 3 |
| 5 | DFN-2B ViT-B/16 CLIP | ViT + filtered CLIP pretrain | Train after slot 4 |

**Why this ensemble works:**
- 2 CNN models (R50-IBN variants with different pretrains): capture texture and local patterns
- 3 ViT models (OpenAI/EVA02/DFN-2B CLIP): capture global structure and semantic features
- Each model has different pretraining data → different error profiles
- EVA02 uses masked image modeling → learns different representations than contrastive CLIP
- VERI-Wild has 70× more vehicle IDs → better generalization for rare vehicle types

### Fusion Strategy

**Phase 1 (2 models):** Score-level fusion with weight sweep
```python
sim = w * sim_primary + (1-w) * sim_secondary
# w ∈ [0.70, 0.75, 0.80, 0.85, 0.90]
```

**Phase 2 (3+ models):** Weighted score fusion with per-model calibration
```python
# Each model's similarity matrix is FIC-whitened independently
# Then fused with learned/tuned weights
sim = w1 * fic(sim1) + w2 * fic(sim2) + w3 * fic(sim3)
# Constraint: w1 + w2 + w3 = 1
# Sweep: w1 ∈ [0.40-0.60], w2 ∈ [0.15-0.30], w3 ∈ [0.10-0.25]
```

**Phase 3 (if score-level insufficient):** Rank-level fusion
```python
# Convert each model's similarity matrix to rank matrix
# Sum ranks, threshold on summed ranks
# Less sensitive to calibration differences
```

**Phase 4 (if rank-level insufficient):** Embedding concatenation + re-PCA
```python
# Concatenate L2-normalized embeddings from all models
# Apply joint PCA to shared dimension (384D or 512D)
# Single FIC + AQE on the fused space
```

---

## Part 4: Novel Approaches Not Yet Tried

### 4A. Test-Time Augmentation (TTA) for Feature Extraction

**Status:** Flip augmentation is already used. Multi-scale TTA was tested and neutral/harmful.

**Untried variant — Random Crop TTA:**
```python
# Instead of single center crop:
# 1. Center crop (standard)
# 2. 4 corner crops (5-crop)
# 3. Average embeddings
# This captures different spatial regions of the vehicle
```

**Expected impact:** +0.0 to +0.5pp
**Risk:** Increases inference time by 5×; may add noise rather than signal
**Priority:** LOW — try only after ensemble is optimized

### 4B. GNN Edge Classification for Association

**Status:** Not implemented. Requires significant new code.

**Concept:** Replace the fixed similarity threshold + connected components with a GNN that learns to classify edges (same-ID vs different-ID) in the similarity graph.

**Architecture:**
```
Input: Pairwise similarity features (appearance, spatiotemporal, camera-pair statistics)
GNN: 2-3 layers of message passing
Output: Edge probability (same vehicle / different vehicle)
Training: Use CityFlowV2 GT associations as supervision
```

**Why it might work:** The current CC approach uses a single global threshold. A GNN can learn camera-pair-specific and context-dependent thresholds, potentially reducing both conflation (27 IDs) and fragmentation (87 IDs).

**Why it might not work:** Only 128 training IDs; GNN may overfit. Also, this is essentially relearning what FIC already does.

**Expected impact:** +1.0 to +3.0pp (if it works)
**Risk:** HIGH effort, HIGH risk of overfitting on 128 IDs
**Priority:** MEDIUM — pursue in parallel with ensemble if resources allow

### 4C. Box-Grained Matching (from AIC22 1st Place)

**Concept:** Instead of matching whole-vehicle embeddings, extract embeddings from sub-regions (upper/lower half, left/right quarter) and match at the part level.

**Why it might work:** Different cameras see different parts of a vehicle. Part-level matching can find correspondence even when overall appearance changes due to viewpoint.

**Implementation:**
1. Split each vehicle crop into a 2×2 or 3×1 grid
2. Extract ReID embeddings per part
3. Match using best-part correspondence (not just global similarity)

**Expected impact:** +0.5 to +2.0pp
**Risk:** MEDIUM — requires new feature extraction code but not a new model
**Priority:** MEDIUM — try after ensemble is established

### 4D. Camera-Pair Specific Thresholds (Learned)

**Status:** CID_BIAS (additive) is a dead end. But MULTIPLICATIVE camera-pair thresholds have NOT been tested.

**Concept:** Instead of a single `sim_thresh=0.50` for all camera pairs, learn separate thresholds per camera pair from GT:
```python
# Example: cameras facing same direction need higher threshold (easier match)
# Cameras at 90° angle need lower threshold (harder match)
thresholds = {
    ("c001", "c002"): 0.55,  # Adjacent cameras, similar angle
    ("c001", "c006"): 0.42,  # Cross-intersection, very different angle
    # etc.
}
```

**Why different from CID_BIAS:** CID_BIAS adds an offset to raw similarity scores. This sets different CC thresholds per pair, which is mathematically distinct — it changes the graph structure rather than the edge weights.

**Expected impact:** +0.0 to +1.0pp
**Risk:** LOW effort, LOW risk (easy to revert)
**Priority:** MEDIUM — worth a quick experiment

### 4E. SAM2 Foreground Masking

**Status:** CONFIRMED DEAD END (-8.7pp MTMC IDF1). DO NOT RETRY.

### 4F. Temporal Ensemble (Feature Averaging Over Time)

**Concept:** For each tracklet, instead of quality-weighted temporal pooling, try:
1. Extract features from ALL frames (not just sampled)
2. Apply temporal attention (not just quality weighting)
3. Use LSTM/Transformer to aggregate temporal features

**Expected impact:** +0.0 to +0.5pp
**Risk:** MEDIUM effort; may not help if the current quality-weighted pooling is already near-optimal
**Priority:** LOW

---

## Part 5: Concrete Action Plan (Priority-Ordered)

### Phase A: Quick Wins (This Week)

| # | Action | Expected Gain | Effort | Dependencies |
|---|--------|:---:|:---:|-------------|
| A1 | Deploy fine-tuned R50-IBN ensemble in 10a/10c | +0.0 to +1.5pp | LOW | 09n complete ✅ |
| A2 | Test camera-pair specific thresholds (learned from GT) | +0.0 to +1.0pp | LOW | None |
| A3 | Test rank-level fusion (if A1 is neutral) | +0.0 to +0.5pp | LOW | A1 results |

### Phase B: Extended Training (Next 1-2 Weeks)

| # | Action | Expected Gain | Effort | Dependencies |
|---|--------|:---:|:---:|-------------|
| B1 | Push R50-IBN from 63% to 70%+ (extended training) | +0.5 to +1.0pp over A1 | MEDIUM | Kaggle GPU time |
| B2 | Extended primary ViT training (120→240 epochs) | +0.0 to +1.0pp | MEDIUM | Kaggle GPU time |
| B3 | Train EVA02 ViT-B/16 CLIP on CityFlowV2 | +0.5 to +1.5pp (3-model) | MEDIUM | timm model availability |

### Phase C: Full Ensemble (Next 2-4 Weeks)

| # | Action | Expected Gain | Effort | Dependencies |
|---|--------|:---:|:---:|-------------|
| C1 | Train VERI-Wild BoT R50-IBN → fine-tune CityFlowV2 | +0.5 to +1.0pp (4-model) | MEDIUM | Download VERI-Wild weights |
| C2 | Train DFN-2B ViT-B/16 CLIP on CityFlowV2 | +0.5 to +1.0pp (5-model) | MEDIUM | B3 results inform approach |
| C3 | Multi-model fusion optimization | +0.5 to +1.0pp | LOW | C1+C2 models available |
| C4 | Re-test DMT camera-aware training in ensemble regime | +0.0 to +1.0pp | MEDIUM | C3 baseline established |

### Phase D: Advanced Techniques (If Needed)

| # | Action | Expected Gain | Effort | Dependencies |
|---|--------|:---:|:---:|-------------|
| D1 | Box-Grained Matching | +0.5 to +2.0pp | HIGH | New feature extraction code |
| D2 | GNN edge classification | +1.0 to +3.0pp | VERY HIGH | New association model |
| D3 | Embedding concatenation + re-PCA | +0.0 to +0.5pp | LOW | Multiple model embeddings |

### Projected Trajectory

| Milestone | MTMC IDF1 | Cumulative Gain | Timeline |
|-----------|:---------:|:-:|---------|
| Current baseline | 0.775 | — | Now |
| After A1 (R50-IBN ensemble) | 0.780-0.790 | +0.5-1.5pp | This week |
| After B1-B2 (extended training) | 0.785-0.800 | +1.0-2.5pp | 1-2 weeks |
| After B3 (3-model ensemble) | 0.790-0.810 | +1.5-3.5pp | 2-3 weeks |
| After C1-C3 (5-model ensemble) | 0.800-0.820 | +2.5-4.5pp | 3-4 weeks |
| After D1-D2 (advanced techniques) | 0.810-0.840 | +3.5-6.5pp | 5-8 weeks |

---

## Risk Assessment

| Risk | Impact | Probability | Mitigation |
|------|--------|:-:|-----------|
| R50-IBN ensemble is neutral (too weak at 63%) | Phase A gives no gain | 30% | Push to 70%+ via extended training (B1) |
| Extended training hurts MTMC (like augoverhaul) | B1/B2 give no gain | 20% | Only deploy if 10c MTMC IDF1 improves, not just mAP |
| EVA02/DFN-2B CLIP too correlated with OpenAI CLIP | Phases B3/C2 give diminishing returns | 25% | Keep CNN diversity (R50-IBN variants) as primary ensemble axis |
| VERI-Wild pretrain doesn't transfer to CityFlowV2 | C1 fails (mAP < 60%) | 35% | Use SBS R50-IBN extended training instead |
| Kaggle GPU quota exhausted | Can't complete all training | 15% | Prioritize A1 > B1 > B3 > C1 > C2 |
| 128-ID CityFlowV2 overfitting caps all models | Fundamental ceiling on fine-tuning | 20% | Use aggressive early stopping; evaluate all checkpoints |

---

## Confirmed Dead Ends (DO NOT RETRY)

Copied from findings.md for reference — **NEVER suggest any of these:**
- CSLS distance (-34.7pp)
- 384px ViT deployment (-2.8pp MTMC IDF1)
- AFLink motion linking (-3.8 to -13.2pp)
- CID_BIAS additive (GT-learned -3.3pp, topology -1.0 to -1.2pp)
- DMT camera-aware training in single-model regime (-1.4pp)
- Hierarchical clustering (-1 to -5pp)
- FAC (-2.5pp)
- K-reciprocal reranking (always hurts)
- Feature concatenation without re-whitening (-1.6pp)
- Network flow solver (-0.24pp, increased conflation)
- CircleLoss on CLIP backbones (inf loss, catastrophic)
- ArcFace on warm-started ResNet (geometry mismatch)
- SGD for any model (catastrophic on small datasets)
- EMA averaging (neutral, dead end)
- Augoverhaul for MTMC (+1.45pp mAP → -5.3pp MTMC IDF1)
- SAM2 foreground masking (-8.7pp)
- ResNet101-IBN-a path FULLY EXHAUSTED (6 variants, all ≤52.77%)
- ResNeXt101-IBN-a (pretrained weight mismatch)
- CLIP RN50x4 CNN (1.55% mAP, QuickGELU mismatch)
- Score-level fusion of two CLIP ViTs (-0.5pp, too correlated)

---

## Key References

- `docs/findings.md` — Full experiment history and dead ends
- `docs/experiment-log.md` — 225+ experiment log
- `docs/subagent-specs/autonomous-improvement.md` — Phase 1 R50-IBN fine-tuning spec
- `docs/subagent-specs/pretrained-ensemble-v2.md` — Original ensemble design
- `src/stage2_features/reid_model.py` — Model dispatchers
- `src/stage2_features/transreid_model.py` — TransReID architecture (timm integration)
- `src/training/model.py` — CNN model architectures
- `configs/default.yaml` — vehicle2/vehicle3 config sections
