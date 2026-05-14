# Dual-Pipeline Breakthrough Plan — Vehicle + Person

**Date**: 2026-04-13
**Goal**: Maximize results on BOTH pipelines for paper submission. Beat or approach SOTA.
**Vehicle**: IDF1=0.775 → target ≥0.82 (ideally 0.85). SOTA=0.8486.
**Person**: IDF1=0.947 → target ≥0.953. SOTA=~0.953.

---

## CRITICAL CORRECTION: Center Loss Already Deployed

The user assumed center loss was never tried. **This is incorrect.** Center loss is already part of the production ViT training recipe:

- **09_vehicle_reid_cityflowv2**: `CenterLoss(num_classes, 768)`, weight=5e-4, delayed start at epoch 15
- **08_vehicle_reid_sota** (VeRi-776 pretrain): Same center loss, delayed at epoch 30
- **09b_vehicle_reid_384px**: Same center loss, delayed at epoch 15

The current best model (mAP=80.14%) already trains with **CE + Triplet + Center Loss**. Re-adding center loss will produce zero gain. All planning below accounts for this.

---

## Root Cause Analysis

### Vehicle: Why We're 7.36pp Behind SOTA

| Factor | Contribution | Evidence |
|--------|:------------:|---------|
| **Single model vs 5-model ensemble** | ~5-6pp | Every AIC top-3 uses 3-5 models; our 52.77% secondary adds noise |
| **Background contamination in crops** | ~0.5-1pp | Crops include road markings, adjacent vehicles, trees |
| **Viewpoint-specific texture capture** | ~0.5-1pp | 384px experiment proved model captures camera-specific details |
| **No motion-based association** | ~0.3-0.5pp | AFLink motion consistency never tried |
| **Association algorithm ceiling** | <0.3pp | 225+ configs exhausted |

**Key insight**: Since we cannot build a 5-model ensemble (ResNet101 at 52.77% is too weak), we must maximize single-model cross-camera invariance and add non-appearance association signals.

### Person: Why We're 0.6pp Behind SOTA

| Factor | Contribution | Evidence |
|--------|:------------:|---------|
| **Kalman tracker simplicity** | ~0.3-0.5pp | Only 5 ID switches, but greedy assignment causes them |
| **No WILDTRACK-specific ReID** | ~0.1-0.3pp | Market1501 pretrained, never fine-tuned on WILDTRACK |
| **Ground-plane detection noise** | ~0.1pp | MODA=92.1%, some FP/FN remaining |

---

## Vehicle Pipeline Plan

### V1. Cross-Camera Invariance Augmentation Overhaul

**Priority**: ★★★★★ (Highest ROI)
**Expected gain**: +0.5–1.5pp MTMC IDF1
**Probability of success**: 70%
**GPU hours**: ~4h (Kaggle P100, single training run)
**Score**: (1.0pp × 0.70) / 4h = **0.175 pp/GPU-h**

#### Hypothesis
The primary model captures viewpoint-specific textures that help within-camera ReID but hurt cross-camera matching (proven by the 384px experiment: +10pp mAP but -2.8pp MTMC IDF1). Stronger augmentations during training will force the model to learn camera-invariant features instead of memorizing viewpoint-specific details.

#### Current Augmentations (09_vehicle_reid_cityflowv2)
```python
RandomHorizontalFlip(p=0.5)
Resize((272, 272), BICUBIC)
Pad(10) + RandomCrop(256, 256)
ColorJitter(brightness=0.2, contrast=0.15, saturation=0.1, hue=0)
RandomErasing(p=0.5, value='random')
```

#### Proposed Augmentations (ADD to existing)
```python
# Force model to not rely on color (cameras have different white balance)
RandomGrayscale(p=0.1)

# Stronger color jitter to simulate cross-camera appearance shift
ColorJitter(brightness=0.3, contrast=0.25, saturation=0.2, hue=0.05)  # was (0.2, 0.15, 0.1, 0)

# Simulate camera blur differences
GaussianBlur(kernel_size=5, sigma=(0.1, 2.0), p=0.2)

# Simulate mild viewpoint changes
RandomPerspective(distortion_scale=0.1, p=0.2)

# Random background patches (aggressive RandomErasing)
RandomErasing(p=0.5, scale=(0.02, 0.4), ratio=(0.3, 3.3), value='random')  # wider scale range
```

#### Implementation
- **Notebook**: Modify `notebooks/kaggle/09_vehicle_reid_cityflowv2/09_vehicle_reid_cityflowv2.ipynb`
- **Changes**: Update the `build_transforms()` or inline transform pipeline in the training cell
- **Keep everything else identical**: Same loss (CE+Triplet+Center), same LR, same epochs
- **Evaluation**: Train → export weights → run 10a/10b/10c chain → compare MTMC IDF1

#### Why This Might Work
- The 384px failure revealed the model is capturing viewpoint-specific textures (badge positions, reflections, shadow patterns). Stronger augmentations during training will destroy these viewpoint-specific signals and force the model to learn body shape, color, and structural features that transfer across cameras.
- RandomGrayscale is particularly important: cameras have different color temperatures, and forcing 10% of training images to grayscale prevents color memorization.
- GaussianBlur simulates the varying image quality across cameras.

#### Risks
- Too aggressive augmentations could hurt mAP without improving cross-camera matching
- The trade-off between discriminative power and invariance is delicate

#### Measurement
- **Metric A**: CityFlowV2 mAP (should stay ≥75%, slight drop acceptable)
- **Metric B**: MTMC IDF1 on 10c (target: >0.78, ideally >0.80)
- **Command**: Full 10a→10b→10c chain on Kaggle

---

### V2. SAM2 Foreground Masking (Inference-Time)

**Priority**: ★★★★
**Expected gain**: +0.3–0.5pp MTMC IDF1
**Probability of success**: 60%
**GPU hours**: ~2h additional per 10a run (SAM2 inference on crops)
**Score**: (0.4pp × 0.60) / 2h = **0.12 pp/GPU-h**

#### Hypothesis
Vehicle crops include background clutter (road markings, adjacent vehicles, trees, buildings) that pollutes ReID embeddings. Background contamination varies by camera viewpoint, making it a source of cross-camera embedding drift. SAM2 can segment vehicle foreground and zero out the background before ReID inference.

#### Design

**Approach**: Inference-time only (not training-time)

**Rationale**: Training-time masking would require pre-computing masks for all training crops, adding complexity. Inference-time masking is simpler and lets the model's learned features focus on the foreground.

**Pipeline Integration**:
```
Stage 2 Pipeline:
  crop_extractor.extract_crops()
  → NEW: sam2_mask_foreground(crops)  # zero out background
  → reid_model.extract_features(masked_crops)
  → quality_weighted_pooling()
  → PCA whitening
  → L2 normalize
```

**Implementation Details**:
1. Load SAM2 `sam2_hiera_tiny` (smallest model, ~39M params) on CUDA
2. For each crop batch: run SAM2 auto-mask → select largest mask (vehicle body)
3. Apply mask: `crop * mask + (1-mask) * mean_pixel` (fill background with dataset mean)
4. Feed masked crop to ReID model as normal

**GPU Cost Estimate**:
- SAM2-tiny on T4: ~15ms per image
- ~941 tracklets × 16 crops = ~15K crops → ~225s (4 minutes)
- Total stage2 overhead: <10 minutes. Very affordable on Kaggle P100.

**Code Changes**:
- New file: `src/stage2_features/foreground_masking.py`
- Modify: `src/stage2_features/pipeline.py` (add masking step between crop extraction and ReID)
- Kaggle notebook: Add `pip install segment-anything-2` and SAM2 model weights as dataset dependency

#### Risks
- SAM2 may fail on partially occluded vehicles → fallback to unmasked crop
- Very small vehicles may get poor segmentation → skip masking for crops < 64×64

---

### V3. AFLink Motion-Based Post-Association Linking

**Priority**: ★★★★
**Expected gain**: +0.2–0.5pp MTMC IDF1
**Probability of success**: 55%
**GPU hours**: 0 (training-free, CPU only)
**Score**: (0.35pp × 0.55) / 0.5h = **0.385 pp/GPU-h** ← Best cost-efficiency

#### Hypothesis
After Stage 4 association, some tracklets remain unlinked because their appearance similarity falls just below the threshold, even though their motion patterns are clearly consistent (same direction, compatible speed, spatiotemporal continuity). AFLink uses spatial-temporal trajectory features to find additional links that appearance alone misses.

#### Design

**What AFLink Does**: Given two trajectories (each a sequence of bounding boxes with timestamps), AFLink predicts whether they belong to the same identity based on:
1. **Spatial proximity**: Are the end of trajectory A and start of trajectory B spatially close?
2. **Velocity consistency**: Do the velocity vectors align?
3. **Direction consistency**: Are they moving in the same direction?
4. **Temporal gap**: Is the time gap between them plausible for a camera transition?

**This is a TRAINING-FREE technique**: No learned parameters. Pure geometric/kinematic heuristics.

**Integration with Current Stage 4**:
```
Stage 4 Pipeline:
  FIC whitening → QE → FAISS → candidate pairs → scoring → conflict_free_cc
  → Global trajectories
  → NEW: AFLink post-association (merge unlinked trajectories with motion consistency)
  → Final global trajectories
```

**Implementation**:
```python
def aflink_post_association(
    trajectories: List[GlobalTrajectory],
    tracklets: Dict[str, List[Tracklet]],
    max_time_gap: float = 30.0,       # seconds
    max_spatial_gap: float = 200.0,    # pixels (projected)
    min_direction_cos: float = 0.7,    # cos(θ) > 0.7 → <45°
    min_velocity_ratio: float = 0.5,   # speed within 2x range
) -> List[GlobalTrajectory]:
    """Post-association linking based on motion consistency."""
    # 1. Compute trajectory endpoints (last bbox of A, first bbox of B)
    # 2. For each unlinked pair in different cameras:
    #    - Check temporal gap (end_A to start_B)
    #    - Check spatial proximity (projected positions)
    #    - Check velocity direction alignment
    #    - Check speed consistency
    # 3. Link pairs that pass all checks (AND logic, not OR)
    # 4. Respect same-camera conflict constraint
```

**Motion Features** (from existing tracklet data):
- Position: bounding box center (x, y) from `Tracklet.boxes`
- Velocity: Δposition / Δtime from consecutive detections
- Direction: atan2(Δy, Δx) heading angle
- These are all available from Stage 1 tracklet data — no new data needed

**Code Changes**:
- New file: `src/stage4_association/aflink.py`
- Modify: `src/stage4_association/pipeline.py` (call AFLink after `merge_tracklets_to_trajectories`)

#### Risks
- CityFlowV2 has only 6 cameras across 2 scenes with limited spatial overlap
- Motion patterns may not be reliable across non-overlapping cameras
- False merges from similar motion patterns (vehicles behind each other)

---

### V4. Knowledge Distillation Redo (ViT-L/14 → ViT-B/16)

**Priority**: ★★★
**Expected gain**: +1.0–2.0pp MTMC IDF1
**Probability of success**: 40%
**GPU hours**: ~12h (6h teacher, 6h student distillation on P100)
**Score**: (1.5pp × 0.40) / 12h = **0.05 pp/GPU-h**

#### Hypothesis
09c failed due to implementation bugs (dimension mismatch, wrong temperature), not because KD doesn't work for ReID. AIC24 top-3 all used ViT-L teacher distillation for +2-4% mAP. A proper implementation could materially improve embedding quality.

#### Design
**Teacher**: ViT-L/14 CLIP (frozen, feature extractor only)
- Load from timm: `vit_large_patch14_clip_224.openai` (304M params)
- Feature dim: 1024D
- No training needed — use frozen CLIP features as soft targets

**Student**: Our existing ViT-B/16 CLIP (768D)
- Initialize from the current best 09 checkpoint (mAP=80.14%)
- Train with combined loss:

```python
loss = (1 - alpha) * task_loss + alpha * kd_loss

# Task loss (unchanged)
task_loss = ce_loss + triplet_loss + center_loss

# KD loss (NEW)
projector = nn.Linear(768, 1024)  # student → teacher dim
kd_loss = MSE(projector(student_feat), teacher_feat.detach())
         + KL_div(student_logits / T, teacher_logits / T, T=2)
```

**Key fixes from failed 09c**:
1. Proper projector: `nn.Linear(768, 1024)` with Xavier init (09c had dimension mismatch)
2. Temperature T=2 (09c used T=4, too soft for fine-grained ReID)
3. Initialize student from the strong baseline (09c trained from scratch)
4. Alpha=0.5 (balanced task vs KD)

**Notebook**: New `notebooks/kaggle/09k_kd_vitl_student/`

#### Risks
- ViT-L/14 frozen features may not transfer well to CityFlowV2's small 128-ID dataset
- KD often needs large datasets to work well; CityFlowV2 has only 7.5K training images
- P100 may not fit both models; may need gradient checkpointing

---

### V5. Viewpoint-Invariant Training with Circle Loss (ViT-Only)

**Priority**: ★★★
**Expected gain**: +0.3–0.8pp MTMC IDF1
**Probability of success**: 45%
**GPU hours**: ~4h
**Score**: (0.5pp × 0.45) / 4h = **0.056 pp/GPU-h**

#### Hypothesis
Circle loss was tried in two contexts and failed in both. But both contexts were confounded:
- **09f v2**: Circle + Triplet on ResNet → catastrophic gradient conflict (16% mAP)
- **09b DMT**: Circle + Camera loss + 384px → -1.4pp (confounded with DMT and resolution)

Circle loss **for ViT alone, without DMT or camera loss**, has never been tested. Circle loss is adaptive — it increases the penalty for hard negatives and reduces it for easy positives, which could improve cross-camera matching specifically.

#### Design
Replace triplet loss with circle loss (NOT both):
```python
# Current: CE + Triplet + Center
# Proposed: CE + Circle + Center (drop Triplet entirely)

circle_loss = CircleLoss(m=0.25, gamma=64)
# m = margin, gamma = scale factor
# Uses cosine similarity of L2-normalized features
```

**Key difference from failed experiments**:
- No triplet loss (avoids gradient conflict)
- No camera loss (avoids DMT penalty)
- Same 256px resolution (avoids 384px penalty)
- Same ViT-B/16 CLIP backbone

**Notebook**: Fork `09_vehicle_reid_cityflowv2`, replace triplet with circle loss

#### Risks
- Circle loss alone may not outperform triplet loss
- The 128-ID CityFlowV2 dataset may be too small for circle loss to converge properly
- No clear evidence that the triplet/circle conflict transfers to ViT (only tested on ResNet)

---

### V6. Larger Batch Size with Harder Triplet Mining

**Priority**: ★★★
**Expected gain**: +0.3–0.8pp MTMC IDF1
**Probability of success**: 50%
**GPU hours**: ~5h (P100 can fit bs=64)
**Score**: (0.5pp × 0.50) / 5h = **0.05 pp/GPU-h**

#### Hypothesis
Current batch size is P=8×K=4=32. With only 128 IDs in CityFlowV2, each batch sees 8/128 = 6.25% of all identities. Increasing to P=16×K=4=64 doubles the identity coverage per batch, providing harder negatives for triplet mining and better gradient signal.

#### Design
```python
# Current
BATCH_SIZE = 32  # P=8, K=4

# Proposed
BATCH_SIZE = 64  # P=16, K=4
# Adjust LR: linear scaling rule → lr *= 2
backbone_lr = 2e-4  # was 1e-4
head_lr = 2e-3      # was 1e-3
```

P100 has 16GB VRAM. ViT-B/16 at 256×256 with fp16: ~200MB per sample in forward pass. Batch 64 ≈ 12.8GB — fits on P100.

**Notebook**: Fork `09_vehicle_reid_cityflowv2`, change batch size + LR

---

### V7. EMA (Exponential Moving Average) Model Averaging

**Priority**: ★★★
**Expected gain**: +0.2–0.5pp mAP (translates to ~0.1-0.3pp MTMC IDF1)
**Probability of success**: 65%
**GPU hours**: ~0.5h additional (same training, just maintain EMA copy)
**Score**: (0.25pp × 0.65) / 0.5h = **0.325 pp/GPU-h**

#### Hypothesis
EMA maintains an exponentially-weighted moving average of model weights during training. The EMA model generalizes better than the final checkpoint because it averages out late-training oscillations. Standard practice in modern training but never used in our pipeline.

#### Design
```python
class ModelEMA:
    def __init__(self, model, decay=0.9999):
        self.ema = copy.deepcopy(model)
        self.decay = decay
    
    def update(self, model):
        for ema_p, p in zip(self.ema.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(p.data, alpha=1 - self.decay)

# In training loop:
ema = ModelEMA(model, decay=0.9999)
for epoch in range(num_epochs):
    for batch in dataloader:
        loss.backward()
        optimizer.step()
        ema.update(model)

# At evaluation/export:
torch.save(ema.ema.state_dict(), 'best_model_ema.pth')
```

**Notebook**: Add EMA to `09_vehicle_reid_cityflowv2` training loop. Export both regular and EMA checkpoints. Test both in 10a pipeline.

---

## Person Pipeline Plan

### P1. Min-Cost Flow Tracker (Replace Kalman)

**Priority**: ★★★★
**Expected gain**: +0.2–0.5pp IDF1 (potentially fixing 2-3 of 5 ID switches)
**Probability of success**: 50%
**GPU hours**: 0 (CPU-only, optimization-based)
**Score**: (0.35pp × 0.50) / 1h = **0.175 pp/GPU-h**

#### Hypothesis
The current Kalman tracker uses greedy Hungarian assignment frame-by-frame, which is locally optimal but globally suboptimal. The 5 remaining ID switches likely occur at moments of close proximity where greedy assignment makes errors. A min-cost flow formulation finds the globally optimal assignment across all frames simultaneously.

#### Design

**Approach**: Min-cost flow (NOT GNN — too heavy for 0.6pp gap)

**Why min-cost flow, not GNN**:
- GNN requires training data and GPU. Overkill for 5 ID switches.
- Min-cost flow is training-free, globally optimal, and has proven MTMC performance.
- Python implementation via `scipy.optimize.linear_sum_assignment` or `networkx.min_cost_flow`.

**Formulation**:
```
Graph:
  - Source → Detection nodes (one per detection per frame)
  - Detection nodes → Detection nodes (across frames, edge cost = -similarity)
  - Detection nodes → Sink (track termination)
  - Source → Detection nodes (track initialization)

Constraints:
  - Each detection assigned to exactly one track
  - Flow conservation at each node
  
Cost function:
  - Appearance similarity (ReID cosine): weight=0.7
  - Position distance (ground-plane Euclidean): weight=0.3
```

**Implementation**:
- Python package: `scipy` (already a dependency)
- Build sparse cost matrix from ground-plane positions + ReID features
- Solve globally over all frames
- Convert flow solution to track assignments

**Integration**: Replace the Kalman tracker in `12b_wildtrack_tracking_reid.ipynb`, keep everything else identical.

#### Alternative: Sliding-Window Global Assignment
If full min-cost flow is too slow (O(V³) for V detections):
- Use sliding window of 10 frames
- Run Hungarian within each window
- Stitch windows with ReID-based matching

#### Risks
- Ground-plane WILDTRACK has ~20 people × ~400 frames × 7 cameras. Full graph could be large.
- Min-cost flow may produce identical results to Hungarian if the cost matrix is well-separated
- Implementation complexity may be higher than expected for marginal gain

---

### P2. WILDTRACK-Specific Person ReID Fine-Tuning

**Priority**: ★★★
**Expected gain**: +0.1–0.3pp IDF1
**Probability of success**: 40%
**GPU hours**: ~2h (fine-tune on WILDTRACK crops)
**Score**: (0.2pp × 0.40) / 2h = **0.04 pp/GPU-h**

#### Hypothesis
The person ReID model is pretrained on Market1501 (1501 IDs, outdoor single-camera crops) and never fine-tuned on WILDTRACK (multi-view indoor/outdoor, 7 cameras with known calibration). Fine-tuning on WILDTRACK-specific crops could improve the ReID merge step that currently finds no additional merges.

#### Design
1. Extract all WILDTRACK training crops (using calibration to project ground-plane tracks to camera views)
2. Label crops with person IDs from WILDTRACK ground truth
3. Fine-tune the person ViT-B/16 for 20-30 epochs on WILDTRACK crops (same recipe as Market1501 but with WILDTRACK-specific camera IDs)
4. Use fine-tuned model in 12b for ReID feature extraction

**Risk**: WILDTRACK has only ~20 person IDs in the training set → severe overfitting risk. May need strong augmentation or freezing early layers.

---

### P3. Track Interpolation Enhancement

**Priority**: ★★★
**Expected gain**: +0.1–0.2pp IDF1
**Probability of success**: 60%
**GPU hours**: 0
**Score**: (0.15pp × 0.60) / 0.2h = **0.45 pp/GPU-h** ← Best person pipeline cost-efficiency

#### Hypothesis
Current interpolation fills gaps ≤2 frames with linear interpolation. The 5 remaining ID switches may include cases where:
1. A person is briefly occluded for 3-4 frames (beyond current gap limit)
2. The Kalman prediction during the gap drifts enough to cause reassignment

Extending the interpolation gap to 3-4 frames and using quadratic (not linear) interpolation based on velocity could reduce 1-2 ID switches.

#### Design
```python
# Current
interpolation_gap = 2  # linear

# Proposed
interpolation_gap = 4  # extended
interpolation_method = "quadratic"  # use velocity for prediction
```

**Implementation**: Modify the interpolation logic in 12b notebook.

---

### P4. Detection Confidence Re-Weighting

**Priority**: ★★
**Expected gain**: +0.05–0.15pp IDF1 (might fix 1 ID switch)
**Probability of success**: 40%
**GPU hours**: 0
**Score**: (0.1pp × 0.40) / 0.2h = **0.20 pp/GPU-h**

#### Hypothesis
Some ID switches might be caused by low-confidence false positive detections that temporarily steal a track ID. Adding a minimum confidence threshold for detection-to-track assignment could prevent these.

---

## Implementation Priority Ranking

### Vehicle Pipeline — Ranked by (gain × probability) / GPU_hours

| Rank | Technique | Expected Gain | P(success) | GPU Hours | Score (pp/GPU-h) | Notebook Needed |
|:----:|-----------|:-------------:|:----------:|:---------:|:----------------:|:---------------:|
| **1** | V3: AFLink Motion Linking | 0.2–0.5pp | 55% | 0.5h (CPU) | **0.385** | No (CPU post-processing) |
| **2** | V7: EMA Model Averaging | 0.1–0.3pp | 65% | 0.5h | **0.325** | Modify 09 |
| **3** | V1: Augmentation Overhaul | 0.5–1.5pp | 70% | 4h | **0.175** | Modify 09 |
| **4** | V2: SAM2 Masking | 0.3–0.5pp | 60% | 2h | **0.120** | Modify 10a |
| **5** | V5: Circle Loss (ViT) | 0.3–0.8pp | 45% | 4h | **0.056** | Fork 09 |
| **6** | V6: Larger Batch | 0.3–0.8pp | 50% | 5h | **0.050** | Fork 09 |
| **7** | V4: KD (ViT-L→ViT-B) | 1.0–2.0pp | 40% | 12h | **0.050** | New 09k |

### Person Pipeline — Ranked by Score

| Rank | Technique | Expected Gain | P(success) | GPU Hours | Score (pp/GPU-h) | Notebook Needed |
|:----:|-----------|:-------------:|:----------:|:---------:|:----------------:|:---------------:|
| **1** | P3: Extended Interpolation | 0.1–0.2pp | 60% | 0h | **0.450** | Modify 12b |
| **2** | P4: Detection Re-Weighting | 0.05–0.15pp | 40% | 0h | **0.200** | Modify 12b |
| **3** | P1: Min-Cost Flow Tracker | 0.2–0.5pp | 50% | 0h | **0.175** | Modify 12b |
| **4** | P2: WILDTRACK ReID Fine-Tune | 0.1–0.3pp | 40% | 2h | **0.040** | New 09w |

---

## Recommended Execution Order

### Wave 1: Zero-GPU-Cost Experiments (1-2 days)
These can run immediately on Kaggle CPU or local machine:

1. **V3: AFLink** — Implement motion-based post-association in Stage 4. Run on existing 10c outputs. CPU-only.
2. **P3: Extended Interpolation** — Increase gap to 4 frames in 12b. Re-run tracking sweep.
3. **P4: Detection Re-Weighting** — Add minimum confidence for assignment in 12b.

**Expected combined gain**: +0.3–0.7pp vehicle, +0.1–0.3pp person

### Wave 2: Single Training Run (3-5 days)
One carefully designed training run that combines multiple small improvements:

4. **V1 + V7 combined**: Train a new ViT with augmentation overhaul AND EMA, in a single 09 notebook run.
   - Add RandomGrayscale, stronger ColorJitter, GaussianBlur, RandomPerspective
   - Add EMA with decay=0.9999
   - Export BOTH regular and EMA checkpoints
   - Run 10a→10b→10c chain for each

**Expected combined gain**: +0.7–1.8pp vehicle

### Wave 3: SAM2 Integration (3-5 days)
5. **V2: SAM2** — Add foreground masking to Stage 2 inference. Test on existing model first, then on Wave 2 model.

**Expected gain**: +0.3–0.5pp vehicle (additive)

### Wave 4: Alternative Training Recipes (1-2 weeks)
6. **V5: Circle Loss** — Fork 09, replace triplet with circle loss. Train and compare.
7. **V6: Larger Batch** — Fork 09, increase to bs=64. Train and compare.
8. **P1: Min-Cost Flow** — Implement in 12b, replace Kalman.

### Wave 5: High-Risk High-Reward (2-3 weeks)
9. **V4: Knowledge Distillation** — Only if Waves 1-3 don't close enough gap. Large time investment.
10. **P2: WILDTRACK ReID** — Only if P1+P3 don't cross 0.953.

---

## Cumulative Projections

### Vehicle (Conservative → Optimistic)

| After Wave | Conservative | Optimistic | Technique |
|:----------:|:------------:|:----------:|-----------|
| Baseline | 0.775 | 0.775 | Current |
| Wave 1 | 0.778 | 0.785 | AFLink |
| Wave 2 | 0.785 | 0.800 | Augmentation + EMA |
| Wave 3 | 0.788 | 0.805 | + SAM2 |
| Wave 4 | 0.793 | 0.815 | + Circle/Batch |
| Wave 5 | 0.800 | 0.830 | + KD |

### Person (Conservative → Optimistic)

| After Wave | Conservative | Optimistic | Technique |
|:----------:|:------------:|:----------:|-----------|
| Baseline | 0.947 | 0.947 | Current |
| Wave 1 | 0.949 | 0.952 | Interpolation + weighting |
| Wave 4 | 0.951 | 0.955 | + Min-cost flow |
| Wave 5 | 0.952 | 0.957 | + WILDTRACK ReID |

---

## Paper Impact Assessment

### With Waves 1-3 Completed (Most Likely Scenario)

| Pipeline | Projected IDF1 | SOTA | % of SOTA | Paper Angle |
|----------|:--------------:|:----:|:---------:|-------------|
| Vehicle | 0.788–0.805 | 0.849 | 93–95% | "93% of SOTA with 1 model vs 5" |
| Person | 0.949–0.952 | 0.953 | 99.6%+ | "Matching SOTA" |

This is significantly stronger for the paper than the current position:
- Vehicle: "91% of SOTA" → "93-95% of SOTA" (AND closing the gap further)
- Person: "99.4% of SOTA" → "99.6-100% of SOTA" (potentially beating SOTA)

### Key Paper Claims Enabled
1. **"Single model achieves 93-95% of 5-model-ensemble SOTA"** — stronger than current 91%
2. **"WILDTRACK SOTA or within 0.1pp"** — publication-worthy on its own
3. **"225+ experiments prove feature quality, not association, is the bottleneck"** — unchanged but more credible with higher numbers
4. **"Cross-camera invariance > discriminative power"** — proven by augmentation experiment (if V1 works, it directly supports the paper's thesis)

---

## What NOT To Do (Confirmed Dead Ends)

| Approach | Status | Why |
|----------|--------|-----|
| Center loss tuning | Already deployed (5e-4, epoch 15 delay) | Already optimal |
| 384px deployment | Dead end (-2.8pp) | Viewpoint-specific textures |
| DMT camera-aware training | Dead end (-1.4pp) | Single-model regime penalty |
| CSLS | Dead end (-34.7pp) | Catastrophic |
| Hierarchical clustering | Dead end (-1 to -5pp) | Centroid averaging |
| FAC | Dead end (-2.5pp) | Cross-camera KNN overwrites |
| Reranking | Dead end | K-reciprocal false positives |
| Feature concatenation | Dead end (-1.6pp) | Uncalibrated spaces |
| ResNet101-IBN ensemble | Dead end (52.77% too weak) | Needs ≥65% mAP |
| More association sweeps | Exhausted (225+ configs) | All within 0.3pp |
| SGD optimizer | Dead end (30.27% mAP) | AdamW essential |
| Circle + Triplet together | Dead end | Gradient conflict |
| VeRi-776→CityFlowV2 ResNet | Dead end (42.7% < 52.77%) | Pretraining hurts transfer |
| Person: better detector alone | Dead end | MODA 90.9→92.1% didn't help tracking |
