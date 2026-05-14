# Low-mAP Recovery: 09g DMT vs 09d Baseline Analysis

## Problem Statement

09g (ResNet101-IBN-a, DMT recipe, 240 epochs) achieved only **mAP=47.92%**, which is **4.85pp worse** than 09d v18 (same architecture, simpler recipe, 150 epochs) at **mAP=52.77%**. This means the more complex DMT training recipe degraded performance rather than improving it.

## Root Cause Analysis

### Primary Cause: Weaker Base Training Recipe

The 09g Stage 1 (supervised) recipe was materially weaker than 09d's proven recipe across **7 dimensions**:

| Hyperparameter | 09d (52.77% mAP) | 09g (47.92% mAP) | Impact |
|----------------|:-----------------:|:-----------------:|--------|
| **Optimizer** | AdamW | Adam | AdamW's decoupled weight decay is strictly better for regularization |
| **Learning rate** | 1e-3 (backbone: 1e-4) | 3.5e-4 (backbone: 3.5e-5) | 09d has ~3x higher LR → stronger gradient signal |
| **Batch size** | 64 (p=16, k=4) | 32 (p=8, k=4) | 09d has 2x batch → more stable PK sampling, better triplet mining |
| **Label smoothing** | 0.05 | 0.1 | 09g's 2x smoothing over-regularizes on a small dataset (128 IDs) |
| **Loss functions** | ID + Triplet | ID + Triplet + Center (0.5) | Center loss adds a 3rd conflicting objective without clear benefit |
| **ColorJitter** | Yes (b=0.2, c=0.15, s=0.1, h=0.05) | No | Missing augmentation diversity |
| **Grad clipping** | max_norm=5.0 | None | Risk of gradient spikes destabilizing training |

### Secondary Cause: DMT Stage 2 (DBSCAN Pseudo-Labeling) Likely Hurt

- DMT reclusters all training images using DBSCAN on FIC-whitened embeddings every 3 epochs
- With only 128 vehicle IDs and already-weak Stage 1 features (base mAP ~47%), the DBSCAN clusters are likely noisy
- The model gets pulled toward noisy pseudo-labels rather than improving
- **Corroborating evidence**: DMT camera-aware training on the primary ViT (87.3% mAP) was already proven harmful: -1.4pp MTMC IDF1 (v46 in findings.md)

### Tertiary Cause: Data Pipeline Difference

- 09d extracts its own crops from raw CityFlowV2 videos (GT boxes, max 15 samples/track, min box area 2000px²)
- 09g uses the pre-built `cityflowv2-reid` Kaggle dataset
- The crop quality, sampling strategy, and ID distribution may differ
- This is a minor factor but could contribute to the gap

## Hyperparameter Comparison Table

| Parameter | 09d v18 | 09g v2 | Notes |
|-----------|---------|--------|-------|
| Architecture | ResNet101-IBN-a + BNNeck + GeM | Same | Identical |
| Image size | 384×384 | 384×384 | Same |
| Optimizer | AdamW | Adam | 09d better |
| Base LR | 1e-3 | 3.5e-4 | 09d 2.86x higher |
| Backbone LR | 1e-4 (0.1x) | 3.5e-5 (0.1x) | 09d 2.86x higher |
| Weight decay | 5e-4 | 5e-4 | Same |
| Warmup epochs | 5 (start_factor=0.01) | 10 (linear) | 09g warms up longer |
| Scheduler | SequentialLR (Linear → Cosine) | LambdaLR (linear → cosine) | Functionally similar |
| eta_min | 1e-6 | N/A (decays to ~0) | Minor |
| Batch size | 64 | 32 | 09d 2x larger |
| PK sampling | p=16, k=4 | p=8, k=4 | 09d 2x more identities per batch |
| Epochs (Stage 1) | 150 | 120 | 09d 25% longer Stage 1 |
| Epochs (Stage 2 DMT) | N/A | 40 | 09g adds DMT stage |
| Eval frequency | Every 5 epochs | Every 10 epochs | 09d evaluates 2x more often |
| Label smoothing | 0.05 | 0.1 | 09g 2x more smoothing |
| ID loss weight | 1.0 | 1.0 | Same |
| Triplet loss | margin=0.3, weight=1.0 | margin=0.3, weight=1.0 | Same |
| Circle loss | weight=0.0 (disabled) | N/A | 09d confirmed circle_weight=0 is optimal |
| Center loss | N/A | weight=0.5, lr=0.5 | 09g adds extra loss |
| FP16 | Yes | Yes | Same |
| Grad clipping | max_norm=5.0 | None | 09d has clipping |
| Random erasing | p=0.5, value=random | p=0.5, value=random | Same |
| ColorJitter | Yes (b=0.2, c=0.15, s=0.1, h=0.05) | No | 09d has extra augmentation |
| Horizontal flip | Yes | Yes | Same |
| Pad+RandomCrop | 10px | 10px | Same |
| Flip eval | Yes | Yes | Same |
| Data source | Self-built crops from raw videos | Pre-built cityflowv2-reid dataset | Different |
| DBSCAN (Stage 2) | N/A | eps=0.58, min_samples=4 | 09g only |
| FIC whitening (Stage 2) | N/A | lambda=5e-4 | 09g only |

## Recovery Strategy

### Recommended Immediate Action: Use 09d Weights as vehicle2 (FASTEST PATH)

**Rationale**: The 09d v18 model at 52.77% mAP is already trained and available. It is strictly better than 09g's 47.9%. No GPU time needed.

**Steps**:
1. Download the 09d v18 checkpoint (`resnet101ibn_cityflowv2_384px_best.pth`) if not already available locally
2. Use it as `vehicle2` in the existing 3-model score-level fusion infrastructure (Stage 2 + Stage 4)
3. Run a quick ensemble test: ViT 256px (primary, ~80% mAP) + ResNet101-IBN-a (secondary, 52.77% mAP), score-level fusion
4. Evaluate MTMC IDF1 on CityFlowV2

**Caveat**: The v49 experiment tested `concat_patch` + vehicle2 ensemble and got -0.3pp. However, v49 used feature concatenation, not score-level fusion. Score-level fusion averages similarity scores and may behave differently. Worth a clean test.

**Expected outcome**: Likely marginal (+/- 0.5pp). The 52.77% model may still be too weak for meaningful ensemble diversity. But this costs zero GPU hours to verify.

### Recommended Follow-Up: Fix the DMT Recipe (Better Training)

**Goal**: Get the ResNet101-IBN-a to ≥60% mAP on CityFlowV2 by using 09d's proven base recipe.

**Recipe for 09g-fixed (new notebook, e.g., 09g v3)**:
```python
# Stage 1: Use 09d's proven recipe
CFG = {
    "img_size": (384, 384),
    "batch_size": 64,           # was 32
    "num_instances": 4,
    "lr": 1e-3,                 # was 3.5e-4
    "weight_decay": 5e-4,
    "warmup_epochs": 5,         # was 10
    "eta_min": 1e-6,
    "label_smoothing": 0.05,    # was 0.1
    "triplet_margin": 0.3,
    "circle_weight": 0.0,       # no circle loss
    "center_loss_weight": 0.0,  # REMOVE center loss
    "id_weight": 1.0,
    "triplet_weight": 1.0,
    "gem_p": 3.0,
    "random_erasing_prob": 0.5,
    "color_jitter": True,       # was False
    "epochs": 150,
    "eval_every": 5,            # was 10
    "fp16": True,
    "grad_clip_norm": 5.0,      # ADD grad clipping
}
# Optimizer: AdamW (not Adam)
# Backbone LR: 0.1x base LR
```

**Stage 2 (DMT, optional)**: Only add if Stage 1 reaches ≥52% mAP. Use conservative settings:
- stage2_epochs: 20 (not 40)
- stage2_lr: 5e-5 (not 1e-4)
- stage2_recluster_every: 5 (not 3)
- dbscan_eps: 0.55 (tighter clusters to reduce noise)
- Only proceed if Stage 1 mAP ≥ 52%

**Account**: ali369 (if GPU hours refreshed) or gumfreddy
**Estimated runtime**: ~5-6 hours on T4

### 09h (ResNeXt101-IBN-a): Hold Until Recipe Is Validated

**Do not retrain 09h until 09g-fixed is validated.**

- ResNeXt101-IBN-a is a more expensive architecture
- If the DMT recipe is still broken, 09h will waste GPU hours for an even worse result
- First validate the fixed recipe on ResNet101 (09g-fixed), then port to ResNeXt101

**When ready**: Use gumfreddy account (mrkdagods is out of hours). Apply the same fixed recipe.

### 09g Output Downloadability

**Yes**, the 09g kernel completed, so its output (including the 47.9% mAP checkpoint) is saved on Kaggle. It can be downloaded with:
```bash
kaggle kernels output mrkdagods/09g-resnet101ibn-dmt -p /tmp/09g_output/
```

However, **there is no reason to download it** — the 09d v18 checkpoint at 52.77% is strictly better.

## Kaggle Account Strategy

| Account | GPU Hours | Assignment |
|---------|-----------|------------|
| **mrkdagods** | EXHAUSTED | Cannot run anything until weekly refresh |
| **ali369** | ~20.5h (may have refreshed) | Use for 09g-fixed (ResNet101 with fixed recipe) |
| **gumfreddy** | Available (unknown hours) | Reserve for 09h (ResNeXt101) after recipe is validated |

## Decision Tree

```
1. Immediately: deploy 09d (52.77%) as vehicle2 for ensemble test
   ├── If ensemble helps (>0.5pp): great, keep it while training better models
   └── If no help: confirms 52% is too weak, proceed to step 2

2. Fix DMT recipe → run 09g-fixed on ali369
   ├── If Stage 1 reaches ≥55%: add DMT Stage 2
   │   ├── If DMT helps: apply same to 09h on gumfreddy
   │   └── If DMT hurts: skip DMT, use supervised-only models
   └── If Stage 1 still <52%: investigate data pipeline differences

3. Only after 09g-fixed is validated: train 09h (ResNeXt101) on gumfreddy
```

## Key Insight

The DMT recipe failed not because DMT is inherently bad, but because the **base supervised training recipe was weakened** in 7 separate dimensions when porting to the DMT framework. The gap is recipe quality, not DMT methodology. Fix the Stage 1 recipe first, then re-evaluate whether DMT Stage 2 helps on top of a strong baseline.