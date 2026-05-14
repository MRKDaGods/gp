# 09f v3 Fix + Person ReID Recovery Plan

> **Created**: 2026-03-29
> **Status**: READY FOR IMPLEMENTATION

---

## Part 1: 09f Vehicle ReID — Root Cause Analysis

### Failure Summary

09f v1 fine-tuned the VeRi-776-pretrained ResNet101-IBN-a (09e, mAP=62.52%) on CityFlowV2 at 384×384. Result: **mAP=16.2%** at epoch 4 (first eval point), declining thereafter. The 52.77% baseline (09d v18, ImageNet→CityFlowV2 directly) is 3× better. The VeRi-776 pretraining was supposed to help, not destroy performance.

### Root Cause 1 (CRITICAL): Circle Loss — Documented Dead End

**The single biggest cause.** 09f uses `circle_weight=1.0` (triplet + circle + ID loss simultaneously). This directly contradicts documented evidence:

| Notebook | circle_weight | mAP | Outcome |
|----------|:------------:|:---:|---------|
| 09d v17 | 0.5 | 29.6% | "Gradient conflict" — documented dead end |
| 09d v18 (ali369) | **0.0** | **52.77%** | Best ResNet without VeRi |
| 09e v2 | **0.0** | **62.52%** | Best VeRi-776 pretrain |
| **09f v1** | **1.0** | **16.2%** | **Catastrophic** |

Triplet loss pulls embeddings to form tight per-ID clusters with margin separation. Circle loss uses a completely different optimization surface (softplus-based pairwise similarity). When applied to the **same `global_feat` tensor**, they produce conflicting gradients that destabilize the feature space.

During warmup (LR≈0), the gradient conflict is negligible — the VeRi-776 features survive mostly intact, yielding 16.2% mAP. Once warmup ends and LR reaches full value at epoch 5, the conflicting gradients destroy the learned representations.

### Root Cause 2 (HIGH): Batch Size Too Small

| Config | batch_size | P (IDs/batch) | Result |
|--------|:----------:|:--------------:|--------|
| 09d v18 (52.77%) | 48 | 12 | Best |
| 09e (62.52%) | implied ~48 | ~12 | Best |
| **09f (16.2%)** | **32** | **8** | Catastrophic |

With only 128 vehicle IDs in CityFlowV2, P=8 means each batch samples 6.25% of all IDs. This severely limits the quality of hard negative mining for triplet loss, producing noisy gradient estimates that compound the circle loss conflict.

### Root Cause 3 (MEDIUM): Label Smoothing Too Aggressive

| Config | label_smoothing | mAP |
|--------|:--------------:|:---:|
| 09d v18 | 0.05 | 52.77% |
| **09f** | **0.1** | **16.2%** |

With only 128 classes, smoothing=0.1 redistributes 10% of the target probability mass uniformly. At 128 classes, each non-target class gets ~0.078% extra probability. This is 2× the proven-optimal value and makes the classifier less confident, slowing convergence on an already tiny dataset.

### Root Cause 4 (LOW): Aggressive Warmup Start Factor

The warmup uses `start_factor=0.01`, meaning LR begins at 1% of target. For the backbone (target=7e-6), this means starting at 7e-8 — essentially zero. The classifier (target=7e-4) starts at 7e-6. This isn't harmful per se, but the 100× ramp from epoch 0→5 means the transition from "harmless warmup" to "full gradient conflict" is abrupt.

### Why Best Model Was Saved at Epoch 4

The training loop evaluates every 5 epochs: epochs 4, 9, 14, 19... (0-indexed). Epoch 4 is the **last warmup epoch** — LR is at ~90% of target but circle loss conflict is still dampened. By epoch 9 (next eval), the model has trained 5 full epochs at peak LR with conflicting triplet+circle gradients. Performance craters.

---

## Part 2: 09f v3 — Fix Specification

### Strategy

Match the proven recipe from 09d v18 + 09e, adapting LR downward for fine-tuning from a stronger initialization point.

### Config Changes

```python
CFG = {
    # === CRITICAL FIXES ===
    "circle_weight": 0.0,          # was 1.0 — REMOVE circle loss (dead end)
    "batch_size": 48,              # was 32 — match 09d v18 (P=12 IDs/batch)
    "label_smoothing": 0.05,       # was 0.1 — match 09d v18
    
    # === LR ADJUSTMENTS FOR FINE-TUNING ===
    "lr": 3.5e-4,                  # was 7e-5 — match 09e base LR
    # Differential LR groups (KEEP from 09f v1):
    #   backbone: lr × 0.1 = 3.5e-5 (was 7e-6)
    #   pool/bottleneck: lr = 3.5e-4 (was 7e-5)
    #   classifier: lr × 10 = 3.5e-3 (was 7e-4)
    
    # === SCHEDULE ===
    "epochs": 120,                 # was 60 — match 09d v18, more time for convergence
    "warmup_epochs": 5,            # keep from 09f
    "warmup_start_factor": 0.1,    # was 0.01 — less aggressive ramp
    "eta_min": 1e-7,               # keep from 09f
    
    # === UNCHANGED (already correct) ===
    "img_size": (384, 384),
    "num_instances": 4,
    "weight_decay": 5e-4,
    "triplet_margin": 0.3,
    "triplet_weight": 1.0,
    "id_weight": 1.0,
    "random_erasing_prob": 0.5,
    "color_jitter": True,
    "eval_every": 5,
    "fp16": True,
}
```

### Loss Function Changes

**Before (09f v1):**
```python
total_loss = loss_id + loss_tri + loss_circle  # 3 losses, circle conflicts with triplet
```

**After (09f v3):**
```python
total_loss = loss_id + loss_tri  # Only ID + triplet, proven recipe
# Circle loss REMOVED entirely — not just weighted to 0, skip the computation
```

### Files to Edit

1. **`notebooks/kaggle/09f_vehicle_reid_resnet101ibn_cityflowv2/09f_vehicle_reid_resnet101ibn_cityflowv2.ipynb`**
   - Cell with CFG dict: update all config values above
   - Cell with loss computation: remove circle loss entirely
   - Cell with warmup scheduler: change start_factor to 0.1
   - Cell with training loop: remove circle_loss_fn from train_one_epoch call

### Expected Outcomes

| Metric | 09f v1 | 09f v3 (expected) | Reasoning |
|--------|:------:|:-----------------:|-----------|
| mAP | 16.2% | **65-75%** | VeRi pretrain (62.52%) + CityFlowV2 fine-tune should exceed 09d v18 (52.77%) |
| Training trajectory | Best at epoch 4, diverging | Improving through epoch 60+, plateau 80-100 |
| Convergence | Never converged | Should start improving by epoch 10 |

### Risk Factors

1. **LR still too high**: If mAP peaks early and declines, reduce base LR to 1e-4. Signs: best mAP before epoch 30.
2. **LR too low**: If mAP plateaus below 55%, increase base LR to 5e-4 or 1e-3. Signs: monotonic but slow improvement.
3. **Overfitting**: CityFlowV2 has only 128 IDs (tiny). If train loss drops but val mAP stalls, increase weight_decay to 1e-3 or add more augmentation.

### Fallback Plan

If 09f v3 still fails (mAP < 50%):
1. Use the proven 09d v18 recipe EXACTLY (lr=1e-3, no differential LR, bs=48) but with VeRi-776 pretrained init instead of ImageNet
2. The only change vs 09d v18 would be the starting checkpoint — eliminates all other variables

---

## Part 3: Person ReID — Feature Collapse Diagnosis

### Problem

42 person tracks in WILDTRACK 12b, all 768-dim L2-normalized. Mean off-diagonal cosine similarity = 0.8737. 844/861 pairs above merge threshold (0.75). Features cannot distinguish people.

### Root Cause: Architecture Mismatch (CONFIRMED)

| Component | Training (09p) | Inference (12b/pipeline) | Match? |
|-----------|---------------|------------------------|:------:|
| timm model | `vit_base_patch16_224` | `vit_base_patch16_clip_224.openai` | **NO** |
| norm_pre layer | Does not exist | Exists (LayerNorm, randomly init) | **NO** |
| Normalization | ImageNet (0.485, 0.456, 0.406) | CLIP (0.481, 0.458, 0.408) | **NO** |
| Architecture class | Standard ViT | CLIP ViT | **NO** |

The consequences:
1. **`norm_pre` layer**: The CLIP ViT has a `norm_pre` LayerNorm before the transformer blocks that standard ViT lacks. When loading 09p weights into the CLIP architecture, `norm_pre` is randomly initialized. This layer processes ALL token embeddings before ANY attention computation — random initialization here corrupts the entire feature pipeline.
2. **Normalization stats**: CLIP uses different mean/std values. Applying CLIP normalization to images then passing through weights trained with ImageNet normalization shifts all activations.
3. **Combined effect**: Near-random features with high mutual similarity (mode collapse).

### Evidence

The pipeline code explicitly creates the person model with:
```yaml
# configs/datasets/wildtrack.yaml line 78-80
vit_model: "vit_base_patch16_clip_224.openai"  # WRONG for 09p weights
clip_normalization: true                        # WRONG for 09p weights
```

But 09p trained with:
```python
# 09p notebook line 296
self.vit = timm.create_model("vit_base_patch16_224", ...)  # Standard ViT
# 09p notebook line 236
transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet
```

---

## Part 4: Person ReID — Short-Term Fix (No GPU Retraining)

### Option A: Fix Config to Match Training (RECOMMENDED — Highest Impact, Zero Cost)

Change the person ReID config to match the actual 09p training architecture:

```yaml
# configs/datasets/wildtrack.yaml
stage2:
  reid:
    person:
      vit_model: "vit_base_patch16_224"       # was: vit_base_patch16_clip_224.openai
      clip_normalization: false                 # was: true

# configs/default.yaml (person section only)
stage2:
  reid:
    person:
      vit_model: "vit_base_patch16_224"       # was: vit_base_patch16_clip_224.openai
      clip_normalization: false                 # was: true
```

**Expected impact**: Features should become actually discriminative. The 09p model achieved **mAP=90.5% on Market-1501** — that quality is completely wasted by the architecture mismatch. Fixing this should collapse off-diagonal similarity from 0.87 to ~0.3-0.5 range, enabling meaningful identity discrimination.

**Risk**: The TransReID inference wrapper (`src/stage2_features/transreid_model.py`) loads weights into `timm.create_model(vit_model, ...)`. The 09p training wrapper is a different class (`TransReIDViT`) with different key names. Need to verify weight key compatibility:

- 09p saves: `vit.blocks.N.attn.qkv.weight`, `bottleneck.weight`, `classifier.weight`, `sie.embed`
- Pipeline expects: potentially different key mapping in `TransReID._load_reid_checkpoint()`

If keys don't align, may need a key-mapping adapter in the weight loading code. The coder agent should verify this.

### Option B: HSV-Only Association (Fallback)

If Option A has weight-loading issues that need additional code changes:

```yaml
# Disable ReID features, rely only on HSV histograms for person association
stage4:
  association:
    person:
      appearance: 0.0     # disable broken ReID
      hsv: 0.90            # HSV histograms are architecture-independent
      spatio: 0.10         # spatiotemporal
```

**Expected impact**: HSV histograms capture color distribution well for WILDTRACK (outdoor, varied clothing). Won't be as discriminative as proper ReID, but far better than mode-collapsed features. Expect IDF1 in the 40-60% range.

### Option C: Off-the-Shelf Pretrained Model (Medium Effort)

Use timm's pretrained `vit_base_patch16_224` weights directly (ImageNet-trained, no fine-tuning) as a general feature extractor. Skip the 09p checkpoint entirely.

```yaml
stage2:
  reid:
    person:
      weights_path: null  # Use timm pretrained
      vit_model: "vit_base_patch16_224"
      clip_normalization: false
```

**Expected impact**: Generic ImageNet features won't be person-discriminative, but they'll at least be coherent features (not mode-collapsed). Better than broken CLIP arch, worse than properly-loaded 09p weights.

### Option D (Long-Term): Retrain with Correct Architecture

Retrain person ReID using the CLIP ViT architecture (`vit_base_patch16_clip_224.openai`) to match the pipeline. This is the proper fix but requires a Kaggle GPU run.

Changes to 09p:
```python
# Change line 296 from:
self.vit = timm.create_model("vit_base_patch16_224", ...)
# To:
self.vit = timm.create_model("vit_base_patch16_clip_224.openai", ...)
```

Expected mAP on Market-1501: 91-93% (CLIP ViT is slightly stronger than standard ViT for person ReID in published benchmarks).

---

## Part 5: Priority Actions

### Immediate (Today)

| # | Action | Impact | Effort |
|:-:|--------|:------:|:------:|
| 1 | **Fix person ReID config** (Option A) | HIGH — restores discriminative person features | 5 min config change |
| 2 | **Verify weight loading** for `vit_base_patch16_224` in TransReID wrapper | Required for #1 | 30 min investigation |
| 3 | **Re-run 12b** with fixed person ReID config | Validates the fix | Kaggle GPU run |

### This Week

| # | Action | Impact | Effort |
|:-:|--------|:------:|:------:|
| 4 | **Build 09f v3 notebook** with fixed recipe (no circle loss, bs=48, etc.) | HIGH — unlocks ResNet ensemble | 1 Kaggle GPU run |
| 5 | **Upload correct 384px ViT checkpoint** (09b v2, 80.14% mAP) | HIGH — +1-2.5pp vehicle IDF1 | Upload task |

### If 09f v3 Succeeds (mAP > 65%)

| # | Action | Impact | Effort |
|:-:|--------|:------:|:------:|
| 6 | Upload 09f checkpoint to Kaggle Models | Required for pipeline | Upload task |
| 7 | Enable 2-model ensemble in 10a pipeline | +1.5-2.5pp vehicle IDF1 | Config change |
| 8 | Re-sweep association with dual features | May recover +0.3-0.5pp | 10c parameter sweep |

---

## Appendix: Config Comparison Table

| Parameter | 09d v18 (52.77%) | 09e (62.52%) | 09f v1 (16.2%) | 09f v3 (proposed) |
|-----------|:-:|:-:|:-:|:-:|
| lr | 1e-3 | 3.5e-4 | 7e-5 | **3.5e-4** |
| differential LR | No | No | Yes (0.1×/1×/10×) | **Yes (0.1×/1×/10×)** |
| batch_size | 48 | ~48 | 32 | **48** |
| P (IDs/batch) | 12 | ~12 | 8 | **12** |
| epochs | 120 | 120 | 60 | **120** |
| circle_weight | 0.0 | 0.0 | 1.0 | **0.0** |
| label_smoothing | 0.05 | 0.1 | 0.1 | **0.05** |
| warmup_start | 0.01 | N/A | 0.01 | **0.1** |
| weight_decay | 5e-4 | 1e-4 | 5e-4 | **5e-4** |
| optimizer | AdamW | AdamW | AdamW | **AdamW** |
| fp16 | Yes | Yes | Yes | **Yes** |
| img_size | 384×384 | 384×384 | 384×384 | **384×384** |
| eval_every | 5 | 10 | 5 | **5** |
| VeRi pretrained | No | N/A (IS VeRi) | Yes | **Yes** |