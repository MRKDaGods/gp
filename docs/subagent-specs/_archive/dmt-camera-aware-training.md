# DMT Camera-Aware Training Spec for TransReID ViT-B/16 CLIP

## 1. Executive Summary

### Goal
Add camera-aware training losses to the TransReID ViT-B/16 CLIP model to improve cross-camera feature invariance, closing the ~6.5pp gap from our current MTMC IDF1 of 0.784 toward the SOTA target of ~0.849.

### Expected Impact
- **+1.5 to +2.5pp MTMC IDF1** (per findings.md estimate)
- Features become more camera-invariant → better cross-camera association
- Combined with CID_BIAS (separate effort), total expected gain: +2.0 to +3.5pp

### Recommended Approach
**Option A: Camera-ID Adversarial Training (Gradient Reversal Layer)** as the primary camera-aware loss, combined with **circle loss on a separate projection head** as a secondary pairwise signal. This is preferred because:
1. GRL is the simplest, most proven approach for domain-invariant features
2. It requires minimal changes to the existing architecture
3. Camera IDs are already passed through the forward pass (`cam_ids` argument)
4. It directly addresses the root problem: features that encode camera-specific appearance

---

## 2. Current Architecture (Baseline)

### Model: TransReID (src/stage2_features/transreid_model.py)
```
Input: (B, 3, 256, 256)
  → Patch embed → (B, 257, 768)  [256 patches + 1 CLS token]
  → Positional embedding
  → SIE camera embed (broadcast to all tokens)  ← ALREADY EXISTS
  → norm_pre (CLIP LayerNorm)
  → 12 Transformer blocks
  → Final LayerNorm
  → CLS token → g_feat (B, 768)
  → BNNeck → bn_feat (B, 768)
  → Projection → proj_feat (B, 768) [Identity for 768→768]
  
  Training outputs:
    cls_logits = cls_head(proj_feat)     → (B, num_classes)
    g_feat                                → (B, 768) [for triplet loss]
    [optional] jpm_logits                 → (B, num_classes)
  
  Inference output:
    L2-normalized proj_feat              → (B, 768)
```

### Current Loss Function
```python
total_loss = CE(cls_logits, pids)           # weight 1.0
           + Triplet(g_feat, pids)          # weight 1.0, margin=0.3
           + 0.5 * CE(jpm_logits, pids)     # weight 0.5 (JPM branch)
           + 5e-4 * CenterLoss(g_feat, pids) # starts at epoch 15
```

### Training Config
- **Optimizer**: AdamW with LLRD (backbone_lr=1e-4, head_lr=1e-3, decay=0.75)
- **Scheduler**: Linear warmup (10 epochs) → Cosine annealing
- **Epochs**: 120
- **Batch**: P=7 identities × K=2 = 14 samples per batch
- **Resolution**: 256×256
- **Mixed precision**: fp16 via torch.amp
- **Gradient clipping**: max_norm=5.0

### Available Metadata
- `cam_ids`: integer tensor (B,) — passed to model in every training step
- 59 unique cameras in CityFlowV2 training set
- Camera info already encoded in crop filenames: `{vid:04d}_{scene}_{camera}_f{frame:06d}.jpg`

---

## 3. Proposed Changes

### 3.1 Gradient Reversal Layer (New Module)

**File**: Add to the training notebook cell (self-contained, no external file needed on Kaggle)

```python
import torch
import torch.nn as nn
from torch.autograd import Function


class GradientReversalFunction(Function):
    """Reverses gradients during backpropagation.
    
    Forward: identity operation (pass-through)
    Backward: negate gradients and scale by lambda
    
    This forces the feature encoder to learn features that CANNOT
    discriminate between cameras, making embeddings camera-invariant.
    """
    @staticmethod
    def forward(ctx, x, lambda_val):
        ctx.lambda_val = lambda_val
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_val * grad_output, None


class GradientReversalLayer(nn.Module):
    """Wraps the GRL function for use in nn.Sequential or direct calls."""
    def __init__(self, lambda_val=1.0):
        super().__init__()
        self.lambda_val = lambda_val

    def set_lambda(self, val):
        self.lambda_val = val

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_val)
```

### 3.2 Camera Classification Head (Model Extension)

Add to the `TransReID.__init__()` method:

```python
# === DMT Camera-Aware Head ===
self.camera_aware = camera_aware  # new __init__ parameter, default=False
if self.camera_aware:
    self.grl = GradientReversalLayer(lambda_val=1.0)
    self.camera_head = nn.Sequential(
        nn.Linear(self.vit_dim, self.vit_dim // 2),  # 768 → 384
        nn.BatchNorm1d(self.vit_dim // 2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(self.vit_dim // 2, num_cameras),    # 384 → 59
    )
    # Initialize camera head
    for m in self.camera_head.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
```

Add to `TransReID.forward()`, after `g_feat = x[:, 0]`:

```python
# === DMT: Camera adversarial prediction ===
cam_logits = None
if self.training and self.camera_aware and cam_ids is not None:
    # GRL is applied to raw features (before BNNeck)
    # This ensures the backbone learns camera-invariant representations
    reversed_feat = self.grl(g_feat)
    cam_logits = self.camera_head(reversed_feat)
```

Modify training-mode return:

```python
if self.training:
    cls = self.cls_head(proj)
    outputs = [cls, g_feat]  # changed from proj to g_feat for triplet
    if self.jpm:
        # ... JPM code ...
        outputs.append(jpm_cls)
    if cam_logits is not None:
        outputs.append(cam_logits)
    return tuple(outputs)
```

**Important**: The return signature becomes:
- Without DMT: `(cls_logits, g_feat)` or `(cls_logits, g_feat, jpm_logits)` — unchanged
- With DMT + no JPM: `(cls_logits, g_feat, cam_logits)`
- With DMT + JPM: `(cls_logits, g_feat, jpm_logits, cam_logits)`

The training loop should detect DMT output by checking the length of the output tuple.

### 3.3 Separate Circle Loss Projection Head (Model Extension)

**Critical lesson**: Circle loss on SHARED features was catastrophic for ResNet101 (-50pp mAP). We MUST use a separate projection head.

Add to `TransReID.__init__()`:

```python
# === Separate Circle Loss Head ===
self.use_circle = use_circle  # new __init__ parameter, default=False
if self.use_circle:
    self.circle_proj = nn.Sequential(
        nn.Linear(self.vit_dim, 512),
        nn.BatchNorm1d(512),
    )
    nn.init.kaiming_normal_(self.circle_proj[0].weight, mode='fan_out')
    self.circle_proj[0].bias = None  # no bias before BN
```

Add to `TransReID.forward()`:

```python
# === Circle loss features (separate head, NOT shared with triplet) ===
circle_feat = None
if self.training and self.use_circle:
    circle_feat = self.circle_proj(g_feat)  # (B, 512)
```

Include `circle_feat` in the output tuple when present.

### 3.4 Training Loop Changes

The core training loop in the 09b notebook needs these modifications:

```python
# === Hyperparameters ===
USE_DMT = True
USE_CIRCLE = True
CAM_LOSS_WEIGHT = 0.3          # Conservative start
CIRCLE_LOSS_WEIGHT = 0.5       # Moderate weight
CIRCLE_SCALE = 64              # Circle loss gamma
CIRCLE_MARGIN = 0.25           # Circle loss margin
DMT_WARMUP_EPOCHS = 15         # Only enable camera loss after N epochs
GRL_LAMBDA_MAX = 1.0           # Max gradient reversal strength
GRL_LAMBDA_SCHEDULE = "linear" # "linear" warmup from 0 → max over remaining epochs

# === Model instantiation ===
model = TransReID(
    num_classes=num_classes,
    num_cameras=NUM_CAMERAS,     # 59 for CityFlowV2
    embed_dim=768,
    vit_model="vit_base_patch16_clip_224.openai",
    pretrained=True,
    sie_camera=True,
    jpm=True,
    img_size=(256, 256),
    camera_aware=USE_DMT,        # NEW
    use_circle=USE_CIRCLE,       # NEW
)

# === Loss functions ===
ce_loss = CrossEntropyLabelSmooth(num_classes, epsilon=0.05)
tri_loss = TripletLossHardMining(margin=0.3)
ctr_loss = CenterLoss(num_classes, feat_dim=768)
cam_ce_loss = nn.CrossEntropyLoss()  # NEW: for camera classification
circle_loss_fn = CircleLoss(m=CIRCLE_MARGIN, gamma=CIRCLE_SCALE)  # NEW

# === Optimizer: add new heads to param groups ===
# Camera head gets HEAD_LR (1e-3), not backbone LR
# Circle projection gets HEAD_LR
# Add these to the existing LLRD groups:
if USE_DMT:
    param_groups.append({"params": model.camera_head.parameters(), "lr": HEAD_LR})
if USE_CIRCLE:
    param_groups.append({"params": model.circle_proj.parameters(), "lr": HEAD_LR})

# === GRL Lambda Schedule ===
def get_grl_lambda(epoch, dmt_warmup, max_epochs, lambda_max):
    """Linear ramp from 0 to lambda_max after warmup."""
    if epoch < dmt_warmup:
        return 0.0
    progress = (epoch - dmt_warmup) / max(max_epochs - dmt_warmup, 1)
    return lambda_max * min(progress, 1.0)

# === Modified training step ===
for epoch in range(EPOCHS):
    # Update GRL lambda
    if USE_DMT:
        lam = get_grl_lambda(epoch, DMT_WARMUP_EPOCHS, EPOCHS, GRL_LAMBDA_MAX)
        model.grl.set_lambda(lam)
    
    for imgs, pids, cams, _ in train_loader:
        imgs = imgs.to(DEVICE)
        pids = pids.to(DEVICE).long()
        cams = cams.to(DEVICE).long()
        
        with torch.amp.autocast("cuda"):
            outputs = model(imgs, cam_ids=cams)
            
            # Parse outputs based on configuration
            # Base: (cls_logits, g_feat)
            # +JPM: (cls_logits, g_feat, jpm_logits)
            # +DMT: appends cam_logits
            # +Circle: appends circle_feat
            idx = 0
            cls_logits = outputs[idx]; idx += 1
            g_feat = outputs[idx]; idx += 1
            
            jpm_logits = None
            if model.jpm:
                jpm_logits = outputs[idx]; idx += 1
            
            cam_logits = None
            if USE_DMT and model.camera_aware:
                cam_logits = outputs[idx]; idx += 1
            
            circle_feat = None
            if USE_CIRCLE and model.use_circle:
                circle_feat = outputs[idx]; idx += 1
            
            # === Core losses (unchanged) ===
            loss = ce_loss(cls_logits, pids) + tri_loss(g_feat, pids)
            
            if jpm_logits is not None:
                loss += 0.5 * ce_loss(jpm_logits, pids)
            
            # === DMT camera adversarial loss ===
            if cam_logits is not None and epoch >= DMT_WARMUP_EPOCHS:
                cam_loss = cam_ce_loss(cam_logits, cams)
                loss += CAM_LOSS_WEIGHT * cam_loss
            
            # === Circle loss on separate head ===
            if circle_feat is not None:
                c_loss = circle_loss_fn(circle_feat, pids)
                loss += CIRCLE_LOSS_WEIGHT * c_loss
            
            # === Center loss (existing, starts at epoch 15) ===
            if epoch >= 15:
                loss += CENTER_WEIGHT * ctr_loss(g_feat.float(), pids)
        
        # ... standard backward pass, optimizer step, etc.
```

### 3.5 Data Loader Changes

**No changes needed.** The existing data loader already:
1. Parses camera IDs from filenames
2. Returns `(img, pid, cam_id, path)` tuples
3. Passes `cam_ids` to `model(imgs, cam_ids=cams)`

The PKSampler (P=7 identities, K=2 instances) naturally creates batches with cross-camera pairs for the same identity, which is exactly what the GRL needs to learn camera-invariant features.

---

## 4. Training Recipe

### 4.1 Schedule Overview

| Epoch Range | Active Losses | GRL λ | Notes |
|---|---|---|---|
| 0-9 | ID + Triplet + JPM | 0.0 | Warmup phase, no camera loss |
| 10-14 | ID + Triplet + JPM + Circle | 0.0 | Circle loss activates, still no camera loss |
| 15-30 | ID + Triplet + JPM + Circle + Center + **DMT** | 0.0→0.14 | Camera loss ramps up slowly |
| 30-120 | All losses active | 0.14→1.0 | Full training with all losses |

### 4.2 Hyperparameter Defaults

| Parameter | Value | Rationale |
|---|---|---|
| `CAM_LOSS_WEIGHT` | 0.3 | Conservative — too high destroys ID discrimination |
| `CIRCLE_LOSS_WEIGHT` | 0.5 | Moderate — separate head isolates risk |
| `CIRCLE_SCALE` (gamma) | 64 | Standard value from Circle Loss paper |
| `CIRCLE_MARGIN` (m) | 0.25 | Standard value from Circle Loss paper |
| `DMT_WARMUP_EPOCHS` | 15 | Let ID features stabilize before adversarial training |
| `GRL_LAMBDA_MAX` | 1.0 | Full gradient reversal at convergence |
| `GRL_LAMBDA_SCHEDULE` | linear | Simple, proven in DANN literature |
| `CENTER_WEIGHT` | 5e-4 | Unchanged from baseline |
| Total epochs | 120 | Unchanged from baseline |
| Batch (P×K) | 7×2 = 14 | Unchanged from baseline |

### 4.3 Expected Training Time

On a single P100 GPU (Kaggle):
- Current 256px training: ~3-4 hours for 120 epochs
- With DMT head: +~5% overhead (one extra Linear layer forward/backward)
- With Circle loss head: +~10% overhead (pairwise similarity computation)
- **Total estimated: ~4-5 hours** — well within Kaggle's 12-hour limit

### 4.4 Evaluation Strategy

After training, evaluate using the **existing 10a→10b→10c pipeline chain**:
1. Export the best checkpoint (by mAP or combined metric)
2. Run 10a (stages 0-2) with the new weights
3. Run 10b (stage 3: indexing)
4. Run 10c (stages 4-5: association + evaluation)
5. Compare MTMC IDF1 against the 0.784 baseline

**Key metric to monitor during training:**
- `cam_accuracy`: camera classification accuracy on the REVERSED features
  - If high: GRL lambda is too low, features still encode camera info
  - If at chance (~1/59 = 1.7%): GRL is working perfectly
  - Target: declining accuracy over training, approaching random

---

## 5. Risk Mitigation

### Risk 1: Camera loss too strong → identity discrimination collapses
- **Mitigation**: DMT_WARMUP_EPOCHS = 15 (features stabilize first)
- **Mitigation**: CAM_LOSS_WEIGHT = 0.3 (conservative)
- **Mitigation**: GRL lambda ramps linearly from 0 (gradual introduction)
- **Monitoring**: Track ID CE loss — if it diverges after DMT activates, reduce CAM_LOSS_WEIGHT
- **Fallback**: Set CAM_LOSS_WEIGHT = 0.0 to disable without restarting

### Risk 2: Circle loss conflicts with triplet loss (repeat of ResNet disaster)
- **Mitigation**: Circle loss operates on a SEPARATE 512D projection head
- **Mitigation**: Triplet loss operates on the raw 768D g_feat (unchanged)
- **Mitigation**: No shared parameters between circle and triplet feature paths
- **Monitoring**: Track triplet loss convergence — should be unaffected by circle loss
- **Fallback**: Set CIRCLE_LOSS_WEIGHT = 0.0 to disable

### Risk 3: SIE embedding conflicts with GRL objective
- **Analysis**: SIE adds camera information INTO the features (encouraging camera-awareness), while GRL tries to REMOVE camera information. These objectives partially conflict.
- **Resolution**: This is actually beneficial — SIE helps the model USE camera position as context (e.g., knowing viewing angle), while GRL prevents camera-SPECIFIC appearance artifacts (lighting, color) from dominating the embedding. They operate at different abstraction levels.
- **Monitoring**: If combined training doesn't converge, try disabling SIE (`sie_camera=False`)

### Risk 4: Training instability with adversarial objectives
- **Mitigation**: Gradient clipping at max_norm=5.0 (already in baseline)
- **Mitigation**: fp16 with GradScaler (already in baseline)
- **Mitigation**: Linear GRL lambda schedule (not sudden activation)
- **Monitoring**: Watch for loss oscillation or NaN values after epoch 15

---

## 6. Monitoring & Logging

Add these metrics to the per-epoch training log:

```python
# After each epoch, compute and log:
metrics = {
    "epoch": epoch,
    "id_loss": avg_id_loss,
    "tri_loss": avg_tri_loss,
    "jpm_loss": avg_jpm_loss,
    "cam_loss": avg_cam_loss if USE_DMT else 0.0,
    "circle_loss": avg_circle_loss if USE_CIRCLE else 0.0,
    "center_loss": avg_center_loss,
    "total_loss": avg_total_loss,
    "grl_lambda": lam if USE_DMT else 0.0,
    "cam_accuracy": cam_correct / cam_total if USE_DMT else 0.0,
    "train_id_accuracy": id_correct / id_total,
    "val_mAP": val_map,
    "val_rank1": val_rank1,
}
```

### Key diagnostic signals:
1. **cam_accuracy declining**: GRL is working, features becoming camera-invariant ✅
2. **cam_accuracy stuck at high**: GRL lambda too low, increase or check gradients ⚠️
3. **id loss diverging after epoch 15**: CAM_LOSS_WEIGHT too high, reduce ⚠️
4. **tri_loss unaffected by circle loss**: Separate heads working correctly ✅
5. **val_mAP improving**: Overall feature quality improving ✅

---

## 7. Implementation Order

### Phase 1: GRL + Camera Head Only (Priority)
1. Add `GradientReversalFunction` and `GradientReversalLayer` classes
2. Add `camera_head` to TransReID model
3. Add `cam_logits` to forward pass output
4. Add camera CE loss to training loop with warmup
5. Add GRL lambda scheduling
6. Train and evaluate MTMC IDF1

### Phase 2: Separate Circle Loss Head (After Phase 1 validates)
1. Add `circle_proj` head to TransReID model
2. Add `circle_feat` to forward pass output
3. Add CircleLoss computation on separate features
4. Train combined model and evaluate

### Phase 3: Hyperparameter Sweep (If Phase 1+2 show improvement)
1. Sweep CAM_LOSS_WEIGHT: [0.1, 0.2, 0.3, 0.5]
2. Sweep CIRCLE_LOSS_WEIGHT: [0.3, 0.5, 0.7, 1.0]
3. Sweep DMT_WARMUP_EPOCHS: [10, 15, 20]
4. Sweep GRL_LAMBDA_MAX: [0.5, 1.0, 2.0]

---

## 8. Files to Modify

| File | Change | Priority |
|---|---|---|
| `notebooks/kaggle/09b_vehicle_reid_cityflowv2/09b_vehicle_reid_cityflowv2.ipynb` | Add GRL classes, modify model init, modify forward, modify training loop | P0 |
| `src/stage2_features/transreid_model.py` | Add camera_aware and circle head support (for local testing only) | P1 |
| `src/training/losses.py` | Already has CircleLoss — no changes needed | — |
| `configs/default.yaml` | Add `stage2.reid.vehicle.camera_aware: true`, `stage2.reid.vehicle.use_circle: true` | P2 |

### Note on Kaggle Notebook
The 09b notebook is self-contained for Kaggle execution. All new code (GRL, camera head, circle head) should be defined INLINE in the notebook cells, not imported from `src/`. The `src/` files are updated for consistency and local testing only.

---

## 9. Ablation Matrix

For rigorous evaluation, run these configurations:

| Experiment | GRL | Circle | CAM_WEIGHT | Baseline? |
|---|---|---|---|---|
| v_baseline | ✗ | ✗ | — | ✅ Current best |
| v_grl_only | ✅ | ✗ | 0.3 | |
| v_circle_only | ✗ | ✅ | — | |
| v_grl_circle | ✅ | ✅ | 0.3 | Target config |
| v_grl_strong | ✅ | ✗ | 0.5 | If 0.3 is too weak |

---

## 10. Why NOT the Other Options

### Option B: Cross-Camera Contrastive Loss
- Requires modifying the PKSampler to guarantee cross-camera positive pairs per batch
- With P=7, K=2 and 59 cameras, many batches won't have useful cross-camera pairs
- More complex implementation for similar theoretical benefit
- **Verdict**: Implement if GRL doesn't work, but GRL is simpler and more principled

### Option C: Camera-Uniform Prior (CUP)
- Requires computing per-camera feature distributions (expensive with 59 cameras)
- KL divergence between 59 distributions is noisy with small batch sizes
- Less proven in vehicle ReID literature
- **Verdict**: Too complex and fragile for our setup

---

## 11. Relationship to CID_BIAS

Camera-aware training (this spec) and CID_BIAS (separate effort) are **complementary**:

- **DMT**: Improves feature quality at TRAINING time → camera-invariant embeddings
- **CID_BIAS**: Adjusts similarity scores at INFERENCE time → camera-pair calibration

They operate at different stages and should stack. The recommended deployment order from findings.md is:
1. CID_BIAS first (ready to implement, no retraining needed)
2. DMT retraining second (this spec)
3. Combine both for maximum gain

---

## 12. Success Criteria

| Metric | Baseline | Target | Stretch |
|---|---|---|---|
| Single-camera mAP | ~80% | ≥80% (no regression) | >82% |
| MTMC IDF1 (Kaggle) | 0.784 | ≥0.800 | ≥0.810 |
| Camera accuracy (train) | N/A | <10% (features are camera-invariant) | <5% |
| Training time (P100) | ~3.5h | <5h | <4h |
