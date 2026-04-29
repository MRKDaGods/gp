# Spec: 09r-v4-fix — ViT-L Training Crash Diagnosis & Fixes

## Notebook
`notebooks/kaggle/09r_vit_large/09r_vit_large_cityflowv2.ipynb`

## Runtime Error Context
- Error at ~2.5h total: crops ~1–1.5h, training ~1h (approx 10–20 epochs completed)
- Hardware: T4 x2 (16 GB each), DataParallel
- BATCH_SIZE=64 (32 per GPU), ViT-L 307M params, AMP enabled, img_size=256
- CENTER_START=0 (center loss active from epoch 1)

---

## Cell 4 — Config (key values)

```python
BATCH_SIZE = 64
PIDS_PER_BATCH = 16
INSTANCES_PER_PID = 4
CENTER_START = 0          # center loss active from epoch 1
CENTER_LOSS_WEIGHT = 0.0005
IMG_SIZE = 256
NUM_EPOCHS = 120
EVAL_CHUNK_SIZE = 1024
```

---

## Cell 7 — Verbatim Source

```python
BACKBONE_ALIASES = {
    "vit_large_patch16_224_TransReID": [
        "vit_large_patch16_224.augreg_in21k_ft_in1k",
        "vit_large_patch16_224.augreg_in21k",
        "vit_large_patch16_224",
    ],
    ...
}

class TransReID(nn.Module):
    def __init__(self, num_classes, num_cameras=0, embed_dim=768,
                 vit_model="vit_base_patch16_224_TransReID", pretrained=True,
                 sie_camera=True, jpm=True, img_size=224, stride_size=16):
        super().__init__()
        ...
        self.vit = timm.create_model(
            self.timm_backbone, pretrained=pretrained, num_classes=0, img_size=img_size,
        )
        # ← NO gradient checkpointing call here
        self.vit_dim = self.vit.embed_dim
        self.num_blocks = len(self.vit.blocks)

        if self.sie_camera:
            self.sie_embed = nn.Parameter(torch.zeros(num_cameras, 1, self.vit_dim))
            nn.init.trunc_normal_(self.sie_embed, std=0.02)
        ...

    def forward(self, x, cam_ids=None):
        batch_size = x.shape[0]
        rot_pos_embed = None

        x = self.vit.patch_embed(x)
        if hasattr(self.vit, "_pos_embed"):
            pos_result = self.vit._pos_embed(x)
            if isinstance(pos_result, tuple):
                x, rot_pos_embed = pos_result
            else:
                x = pos_result
        else:
            cls_token = self.vit.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_token, x], dim=1) + self.vit.pos_embed
            if hasattr(self.vit, "pos_drop"):
                x = self.vit.pos_drop(x)

        if self.sie_camera and cam_ids is not None:
            x = x + self.sie_embed[cam_ids]    # BUG-3: broadcasts to ALL tokens [B,257,dim]

        ...

        for block in self.vit.blocks:           # BUG-1: no gradient checkpointing — all 24 layers live in GPU mem
            if rot_pos_embed is not None:
                x = block(x, rope=rot_pos_embed)
            else:
                x = block(x)
        x = self.vit.norm(x)

        global_feat = x[:, 0]
        bn_feat = self.bn(global_feat)
        proj_feat = self.proj(bn_feat)

        if self.training:
            cls_logits = self.cls_head(proj_feat)
            if self.jpm:
                patches = x[:, 1:]
                shuffle_index = torch.randperm(patches.size(1), device=x.device)
                shuffled = patches[:, shuffle_index]
                midpoint = shuffled.size(1) // 2
                jpm_feat = (shuffled[:, :midpoint].mean(1) + shuffled[:, midpoint:].mean(1)) / 2
                jpm_logits = self.jpm_cls(self.bn_jpm(jpm_feat))
                return cls_logits, proj_feat, jpm_logits
            return cls_logits, proj_feat
        return F.normalize(proj_feat, p=2, dim=1)

    def get_llrd_param_groups(self, backbone_lr, head_lr, decay=0.75):
        groups = {}
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith("vit."):
                if "blocks." in name:
                    block_index = int(name.split("blocks.")[1].split(".")[0])
                    depth = block_index + 1
                elif any(t in name for t in ("patch_embed", "cls_token", "pos_embed", "norm_pre")):
                    depth = 0
                else:
                    depth = self.num_blocks + 1
                scale = decay ** (self.num_blocks + 1 - depth)
                lr = backbone_lr * scale
                group_key = f"backbone_{depth}"
            else:
                lr = head_lr
                group_key = "head"
            groups.setdefault(group_key, {"params": [], "lr": lr})["params"].append(parameter)
        return sorted(groups.values(), key=lambda item: item["lr"])

model = TransReID(
    num_classes=num_classes,
    num_cameras=num_cameras,
    embed_dim=1024,
    vit_model=VIT_MODEL,
    pretrained=True,
    sie_camera=True,
    jpm=True,
    img_size=IMG_SIZE,
    stride_size=STRIDE_SIZE,
).to(DEVICE)

if torch.cuda.device_count() > 1:
    print(f"Using DataParallel across {torch.cuda.device_count()} GPUs")
    model = torch.nn.DataParallel(model)

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

---

## Cell 8 — Verbatim Source

```python
class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon

    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=1)
        target_probs = F.one_hot(targets, num_classes=self.num_classes).float()
        target_probs = (1.0 - self.epsilon) * target_probs + self.epsilon / self.num_classes
        return (-target_probs * log_probs).sum(dim=1).mean()

class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, features, labels):
        labels = labels.long()
        batch_centers = self.centers.index_select(0, labels)
        return 0.5 * (features - batch_centers).pow(2).sum(dim=1).mean()

def batch_hard_triplet_loss(features, labels, margin=0.3):
    normalized = F.normalize(features, p=2, dim=1)
    distance = torch.cdist(normalized, normalized, p=2)
    labels = labels.view(-1, 1)
    positive_mask = labels.eq(labels.t())
    eye_mask = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
    positive_mask = positive_mask & ~eye_mask
    negative_mask = ~labels.eq(labels.t())
    if positive_mask.sum() == 0 or negative_mask.sum() == 0:
        return features.sum() * 0.0
    hardest_positive = distance.masked_fill(~positive_mask, float("-inf")).max(dim=1).values
    hardest_negative = distance.masked_fill(~negative_mask, float("inf")).min(dim=1).values
    valid = torch.isfinite(hardest_positive) & torch.isfinite(hardest_negative)
    if valid.sum() == 0:
        return features.sum() * 0.0
    return F.relu(hardest_positive[valid] - hardest_negative[valid] + margin).mean()

ce_loss = CrossEntropyLabelSmooth(num_classes=num_classes, epsilon=0.05).to(DEVICE)
center_loss_fn = CenterLoss(num_classes=num_classes, feat_dim=1024).to(DEVICE)

raw_model = model.module if hasattr(model, "module") else model
optimizer = torch.optim.AdamW(
    raw_model.get_llrd_param_groups(BACKBONE_LR, HEAD_LR, decay=0.75),
    weight_decay=WEIGHT_DECAY,
)
center_optimizer = torch.optim.SGD(center_loss_fn.parameters(), lr=0.5)
base_lrs = [group["lr"] for group in optimizer.param_groups]
scaler = torch.amp.GradScaler("cuda", enabled=DEVICE == "cuda")

def set_epoch_lrs(epoch_index):
    if epoch_index < WARMUP_EPOCHS:
        scale = float(epoch_index + 1) / float(max(WARMUP_EPOCHS, 1))
    else:
        progress = float(epoch_index - WARMUP_EPOCHS + 1) / float(max(NUM_EPOCHS - WARMUP_EPOCHS, 1))
        scale = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    for group, base_lr in zip(optimizer.param_groups, base_lrs):
        group["lr"] = base_lr * scale

history = {"loss": [], "train_acc": []}
best_train_acc = -1.0
best_epoch = -1
final_train_acc = 0.0
eval_metrics = {}

if len(train_loader) == 0:
    raise RuntimeError("Train loader is empty; crop extraction or splitting failed")

for epoch in range(1, NUM_EPOCHS + 1):
    set_epoch_lrs(epoch - 1)
    model.train()
    running_loss = 0.0
    total_seen = 0
    total_correct = 0
    total_batches = 0
    use_center = (epoch - 1) >= CENTER_START

    for images, pids, cams, _ in train_loader:
        images = images.to(DEVICE, non_blocking=True)
        pids   = pids.to(DEVICE, non_blocking=True).long()
        cams   = cams.to(DEVICE, non_blocking=True).long()

        optimizer.zero_grad(set_to_none=True)
        if use_center:
            center_optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=DEVICE == "cuda"):
            outputs = model(images, cam_ids=cams)
            if len(outputs) == 3:
                cls_logits, features, jpm_logits = outputs
                id_loss = ce_loss(cls_logits, pids) + 0.5 * ce_loss(jpm_logits, pids)
            else:
                cls_logits, features = outputs
                id_loss = ce_loss(cls_logits, pids)
            triplet_loss = batch_hard_triplet_loss(features, pids, margin=MARGIN)
            total_loss = id_loss + triplet_loss
            if use_center:
                center_loss = center_loss_fn(features.float(), pids)   # BUG-2: fp32 cast in fp16 block
                total_loss = total_loss + CENTER_LOSS_WEIGHT * center_loss

        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)                             # only unscales main optimizer
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        if use_center:
            scaler.step(center_optimizer)                      # BUG-2: center grads not clipped
        scaler.update()

        predictions = cls_logits.argmax(dim=1)
        total_correct += int((predictions == pids).sum().item())
        total_seen    += int(pids.numel())
        running_loss  += float(total_loss.detach().item())
        total_batches += 1

    epoch_loss = running_loss / max(total_batches, 1)
    train_acc  = total_correct / max(total_seen, 1)
    history["loss"].append(epoch_loss)
    history["train_acc"].append(train_acc)
    final_train_acc = train_acc

    if train_acc > best_train_acc:
        best_train_acc = train_acc
        best_epoch     = epoch
        torch.save(raw_model.state_dict(), BEST_MODEL_PATH)

    if epoch == 1 or epoch % 10 == 0 or epoch == NUM_EPOCHS:
        print(
            f"Epoch {epoch:03d}: loss={epoch_loss:.4f}, acc={train_acc:.4f}, "
            f"head_lr={optimizer.param_groups[-1]['lr']:.2e}"
        )

print(f"Training complete | Best epoch: {best_epoch} | Final train acc: {final_train_acc:.4f}")
```

---

## Cell 9 — Verbatim Source

```python
@torch.no_grad()
def extract_features(model, loader, device="cuda", flip=True, pass_cams=False):
    model.eval()
    features, pids, cams = [], [], []
    for images, pid, cam, _ in loader:
        images = images.to(device)
        kwargs = {"cam_ids": cam.to(device).long()} if pass_cams else {}
        feats = model(images, **kwargs)
        if isinstance(feats, (tuple, list)):
            feats = feats[-1]
        if flip:
            flipped = model(torch.flip(images, [3]), **kwargs)
            if isinstance(flipped, (tuple, list)):
                flipped = flipped[-1]
            feats = (feats + flipped) / 2.0
        feats = F.normalize(feats, p=2, dim=1)
        features.append(feats.cpu().numpy())
        pids.append(pid.numpy())
        cams.append(cam.numpy())
    if not features:
        return (np.zeros((0, 1024), np.float32), np.zeros((0,), np.int64), np.zeros((0,), np.int64))
    return np.concatenate(features), np.concatenate(pids), np.concatenate(cams)

def eval_market1501_chunked(query_features, gallery_features, q_pids, g_pids,
                             q_cams, g_cams, chunk_size=1024, max_rank=50):
    # CPU-only numpy distance matrix + CMC + AP — no GPU memory risk
    ...

if BEST_MODEL_PATH.exists():
    best_state = torch.load(BEST_MODEL_PATH, map_location="cpu", weights_only=False)
    raw_model.load_state_dict(best_state, strict=False)

if CAN_EVAL:
    query_features, query_pids, query_cams = extract_features(
        raw_model, query_loader, DEVICE, pass_cams=True
    )
    gallery_features, gallery_pids, gallery_cams = extract_features(
        raw_model, gallery_loader, DEVICE, pass_cams=True
    )
    mean_ap, cmc = eval_market1501_chunked(...)
    ...
```

---

## Bug Analysis

### Bug 1 — CRITICAL (PRIMARY CRASH CAUSE): No gradient checkpointing on ViT-L
**Location**: Cell 7, `TransReID.__init__()`, after `self.vit = timm.create_model(...)`

**Description**: ViT-L has 24 transformer blocks. Without gradient checkpointing,
ALL 24 layers' activations are retained in GPU memory during the backward pass.

**Memory breakdown** (batch=32/GPU, seq_len=257, fp16 AMP):

| Component | Size |
|---|---|
| Per-block attn weights [32, 16, 257, 257] fp16 | 68 MB |
| Per-block MLP intermediate [32, 257, 4096] fp16 | 67 MB |
| Per-block input activation [32, 257, 1024] fp16 | 17 MB |
| Per-block total | ~150 MB |
| 24 blocks total | **~3.6 GB** |

**GPU 0 peak (DataParallel: optimizer state lives on GPU 0)**:

| Component | Memory |
|---|---|
| Model master weights fp32 | 1.2 GB |
| AdamW m+v state fp32 | 2.4 GB |
| Gradients fp32 | 1.2 GB |
| Activations fp16 (no grad ckpt) | 3.6 GB |
| DataParallel gradient reduction | ~1.0 GB |
| **Total peak** | **~9.4 GB** |

9.4 GB is within 16 GB but TIGHT. PyTorch CUDA allocator fragmentation
accumulates over 15-20 training epochs (many alloc/free cycles of different
tensor sizes). Eventually it cannot find a contiguous block for a new allocation
even though total free memory > needed. This triggers a CUDA OOM error.

**Fix** — Cell 7, `TransReID.__init__`, right after `self.vit = timm.create_model(...)`:
```python
# Enable gradient checkpointing to reduce activation memory from ~3.6GB to ~0.15GB
if hasattr(self.vit, "set_grad_checkpointing"):
    self.vit.set_grad_checkpointing(enable=True)
    print("[INFO] Gradient checkpointing enabled for ViT backbone")
```

With gradient checkpointing: only 1 block's activations live in memory at a time.
GPU 0 peak drops to ~5.8 GB. Trade-off: ~20-25% slower training (recomputes forward
through each block during backward). Fully acceptable for a 120-epoch run.

---

### Bug 2 — HIGH: Center optimizer gradients not clipped; AMP inconsistency
**Location**: Cell 8, training loop, ~line after `scaler.unscale_(optimizer)`

**Description**:
```python
scaler.unscale_(optimizer)           # only unscales raw_model params
torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=5.0)  # clips model only
scaler.step(optimizer)
if use_center:
    scaler.step(center_optimizer)    # center_loss_fn.centers: NOT clipped
```

`center_loss_fn.centers` gradients are never clipped. With SGD lr=0.5 and
CENTER_START=0 (active from epoch 1), early-training center gradients can be
very large (features are near-random, distances large). This can cause:
1. Exploding center updates → NaN in center loss → NaN in total_loss → AMP
   scales down → training stalls / output diverges within a few epochs
2. Inf/NaN in the scaled center gradients causing `scaler.step()` to skip
   steps, compounding the scaling issue

Also: `center_loss_fn(features.float(), pids)` performs fp32 arithmetic inside
the fp16 autocast context. The gradient from center loss to `features` is fp32
while ID/triplet loss gradients are fp16. PyTorch internally accumulates fp32
gradients for the model, but the mixed-dtype gradient flow can cause the AMP
scaler to see numerical issues in early epochs.

**Fix**:
```python
scaler.unscale_(optimizer)
if use_center:
    scaler.unscale_(center_optimizer)                                        # ADD
torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=5.0)
if use_center:
    torch.nn.utils.clip_grad_norm_(center_loss_fn.parameters(), max_norm=5.0)  # ADD
scaler.step(optimizer)
if use_center:
    scaler.step(center_optimizer)
scaler.update()
```

---

### Bug 3 — MEDIUM (quality, not crash): SIE embed added to ALL 257 tokens
**Location**: Cell 7, `TransReID.forward()`, sie_camera block

**Description**:
```python
# Current — broadcasts camera embed to ALL tokens including cls
x = x + self.sie_embed[cam_ids]    # [B,1,dim] + [B,257,dim] → broadcasts
```
In the original TransReID paper, SIE adds camera-specific offsets to the patch
position embeddings only (not cls token). Adding it to the cls token contaminates
the classification token with camera identity bias, hurting cross-camera matching.

**Fix**:
```python
if self.sie_camera and cam_ids is not None:
    # Apply SIE to patch tokens only (index 1:), preserve cls token (index 0)
    x[:, 1:] = x[:, 1:] + self.sie_embed[cam_ids]
```

---

## Memory Estimation Summary

### Is BATCH_SIZE=64 safe for ViT-L on T4 x2?

| Config | GPU 0 Peak | Status |
|---|---|---|
| batch=64, no grad ckpt | ~9.4 GB + fragmentation risk | UNSAFE long-term |
| batch=64, grad ckpt enabled | ~5.8 GB | SAFE |
| batch=48, no grad ckpt | ~8.0 GB + fragmentation risk | BORDERLINE |
| batch=32, no grad ckpt | ~6.4 GB | SAFE |

**Recommendation**: Keep BATCH_SIZE=64 AND enable gradient checkpointing (Bug 1 fix).
If gradient checkpointing is skipped, reduce to BATCH_SIZE=32 (PIDS_PER_BATCH=16,
INSTANCES_PER_PID=2). Do NOT try batch=48 with INSTANCES_PER_PID=3 — PK sampler
requires BATCH_SIZE == PIDS_PER_BATCH * INSTANCES_PER_PID.

### Should gradient checkpointing be added to ViT-L?
YES. ViT-L is 307M params with 24 blocks. Without it, the activation memory alone
(3.6 GB) plus the optimizer state on GPU 0 (3.6 GB) leaves only ~2.8 GB headroom
on a 16 GB T4 — insufficient to survive 100+ epochs of varying allocation patterns.
timm's `set_grad_checkpointing()` is a one-line fix with full autograd support.

---

## Implementation Checklist for Coder

- [ ] Cell 7 `__init__`: Add `self.vit.set_grad_checkpointing(enable=True)` after timm.create_model
- [ ] Cell 7 `forward`: Change SIE to `x[:, 1:] = x[:, 1:] + self.sie_embed[cam_ids]`
- [ ] Cell 8: Add `scaler.unscale_(center_optimizer)` before clip_grad_norm
- [ ] Cell 8: Add `clip_grad_norm_` for `center_loss_fn.parameters()`
- [ ] Cell 4: Optionally add print confirming grad checkpointing is active
- [ ] No changes needed in Cell 9