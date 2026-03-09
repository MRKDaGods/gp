"""Create v14 of NB08: 256x256 resolution + v8 proven settings.
Modifies the notebook in-place."""
import json, pathlib

NB_PATH = pathlib.Path(__file__).resolve().parent.parent / '08_vehicle_reid_sota' / '08_vehicle_reid_sota.ipynb'
nb = json.load(open(NB_PATH, 'r', encoding='utf-8'))

def get_src(ci):
    return ''.join(nb['cells'][ci]['source'])

def set_src(ci, text):
    lines = text.split('\n')
    nb['cells'][ci]['source'] = [l + '\n' for l in lines[:-1]] + [lines[-1]]

# ════════════════════════════════════════════════════════════════════
# Cell 0: Header
# ════════════════════════════════════════════════════════════════════
print("Cell 0: Header")
set_src(0, """# Notebook 08: Vehicle ReID — TransReID Training (v14)
**Multi-Camera Tracking System — Kaggle Training Pipeline**

Train SOTA vehicle re-identification on VeRi-776 using TransReID ViT-Base with CLIP pretraining.

## Model
| Model | Architecture | Dim | Target mAP | Target R1 |
|-------|-------------|-----|-------------|-------------|
| **TransReID** | ViT-Base/16 (CLIP) + SIE + JPM | 768 | >82% | >97% |

## Key Features (v14)
- All v6 fixes: norm_pre, SIE all tokens, LLRD(0.75), OpenAI CLIP
- v6 proven hyperparameters: backbone_lr=3.5e-4, head_lr=3.5e-3
- **v8 proven recipe**: Triplet(0.3), ε=0.1, 140 epochs (80.4% mAP)
- **NEW: 256×256 input** (up from 224 — more spatial detail for vehicles)
- Center loss activated after epoch 30 (delayed start for stability)
- Random Erasing (p=0.5, random fill — proven in TransReID paper)
- TTA: horizontal flip averaging at eval time

**v8 → v14 changes (one change only):**
| Change | v8 | v14 | Rationale |
|--------|----|----|-----------|
| Input resolution | 224×224 | **256×256** | 30% more pixels → finer vehicle details (plates, decals) |
| Batch size | 96 (P=24,K=4) | 80 (P=20,K=4) | Reduced for higher-res memory budget |

**Reverted from v9/v8+ (proven harmful):**
- ε=0.15 → ε=0.1 (0.15 hurt by ~3% mAP)
- 180 epochs → 140 (peaked at 120-140 then degraded)
- Circle loss → Triplet (Circle gave no benefit after weighting fix)

**Runtime**: GPU T4 x2 (32GB) | **Duration**: ~4h (larger images)""")

# ════════════════════════════════════════════════════════════════════
# Cell 8: Transforms — 256×256, batch 80
# ════════════════════════════════════════════════════════════════════
print("Cell 8: Transforms + loaders")
src8 = get_src(8)

# Resolution
src8 = src8.replace('H, W = 224, 224', 'H, W = 256, 256  # v14: higher resolution (v8 was 224)')

# Augmentation comment
src8 = src8.replace('v6-v9: proven augmentation', 'v6/v8/v14: proven augmentation')

# Batch size: 48*NUM_GPUS → 40*NUM_GPUS for memory safety at 256
src8 = src8.replace(
    'BATCH = 48 * NUM_GPUS  # 96 on 2xT4\n'
    'P_IDS = 12 * NUM_GPUS  # 24 IDs per batch',
    'BATCH = 40 * NUM_GPUS  # 80 on 2xT4 (reduced for 256px memory)\n'
    'P_IDS = 10 * NUM_GPUS  # 20 IDs per batch'
)

set_src(8, src8)

# ════════════════════════════════════════════════════════════════════
# Cell 9: Losses — TripletLoss + ε=0.1
# ════════════════════════════════════════════════════════════════════
print("Cell 9: Losses")
set_src(9, r'''# ── Losses (numerically stable for fp16 + DataParallel) ──
class CrossEntropyLabelSmooth(nn.Module):
    """Label-smooth CE using F.log_softmax in fp32 for numerical stability."""
    def __init__(self, num_classes, epsilon=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon

    def forward(self, inputs, targets):
        # Force fp32 for log_softmax to prevent overflow
        log_probs = F.log_softmax(inputs.float(), dim=1)
        with torch.no_grad():
            oh = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
            smooth = (1 - self.epsilon) * oh + self.epsilon / self.num_classes
        loss = (-smooth * log_probs).sum(dim=1).mean()
        return loss


class TripletLossHardMining(nn.Module):
    """Triplet loss with batch hard mining — fp32 computation.
    
    For each anchor, find hardest positive (max dist) and hardest negative
    (min dist) within the batch. Standard in TransReID / BoT.
    """
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, feats, pids):
        feats = F.normalize(feats.float(), p=2, dim=1)
        n = feats.size(0)
        dist = torch.cdist(feats, feats, p=2)  # (n, n) Euclidean
        
        mask_pos = pids.unsqueeze(0).eq(pids.unsqueeze(1))
        mask_neg = ~mask_pos
        
        # For each anchor: hardest positive (largest distance)
        dist_pos = dist.clone()
        dist_pos[~mask_pos] = 0
        hardest_pos, _ = dist_pos.max(dim=1)
        
        # For each anchor: hardest negative (smallest distance)
        dist_neg = dist.clone()
        dist_neg[~mask_neg] = float('inf')
        hardest_neg, _ = dist_neg.min(dim=1)
        
        y = torch.ones(n, device=feats.device)
        return self.ranking_loss(hardest_neg, hardest_pos, y)


class CenterLoss(nn.Module):
    """Center loss (Wen et al., ECCV 2016) — fp32 computation."""
    def __init__(self, num_classes, feat_dim):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, feats, labels):
        feats = feats.float()
        batch_size = feats.size(0)
        centers_batch = self.centers[labels]  # (B, D)
        diff = feats - centers_batch
        loss = (diff * diff).sum(1).mean()
        return loss


print("Losses defined: CE+LS(ε=0.1), TripletLoss(m=0.3), CenterLoss (fp32-stable)")''')

# ════════════════════════════════════════════════════════════════════
# Cell 12: Model markdown
# ════════════════════════════════════════════════════════════════════
print("Cell 12: Model markdown")
set_src(12, """## 5. TransReID Model

Camera-aware SIE is especially important for vehicles (same car looks very
different across 20 cameras with different viewpoints/lighting).

**v6 Critical Fixes (carried through all versions):**
- **norm_pre**: CLIP ViTs have a pre-LayerNorm before transformer blocks.
- **SIE on ALL tokens**: Original TransReID broadcasts camera embedding to all patch tokens.
- **LLRD**: Layer-wise LR Decay protects shallow CLIP features.
- **Pure OpenAI CLIP**: Uses `vit_base_patch16_clip_224.openai`.

**v14: 256×256 Resolution**
- ViT-Base/16 at 256px → 16×16 = 256 patches (vs 14×14 = 196 at 224px)
- timm interpolates positional embeddings from 224→256 automatically
- ~30% more spatial information for finer vehicle details (plates, decals, trim)
- v8 proven base: Triplet(0.3), ε=0.1, 140 epochs""")

# ════════════════════════════════════════════════════════════════════
# Cell 13: Model — add img_size=256
# ════════════════════════════════════════════════════════════════════
print("Cell 13: Model (img_size=256)")
src13 = get_src(13)

# The timm.create_model call needs img_size
src13 = src13.replace(
    "self.vit = timm.create_model(vit_model, pretrained=pretrained, num_classes=0)",
    "self.vit = timm.create_model(vit_model, pretrained=pretrained, num_classes=0,\n"
    "                                         img_size=img_size)"
)

# Add img_size parameter to __init__
src13 = src13.replace(
    "def __init__(self, num_classes, num_cameras=0, embed_dim=768,\n"
    "                 vit_model=\"vit_base_patch16_clip_224\", pretrained=True,",
    "def __init__(self, num_classes, num_cameras=0, embed_dim=768,\n"
    "                 vit_model=\"vit_base_patch16_clip_224\", pretrained=True,\n"
    "                 img_size=224,"
)

# Add img_size to the print summary
src13 = src13.replace(
    'f"SIE={self.sie_camera}({num_cameras}), JPM={jpm}, blocks={self.num_blocks}")',
    'f"SIE={self.sie_camera}({num_cameras}), JPM={jpm}, blocks={self.num_blocks}, img={img_size}")'
)

# Model instantiation: add img_size=256
src13 = src13.replace(
    "model = TransReID(\n"
    "    num_classes=num_classes, num_cameras=num_cameras,\n"
    "    embed_dim=768, vit_model=VIT_MODEL, sie_camera=True, jpm=True,\n"
    ").to(DEVICE)",
    "model = TransReID(\n"
    "    num_classes=num_classes, num_cameras=num_cameras,\n"
    "    embed_dim=768, vit_model=VIT_MODEL, sie_camera=True, jpm=True,\n"
    "    img_size=256,  # v14: higher resolution (v8 was 224)\n"
    ").to(DEVICE)"
)

set_src(13, src13)

# ════════════════════════════════════════════════════════════════════
# Cell 14: Training — revert to v8 + fix bugs
# ════════════════════════════════════════════════════════════════════
print("Cell 14: Training")
set_src(14, r'''# -- Training TransReID (CLIP ViT-Base) on VeRi-776 --
# v14: v8 proven recipe + 256×256 resolution

# v8 proven: Label smoothing ε=0.1
ce_loss = CrossEntropyLabelSmooth(num_classes, 0.1).to(DEVICE)

# v8 proven: Triplet loss with hard mining (margin=0.3)
tri_loss = TripletLossHardMining(margin=0.3).to(DEVICE)

# v8: Center loss -- delayed start at epoch 30 for stability
ctr_loss = CenterLoss(num_classes, 768).to(DEVICE)
CENTER_WEIGHT = 5e-4
CENTER_START = 30  # Only activate AFTER classifier has converged somewhat

raw_model = model.module if hasattr(model, 'module') else model

# v6/v8: Proven CLIP fine-tuning hyperparameters (unchanged)
backbone_lr = 3.5e-4
head_lr = 3.5e-3
wd = 5e-4
llrd_factor = 0.75  # v6 proven value

print(f"LLRD config: decay={llrd_factor}")
print(f"  Deepest backbone layer lr: {backbone_lr:.2e}")
print(f"  Shallowest (embed) layer lr: {backbone_lr * llrd_factor**(raw_model.num_blocks+1):.2e}")
print(f"  Metric loss: TripletLoss(m=0.3)")
print(f"  Center loss weight: {CENTER_WEIGHT} (starts at epoch {CENTER_START})")
print(f"  Label smoothing: ε=0.1")

param_groups = raw_model.get_llrd_param_groups(backbone_lr, head_lr, decay=llrd_factor)
optimizer = torch.optim.AdamW(param_groups, weight_decay=wd)

# v8: Separate center loss optimizer (SGD, lr=0.5, no weight decay -- standard)
center_optimizer = torch.optim.SGD(ctr_loss.parameters(), lr=0.5)

# Store base LRs for warmup
base_lrs = [pg["lr"] for pg in optimizer.param_groups]

n_bb = sum(p.numel() for n, p in raw_model.named_parameters() if "vit" in n)
n_hd = sum(p.numel() for n, p in raw_model.named_parameters() if "vit" not in n)
print(f"Backbone params: {n_bb:,} (max_lr={backbone_lr})")
print(f"Head params:     {n_hd:,} (lr={head_lr})")

# v8 proven: 140 epochs (v9/v8+ showed 180 degrades after ep140)
EPOCHS = 140
WARMUP = 10

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP)
scaler = torch.amp.GradScaler("cuda")

history = {"loss": [], "mAP": [], "R1": [], "mAP_rr": [], "R1_rr": []}
best_mAP = 0.0

print("=" * 70)
print(f"  Training TransReID ViT-Base (CLIP) on VeRi-776 — v14")
print(f"  Losses: CE(ε=0.1) + Triplet(0.3) + Center(5e-4, delayed@ep{CENTER_START})")
print(f"  v8→v14: 224→256px input, batch 96→80 (1 change from proven v8 base)")
print(f"  LLRD factor={llrd_factor}, warmup={WARMUP}, epochs={EPOCHS}")
print("=" * 70)

t0 = time.time()
for epoch in range(EPOCHS):
    model.train()
    rl, nb = 0.0, 0
    use_center = (epoch >= CENTER_START)  # v8: delayed center loss

    # Warmup: linearly scale all LRs from 0 to base
    if epoch < WARMUP:
        wf = (epoch + 1) / WARMUP
        for i, pg in enumerate(optimizer.param_groups):
            pg["lr"] = base_lrs[i] * wf
    elif epoch == WARMUP:
        for i, pg in enumerate(optimizer.param_groups):
            pg["lr"] = base_lrs[i]

    for imgs, pids, cams, _ in train_loader:
        imgs, pids, cams = imgs.to(DEVICE), pids.to(DEVICE).long(), cams.to(DEVICE).long()
        optimizer.zero_grad()
        if use_center:
            center_optimizer.zero_grad()

        with torch.amp.autocast("cuda"):
            out = model(imgs, cam_ids=cams)
            if len(out) == 3:
                c, f, jc = out
                loss = ce_loss(c, pids) + tri_loss(f, pids) + 0.5 * ce_loss(jc, pids)
            else:
                c, f = out
                loss = ce_loss(c, pids) + tri_loss(f, pids)

        if use_center:
            # Center loss in fp32 (outside autocast for numerical stability)
            center_l = ctr_loss(f.float(), pids)
            total_loss = loss + CENTER_WEIGHT * center_l
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            scaler.unscale_(center_optimizer)
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.step(center_optimizer)
        else:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=5.0)
            scaler.step(optimizer)

        scaler.update()
        rl += loss.item() if not torch.isnan(loss) else 0.0
        nb += 1

    if epoch >= WARMUP:
        scheduler.step()
    history["loss"].append(rl / max(nb, 1))

    if (epoch + 1) % 10 == 0:
        hd_lr = optimizer.param_groups[-1]["lr"]
        top_bb_lr = max(pg["lr"] for pg in optimizer.param_groups[:-1])
        ctr_tag = " [+center]" if use_center else ""
        print(f"Epoch {epoch+1:3d} | Loss {rl/max(nb,1):.4f} | "
              f"top_bb_lr={top_bb_lr:.2e} head_lr={hd_lr:.2e}{ctr_tag}")

    # Evaluate every 20 epochs (and final)
    if (epoch + 1) % 20 == 0 or epoch == EPOCHS - 1:
        mAP, cmc, mAP_rr, cmc_rr = full_eval(model, query_loader, gallery_loader,
                                               DEVICE, pass_cams=True)
        history["mAP"].append(mAP); history["R1"].append(cmc[0])
        history["mAP_rr"].append(mAP_rr or 0)
        history["R1_rr"].append(cmc_rr[0] if cmc_rr is not None else 0)
        is_best = mAP > best_mAP
        if is_best:
            best_mAP = mAP
            _state = (model.module if hasattr(model, 'module') else model).state_dict()
            torch.save(_state, OUTPUT_DIR / "transreid_veri_best.pth")
        tag = " ★" if is_best else ""
        print(f"  → mAP: {mAP:.4f}, R1: {cmc[0]:.4f}")
        if mAP_rr: print(f"  → mAP(RR): {mAP_rr:.4f}, R1(RR): {cmc_rr[0]:.4f}{tag}")

elapsed = time.time() - t0
print(f"\nTransReID (CLIP) v14 done in {elapsed/3600:.1f}h | Best mAP: {best_mAP:.4f}")''')

# ════════════════════════════════════════════════════════════════════
# Cell 16: Training curves title
# ════════════════════════════════════════════════════════════════════
print("Cell 16: Curves")
src16 = get_src(16)
src16 = src16.replace("v9", "v14")
set_src(16, src16)

# ════════════════════════════════════════════════════════════════════
# Cell 18: Metadata
# ════════════════════════════════════════════════════════════════════
print("Cell 18: Metadata")
src18 = get_src(18)
src18 = src18.replace(
    "TransReID ViT-Base CLIP (v9, 0.2×circle_loss, center_loss delayed@ep30, ε=0.15, 180ep)",
    "TransReID ViT-Base CLIP (v14, 256px, triplet, center_loss delayed@ep30, ε=0.1, 140ep)"
)
src18 = src18.replace('"0.2×CircleLoss(m=0.25,γ=64)"', '"TripletLoss(m=0.3)"')
src18 = src18.replace('"0.2 * CircleLoss(m=0.25, gamma=64)"', '"TripletLoss(margin=0.3)"')
src18 = src18.replace('"0.15"', '"0.1"')
src18 = src18.replace('"180"', '"140"')
src18 = src18.replace(
    '"Triplet(0.3) → 0.2×CircleLoss(m=0.25, γ=64): adaptive pair weighting, weight balances with CE"',
    '"v8 proven Triplet(0.3); v14 adds 256px resolution"'
)
# Also fix img_size references
src18 = src18.replace('"224"', '"256"')
set_src(18, src18)

# ════════════════════════════════════════════════════════════════════
# Cell 19: Results
# ════════════════════════════════════════════════════════════════════
print("Cell 19: Results")
src19 = get_src(19)
src19 = src19.replace("v9", "v14")
set_src(19, src19)

# ════════════════════════════════════════════════════════════════════
# Cell 20: Integration markdown
# ════════════════════════════════════════════════════════════════════
print("Cell 20: Integration")
set_src(20, """## Local Integration

```yaml
# configs/default.yaml -- Stage 2 config
stage2:
  reid:
    model_name: transreid
    weights: models/reid/vehicle_transreid_vit_base_veri776.pth
    embedding_dim: 768
    input_size: [256, 256]  # v14: higher resolution
```

### v14 Strategy: One Surgical Change
- **v8 proven base** (80.4% mAP): Triplet(0.3), ε=0.1, 140 epochs, all v6 fixes
- **Single change**: 224→256 input resolution (+30% pixels, finer vehicle detail)
- Batch reduced 96→80 for GPU memory safety at higher resolution
- v9 experiments showed ε=0.15 and 180 epochs both HURT (reverted)
- If 256px works, consider 288px or 384px in future versions""")

# Write
with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\nDone! v14 written to {NB_PATH}")
