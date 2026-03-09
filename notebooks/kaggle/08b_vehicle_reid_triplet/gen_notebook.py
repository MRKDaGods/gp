"""Generate 08b notebook: v8+ (Triplet, ε=0.15, 180ep) from 08 v9."""
import json, copy, re

import pathlib, sys
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
SRC = SCRIPT_DIR.parent / '08_vehicle_reid_sota' / '08_vehicle_reid_sota.ipynb'
OUT = SCRIPT_DIR / '08b_vehicle_reid_triplet.ipynb'
with open(SRC, 'r', encoding='utf-8') as f:
    nb = json.load(f)

nb2 = copy.deepcopy(nb)

def replace_cell_source(cell_idx, old, new):
    """Replace text in a specific cell's source."""
    src = ''.join(nb2['cells'][cell_idx]['source'])
    if old not in src:
        print(f"  WARNING: '{old[:60]}...' not found in cell {cell_idx}")
        return False
    src = src.replace(old, new, 1)
    nb2['cells'][cell_idx]['source'] = [line + '\n' for line in src.split('\n')]
    # Remove trailing \n from last line if original didn't have it
    if nb2['cells'][cell_idx]['source'][-1] == '\n':
        nb2['cells'][cell_idx]['source'] = nb2['cells'][cell_idx]['source'][:-1]
    return True

def set_cell_source(cell_idx, new_source):
    """Replace entire cell source."""
    lines = new_source.split('\n')
    nb2['cells'][cell_idx]['source'] = [line + '\n' for line in lines[:-1]] + [lines[-1]]


# ── Cell 0: Header markdown ──
print("Cell 0: Header")
set_cell_source(0, """# Notebook 08b: Vehicle ReID — TransReID Training (v8+, Triplet)
**Multi-Camera Tracking System — Kaggle Training Pipeline**

Train SOTA vehicle re-identification on VeRi-776 using TransReID ViT-Base with CLIP pretraining.

## Model
| Model | Architecture | Dim | Target mAP | Target R1 |
|-------|-------------|-----|-------------|-------------|
| **TransReID** | ViT-Base/16 (CLIP) + SIE + JPM | 768 | >82% | >97% |

## Key Features (v8+)
- All v6 fixes: norm_pre, SIE all tokens, LLRD(0.75), OpenAI CLIP
- v6 proven hyperparameters: backbone_lr=3.5e-4, head_lr=3.5e-3
- **Triplet loss** (margin=0.3) — proven in v8 (80.4% mAP)
- **Center loss** activated after epoch 30 (delayed start for stability)
- **Label smoothing ε=0.15** (up from v8's 0.1 — helps with similar vehicles)
- **180 epochs** (v8 was 140; longer cosine tail for fine-tuning)
- **Random Erasing** (p=0.5, random fill — proven in TransReID paper)
- v6 augmentation base (NO AutoAugment — v7 lesson)

**v8 → v8+ changes:**
| Change | v8 | v8+ | Rationale |
|--------|----|----|-----------|
| Label smooth ε | 0.1 | 0.15 | VeRi has many similar vehicles; stronger smoothing helps |
| Epochs | 140 | 180 | Longer cosine tail benefits CLIP fine-tuning |

**Runtime**: GPU T4 x2 (32GB) | **Duration**: ~3.5h
**Parallel run with 08 v9 (Circle Loss) — head-to-head comparison.**""")

# ── Cell 9: Losses — replace CircleLoss with TripletLoss ──
print("Cell 9: Losses")
set_cell_source(9, r'''# ── Losses (numerically stable for fp16 + DataParallel) ──
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


print("Losses defined: CE+LS(ε=0.15), TripletLoss(m=0.3), CenterLoss (fp32-stable)")''')

# ── Cell 12: Model description markdown ──
print("Cell 12: Model description")
set_cell_source(12, """## 5. TransReID Model

Camera-aware SIE is especially important for vehicles (same car looks very
different across 20 cameras with different viewpoints/lighting).

**v6 Critical Fixes (carried into v8+):**
- **norm_pre**: CLIP ViTs have a pre-LayerNorm before transformer blocks.
- **SIE on ALL tokens**: Original TransReID broadcasts camera embedding to all patch tokens.
- **LLRD**: Layer-wise LR Decay protects shallow CLIP features.
- **Pure OpenAI CLIP**: Uses `vit_base_patch16_clip_224.openai`.

**v8+ Training (conservative, proven base):**
- **Triplet Loss** (margin=0.3): Same as v8. Proven, reliable metric learning.
- **Label smoothing ε=0.15**: Up from 0.1.
- **180 epochs**: Extended from 140.""")

# ── Cell 14: Training cell — swap Circle for Triplet ──
print("Cell 14: Training setup + loop")
src14 = ''.join(nb2['cells'][14]['source'])

# Replace circle_loss setup
src14 = src14.replace(
    "# v9: Circle loss replaces triplet (adaptive weighting, better gradients)\n"
    "# Weight=0.2 balances circle loss magnitude (~8-10) with CE (~2-3)\n"
    "# Without weighting, circle loss dominates gradients and starves the classifier\n"
    "circle_loss = CircleLoss(m=0.25, gamma=64).to(DEVICE)\n"
    "CIRCLE_WEIGHT = 0.2",
    "# v8+: Triplet loss with hard mining (proven in v8: 80.4% mAP)\n"
    "tri_loss = TripletLossHardMining(margin=0.3).to(DEVICE)"
)

# Replace metric loss print
src14 = src14.replace(
    'print(f"  Metric loss: {CIRCLE_WEIGHT}×CircleLoss(m=0.25, γ=64)")',
    'print(f"  Metric loss: TripletLoss(m=0.3)")'
)

# Replace duplicate old print if present
src14 = src14.replace(
    'print(f"  Metric loss: CircleLoss(m=0.25, γ=64)")',
    'print(f"  Metric loss: TripletLoss(m=0.3)")'
)

# Replace banner
src14 = src14.replace(
    "Training TransReID ViT-Base (CLIP) on VeRi-776 — v9",
    "Training TransReID ViT-Base (CLIP) on VeRi-776 — v8+"
)

# Replace losses banner lines
src14 = src14.replace(
    'print(f"  Losses: CE(ε=0.15) + {CIRCLE_WEIGHT}×CircleLoss(m=0.25,γ=64) + Center(5e-4, delayed@ep{CENTER_START})")',
    'print(f"  Losses: CE(ε=0.15) + Triplet(0.3) + Center(5e-4, delayed@ep{CENTER_START})")'
)
src14 = src14.replace(
    'print(f"  v8→v9: Circle(w=0.2) replaces Triplet, ε 0.1→0.15, epochs 140→180")',
    'print(f"  v8→v8+: same Triplet, ε 0.1→0.15, epochs 140→180")'
)

# Remove old duplicate banner lines if present
src14 = src14.replace(
    'print(f"  Losses: CE(ε=0.15) + CircleLoss(m=0.25,γ=64) + Center(5e-4, delayed@ep{CENTER_START})")',
    'print(f"  Losses: CE(ε=0.15) + Triplet(0.3) + Center(5e-4, delayed@ep{CENTER_START})")'
)
src14 = src14.replace(
    'print(f"  v8→v9: Circle replaces Triplet, ε 0.1→0.15, epochs 140→180")',
    'print(f"  v8→v8+: same Triplet, ε 0.1→0.15, epochs 140→180")'
)

# Replace loss computation in training loop (JPM branch)
src14 = src14.replace(
    "                # v9: Circle loss replaces triplet loss (weighted to balance magnitudes)\n"
    "                loss = ce_loss(c, pids) + CIRCLE_WEIGHT * circle_loss(f, pids) + 0.5 * ce_loss(jc, pids)",
    "                # v8+: Triplet loss (proven in v8)\n"
    "                loss = ce_loss(c, pids) + tri_loss(f, pids) + 0.5 * ce_loss(jc, pids)"
)

# Replace loss computation (non-JPM branch)
src14 = src14.replace(
    "                loss = ce_loss(c, pids) + CIRCLE_WEIGHT * circle_loss(f, pids)",
    "                loss = ce_loss(c, pids) + tri_loss(f, pids)"
)

# Replace final print
src14 = src14.replace(
    "TransReID (CLIP) v9 done",
    "TransReID (CLIP) v8+ done"
)

# Write back
lines14 = src14.split('\n')
nb2['cells'][14]['source'] = [line + '\n' for line in lines14[:-1]] + [lines14[-1]]

# ── Cell 16: Training curves title ──
print("Cell 16: Curves title")
replace_cell_source(16, "TransReID (CLIP) v9", "TransReID (CLIP) v8+")

# ── Cell 18: Metadata ──
print("Cell 18: Metadata")
src18 = ''.join(nb2['cells'][18]['source'])
src18 = src18.replace(
    "TransReID ViT-Base CLIP (v9, 0.2×circle_loss, center_loss delayed@ep30, ε=0.15, 180ep)",
    "TransReID ViT-Base CLIP (v8+, triplet, center_loss delayed@ep30, ε=0.15, 180ep)"
)
src18 = src18.replace('"0.2×CircleLoss(m=0.25,γ=64)"', '"TripletLoss(m=0.3)"')
src18 = src18.replace('"0.2 * CircleLoss(m=0.25, gamma=64)"', '"TripletLoss(margin=0.3)"')
src18 = src18.replace(
    '"Triplet(0.3) → 0.2×CircleLoss(m=0.25, γ=64): adaptive pair weighting, weight balances with CE"',
    '"Kept Triplet(0.3) from v8 (proven baseline)"'
)
lines18 = src18.split('\n')
nb2['cells'][18]['source'] = [line + '\n' for line in lines18[:-1]] + [lines18[-1]]

# ── Cell 19: Results ──
print("Cell 19: Results")
replace_cell_source(19, "VEHICLE ReID RESULTS — VeRi-776 (v9)", "VEHICLE ReID RESULTS — VeRi-776 (v8+)")
replace_cell_source(19, "TransReID ViT-Base (CLIP) v9", "TransReID ViT-Base (CLIP) v8+")

# ── Cell 20: Integration markdown ──
print("Cell 20: Integration")
set_cell_source(20, """## Local Integration

```yaml
# configs/default.yaml -- Stage 2 config
stage2:
  reid:
    model_name: transreid
    weights: models/reid/vehicle_transreid_vit_base_veri776.pth
    embedding_dim: 768
    input_size: [224, 224]
```

### v8+ Strategy
- **v6 proven base** (79.6% mAP): same LR, LLRD, augmentation
- **v8 proven Triplet** (80.4% mAP): kept Triplet(0.3) + center loss delayed@ep30
- **v8+ tweaks** (conservative, 2 changes only):
  1. **Label smoothing ε=0.15**: stronger regularisation for visually similar vehicles
  2. **180 epochs**: longer cosine tail extracts more from CLIP backbone
- **Parallel run** with 08b v9 (Circle Loss) for head-to-head comparison""")

# Clean up cell IDs (generate new ones)
import uuid
for cell in nb2['cells']:
    cell['id'] = uuid.uuid4().hex[:8]

# Write
with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(nb2, f, indent=1, ensure_ascii=False)

print("\nDone! Notebook written.")
