"""Create v15: BNNeck routing fix + revert to v8 base (224px, batch 96).
Modifies the notebook in-place."""
import json, pathlib

NB_PATH = pathlib.Path(__file__).resolve().parent / '08_vehicle_reid_sota.ipynb'
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
set_src(0, """# Notebook 08: Vehicle ReID — TransReID Training (v15)
**Multi-Camera Tracking System — Kaggle Training Pipeline**

Train SOTA vehicle re-identification on VeRi-776 using TransReID ViT-Base with CLIP pretraining.

## Model
| Model | Architecture | Dim | Target mAP | Target R1 |
|-------|-------------|-----|-------------|-------------|
| **TransReID** | ViT-Base/16 (CLIP) + SIE + JPM | 768 | >82% | >97% |

## Key Features (v15)
- All v6 fixes: norm_pre, SIE all tokens, LLRD(0.75), OpenAI CLIP
- v8 proven base: Triplet(0.3), ε=0.1, 140 epochs, 224×224
- **NEW: BNNeck routing fix** — triplet+center loss receive PRE-BN features

**v8 → v15 changes (one fix only):**
| Change | v8 (bug) | v15 (fix) | Rationale |
|--------|---------|-----------|-----------|
| Triplet/Center input | post-BN features | **pre-BN features** | BoT paper: BN whitens features, destroying distance structure for metric losses. CE should get post-BN, triplet/center should get pre-BN. ~1-2% mAP gain in literature. |

**Previous experiments (all reverted):**
| Experiment | Result | Conclusion |
|-----------|--------|------------|
| ε=0.15 (v9, v8+) | −3% mAP | Too much smoothing for 576-class VeRi |
| 180 epochs (v9, v8+) | Peaked at 140 | Extra epochs caused mild overfitting |
| Circle Loss (v13) | Same as Triplet | No benefit after weighting fix |
| 256×256 (v14) | −2% mAP | Positional embedding interpolation corrupts CLIP |

**Runtime**: GPU T4 x2 (32GB) | **Duration**: ~3h""")

# ════════════════════════════════════════════════════════════════════
# Cell 8: Revert to 224px, batch 96
# ════════════════════════════════════════════════════════════════════
print("Cell 8: Revert 224px + batch 96")
src8 = get_src(8)
src8 = src8.replace(
    'H, W = 256, 256  # v14: higher resolution (v8 was 224)',
    'H, W = 224, 224'
)
src8 = src8.replace(
    'BATCH = 40 * NUM_GPUS  # 80 on 2xT4 (reduced for 256px memory)\n'
    'P_IDS = 10 * NUM_GPUS  # 20 IDs per batch',
    'BATCH = 48 * NUM_GPUS  # 96 on 2xT4\n'
    'P_IDS = 12 * NUM_GPUS  # 24 IDs per batch'
)
set_src(8, src8)

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

**v15: BNNeck Routing Fix**
Standard BoT/TransReID design (Luo et al., CVPRW 2019):
- **CE classifier** ← features AFTER BNNeck (normalized distribution → better softmax convergence)
- **Triplet + Center loss** ← features BEFORE BNNeck (raw distances → proper metric learning)
- **Inference** ← features AFTER BNNeck + L2 normalize

v8 had a bug: triplet+center received post-BN features, which destroys the
natural distance structure needed for hard mining. BoT reports ~1.9% mAP gain
from correct routing on Market-1501.""")

# ════════════════════════════════════════════════════════════════════
# Cell 13: Model — BNNeck routing fix + revert img_size
# ════════════════════════════════════════════════════════════════════
print("Cell 13: BNNeck routing fix")
src13 = get_src(13)

# Revert img_size parameter from __init__ signature
src13 = src13.replace(
    '                 vit_model="vit_base_patch16_clip_224", pretrained=True,\n'
    '                 img_size=224,\n'
    '                 sie_camera=True, jpm=True):',
    '                 vit_model="vit_base_patch16_clip_224", pretrained=True,\n'
    '                 sie_camera=True, jpm=True):'
)

# Revert timm create_model to remove img_size
src13 = src13.replace(
    'self.vit = timm.create_model(vit_model, pretrained=pretrained, num_classes=0,\n'
    '                                         img_size=img_size)',
    'self.vit = timm.create_model(vit_model, pretrained=pretrained, num_classes=0)'
)

# Fix the print summary to remove img={img_size}
src13 = src13.replace(
    'f"SIE={self.sie_camera}({num_cameras}), JPM={jpm}, blocks={self.num_blocks}, img={img_size}")',
    'f"SIE={self.sie_camera}({num_cameras}), JPM={jpm}, blocks={self.num_blocks}")'
)

# BNNeck routing fix: return g (pre-BN) for triplet/center, cls uses post-BN
# Old forward return section:
src13 = src13.replace(
    "        # Global feature (CLS token)\n"
    "        g = x[:, 0]\n"
    "        bn = self.bn(g)\n"
    "        proj = self.proj(bn)\n"
    "\n"
    "        if self.training:\n"
    "            cls = self.cls_head(proj)\n"
    "            if self.jpm:\n"
    "                patches = x[:, 1:]\n"
    "                idx = torch.randperm(patches.size(1), device=x.device)\n"
    "                s = patches[:, idx]\n"
    "                mid = s.size(1) // 2\n"
    "                jf = (s[:, :mid].mean(1) + s[:, mid:].mean(1)) / 2\n"
    "                return cls, proj, self.jpm_cls(self.bn_jpm(jf))\n"
    "            return cls, proj\n"
    "        return F.normalize(proj, p=2, dim=1)",

    "        # Global feature (CLS token)\n"
    "        g = x[:, 0]  # raw features (pre-BN) — for triplet + center loss\n"
    "        bn = self.bn(g)  # BNNeck — for CE classifier + inference\n"
    "        proj = self.proj(bn)\n"
    "\n"
    "        if self.training:\n"
    "            cls = self.cls_head(proj)  # CE gets post-BN features\n"
    "            if self.jpm:\n"
    "                patches = x[:, 1:]\n"
    "                idx = torch.randperm(patches.size(1), device=x.device)\n"
    "                s = patches[:, idx]\n"
    "                mid = s.size(1) // 2\n"
    "                jf = (s[:, :mid].mean(1) + s[:, mid:].mean(1)) / 2\n"
    "                return cls, g, self.jpm_cls(self.bn_jpm(jf))  # v15: g (pre-BN) for triplet\n"
    "            return cls, g  # v15: g (pre-BN) for triplet+center\n"
    "        return F.normalize(proj, p=2, dim=1)  # inference: post-BN"
)

# Revert model instantiation to remove img_size=256
src13 = src13.replace(
    "model = TransReID(\n"
    "    num_classes=num_classes, num_cameras=num_cameras,\n"
    "    embed_dim=768, vit_model=VIT_MODEL, sie_camera=True, jpm=True,\n"
    "    img_size=256,  # v14: higher resolution (v8 was 224)\n"
    ").to(DEVICE)",
    "model = TransReID(\n"
    "    num_classes=num_classes, num_cameras=num_cameras,\n"
    "    embed_dim=768, vit_model=VIT_MODEL, sie_camera=True, jpm=True,\n"
    ").to(DEVICE)"
)

set_src(13, src13)

# ════════════════════════════════════════════════════════════════════
# Cell 14: Training — v8 base, update banner only
# ════════════════════════════════════════════════════════════════════
print("Cell 14: Training banner")
src14 = get_src(14)
src14 = src14.replace(
    '# v14: v8 proven recipe + 256\u00d7256 resolution',
    '# v15: v8 proven recipe + BNNeck routing fix'
)
src14 = src14.replace(
    '  Training TransReID ViT-Base (CLIP) on VeRi-776 \u2014 v14',
    '  Training TransReID ViT-Base (CLIP) on VeRi-776 \u2014 v15'
)
src14 = src14.replace(
    '  v8\u2192v14: 224\u2192256px input, batch 96\u219280 (1 change from proven v8 base)',
    '  v8\u2192v15: BNNeck fix (triplet+center get pre-BN features)'
)
src14 = src14.replace(
    'TransReID (CLIP) v14 done',
    'TransReID (CLIP) v15 done'
)
set_src(14, src14)

# ════════════════════════════════════════════════════════════════════
# Cell 16: Curves title
# ════════════════════════════════════════════════════════════════════
print("Cell 16: Curves")
src16 = get_src(16)
src16 = src16.replace('v14', 'v15')
set_src(16, src16)

# ════════════════════════════════════════════════════════════════════
# Cell 18: Metadata
# ════════════════════════════════════════════════════════════════════
print("Cell 18: Metadata")
src18 = get_src(18)
src18 = src18.replace(
    'TransReID ViT-Base CLIP (v14, 256px, triplet, center_loss delayed@ep30, \u03b5=0.1, 140ep)',
    'TransReID ViT-Base CLIP (v15, BNNeck fix, triplet, center_loss delayed@ep30, \u03b5=0.1, 140ep)'
)
src18 = src18.replace('"256"', '"224"')
set_src(18, src18)

# ════════════════════════════════════════════════════════════════════
# Cell 19: Results
# ════════════════════════════════════════════════════════════════════
print("Cell 19: Results")
src19 = get_src(19)
src19 = src19.replace('v14', 'v15')
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
    input_size: [224, 224]
```

### v15 Strategy: Fix the BNNeck Bug
- **v8 proven base** (80.4% mAP): Triplet(0.3), ε=0.1, 140 epochs, 224×224
- **Single fix**: BNNeck routing — triplet+center get pre-BN features (standard BoT design)
- BoT paper reports ~1.9% mAP improvement from correct routing
- All v9-v14 experiments reverted (ε=0.15, 180ep, Circle, 256px all hurt)""")

# Write
with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\nDone! v15 written to {NB_PATH}")
