# 09o — EVA02 ViT-B/16 CLIP Training on CityFlowV2

## Goal

Train `eva02_base_patch16_clip_224.merged2b` (timm) for vehicle ReID on CityFlowV2, producing a model that plugs into our Stage 2 pipeline as a secondary/replacement for the current LAION-2B CLIP ViT-B/16.

**Why EVA02?** It's a ViT-B/16 pretrained with MIM (Masked Image Modeling) + CLIP on Merged-2B — a different pretraining recipe from our current OpenAI CLIP and LAION-2B CLIP models. This gives us a **decorrelated secondary** that might complement our primary in ensemble, unlike the too-correlated LAION-2B variant (10c v56: -0.5pp).

## Critical Architecture Incompatibility ⚠️

**EVA02 is NOT a drop-in replacement in the existing TransReID class.** Three breaking differences:

| Issue | Standard ViT (current) | EVA02 |
|-------|------------------------|-------|
| `_pos_embed()` return | `Tensor` | `Tuple[Tensor, Tensor]` (x, rot_pos_embed) |
| `Block.forward()` signature | `forward(x)` | `forward(x, rope=rot_pos_embed)` |
| Position encoding | Absolute pos_embed only | Absolute + RoPE (rotary) in attention |

**Root cause:** EVA02 uses **Rotary Position Embeddings (RoPE)** inside attention blocks. timm's `Eva` class:
- `_pos_embed()` returns `(x, rot_pos_embed)` — a tuple, not a single tensor
- `EvaBlock.forward()` takes `rope` kwarg and passes it to attention
- Without `rope`, attention degrades to position-unaware (defeats RoPE)

Additionally, EVA02 uses **SwiGLU** FFN (instead of standard MLP) and sub-layer normalization, but these are encapsulated within blocks and don't affect the external interface.

## Architecture: Training Model Class

Define `EVA02ReID` in the notebook. Based on TransReID but with RoPE handling:

```python
class EVA02ReID(nn.Module):
    """EVA02 ViT-B/16 + BNNeck + SIE + JPM for ReID.
    
    Key difference from TransReID: handles EVA02's RoPE by unpacking
    _pos_embed() tuple and passing rope to each block.
    """
    def __init__(self, num_classes, num_cameras=0, embed_dim=768,
                 vit_model="eva02_base_patch16_clip_224.merged2b",
                 pretrained=True, sie_camera=True, jpm=True, img_size=256):
        super().__init__()
        self.sie_camera = sie_camera and num_cameras > 0
        self.jpm = jpm
        
        self.vit = timm.create_model(vit_model, pretrained=pretrained,
                                     num_classes=0, img_size=img_size)
        self.vit_dim = self.vit.embed_dim  # 768
        self.num_blocks = len(self.vit.blocks)
        
        # SIE camera embedding
        if self.sie_camera:
            self.sie_embed = nn.Parameter(torch.zeros(num_cameras, 1, self.vit_dim))
            nn.init.trunc_normal_(self.sie_embed, std=0.02)
        
        # BNNeck (key='bn' to match pipeline checkpoint loading)
        self.bn = nn.BatchNorm1d(self.vit_dim)
        self.bn.bias.requires_grad_(False)
        
        # Projection (Identity since embed_dim == vit_dim == 768)
        self.proj = (nn.Linear(self.vit_dim, embed_dim, bias=False)
                     if embed_dim != self.vit_dim else nn.Identity())
        self.cls_head = nn.Linear(embed_dim, num_classes, bias=False)
        
        if isinstance(self.proj, nn.Linear):
            nn.init.kaiming_normal_(self.proj.weight, mode="fan_out")
        nn.init.normal_(self.cls_head.weight, std=0.001)
        
        if self.jpm:
            self.bn_jpm = nn.BatchNorm1d(self.vit_dim)
            self.bn_jpm.bias.requires_grad_(False)
            self.jpm_cls = nn.Linear(self.vit_dim, num_classes, bias=False)
            nn.init.normal_(self.jpm_cls.weight, std=0.001)

    def forward(self, x, cam_ids=None):
        B = x.shape[0]
        
        # 1. Patch embedding
        x = self.vit.patch_embed(x)
        
        # 2. Pos embed — EVA02 returns (x, rot_pos_embed) tuple
        result = self.vit._pos_embed(x)
        if isinstance(result, tuple):
            x, rot_pos_embed = result
        else:
            x = result
            rot_pos_embed = None
        
        # 3. SIE: camera embedding broadcast to all tokens
        if self.sie_camera and cam_ids is not None:
            x = x + self.sie_embed[cam_ids]
        
        # 4. Patch drop + norm_pre (EVA02 has active norm_pre)
        if hasattr(self.vit, 'patch_drop'):
            x = self.vit.patch_drop(x)
        if hasattr(self.vit, 'norm_pre'):
            x = self.vit.norm_pre(x)
        
        # 5. Transformer blocks — MUST pass rope for EVA02
        for blk in self.vit.blocks:
            if rot_pos_embed is not None:
                x = blk(x, rope=rot_pos_embed)
            else:
                x = blk(x)
        x = self.vit.norm(x)
        
        # 6. CLS token → features
        g = x[:, 0]
        bn = self.bn(g)
        proj = self.proj(bn)
        
        if self.training:
            cls = self.cls_head(proj)
            if self.jpm:
                patches = x[:, 1:]
                idx = torch.randperm(patches.size(1), device=x.device)
                s = patches[:, idx]
                mid = s.size(1) // 2
                jf = (s[:, :mid].mean(1) + s[:, mid:].mean(1)) / 2
                return cls, g, self.jpm_cls(self.bn_jpm(jf))
            return cls, g
        
        return F.normalize(proj, p=2, dim=1)

    def get_llrd_param_groups(self, backbone_lr, head_lr, decay=0.75):
        """Layer-wise learning rate decay — identical to TransReID."""
        groups = {}
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith('vit.'):
                if 'blocks.' in name:
                    block_idx = int(name.split('blocks.')[1].split('.')[0])
                    depth = block_idx + 1
                elif any(k in name for k in ['patch_embed', 'cls_token', 'pos_embed', 'norm_pre', 'rope']):
                    depth = 0
                else:
                    depth = self.num_blocks + 1
                scale = decay ** (self.num_blocks + 1 - depth)
                lr = backbone_lr * scale
                gk = f"bb_d{depth}"
            else:
                lr = head_lr
                gk = "head"
            if gk not in groups:
                groups[gk] = {"params": [], "lr": lr}
            groups[gk]["params"].append(param)
        return sorted(groups.values(), key=lambda x: x["lr"])
```

**Key differences from TransReID:**
1. `_pos_embed()` — unpacks tuple, captures `rot_pos_embed`
2. Block iteration — passes `rope=rot_pos_embed` to each block
3. LLRD — includes `rope` params at depth 0 (embedding-level LR)
4. State dict keys match TransReID exactly (`bn`, `cls_head`, `proj`, `sie_embed`, `bn_jpm`, `jpm_cls`) for pipeline compatibility

## Training Recipe

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Backbone** | `eva02_base_patch16_clip_224.merged2b` | MIM+CLIP pretrain, decorrelated from OpenAI/LAION CLIP |
| **Input size** | 256×256 | Pipeline consistency; RoPE scales gracefully to non-native resolutions |
| **Embed dim** | 768 | Same as ViT-B/16 |
| **Epochs** | 120 | EVA02 has stronger pretrain (MIM+CLIP); may converge faster than LAION-2B's 300ep |
| **Batch size** | 64 (P=16 × K=4) | PK sampler, same as 09l |
| **Backbone LR** | 1e-4 | With LLRD decay=0.75 |
| **Head LR** | 1e-3 | ReID heads (bn, proj, cls_head, jpm) |
| **Weight decay** | 5e-4 | AdamW |
| **LLRD factor** | 0.75 | Layer-wise learning rate decay across 12 blocks |
| **Warmup** | 10 epochs | Linear warmup |
| **LR schedule** | Cosine annealing | After warmup, cosine decay to 0 |
| **Grad clipping** | max_norm=5.0 | Stability |
| **AMP** | Yes | fp16 autocast + GradScaler |
| **EMA** | Yes, decay=0.9999 | Exponential moving average (eval on EMA model) |

### Losses

| Loss | Weight | Config | Start |
|------|--------|--------|-------|
| **CE + Label Smoothing** | 1.0 | ε=0.05 | Epoch 0 |
| **Triplet (hard mining)** | 1.0 | margin=0.3 | Epoch 0 |
| **JPM auxiliary CE** | 0.5 | ε=0.05 | Epoch 0 |
| **Center Loss** | 5e-4 | SGD lr=0.5 | Epoch 15 (delayed) |

### Augmentation

```python
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]

train_tf = T.Compose([
    T.Resize((272, 272), interpolation=T.InterpolationMode.BICUBIC),
    T.RandomHorizontalFlip(p=0.5),
    T.Pad(10),
    T.RandomCrop((256, 256)),
    T.ColorJitter(brightness=0.2, contrast=0.15, saturation=0.1, hue=0.0),
    T.ToTensor(),
    T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    T.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value="random"),
])

val_tf = T.Compose([
    T.Resize((256, 256), interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
])
```

**Normalization**: CLIP stats — auto-detected because `"clip"` appears in `eva02_base_patch16_clip_224.merged2b`.

### SIE (Side Information Embedding)

- `num_cameras` = number of unique cameras in CityFlowV2 training set (~46)
- SIE broadcast to ALL tokens (per TransReID paper)
- Camera IDs passed during training and evaluation
- SIE is disabled at MTMC inference (sie_camera_map={}) — same as primary model

### No VeRi-776 Intermediate

Go directly from EVA02 pretrained → CityFlowV2 fine-tune. No intermediate VeRi-776 stage.

**Rationale:** findings.md shows VeRi-776→CityFlowV2 pretrain was harmful for ResNet (42.7% vs 52.77%). EVA02's MIM+CLIP pretrain is already stronger than VeRi-776 features, so the intermediate step would likely overwrite useful representations.

## Notebook Structure (09o_eva02_vit_reid)

### Cell Layout

| Cell | Content |
|------|---------|
| 1 | Metadata comment + pip installs (`timm>=1.0`, `gdown`, `opencv-python-headless`) |
| 2 | Imports + constants (H=256, W=256, CLIP_MEAN, CLIP_STD, VIT_MODEL, EPOCHS=120) |
| 3 | CityFlowV2 download via gdown (reuse 09n's exact download logic) |
| 4 | GT parsing + crop extraction (reuse 09l/09n's extract_crops_from_camera, MAX_CROPS=20) |
| 5 | Train/query/gallery split + dataset class + PK sampler |
| 6 | EVA02ReID model class (as defined above) |
| 7 | Loss functions: CrossEntropyLabelSmooth, TripletLossHardMining, CenterLoss |
| 8 | EMA class |
| 9 | LLRD optimizer setup + cosine scheduler |
| 10 | Training loop (120 epochs, eval every 20, center loss from ep15) |
| 11 | Final evaluation + best model metrics |
| 12 | Export: save state_dict + metadata JSON |

### Data Download (Cell 3)

```python
GDRIVE_ID = "13wNJpS_Oaoe-7y5Dzexg_Ol7bKu1OWuC"
ARCHIVE_NAME = "AIC22_Track1_MTMC_Tracking.zip"
# ... exact same gdown logic as 09n ...
```

### DataParallel Setup

```python
NUM_GPUS = torch.cuda.device_count()
model = EVA02ReID(
    num_classes=num_classes,
    num_cameras=num_cameras,
    embed_dim=768,
    vit_model="eva02_base_patch16_clip_224.merged2b",
    pretrained=True,
    sie_camera=True,
    jpm=True,
    img_size=256,
).to(DEVICE)

raw_model = model
if NUM_GPUS > 1:
    model = nn.DataParallel(model)
    print(f"DataParallel on {NUM_GPUS} GPUs")
```

### Evaluation

- Chunked extraction (batch=64, move to CPU after each batch)
- Flip TTA (average forward + horizontally-flipped forward)
- Market-1501 metrics (mAP, R1, CMC@50)
- Eval every 20 epochs + final epoch
- Save best EMA model by mAP

### Model Export (Cell 12)

```python
# Save as state_dict only (compatible with pipeline loading)
best_state = torch.load(best_model_path, map_location="cpu", weights_only=True)
torch.save({"state_dict": best_state}, "/kaggle/working/exported_models/eva02_vit_cityflowv2_final.pth")

# Metadata JSON
metadata = {
    "task": "vehicle_reid",
    "dataset": "cityflowv2",
    "model": {
        "architecture": "eva02_base_patch16_clip_224.merged2b",
        "type": "eva02_reid",
        "embedding_dim": 768,
        "input_size": [256, 256],
        "normalization": {"mean": CLIP_MEAN, "std": CLIP_STD},
        "num_cameras": num_cameras,
        "num_classes": num_classes,
        "tricks": ["SIE", "JPM", "BNNeck", "CE+LS(0.05)", "Triplet(m=0.3)",
                   "CenterLoss(delayed@ep15)", "CosLR", "RE", "CLIP-norm",
                   "LLRD(0.75)", "EMA", "RoPE"],
        "best_mAP": best_mAP,
        "best_R1": best_r1,
        "epochs": EPOCHS,
        "ema_decay": 0.9999,
    },
}
import json
with open("/kaggle/working/exported_models/eva02_vit_cityflowv2_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=True)
```

### Kernel Metadata

```json
{
    "id": "gumfreddy/09o-eva02-vit-reid-cityflowv2",
    "title": "09o-eva02-vit-reid-cityflowv2",
    "code_file": "09o_eva02_vit_reid.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": true,
    "enable_gpu": true,
    "machine_shape": "NvidiaTeslaT4x2",
    "enable_internet": true,
    "dataset_sources": [],
    "kernel_sources": [],
    "competition_sources": []
}
```

## Inference Integration

### Changes to `transreid_model.py`

Modify the `TransReID.forward()` method to handle EVA02's RoPE. This is a **backward-compatible** change that works for all existing ViTs:

```python
# In forward(), replace:
#   if hasattr(self.vit, "_pos_embed"):
#       x = self.vit._pos_embed(x)
# With:
rot_pos_embed = None
if hasattr(self.vit, "_pos_embed"):
    result = self.vit._pos_embed(x)
    if isinstance(result, tuple):
        x, rot_pos_embed = result  # EVA02 returns (x, rope)
    else:
        x = result
else:
    cls_tok = self.vit.cls_token.expand(B, -1, -1)
    x = torch.cat([cls_tok, x], dim=1) + self.vit.pos_embed
    if hasattr(self.vit, "pos_drop"):
        x = self.vit.pos_drop(x)

# ... SIE injection (unchanged) ...
# ... patch_drop (unchanged) ...
# ... norm_pre (unchanged) ...

# Replace:
#   for blk in self.vit.blocks:
#       x = blk(x)
# With:
for blk in self.vit.blocks:
    if rot_pos_embed is not None:
        x = blk(x, rope=rot_pos_embed)
    else:
        x = blk(x)
```

**Why backward-compatible:** Standard ViTs' `_pos_embed()` returns a plain tensor (not tuple), so `isinstance(result, tuple)` is `False` → no rope → `blk(x)` as before.

### Changes to `reid_model.py`

Add `"eva02_vit"` to the TransReID routing set:

```python
_TRANSREID_NAMES = {"transreid", "vit_small", "vit_base", "transreid_vit", "eva02_vit"}
```

### Config Integration

To use EVA02 as **primary** model:

```yaml
stage2:
  reid:
    vehicle:
      model_name: "eva02_vit"
      weights_path: "models/reid/eva02_vit_cityflowv2_final.pth"
      embedding_dim: 768
      input_size: [256, 256]
      vit_model: "eva02_base_patch16_clip_224.merged2b"
      clip_normalization: true  # auto-detected from "clip" in vit_model name
      num_cameras: 59
```

To use EVA02 as **secondary** (ensemble with primary OpenAI CLIP ViT):

```yaml
stage2:
  reid:
    vehicle2:
      enabled: true
      save_separate: true
      model_name: "eva02_vit"
      weights_path: "models/reid/eva02_vit_cityflowv2_final.pth"
      embedding_dim: 768
      input_size: [256, 256]
      vit_model: "eva02_base_patch16_clip_224.merged2b"
      clip_normalization: true
      num_cameras: 59
```

### 10a Pipeline Integration

In notebook 10a, EVA02 weights are loaded via the same `build_transreid()` → `timm.create_model()` path. The only requirement is that `transreid_model.py` has the RoPE-aware forward pass.

The checkpoint key structure is **identical** to TransReID:
- `vit.*` — backbone weights (timm Eva module)
- `bn.*` — BNNeck
- `proj.*` — projection (Identity, so no keys)
- `cls_head.*` — classifier (dropped at inference, num_classes mismatch OK)
- `sie_embed` — camera embedding (zero-padded if fewer cameras in checkpoint)
- `bn_jpm.*`, `jpm_cls.*` — JPM (dropped at inference)

## Estimated Training Time

| Phase | Time |
|-------|------|
| Pip installs + data download | ~10 min |
| Crop extraction (46 cameras) | ~20 min |
| Training (120 epochs × ~188 iter/ep) | ~2.5 hr |
| Eval (6 evals × ~3 min) | ~18 min |
| **Total** | **~3 hours** |

Well within Kaggle's 12-hour limit on T4 x2.

**Note:** EVA02's SwiGLU FFN is ~15% heavier than standard MLP, so expect ~0.4s/iteration vs ~0.35s for standard ViT-B/16.

## Expected Performance

**Optimistic:** mAP ≥ 80%, R1 ≥ 91% — competitive with primary CLIP ViT (81.59% mAP)
**Realistic:** mAP 75-80% — EVA02's MIM pretraining may not transfer as well to vehicle ReID as pure CLIP
**Pessimistic:** mAP < 70% — architectural mismatch or RoPE scaling issues at 256×256

**Key success metric:** Not just mAP, but **decorrelation from primary model**. Even at 75% mAP, if EVA02 makes different errors than the primary CLIP ViT, score-level fusion in Stage 4 could improve MTMC IDF1.

## Risks

1. **RoPE at 256×256:** EVA02's RoPE was trained at 224×224 (14×14 grid). At 256×256 (16×16 grid), timm auto-scales RoPE, but this is untested for ReID. Fallback: train at 224×224 if 256 underperforms.

2. **SwiGLU + AMP stability:** SwiGLU uses gated activation that could have fp16 overflow. Mitigated by grad clipping (max_norm=5.0) and GradScaler. Monitor for NaN losses.

3. **Correlation with primary:** If EVA02 learns features too similar to OpenAI CLIP ViT, ensemble won't help. The MIM pretraining component should provide decorrelation, but this needs empirical validation.

4. **Center loss interaction:** Center loss with SwiGLU features may behave differently. Delayed start (epoch 15) provides safety margin.

## Checklist Before Implementation

- [ ] Verify `timm.create_model("eva02_base_patch16_clip_224.merged2b", pretrained=True, num_classes=0, img_size=256)` works on Kaggle's timm version
- [ ] Verify EVA02's `_pos_embed()` returns tuple on Kaggle's timm version
- [ ] Verify `EvaBlock.forward(x, rope=...)` signature on Kaggle's timm version
- [ ] Confirm T4 x2 VRAM is sufficient for batch=64 at 256×256 (EVA02-B ~86M params, similar to ViT-B)
- [ ] Test that exported state_dict loads correctly in pipeline's `build_transreid()` after the RoPE-aware modification
