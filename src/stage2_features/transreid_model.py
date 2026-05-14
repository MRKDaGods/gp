"""TransReID model for inference in the MTMC pipeline.

TransReID (He et al., ICCV 2021) with:
- ViT backbone (via timm) — supports CLIP ViT-Base and standard ViTs
- Side Information Embedding (SIE) — camera-aware tokens broadcast to ALL tokens
- Jigsaw Patch Module (JPM) — used only during training
- BNNeck + optional projection for deployment features
- norm_pre support for CLIP ViT compatibility (critical for CLIP backbones)

This module is used by Stage 2 (feature extraction) to load
TransReID weights trained on Kaggle (Notebook 07/08).

Architecture must match the training notebook (NB08) exactly:
- Weight key names: ``bn`` (not ``bn_global``), ``cls_head`` (not ``classifier``)
- SIE broadcasts to ALL tokens (not just CLS)
- norm_pre called before transformer blocks (critical for CLIP ViTs)
- Identity projection when embed_dim == vit_dim
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


class TransReID(nn.Module):
    """TransReID: ViT + SIE + JPM for re-identification.

    During inference, returns L2-normalized ``embed_dim``-dimensional features.
    During training, returns (cls_score, proj_feat[, jpm_score]).

    v6 CRITICAL FIX: Includes timm's norm_pre for CLIP compatibility.
    CLIP ViTs use pre-LayerNorm that standard ViTs lack — skipping it
    completely destroys pretrained features.
    """

    def __init__(
        self,
        num_classes: int = 1,
        num_cameras: int = 0,
        embed_dim: int = 768,
        vit_model: str = "vit_base_patch16_clip_224.openai",
        pretrained: bool = False,
        sie_camera: bool = True,
        jpm: bool = True,
        img_size: tuple[int, int] | None = None,
    ):
        super().__init__()
        import timm

        self.sie_camera = sie_camera and num_cameras > 0
        self.jpm = jpm

        # ViT backbone — pass img_size when it differs from the timm default
        # (e.g. person ReID uses 256×128 → grid 16×8 = 128 patches)
        timm_kwargs = dict(pretrained=pretrained, num_classes=0)
        if img_size is not None:
            timm_kwargs["img_size"] = img_size
        self.vit = timm.create_model(vit_model, **timm_kwargs)
        self.vit_dim = self.vit.embed_dim  # 768 for ViT-Base, 384 for ViT-Small
        self.num_blocks = len(self.vit.blocks)

        # Detect architecture features
        has_norm_pre = hasattr(self.vit, "norm_pre") and not isinstance(
            self.vit.norm_pre, nn.Identity
        )
        logger.debug(
            f"TransReID: {vit_model}, vit_dim={self.vit_dim}, "
            f"norm_pre={type(self.vit.norm_pre).__name__} (active={has_norm_pre}), "
            f"blocks={self.num_blocks}"
        )

        # SIE: camera embedding broadcast to ALL tokens (per TransReID paper)
        if self.sie_camera:
            self.sie_embed = nn.Parameter(
                torch.zeros(num_cameras, 1, self.vit_dim)
            )
            nn.init.trunc_normal_(self.sie_embed, std=0.02)

        # BNNeck (named 'bn' to match NB08 training checkpoint keys)
        self.bn = nn.BatchNorm1d(self.vit_dim)
        self.bn.bias.requires_grad_(False)

        # Projection: Identity when embed_dim == vit_dim (e.g., 768 → 768)
        self.proj = (
            nn.Linear(self.vit_dim, embed_dim, bias=False)
            if embed_dim != self.vit_dim
            else nn.Identity()
        )

        # Classifier head (named 'cls_head' to match NB08 checkpoint keys)
        self.cls_head = nn.Linear(embed_dim, num_classes, bias=False)
        if isinstance(self.proj, nn.Linear):
            nn.init.kaiming_normal_(self.proj.weight, mode="fan_out")
        nn.init.normal_(self.cls_head.weight, std=0.001)

        # JPM branch (training only)
        if self.jpm:
            self.bn_jpm = nn.BatchNorm1d(self.vit_dim)
            self.bn_jpm.bias.requires_grad_(False)
            self.jpm_cls = nn.Linear(self.vit_dim, num_classes, bias=False)
            nn.init.normal_(self.jpm_cls.weight, std=0.001)

    def forward(self, x: torch.Tensor, cam_ids: torch.Tensor | None = None):
        B = x.shape[0]
        rot_pos_embed = None

        # 1. Patch embedding
        x = self.vit.patch_embed(x)

        # 2. CLS token + positional embedding + pos_drop (use timm's method)
        if hasattr(self.vit, "_pos_embed"):
            result = self.vit._pos_embed(x)
            if isinstance(result, tuple):
                x, rot_pos_embed = result
            else:
                x = result
        else:
            cls_tok = self.vit.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_tok, x], dim=1) + self.vit.pos_embed
            if hasattr(self.vit, "pos_drop"):
                x = self.vit.pos_drop(x)

        # 3. SIE: camera embedding broadcast to ALL tokens (per TransReID paper)
        if self.sie_camera and cam_ids is not None:
            x = x + self.sie_embed[cam_ids]  # (B,1,D) broadcasts to (B,N+1,D)

        # 4. Patch drop (Identity for most models, but call if present)
        if hasattr(self.vit, "patch_drop"):
            x = self.vit.patch_drop(x)

        # 5. CRITICAL: Pre-normalization (CLIP uses LayerNorm here!)
        #    Standard ViTs have Identity here, so this is a no-op for them.
        #    Skipping this for CLIP ViTs completely destroys pretrained features.
        if hasattr(self.vit, "norm_pre"):
            x = self.vit.norm_pre(x)

        # 6. Transformer blocks + final norm
        for blk in self.vit.blocks:
            if rot_pos_embed is not None:
                x = blk(x, rope=rot_pos_embed)
            else:
                x = blk(x)
        x = self.vit.norm(x)

        # CLS token → global feature
        g_feat = x[:, 0]
        bn = self.bn(g_feat)
        proj = self.proj(bn)

        if self.training:
            cls = self.cls_head(proj)
            if self.jpm:
                patches = x[:, 1:]
                idx = torch.randperm(patches.size(1), device=x.device)
                shuffled = patches[:, idx]
                mid = patches.size(1) // 2
                jpm_feat = (shuffled[:, :mid].mean(1) + shuffled[:, mid:].mean(1)) / 2
                jpm_cls = self.jpm_cls(self.bn_jpm(jpm_feat))
                return cls, proj, jpm_cls
            return cls, proj

        # Inference: L2-normalized embedding
        # If concat_patch is set, concatenate CLS with GeM-pooled patches
        proj_normed = F.normalize(proj, p=2, dim=1)
        if getattr(self, "_concat_patch", False):
            patches = x[:, 1:]  # (B, N, D) — patch tokens
            # GeM pooling: generalized mean with p=3 (more discriminative than avg)
            gem_p = getattr(self, "_gem_p", 3.0)
            patch_gem = (patches.clamp(min=1e-6) ** gem_p).mean(dim=1) ** (1.0 / gem_p)
            patch_normed = F.normalize(patch_gem, p=2, dim=1)
            return torch.cat([proj_normed, patch_normed], dim=1)
        return proj_normed


def build_transreid(
    num_classes: int = 1,
    num_cameras: int = 0,
    embed_dim: int = 768,
    vit_model: str = "vit_base_patch16_clip_224.openai",
    pretrained: bool = False,
    weights_path: str | None = None,
    img_size: tuple[int, int] | None = None,
) -> TransReID:
    """Build TransReID model and optionally load weights.

    Args:
        num_classes: Number of identity classes (1 for inference).
        num_cameras: Number of cameras for SIE (0 to disable).
        embed_dim: Output embedding dimension.
        vit_model: timm model name for the ViT backbone.
        pretrained: Load pretrained ViT weights from timm.
        weights_path: Path to trained TransReID checkpoint.
        img_size: (H, W) input image size.  When different from the timm
            default (224×224), timm creates the ViT with the correct
            patch grid and positional embedding length.

    Returns:
        TransReID model instance.
    """
    model = TransReID(
        num_classes=num_classes,
        num_cameras=num_cameras,
        embed_dim=embed_dim,
        vit_model=vit_model,
        pretrained=pretrained,
        sie_camera=num_cameras > 0,
        jpm=True,
        img_size=img_size,
    )

    if weights_path:
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "model" in state_dict:
            state_dict = state_dict["model"]
        elif "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

        # Strip module. prefix (from DataParallel)
        state_dict = {
            k.replace("module.", "", 1): v for k, v in state_dict.items()
        }

        # Remap 09p-style TransReIDViT keys to pipeline TransReID keys.
        remap_prefixes = {
            "bottleneck.": "bn.",
            "classifier.": "cls_head.",
        }
        remapped = {}
        for key, value in state_dict.items():
            new_key = key
            for old_prefix, new_prefix in remap_prefixes.items():
                if key.startswith(old_prefix):
                    new_key = new_prefix + key[len(old_prefix):]
                    break
            if new_key == "sie.camera_embed.weight":
                new_key = "sie_embed"
                value = value.unsqueeze(1)
            remapped[new_key] = value
        state_dict = remapped

        # ------------------------------------------------------------------
        # Handle shape mismatches between checkpoint and model:
        #   - pos_embed: training resolution may differ from timm default
        #   - sie_embed: checkpoint may have fewer cameras than deployment
        #   - cls_head / jpm_cls: num_classes differs at inference
        # Strategy: drop mismatched keys and let `strict=False` handle them.
        # For sie_embed specifically, we zero-pad if the checkpoint has fewer
        # cameras (safe: extra cameras just get zero SIE bias).
        # ------------------------------------------------------------------
        model_sd = model.state_dict()
        keys_to_drop = []
        for key in list(state_dict.keys()):
            if key not in model_sd:
                continue
            if state_dict[key].shape != model_sd[key].shape:
                if key == "sie_embed" and state_dict[key].shape[0] < model_sd[key].shape[0]:
                    # Zero-pad SIE: copy trained cameras, leave extras as zero
                    ckpt_cams = state_dict[key].shape[0]
                    padded = torch.zeros_like(model_sd[key])
                    padded[:ckpt_cams] = state_dict[key]
                    state_dict[key] = padded
                    logger.info(
                        f"SIE embed: padded {ckpt_cams} → "
                        f"{model_sd[key].shape[0]} cameras"
                    )
                elif key == "vit.pos_embed":
                    ckpt_pe = state_dict[key]  # (1, ckpt_tokens, D)
                    model_pe = model_sd[key]   # (1, model_tokens, D)
                    ckpt_tokens = ckpt_pe.shape[1]
                    model_tokens = model_pe.shape[1]
                    if ckpt_tokens == model_tokens:
                        pass  # same size, load directly
                    elif model_tokens > ckpt_tokens:
                        # Model requests larger grid (e.g. 384×384) than checkpoint
                        # (e.g. 256×256). Bicubic-interpolate grid embeddings.
                        cls_token = ckpt_pe[:, :1, :]  # (1, 1, D)
                        grid_pe = ckpt_pe[:, 1:, :]    # (1, N_ckpt, D)
                        ckpt_grid = int(grid_pe.shape[1] ** 0.5)
                        model_grid = int((model_tokens - 1) ** 0.5)
                        grid_pe = grid_pe.reshape(1, ckpt_grid, ckpt_grid, -1).permute(0, 3, 1, 2)
                        grid_pe = F.interpolate(
                            grid_pe.float(), size=(model_grid, model_grid),
                            mode="bicubic", align_corners=False,
                        ).to(ckpt_pe.dtype)
                        grid_pe = grid_pe.permute(0, 2, 3, 1).reshape(1, model_grid * model_grid, -1)
                        state_dict[key] = torch.cat([cls_token, grid_pe], dim=1)
                        logger.info(
                            f"Interpolated vit.pos_embed: {ckpt_grid}×{ckpt_grid} → "
                            f"{model_grid}×{model_grid} grid (bicubic)"
                        )
                    else:
                        # Checkpoint has more tokens — resize model to match
                        logger.info(
                            f"Resizing vit.pos_embed: model {model_sd[key].shape} → "
                            f"checkpoint {state_dict[key].shape}"
                        )
                        model.vit.pos_embed = nn.Parameter(
                            torch.zeros_like(state_dict[key]), requires_grad=False,
                        )
                else:
                    # Shape mismatch on classifier heads etc. — drop
                    logger.debug(
                        f"Dropping key {key}: ckpt {state_dict[key].shape} "
                        f"vs model {model_sd[key].shape}"
                    )
                    keys_to_drop.append(key)
        for key in keys_to_drop:
            del state_dict[key]

        # Load with relaxed strictness (num_classes may differ)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            # Filter out classifier/cls_head which are expected to mismatch
            critical_missing = [
                k for k in missing
                if not any(skip in k for skip in ("cls_head", "classifier", "jpm_cls"))
            ]
            if critical_missing:
                logger.warning(f"TransReID critical missing keys: {critical_missing}")
            else:
                logger.debug(f"TransReID missing keys (non-critical): {len(missing)}")
        if unexpected:
            logger.debug(f"TransReID unexpected keys: {len(unexpected)}")
        logger.info(f"Loaded TransReID weights from {weights_path}")

    return model
