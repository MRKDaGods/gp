"""DINOv2-Large TransReID variant — verbatim port of Kaggle kernel
``yahiaakhalafallah/09s-dinov2-large-cityflowv2`` (cells 4/6/7).

This is the arch that trained ``vehicle_transreid_dinov2_large_cityflowv2_
final.pth`` — the 14e B1 TERTIARY fusion stream (w_tertiary=0.525 in the
0.77936 CityFlowV2 headline). It is a timm-backboned TransReID: ViT-L/14
DINOv2 backbone, SIE camera embedding on patch tokens, BN-neck, identity
projection (embed_dim == vit_dim == 1024 for ViT-L), JPM head at train
time. Eval forward returns the L2-normalized projected feature.

Port deviations (matching every prior kernel port): ``print`` -> stdlib
logging with %-style args; module-level notebook constants kept as
UPPER_CASE globals. Math, structure, init, and forward are byte-faithful.
Training-only pieces of the notebook (losses, LLRD optimizer wiring,
samplers) are NOT ported — training stays on Kaggle per [[gp-workflow-rules]].
"""

from __future__ import annotations

import logging

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---- cell 4 constants (inference-relevant subset) -------------------------
VIT_MODEL = "vit_large_patch14_dinov2"
IMG_SIZE = 252
STRIDE_SIZE = 14
EMBED_DIM = 1024

# ---- cell 6: eval-time preprocessing --------------------------------------
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def build_test_transform(img_size: int = IMG_SIZE):
    """The notebook's ``test_tf`` (RGB PIL in, normalized tensor out)."""
    import torchvision.transforms as T

    return T.Compose([
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])


# ---- cell 7: backbone resolution + model ----------------------------------
BACKBONE_ALIASES = {
    "vit_large_patch16_224_TransReID": [
        "vit_large_patch16_224.augreg_in21k_ft_in1k",
        "vit_large_patch16_224.augreg_in21k",
        "vit_large_patch16_224",
    ],
    "vit_base_patch16_224_TransReID": [
        "vit_base_patch16_224.augreg_in21k_ft_in1k",
        "vit_base_patch16_224.augreg_in21k",
        "vit_base_patch16_224",
    ],
}

BACKBONE_ALIASES["vit_large_patch14_dinov2"] = [
    "vit_large_patch14_dinov2.lvd142m",
    "vit_large_patch14_dinov2",
]


def resolve_backbone_name(vit_model: str) -> str:
    candidates = BACKBONE_ALIASES.get(vit_model, [vit_model])
    fallback = BACKBONE_ALIASES["vit_base_patch16_224_TransReID"]
    if vit_model == "vit_large_patch16_224_TransReID":
        candidates = candidates + fallback
    last_error = None
    for candidate in candidates:
        try:
            probe = timm.create_model(candidate, pretrained=False, num_classes=0, img_size=IMG_SIZE)
            del probe
            if candidate not in BACKBONE_ALIASES.get(vit_model, []):
                logger.warning("Falling back to backbone alias %s for %s", candidate, vit_model)
            return candidate
        except Exception as error:
            last_error = error
    raise RuntimeError(f"Could not resolve a timm backbone for {vit_model}: {last_error}")


class TransReID(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_cameras: int = 0,
        embed_dim: int = 768,
        vit_model: str = "vit_base_patch16_224_TransReID",
        pretrained: bool = True,
        sie_camera: bool = True,
        jpm: bool = True,
        img_size: int | tuple[int, int] = 224,
        stride_size: int = 16,
    ):
        super().__init__()
        self.sie_camera = sie_camera and num_cameras > 0
        self.jpm = jpm
        self.stride_size = stride_size
        self.requested_vit_model = vit_model
        self.timm_backbone = resolve_backbone_name(vit_model)
        self.vit = timm.create_model(
            self.timm_backbone,
            pretrained=pretrained,
            num_classes=0,
            img_size=img_size,
        )
        if hasattr(self.vit, "set_grad_checkpointing"):
            self.vit.set_grad_checkpointing(enable=True)
            logger.info("Gradient checkpointing enabled for ViT backbone")
        self.vit_dim = self.vit.embed_dim
        self.num_blocks = len(self.vit.blocks)

        if self.sie_camera:
            self.sie_embed = nn.Parameter(torch.zeros(num_cameras, self.vit_dim))
            nn.init.trunc_normal_(self.sie_embed, std=0.02)

        self.bn = nn.BatchNorm1d(self.vit_dim)
        self.bn.bias.requires_grad_(False)
        self.proj = nn.Linear(self.vit_dim, embed_dim, bias=False) if embed_dim != self.vit_dim else nn.Identity()
        self.cls_head = nn.Linear(embed_dim, num_classes, bias=False)
        if isinstance(self.proj, nn.Linear):
            nn.init.kaiming_normal_(self.proj.weight, mode="fan_out")
        nn.init.normal_(self.cls_head.weight, std=0.001)

        if self.jpm:
            self.bn_jpm = nn.BatchNorm1d(self.vit_dim)
            self.bn_jpm.bias.requires_grad_(False)
            self.jpm_cls = nn.Linear(self.vit_dim, num_classes, bias=False)
            nn.init.normal_(self.jpm_cls.weight, std=0.001)

        logger.info(
            "TransReID backbone=%s -> %s | vit_dim=%s | embed_dim=%s | cameras=%s",
            self.requested_vit_model, self.timm_backbone,
            self.vit_dim, embed_dim, num_cameras,
        )

    def forward(self, x: torch.Tensor, cam_ids: torch.Tensor | None = None):
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
            # only add SIE to patch tokens, not cls token
            x[:, 1:] = x[:, 1:] + self.sie_embed[cam_ids].unsqueeze(1)

        if hasattr(self.vit, "patch_drop"):
            x = self.vit.patch_drop(x)
        if hasattr(self.vit, "norm_pre"):
            x = self.vit.norm_pre(x)

        for block in self.vit.blocks:
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

    def get_llrd_param_groups(self, backbone_lr: float, head_lr: float, decay: float = 0.75):
        groups = {}
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith("vit."):
                if "blocks." in name:
                    block_index = int(name.split("blocks.")[1].split(".")[0])
                    depth = block_index + 1
                elif any(token in name for token in ("patch_embed", "cls_token", "pos_embed", "norm_pre")):
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


# ---- v2 glue (NOT from the kernel) ----------------------------------------

def infer_checkpoint_dims(state_dict: dict) -> tuple[int, int]:
    """(num_classes, num_cameras) from the saved full state dict — the
    notebook derived both from its CityFlowV2 train split; the shapes of
    ``cls_head.weight`` / ``sie_embed`` carry them, so a strict offline
    reconstruction never has to guess."""
    if "cls_head.weight" not in state_dict:
        raise KeyError("not a 09s TransReID checkpoint: cls_head.weight missing")
    num_classes = int(state_dict["cls_head.weight"].shape[0])
    num_cameras = (
        int(state_dict["sie_embed"].shape[0]) if "sie_embed" in state_dict else 0
    )
    return num_classes, num_cameras
