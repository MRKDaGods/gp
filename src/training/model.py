"""ReID model with BNNeck for BoT-style training.

Implements the strong baseline architecture from:
  Luo et al., "Bag of Tricks and A Strong Baseline for Deep Person
  Re-identification" (CVPRW 2019, TMM 2019).

Key features:
  - ResNet50-IBN-a backbone with configurable last stride
  - BNNeck: BN layer between feature and classifier
  - Supports ID loss (on BN features) + Triplet loss (on raw features)
  - Compatible with torchreid model zoo weights
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


def _build_backbone(model_name: str, last_stride: int = 1, pretrained: bool = True):
    """Build backbone using torchreid model zoo.

    Args:
        model_name: torchreid model name (e.g., 'resnet50_ibn_a').
        last_stride: Stride of the last conv block (1 for higher resolution).
        pretrained: Whether to load ImageNet pre-trained weights.
    """
    import torchreid

    model = torchreid.models.build_model(
        name=model_name,
        num_classes=1,  # placeholder
        loss="softmax",
        pretrained=pretrained,
    )

    # Modify last stride for ResNet-family models
    if hasattr(model, "layer4"):
        # ResNet / ResNet-IBN
        for module in model.layer4.modules():
            if hasattr(module, "stride"):
                if isinstance(module.stride, tuple) and module.stride == (2, 2):
                    module.stride = (last_stride, last_stride)
                elif isinstance(module, nn.Conv2d) and module.stride == (2, 2):
                    module.stride = (last_stride, last_stride)

    return model


class IBN_a(nn.Module):
    """Instance-Batch Normalization (IBN-a) layer."""

    def __init__(self, planes: int):
        super().__init__()
        half = planes // 2
        self.IN = nn.InstanceNorm2d(half, affine=True)
        self.BN = nn.BatchNorm2d(planes - half)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        split = x.shape[1] // 2
        out1 = self.IN(x[:, :split])
        out2 = self.BN(x[:, split:])
        return torch.cat([out1, out2], dim=1)


class GeM(nn.Module):
    """Generalized Mean Pooling."""

    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), 1
        ).pow(1.0 / self.p)


def _build_resnet101_ibn_a(last_stride: int = 1, pretrained: bool = True) -> nn.Module:
    """Build ResNet101 with IBN-a layers on layer1 and layer2."""
    import torchvision.models as tv_models

    weights = tv_models.ResNet101_Weights.DEFAULT if pretrained else None
    base = tv_models.resnet101(weights=weights)

    for layer in [base.layer1, base.layer2]:
        for block in layer:
            if hasattr(block, "bn1"):
                block.bn1 = IBN_a(block.bn1.num_features)

    if last_stride == 1:
        for module in base.layer4.modules():
            if isinstance(module, nn.Conv2d) and module.stride == (2, 2):
                module.stride = (1, 1)

    base.fc = nn.Identity()
    base.avgpool = nn.Identity()
    return base


class ReIDModelResNet101IBN(nn.Module):
    """ResNet101-IBN-a with GeM pooling and BNNeck for BoT-style training."""

    def __init__(
        self,
        num_classes: int = 751,
        last_stride: int = 1,
        pretrained: bool = True,
        gem_p: float = 3.0,
    ):
        super().__init__()
        self.backbone = _build_resnet101_ibn_a(last_stride, pretrained)
        self.feat_dim = 2048
        self.pool = GeM(p=gem_p)

        self.bottleneck = nn.BatchNorm1d(self.feat_dim)
        self.bottleneck.bias.requires_grad_(False)
        nn.init.constant_(self.bottleneck.weight, 1)
        nn.init.constant_(self.bottleneck.bias, 0)

        self.classifier = nn.Linear(self.feat_dim, num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)

        logger.info(
            f"ReIDModelResNet101IBN: classes={num_classes}, "
            f"feat_dim={self.feat_dim}, last_stride={last_stride}, gem_p={gem_p}"
        )

    def _backbone_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        return x

    def forward(self, x: torch.Tensor):
        feat_map = self._backbone_forward(x)
        global_feat = self.pool(feat_map)
        global_feat = global_feat.view(global_feat.shape[0], -1)
        bn_feat = self.bottleneck(global_feat)

        if self.training:
            cls_score = self.classifier(bn_feat)
            return cls_score, global_feat, bn_feat
        return F.normalize(bn_feat, p=2, dim=1)

    @torch.no_grad()
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.forward(x)


class ReIDModelBoT(nn.Module):
    """ReID model with Bag-of-Tricks (BoT) training setup.

    Architecture:
        backbone → GAP → feat (for triplet loss)
                       → BNNeck → classifier (for ID loss)

    During inference, extracts the BN-normalized feature (after BNNeck)
    for cosine-distance matching, or the raw feature for Euclidean matching.
    """

    def __init__(
        self,
        model_name: str = "resnet50_ibn_a",
        num_classes: int = 751,
        last_stride: int = 1,
        pretrained: bool = True,
        feat_dim: int = 2048,
        neck: str = "bnneck",  # 'bnneck' or 'no'
    ):
        super().__init__()
        self.model_name = model_name
        self.feat_dim = feat_dim
        self.neck_type = neck

        # Build backbone
        self.backbone = _build_backbone(model_name, last_stride, pretrained)

        # Remove the original classifier and global pool
        # torchreid models have .classifier and .global_avgpool
        self.gap = nn.AdaptiveAvgPool2d(1)

        # BNNeck
        self.bottleneck = nn.BatchNorm1d(feat_dim)
        self.bottleneck.bias.requires_grad_(False)  # no bias
        nn.init.constant_(self.bottleneck.weight, 1)
        nn.init.constant_(self.bottleneck.bias, 0)

        # Classifier
        self.classifier = nn.Linear(feat_dim, num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)

        logger.info(
            f"ReIDModelBoT: backbone={model_name}, classes={num_classes}, "
            f"feat_dim={feat_dim}, last_stride={last_stride}, neck={neck}"
        )

    def _backbone_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from backbone before the classifier head."""
        # torchreid ResNet models implement featuremaps()
        if hasattr(self.backbone, "featuremaps"):
            feat_map = self.backbone.featuremaps(x)
        else:
            # Fallback: run through all layers except classifier
            feat_map = self.backbone.conv1(x)
            feat_map = self.backbone.bn1(feat_map)
            feat_map = self.backbone.relu(feat_map)
            feat_map = self.backbone.maxpool(feat_map)
            feat_map = self.backbone.layer1(feat_map)
            feat_map = self.backbone.layer2(feat_map)
            feat_map = self.backbone.layer3(feat_map)
            feat_map = self.backbone.layer4(feat_map)
        return feat_map

    def forward(self, x: torch.Tensor):
        """Forward pass.

        During training, returns (cls_score, global_feat, bn_feat).
        During eval, returns bn_feat (for cosine matching).
        """
        feat_map = self._backbone_forward(x)   # (B, C, H, W)
        global_feat = self.gap(feat_map)         # (B, C, 1, 1)
        global_feat = global_feat.view(global_feat.shape[0], -1)  # (B, C)

        if self.neck_type == "bnneck":
            bn_feat = self.bottleneck(global_feat)  # (B, C)
        else:
            bn_feat = global_feat

        if self.training:
            cls_score = self.classifier(bn_feat)
            return cls_score, global_feat, bn_feat
        else:
            return bn_feat  # Use BN feature for inference

    def load_pretrained_reid(self, weights_path: str):
        """Load pre-trained ReID weights (e.g., from torchreid checkpoint)."""
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "model" in state_dict:
            state_dict = state_dict["model"]

        # Strip module. prefix
        state_dict = {
            k.replace("module.", "", 1): v for k, v in state_dict.items()
        }

        # Try to load backbone weights
        backbone_state = {}
        for k, v in state_dict.items():
            if k.startswith("classifier"):
                continue
            backbone_state[k] = v

        missing, unexpected = self.backbone.load_state_dict(
            backbone_state, strict=False
        )
        if missing:
            logger.debug(f"Missing keys in backbone: {len(missing)}")
        if unexpected:
            logger.debug(f"Unexpected keys in backbone: {len(unexpected)}")
        logger.info(f"Loaded pre-trained ReID weights from {weights_path}")

    @torch.no_grad()
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract inference features (BN-normalized)."""
        self.eval()
        return self.forward(x)
