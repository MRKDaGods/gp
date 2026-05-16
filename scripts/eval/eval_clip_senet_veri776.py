"""Standalone CLIP-SENet v6 VeRi-776 evaluation extracted from 13e.

Source notebook: notebooks/kaggle/13e_clip_senet_eval/13e_clip_senet_eval.ipynb.
The model definition, checkpoint payload handling, ImageNet normalization,
feature extraction, Market1501-style metric, AQE, and k-reciprocal rerank
logic are lifted from the notebook with CLI/output plumbing around them.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

logger = logging.getLogger(__name__)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)

DEFAULT_NUM_CLASSES = 575
IMAGE_SIZE = (320, 320)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
NUM_WORKERS = 2
FILENAME_RE = re.compile(r"^(?P<pid>-?\d+)_c(?P<camid>\d+)")
RERANK_K1_VALUES = [10, 20, 30, 50]
RERANK_K2_VALUES = [3, 6, 10, 15]
RERANK_LAMBDAS = [0.1, 0.2, 0.3, 0.5, 0.7]


@dataclass(frozen=True)
class LoadedBackboneInfo:
    """Describes the backbone variant that was successfully loaded."""

    family: str
    model_name: str
    pretrained_tag: str | None = None


class AFEMBlock(nn.Module):
    """Adaptive Fine-grained Enhancement Module.

    This implements the paper's ambiguous Eq. (4) using the `(G + 1)`
    interpretation: `G` grouped weighted residual chunks plus one identity
    residual path. Set `residual_mode="sum_only"` to drop the identity term and
    return only the weighted grouped sum.
    """

    def __init__(
        self,
        in_dim: int = 2048,
        out_dim: int = 2048,
        num_groups: int = 32,
        residual_mode: str = "grouped_identity",
    ):
        super().__init__()
        if out_dim % num_groups != 0:
            raise ValueError(
                f"AFEM out_dim={out_dim} must be divisible by num_groups={num_groups}"
            )
        if residual_mode not in {"grouped_identity", "sum_only"}:
            raise ValueError(
                "residual_mode must be 'grouped_identity' or 'sum_only'"
            )

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_groups = num_groups
        self.group_dim = out_dim // num_groups
        self.residual_mode = residual_mode

        self.shared = nn.Sequential(
            nn.Linear(in_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
        )
        self.group_weights = nn.Parameter(torch.randn(num_groups, self.group_dim))

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        linear = self.shared[0]
        nn.init.kaiming_normal_(linear.weight, mode="fan_out")
        bn = self.shared[1]
        nn.init.ones_(bn.weight)
        nn.init.zeros_(bn.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.shared(x)
        grouped = h.view(h.shape[0], self.num_groups, self.group_dim)
        weighted = grouped * self.group_weights.unsqueeze(0)
        enhanced = weighted.reshape(h.shape[0], self.out_dim)
        if self.residual_mode == "sum_only":
            return enhanced
        return h + enhanced


class _ResNetFeatureWrapper(nn.Module):
    """Wrap torchvision-style ResNet backbones to expose pooled 2048-d features."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        return torch.flatten(x, 1)


class ResNet101IBNBranch(nn.Module):
    """Appearance branch backed by real ResNet101 IBN-a with deterministic fallbacks."""

    _IBN_MODEL = "resnet101_ibn_a"
    _FALLBACK_MODEL = "resnet101"

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.output_dim = 2048
        self.backbone: nn.Module
        self.loaded_backbone: LoadedBackboneInfo

        for loader in (
            self._load_pretrainedmodels_ibn,
            self._load_torch_hub_ibn,
            self._load_timm_ibn,
            self._load_timm_plain,
        ):
            loaded = loader(pretrained=pretrained)
            if loaded is None:
                continue
            self.backbone, self.loaded_backbone = loaded
            logger.info(
                "Appearance branch loaded via '%s' model='%s' pretrained_tag='%s'",
                self.loaded_backbone.family,
                self.loaded_backbone.model_name,
                self.loaded_backbone.pretrained_tag,
            )
            return

        raise ImportError(
            "Unable to load appearance backbone via pretrainedmodels, torch.hub, or timm"
        )

    def _load_pretrainedmodels_ibn(
        self, pretrained: bool
    ) -> tuple[nn.Module, LoadedBackboneInfo] | None:
        try:
            import pretrainedmodels
        except ImportError:
            logger.warning(
                "Appearance branch loader 'pretrainedmodels' is unavailable; trying torch.hub"
            )
            return None

        constructor = getattr(pretrainedmodels, self._IBN_MODEL, None)
        if constructor is None:
            logger.warning(
                "Appearance branch loader 'pretrainedmodels' has no '%s' entry; trying torch.hub",
                self._IBN_MODEL,
            )
            return None

        pretrained_tag = "imagenet" if pretrained else None
        try:
            raw_model = constructor(pretrained=pretrained_tag)
        except Exception as exc:  # noqa: BLE001 - keep fallback chain moving
            logger.warning(
                "Appearance branch loader 'pretrainedmodels' failed for '%s': %s",
                self._IBN_MODEL,
                exc,
            )
            return None

        if hasattr(raw_model, "last_linear"):
            raw_model.last_linear = nn.Identity()
        backbone = _ResNetFeatureWrapper(raw_model)
        return backbone, LoadedBackboneInfo(
            family="pretrainedmodels",
            model_name=self._IBN_MODEL,
            pretrained_tag=pretrained_tag or "random_init",
        )

    def _load_torch_hub_ibn(
        self, pretrained: bool
    ) -> tuple[nn.Module, LoadedBackboneInfo] | None:
        try:
            raw_model = torch.hub.load(
                "XingangPan/IBN-Net",
                self._IBN_MODEL,
                pretrained=pretrained,
                trust_repo=True,
            )
        except Exception as exc:  # noqa: BLE001 - keep fallback chain moving
            logger.warning(
                "Appearance branch loader 'torch.hub' failed for '{}': {}",
                self._IBN_MODEL,
                exc,
            )
            return None

        if hasattr(raw_model, "fc"):
            raw_model.fc = nn.Identity()
        backbone = _ResNetFeatureWrapper(raw_model)
        return backbone, LoadedBackboneInfo(
            family="torch.hub",
            model_name=self._IBN_MODEL,
            pretrained_tag="official_pretrained" if pretrained else "random_init",
        )

    def _load_timm_ibn(
        self, pretrained: bool
    ) -> tuple[nn.Module, LoadedBackboneInfo] | None:
        try:
            import timm
        except ImportError as exc:
            raise ImportError("timm is required for ResNet101IBNBranch fallbacks") from exc

        available = set(timm.list_models())
        if self._IBN_MODEL not in available:
            logger.warning(
                "Appearance branch loader 'timm' has no '%s' entry; trying plain '%s'",
                self._IBN_MODEL,
                self._FALLBACK_MODEL,
            )
            return None

        try:
            backbone = timm.create_model(
                self._IBN_MODEL,
                pretrained=pretrained,
                num_classes=0,
                global_pool="avg",
            )
        except Exception as exc:  # noqa: BLE001 - keep fallback chain moving
            logger.warning(
                "Appearance branch loader 'timm' failed for '%s': %s",
                self._IBN_MODEL,
                exc,
            )
            return None

        return backbone, LoadedBackboneInfo(
            family="timm",
            model_name=self._IBN_MODEL,
            pretrained_tag="timm_pretrained" if pretrained else "random_init",
        )

    def _load_timm_plain(
        self, pretrained: bool
    ) -> tuple[nn.Module, LoadedBackboneInfo] | None:
        try:
            import timm
        except ImportError as exc:
            raise ImportError("timm is required for ResNet101IBNBranch fallbacks") from exc

        try:
            backbone = timm.create_model(
                self._FALLBACK_MODEL,
                pretrained=pretrained,
                num_classes=0,
                global_pool="avg",
            )
        except Exception as exc:  # noqa: BLE001 - keep fallback chain moving
            logger.warning(
                "Appearance branch loader 'timm' failed for plain '%s': %s",
                self._FALLBACK_MODEL,
                exc,
            )
            return None

        logger.warning(
            "Appearance branch fell back to plain '%s' because no IBN-a loader succeeded",
            self._FALLBACK_MODEL,
        )
        return backbone, LoadedBackboneInfo(
            family="timm",
            model_name=self._FALLBACK_MODEL,
            pretrained_tag="timm_pretrained" if pretrained else "random_init",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x)
        if out.ndim != 2:
            raise RuntimeError(
                f"Appearance branch expected pooled 2D output, got shape {tuple(out.shape)}"
            )
        return out


class TinyCLIPImageBranch(nn.Module):
    """Semantic branch that loads TinyCLIP with a deterministic fallback chain."""

    _OPEN_CLIP_CHAIN = (
        {
            "model_name": "hf-hub:wkcn/TinyCLIP-ViT-45M-32-Text-21M-LAION400M",
            "pretrained_tag": None,
        },
        {
            "model_name": "TinyCLIP-ViT-40M-32-Text-19M",
            "pretrained_tag": "laion400m_e32",
        },
    )
    _TIMM_TINYCLIP_CHAIN = (
        "vit_medium_patch32_clip_224.tinyclip_laion400m",
    )
    _LAST_RESORT_OPEN_CLIP = ("ViT-B-32", "openai")

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.provider = ""
        self.model = None
        self.loaded_backbone: LoadedBackboneInfo | None = None
        last_error = self._try_load_open_clip(pretrained=pretrained)
        if self.model is None:
            last_error = self._try_load_timm_tinyclip(pretrained=pretrained) or last_error
        if self.model is None:
            last_error = self._try_load_open_clip_last_resort(pretrained=pretrained) or last_error

        if self.model is None or self.loaded_backbone is None:
            raise RuntimeError(
                "Unable to load any TinyCLIP/OpenCLIP visual backbone"
            ) from last_error

        self.image_size = self._infer_image_size(self.model)

    def _try_load_open_clip(self, pretrained: bool) -> Exception | None:
        try:
            import open_clip
        except ImportError as exc:
            return exc

        last_error: Exception | None = None
        for candidate in self._OPEN_CLIP_CHAIN:
            model_name = candidate["model_name"]
            pretrained_tag = candidate["pretrained_tag"]
            try:
                if pretrained_tag is None:
                    model, _, _ = open_clip.create_model_and_transforms(model_name)
                else:
                    model, _, _ = open_clip.create_model_and_transforms(
                        model_name,
                        pretrained=pretrained_tag if pretrained else None,
                    )
            except Exception as exc:  # noqa: BLE001 - preserve fallback chain context
                last_error = exc
                logger.warning(
                    "TinyCLIP load failed for model='%s' pretrained='%s': %s",
                    model_name,
                    pretrained_tag or "hf-hub-default",
                    exc,
                )
                continue

            self.model = model
            self.provider = "open_clip"
            self.loaded_backbone = LoadedBackboneInfo(
                family="semantic",
                model_name=model_name,
                pretrained_tag=pretrained_tag if pretrained else "random_init",
            )
            self.output_dim = self._infer_open_clip_output_dim(model)
            logger.info(
                "TinyCLIP branch loaded model='%s' pretrained='%s' via open_clip output_dim=%s",
                model_name,
                pretrained_tag if pretrained_tag is not None and pretrained else "hf-hub-default",
                self.output_dim,
            )
            return None

        return last_error

    def _try_load_timm_tinyclip(self, pretrained: bool) -> Exception | None:
        try:
            import timm
        except ImportError as exc:
            return exc

        last_error: Exception | None = None
        for model_name in self._TIMM_TINYCLIP_CHAIN:
            try:
                model = timm.create_model(
                    model_name,
                    pretrained=pretrained,
                    num_classes=0,
                )
            except Exception as exc:  # noqa: BLE001 - preserve fallback chain context
                last_error = exc
                logger.warning(
                    "TinyCLIP-equivalent timm load failed for model='%s': %s",
                    model_name,
                    exc,
                )
                continue

            self.model = model
            self.provider = "timm"
            self.loaded_backbone = LoadedBackboneInfo(
                family="semantic",
                model_name=model_name,
                pretrained_tag="timm_pretrained" if pretrained else "random_init",
            )
            self.output_dim = self._infer_timm_output_dim(model)
            logger.info(
                "TinyCLIP branch loaded model='%s' via timm output_dim=%s",
                model_name,
                self.output_dim,
            )
            return None

        return last_error

    def _try_load_open_clip_last_resort(self, pretrained: bool) -> Exception | None:
        try:
            import open_clip
        except ImportError as exc:
            return exc

        model_name, pretrained_tag = self._LAST_RESORT_OPEN_CLIP
        try:
            model, _, _ = open_clip.create_model_and_transforms(
                model_name,
                pretrained=pretrained_tag if pretrained else None,
            )
        except Exception as exc:  # noqa: BLE001 - explicit last resort context
            logger.warning(
                "OpenCLIP last resort load failed for model='%s' pretrained='%s': %s",
                model_name,
                pretrained_tag,
                exc,
            )
            return exc

        self.model = model
        self.provider = "open_clip"
        self.loaded_backbone = LoadedBackboneInfo(
            family="semantic",
            model_name=model_name,
            pretrained_tag=pretrained_tag if pretrained else "random_init",
        )
        self.output_dim = self._infer_open_clip_output_dim(model)
        logger.info(
            "TinyCLIP branch loaded model='%s' pretrained='%s' via open_clip output_dim=%s",
            model_name,
            pretrained_tag if pretrained else "random_init",
            self.output_dim,
        )
        return None

    @staticmethod
    def _infer_open_clip_output_dim(model: nn.Module) -> int:
        visual = getattr(model, "visual", None)
        output_dim = getattr(visual, "output_dim", None)
        if isinstance(output_dim, int):
            return output_dim

        visual_proj = getattr(model, "visual_projection", None)
        if isinstance(visual_proj, torch.Tensor) and visual_proj.ndim == 2:
            return int(visual_proj.shape[-1])

        visual_proj = getattr(visual, "proj", None)
        if isinstance(visual_proj, torch.Tensor):
            if visual_proj.ndim == 1:
                return int(visual_proj.shape[0])
            if visual_proj.ndim == 2:
                return int(visual_proj.shape[-1])

        raise RuntimeError("Could not infer TinyCLIP visual output dimension")

    @staticmethod
    def _infer_timm_output_dim(model: nn.Module) -> int:
        output_dim = getattr(model, "num_features", None)
        if isinstance(output_dim, int):
            return output_dim
        raise RuntimeError("Could not infer timm TinyCLIP visual output dimension")

    @staticmethod
    def _infer_image_size(model: nn.Module) -> tuple[int, int]:
        pretrained_cfg = getattr(model, "pretrained_cfg", None)
        if isinstance(pretrained_cfg, dict):
            input_size = pretrained_cfg.get("input_size")
            if isinstance(input_size, tuple) and len(input_size) == 3:
                return (int(input_size[-2]), int(input_size[-1]))

        visual = getattr(model, "visual", None)
        image_size = getattr(visual, "image_size", None)
        if isinstance(image_size, int):
            return (image_size, image_size)
        if isinstance(image_size, tuple) and len(image_size) == 2:
            return (int(image_size[0]), int(image_size[1]))
        return (224, 224)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if tuple(x.shape[-2:]) != self.image_size:
            x = F.interpolate(
                x,
                size=self.image_size,
                mode="bilinear",
                align_corners=False,
            )
        if self.provider == "open_clip":
            features = self.model.encode_image(x, normalize=False)
        else:
            features = self.model(x)
        if features.ndim != 2:
            raise RuntimeError(
                f"TinyCLIP branch expected 2D image features, got shape {tuple(features.shape)}"
            )
        return features


class CLIPSENet(nn.Module):
    """CLIP-SENet with a CNN appearance branch and a CLIP semantic branch."""

    def __init__(
        self,
        num_classes: int,
        embed_dim: int = 2048,
        afem_groups: int = 32,
        feat_dim_appearance: int = 2048,
        feat_dim_semantic: int = 512,
        dropout: float = 0.0,
        appearance_pretrained: bool = True,
        semantic_pretrained: bool = True,
        residual_mode: str = "grouped_identity",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        self.appearance_branch = ResNet101IBNBranch(pretrained=appearance_pretrained)
        self.semantic_branch = TinyCLIPImageBranch(pretrained=semantic_pretrained)

        detected_app_dim = self.appearance_branch.output_dim
        detected_sem_dim = self.semantic_branch.output_dim
        if feat_dim_appearance != detected_app_dim:
            logger.warning(
                "Requested feat_dim_appearance=%s but backbone reports %s. Using detected dim.",
                feat_dim_appearance,
                detected_app_dim,
            )
        if feat_dim_semantic != detected_sem_dim:
            logger.warning(
                "Requested feat_dim_semantic=%s but backbone reports %s. Using detected dim.",
                feat_dim_semantic,
                detected_sem_dim,
            )

        self.feat_dim_appearance = detected_app_dim
        self.feat_dim_semantic = detected_sem_dim
        self.fusion_fc = nn.Linear(
            self.feat_dim_appearance + self.feat_dim_semantic,
            embed_dim,
            bias=False,
        )
        self.afem = AFEMBlock(
            in_dim=embed_dim,
            out_dim=embed_dim,
            num_groups=afem_groups,
            residual_mode=residual_mode,
        )
        self.bnneck = nn.BatchNorm1d(embed_dim)
        self.bnneck.bias.requires_grad_(False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(embed_dim, num_classes, bias=False)

        nn.init.kaiming_normal_(self.fusion_fc.weight, mode="fan_out")
        nn.init.normal_(self.classifier.weight, std=0.001)

        self.loaded_resnext_model = self.appearance_branch.loaded_backbone.model_name
        self.loaded_tinyclip_model = self.semantic_branch.loaded_backbone.model_name

    def forward(self, x: torch.Tensor):
        f_app = self.appearance_branch(x)
        f_sem = self.semantic_branch(x)
        t_u = self.fusion_fc(torch.cat([f_app, f_sem], dim=1))
        t_s_prime = self.afem(t_u)
        t = t_u + t_s_prime
        t_bn = self.bnneck(t)
        t_bn_normalized = F.normalize(t_bn, p=2, dim=1)

        if self.training:
            logits = self.classifier(self.dropout(t_bn))
            return t_bn_normalized, logits

        return t_bn_normalized


def build_clip_senet(num_classes: int, **kwargs) -> CLIPSENet:
    """Build a CLIP-SENet model for M1 architecture validation."""

    return CLIPSENet(num_classes=num_classes, **kwargs)


def torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_checkpoint(path: Path):
    payload = torch_load(path)
    if isinstance(payload, dict) and "model_state" in payload:
        state_dict = payload["model_state"]
        checkpoint_kind = "payload:model_state"
    elif isinstance(payload, dict) and "model" in payload and isinstance(payload["model"], dict):
        state_dict = payload["model"]
        checkpoint_kind = "payload:model"
    elif isinstance(payload, dict) and payload and all(hasattr(value, "shape") for value in payload.values()):
        state_dict = payload
        checkpoint_kind = "state_dict"
    else:
        raise TypeError(f"Unsupported checkpoint format at {path}: {type(payload).__name__}")

    classifier_weight = state_dict.get("classifier.weight")
    inferred_num_classes = int(classifier_weight.shape[0]) if classifier_weight is not None else DEFAULT_NUM_CLASSES
    return state_dict, checkpoint_kind, inferred_num_classes


def parse_veri_record(img_path: Path):
    match = FILENAME_RE.match(img_path.stem)
    if match is None:
        raise RuntimeError(f"Unexpected VeRi filename: {img_path.name}")
    return {
        "path": str(img_path),
        "pid": int(match.group("pid")),
        "camid": int(match.group("camid")) - 1,
    }


def parse_split(split_dir: Path):
    items = []
    pid_set = set()
    for img_path in sorted(split_dir.glob("*.jpg")):
        record = parse_veri_record(img_path)
        if record["pid"] == -1:
            continue
        pid_set.add(record["pid"])
        items.append(record)
    return items, len(pid_set)


class VeRiEvalDataset(Dataset):
    def __init__(self, items, transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        image = Image.open(item["path"]).convert("RGB")
        tensor = self.transform(image)
        return tensor, int(item["pid"]), int(item["camid"]), item["path"]


def build_transform(image_size: tuple[int, int]):
    return T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_loader(items, image_size: tuple[int, int], batch_size: int):
    return DataLoader(
        VeRiEvalDataset(items, build_transform(image_size)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )


def build_veri_loaders(veri_root: Path, image_size: tuple[int, int], batch_size: int):
    required = ("image_query", "image_test")
    missing = [split for split in required if not (veri_root / split).is_dir()]
    if missing:
        raise FileNotFoundError(f"VeRi root {veri_root} is missing required splits: {missing}")
    query_items, query_ids = parse_split(veri_root / "image_query")
    gallery_items, gallery_ids = parse_split(veri_root / "image_test")
    if not query_items or not gallery_items:
        raise RuntimeError(
            f"VeRi split is empty: query={len(query_items)} gallery={len(gallery_items)}"
        )
    query_loader = build_loader(query_items, image_size, batch_size)
    gallery_loader = build_loader(gallery_items, image_size, batch_size)
    return query_loader, gallery_loader, query_items, gallery_items, query_ids, gallery_ids


@torch.no_grad()
def extract_features(model, dataloader, device: str):
    model.eval()
    features = []
    pids = []
    camids = []

    for images, batch_pids, batch_camids, _ in dataloader:
        images = images.to(device, non_blocking=True)
        batch_features = model(images)
        if isinstance(batch_features, (tuple, list)):
            batch_features = batch_features[-1]
        batch_features = F.normalize(batch_features.float(), p=2, dim=1)
        features.append(batch_features.cpu().numpy())
        pids.append(batch_pids.numpy())
        camids.append(batch_camids.numpy())

    return (
        np.concatenate(features, axis=0),
        np.concatenate(pids, axis=0),
        np.concatenate(camids, axis=0),
    )


def compute_distance_matrix(query_features, gallery_features, metric="cosine"):
    if metric == "cosine":
        sim = query_features @ gallery_features.T
        dist = 1.0 - sim
    elif metric == "euclidean":
        dist = (
            np.sum(query_features ** 2, axis=1, keepdims=True)
            + np.sum(gallery_features ** 2, axis=1, keepdims=True).T
            - 2 * query_features @ gallery_features.T
        )
        dist = np.clip(dist, 0, None)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    return dist


def eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    all_cmc = []
    all_ap = []
    num_valid = 0
    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = ~remove
        if not np.any(matches[q_idx][keep]):
            continue
        raw_cmc = matches[q_idx][keep]
        num_valid += 1
        cmc = raw_cmc.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        precision = tmp_cmc / (np.arange(len(tmp_cmc)) + 1.0)
        ap = (precision * raw_cmc.astype(bool)).sum() / num_rel if num_rel > 0 else 0.0
        all_ap.append(ap)

    if num_valid == 0:
        raise RuntimeError("No valid query found during VeRi evaluation")

    cmc = np.asarray(all_cmc, dtype=np.float32).mean(axis=0)
    mAP = float(np.mean(all_ap))
    return mAP, cmc


def to_metric_dict(mAP, cmc):
    ranks = list(cmc)
    return {
        "mAP": float(mAP),
        "R1": float(ranks[min(0, len(ranks) - 1)]),
        "R5": float(ranks[min(4, len(ranks) - 1)]),
        "R10": float(ranks[min(9, len(ranks) - 1)]),
    }


def metric_sort_key(metrics):
    return (metrics["mAP"], metrics["R1"], metrics["R5"], metrics["R10"])


def print_metrics(label, metrics):
    print(
        f"{label}: mAP={metrics['mAP'] * 100:.4f}%  "
        f"R1={metrics['R1'] * 100:.4f}%  "
        f"R5={metrics['R5'] * 100:.2f}%  "
        f"R10={metrics['R10'] * 100:.2f}%"
    )


def average_query_expansion(features, k, iterations=1):
    current = features.astype(np.float32, copy=True)
    if k <= 1:
        return current
    for _ in range(iterations):
        sim = current @ current.T
        topk = min(k, sim.shape[1])
        kth = max(topk - 1, 0)
        topk_idx = np.argpartition(-sim, kth=kth, axis=1)[:, :topk]
        expanded = np.zeros_like(current)
        for index in range(current.shape[0]):
            expanded[index] = current[topk_idx[index]].mean(axis=0)
        norms = np.linalg.norm(expanded, axis=1, keepdims=True) + 1e-12
        current = expanded / norms
    return current


@torch.no_grad()
def build_rerank_state(all_features, max_k1, device: str):
    features = torch.as_tensor(all_features, dtype=torch.float32, device=device)
    features = F.normalize(features, p=2, dim=1)
    similarity = torch.matmul(features, features.T)
    original_dist = (2.0 - 2.0 * similarity).clamp_min_(0).cpu().numpy().astype(np.float32)
    initial_rank = torch.topk(
        similarity,
        k=min(max_k1 + 1, similarity.shape[1]),
        dim=1,
        largest=True,
        sorted=True,
    ).indices.cpu().numpy().astype(np.int32)
    del features, similarity
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return original_dist, initial_rank


def compute_reranking_torch(original_dist, initial_rank, query_num, k1=20, k2=6, lambda_value=0.3):
    all_num = original_dist.shape[0]
    V = np.zeros((all_num, all_num), dtype=np.float16)
    half_k1 = int(np.round(k1 / 2.0))

    for index in range(all_num):
        forward = initial_rank[index, :k1 + 1]
        backward = initial_rank[forward, :k1 + 1]
        reciprocal = forward[np.any(backward == index, axis=1)]
        reciprocal_expansion = reciprocal.copy()

        for candidate in reciprocal:
            candidate_forward = initial_rank[candidate, :half_k1 + 1]
            candidate_backward = initial_rank[candidate_forward, :half_k1 + 1]
            candidate_reciprocal = candidate_forward[np.any(candidate_backward == candidate, axis=1)]
            if candidate_reciprocal.size == 0:
                continue
            overlap = np.intersect1d(candidate_reciprocal, reciprocal)
            if overlap.size > (2.0 / 3.0) * candidate_reciprocal.size:
                reciprocal_expansion = np.concatenate((reciprocal_expansion, candidate_reciprocal))

        reciprocal_expansion = np.unique(reciprocal_expansion)
        weights = np.exp(-original_dist[index, reciprocal_expansion]).astype(np.float32)
        V[index, reciprocal_expansion] = (weights / (weights.sum() + 1e-12)).astype(np.float16)

    if k2 > 1:
        V_qe = np.zeros_like(V, dtype=np.float16)
        for index in range(all_num):
            V_qe[index] = V[initial_rank[index, :k2]].mean(axis=0)
        V = V_qe

    inv_index = [np.flatnonzero(V[:, column]) for column in range(all_num)]
    jaccard_dist = np.zeros((query_num, all_num), dtype=np.float32)

    for index in range(query_num):
        temp_min = np.zeros(all_num, dtype=np.float32)
        non_zero = np.flatnonzero(V[index])
        for nz in non_zero:
            related = inv_index[nz]
            temp_min[related] += np.minimum(np.float32(V[index, nz]), V[related, nz].astype(np.float32))
        jaccard_dist[index] = 1.0 - temp_min / (2.0 - temp_min)

    final_dist = jaccard_dist * (1.0 - lambda_value) + original_dist[:query_num] * lambda_value
    return final_dist[:, query_num:]


def evaluate_rerank_sweep(all_features, qf_len, q_pids, g_pids, q_camids, g_camids, device: str):
    rerank_state = build_rerank_state(all_features, max_k1=max(RERANK_K1_VALUES), device=device)
    records = []
    for k1 in RERANK_K1_VALUES:
        for k2 in RERANK_K2_VALUES:
            for lambda_value in RERANK_LAMBDAS:
                distmat = compute_reranking_torch(
                    original_dist=rerank_state[0],
                    initial_rank=rerank_state[1],
                    query_num=qf_len,
                    k1=k1,
                    k2=k2,
                    lambda_value=lambda_value,
                )
                mAP, cmc = eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids)
                records.append({
                    "k1": int(k1),
                    "k2": int(k2),
                    "lambda": float(lambda_value),
                    "metrics": to_metric_dict(mAP, cmc),
                })
    return max(records, key=lambda record: metric_sort_key(record["metrics"])), records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the 13e CLIP-SENet v6 checkpoint on VeRi-776."
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to best.pth or clipsenet_v6_veri776_best.pth")
    parser.add_argument("--veri-root", type=Path, required=True, help="VeRi-776 root containing image_query and image_test")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda", help="Evaluation device")
    parser.add_argument("--batch-size", type=int, default=64, help="Eval batch size")
    parser.add_argument("--img-size", type=int, nargs=2, metavar=("H", "W"), default=list(IMAGE_SIZE), help="Input image size")
    parser.add_argument("--output-json", type=Path, required=True, help="Path to write metric JSON")
    rerank_group = parser.add_mutually_exclusive_group()
    rerank_group.add_argument("--rerank", dest="rerank", action="store_true", help="Enable notebook k-reciprocal rerank sweep")
    rerank_group.add_argument("--no-rerank", dest="rerank", action="store_false", help="Disable rerank sweep")
    parser.set_defaults(rerank=False)
    parser.add_argument("--aqe-k", type=int, default=1, help="AQE k applied before rerank; k<=1 preserves notebook rerank behavior")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is False")

    image_size = (int(args.img_size[0]), int(args.img_size[1]))
    checkpoint_path = args.checkpoint.expanduser().resolve()
    veri_root = args.veri_root.expanduser().resolve()
    output_json = args.output_json.expanduser().resolve()

    state_dict, checkpoint_kind, inferred_num_classes = load_checkpoint(checkpoint_path)
    print("DEVICE:", args.device)
    print("CHECKPOINT_PATH:", checkpoint_path)
    print("CHECKPOINT_KIND:", checkpoint_kind)
    print("NUM_CLASSES:", inferred_num_classes)

    model = build_clip_senet(num_classes=inferred_num_classes).to(args.device)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys or unexpected_keys:
        print("Missing keys:", missing_keys)
        print("Unexpected keys:", unexpected_keys)
        raise RuntimeError("Checkpoint load was not strict; aborting eval setup")
    model.eval()

    query_loader, gallery_loader, query_items, gallery_items, query_ids, gallery_ids = build_veri_loaders(
        veri_root=veri_root,
        image_size=image_size,
        batch_size=args.batch_size,
    )
    print("VERI_ROOT:", veri_root)
    print(f"Query:   {len(query_items):,} images, {query_ids} IDs")
    print(f"Gallery: {len(gallery_items):,} images, {gallery_ids} IDs")

    qf, q_pids, q_camids = extract_features(model, query_loader, args.device)
    gf, g_pids, g_camids = extract_features(model, gallery_loader, args.device)
    all_features = np.concatenate([qf, gf], axis=0)

    print("qf:", qf.shape)
    print("gf:", gf.shape)

    base_distmat = compute_distance_matrix(qf, gf, metric="cosine")
    base_mAP, base_cmc = eval_market1501(base_distmat, q_pids, g_pids, q_camids, g_camids)
    output = to_metric_dict(base_mAP, base_cmc)
    print_metrics("Base cosine", output)

    if args.rerank:
        rerank_features = all_features
        if args.aqe_k > 1:
            rerank_features = average_query_expansion(all_features, k=args.aqe_k, iterations=1)
        best_rerank, _ = evaluate_rerank_sweep(
            all_features=rerank_features,
            qf_len=len(qf),
            q_pids=q_pids,
            g_pids=g_pids,
            q_camids=q_camids,
            g_camids=g_camids,
            device=args.device,
        )
        output["rerank_aqe"] = best_rerank["metrics"]
        print_metrics(
            f"Rerank+AQE aqe_k={args.aqe_k} k1={best_rerank['k1']} k2={best_rerank['k2']} lambda={best_rerank['lambda']:.1f}",
            output["rerank_aqe"],
        )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print(f"Wrote {output_json}")


if __name__ == "__main__":
    main()
