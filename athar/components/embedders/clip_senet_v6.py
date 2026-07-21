"""CLIP-SENet v6 — canonical architecture + checkpoint/loader plumbing.

MOVED VERBATIM (2026-07-21 serving inversion) from
scripts/eval/eval_clip_senet_veri776.py — the copy that produced the
registered VeRi-776 numbers (CLIP-SENet solo 91.36 mAP; fusion 93.3).
The eval script now imports from here; athar.serving no longer imports
from scripts (the loader<->script cycle is gone).

The OTHER copy, athar/components/embedders/clip_senet_model.py, is the
v1 pipeline-side variant (different loader order, HF-hub-first air-gap
bug) kept only until the stage2 glue port retires it — do not extend it.

Original provenance: notebooks/kaggle/13e_clip_senet_eval.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

logger = logging.getLogger(__name__)

DEFAULT_NUM_CLASSES = 575
IMAGE_SIZE = (320, 320)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
NUM_WORKERS = 2


@contextmanager
def _cpu_safe_hub_deserialization():
    original_hub_loader = torch.hub.load_state_dict_from_url
    original_model_zoo_loader = None
    try:
        import torch.utils.model_zoo as model_zoo
        original_model_zoo_loader = model_zoo.load_url
    except Exception:  # noqa: BLE001 - model_zoo is best-effort compatibility
        model_zoo = None

    def load_state_dict_from_url_cpu(*args, **kwargs):
        kwargs.setdefault("map_location", "cpu")
        return original_hub_loader(*args, **kwargs)

    def load_url_cpu(*args, **kwargs):
        kwargs.setdefault("map_location", "cpu")
        return original_model_zoo_loader(*args, **kwargs)

    torch.hub.load_state_dict_from_url = load_state_dict_from_url_cpu
    if model_zoo is not None and original_model_zoo_loader is not None:
        model_zoo.load_url = load_url_cpu
    try:
        yield
    finally:
        torch.hub.load_state_dict_from_url = original_hub_loader
        if model_zoo is not None and original_model_zoo_loader is not None:
            model_zoo.load_url = original_model_zoo_loader




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
            with _cpu_safe_hub_deserialization():
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
            with _cpu_safe_hub_deserialization():
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


def torch_load(path: Path, map_location: str | torch.device = "cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_checkpoint(path: Path, map_location: str | torch.device = "cpu"):
    payload = torch_load(path, map_location=map_location)
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


