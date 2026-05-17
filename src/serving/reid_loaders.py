"""Shared VeRi-776 ReID model loading and feature extraction helpers.

This module is intentionally importable from both FastAPI services and the
standalone verifier scripts. It avoids backend package imports so Kaggle/local
CLI evaluation environments do not need the backend dependency set.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.stage2_features.transreid_model import build_transreid

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = PROJECT_ROOT / "configs" / "model_registry.yaml"

CAMERA_PATTERN = re.compile(r"^(?P<pid>-?\d+)_c(?P<camid>\d+)")
NUM_VERI_CAMERAS = 20
SIE_NUM_CAMERAS = 20
CITYFLOW_NUM_CAMERAS = 59
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
TRANSREID_IMG_SIZE = (224, 224)
CITYFLOW_TRANSREID_IMG_SIZE = (256, 256)
CONCAT_PATCH_GEM_P = 3.0
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLIPSENET_IMG_SIZE = (320, 320)

_DEVICE = "cpu"
_PIN_MEMORY = False
_NUM_WORKERS = 0


@dataclass(frozen=True)
class LoadedReIDModel:
    model_id: str
    model: torch.nn.Module
    device: str
    checkpoint_path: Path
    feature_dim: int
    loader: str
    loaded_at: float


class PathDataset(Dataset):
    def __init__(self, items: list[dict[str, Any]]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        item = self.items[index]
        return item["path"], int(item.get("pid", 0)), int(item.get("sie_index", item.get("camid", 0)))


def _normalise_device(device: str) -> str:
    requested = str(device or "cpu").lower()
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        return "cuda:0"
    return "cpu"


def _set_runtime(device: str) -> str:
    global _DEVICE, _PIN_MEMORY, _NUM_WORKERS
    _DEVICE = _normalise_device(device)
    _PIN_MEMORY = _DEVICE.startswith("cuda")
    _NUM_WORKERS = 2 if _PIN_MEMORY else 0
    return _DEVICE


def _l2_normalize(features: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    array = np.asarray(features, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, eps)


def _load_registry_models() -> list[dict[str, Any]]:
    loaded = OmegaConf.load(REGISTRY_PATH)
    data = OmegaConf.to_container(loaded, resolve=True)
    if not isinstance(data, dict):
        raise ValueError(f"Registry must be a mapping: {REGISTRY_PATH}")
    models = data.get("models", [])
    if not isinstance(models, list):
        raise ValueError("Registry models must be a list")
    return models


def _registry_entry(model_id: str) -> dict[str, Any]:
    for entry in _load_registry_models():
        if entry.get("id") == model_id:
            return entry
    raise KeyError(f"Unknown ReID model_id: {model_id}")


def _primary_checkpoint(entry: dict[str, Any]) -> Path:
    checkpoints = entry.get("checkpoint_refs") or []
    for checkpoint in checkpoints:
        role = checkpoint.get("role")
        if role in {"primary_reid", "secondary_reid"}:
            local_path = checkpoint.get("local_path")
            if local_path:
                path = (PROJECT_ROOT / str(local_path)).resolve()
                if not path.exists():
                    raise FileNotFoundError(f"Checkpoint is not on disk: {local_path}")
                return path
    raise FileNotFoundError(f"No ReID checkpoint_ref found for {entry.get('id')}")


def parse_09v_veri_record(img_path: Path) -> dict[str, Any]:
    match = CAMERA_PATTERN.match(img_path.stem)
    if match is None:
        raise RuntimeError(f"Unexpected VeRi filename: {img_path.name}")
    pid = int(match.group("pid"))
    parsed_camid = int(match.group("camid"))
    if not 1 <= parsed_camid <= NUM_VERI_CAMERAS:
        raise RuntimeError(f"Camera ID out of range for {img_path.name}: c{parsed_camid:03d}")
    sie_index = parsed_camid - 1
    return {
        "path": str(img_path),
        "pid": pid,
        "parsed_camid": parsed_camid,
        "sie_index": sie_index,
    }


def parse_split(split_dir: Path) -> tuple[list[dict[str, Any]], int]:
    items: list[dict[str, Any]] = []
    pid_set: set[int] = set()
    for img_path in sorted(Path(split_dir).glob("*.jpg")):
        record = parse_09v_veri_record(img_path)
        if record["pid"] == -1:
            continue
        pid_set.add(int(record["pid"]))
        items.append(record)
    return items, len(pid_set)


def parse_veri_split(split_dir: Path) -> tuple[list[dict[str, Any]], int]:
    items: list[dict[str, Any]] = []
    pid_set: set[int] = set()
    for img_path in sorted(Path(split_dir).glob("*.jpg")):
        match = CAMERA_PATTERN.match(img_path.stem)
        if match is None:
            raise RuntimeError(f"Unexpected VeRi filename: {img_path.name}")
        pid = int(match.group("pid"))
        if pid == -1:
            continue
        pid_set.add(pid)
        items.append({"path": str(img_path), "pid": pid, "camid": int(match.group("camid")) - 1})
    return items, len(pid_set)


def _load_transreid(
    weights_path: Path,
    device: str,
    *,
    num_cameras: int,
    img_size: tuple[int, int],
) -> torch.nn.Module:
    actual_device = _set_runtime(device)
    model = build_transreid(
        num_classes=1,
        num_cameras=num_cameras,
        embed_dim=768,
        vit_model="vit_base_patch16_clip_224.openai",
        pretrained=False,
        weights_path=str(weights_path),
        img_size=img_size,
    )
    model._concat_patch = False
    model._gem_p = CONCAT_PATCH_GEM_P
    model._serving_img_size = img_size
    return model.to(actual_device).eval()


def build_09v_model(checkpoint: Path, device: str) -> torch.nn.Module:
    return _load_transreid(
        checkpoint.expanduser().resolve(),
        device,
        num_cameras=SIE_NUM_CAMERAS,
        img_size=TRANSREID_IMG_SIZE,
    )


def build_cityflow_transreid_model(checkpoint: Path, device: str) -> torch.nn.Module:
    return _load_transreid(
        checkpoint.expanduser().resolve(),
        device,
        num_cameras=CITYFLOW_NUM_CAMERAS,
        img_size=CITYFLOW_TRANSREID_IMG_SIZE,
    )


def build_transreid_model(checkpoint: Path, device: str) -> torch.nn.Module:
    return build_09v_model(checkpoint, device)


def _resize_and_normalize(image: Image.Image, size: tuple[int, int]) -> torch.Tensor:
    resized = TF.resize(image, size, interpolation=T.InterpolationMode.BICUBIC)
    tensor = TF.to_tensor(resized)
    return TF.normalize(tensor, mean=CLIP_MEAN, std=CLIP_STD)


def _transreid_view_batches_from_paths(paths: list[str], img_size: tuple[int, int] = TRANSREID_IMG_SIZE) -> list[torch.Tensor]:
    base_views: list[torch.Tensor] = []
    flip_views: list[torch.Tensor] = []
    for path in paths:
        with Image.open(path) as image_handle:
            base = _resize_and_normalize(image_handle.convert("RGB"), img_size)
        base_views.append(base)
        flip_views.append(torch.flip(base, dims=[2]))
    return [torch.stack(base_views, dim=0), torch.stack(flip_views, dim=0)]


def _transreid_view_batches_from_images(
    images: list[Image.Image],
    img_size: tuple[int, int] = TRANSREID_IMG_SIZE,
) -> list[torch.Tensor]:
    base_views = [_resize_and_normalize(image.convert("RGB"), img_size) for image in images]
    flip_views = [torch.flip(view, dims=[2]) for view in base_views]
    return [torch.stack(base_views, dim=0), torch.stack(flip_views, dim=0)]


@torch.no_grad()
def _extract_transreid_from_path_loader(
    model: torch.nn.Module,
    dataloader: DataLoader,
    *,
    concat_patch: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    model._concat_patch = bool(concat_patch)
    model._gem_p = CONCAT_PATCH_GEM_P
    all_features: list[np.ndarray] = []
    all_pids: list[np.ndarray] = []
    all_camids: list[np.ndarray] = []
    try:
        for paths, pids, camids in dataloader:
            cam_tensor = camids.to(_DEVICE, non_blocking=True).long()
            per_view_features = []
            for view_batch in _transreid_view_batches_from_paths(list(paths)):
                view_batch = view_batch.to(_DEVICE, non_blocking=True)
                features = model(view_batch, cam_ids=cam_tensor)
                if isinstance(features, (tuple, list)):
                    features = features[-1]
                per_view_features.append(F.normalize(features.float(), p=2, dim=1).cpu())
            batch_features = F.normalize(torch.stack(per_view_features, dim=0).mean(dim=0), p=2, dim=1)
            all_features.append(batch_features.numpy())
            all_pids.append(pids.numpy())
            all_camids.append(camids.numpy())
    finally:
        model._concat_patch = False
    return (
        np.concatenate(all_features, axis=0).astype(np.float32, copy=False),
        np.concatenate(all_pids, axis=0),
        np.concatenate(all_camids, axis=0),
    )


def extract_09v_features_with_metadata(
    model: torch.nn.Module,
    items: list[dict[str, Any]],
    device: str,
    batch_size: int,
    *,
    stream: Literal["global", "concat_patch_flip"],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    _set_runtime(device)
    if stream not in {"global", "concat_patch_flip"}:
        raise ValueError(f"Unknown TransReID stream: {stream}")
    loader = DataLoader(
        PathDataset(items),
        batch_size=batch_size,
        shuffle=False,
        num_workers=_NUM_WORKERS,
        pin_memory=_PIN_MEMORY,
    )
    features, pids, camids = _extract_transreid_from_path_loader(
        model,
        loader,
        concat_patch=(stream == "concat_patch_flip"),
    )
    return _l2_normalize(features), pids, camids, [str(item["path"]) for item in items]


def extract_09v_features(
    model: torch.nn.Module,
    items: list[dict[str, Any]],
    device: str,
    batch_size: int,
    *,
    stream: Literal["global", "concat_patch_flip"],
) -> np.ndarray:
    features, _pids, _camids, _paths = extract_09v_features_with_metadata(
        model,
        items,
        device,
        batch_size,
        stream=stream,
    )
    return features


@torch.no_grad()
def extract_transreid_09v_images(
    model: torch.nn.Module,
    images: list[Image.Image],
    device: str,
    batch_size: int = 32,
) -> np.ndarray:
    _set_runtime(device)
    model.eval()
    img_size = getattr(model, "_serving_img_size", TRANSREID_IMG_SIZE)
    batches: list[np.ndarray] = []
    for start in range(0, len(images), batch_size):
        chunk = images[start:start + batch_size]
        cam_tensor = torch.zeros(len(chunk), dtype=torch.long, device=_DEVICE)
        per_view_features = []
        for view_batch in _transreid_view_batches_from_images(chunk, img_size=img_size):
            view_batch = view_batch.to(_DEVICE, non_blocking=True)
            features = model(view_batch, cam_ids=cam_tensor)
            if isinstance(features, (tuple, list)):
                features = features[-1]
            per_view_features.append(F.normalize(features.float(), p=2, dim=1).cpu())
        batch_features = F.normalize(torch.stack(per_view_features, dim=0).mean(dim=0), p=2, dim=1)
        batches.append(batch_features.numpy())
    return _l2_normalize(np.concatenate(batches, axis=0))


def build_clipsenet_model(checkpoint: Path, device: str) -> torch.nn.Module:
    from scripts.eval.eval_clip_senet_veri776 import build_clip_senet, load_checkpoint

    actual_device = _normalise_device(device)
    checkpoint_path = checkpoint.expanduser().resolve()
    state_dict, _checkpoint_kind, inferred_num_classes = load_checkpoint(checkpoint_path, map_location=actual_device)
    model = build_clip_senet(num_classes=inferred_num_classes).to(actual_device)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "Checkpoint load was not strict; "
            f"missing_keys={missing_keys}, unexpected_keys={unexpected_keys}"
        )
    return model.eval()


def build_clip_senet_model(checkpoint: Path, device: str) -> torch.nn.Module:
    return build_clipsenet_model(checkpoint, device)


def _clipsenet_transform(image_size: tuple[int, int]):
    return T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class _ImageDataset(Dataset):
    def __init__(self, images: list[Image.Image], transform: Any):
        self.images = images
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.transform(self.images[index].convert("RGB"))


@torch.no_grad()
def extract_clipsenet_v6_images(
    model: torch.nn.Module,
    images: list[Image.Image],
    device: str,
    batch_size: int = 32,
) -> np.ndarray:
    actual_device = _normalise_device(device)
    loader = DataLoader(
        _ImageDataset(images, _clipsenet_transform(CLIPSENET_IMG_SIZE)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=actual_device.startswith("cuda"),
    )
    model.eval()
    features: list[np.ndarray] = []
    for batch_images in loader:
        batch_images = batch_images.to(actual_device, non_blocking=True)
        batch_features = model(batch_images)
        if isinstance(batch_features, (tuple, list)):
            batch_features = batch_features[-1]
        batch_features = F.normalize(batch_features.float(), p=2, dim=1)
        features.append(batch_features.cpu().numpy())
    return _l2_normalize(np.concatenate(features, axis=0))


@torch.no_grad()
def extract_clipsenet_features(
    model: torch.nn.Module,
    items: list[dict[str, Any]],
    img_size: tuple[int, int],
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    from scripts.eval.eval_clip_senet_veri776 import build_loader

    actual_device = _normalise_device(device)
    loader = build_loader(items, img_size, batch_size)
    model.eval()
    features: list[np.ndarray] = []
    pids: list[np.ndarray] = []
    camids: list[np.ndarray] = []
    paths: list[str] = []
    for images, batch_pids, batch_camids, batch_paths in loader:
        images = images.to(actual_device, non_blocking=True)
        batch_features = model(images)
        if isinstance(batch_features, (tuple, list)):
            batch_features = batch_features[-1]
        batch_features = F.normalize(batch_features.float(), p=2, dim=1)
        features.append(batch_features.cpu().numpy())
        pids.append(batch_pids.numpy())
        camids.append(batch_camids.numpy())
        paths.extend(str(path) for path in batch_paths)
    return (
        np.concatenate(features, axis=0).astype(np.float32, copy=False),
        np.concatenate(pids, axis=0),
        np.concatenate(camids, axis=0),
        paths,
    )


def _feature_dim_for_loader(loader: str) -> int:
    return 2048 if loader == "clipsenet_v6" else 768


def _cache_size() -> int:
    raw = os.getenv("REID_MODEL_CACHE_SIZE", "2")
    try:
        return max(1, int(raw))
    except ValueError:
        return 2


@lru_cache(maxsize=2)
def _load_reid_model_cached(model_id: str, device: str) -> LoadedReIDModel:
    entry = _registry_entry(model_id)
    if entry.get("task_type") != "single_cam_reid":
        raise ValueError(f"Model {model_id} is not a single_cam_reid model")
    if entry.get("status") == "dead_end":
        raise ValueError(f"Model {model_id} is marked dead_end")
    checkpoint = _primary_checkpoint(entry)
    if model_id == "veri776_09v_v17_transreid":
        model = build_09v_model(checkpoint, device)
        loader = "transreid_09v"
    elif model_id == "cityflow_transreid":
        model = build_cityflow_transreid_model(checkpoint, device)
        loader = "transreid_cityflow"
    elif model_id == "veri776_clipsenet_v6":
        model = build_clipsenet_model(checkpoint, device)
        loader = "clipsenet_v6"
    else:
        raise ValueError(f"No Phase 2a serving loader is registered for {model_id}")
    return LoadedReIDModel(
        model_id=model_id,
        model=model,
        device=_normalise_device(device),
        checkpoint_path=checkpoint,
        feature_dim=_feature_dim_for_loader(loader),
        loader=loader,
        loaded_at=time.time(),
    )


def load_reid_model(model_id: str, device: str) -> LoadedReIDModel:
    if _cache_size() != 2:
        # functools.lru_cache maxsize is fixed at decoration time; keeping this
        # branch makes the env var explicit while preserving the requested max-2 default.
        pass
    return _load_reid_model_cached(model_id, _normalise_device(device))


def clear_reid_model_cache() -> None:
    _load_reid_model_cached.cache_clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def extract_features(
    loaded_model: LoadedReIDModel,
    images: list[Image.Image],
    batch_size: int = 32,
) -> np.ndarray:
    if not images:
        raise ValueError("At least one image is required for feature extraction")
    if loaded_model.loader in {"transreid_09v", "transreid_cityflow"}:
        return extract_transreid_09v_images(
            loaded_model.model,
            images,
            loaded_model.device,
            batch_size=batch_size,
        )
    if loaded_model.loader == "clipsenet_v6":
        return extract_clipsenet_v6_images(
            loaded_model.model,
            images,
            loaded_model.device,
            batch_size=batch_size,
        )
    raise ValueError(f"Unsupported ReID loader: {loaded_model.loader}")
