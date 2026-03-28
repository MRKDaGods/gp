"""Configuration loader using OmegaConf."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

from omegaconf import DictConfig, OmegaConf


def is_torch_cuda_available() -> bool:
    """True if PyTorch is installed and can use at least one CUDA device."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def apply_cpu_when_no_cuda(cfg: DictConfig) -> bool:
    """If CUDA is not available, set stage1/stage2 torch devices to CPU and disable half.

    Mutates cfg in place. Call after load_config (and after CLI overrides are merged).

    Returns:
        True if the config was modified.
    """
    if is_torch_cuda_available():
        return False

    if "stage1" in cfg:
        s1 = cfg.stage1
        if "detector" in s1:
            s1.detector.device = "cpu"
            if "half" in s1.detector:
                s1.detector.half = False
        if "tracker" in s1:
            s1.tracker.device = "cpu"
            if "half" in s1.tracker:
                s1.tracker.half = False

    if "stage2" in cfg and "reid" in cfg.stage2:
        cfg.stage2.reid.device = "cpu"
        if "half" in cfg.stage2.reid:
            cfg.stage2.reid.half = False

    return True


def load_config(
    config_path: str | Path,
    overrides: Optional[Sequence[str]] = None,
    dataset_config: Optional[str | Path] = None,
) -> DictConfig:
    """Load YAML configuration with optional overrides and dataset config merge.

    Args:
        config_path: Path to the main YAML config file.
        overrides: List of dotlist overrides, e.g. ["stage1.detector.model=yolo26s.pt"].
        dataset_config: Optional path to a dataset-specific YAML to merge in.

    Returns:
        Merged OmegaConf DictConfig.
    """
    cfg = OmegaConf.load(config_path)

    if dataset_config is not None:
        ds_cfg = OmegaConf.load(dataset_config)
        cfg = OmegaConf.merge(cfg, ds_cfg)

    if overrides:
        override_cfg = OmegaConf.from_dotlist(list(overrides))
        cfg = OmegaConf.merge(cfg, override_cfg)

    OmegaConf.resolve(cfg)
    return cfg


def save_config(cfg: DictConfig, path: str | Path) -> None:
    """Save a config to YAML file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, str(path))


def config_to_dict(cfg: DictConfig) -> dict[str, Any]:
    """Convert OmegaConf config to a plain Python dict."""
    return OmegaConf.to_container(cfg, resolve=True)
