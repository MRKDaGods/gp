"""Configuration loader using OmegaConf."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

from omegaconf import DictConfig, OmegaConf


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
