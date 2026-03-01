"""Tests for core configuration loader."""

from pathlib import Path

import pytest
from omegaconf import DictConfig

from src.core.config import config_to_dict, load_config, save_config


def test_load_config(tmp_path):
    """Test loading a YAML config file."""
    config_content = """
project:
  name: test
  output_dir: /tmp/test
stage1:
  detector:
    model: yolo26m.pt
    confidence_threshold: 0.25
"""
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(config_content)

    cfg = load_config(config_path)
    assert isinstance(cfg, DictConfig)
    assert cfg.project.name == "test"
    assert cfg.stage1.detector.confidence_threshold == 0.25


def test_load_config_with_overrides(tmp_path):
    """Test config overrides via dotlist."""
    config_content = """
project:
  name: test
stage1:
  detector:
    model: yolo26m.pt
"""
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(config_content)

    cfg = load_config(config_path, overrides=["stage1.detector.model=yolo26s.pt"])
    assert cfg.stage1.detector.model == "yolo26s.pt"


def test_save_and_load_config(tmp_path):
    """Test config save/load roundtrip."""
    config_content = """
project:
  name: roundtrip_test
"""
    config_path = tmp_path / "original.yaml"
    config_path.write_text(config_content)

    cfg = load_config(config_path)
    save_path = tmp_path / "saved.yaml"
    save_config(cfg, save_path)

    cfg2 = load_config(save_path)
    assert cfg2.project.name == "roundtrip_test"


def test_config_to_dict(tmp_path):
    """Test converting config to plain dict."""
    config_content = """
project:
  name: test
  values: [1, 2, 3]
"""
    config_path = tmp_path / "test.yaml"
    config_path.write_text(config_content)

    cfg = load_config(config_path)
    d = config_to_dict(cfg)
    assert isinstance(d, dict)
    assert d["project"]["name"] == "test"
    assert d["project"]["values"] == [1, 2, 3]
