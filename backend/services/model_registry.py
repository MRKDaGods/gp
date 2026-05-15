"""YAML-backed model registry loader and validators."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from jsonschema import Draft202012Validator
from omegaconf import OmegaConf

from backend.models.registry import ModelEntry, Registry

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY_PATH = PROJECT_ROOT / "configs" / "model_registry.yaml"
DEFAULT_SCHEMA_PATH = PROJECT_ROOT / "configs" / "model_registry.schema.json"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


def _to_plain_data(path: Path) -> Dict[str, Any]:
    loaded = OmegaConf.load(path)
    data = OmegaConf.to_container(loaded, resolve=True)
    if not isinstance(data, dict):
        raise ValueError(f"Registry must be a mapping: {path}")
    return data


def _load_schema(schema_path: Path = DEFAULT_SCHEMA_PATH) -> Dict[str, Any]:
    return json.loads(schema_path.read_text(encoding="utf-8"))


def _path_exists(repo_relative_path: str) -> bool:
    return (PROJECT_ROOT / repo_relative_path).exists()


def _validate_metric_sources(data: Dict[str, Any]) -> None:
    for entry in data.get("models", []):
        model_id = entry.get("id", "<unknown>")
        for metric in entry.get("metrics", []):
            source = metric.get("source", {})
            source_path = source.get("path")
            if not source_path or not _path_exists(str(source_path)):
                raise FileNotFoundError(
                    f"Metric source path for {model_id}.{metric.get('name')} does not exist: "
                    f"{source_path}"
                )


def _merged_pipeline_config(pipeline_config: str) -> Dict[str, Any]:
    default_cfg = OmegaConf.load(DEFAULT_CONFIG_PATH)
    cfg_path = PROJECT_ROOT / pipeline_config
    if not cfg_path.exists():
        raise FileNotFoundError(f"pipeline_config does not exist: {pipeline_config}")
    dataset_cfg = OmegaConf.load(cfg_path)
    merged = OmegaConf.merge(default_cfg, dataset_cfg)
    data = OmegaConf.to_container(merged, resolve=False)
    if not isinstance(data, dict):
        raise ValueError(f"pipeline_config must resolve to a mapping: {pipeline_config}")
    return data


def _has_dotted_key(config: Dict[str, Any], dotted_key: str) -> bool:
    cursor: Any = config
    for part in dotted_key.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return False
        cursor = cursor[part]
    return True


def _validate_model_overrides(data: Dict[str, Any]) -> None:
    config_cache: Dict[str, Dict[str, Any]] = {}
    for entry in data.get("models", []):
        pipeline_config = entry.get("pipeline_config")
        overrides = entry.get("model_overrides") or []
        if not overrides:
            continue
        if not pipeline_config:
            raise ValueError(f"{entry.get('id')} defines model_overrides without pipeline_config")
        if pipeline_config not in config_cache:
            config_cache[pipeline_config] = _merged_pipeline_config(str(pipeline_config))
        config = config_cache[pipeline_config]
        for override in overrides:
            dotted_key = str(override).split("=", 1)[0]
            if not _has_dotted_key(config, dotted_key):
                raise ValueError(
                    f"Unknown model override path for {entry.get('id')}: {dotted_key}"
                )


def _reconcile_checkpoint_presence(data: Dict[str, Any]) -> Dict[str, Any]:
    for entry in data.get("models", []):
        missing: List[str] = []
        for checkpoint in entry.get("checkpoint_refs", []):
            local_path = str(checkpoint.get("local_path"))
            on_disk = _path_exists(local_path)
            checkpoint["on_disk"] = on_disk
            if not on_disk:
                missing.append(local_path)
        entry["missing_checkpoints"] = missing
    return data


def validate_registry_data(data: Dict[str, Any], schema_path: Path = DEFAULT_SCHEMA_PATH) -> None:
    schema = _load_schema(schema_path)
    Draft202012Validator(schema).validate(data)
    _validate_metric_sources(data)
    _validate_model_overrides(data)


def load_registry(registry_path: Path = DEFAULT_REGISTRY_PATH) -> Registry:
    data = _to_plain_data(registry_path)
    validate_registry_data(data)
    data = _reconcile_checkpoint_presence(data)
    return Registry.model_validate(data)


@lru_cache(maxsize=1)
def get_registry() -> Registry:
    return load_registry(DEFAULT_REGISTRY_PATH)


def get_model(model_id: str) -> Optional[ModelEntry]:
    for model in get_registry().models:
        if model.id == model_id:
            return model
    return None


def list_models(
    *,
    task_type: Optional[str] = None,
    status: Optional[str] = None,
    include_dead_ends: bool = False,
) -> List[ModelEntry]:
    models = get_registry().models
    filtered: List[ModelEntry] = []
    for model in models:
        if model.status == "dead_end" and not include_dead_ends:
            continue
        if task_type is not None and model.task_type != task_type:
            continue
        if status is not None and model.status != status:
            continue
        filtered.append(model)
    return filtered
