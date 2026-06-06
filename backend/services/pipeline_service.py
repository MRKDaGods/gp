"""Pipeline subprocess orchestration, run ID allocation, and background tasks."""
import asyncio
import json
import shutil
import subprocess
import sys
import threading
import traceback as _traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.config import (
    _CAMERA_LINE_RE,
    _CAMERAS_TOTAL_RE,
    _FRAME_LINE_RE,
    _HAS_CV2,
    _PIPELINE_PYTHON,
    _PROJECT_ROOT,
    _STAGE_LINE_RE,
    _STAGE_NAMES,
    DATASET_DIR,
    OUTPUT_DIR,
    PRECOMPUTE_RUN_ID,
    VIDEO_EXTENSIONS,
    list_run_dirs,
    resolve_run_dir,
)
from backend.models.requests import FusionConfig
from backend.models.registry import CheckpointRef, ModelArchitecture, ModelEntry
from backend.services.model_registry import _merged_pipeline_config, get_model
from backend.services.tracklet_service import _persist_probe_link
from backend.services.video_service import (
    _detect_camera_for_video,
    _extract_camera_id,
    _normalize_camera_id,
    _safe_reid_batch_size,
)
from backend.state import app_state


DATASET_CONFIG_BY_NAME = {
    "cityflowv2": "configs/datasets/cityflowv2.yaml",
    "wildtrack": "configs/datasets/wildtrack.yaml",
}

DATASET_TASK_BY_NAME = {
    "cityflowv2": "mtmc_vehicle",
    "wildtrack": "mtmc_person",
    "veri776": "single_cam_reid",
}


class RunCancelled(RuntimeError):
    """Raised when a pipeline run is cancelled by the user mid-execution."""


def _is_run_cancelled(run_id: str) -> bool:
    return app_state.active_runs.get(run_id, {}).get("status") == "cancelled"


def _finalize_run_failure(run_id: str, exc: BaseException, tb: str, label: str) -> None:
    """Mark a run as cancelled (user-initiated) or errored. Keeps a cancel from
    being mislabelled as an error when the subprocess was killed on purpose."""
    run = app_state.active_runs.get(run_id)
    if run is None:
        return
    if isinstance(exc, RunCancelled) or run.get("status") == "cancelled":
        run["status"] = "cancelled"
        run["message"] = "Run cancelled by user"
        run.pop("error", None)
        run.pop("errorDetail", None)
        return
    err_type = type(exc).__name__
    err_msg = str(exc) or f"({err_type} with no message)"
    full_error = f"{err_type}: {err_msg}"
    run["status"] = "error"
    run["error"] = full_error
    run["errorDetail"] = tb[-3000:]
    run["message"] = f"{label} - {full_error[:300]}"


@dataclass(frozen=True)
class PipelineModelResolution:
    model_id: Optional[str]
    resolved_config: str
    applied_overrides: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    dataset: Optional[str] = None
    fusion_resolved: Optional[Dict[str, Any]] = None


class PipelineModelValidationError(ValueError):
    def __init__(self, message: str, status_code: int = 422) -> None:
        super().__init__(message)
        self.status_code = status_code


def resolve_pipeline_config_path(dataset: Optional[str] = None) -> str:
    """Resolve backend dataset selector to the run_pipeline --config path."""
    selector = (dataset or "").strip().lower()
    if not selector or selector in {"default", "demo"}:
        return "configs/default.yaml"
    if selector not in DATASET_CONFIG_BY_NAME:
        allowed = ", ".join(sorted(DATASET_CONFIG_BY_NAME))
        raise ValueError(f"Unsupported dataset '{dataset}'. Expected one of: {allowed}")
    return DATASET_CONFIG_BY_NAME[selector]


def _normalise_dataset(dataset: Optional[str]) -> Optional[str]:
    value = (dataset or "").strip().lower()
    return value or None


def _lookup_registry_model(model_id: str):
    model = get_model(model_id)
    if model is not None:
        return model
    alternate_id = model_id.replace("-", "_")
    if alternate_id != model_id:
        return get_model(alternate_id)
    return None


def _format_override_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, list):
        return "[" + ",".join(str(item) for item in value) + "]"
    return str(value)


def _fusion_checkpoint_path(model: ModelEntry) -> str:
    for checkpoint in model.checkpoint_refs:
        if checkpoint.role == "primary_reid":
            return checkpoint.local_path
    for checkpoint in model.checkpoint_refs:
        if str(checkpoint.role).endswith("_reid"):
            return checkpoint.local_path

    raise PipelineModelValidationError(
        f"Model '{model.id}' does not define a ReID checkpoint path for fusion. "
        "Fusion requires a ReID checkpoint in model_registry.yaml."
    )


def _get_arch_metadata(model: ModelEntry) -> ModelArchitecture:
    if model.architecture is None:
        raise PipelineModelValidationError(
            f"Model '{model.id}' is missing the 'architecture' block in model_registry.yaml. "
            "Fusion requires arch metadata. See configs/model_registry.yaml for the schema."
        )
    return model.architecture


def _resolve_fusion_model(model_id: str) -> ModelEntry:
    model = _lookup_registry_model(model_id)
    if model is None:
        raise ValueError(f"Unknown model_id in fusion.models: '{model_id}'")
    return model


def _append_stage2_fusion_overrides(
    overrides: List[str],
    slot: str,
    model: ModelEntry,
    architecture: ModelArchitecture,
    *,
    include_slot_flags: bool = True,
) -> Dict[str, Any]:
    checkpoint_path = _fusion_checkpoint_path(model)
    slot_config: Dict[str, Any] = {
        "model_name": architecture.arch,
        "weights_path": checkpoint_path,
        "embedding_dim": architecture.embedding_dim,
        "input_size": architecture.input_size,
        "clip_normalization": architecture.clip_normalization,
    }
    if include_slot_flags:
        slot_config = {
            "enabled": True,
            "save_separate": True,
            **slot_config,
        }
    if architecture.arch == "transreid" and architecture.vit_model:
        slot_config["vit_model"] = architecture.vit_model

    for key, value in slot_config.items():
        overrides.append(f"stage2.reid.{slot}.{key}={_format_override_value(value)}")

    return {
        "model_id": model.id,
        "arch": architecture.arch,
        "checkpoint": checkpoint_path,
        "stage2_slot": slot,
        "stage2_config": slot_config,
    }


# Bundled-fusion stream wiring (single model_id whose model_overrides declare
# extra Stage-4 ensemble streams). Distinct from _resolve_fusion_pipeline_model,
# which wires a user-supplied FusionConfig of separate model_ids.

# Stage-4 ensemble slot -> (Stage-2 vehicle slot that produces it,
#                           Stage-2 output filename, checkpoint_ref role).
# NOTE: there is no `vehicle4` Stage-2 slot and no `embeddings_quaternary.npy`
# producer (src/stage2_features/pipeline.py only writes embeddings_secondary.npy
# from vehicle2 and embeddings_tertiary.npy from vehicle3). The quaternary
# ensemble stream therefore REUSES the vehicle2 Stage-2 slot (-> _secondary.npy);
# this is only sound when the secondary slot itself is unused (weight 0), which
# holds for the K7 model.
_BUNDLED_SLOT_TABLE: Dict[str, Dict[str, str]] = {
    "secondary": {
        "stage2_slot": "vehicle2",
        "stage2_file": "embeddings_secondary.npy",
        "checkpoint_role": "secondary_reid",
    },
    "tertiary": {
        "stage2_slot": "vehicle3",
        "stage2_file": "embeddings_tertiary.npy",
        "checkpoint_role": "tertiary_reid",
    },
    "quaternary": {
        "stage2_slot": "vehicle2",
        "stage2_file": "embeddings_secondary.npy",
        "checkpoint_role": "quaternary_reid",
    },
}

# Registry ArchitectureName -> the model_name string that
# src/stage2_features/reid_model.py actually knows how to build. Most names are
# passed through unchanged (transreid/dinov2 go through the TransReID+timm path);
# FastReID R50-IBN is registered as arch `resnet50_ibn` but the builder keys on
# `fastreid_sbs_r50_ibn` (see ReIDModel._build_model / _build_fastreid_sbs_r50_ibn).
_ARCH_TO_STAGE2_MODEL_NAME: Dict[str, str] = {
    "resnet50_ibn": "fastreid_sbs_r50_ibn",
}


def _append_stage2_checkpoint_overrides(
    overrides: List[str],
    stage2_slot: str,
    checkpoint: CheckpointRef,
    architecture: ModelArchitecture,
) -> Dict[str, Any]:
    """Emit the Stage-2 enable overrides for a bundled stream described by a
    per-checkpoint architecture block. Mirrors _append_stage2_fusion_overrides
    but sources the weights from the checkpoint_ref (not the primary checkpoint)
    and translates the registry arch name to the Stage-2 builder's model_name."""
    model_name = _ARCH_TO_STAGE2_MODEL_NAME.get(architecture.arch, architecture.arch)
    slot_config: Dict[str, Any] = {
        "enabled": True,
        "save_separate": True,
        "model_name": model_name,
        "weights_path": checkpoint.local_path,
        "embedding_dim": architecture.embedding_dim,
        "input_size": architecture.input_size,
        "clip_normalization": architecture.clip_normalization,
    }
    if architecture.arch == "transreid" and architecture.vit_model:
        slot_config["vit_model"] = architecture.vit_model
    for key, value in slot_config.items():
        overrides.append(f"stage2.reid.{stage2_slot}.{key}={_format_override_value(value)}")
    return {
        "arch": architecture.arch,
        "model_name": model_name,
        "checkpoint": checkpoint.local_path,
        "stage2_slot": stage2_slot,
        "stage2_config": slot_config,
    }


def _parse_ensemble_slot_weights(model_overrides: List[str]) -> Dict[str, float]:
    """Extract `stage4.association.<slot>_embeddings.weight=W` (slot in
    secondary/tertiary/quaternary) from a model's bare overrides."""
    weights: Dict[str, float] = {}
    for override in model_overrides:
        key, _, raw_value = str(override).partition("=")
        key = key.strip()
        for slot in _BUNDLED_SLOT_TABLE:
            if key == f"stage4.association.{slot}_embeddings.weight":
                try:
                    weights[slot] = float(raw_value.strip())
                except ValueError:
                    weights[slot] = 0.0
    return weights


def _stage2_slot_enabled_in_config(model: ModelEntry, stage2_slot: str) -> bool:
    """Whether the model's pipeline_config already enables a Stage-2 reid slot
    (e.g. cityflowv2.yaml enables vehicle3/DINOv2 itself)."""
    if not model.pipeline_config:
        return False
    try:
        merged = _merged_pipeline_config(model.pipeline_config)
    except Exception:
        return False
    slot_cfg = (
        merged.get("stage2", {}).get("reid", {}).get(stage2_slot, {})
        if isinstance(merged, dict)
        else {}
    )
    return bool(isinstance(slot_cfg, dict) and slot_cfg.get("enabled", False))


def _checkpoint_for_role(model: ModelEntry, role: str) -> Optional[CheckpointRef]:
    for checkpoint in model.checkpoint_refs:
        if checkpoint.role == role:
            return checkpoint
    return None


def _wire_bundled_fusion_streams(
    model: ModelEntry,
    overrides: List[str],
) -> Optional[Dict[str, Any]]:
    """Wire the Stage-4 ensemble streams that a single bundled model declares via
    `stage4.association.<slot>_embeddings.weight=W` overrides.

    For each slot with W>0 this appends a DYNAMIC, run-scoped Stage-4 embedding
    path (so the stream actually loads - the YAML default `run_latest` path never
    exists) and, when the stream needs its own Stage-2 extractor, the Stage-2
    enable overrides. A weight>0 stream that is neither wireable (no checkpoint
    architecture) nor already enabled in the pipeline_config raises rather than
    silently degrading to primary-only.

    Returns a fusion_resolved dict (so callers no longer report these as single),
    or None when the model declares no ensemble streams.
    """
    slot_weights = _parse_ensemble_slot_weights(list(model.model_overrides))
    active = {slot: w for slot, w in slot_weights.items() if w > 0.0}
    if not active:
        return None

    dyn_root = "${project.output_dir}/${project.run_name}/stage2"
    wired_streams: List[Dict[str, Any]] = []

    # Deterministic order: secondary, tertiary, quaternary.
    for slot in ("secondary", "tertiary", "quaternary"):
        if slot not in active:
            continue
        weight = active[slot]
        table = _BUNDLED_SLOT_TABLE[slot]
        stage2_slot = table["stage2_slot"]
        stage2_file = table["stage2_file"]
        role = table["checkpoint_role"]

        checkpoint = _checkpoint_for_role(model, role)
        stream: Dict[str, Any] = {
            "slot": slot,
            "weight": weight,
            "stage2_slot": stage2_slot,
            "stage2_file": stage2_file,
        }

        if checkpoint is not None and checkpoint.architecture is not None:
            # The stream brings its own extractor -> emit Stage-2 enable overrides.
            stage2_info = _append_stage2_checkpoint_overrides(
                overrides, stage2_slot, checkpoint, checkpoint.architecture
            )
            stream.update(stage2_info)
            stream["wired_via"] = "checkpoint_architecture"
        elif _stage2_slot_enabled_in_config(model, stage2_slot):
            # The pipeline_config already enables the producing Stage-2 slot
            # (e.g. cityflowv2.yaml vehicle3/DINOv2). Only the dynamic Stage-4
            # path needs fixing; no Stage-2 overrides required.
            stream["wired_via"] = "pipeline_config"
        else:
            raise PipelineModelValidationError(
                f"Model '{model.id}' sets {slot}_embeddings.weight={weight} but its "
                f"stream is not wired: no '{role}' checkpoint_ref with an architecture "
                f"block, and pipeline_config '{model.pipeline_config}' does not enable "
                f"stage2.reid.{stage2_slot}. Refusing to silently degrade to "
                "primary-only - add the checkpoint architecture or enable the Stage-2 "
                "slot in the pipeline_config."
            )

        # DYNAMIC, run-scoped Stage-4 embedding path + enable flag. The YAML
        # default (e.g. data/outputs/run_latest/...) points at a directory that
        # never exists, so without this the stream's .path fails to load and the
        # ensemble silently collapses to primary-only.
        overrides.append(
            f"stage4.association.{slot}_embeddings.path={dyn_root}/{stage2_file}"
        )
        overrides.append(f"stage4.association.{slot}_embeddings.enabled=true")
        wired_streams.append(stream)

    return {
        "mode": "bundled",
        "primary_model_id": model.id,
        "streams": wired_streams,
    }


def _resolve_fusion_pipeline_model(
    fusion: FusionConfig,
    dataset: Optional[str] = None,
) -> PipelineModelResolution:
    ordered = sorted(enumerate(fusion.models), key=lambda item: (-item[1].weight, item[0]))
    primary_entry = ordered[0][1]

    for entry in fusion.models:
        model = _resolve_fusion_model(entry.model_id)
        _get_arch_metadata(model)

    primary_model = _resolve_fusion_model(primary_entry.model_id)
    try:
        base_resolution = resolve_pipeline_model(
            model_id=primary_entry.model_id,
            dataset=dataset,
            fusion=None,
            _wire_bundled_streams=False,
        )
    except PipelineModelValidationError:
        base_resolution = PipelineModelResolution(
            model_id=primary_model.id,
            resolved_config=resolve_pipeline_config_path(dataset),
            applied_overrides=[],
            warnings=[],
            dataset=dataset,
        )
    overrides = list(base_resolution.applied_overrides)
    primary_architecture = _get_arch_metadata(primary_model)
    primary_stage2 = _append_stage2_fusion_overrides(
        overrides,
        "vehicle",
        primary_model,
        primary_architecture,
        include_slot_flags=False,
    )

    resolved_models: List[Dict[str, Any]] = [
        {
            "model_id": primary_entry.model_id,
            "weight": primary_entry.weight,
            "role": "primary",
            "primary": True,
            "arch": primary_stage2["arch"],
            "checkpoint": primary_stage2["checkpoint"],
            "stage2_slot": primary_stage2["stage2_slot"],
            "stage2_config": primary_stage2["stage2_config"],
        }
    ]

    secondary_path = "${project.output_dir}/${project.run_name}/stage2/embeddings_secondary.npy"
    tertiary_path = "${project.output_dir}/${project.run_name}/stage2/embeddings_tertiary.npy"
    slot_specs = [
        ("vehicle2", "secondary", "secondary_embeddings", secondary_path),
        ("vehicle3", "tertiary", "tertiary_embeddings", tertiary_path),
    ]

    for slot_index, (_, entry) in enumerate(ordered[1:]):
        stage2_slot, role, stage4_slot, embedding_path = slot_specs[slot_index]
        model = _resolve_fusion_model(entry.model_id)
        architecture = _get_arch_metadata(model)
        resolved_model = _append_stage2_fusion_overrides(overrides, stage2_slot, model, architecture)
        resolved_model.update({"weight": entry.weight, "role": role, "primary": False})
        resolved_models.append(resolved_model)

        overrides.extend(
            [
                f"stage4.association.{stage4_slot}.path={embedding_path}",
                f"stage4.association.{stage4_slot}.enabled=true",
                f"stage4.association.{stage4_slot}.weight={entry.weight}",
            ]
        )

    overrides.extend(
        [
            f"stage4.association.query_expansion.k={fusion.aqe_k}",
            f"stage4.association.reranking.enabled={_format_override_value(fusion.rerank)}",
            f"stage4.association.reranking.k1={fusion.k1}",
            f"stage4.association.reranking.k2={fusion.k2}",
            f"stage4.association.reranking.lambda_value={fusion.lambda_}",
        ]
    )

    fusion_resolved = {
        "models": resolved_models,
        "primary_model_id": primary_entry.model_id,
        "aqe_k": fusion.aqe_k,
        "k1": fusion.k1,
        "k2": fusion.k2,
        "lambda": fusion.lambda_,
        "rerank": fusion.rerank,
    }

    return PipelineModelResolution(
        model_id=base_resolution.model_id,
        resolved_config=base_resolution.resolved_config,
        applied_overrides=overrides,
        warnings=base_resolution.warnings,
        dataset=base_resolution.dataset,
        fusion_resolved=fusion_resolved,
    )


def resolve_pipeline_model(
    model_id: Optional[str] = None,
    dataset: Optional[str] = None,
    fusion: Optional[FusionConfig] = None,
    *,
    _wire_bundled_streams: bool = True,
) -> PipelineModelResolution:
    """Resolve an optional registry model selection into pipeline run settings.

    `_wire_bundled_streams` is an internal flag: the FusionConfig path reuses this
    function to build the primary's base resolution and re-wires the ensemble
    itself, so it disables the bundled-stream wiring to avoid double-wiring.
    """
    requested_dataset = _normalise_dataset(dataset)

    if fusion is not None:
        return _resolve_fusion_pipeline_model(fusion=fusion, dataset=requested_dataset)

    if not model_id:
        return PipelineModelResolution(
            model_id=None,
            resolved_config=resolve_pipeline_config_path(requested_dataset),
            applied_overrides=[],
            warnings=[],
            dataset=requested_dataset,
        )

    model_key = model_id.strip()
    model = _lookup_registry_model(model_key)
    if model is None:
        raise ValueError(f"Unknown model_id '{model_id}'")

    warnings: List[str] = []
    if requested_dataset and requested_dataset != model.dataset:
        warnings.append(
            f"model_id '{model.id}' overrides requested dataset '{requested_dataset}' "
            f"with registry dataset '{model.dataset}'"
        )

    effective_dataset = model.dataset
    if not model.runnable_locally:
        kernel_hint = model.notebook_or_kernel_ref or "no Kaggle kernel reference recorded"
        raise PipelineModelValidationError(
            f"Model '{model.id}' is not runnable through the local pipeline API. "
            f"Run or reproduce it on Kaggle via: {kernel_hint}"
        )

    expected_task = DATASET_TASK_BY_NAME.get(effective_dataset)
    if expected_task is not None and model.task_type != expected_task:
        raise PipelineModelValidationError(
            f"Model '{model.id}' has task_type '{model.task_type}', which is not compatible "
            f"with dataset '{effective_dataset}' ({expected_task})."
        )

    if not model.pipeline_config:
        raise PipelineModelValidationError(
            f"Model '{model.id}' does not define a pipeline_config for MTMC pipeline runs."
        )

    applied_overrides = list(model.model_overrides)
    # A registered model can BUNDLE a multi-stream fusion ensemble in its
    # model_overrides (declaring stage4.association.<slot>_embeddings.weight=W
    # without a usable path). Wire those streams now: add run-scoped Stage-4
    # embedding paths + any Stage-2 enable overrides, or fail loud if a weighted
    # stream is unwireable. Returns None for ordinary single-model selections.
    fusion_resolved = (
        _wire_bundled_fusion_streams(model, applied_overrides)
        if _wire_bundled_streams
        else None
    )

    return PipelineModelResolution(
        model_id=model.id,
        resolved_config=model.pipeline_config,
        applied_overrides=applied_overrides,
        warnings=warnings,
        dataset=effective_dataset,
        fusion_resolved=fusion_resolved,
    )


def _allocate_numeric_run_id() -> str:
    """Allocate the next numeric run id under outputs/ (1, 2, 3, ...)."""
    with app_state.run_id_lock:
        max_num = 0
        try:
            for child in OUTPUT_DIR.iterdir():
                if child.is_dir() and child.name.isdigit():
                    max_num = max(max_num, int(child.name))
        except Exception:
            pass

        next_num = max_num + 1
        while True:
            run_id = str(next_num)
            run_dir = OUTPUT_DIR / run_id
            try:
                run_dir.mkdir(parents=True, exist_ok=False)
                return run_id
            except FileExistsError:
                next_num += 1


def _resolve_run_id(requested_run_id: Optional[str]) -> str:
    """Resolve a run id: keep explicit id, otherwise allocate numeric id."""
    if requested_run_id is not None:
        txt = str(requested_run_id).strip()
        if txt:
            return txt
    return _allocate_numeric_run_id()


def _write_run_context(run_id: str, payload: Dict[str, Any]) -> None:
    """Persist lightweight run metadata to help auditing and dataset discovery.

    Merges into any existing context so per-stage runs don't drop fields written
    at run creation (videos, inputDir, etc.)."""
    try:
        run_dir = OUTPUT_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        ctx_path = run_dir / "run_context.json"
        existing: Dict[str, Any] = {}
        if ctx_path.exists():
            try:
                existing = json.loads(ctx_path.read_text(encoding="utf-8"))
            except Exception:
                existing = {}
        context = {
            **existing,
            "runId": run_id,
            "createdAt": existing.get("createdAt") or datetime.now().isoformat(),
            "updatedAt": datetime.now().isoformat(),
            **payload,
        }
        ctx_path.write_text(json.dumps(context, indent=2), encoding="utf-8")
    except Exception as exc:
        print(f"[WARN] Failed to write run_context.json for run {run_id}: {exc}", flush=True)


def read_run_context(run_dir: Path) -> Dict[str, Any]:
    """Read run_context.json for a run dir (empty dict if missing/unreadable)."""
    ctx_path = run_dir / "run_context.json"
    if ctx_path.exists():
        try:
            return json.loads(ctx_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def stages_present(run_dir: Path) -> Dict[str, bool]:
    """Which pipeline stages produced (non-empty) output for a run."""
    out: Dict[str, bool] = {}
    for i in range(7):
        d = run_dir / f"stage{i}"
        present = False
        if d.is_dir():
            try:
                present = any(d.iterdir())
            except Exception:
                present = False
        out[f"stage{i}"] = present
    return out


def _input_dir_from_config(run_dir: Path) -> Optional[str]:
    """Recover stage0.input_dir from a run's merged config.yaml (normalised to
    posix). Lets runs created before run_context stored inputDir be re-opened."""
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        return None
    try:
        import yaml
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        raw = (cfg.get("stage0") or {}).get("input_dir")
        return str(raw).replace("\\", "/") if raw else None
    except Exception:
        return None


def _cameras_from_disk(run_dir: Path) -> List[str]:
    """Recover camera ids from output artifacts (stage1 tracklets, else stage0
    camera folders) for runs whose run_context lacks them."""
    cams: List[str] = []
    s1 = run_dir / "stage1"
    if s1.is_dir():
        for f in sorted(s1.glob("tracklets_*.json")):
            cams.append(f.stem[len("tracklets_"):])
    if not cams:
        s0 = run_dir / "stage0"
        if s0.is_dir():
            try:
                cams = sorted(d.name for d in s0.iterdir() if d.is_dir())
            except Exception:
                cams = []
    return cams


def describe_run(run_dir: Path) -> Dict[str, Any]:
    """A summary record for a run on disk (for listing / loading in the UI)."""
    ctx = read_run_context(run_dir)
    present = stages_present(run_dir)
    traj_path = run_dir / "stage4" / "global_trajectories.json"
    trajectory_count: Optional[int] = None
    if traj_path.exists():
        try:
            data = json.loads(traj_path.read_text(encoding="utf-8"))
            trajectory_count = (
                len(data) if isinstance(data, list) else len(data.get("trajectories", []))
            )
        except Exception:
            trajectory_count = None
    videos = ctx.get("videos") or []
    cameras = (
        ctx.get("selectedCameras")
        or [v.get("cameraId") for v in videos if v.get("cameraId")]
        or _cameras_from_disk(run_dir)
    )
    # Recover the source folder from the run's merged config when run_context
    # predates the inputDir field - so old runs can still restore their footage.
    input_dir = ctx.get("inputDir") or _input_dir_from_config(run_dir)
    # Fall back to the directory's modified time when no context timestamps exist.
    try:
        dir_mtime = datetime.fromtimestamp(run_dir.stat().st_mtime).isoformat()
    except Exception:
        dir_mtime = None
    created_at = ctx.get("createdAt") or dir_mtime
    updated_at = ctx.get("updatedAt") or dir_mtime
    live = app_state.active_runs.get(run_dir.name, {})
    status = live.get("status")
    if status not in ("running", "queued", "error", "cancelled"):
        status = "ready" if any(present.values()) else "empty"
    # Per-stage status: a stage that wrote output is "done"; the in-flight stage
    # of a running/errored run is marked "running"/"error" even though it hasn't
    # written artifacts yet - so the UI shows "Detect: running" instead of
    # misreporting the run as ingestion-only while detection is still working.
    active_stage = live.get("currentStageNum")
    stage_status: Dict[str, str] = {}
    for i in range(7):
        key = f"stage{i}"
        if present.get(key):
            stage_status[key] = "done"
        elif status == "running" and active_stage == i:
            stage_status[key] = "running"
        elif status == "error" and active_stage == i:
            stage_status[key] = "error"
        else:
            stage_status[key] = "idle"
    try:
        size_bytes = sum(f.stat().st_size for f in run_dir.rglob("*") if f.is_file())
    except Exception:
        size_bytes = 0
    return {
        "runId": run_dir.name,
        "root": str(run_dir.parent).replace("\\", "/"),
        "name": ctx.get("datasetName") or ctx.get("name"),
        "source": ctx.get("source"),
        "inputDir": input_dir,
        "cameras": cameras,
        "smoke": bool(ctx.get("smoke", False)),
        "videos": videos,
        "createdAt": created_at,
        "updatedAt": updated_at,
        "stages": present,
        "stageStatus": stage_status,
        "activeStage": active_stage,
        "currentStageName": live.get("currentStageName"),
        "message": live.get("message"),
        "error": live.get("error"),
        "trajectoryCount": trajectory_count,
        "status": status,
        "progress": live.get("progress"),
        "sizeBytes": size_bytes,
    }


def _cleanup_empty_run_dirs() -> int:
    """Remove orphan numeric run dirs that are completely empty - leftovers from a
    run id that was allocated (the allocator pre-creates the dir) but never wrote
    config/context (e.g. a request that failed validation after allocation). They
    don't show in the runs list but would otherwise accumulate as phantom ids."""
    removed = 0
    try:
        for child in OUTPUT_DIR.iterdir():
            if not (child.is_dir() and child.name.isdigit()):
                continue
            try:
                if any(child.iterdir()):
                    continue  # has content - a real run
                child.rmdir()
                removed += 1
            except OSError:
                pass
    except Exception:
        pass
    return removed


def rehydrate_runs_from_disk() -> int:
    """Rebuild in-memory run state from disk on startup so runs survive a backend
    restart. Registers light video records + video->run mapping (no video probing)
    and a placeholder active_runs entry per run. Existing in-memory state wins."""
    _cleanup_empty_run_dirs()
    count = 0
    # Process in id order so the most recent run wins the video->run mapping.
    def _sort_key(d: Path):
        return (0, int(d.name)) if d.name.isdigit() else (1, d.name)

    for run_dir in sorted(list_run_dirs(), key=_sort_key):
        run_id = run_dir.name
        ctx = read_run_context(run_dir)
        present = stages_present(run_dir)
        if run_id not in app_state.active_runs:
            app_state.active_runs[run_id] = {
                "id": run_id,
                "runId": run_id,
                "status": "completed" if any(present.values()) else "idle",
                "progress": 100 if present.get("stage4") else 0,
                "message": "Loaded from disk",
                "datasetFolder": ctx.get("datasetName"),
                "inputDir": ctx.get("inputDir"),
                "selectedCameras": ctx.get("selectedCameras"),
                "source": ctx.get("source", "disk"),
                "runDir": str(run_dir),
                "rehydrated": True,
            }
        for v in ctx.get("videos") or []:
            vid_id = v.get("id")
            vpath = v.get("path")
            if not vid_id or not vpath:
                continue
            if vid_id not in app_state.uploaded_videos:
                # Light record (no probing): enough for streaming / detections.
                app_state.uploaded_videos[vid_id] = {
                    "id": vid_id,
                    "name": v.get("name") or vid_id,
                    "filename": Path(vpath).name,
                    "path": vpath,
                }
            app_state.video_to_latest_run[vid_id] = run_id
        # Legacy fallback for runs without a videos[] block.
        link = run_dir / "probe_video_id.txt"
        if link.exists():
            try:
                vid = link.read_text(encoding="utf-8").strip()
                if vid:
                    app_state.video_to_latest_run.setdefault(vid, run_id)
            except Exception:
                pass
        count += 1
    return count


def delete_run(run_id: str) -> bool:
    """Delete a run's directory from disk and purge its in-memory state. Returns
    True if a directory was removed."""
    run_dir = resolve_run_dir(run_id)
    removed = False
    if run_dir is not None and run_dir.exists():
        # Terminate a still-running subprocess for this run before deleting.
        proc = app_state.run_processes.get(run_id)
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass
        shutil.rmtree(run_dir, ignore_errors=True)
        removed = not run_dir.exists()
    app_state.active_runs.pop(run_id, None)
    app_state.run_processes.pop(run_id, None)
    for vid_id in [v for v, r in app_state.video_to_latest_run.items() if r == run_id]:
        app_state.video_to_latest_run.pop(vid_id, None)
    return removed


def _prepare_input_for_run(run_id: str, source_video_path: Path, camera_id: str) -> Path:
    run_input_dir = OUTPUT_DIR / run_id / "input" / camera_id
    run_input_dir.mkdir(parents=True, exist_ok=True)

    target_video_path = run_input_dir / source_video_path.name
    shutil.copy2(source_video_path, target_video_path)

    return run_input_dir.parent


def _prepare_dataset_input_for_run(run_id: str, dataset_path: Path) -> Path:
    """Copy dataset input videos into outputs/{run_id}/input/ for full run reproducibility."""
    run_input_root = OUTPUT_DIR / run_id / "input"
    run_input_root.mkdir(parents=True, exist_ok=True)

    copied: List[Dict[str, str]] = []

    for child in sorted(dataset_path.iterdir()):
        if not child.is_dir():
            continue
        camera_dir = run_input_root / child.name
        camera_dir.mkdir(parents=True, exist_ok=True)
        for src in sorted(child.iterdir()):
            if not src.is_file() or src.suffix.lower() not in VIDEO_EXTENSIONS:
                continue
            dst = camera_dir / src.name
            shutil.copy2(src, dst)
            copied.append({"source": str(src), "copiedTo": str(dst.relative_to(OUTPUT_DIR / run_id).as_posix())})

    if not copied:
        misc_dir = run_input_root / "misc"
        misc_dir.mkdir(parents=True, exist_ok=True)
        for src in sorted(dataset_path.iterdir()):
            if not src.is_file() or src.suffix.lower() not in VIDEO_EXTENSIONS:
                continue
            dst = misc_dir / src.name
            shutil.copy2(src, dst)
            copied.append({"source": str(src), "copiedTo": str(dst.relative_to(OUTPUT_DIR / run_id).as_posix())})

    manifest = {
        "sourceDatasetPath": str(dataset_path),
        "copiedAt": datetime.now().isoformat(),
        "copiedVideoCount": len(copied),
        "videos": copied,
    }
    (run_input_root / "input_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return run_input_root


def _cuda_available_for_pipeline() -> bool:
    """Match subprocess torch CUDA visibility (same check as src.core.config)."""
    try:
        from src.core.config import is_torch_cuda_available
        return is_torch_cuda_available()
    except Exception:
        try:
            import torch
            return bool(torch.cuda.is_available())
        except Exception:
            return False


def _build_pipeline_cmd(
    stages: str,
    run_id: str,
    input_dir: str,
    camera_id: Optional[str] = None,
    smoke_test: bool = False,
    use_cpu: bool = False,
    reid_model_path: Optional[str] = None,
    tracker: Optional[str] = None,
    dataset: Optional[str] = None,
    pipeline_config: Optional[str] = None,
    model_overrides: Optional[List[str]] = None,
) -> list:
    """Build the subprocess command for run_pipeline.py."""
    effective_use_cpu = use_cpu or not _cuda_available_for_pipeline()
    config_path = pipeline_config or resolve_pipeline_config_path(dataset)
    cmd = [
        _PIPELINE_PYTHON,
        "scripts/run_pipeline.py",
        "--config",
        config_path,
        "--stages",
        stages,
        "--override",
        f"project.output_dir={OUTPUT_DIR.as_posix()}",
        "--override",
        f"project.run_name='{run_id}'",
        "--override",
        f"stage0.input_dir={input_dir}",
        "--override",
        "stage4.global_gallery.enabled=true",
    ]
    for override in model_overrides or []:
        cmd.extend(["--override", override])
    if camera_id:
        cmd.extend(["--override", f"stage0.cameras=[{camera_id}]"])
    if smoke_test:
        cmd.append("--smoke-test")
    if effective_use_cpu:
        cmd.extend([
            "--override", "stage1.detector.device=cpu",
            "--override", "stage1.tracker.device=cpu",
            "--override", "stage1.detector.half=false",
            "--override", "stage1.tracker.half=false",
            "--override", "stage2.reid.device=cpu",
            "--override", "stage2.reid.half=false",
            "--override", "stage2.reid.batch_size=4",
        ])
    else:
        if sys.platform == "win32":
            cmd.extend([
                "--override", "stage2.reid.half=false",
                "--override", f"stage2.reid.batch_size={_safe_reid_batch_size()}",
            ])
    if reid_model_path:
        cmd.extend([
            "--override", f"stage2.reid.vehicle.weights_path={reid_model_path}",
        ])
    if tracker:
        cmd.extend([
            "--override", f"stage1.tracker.type={tracker}",
        ])
    return cmd


async def _run_pipeline_streaming(
    run_id: str,
    cmd: list,
    stage_nums: list,
) -> Dict[str, Any]:
    """Run a pipeline subprocess using threads so it works on any asyncio event loop."""
    total_stages = max(len(stage_nums), 1)
    # Progress model: each stage owns an equal slice of the 0-95% bar. A stage
    # starts at its base and fills its slice as work streams in (per-frame for the
    # detection stage), so the bar advances smoothly instead of jumping milestones.
    span = 95.0 / total_stages
    stages_started = 0
    stage_base = 0.0
    cameras_total = 0
    cameras_seen: list = []
    log_lines: list = []

    loop = asyncio.get_event_loop()
    line_queue: asyncio.Queue = asyncio.Queue()

    def _set_progress(value: float) -> None:
        """Clamp + monotonically advance the run's progress (never moves backward)."""
        if run_id not in app_state.active_runs:
            return
        run = app_state.active_runs[run_id]
        prev = float(run.get("progress", 0) or 0)
        run["progress"] = int(max(prev, min(value, 95.0)))

    def _handle_line(line: str) -> None:
        nonlocal stages_started, stage_base, cameras_total
        log_lines.append(line)
        # Publish a rolling tail of the subprocess output so the UI can show
        # live, verbose progress (what the pipeline is actually doing right now).
        if run_id in app_state.active_runs:
            app_state.active_runs[run_id]["logTail"] = "\n".join(log_lines[-40:])

        m = _STAGE_LINE_RE.search(line)
        if m:
            stage_num = int(m.group(1))
            stage_label = _STAGE_NAMES.get(stage_num, f"Stage {stage_num}")
            stage_base = stages_started * span
            stages_started += 1
            cameras_seen.clear()
            cameras_total = 0
            if run_id in app_state.active_runs:
                run = app_state.active_runs[run_id]
                run["message"] = f"Running {stage_label}..."
                run["currentStageName"] = stage_label
                run["currentStageNum"] = stage_num
                run["completedStages"] = stages_started
                run["totalStages"] = total_stages
                run["currentFrame"] = 0
                run["totalFrames"] = 0
            _set_progress(stage_base)

        ctm = _CAMERAS_TOTAL_RE.search(line)
        if ctm:
            cameras_total = int(ctm.group(1))
            if run_id in app_state.active_runs:
                app_state.active_runs[run_id]["camerasTotal"] = cameras_total

        cm = _CAMERA_LINE_RE.search(line)
        if cm and run_id in app_state.active_runs:
            cam_id = cm.group(1)
            if cam_id not in cameras_seen:
                cameras_seen.append(cam_id)
            cam_index = cameras_seen.index(cam_id) + 1
            run = app_state.active_runs[run_id]
            run["currentCamera"] = cam_id
            run["camerasProcessed"] = cam_index
            stage_name = run.get("currentStageName", "Processing")
            run["message"] = (
                f"{stage_name} - camera {cam_id} ({cam_index} processed)"
            )

        fm = _FRAME_LINE_RE.search(line)
        if fm and run_id in app_state.active_runs:
            cur_frame = int(fm.group(2))
            cam_frames_total = max(int(fm.group(3)), 1)
            run = app_state.active_runs[run_id]
            run["currentFrame"] = cur_frame
            run["totalFrames"] = cam_frames_total
            # Interpolate within the active stage: completed cameras + the current
            # camera's frame fraction, divided by the known camera total.
            cam_index = max(int(run.get("camerasProcessed", 1)), 1)
            cam_denom = max(cameras_total, cam_index, 1)
            cam_frac = cur_frame / cam_frames_total
            within = ((cam_index - 1) + cam_frac) / cam_denom
            _set_progress(stage_base + within * span)

    def _drain_stream(stream) -> None:
        try:
            for raw_line in stream:
                line = raw_line.rstrip() if isinstance(raw_line, str) else raw_line.decode(errors="ignore").rstrip()
                loop.call_soon_threadsafe(line_queue.put_nowait, ("line", line))
        except Exception:
            pass
        finally:
            loop.call_soon_threadsafe(line_queue.put_nowait, ("done", None))

    def _run_blocking() -> int:
        proc = subprocess.Popen(
            cmd,
            cwd=str(_PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=0,
        )
        # Publish the handle so cancel_pipeline() can terminate this run.
        app_state.run_processes[run_id] = proc
        t_out = threading.Thread(target=_drain_stream, args=(proc.stdout,), daemon=True)
        t_err = threading.Thread(target=_drain_stream, args=(proc.stderr,), daemon=True)
        t_out.start()
        t_err.start()
        proc.wait()
        try:
            proc.stdout.close()
        except Exception:
            pass
        try:
            proc.stderr.close()
        except Exception:
            pass
        t_out.join(timeout=30)
        t_err.join(timeout=30)
        app_state.run_processes.pop(run_id, None)
        return proc.returncode

    future = loop.run_in_executor(None, _run_blocking)

    sentinels = 0
    while sentinels < 2:
        kind, payload = await line_queue.get()
        if kind == "done":
            sentinels += 1
        else:
            _handle_line(payload)

    returncode = await future
    run_dir = OUTPUT_DIR / run_id

    # A user cancel terminates the subprocess (non-zero return). Treat that as a
    # cancellation, not a pipeline error, so the UI can show a clean cancelled state.
    if _is_run_cancelled(run_id):
        raise RunCancelled(f"Run {run_id} cancelled by user")

    if returncode != 0:
        stderr_tail = "\n".join(log_lines[-80:])[-4000:]
        raise RuntimeError(
            f"Pipeline exited with code {returncode}.\n\n"
            f"Last log output:\n{stderr_tail}"
        )

    return {
        "runDir": str(run_dir),
        "logTail": "\n".join(log_lines[-50:]),
    }


async def _run_pipeline_stages(
    run_id: str,
    stages: str,
    video_id: str,
    camera_id: str,
    use_cpu: bool = False,
    smoke_test: bool = False,
    reid_model_path: Optional[str] = None,
    tracker: Optional[str] = None,
    dataset: Optional[str] = None,
    pipeline_config: Optional[str] = None,
    model_overrides: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run one or more pipeline stages via subprocess with streaming progress."""
    video_meta = app_state.uploaded_videos[video_id]
    source_video_path = Path(video_meta["path"]).resolve()
    if not source_video_path.exists():
        raise FileNotFoundError(f"Video file does not exist: {source_video_path}")

    input_dir = _prepare_input_for_run(run_id, source_video_path, camera_id)

    stage_nums = [int(s.strip()) for s in stages.split(",")]

    cmd = _build_pipeline_cmd(
        stages=stages,
        run_id=run_id,
        input_dir=input_dir.as_posix(),
        camera_id=camera_id,
        smoke_test=smoke_test,
        use_cpu=use_cpu,
        reid_model_path=reid_model_path,
        tracker=tracker,
        dataset=dataset,
        pipeline_config=pipeline_config,
        model_overrides=model_overrides,
    )

    return await _run_pipeline_streaming(run_id, cmd, stage_nums)


def _materialize_import_tree(extracted_root: Path, run_dir: Path) -> None:
    """Copy extracted Kaggle artifacts into a normalized run directory."""
    stage_names = {"stage0", "stage1", "stage2", "stage3", "stage4", "stage5", "stage6"}

    candidate_root = extracted_root
    direct_dirs = {p.name for p in extracted_root.iterdir() if p.is_dir()}
    if not stage_names.intersection(direct_dirs):
        children = [p for p in extracted_root.iterdir() if p.is_dir()]
        if len(children) == 1:
            nested_dirs = {p.name for p in children[0].iterdir() if p.is_dir()}
            if stage_names.intersection(nested_dirs):
                candidate_root = children[0]

    run_dir.mkdir(parents=True, exist_ok=True)
    for child in candidate_root.iterdir():
        destination = run_dir / child.name
        if child.is_dir():
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(child, destination)
        else:
            shutil.copy2(child, destination)


async def _background_precompute_dataset() -> None:
    """Run the full pipeline (stages 0-4) on the S01 dataset at startup."""
    dataset_s01 = DATASET_DIR / "S01"
    if not dataset_s01.exists():
        return

    run_dir = OUTPUT_DIR / PRECOMPUTE_RUN_ID
    if any((run_dir / "stage1").glob("tracklets_*.json")):
        for vid_id, vid_meta in list(app_state.uploaded_videos.items()):
            cam_id = _extract_camera_id(str(vid_meta.get("path", "")))
            if cam_id and cam_id.startswith("S01_"):
                app_state.video_to_latest_run[vid_id] = PRECOMPUTE_RUN_ID
        return

    try:
        cmd = [
            _PIPELINE_PYTHON,
            "scripts/run_pipeline.py",
            "--config", "configs/default.yaml",
            "--stages", "0,1,2,3,4",
            "--override", f"project.output_dir={OUTPUT_DIR.as_posix()}",
            "--override", f"project.run_name={PRECOMPUTE_RUN_ID}",
            "--override", f"stage0.input_dir={dataset_s01.as_posix()}",
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(_PROJECT_ROOT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            err = stderr.decode(errors="ignore")[-2000:]
            print(f"[PRECOMPUTE] Pipeline failed: {err}")
            return

        for vid_id, vid_meta in list(app_state.uploaded_videos.items()):
            cam_id = _extract_camera_id(str(vid_meta.get("path", "")))
            if cam_id and cam_id.startswith("S01_"):
                app_state.video_to_latest_run[vid_id] = PRECOMPUTE_RUN_ID

        print(
            f"[PRECOMPUTE] S01 pipeline complete - "
            f"{len(list((run_dir / 'stage1').glob('tracklets_*.json')))} cameras processed"
        )

    except Exception as exc:
        print(f"[PRECOMPUTE] Background precompute error: {exc}")


async def execute_stage(run_id: str, stage: int, config: Dict[str, Any]):
    """Execute a real pipeline stage for a selected video."""
    try:
        if run_id not in app_state.active_runs:
            return

        video_id = config.get("videoId")
        camera_id = config.get("cameraId")
        smoke_test = bool(config.get("smokeTest", False))
        use_cpu = bool(config.get("useCpu", False))
        reid_model_path = config.get("reidModelPath")
        tracker = str(config.get("tracker") or "deepocsort")
        dataset = config.get("dataset")
        pipeline_config = config.get("resolvedConfig")
        model_overrides = list(config.get("appliedOverrides") or [])

        if not video_id or video_id not in app_state.uploaded_videos:
            raise RuntimeError(f"Stage {stage} requires a valid videoId")

        if not camera_id:
            camera_id = _detect_camera_for_video(app_state.uploaded_videos[video_id], None)

        app_state.active_runs[run_id]["cameraId"] = camera_id

        if stage == 1:
            app_state.active_runs[run_id]["message"] = f"Running detection & tracking for {camera_id}..."
            app_state.active_runs[run_id]["progress"] = 10

            run_meta = await _run_pipeline_stages(
                run_id=run_id,
                stages="0,1",
                video_id=video_id,
                camera_id=camera_id,
                use_cpu=use_cpu,
                smoke_test=smoke_test,
                tracker=tracker,
                dataset=dataset,
                pipeline_config=pipeline_config,
                model_overrides=model_overrides,
            )

            app_state.video_to_latest_run[video_id] = run_id
            _persist_probe_link(video_id, run_id)
            app_state.active_runs[run_id]["status"] = "completed"
            app_state.active_runs[run_id]["progress"] = 100
            app_state.active_runs[run_id]["message"] = "Detection & tracking complete"
            app_state.active_runs[run_id]["runDir"] = run_meta["runDir"]
            app_state.active_runs[run_id]["completedAt"] = datetime.now().isoformat()
            return

        if stage in (2, 3):
            run_dir = OUTPUT_DIR / run_id
            stage2_done = (run_dir / "stage2" / "embeddings.npy").exists()

            if stage == 3 and stage2_done:
                stages_to_run = "3"
                app_state.active_runs[run_id]["message"] = "Running indexing (embeddings already extracted)..."
            else:
                stages_to_run = "2,3"
                app_state.active_runs[run_id]["message"] = "Running feature extraction & indexing..."

            app_state.active_runs[run_id]["progress"] = 10

            run_meta = await _run_pipeline_stages(
                run_id=run_id,
                stages=stages_to_run,
                video_id=video_id,
                camera_id=camera_id,
                use_cpu=use_cpu,
                smoke_test=smoke_test,
                reid_model_path=reid_model_path,
                dataset=dataset,
                pipeline_config=pipeline_config,
                model_overrides=model_overrides,
            )

            app_state.active_runs[run_id]["status"] = "completed"
            app_state.active_runs[run_id]["progress"] = 100
            app_state.active_runs[run_id]["message"] = f"Stage {stage} complete"
            app_state.active_runs[run_id]["runDir"] = run_meta["runDir"]
            app_state.active_runs[run_id]["completedAt"] = datetime.now().isoformat()
            return

        if stage == 4:
            app_state.active_runs[run_id]["message"] = "Running cross-camera association..."
            app_state.active_runs[run_id]["progress"] = 10

            run_meta = await _run_pipeline_stages(
                run_id=run_id,
                stages="4",
                video_id=video_id,
                camera_id=camera_id,
                use_cpu=use_cpu,
                smoke_test=smoke_test,
                dataset=dataset,
                pipeline_config=pipeline_config,
                model_overrides=model_overrides,
            )

            app_state.active_runs[run_id]["status"] = "completed"
            app_state.active_runs[run_id]["progress"] = 100
            app_state.active_runs[run_id]["message"] = "Association complete"
            app_state.active_runs[run_id]["runDir"] = run_meta["runDir"]
            app_state.active_runs[run_id]["completedAt"] = datetime.now().isoformat()
            return

        stage_name = {5: "evaluation", 6: "visualization"}.get(stage, str(stage))
        app_state.active_runs[run_id]["message"] = f"Running {stage_name}..."
        app_state.active_runs[run_id]["progress"] = 10

        run_meta = await _run_pipeline_stages(
            run_id=run_id,
            stages=str(stage),
            video_id=video_id,
            camera_id=camera_id,
            use_cpu=use_cpu,
            smoke_test=smoke_test,
            dataset=dataset,
            pipeline_config=pipeline_config,
            model_overrides=model_overrides,
        )

        app_state.active_runs[run_id]["status"] = "completed"
        app_state.active_runs[run_id]["progress"] = 100
        app_state.active_runs[run_id]["completedAt"] = datetime.now().isoformat()

    except BaseException as e:
        tb = _traceback.format_exc()
        print(f"[PIPELINE ERROR] run={run_id} stage={stage}\n{tb}", flush=True)
        _finalize_run_failure(run_id, e, tb, f"Stage {stage} failed")
        if isinstance(e, (asyncio.CancelledError, KeyboardInterrupt)):
            raise


async def execute_full_pipeline(run_id: str, config: Dict[str, Any]):
    """Execute all pipeline stages (0-4) in sequence."""
    try:
        video_id = config.get("videoId")
        camera_id = config.get("cameraId")
        smoke_test = bool(config.get("smokeTest", False))
        use_cpu = bool(config.get("useCpu", False))
        reid_model_path = config.get("reidModelPath")
        dataset = config.get("dataset")
        pipeline_config = config.get("resolvedConfig")
        model_overrides = list(config.get("appliedOverrides") or [])

        if not video_id or video_id not in app_state.uploaded_videos:
            raise RuntimeError("Full pipeline requires a valid videoId")

        if not camera_id:
            camera_id = _detect_camera_for_video(app_state.uploaded_videos[video_id], None)

        app_state.active_runs[run_id]["cameraId"] = camera_id

        app_state.active_runs[run_id]["message"] = "Running full pipeline (stages 0-4)..."
        app_state.active_runs[run_id]["progress"] = 5

        run_meta = await _run_pipeline_stages(
            run_id=run_id,
            stages="0,1,2,3,4",
            video_id=video_id,
            camera_id=camera_id,
            use_cpu=use_cpu,
            smoke_test=smoke_test,
            reid_model_path=reid_model_path,
            dataset=dataset,
            pipeline_config=pipeline_config,
            model_overrides=model_overrides,
        )

        app_state.video_to_latest_run[video_id] = run_id
        _persist_probe_link(video_id, run_id)
        app_state.active_runs[run_id]["status"] = "completed"
        app_state.active_runs[run_id]["progress"] = 100
        app_state.active_runs[run_id]["message"] = "Full pipeline complete"
        app_state.active_runs[run_id]["runDir"] = run_meta["runDir"]
        app_state.active_runs[run_id]["completedAt"] = datetime.now().isoformat()

    except BaseException as e:
        tb = _traceback.format_exc()
        print(f"[PIPELINE ERROR] full-pipeline run={run_id}\n{tb}", flush=True)
        _finalize_run_failure(run_id, e, tb, "Pipeline failed")
        if isinstance(e, (asyncio.CancelledError, KeyboardInterrupt)):
            raise


async def _execute_dataset_pipeline(run_id: str, dataset_path: Path, folder_name: str):
    """Background task: run stages 0-4 on a full dataset folder."""
    try:
        stage_nums = [0, 1, 2, 3, 4]
        app_state.active_runs[run_id]["message"] = "Preparing run-local dataset input copy..."
        app_state.active_runs[run_id]["progress"] = 1
        input_dir = _prepare_dataset_input_for_run(run_id, dataset_path)

        cmd = _build_pipeline_cmd(
            stages="0,1,2,3,4",
            run_id=run_id,
            input_dir=input_dir.as_posix(),
        )

        app_state.active_runs[run_id]["message"] = "Running Ingestion & Pre-Processing..."
        app_state.active_runs[run_id]["progress"] = 2

        run_meta = await _run_pipeline_streaming(run_id, cmd, stage_nums)

        scene_prefix = folder_name.upper()
        for vid_id, vid_meta in list(app_state.uploaded_videos.items()):
            cam_id = _extract_camera_id(str(vid_meta.get("path", "")))
            if cam_id and cam_id.startswith(f"{scene_prefix}_"):
                app_state.video_to_latest_run[vid_id] = run_id

        app_state.active_runs[run_id]["status"] = "completed"
        app_state.active_runs[run_id]["progress"] = 100
        app_state.active_runs[run_id]["message"] = f"Pipeline complete for {folder_name}"
        app_state.active_runs[run_id]["runDir"] = run_meta["runDir"]
        app_state.active_runs[run_id]["completedAt"] = datetime.now().isoformat()

    except Exception as e:
        _finalize_run_failure(run_id, e, _traceback.format_exc(), "Error")


def _link_input_dir_videos_to_run(
    run_id: str, input_dir: str, cameras: Optional[List[str]]
) -> None:
    """Map registered camera videos under `input_dir` to `run_id` so the detections
    endpoint can resolve each camera's stage-1 tracklets for this run."""
    try:
        root = Path(input_dir).resolve()
    except Exception:
        return
    selected_norm = (
        {_normalize_camera_id(c) for c in cameras} if cameras else None
    )
    for vid_id, vid_meta in list(app_state.uploaded_videos.items()):
        vpath = str(vid_meta.get("path", ""))
        if not vpath:
            continue
        try:
            resolved = Path(vpath).resolve()
        except Exception:
            continue
        if resolved != root and root not in resolved.parents:
            continue
        if selected_norm is not None:
            cam = _extract_camera_id(vpath) or _extract_camera_id(str(vid_meta.get("name", "")))
            if cam is None or _normalize_camera_id(cam) not in selected_norm:
                continue
        app_state.video_to_latest_run[vid_id] = run_id
        _persist_probe_link(vid_id, run_id)


async def _execute_input_dir_pipeline(
    run_id: str,
    input_dir: str,
    stages: str,
    smoke_test: bool,
    label: str,
    cameras: Optional[List[str]] = None,
):
    """Background task: run the selected stages directly against a source folder.

    Unlike `_execute_dataset_pipeline`, this does NOT copy videos into the run
    dir - it points `stage0.input_dir` at the chosen folder. Stage 0's own video
    discovery handles both the per-camera (`<cam>/vdo.avi`) and flat
    (`C1.mp4`, `C2.mp4`, ...) layouts, so this works for every dataset.

    If `cameras` is given, only those cameras are processed (via a
    `stage0.cameras=[...]` override), enabling multi-camera subset selection.
    """
    try:
        stage_nums = [int(s) for s in stages.split(",") if s.strip().isdigit()]
        app_state.active_runs[run_id]["message"] = f"Running ingestion on {label}..."
        app_state.active_runs[run_id]["progress"] = 2

        cmd = _build_pipeline_cmd(
            stages=stages,
            run_id=run_id,
            input_dir=Path(input_dir).as_posix(),
            smoke_test=smoke_test,
        )
        if cameras:
            cam_list = ",".join(cameras)
            cmd.extend(["--override", f"stage0.cameras=[{cam_list}]"])

        run_meta = await _run_pipeline_streaming(run_id, cmd, stage_nums)

        # Link this run to every registered camera video under the input folder so
        # the detections viewer can resolve outputs/<run_id>/stage1/tracklets_*.json
        # for the selected camera. Without this, the per-camera detection display
        # finds no run for the video and shows nothing.
        _link_input_dir_videos_to_run(run_id, input_dir, cameras)

        app_state.active_runs[run_id]["status"] = "completed"
        app_state.active_runs[run_id]["progress"] = 100
        app_state.active_runs[run_id]["message"] = f"Pipeline complete for {label}"
        app_state.active_runs[run_id]["runDir"] = run_meta["runDir"]
        app_state.active_runs[run_id]["completedAt"] = datetime.now().isoformat()

    except Exception as e:
        _finalize_run_failure(run_id, e, _traceback.format_exc(), "Error")
