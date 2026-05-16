"""Controlled subprocess-backed evaluation jobs for verified models."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from backend.services.job_service import Job, job_service

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class EvalSpec:
    eval_type: str
    script: Path
    defaults: dict[str, Any]
    cli_flag_style: str = "hyphen"
    boolean_flags: dict[str, tuple[str, str]] = field(default_factory=dict)
    allowed_overrides: set[str] = field(default_factory=set)


def _repo_path(path: str) -> Path:
    return PROJECT_ROOT / path


EVAL_SPECS: dict[str, EvalSpec] = {
    "veri776_transreid": EvalSpec(
        eval_type="veri776_transreid",
        script=_repo_path("scripts/eval/eval_09v_transreid_veri776.py"),
        defaults={
            "checkpoint": _repo_path("models/reid/vehicle_transreid_vit_base_veri776.pth"),
            "veri_root": _repo_path("data/raw/veri776"),
            "device": "cpu",
            "batch_size": 64,
            "rerank": True,
            "aqe_k": 2,
        },
        boolean_flags={"rerank": ("--rerank", "--no-rerank")},
        allowed_overrides={"checkpoint", "veri_root", "device", "batch_size", "rerank", "aqe_k"},
    ),
    "veri776_clipsenet": EvalSpec(
        eval_type="veri776_clipsenet",
        script=_repo_path("scripts/eval/eval_clip_senet_veri776.py"),
        defaults={
            "checkpoint": _repo_path("models/reid/clipsenet_v6_veri776_best.pth"),
            "veri_root": _repo_path("data/raw/veri776"),
            "device": "cpu",
            "batch_size": 64,
            "img_size": [320, 320],
            "rerank": False,
            "aqe_k": 1,
        },
        boolean_flags={"rerank": ("--rerank", "--no-rerank")},
        allowed_overrides={"checkpoint", "veri_root", "device", "batch_size", "img_size", "rerank", "aqe_k"},
    ),
    "cityflow_transreid": EvalSpec(
        eval_type="cityflow_transreid",
        script=_repo_path("scripts/eval_cityflowv2_reid.py"),
        cli_flag_style="underscore",
        defaults={
            "weights": _repo_path("models/reid/transreid_cityflowv2_best.pth"),
            "data_root": _repo_path("data/raw/cityflowv2"),
            "crop_dir": _repo_path("data/processed/cityflowv2_crops"),
            "device": "cpu",
            "batch_size": 64,
            "num_workers": 0,
            "img_size": [256, 256],
            "max_crops": 2,
            "max_ids": 32,
            "qe_k": 0,
            "rerank": True,
            "k1": 20,
            "k2": 6,
            "lambda_value": 0.3,
        },
        boolean_flags={"rerank": ("--rerank", "--no-rerank")},
        allowed_overrides={
            "weights",
            "data_root",
            "crop_dir",
            "device",
            "batch_size",
            "num_workers",
            "img_size",
            "max_crops",
            "max_ids",
            "qe_k",
            "rerank",
            "k1",
            "k2",
            "lambda_value",
        },
    ),
    "veri776_14t_fusion": EvalSpec(
        eval_type="veri776_14t_fusion",
        script=_repo_path("scripts/eval/eval_14t_fusion_veri776.py"),
        defaults={
            "transreid_checkpoint": _repo_path("models/reid/vehicle_transreid_vit_base_veri776.pth"),
            "clipsenet_checkpoint": _repo_path("models/reid/clipsenet_v6_veri776_best.pth"),
            "veri_root": _repo_path("data/raw/veri776"),
            "device": "cpu",
            "w_clipsenet": 0.7,
            "transreid_stream": "global",
            "aqe_k": 2,
            "rerank_k1": 30,
            "rerank_k2": 10,
            "rerank_lambda": 0.2,
            "transreid_batch_size": 64,
            "clipsenet_batch_size": 64,
            "clipsenet_img_size": [320, 320],
            "skip_drift_parents": True,
            "weights_sweep": False,
            "concat_sweep": False,
        },
        boolean_flags={
            "skip_drift_parents": ("--skip-drift-parents", ""),
            "weights_sweep": ("--weights-sweep", ""),
            "concat_sweep": ("--concat-sweep", ""),
        },
        allowed_overrides={
            "transreid_checkpoint",
            "clipsenet_checkpoint",
            "veri_root",
            "device",
            "w_clipsenet",
            "transreid_stream",
            "aqe_k",
            "rerank_k1",
            "rerank_k2",
            "rerank_lambda",
            "transreid_batch_size",
            "clipsenet_batch_size",
            "clipsenet_img_size",
            "skip_drift_parents",
            "weights_sweep",
            "concat_sweep",
        },
    ),
}


class EvalService:
    def __init__(self) -> None:
        for eval_type in EVAL_SPECS:
            job_service.register_handler(eval_type, self._run_eval_job)

    def submit_eval(
        self,
        eval_type: str,
        config_overrides: dict[str, Any] | None = None,
        background_tasks: Any | None = None,
    ) -> str:
        if eval_type not in EVAL_SPECS:
            raise ValueError(f"Unsupported eval_type: {eval_type}")
        return job_service.submit_job(
            eval_type,
            {"config_overrides": config_overrides or {}},
            background_tasks=background_tasks,
        )

    def _run_eval_job(self, payload: dict[str, Any], job: Job) -> dict[str, Any]:
        spec = EVAL_SPECS[job.eval_type]
        overrides = payload.get("config_overrides") or {}
        if not isinstance(overrides, dict):
            raise ValueError("config_overrides must be an object")

        unknown = sorted(set(overrides) - spec.allowed_overrides)
        if unknown:
            raise ValueError(f"Unsupported override(s) for {job.eval_type}: {', '.join(unknown)}")

        job_dir = job_service.job_dir / job.id
        job_dir.mkdir(parents=True, exist_ok=True)
        result_path = job_dir / "result.json"
        stderr_path = job_dir / "stderr.log"
        stdout_path = job_dir / "stdout.log"
        command = self._build_command(spec, overrides, result_path)
        job.progress = {"stage": "subprocess", "percent": 10, "command": self._public_command(command)}
        job_service._persist(job)

        with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open("w", encoding="utf-8") as stderr_handle:
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
                check=False,
            )

        if completed.returncode != 0:
            tail = _tail_text(stderr_path)
            raise RuntimeError(f"Eval subprocess failed with exit code {completed.returncode}: {tail}")
        if not result_path.is_file():
            raise RuntimeError(f"Eval subprocess completed but did not write {result_path}")

        job.progress = {"stage": "parsing_result", "percent": 95}
        job_service._persist(job)
        result = json.loads(result_path.read_text(encoding="utf-8"))
        return {"evalType": job.eval_type, "summary": summarize_eval_result(result), "result": result}

    def _build_command(self, spec: EvalSpec, overrides: dict[str, Any], result_path: Path) -> list[str]:
        values = {**spec.defaults, **overrides}
        command = [sys.executable, str(spec.script)]
        for key, value in values.items():
            if key in spec.boolean_flags:
                true_flag, false_flag = spec.boolean_flags[key]
                flag = true_flag if bool(value) else false_flag
                if flag:
                    command.append(flag)
                continue
            command.append(_cli_flag(key, spec.cli_flag_style))
            command.extend(_stringify_cli_values(value))
        command.extend(["--output-json", str(result_path)])
        return command

    def _public_command(self, command: list[str]) -> list[str]:
        public: list[str] = []
        for part in command:
            try:
                path = Path(part)
                if path.is_absolute():
                    public.append(os.path.relpath(path, PROJECT_ROOT))
                    continue
            except ValueError:
                pass
            public.append(part)
        return public


def _stringify_cli_values(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def _cli_flag(key: str, style: str) -> str:
    if style == "underscore":
        return f"--{key}"
    if style == "hyphen":
        return f"--{key.replace('_', '-')}"
    raise ValueError(f"Unsupported CLI flag style: {style}")


def _tail_text(path: Path, limit: int = 2000) -> str:
    if not path.exists():
        return "stderr.log missing"
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-limit:].strip() or "stderr was empty"


def summarize_eval_result(result: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    metric_keys = [
        "mAP",
        "map",
        "rank1",
        "r1",
        "R1",
        "R5",
        "R10",
        "cosine_mAP",
        "rerank_mAP",
        "best_mAP",
        "best_R1",
        "idf1",
        "mtmc_idf1",
        "moda",
    ]
    for key in metric_keys:
        value = _find_metric(result, key)
        if value is not None:
            summary[key] = value
    if not summary and isinstance(result, dict):
        for key, value in result.items():
            if isinstance(value, (int, float, str, bool)) or value is None:
                summary[key] = value
            if len(summary) >= 8:
                break
    return summary


def _find_metric(value: Any, target_key: str) -> Any | None:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() == target_key.lower() and isinstance(item, (int, float, str, bool)):
                return item
        for item in value.values():
            found = _find_metric(item, target_key)
            if found is not None:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _find_metric(item, target_key)
            if found is not None:
                return found
    return None


eval_service = EvalService()