from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path
from typing import Any, Dict, List, Optional

_KERNEL_OWNER_RE = re.compile(r"^[a-z0-9-]+$")
_DATASET_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*/[a-z0-9][a-z0-9_-]*$")
_KERNEL_NAME_MAX_LENGTH = 50


@dataclass(frozen=True, kw_only=True)
class KernelTemplateContext:
    run_id: str
    stages: List[int]
    config_path: str
    dataset_slugs: List[str]
    project_dataset_slug: str
    kernel_owner: str
    kernel_title: str
    model_id: Optional[str] = None
    fusion: Optional[Dict[str, Any]] = None
    config_overrides: List[str] = field(default_factory=list)
    enable_gpu: bool = True
    enable_internet: bool = False
    custom_python_imports: List[str] = field(default_factory=list)


def render_kernel(ctx: KernelTemplateContext, output_dir: Path) -> Path:
    """Render notebook.ipynb and kernel-metadata.json into output_dir."""
    _validate_context(ctx)
    output_dir.mkdir(parents=True, exist_ok=True)

    kernel_name = _kernel_name(ctx)
    kernel_slug = f"{ctx.kernel_owner}/{kernel_name}"
    notebook = build_notebook(ctx)
    metadata = build_kernel_metadata(ctx, kernel_slug)

    (output_dir / "notebook.ipynb").write_text(
        json.dumps(notebook, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    (output_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return output_dir


def build_notebook(ctx: KernelTemplateContext) -> Dict[str, Any]:
    """Build a parameterized Kaggle notebook dictionary for the requested stages."""
    _validate_context(ctx)
    project_mount_name = _dataset_mount_name(ctx.project_dataset_slug)
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    custom_imports = "\n".join(ctx.custom_python_imports)
    custom_imports_block = f"\n{custom_imports}\n" if custom_imports else ""

    cells = [
        _markdown_cell(
            f"""# {ctx.kernel_title}

Run ID: `{ctx.run_id}`

Stages: `{','.join(str(stage) for stage in ctx.stages)}`

<!-- Generated {timestamp} -->"""
        ),
        _code_cell(
            f"""import os, sys, json, subprocess
from pathlib import Path
{custom_imports_block}
# Mounted dataset paths
PROJECT_ROOT = Path("/kaggle/input/{project_mount_name}")
INPUT_ROOT = Path("/kaggle/input")
OUTPUT_ROOT = Path("/kaggle/working")
RUN_ID = {json.dumps(ctx.run_id, ensure_ascii=True)}
STAGES = {json.dumps(ctx.stages, ensure_ascii=True)}
CONFIG_PATH = PROJECT_ROOT / {json.dumps(ctx.config_path, ensure_ascii=True)}

# Add project to PYTHONPATH
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

print("Project:", PROJECT_ROOT)
print("Run ID:", RUN_ID)
print("Stages:", STAGES)
print("Config:", CONFIG_PATH)"""
        ),
        _code_cell(
            """requirements_kaggle = PROJECT_ROOT / "requirements-kaggle.txt"
requirements_default = PROJECT_ROOT / "requirements.txt"
requirements_file = requirements_kaggle if requirements_kaggle.exists() else requirements_default

if requirements_file.exists():
    print("Installing dependencies from", requirements_file)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements_file)],
        check=True,
    )
else:
    print("No requirements file found; skipping dependency install")"""
        ),
        _code_cell(
            f"""overrides = {json.dumps(ctx.config_overrides, ensure_ascii=True)}
model_id = {json.dumps(ctx.model_id, ensure_ascii=True)}
fusion = {json.dumps(ctx.fusion, ensure_ascii=True)}

run_request = {{
    "run_id": RUN_ID,
    "stages": STAGES,
    "config_path": str(CONFIG_PATH),
    "config_overrides": overrides,
    "model_id": model_id,
    "fusion": fusion,
}}

REQUEST_FILE = OUTPUT_ROOT / "run_request.json"
with REQUEST_FILE.open("w") as f:
    json.dump(run_request, f, indent=2)
print(json.dumps(run_request, indent=2))"""
        ),
        _code_cell(
            """# Build the CLI command
cmd = [
    sys.executable,
    str(PROJECT_ROOT / "scripts" / "run_pipeline.py"),
    "--config", str(CONFIG_PATH),
    "--stages", ",".join(map(str, STAGES)),
    "--run-id", RUN_ID,
    "--output-dir", str(OUTPUT_ROOT / "outputs" / RUN_ID),
]
if model_id:
    cmd.extend(["--override", f"run.model_id={model_id}"])
for override in overrides:
    cmd.extend(["--override", override])

print("Running:", " ".join(cmd))
result = subprocess.run(cmd, check=False)
print("Exit code:", result.returncode)"""
        ),
        _code_cell(
            """import zipfile, shutil
ZIP_PATH = OUTPUT_ROOT / f"{RUN_ID}_outputs.zip"
with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
    out_dir = OUTPUT_ROOT / "outputs" / RUN_ID
    if out_dir.exists():
        for p in out_dir.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(out_dir))
print(f"Wrote {ZIP_PATH} ({ZIP_PATH.stat().st_size} bytes)")"""
        ),
        _code_cell(
            """manifest = {
    "run_id": RUN_ID,
    "stages": STAGES,
    "exit_code": result.returncode,
    "output_zip": str(ZIP_PATH.relative_to(OUTPUT_ROOT)),
}
with (OUTPUT_ROOT / "manifest.json").open("w") as f:
    json.dump(manifest, f, indent=2)
print(json.dumps(manifest, indent=2))"""
        ),
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build_kernel_metadata(ctx: KernelTemplateContext, kernel_slug: str) -> Dict[str, Any]:
    """Build Kaggle kernel-metadata.json for a rendered notebook."""
    _validate_context(ctx)
    if kernel_slug != f"{ctx.kernel_owner}/{_kernel_name(ctx)}":
        raise ValueError("kernel_slug must match the context-derived owner/name.")

    return {
        "id": kernel_slug,
        "id_no": None,
        "title": ctx.kernel_title,
        "code_file": "notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": ctx.enable_gpu,
        "enable_tpu": False,
        "enable_internet": ctx.enable_internet,
        "dataset_sources": list(ctx.dataset_slugs),
        "competition_sources": [],
        "kernel_sources": [],
    }


def kernel_for_stages_012(
    *,
    run_id: str,
    kernel_owner: str,
    config_path: str,
    project_dataset_slug: str,
    video_dataset_slug: str,
    model_id: Optional[str] = None,
    fusion: Optional[Dict[str, Any]] = None,
    config_overrides: Optional[List[str]] = None,
) -> KernelTemplateContext:
    """Build a GPU-enabled context for Stages 0-2."""
    return KernelTemplateContext(
        run_id=run_id,
        stages=[0, 1, 2],
        config_path=config_path,
        dataset_slugs=[project_dataset_slug, video_dataset_slug],
        project_dataset_slug=project_dataset_slug,
        kernel_owner=kernel_owner,
        kernel_title=f"MTMC {run_id} - Stages 0-2",
        model_id=model_id,
        fusion=fusion,
        config_overrides=list(config_overrides or []),
        enable_gpu=True,
        enable_internet=False,
    )


def kernel_for_stage_4(
    *,
    run_id: str,
    kernel_owner: str,
    config_path: str,
    project_dataset_slug: str,
    stage2_outputs_dataset_slug: str,
    model_id: Optional[str] = None,
    fusion: Optional[Dict[str, Any]] = None,
    config_overrides: Optional[List[str]] = None,
) -> KernelTemplateContext:
    """Build a CPU-only context for Stage 4 association."""
    return KernelTemplateContext(
        run_id=run_id,
        stages=[4],
        config_path=config_path,
        dataset_slugs=[project_dataset_slug, stage2_outputs_dataset_slug],
        project_dataset_slug=project_dataset_slug,
        kernel_owner=kernel_owner,
        kernel_title=f"MTMC {run_id} - Stage 4",
        model_id=model_id,
        fusion=fusion,
        config_overrides=list(config_overrides or []),
        enable_gpu=False,
        enable_internet=False,
    )


def _validate_context(ctx: KernelTemplateContext) -> None:
    if not ctx.run_id or not ctx.run_id.strip():
        raise ValueError("run_id is required.")
    if not ctx.stages:
        raise ValueError("stages must be non-empty.")
    if any(not isinstance(stage, int) or stage < 0 or stage > 6 for stage in ctx.stages):
        raise ValueError("stages must be integers from 0 through 6.")
    if not _KERNEL_OWNER_RE.fullmatch(ctx.kernel_owner):
        raise ValueError("kernel_owner must match ^[a-z0-9-]+$.")
    if not ctx.config_path or Path(ctx.config_path).is_absolute():
        raise ValueError("config_path must be a repo-relative path.")
    if not ctx.dataset_slugs:
        raise ValueError("dataset_slugs must be non-empty.")
    if ctx.project_dataset_slug not in ctx.dataset_slugs:
        raise ValueError("project_dataset_slug must be included in dataset_slugs.")
    for slug in ctx.dataset_slugs:
        if not _DATASET_SLUG_RE.fullmatch(slug):
            raise ValueError(f"Invalid dataset slug: {slug!r}.")


def _kernel_name(ctx: KernelTemplateContext) -> str:
    run_id_slug = _slug_fragment(ctx.run_id)
    stages_str = "".join(str(stage) for stage in ctx.stages)
    suffix = f"-stages-{stages_str}"
    prefix = "mtmc-"
    max_run_length = _KERNEL_NAME_MAX_LENGTH - len(prefix) - len(suffix)
    if max_run_length < 1:
        raise ValueError("Kernel name budget is too small for the requested stages.")
    run_id_slug = run_id_slug[:max_run_length].strip("-") or "run"
    return f"{prefix}{run_id_slug}{suffix}"


def _slug_fragment(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "run"


def _dataset_mount_name(dataset_slug: str) -> str:
    return dataset_slug.split("/", 1)[1]


def _markdown_cell(source: str) -> Dict[str, Any]:
    return {
        "cell_type": "markdown",
        "id": _cell_id("markdown", source),
        "metadata": {"language": "markdown"},
        "source": _source_lines(source),
    }


def _code_cell(source: str) -> Dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": _cell_id("code", source),
        "metadata": {"language": "python"},
        "outputs": [],
        "source": _source_lines(source),
    }


def _cell_id(kind: str, source: str) -> str:
    digest = sha1(f"{kind}\n{source}".encode("utf-8")).hexdigest()[:8]
    return f"mtmc-{digest}"


def _source_lines(source: str) -> List[str]:
    lines = source.splitlines()
    if not lines:
        return [""]
    return [line + "\n" for line in lines[:-1]] + [lines[-1]]
