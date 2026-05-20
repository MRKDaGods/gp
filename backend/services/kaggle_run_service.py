from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from backend.config import _PROJECT_ROOT
from backend.models.requests import KaggleConfig
from backend.services import kaggle_kernel_templates, kaggle_service
from backend.services.kaggle_service import (
    KaggleAuthError,
    KaggleConcurrencyError,
    KaggleValidationError,
)


@dataclass(frozen=True)
class KaggleRunResult:
    run_id: str
    kernel_slug: str
    kernel_url: str
    dataset_slug: str
    project_dataset_slug: str
    status: str
    metadata_path: str

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


def dispatch_stage_to_kaggle(
    *,
    run_id: str,
    stages: List[int],
    config_path: str,
    config_overrides: List[str],
    model_id: Optional[str],
    fusion: Optional[Dict[str, Any]],
    kaggle_cfg: KaggleConfig,
    user_video_path: Optional[Path] = None,
    output_root: Path = Path("data/outputs"),
) -> KaggleRunResult:
    username, key = _resolve_credentials(kaggle_cfg)
    owner = kaggle_service.whoami(username=username, key=key)

    if kaggle_service.count_active_kernels(username=username, key=key) >= 2:
        raise KaggleConcurrencyError("Kaggle has reached the active kernel limit for this account.")

    run_dir = output_root / run_id
    kaggle_work_dir = run_dir / "_kaggle"
    kaggle_work_dir.mkdir(parents=True, exist_ok=True)

    project_dataset_slug = _ensure_project_source_dataset(
        owner=owner,
        run_id=run_id,
        output_root=output_root,
        kaggle_work_dir=kaggle_work_dir,
        username=username,
        key=key,
    )
    dataset_slug = _resolve_video_dataset(
        owner=owner,
        run_id=run_id,
        kaggle_cfg=kaggle_cfg,
        user_video_path=user_video_path,
        kaggle_work_dir=kaggle_work_dir,
        username=username,
        key=key,
        output_root=output_root,
    )

    kernel_dir = kaggle_work_dir / "kernel"
    ctx = _kernel_context(
        run_id=run_id,
        stages=stages,
        owner=owner,
        config_path=config_path,
        project_dataset_slug=project_dataset_slug,
        dataset_slug=dataset_slug,
        model_id=model_id,
        fusion=fusion,
        config_overrides=config_overrides,
    )
    metadata_dir = kaggle_kernel_templates.render_kernel(ctx, kernel_dir)
    push_result = kaggle_service.push_kernel(metadata_dir, username=username, key=key)

    kernel_slug = push_result.slug or _read_rendered_kernel_slug(metadata_dir)
    kernel_url = push_result.kernel_url or f"https://www.kaggle.com/code/{kernel_slug}"
    metadata_path = run_dir / "kaggle_job.json"
    state = {
        "run_id": run_id,
        "kernel_slug": kernel_slug,
        "kernel_url": kernel_url,
        "dataset_slug": dataset_slug,
        "project_dataset_slug": project_dataset_slug,
        "status": "queued",
        "stages": list(stages),
        "started_at": _utc_now_iso(),
        "last_polled_at": None,
        "exit_code": None,
        "outputs_downloaded_to": None,
    }
    metadata_path.write_text(json.dumps(state, indent=2, ensure_ascii=True), encoding="utf-8")

    return KaggleRunResult(
        run_id=run_id,
        kernel_slug=kernel_slug,
        kernel_url=kernel_url,
        dataset_slug=dataset_slug,
        project_dataset_slug=project_dataset_slug,
        status="queued",
        metadata_path=str(metadata_path),
    )


def get_kaggle_job_state(
    run_id: str,
    output_root: Path = Path("data/outputs"),
) -> Optional[Dict[str, Any]]:
    metadata_path = output_root / run_id / "kaggle_job.json"
    if not metadata_path.exists():
        return None
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def refresh_kaggle_job_status(
    run_id: str,
    kaggle_cfg: Optional[KaggleConfig] = None,
    output_root: Path = Path("data/outputs"),
) -> Dict[str, Any]:
    state = get_kaggle_job_state(run_id, output_root=output_root)
    if state is None:
        raise FileNotFoundError(f"No Kaggle job found for run_id {run_id}")

    username: Optional[str] = None
    key: Optional[str] = None
    if kaggle_cfg is not None:
        username, key = _resolve_credentials(kaggle_cfg)
    status = kaggle_service.kernel_status(str(state["kernel_slug"]), username=username, key=key)
    state["status"] = status.status
    state["last_polled_at"] = status.last_polled_iso

    metadata_path = output_root / run_id / "kaggle_job.json"
    metadata_path.write_text(json.dumps(state, indent=2, ensure_ascii=True), encoding="utf-8")
    return state


def _resolve_credentials(kaggle_cfg: KaggleConfig) -> Tuple[Optional[str], Optional[str]]:
    if kaggle_cfg.username and kaggle_cfg.key:
        return kaggle_cfg.username, kaggle_cfg.key

    credential_path = Path.home() / ".kaggle" / "kaggle.json"
    if not credential_path.exists():
        raise KaggleAuthError(
            "Kaggle credentials are required. Provide username/key or configure "
            "~/.kaggle/kaggle.json."
        )

    try:
        payload = json.loads(credential_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise KaggleAuthError("Unable to read ~/.kaggle/kaggle.json.") from exc

    username = payload.get("username")
    key = payload.get("key")
    if not username or not key:
        raise KaggleAuthError("~/.kaggle/kaggle.json must contain username and key.")
    return str(username), str(key)


def _ensure_project_source_dataset(
    *,
    owner: str,
    run_id: str,
    output_root: Path,
    kaggle_work_dir: Path,
    username: Optional[str],
    key: Optional[str],
) -> str:
    slug = f"{owner}/mtmc-tracker-source"
    source_dir = kaggle_work_dir / "project_source"
    if source_dir.exists():
        shutil.rmtree(source_dir)
    _build_project_source_archive(source_dir)

    marker_dir = output_root / "_kaggle"
    marker_dir.mkdir(parents=True, exist_ok=True)
    marker_path = marker_dir / f"project-source-{owner}.json"
    update_only = marker_path.exists()
    kaggle_service.dataset_create_or_update(
        slug,
        source_dir,
        title="MTMC Tracker source",
        description=f"Project source bundle for MTMC Kaggle run {run_id}",
        update_only=update_only,
        username=username,
        key=key,
    )
    marker_path.write_text(
        json.dumps({"slug": slug, "updated_at": _utc_now_iso()}, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return slug


def _resolve_video_dataset(
    *,
    owner: str,
    run_id: str,
    kaggle_cfg: KaggleConfig,
    user_video_path: Optional[Path],
    kaggle_work_dir: Path,
    username: Optional[str],
    key: Optional[str],
    output_root: Path,
) -> str:
    if kaggle_cfg.dataset_slug:
        return kaggle_cfg.dataset_slug
    if user_video_path is None:
        raise ValueError("Need either dataset_slug or user_video_path")

    slug = f"{owner}/mtmc-user-video-{_slug_fragment(run_id)}"
    existing_state = get_kaggle_job_state(run_id, output_root=output_root)
    if existing_state and existing_state.get("dataset_slug") == slug:
        return slug

    video_dir = kaggle_work_dir / "video_dataset"
    if video_dir.exists():
        shutil.rmtree(video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)
    _copy_dataset_input(user_video_path, video_dir)
    kaggle_service.dataset_create_or_update(
        slug,
        video_dir,
        title=f"MTMC user video {run_id}",
        description=f"Auto-uploaded by MTMC backend for run {run_id}",
        username=username,
        key=key,
    )
    return slug


def _kernel_context(
    *,
    run_id: str,
    stages: List[int],
    owner: str,
    config_path: str,
    project_dataset_slug: str,
    dataset_slug: str,
    model_id: Optional[str],
    fusion: Optional[Dict[str, Any]],
    config_overrides: List[str],
) -> kaggle_kernel_templates.KernelTemplateContext:
    if stages == [4]:
        return kaggle_kernel_templates.kernel_for_stage_4(
            run_id=run_id,
            kernel_owner=owner,
            config_path=config_path,
            project_dataset_slug=project_dataset_slug,
            stage2_outputs_dataset_slug=dataset_slug,
            model_id=model_id,
            fusion=fusion,
            config_overrides=config_overrides,
        )
    if all(stage in {0, 1, 2} for stage in stages):
        stage_list = ",".join(str(stage) for stage in stages)
        title_stage_word = "Stages" if len(stages) > 1 else "Stage"
        return kaggle_kernel_templates.KernelTemplateContext(
            run_id=run_id,
            stages=list(stages),
            config_path=config_path,
            dataset_slugs=[project_dataset_slug, dataset_slug],
            project_dataset_slug=project_dataset_slug,
            kernel_owner=owner,
            kernel_title=f"MTMC {run_id} - {title_stage_word} {stage_list}",
            model_id=model_id,
            fusion=fusion,
            config_overrides=list(config_overrides),
            enable_gpu=True,
            enable_internet=False,
        )
    raise KaggleValidationError(
        f"Kaggle dispatch currently supports stages 0, 1, 2, and 4; got {stages}"
    )


def _build_project_source_archive(dest_dir: Path) -> None:
    include_patterns = [
        "backend/**/*.py",
        "src/**/*.py",
        "configs/**/*.yaml",
        "configs/**/*.json",
        "scripts/eval/*.py",
        "requirements*.txt",
        "pyproject.toml",
    ]
    excludes = {".venv", ".git", "data", "__pycache__", "node_modules", ".next"}
    dest_dir.mkdir(parents=True, exist_ok=True)

    files: set[Path] = set()
    for pattern in include_patterns:
        files.update(path for path in _PROJECT_ROOT.glob(pattern) if path.is_file())
    run_pipeline_path = _PROJECT_ROOT / "scripts" / "run_pipeline.py"
    if run_pipeline_path.exists():
        files.add(run_pipeline_path)

    for source_path in sorted(files):
        relative = source_path.relative_to(_PROJECT_ROOT)
        if any(part in excludes for part in relative.parts):
            continue
        if source_path.suffix == ".pyc":
            continue
        target_path = dest_dir / relative
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)


def _copy_dataset_input(source_path: Path, dest_dir: Path) -> None:
    source_path = source_path.resolve()
    if not source_path.exists():
        raise ValueError(f"Input path does not exist: {source_path}")
    if source_path.is_file():
        shutil.copy2(source_path, dest_dir / source_path.name)
        return
    for child in source_path.rglob("*"):
        if not child.is_file():
            continue
        relative = child.relative_to(source_path)
        target = dest_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(child, target)


def _read_rendered_kernel_slug(metadata_dir: Path) -> str:
    metadata_path = metadata_dir / "kernel-metadata.json"
    if not metadata_path.exists():
        return ""
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return str(metadata.get("id") or "")


def _slug_fragment(value: str) -> str:
    slug = "".join(
        char if char.isalnum() or char in {"-", "_"} else "-"
        for char in value.lower()
    )
    return slug.strip("-") or "run"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
