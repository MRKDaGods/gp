from __future__ import annotations

import json
from pathlib import Path

import nbformat
import pytest

from backend.services.kaggle_kernel_templates import (
    KernelTemplateContext,
    build_kernel_metadata,
    build_notebook,
    kernel_for_stage_4,
    kernel_for_stages_012,
    render_kernel,
)


def _ctx() -> KernelTemplateContext:
    return kernel_for_stages_012(
        run_id="Run 42_Test",
        kernel_owner="gumfreddy",
        config_path="configs/datasets/cityflowv2.yaml",
        project_dataset_slug="gumfreddy/mtmc-project-src",
        video_dataset_slug="thanhnguyenle/data-aicity-2023-track-2",
        model_id="vehicle-transreid",
        fusion={"enabled": True, "weight": 0.5},
        config_overrides=["stage1.tracker.min_hits=2"],
    )


def _assert_source_newlines(notebook: dict) -> None:
    for cell in notebook["cells"]:
        source = cell["source"]
        assert source
        for line in source[:-1]:
            assert line.endswith("\n")
        assert not source[-1].endswith("\n")


def test_kernel_for_stages_012_builds_gpu_context() -> None:
    ctx = _ctx()

    assert ctx.stages == [0, 1, 2]
    assert ctx.enable_gpu is True
    assert ctx.dataset_slugs == [
        "gumfreddy/mtmc-project-src",
        "thanhnguyenle/data-aicity-2023-track-2",
    ]


def test_kernel_for_stage_4_builds_cpu_context() -> None:
    ctx = kernel_for_stage_4(
        run_id="run-42",
        kernel_owner="gumfreddy",
        config_path="configs/datasets/cityflowv2.yaml",
        project_dataset_slug="gumfreddy/mtmc-project-src",
        stage2_outputs_dataset_slug="gumfreddy/run-42-stage2-outputs",
    )

    assert ctx.stages == [4]
    assert ctx.enable_gpu is False
    assert ctx.dataset_slugs == ["gumfreddy/mtmc-project-src", "gumfreddy/run-42-stage2-outputs"]


def test_build_notebook_produces_seven_cells_with_expected_source_newlines() -> None:
    notebook = build_notebook(_ctx())

    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) == 7
    assert [cell["cell_type"] for cell in notebook["cells"]] == [
        "markdown",
        "code",
        "code",
        "code",
        "code",
        "code",
        "code",
    ]
    assert all("language" in cell["metadata"] for cell in notebook["cells"])
    _assert_source_newlines(notebook)


def test_build_kernel_metadata_uses_context_slug_datasets_and_gpu_flag() -> None:
    ctx = _ctx()
    metadata = build_kernel_metadata(ctx, "gumfreddy/mtmc-run-42-test-stages-012")

    assert metadata["id"] == "gumfreddy/mtmc-run-42-test-stages-012"
    assert metadata["id_no"] is None
    assert metadata["title"] == "MTMC Run 42_Test - Stages 0-2"
    assert metadata["code_file"] == "notebook.ipynb"
    assert metadata["language"] == "python"
    assert metadata["kernel_type"] == "notebook"
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is True
    assert metadata["dataset_sources"] == ctx.dataset_slugs
    assert metadata["competition_sources"] == []
    assert metadata["kernel_sources"] == []


def test_render_kernel_writes_valid_json_and_loadable_nbformat(tmp_path: Path) -> None:
    output_dir = render_kernel(_ctx(), tmp_path / "kernel")
    notebook_path = output_dir / "notebook.ipynb"
    metadata_path = output_dir / "kernel-metadata.json"

    assert notebook_path.exists()
    assert metadata_path.exists()

    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["id"] == "gumfreddy/mtmc-run-42-test-stages-012"
    assert metadata["code_file"] == "notebook.ipynb"
    assert metadata["dataset_sources"] == [
        "gumfreddy/mtmc-project-src",
        "thanhnguyenle/data-aicity-2023-track-2",
    ]

    nbformat.read(str(notebook_path), as_version=4)
    _assert_source_newlines(notebook)


def test_invalid_kernel_owner_raises_value_error() -> None:
    ctx = KernelTemplateContext(
        run_id="run-42",
        stages=[4],
        config_path="configs/datasets/cityflowv2.yaml",
        dataset_slugs=["gumfreddy/mtmc-project-src"],
        project_dataset_slug="gumfreddy/mtmc-project-src",
        kernel_owner="Gum_Freddy",
        kernel_title="Invalid owner",
    )

    with pytest.raises(ValueError, match="kernel_owner"):
        build_notebook(ctx)


def test_empty_stages_raises_value_error() -> None:
    ctx = KernelTemplateContext(
        run_id="run-42",
        stages=[],
        config_path="configs/datasets/cityflowv2.yaml",
        dataset_slugs=["gumfreddy/mtmc-project-src"],
        project_dataset_slug="gumfreddy/mtmc-project-src",
        kernel_owner="gumfreddy",
        kernel_title="Empty stages",
    )

    with pytest.raises(ValueError, match="stages"):
        build_notebook(ctx)


def test_invalid_dataset_slug_raises_value_error() -> None:
    ctx = KernelTemplateContext(
        run_id="run-42",
        stages=[4],
        config_path="configs/datasets/cityflowv2.yaml",
        dataset_slugs=["not-a-dataset-slug"],
        project_dataset_slug="not-a-dataset-slug",
        kernel_owner="gumfreddy",
        kernel_title="Invalid dataset",
    )

    with pytest.raises(ValueError, match="Invalid dataset slug"):
        build_notebook(ctx)


def test_rendered_notebook_final_cell_last_source_line_has_no_trailing_newline(
    tmp_path: Path,
) -> None:
    output_dir = render_kernel(_ctx(), tmp_path / "kernel")
    notebook = json.loads((output_dir / "notebook.ipynb").read_text(encoding="utf-8"))

    final_cell_source = notebook["cells"][-1]["source"]
    assert final_cell_source[-1] == "print(json.dumps(manifest, indent=2))"
    assert not final_cell_source[-1].endswith("\n")
