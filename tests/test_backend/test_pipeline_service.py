from backend.services.pipeline_service import _build_pipeline_cmd, resolve_pipeline_config_path


def test_resolve_pipeline_config_path_defaults_to_existing_behavior() -> None:
    assert resolve_pipeline_config_path(None) == "configs/default.yaml"
    assert resolve_pipeline_config_path("default") == "configs/default.yaml"


def test_resolve_pipeline_config_path_accepts_known_datasets() -> None:
    assert resolve_pipeline_config_path("cityflowv2") == "configs/datasets/cityflowv2.yaml"
    assert resolve_pipeline_config_path("wildtrack") == "configs/datasets/wildtrack.yaml"


def test_build_pipeline_cmd_uses_dataset_config_selector() -> None:
    cmd = _build_pipeline_cmd(
        stages="0",
        run_id="run-test",
        input_dir="data/raw/wildtrack/videos",
        use_cpu=True,
        dataset="wildtrack",
    )

    assert cmd[cmd.index("--config") + 1] == "configs/datasets/wildtrack.yaml"
    assert "project.run_name='run-test'" in cmd