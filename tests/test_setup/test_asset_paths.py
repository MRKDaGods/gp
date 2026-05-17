from pathlib import Path
import importlib.util
import sys

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]

_spec = importlib.util.spec_from_file_location(
    "download_assets", REPO_ROOT / "scripts/download_assets.py"
)
assert _spec is not None and _spec.loader is not None
download_assets = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = download_assets
_spec.loader.exec_module(download_assets)

ASSETS_BY_ID = download_assets.ASSETS_BY_ID
REQUIRED_MODEL_ASSET_IDS = download_assets.REQUIRED_MODEL_ASSET_IDS


def _posix(asset_id: str) -> str:
    return ASSETS_BY_ID[asset_id].final_path.as_posix()


def test_required_model_assets_are_manifested() -> None:
    assert set(REQUIRED_MODEL_ASSET_IDS) == {
        "clipsenet_v6",
        "transreid_veri776_09v",
        "transreid_cityflowv2",
        "mvdetr_wildtrack",
    }


def test_vehicle_config_paths_match_download_manifest() -> None:
    default_cfg = OmegaConf.load(REPO_ROOT / "configs/default.yaml")
    cityflow_cfg = OmegaConf.load(REPO_ROOT / "configs/datasets/cityflowv2.yaml")

    assert default_cfg.stage2.reid.vehicle.weights_path == _posix("transreid_cityflowv2")
    assert default_cfg.stage2.reid.vehicle.weights_fallback == _posix("transreid_veri776_09v")
    assert cityflow_cfg.stage2.reid.vehicle.weights_fallback == _posix("transreid_cityflowv2")


def test_veri776_fusion_paths_match_download_manifest() -> None:
    fusion_text = (REPO_ROOT / "configs/models/veri776_14t_fusion.yaml").read_text(
        encoding="utf-8"
    )

    assert f"checkpoint: {_posix('transreid_veri776_09v')}" in fusion_text
    assert f"checkpoint: {_posix('clipsenet_v6')}" in fusion_text


def test_mvdetr_checkpoint_path_matches_registry_convention() -> None:
    registry_text = (REPO_ROOT / "configs/model_registry.yaml").read_text(encoding="utf-8")
    expected_path = _posix("mvdetr_wildtrack")

    assert expected_path == "models/person_detection/MultiviewDetector.pth"
    assert f"local_path: {expected_path}" in registry_text