"""Checkpoint-contract tests for the ported TransReID (v1 parity-critical).

The v1 checkpoints encode a precise state-dict key contract (``bn.*``,
``cls_head.*``, ``sie_embed``, norm_pre) that ``build_transreid`` remaps and
validates. These tests prove the frozen checkpoints load into the PORTED
module exactly as they did in v1: no critical missing keys, only the two
expected classifier-head drops, and a unit-norm 768-d embedding out.

Skipped automatically when torch/timm or the local checkpoints are absent
(CI without model assets); they are REQUIRED locally before any Phase 2
porting is trusted — see ROADMAP Phase 0/2.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from athar.components.embedders.transreid_model import build_transreid  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
VERI_CKPT = REPO_ROOT / "models" / "reid" / "vehicle_transreid_vit_base_veri776.pth"
CITYFLOW_CKPT = REPO_ROOT / "models" / "reid" / "transreid_cityflowv2_best.pth"

# Exact v1 build recipes:
#  - VeRi:     scripts/eval/eval_09v_transreid_veri776.py (IMG_SIZE, SIE_NUM_CAMERAS)
#  - CityFlow: configs/default.yaml stage2.reid.vehicle (input_size, num_cameras)
RECIPES = {
    "veri776": dict(
        checkpoint=VERI_CKPT,
        num_cameras=20,
        img_size=(224, 224),
    ),
    "cityflowv2": dict(
        checkpoint=CITYFLOW_CKPT,
        num_cameras=59,
        img_size=(256, 256),
    ),
}


def _build(recipe: dict, caplog):
    with caplog.at_level(logging.DEBUG, logger="athar.components.embedders.transreid_model"):
        model = build_transreid(
            num_classes=1,
            num_cameras=recipe["num_cameras"],
            embed_dim=768,
            vit_model="vit_base_patch16_clip_224.openai",
            pretrained=False,
            weights_path=str(recipe["checkpoint"]),
            img_size=recipe["img_size"],
        )
    return model


@pytest.mark.parametrize("name", sorted(RECIPES))
def test_checkpoint_loads_with_v1_key_contract(name, caplog):
    recipe = RECIPES[name]
    if not recipe["checkpoint"].is_file():
        pytest.skip(f"checkpoint not on disk: {recipe['checkpoint']}")

    model = _build(recipe, caplog)

    text = caplog.text
    assert "critical missing keys" not in text, f"key contract broken:\n{text}"
    assert "Loaded TransReID weights" in text
    # The ONLY tolerated drops are the training-time classifier heads.
    dropped = [line for line in text.splitlines() if "Dropping key" in line]
    for line in dropped:
        assert "cls_head" in line or "jpm_cls" in line, f"unexpected dropped key: {line}"

    model.eval()
    h, w = recipe["img_size"]
    with torch.no_grad():
        feats = model(torch.randn(2, 3, h, w))
    assert feats.shape == (2, 768)
    norms = feats.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4), (
        "inference output must be L2-normalized"
    )
