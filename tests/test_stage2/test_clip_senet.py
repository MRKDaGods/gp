from __future__ import annotations

import importlib.util

import pytest
import torch


def _build_or_skip():
    if importlib.util.find_spec("timm") is None:
        pytest.skip("skipped: timm is not installed")
    if importlib.util.find_spec("open_clip") is None:
        pytest.skip("skipped: no network or missing dependency (open_clip not installed)")

    from src.stage2_features.clip_senet_model import build_clip_senet

    try:
        return build_clip_senet(num_classes=576)
    except ImportError as exc:
        pytest.skip(f"skipped: {exc}")
    except Exception as exc:  # noqa: BLE001 - convert download failures into clear skips
        message = str(exc).lower()
        network_markers = (
            "temporary failure in name resolution",
            "name or service not known",
            "connection",
            "download",
            "timed out",
            "http error",
            "certificate",
            "urlopen",
            "remote disconnected",
        )
        if any(marker in message for marker in network_markers):
            pytest.skip(f"skipped: no network ({exc})")
        raise


@pytest.fixture(scope="module")
def clip_senet_model():
    return _build_or_skip()


def test_afem_block():
    from src.stage2_features.clip_senet_model import AFEMBlock

    block = AFEMBlock(in_dim=2048, out_dim=2048, num_groups=32)
    x = torch.randn(4, 2048)
    out = block(x)
    assert out.shape == (4, 2048)


@pytest.mark.slow
@pytest.mark.requires_network
def test_forward_shapes(clip_senet_model):
    model = clip_senet_model
    x = torch.randn(2, 3, 320, 320)

    model.train()
    feat, logits = model(x)
    assert feat.shape == (2, 2048)
    assert logits.shape == (2, 576)

    model.eval()
    with torch.no_grad():
        feat_eval = model(x)
    assert feat_eval.shape == (2, 2048)


@pytest.mark.slow
@pytest.mark.requires_network
def test_param_count(clip_senet_model):
    param_count = sum(parameter.numel() for parameter in clip_senet_model.parameters())
    assert 80_000_000 <= param_count <= 150_000_000


@pytest.mark.slow
@pytest.mark.requires_network
def test_eval_mode_no_logits(clip_senet_model):
    model = clip_senet_model
    model.eval()
    x = torch.randn(2, 3, 320, 320)
    with torch.no_grad():
        out = model(x)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 2048)
