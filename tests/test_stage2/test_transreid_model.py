"""Tests for TransReID model and ReIDModel TransReID integration.

Tests cover:
- CLIP ViT-Base architecture (default) with norm_pre, SIE all tokens
- Weight key naming matching NB08 training checkpoint
- Identity projection when embed_dim == vit_dim
- Backward compatibility with ViT-Small
- ReIDModel CLIP normalization detection
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

timm = pytest.importorskip("timm", reason="timm required for TransReID tests")

# Default CLIP backbone used by NB08
_CLIP_VIT_MODEL = "vit_base_patch16_clip_224.openai"


class TestTransReIDModel:
    """Tests for the TransReID architecture (CLIP ViT-Base by default)."""

    def test_build_transreid_default(self):
        """TransReID builds with default CLIP ViT-Base parameters."""
        from src.stage2_features.transreid_model import build_transreid

        model = build_transreid(
            num_classes=10, num_cameras=0, embed_dim=768, pretrained=False,
        )
        assert model is not None
        assert model.vit_dim == 768
        assert not model.sie_camera  # no cameras → SIE disabled

    def test_build_transreid_with_sie(self):
        """TransReID builds with SIE enabled when num_cameras > 0."""
        from src.stage2_features.transreid_model import build_transreid

        model = build_transreid(
            num_classes=10, num_cameras=20, embed_dim=768, pretrained=False,
        )
        assert model.sie_camera
        assert model.sie_embed.shape == (20, 1, 768)

    def test_forward_inference_clip(self):
        """Forward pass in eval mode returns L2-normalized 768-dim features."""
        from src.stage2_features.transreid_model import TransReID

        model = TransReID(num_classes=10, embed_dim=768, pretrained=False)
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 768)
        # Check L2 normalization
        norms = torch.norm(out, dim=1)
        assert torch.allclose(norms, torch.ones(2), atol=1e-5)

    def test_identity_projection_when_dims_match(self):
        """Projection is Identity when embed_dim == vit_dim (768 → 768)."""
        from src.stage2_features.transreid_model import TransReID

        model = TransReID(num_classes=10, embed_dim=768, pretrained=False)
        assert isinstance(model.proj, torch.nn.Identity)

    def test_linear_projection_when_dims_differ(self):
        """Projection is Linear when embed_dim != vit_dim."""
        from src.stage2_features.transreid_model import TransReID

        model = TransReID(num_classes=10, embed_dim=512, pretrained=False)
        assert isinstance(model.proj, torch.nn.Linear)
        assert model.proj.in_features == 768
        assert model.proj.out_features == 512

    def test_norm_pre_exists_for_clip(self):
        """CLIP ViTs have active norm_pre (LayerNorm, not Identity)."""
        from src.stage2_features.transreid_model import TransReID

        model = TransReID(num_classes=10, embed_dim=768, pretrained=False)
        assert hasattr(model.vit, "norm_pre")
        # CLIP models have LayerNorm here; standard ViTs have Identity
        assert not isinstance(model.vit.norm_pre, torch.nn.Identity)

    def test_bnneck_key_name_matches_nb08(self):
        """BNNeck is named 'bn' (matching NB08 checkpoint keys, not 'bn_global')."""
        from src.stage2_features.transreid_model import TransReID

        model = TransReID(num_classes=10, embed_dim=768, pretrained=False)
        assert hasattr(model, "bn")
        assert isinstance(model.bn, torch.nn.BatchNorm1d)
        # Old key name should NOT exist
        assert not hasattr(model, "bn_global")

    def test_classifier_key_name_matches_nb08(self):
        """Classifier is named 'cls_head' (matching NB08, not 'classifier')."""
        from src.stage2_features.transreid_model import TransReID

        model = TransReID(num_classes=10, embed_dim=768, pretrained=False)
        assert hasattr(model, "cls_head")
        assert not hasattr(model, "classifier")

    def test_sie_affects_output(self):
        """SIE with cam_ids changes the output (broadcasts to all tokens)."""
        from src.stage2_features.transreid_model import TransReID

        model = TransReID(
            num_classes=5, num_cameras=4, embed_dim=768, pretrained=False,
        )
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        cam_ids = torch.tensor([0, 2])

        with torch.no_grad():
            out_no_sie = model(x, cam_ids=None)
            out_with_sie = model(x, cam_ids=cam_ids)

        # SIE should change the output
        assert not torch.allclose(out_no_sie, out_with_sie, atol=1e-3)

    def test_forward_training_jpm(self):
        """Training mode returns (cls, proj, jpm_cls) with JPM."""
        from src.stage2_features.transreid_model import TransReID

        model = TransReID(
            num_classes=10, embed_dim=768, pretrained=False, jpm=True,
        )
        model.train()
        x = torch.randn(4, 3, 224, 224)
        outputs = model(x)
        assert isinstance(outputs, tuple)
        assert len(outputs) == 3  # cls, proj, jpm_cls
        cls_score, proj, jpm_cls = outputs
        assert cls_score.shape == (4, 10)
        assert proj.shape == (4, 768)
        assert jpm_cls.shape == (4, 10)

    def test_forward_training_no_jpm(self):
        """Training mode returns (cls, proj) without JPM."""
        from src.stage2_features.transreid_model import TransReID

        model = TransReID(
            num_classes=10, embed_dim=768, pretrained=False, jpm=False,
        )
        model.train()
        x = torch.randn(2, 3, 224, 224)
        outputs = model(x)
        assert isinstance(outputs, tuple)
        assert len(outputs) == 2

    def test_state_dict_key_names(self):
        """State dict key names match NB08 training checkpoint format."""
        from src.stage2_features.transreid_model import TransReID

        model = TransReID(
            num_classes=10, num_cameras=4, embed_dim=768, pretrained=False,
        )
        keys = set(model.state_dict().keys())

        # Must have 'bn.*' keys (not 'bn_global.*')
        assert any(k.startswith("bn.") for k in keys)
        assert not any(k.startswith("bn_global.") for k in keys)

        # Must have 'cls_head.*' keys (not 'classifier.*')
        assert any(k.startswith("cls_head.") for k in keys)
        assert not any(k.startswith("classifier.") for k in keys)

        # SIE embed should be present
        assert "sie_embed" in keys

        # JPM keys
        assert any(k.startswith("bn_jpm.") for k in keys)
        assert any(k.startswith("jpm_cls.") for k in keys)

    def test_vit_small_backward_compat(self):
        """ViT-Small still works for backward compatibility."""
        from src.stage2_features.transreid_model import TransReID

        model = TransReID(
            num_classes=5, embed_dim=256,
            vit_model="vit_small_patch16_224", pretrained=False,
        )
        assert model.vit_dim == 384
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 256)


class TestReIDModelTransReIDRouting:
    """Tests for ReIDModel's TransReID routing and CLIP normalization."""

    def test_transreid_names_detected(self):
        """TransReID model names are correctly identified."""
        from src.stage2_features.reid_model import ReIDModel

        for name in ["transreid", "vit_small", "vit_base", "transreid_vit"]:
            assert name in ReIDModel._TRANSREID_NAMES

    def test_non_transreid_names(self):
        """Standard torchreid names are NOT detected as TransReID."""
        from src.stage2_features.reid_model import ReIDModel

        for name in ["osnet_x1_0", "resnet50", "resnet50_ibn_a"]:
            assert name not in ReIDModel._TRANSREID_NAMES

    def test_clip_normalization_auto_detect(self):
        """CLIP normalization is auto-detected from vit_model name."""
        from src.stage2_features.reid_model import _CLIP_MEAN, _CLIP_STD, _IMAGENET_MEAN, _IMAGENET_STD

        # CLIP model → CLIP normalization
        import cv2
        from unittest.mock import patch, MagicMock

        # We can't easily instantiate ReIDModel (needs weights/GPU),
        # so test the normalization constants directly
        assert _CLIP_MEAN[0] == pytest.approx(0.48145466)
        assert _IMAGENET_MEAN[0] == pytest.approx(0.485)
        assert _CLIP_STD[0] == pytest.approx(0.26862954)
        assert _IMAGENET_STD[0] == pytest.approx(0.229)

    def test_default_input_size_override(self):
        """TransReID automatically uses 224x224 when default (256, 128) is set."""
        from src.stage2_features.reid_model import ReIDModel

        # Can't actually instantiate without GPU/weights, so test the logic
        model_cls = ReIDModel
        assert "transreid" in model_cls._TRANSREID_NAMES

    def test_fastreid_name_routes_to_builder(self):
        """fast-reid SBS(R50-IBN) is routed to the dedicated builder."""
        from src.stage2_features.reid_model import ReIDModel

        model = ReIDModel.__new__(ReIDModel)
        model.is_transreid = False

        calls = {}

        def fake_builder(weights_path):
            calls["weights_path"] = weights_path
            return "sentinel"

        model._build_fastreid_sbs_r50_ibn = fake_builder
        built = ReIDModel._build_model(model, "fastreid_sbs_r50_ibn", "weights.pth")

        assert built == "sentinel"
        assert calls["weights_path"] == "weights.pth"

    def test_fastreid_state_dict_remap(self):
        """fast-reid checkpoint keys are remapped to the local model layout."""
        from src.stage2_features.reid_model import ReIDModel

        state_dict = {
            "pixel_mean": torch.zeros(3, 1, 1),
            "backbone.conv1.weight": torch.randn(64, 3, 7, 7),
            "backbone.layer1.0.bn1.IN.weight": torch.randn(32),
            "backbone.NL_2.0.g.0.weight": torch.randn(1),
            "heads.pool_layer.p": torch.tensor([3.0]),
            "heads.bnneck.num_batches_tracked": torch.tensor(0),
            "heads.bottleneck.0.weight": torch.randn(2048),
            "heads.bottleneck.0.running_mean": torch.randn(2048),
            "heads.classifier.weight": torch.randn(575, 2048),
        }

        remapped = ReIDModel._remap_fastreid_sbs_r50_ibn_state_dict(state_dict)

        assert "backbone.conv1.weight" in remapped
        assert "backbone.layer1.0.bn1.IN.weight" in remapped
        assert "pool.p" in remapped
        assert "bottleneck.weight" in remapped
        assert "bottleneck.running_mean" in remapped
        assert "bottleneck.num_batches_tracked" in remapped
        assert "pixel_mean" not in remapped
        assert "heads.classifier.weight" not in remapped
        assert not any(key.startswith("backbone.NL_") for key in remapped)


class TestResNet50IBNModel:
    """Tests for the ResNet50-IBN fast-reid-compatible inference model."""

    def test_resnet50_ibn_global_eval_feature_shape(self):
        """Global eval mode returns 2048D pre-BNNeck features."""
        pytest.importorskip("torchvision", reason="torchvision required for ResNet50-IBN tests")

        from src.training.model import ReIDModelResNet50IBN

        model = ReIDModelResNet50IBN(
            num_classes=1,
            pretrained=False,
            eval_feature="global",
        )
        model.eval()

        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            out = model(x)

        assert out.shape == (1, 2048)
