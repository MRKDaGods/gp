"""Tests for ReID training losses."""

from __future__ import annotations

import pytest
import torch

from src.training.losses import (
    CenterLoss,
    CircleLoss,
    CrossEntropyLabelSmooth,
    TripletLoss,
)


class TestCrossEntropyLabelSmooth:
    """Tests for label-smoothed cross-entropy loss."""

    def test_output_scalar(self):
        """Loss returns a scalar tensor."""
        loss_fn = CrossEntropyLabelSmooth(num_classes=10, epsilon=0.1)
        inputs = torch.randn(8, 10)
        targets = torch.randint(0, 10, (8,))
        loss = loss_fn(inputs, targets)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_zero_smoothing_equals_ce(self):
        """With epsilon=0, should approximate standard cross-entropy."""
        loss_smooth = CrossEntropyLabelSmooth(num_classes=5, epsilon=0.0)
        inputs = torch.randn(4, 5)
        targets = torch.randint(0, 5, (4,))
        loss = loss_smooth(inputs, targets)
        # Standard CE for comparison
        ce_loss = torch.nn.functional.cross_entropy(inputs, targets)
        assert abs(loss.item() - ce_loss.item()) < 0.01

    def test_smoothing_increases_loss(self):
        """Label smoothing generally increases loss on correct predictions."""
        inputs = torch.zeros(4, 5)
        targets = torch.arange(4) % 5
        # Perfect predictions (one-hot logits)
        for i in range(4):
            inputs[i, targets[i]] = 10.0

        loss_no_smooth = CrossEntropyLabelSmooth(num_classes=5, epsilon=0.0)
        loss_smooth = CrossEntropyLabelSmooth(num_classes=5, epsilon=0.1)
        assert loss_smooth(inputs, targets).item() > loss_no_smooth(inputs, targets).item()

    def test_gradient_flows(self):
        """Gradients flow through the loss."""
        loss_fn = CrossEntropyLabelSmooth(num_classes=10)
        inputs = torch.randn(8, 10, requires_grad=True)
        targets = torch.randint(0, 10, (8,))
        loss = loss_fn(inputs, targets)
        loss.backward()
        assert inputs.grad is not None


class TestTripletLoss:
    """Tests for hard-mining triplet loss."""

    def _make_batch(self, p=4, k=4, feat_dim=128):
        """Create a PK-style batch with P identities, K samples each."""
        features = []
        labels = []
        for pid in range(p):
            center = torch.randn(feat_dim) * 5
            for _ in range(k):
                features.append(center + torch.randn(feat_dim) * 0.1)
                labels.append(pid)
        return torch.stack(features), torch.tensor(labels)

    def test_output_scalar(self):
        features, labels = self._make_batch()
        loss_fn = TripletLoss(margin=0.3)
        loss = loss_fn(features, labels)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_zero_loss_perfect_separation(self):
        """Well-separated clusters should yield near-zero loss."""
        feat_dim = 64
        features = []
        labels = []
        for pid in range(3):
            center = torch.zeros(feat_dim)
            center[pid * 20 : (pid + 1) * 20] = 10.0  # Very separated
            for _ in range(4):
                features.append(center + torch.randn(feat_dim) * 0.01)
                labels.append(pid)
        features = torch.stack(features)
        labels = torch.tensor(labels)
        loss_fn = TripletLoss(margin=0.3)
        loss = loss_fn(features, labels)
        assert loss.item() < 0.1

    def test_soft_margin(self):
        """Soft margin variant runs without error."""
        features, labels = self._make_batch()
        loss_fn = TripletLoss(margin=0.3, soft_margin=True)
        loss = loss_fn(features, labels)
        assert loss.dim() == 0

    def test_gradient_flows(self):
        features, labels = self._make_batch()
        features.requires_grad_(True)
        loss_fn = TripletLoss(margin=0.3)
        loss = loss_fn(features, labels)
        loss.backward()
        assert features.grad is not None


class TestCenterLoss:
    """Tests for center loss."""

    def test_output_scalar(self):
        loss_fn = CenterLoss(num_classes=10, feat_dim=128)
        x = torch.randn(8, 128)
        labels = torch.randint(0, 10, (8,))
        loss = loss_fn(x, labels)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_centers_are_parameters(self):
        loss_fn = CenterLoss(num_classes=5, feat_dim=64)
        assert loss_fn.centers.shape == (5, 64)
        assert loss_fn.centers.requires_grad

    def test_gradient_flows_to_centers(self):
        loss_fn = CenterLoss(num_classes=5, feat_dim=32)
        x = torch.randn(4, 32)
        labels = torch.randint(0, 5, (4,))
        loss = loss_fn(x, labels)
        loss.backward()
        assert loss_fn.centers.grad is not None


class TestCircleLoss:
    """Tests for circle loss."""

    def test_output_scalar(self):
        loss_fn = CircleLoss(m=0.25, gamma=64)
        x = torch.randn(8, 128)
        labels = torch.randint(0, 4, (8,))
        # Ensure at least 2 samples per class
        labels[:2] = 0
        labels[2:4] = 1
        loss = loss_fn(x, labels)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_gradient_flows(self):
        loss_fn = CircleLoss()
        x = torch.randn(8, 64, requires_grad=True)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        loss = loss_fn(x, labels)
        loss.backward()
        assert x.grad is not None
