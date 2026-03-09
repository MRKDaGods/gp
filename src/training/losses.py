"""ReID training losses: Triplet, Center, Circle, and Label Smoothing CE.

Implements the "Bag of Tricks" (BoT) loss functions for training strong
ReID models on Market-1501, VeRi-776, and other datasets.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLabelSmooth(nn.Module):
    """Cross-entropy loss with label smoothing.

    Reference: Szegedy et al., "Rethinking the Inception Architecture" (CVPR 2016).
    """

    def __init__(self, num_classes: int, epsilon: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = self.logsoftmax(inputs)
        targets_one_hot = torch.zeros_like(log_probs).scatter_(
            1, targets.unsqueeze(1), 1
        )
        targets_smooth = (
            (1 - self.epsilon) * targets_one_hot
            + self.epsilon / self.num_classes
        )
        loss = (-targets_smooth * log_probs).mean(0).sum()
        return loss


class TripletLoss(nn.Module):
    """Hard-mining triplet loss with optional soft margin.

    Mines the hardest positive and hardest negative within each batch
    for every anchor. Uses margin-based or soft-margin formulation.

    Reference: Hermans et al., "In Defense of the Triplet Loss for Person
    Re-Identification" (arXiv 2017).
    """

    def __init__(self, margin: float = 0.3, soft_margin: bool = False):
        super().__init__()
        self.margin = margin
        self.soft_margin = soft_margin
        if not soft_margin:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute triplet loss.

        Args:
            inputs: Feature matrix (batch_size, feat_dim).
            targets: Ground truth labels (batch_size,).
        """
        n = inputs.size(0)
        # Pairwise distance
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()

        # Hard mining
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            # Hardest positive: max distance among same-ID samples
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            # Hardest negative: min distance among different-ID samples
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        y = torch.ones_like(dist_an)
        if self.soft_margin:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss


class CenterLoss(nn.Module):
    """Center loss for discriminative feature learning.

    Learns a center for each class and penalizes the distance between
    features and their corresponding class center.

    Reference: Wen et al., "A Discriminative Feature Learning Approach
    for Deep Face Recognition" (ECCV 2016).
    """

    def __init__(self, num_classes: int, feat_dim: int):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        # Expand and compute distances
        distmat = (
            torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes)
            + torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        )
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_classes, device=x.device).long()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        return loss


class CircleLoss(nn.Module):
    """Circle loss: A unified perspective of pair similarity optimization.

    Reference: Sun et al., "Circle Loss" (CVPR 2020).
    """

    def __init__(self, m: float = 0.25, gamma: float = 64):
        super().__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute circle loss using cosine similarity."""
        # L2 normalize features
        inputs = F.normalize(inputs, p=2, dim=1)
        n = inputs.size(0)

        # Cosine similarity
        sim_mat = torch.matmul(inputs, inputs.t())
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())

        pos_pair = []
        neg_pair = []
        for i in range(n):
            pos_pair.append(sim_mat[i][mask[i]])
            neg_pair.append(sim_mat[i][mask[i] == 0])

        # Weighting
        Op = 1 + self.m
        On = -self.m
        delta_p = 1 - self.m
        delta_n = self.m

        loss = 0
        for i in range(n):
            ap = torch.clamp_min(-pos_pair[i].detach() + Op, min=0.0)
            an = torch.clamp_min(neg_pair[i].detach() + On, min=0.0)

            logit_p = -ap * (pos_pair[i] - delta_p) * self.gamma
            logit_n = an * (neg_pair[i] - delta_n) * self.gamma

            loss += self.soft_plus(
                torch.logsumexp(logit_n, dim=0) + torch.logsumexp(-logit_p, dim=0)
            )
        loss /= n
        return loss
