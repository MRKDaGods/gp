"""BoT ReID trainer: Train strong ReID models with Bag-of-Tricks recipe.

Usage:
    python -m src.training.train_reid --dataset market1501 --root data/raw/market1501
    python -m src.training.train_reid --dataset veri776 --root data/raw/veri776 --height 224 --width 224
    python -m src.training.train_reid --dataset market1501 --root data/raw/market1501 --loss circle

Implements:
    - ResNet50-IBN-a with last_stride=1
    - BNNeck (batch normalization neck)
    - ID loss (cross-entropy with label smoothing) + Triplet loss (hard mining)
    - Optional: Center loss, Circle loss
    - Warm-up learning rate schedule
    - Random erasing augmentation
    - Mixed precision training (fp16/bf16)
    - Evaluation with re-ranking
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from src.training.model import ReIDModelBoT
from src.training.losses import (
    CrossEntropyLabelSmooth,
    TripletLoss,
    CenterLoss,
    CircleLoss,
)
from src.training.datasets import build_dataloader
from src.training.evaluate_reid import evaluate_reid, compute_reranking


def build_optimizer(
    model: ReIDModelBoT,
    center_loss: CenterLoss | None,
    lr: float = 3.5e-4,
    weight_decay: float = 5e-4,
    center_lr: float = 0.5,
) -> tuple:
    """Build optimizers for model and center loss."""
    params = [
        {"params": model.backbone.parameters(), "lr": lr * 0.1},  # lower LR for backbone
        {"params": model.bottleneck.parameters(), "lr": lr},
        {"params": model.classifier.parameters(), "lr": lr},
    ]
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    center_optimizer = None
    if center_loss is not None:
        center_optimizer = torch.optim.SGD(
            center_loss.parameters(), lr=center_lr
        )

    return optimizer, center_optimizer


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int = 10,
    total_epochs: int = 120,
    milestones: list | None = None,
    gamma: float = 0.1,
) -> torch.optim.lr_scheduler._LRScheduler:
    """Build LR scheduler with linear warmup."""
    if milestones is None:
        milestones = [40, 70]

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        for i, m in enumerate(milestones):
            if epoch < m:
                return gamma ** i
        return gamma ** len(milestones)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(
    model: ReIDModelBoT,
    train_loader,
    id_loss_fn,
    triplet_loss_fn,
    center_loss_fn,
    optimizer,
    center_optimizer,
    scaler,
    device: str,
    epoch: int,
    id_weight: float = 1.0,
    triplet_weight: float = 1.0,
    center_weight: float = 0.0005,
):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    running_id = 0.0
    running_tri = 0.0
    running_cen = 0.0
    n_batches = 0

    for batch_idx, (imgs, pids, _, _) in enumerate(train_loader):
        imgs = imgs.to(device)
        pids = pids.to(device).long()

        optimizer.zero_grad()
        if center_optimizer:
            center_optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=scaler is not None):
            cls_score, global_feat, bn_feat = model(imgs)

            loss_id = id_loss_fn(cls_score, pids) * id_weight
            loss_tri = triplet_loss_fn(global_feat, pids) * triplet_weight

            loss = loss_id + loss_tri

            if center_loss_fn is not None:
                loss_cen = center_loss_fn(global_feat, pids) * center_weight
                loss += loss_cen
            else:
                loss_cen = torch.tensor(0.0)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if center_optimizer is not None:
            # Center loss has its own optimizer
            for param in center_loss_fn.parameters():
                param.grad.data *= (1.0 / center_weight)
            center_optimizer.step()

        running_loss += loss.item()
        running_id += loss_id.item()
        running_tri += loss_tri.item()
        running_cen += loss_cen.item() if isinstance(loss_cen, torch.Tensor) else loss_cen
        n_batches += 1

        if batch_idx % 50 == 0:
            logger.info(
                f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f} "
                f"(ID: {loss_id.item():.4f}, Tri: {loss_tri.item():.4f}, "
                f"Cen: {loss_cen.item() if isinstance(loss_cen, torch.Tensor) else 0:.4f})"
            )

    return {
        "loss": running_loss / max(n_batches, 1),
        "id_loss": running_id / max(n_batches, 1),
        "tri_loss": running_tri / max(n_batches, 1),
        "cen_loss": running_cen / max(n_batches, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="BoT ReID Training")
    parser.add_argument("--dataset", type=str, default="market1501",
                        choices=["market1501", "veri776", "msmt17"])
    parser.add_argument("--root", type=str, required=True,
                        help="Path to dataset root")
    parser.add_argument("--output-dir", type=str, default="models/reid/trained",
                        help="Output directory for checkpoints")
    parser.add_argument("--backbone", type=str, default="resnet50_ibn_a")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-instances", type=int, default=4,
                        help="Number of instances per identity in batch")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=3.5e-4)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--milestones", type=int, nargs="+", default=[40, 70])
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--random-erasing", type=float, default=0.5)
    parser.add_argument("--last-stride", type=int, default=1)
    parser.add_argument("--loss", type=str, default="triplet",
                        choices=["triplet", "circle", "triplet+center"])
    parser.add_argument("--triplet-margin", type=float, default=0.3)
    parser.add_argument("--center-weight", type=float, default=0.0005)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--no-fp16", action="store_false", dest="fp16")
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--rerank", action="store_true", default=False,
                        help="Use re-ranking during evaluation")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--pretrained-reid", type=str, default=None,
                        help="Path to pre-trained ReID weights to init from")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if torch.cuda.is_available() else "cpu"

    # Build data
    logger.info(f"Loading dataset: {args.dataset} from {args.root}")
    train_loader, query_loader, gallery_loader, num_classes, num_cameras = (
        build_dataloader(
            dataset_name=args.dataset,
            root=args.root,
            height=args.height,
            width=args.width,
            batch_size=args.batch_size,
            num_instances=args.num_instances,
            num_workers=args.num_workers,
            random_erasing_prob=args.random_erasing,
        )
    )
    logger.info(
        f"Dataset: {args.dataset}, classes={num_classes}, cameras={num_cameras}, "
        f"train batches={len(train_loader)}"
    )

    # Build model
    model = ReIDModelBoT(
        model_name=args.backbone,
        num_classes=num_classes,
        last_stride=args.last_stride,
        pretrained=True,
        feat_dim=2048 if "resnet50" in args.backbone else 512,
        neck="bnneck",
    ).to(device)

    if args.pretrained_reid:
        model.load_pretrained_reid(args.pretrained_reid)

    # Build losses
    id_loss_fn = CrossEntropyLabelSmooth(num_classes, epsilon=args.label_smoothing)

    if args.loss == "circle":
        triplet_loss_fn = CircleLoss(m=0.25, gamma=64)
    else:
        triplet_loss_fn = TripletLoss(margin=args.triplet_margin)

    center_loss_fn = None
    if "center" in args.loss:
        feat_dim = 2048 if "resnet50" in args.backbone else 512
        center_loss_fn = CenterLoss(num_classes=num_classes, feat_dim=feat_dim).to(device)

    # Build optimizer and scheduler
    optimizer, center_optimizer = build_optimizer(
        model, center_loss_fn, lr=args.lr, center_lr=0.5
    )
    scheduler = build_scheduler(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        milestones=args.milestones,
    )

    # Mixed precision
    scaler = torch.amp.GradScaler("cuda") if args.fp16 and "cuda" in device else None

    # Resume
    start_epoch = 0
    best_mAP = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_mAP = ckpt.get("best_mAP", 0.0)
        logger.info(f"Resumed from epoch {start_epoch}, best_mAP={best_mAP:.4f}")

    # Training loop
    logger.info(f"Starting training for {args.epochs} epochs")
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_metrics = train_one_epoch(
            model, train_loader, id_loss_fn, triplet_loss_fn,
            center_loss_fn, optimizer, center_optimizer, scaler,
            device, epoch,
            center_weight=args.center_weight,
        )
        scheduler.step()

        elapsed = time.time() - t0
        lr_current = optimizer.param_groups[0]["lr"]
        logger.info(
            f"Epoch {epoch}/{args.epochs} done in {elapsed:.1f}s — "
            f"Loss: {train_metrics['loss']:.4f}, LR: {lr_current:.6f}"
        )

        # Evaluate
        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            logger.info(f"Evaluating at epoch {epoch}...")
            mAP, cmc, mAP_rr, cmc_rr = evaluate_reid(
                model, query_loader, gallery_loader, device,
                rerank=args.rerank,
            )

            is_best = mAP > best_mAP
            if is_best:
                best_mAP = mAP

            logger.info(
                f"  mAP: {mAP:.4f}, R1: {cmc[0]:.4f}, R5: {cmc[4]:.4f}"
            )
            if args.rerank and mAP_rr is not None:
                logger.info(
                    f"  mAP(RR): {mAP_rr:.4f}, R1(RR): {cmc_rr[0]:.4f}"
                )

            # Save checkpoint
            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "mAP": mAP,
                "best_mAP": best_mAP,
            }
            torch.save(ckpt, output_dir / f"checkpoint_epoch{epoch}.pth")

            if is_best:
                torch.save(ckpt, output_dir / "best_model.pth")
                logger.info(f"  ★ New best mAP: {best_mAP:.4f}")

                # Also save just the model weights for deployment
                torch.save(
                    {"state_dict": model.state_dict()},
                    output_dir / f"reid_{args.dataset}_{args.backbone}_best.pth",
                )

    logger.info(f"Training complete! Best mAP: {best_mAP:.4f}")
    logger.info(f"Checkpoints saved to {output_dir}")


if __name__ == "__main__":
    main()
