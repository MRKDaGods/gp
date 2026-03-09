"""Evaluate existing ReID models on Market-1501 and VeRi-776.

Quick script to establish baseline numbers with our current pre-trained
weights before training with BoT recipe.

Usage:
    python scripts/eval_reid_baseline.py --dataset market1501 --root data/raw/market1501
    python scripts/eval_reid_baseline.py --dataset veri776 --root data/raw/veri776 --height 224 --width 224
    python scripts/eval_reid_baseline.py --dataset market1501 --root data/raw/market1501 --rerank
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.model import ReIDModelBoT
from src.training.datasets import build_dataloader
from src.training.evaluate_reid import evaluate_reid


def main():
    parser = argparse.ArgumentParser(description="Evaluate ReID baseline models")
    parser.add_argument("--dataset", type=str, default="market1501",
                        choices=["market1501", "veri776", "msmt17"])
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to model weights. If not given, uses default for dataset.")
    parser.add_argument("--backbone", type=str, default="resnet50_ibn_a")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--rerank", action="store_true", default=False)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    # Default weights
    if args.weights is None:
        defaults = {
            "market1501": "models/reid/person_resnet50ibn_market1501.pth",
            "veri776": "models/reid/vehicle_resnet50ibn_veri776.pth",
        }
        args.weights = defaults.get(args.dataset)

    # Load data
    logger.info(f"Loading {args.dataset} from {args.root}")
    _, query_loader, gallery_loader, num_classes, num_cameras = build_dataloader(
        dataset_name=args.dataset,
        root=args.root,
        height=args.height,
        width=args.width,
        batch_size=args.batch_size,
        num_workers=4,
    )

    # Build model (eval mode, no training head needed but we need BNNeck)
    feat_dim = 2048 if "resnet50" in args.backbone else 512
    model = ReIDModelBoT(
        model_name=args.backbone,
        num_classes=num_classes,
        last_stride=1,
        pretrained=True,
        feat_dim=feat_dim,
        neck="bnneck",
    ).to(device)

    # Load weights
    if args.weights and Path(args.weights).exists():
        logger.info(f"Loading weights from {args.weights}")
        model.load_pretrained_reid(args.weights)
    else:
        logger.warning(f"No weights found at {args.weights}, using ImageNet pretrained")

    # Evaluate
    mAP, cmc, mAP_rr, cmc_rr = evaluate_reid(
        model, query_loader, gallery_loader, device,
        rerank=args.rerank,
    )

    print("\n" + "=" * 60)
    print(f"  Dataset:  {args.dataset}")
    print(f"  Backbone: {args.backbone}")
    print(f"  Weights:  {args.weights}")
    print("=" * 60)
    print(f"  mAP:    {mAP * 100:.2f}%")
    print(f"  Rank-1: {cmc[0] * 100:.2f}%")
    print(f"  Rank-5: {cmc[4] * 100:.2f}%")
    print(f"  Rank-10:{cmc[9] * 100:.2f}%")
    if args.rerank and mAP_rr is not None:
        print("  --- with Re-ranking ---")
        print(f"  mAP:    {mAP_rr * 100:.2f}%")
        print(f"  Rank-1: {cmc_rr[0] * 100:.2f}%")
        print(f"  Rank-5: {cmc_rr[4] * 100:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
