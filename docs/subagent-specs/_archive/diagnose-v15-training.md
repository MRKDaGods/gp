# v15 Training Recipe Fix (v16)

## Problem: v15 mAP declining (16.4% -> 15.2% over 29 epochs)

## Root Causes (6 found):
1. CRITICAL: Adam with L2 reg instead of AdamW -> destroys pretrained backbone features
2. CRITICAL: Circle + Triplet loss conflict on same features -> conflicting gradients
3. HIGH: LR 3x too low (3.5e-4 vs 09b's 1e-3) -> backbone barely adapts
4. MEDIUM: Only 8 IDs/batch (batch_size=32, K=4) -> poor hard negative mining
5. LOW: Label smoothing 0.1 too aggressive -> slower convergence
6. LOW: eval_every=20 too sparse -> can't see peak mAP

## Fix (v16 -> all changes at once):
- Adam -> AdamW
- circle_weight: 1.0 -> 0.0
- lr: 3.5e-4 -> 1e-3
- batch_size: 32 -> 48 (P=12 IDs/batch)
- warmup_epochs: 10 -> 5
- label_smoothing: 0.1 -> 0.05
- eval_every: 20 -> 5
- epochs: 60 -> 120

## Expected: mAP 22-30% with increasing trajectory (not declining)
## Target: mAP > 25% for ensemble contribution