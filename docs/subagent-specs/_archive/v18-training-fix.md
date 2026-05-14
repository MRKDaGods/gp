# v18 ResNet101-IBN-a Training Fix

## Root Cause
v17 mAP=29.62% (down from v16's 50.61%) due to:
1. lr=3.5e-4 (ViT recipe, not ResNet) — backbone gets 3.5e-5, barely adapts
2. circle_weight=0.5 — gradient conflict with triplet loss on same features

## Approach A (ali369): Revert to v16 + batch=64 + epochs=150
CFG changes: lr=1e-3, warmup=5, wd=5e-4, label_smoothing=0.05, circle_weight=0.0, epochs=150

## Approach B (mrkdagods): SGD lr=0.008
CFG changes: optimizer=sgd, lr=0.008, momentum=0.9, warmup=10, wd=5e-4, label_smoothing=0.1, circle_weight=0.0, epochs=150
Plus: optimizer cell needs SGD branch

## Success Gate: mAP >= 50% (matches v16)