# 384px Model Regression Fix — Root Cause Analysis & Experiment Plan

> **Status**: INVESTIGATION  
> **Priority**: P0 — blocking deployment of 384px model
> **Created**: 2026-03-27

## Problem Statement
384px TransReID ViT (mAP=80.14%) deployed replacing 256px (~75% mAP). Despite higher mAP, MTMC IDF1 dropped from 0.784 to 0.769 (-1.5pp).

## Root Cause Analysis (Ranked)

1. **Secondary Model Dilution (70%)**: 40% fusion weight on 52.77% ResNet degrades stronger 384px signal
2. **Double Camera Normalization (50%)**: Camera BN + FIC over-corrects for more camera-invariant features
3. **PCA Whitening (40%)**: Amplifies noise dimensions that were negligible with 256px
4. **FIC Reg Mismatch (30%)**: 0.1 tuned for 256px covariance structure
5. **ViT Structural (15%)**: 576 patches vs 256 diffuses CLS token

## Experiments (Priority Order)
- Exp1: secondary_embeddings.weight=0.0 (10c only, NO GPU)
- Exp2: secondary weight 0.10/0.15/0.20 sweep (10c only)
- Exp3: camera_bn.enabled=false (10a GPU needed)
- Exp4: pca.whiten=false (code change + 10a)
- Exp5: FIC reg sweep 0.3/0.5/1.0/3.0 (10c only)

## Experiment Log
- Exp1 result: PENDING