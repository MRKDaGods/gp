# Post-384px Strategy: Closing the 6.5pp Gap to SOTA

**Date**: 2026-03-30
**Current Best**: MTMC IDF1 = 78.4% (256px ViT-B/16 CLIP, v80)
**SOTA Target**: MTMC IDF1 ≈ 84.86% (AIC22 1st place)
**Gap**: 6.46pp

## 384px Result (Definitive Dead End)

| Run | Model | min_hits | MTMC IDF1 |
|-----|:-----:|:--------:|:---------:|
| v80 baseline | 256px | 2 | **0.784** |
| v43 (384px, tuned thresholds) | 384px | 3 | 0.7585 |
| v44 (384px, exact v43 config) | 384px | 2 | 0.7562 |

384px is **conclusively worse** (-2.8pp) despite higher single-camera mAP (80.14% vs ~70%).
Root cause: higher resolution captures viewpoint-specific textures that hurt cross-camera matching.

## Prioritized Action Plan

| Priority | Approach | Expected Impact | Effort |
|:--------:|---------|:--------------:|:------:|
| **1** | CID_BIAS on 256px features | +0.5-1.0pp | Low (CPU-only) |
| **2** | Camera-aware training (DMT + circle loss) | +1.5-2.5pp | High (retrain) |
| **3** | Multi-query track matching | +0.3-0.8pp | Medium |

### Phase 1: CID_BIAS (IMMEDIATE)
- Code exists: scripts/compute_cid_bias.py + stage4 loading at L383-415
- Compute per-camera-pair additive similarity bias from GT associations
- Expected: 78.4% → 78.9-79.4%

### Phase 2: Camera-Aware Retrain (DMT)
- Add camera-domain adversarial loss + separate-head circle loss to 09b notebook
- Full Kaggle retrain (6-12h P100)
- Expected: +1.5-2.5pp cumulative

### Phase 3: Multi-Query Matching
- Save top-K diverse embeddings per track in stage 2
- Use max-of-pairs similarity in stage 4
- Expected: +0.3-0.8pp

## Realistic Ceiling: 81-83% MTMC IDF1

Without multi-model ensemble (the main SOTA differentiator), 84.86% is not achievable.