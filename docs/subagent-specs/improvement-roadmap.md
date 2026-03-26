# Improvement Roadmap: IDF1 77.5% → 84.86% SOTA

**Date**: 2025-03-25
**Current Best**: IDF1 = 77.5% (384px v2 ViT-Base/16 CLIP, mAP=80.14%)
**SOTA Target**: IDF1 = 84.86%
**Gap**: 7.35pp

## Prioritized Approaches

| Rank | Approach | Est. IDF1 Gain | Complexity | Status |
|:---:|---|:---:|:---:|---|
| 1 | Enable CamTTA | +0.3-0.7pp | Trivial | TODO |
| 2 | ResNet101-IBN-a ensemble | +1.0-2.0pp | Low | 09d v9 training |
| 3 | Re-test reranking w/ ensemble | +0.3-0.8pp | Low | Blocked on #2 |
| 4 | DMT camera-aware training | +1.0-1.5pp | Medium | Not started |
| 5 | Fix KD (ViT-L to ViT-B) | +1.0-2.0pp | Medium | Not started |
| 6 | Box-grained matching | +0.5-1.5pp | Medium | Not started |
| 7 | Hand-annotated zones | +0.5-1.0pp | Medium | Not started |

## Phase 0: Free Gains (Today)
- Enable CamTTA: config toggle stage2.camera_tta.enabled=true

## Phase 1: Ensemble + Reranking
- ResNet101-IBN-a ensemble with score-level fusion (FUSION_WEIGHT sweep 0.2-0.5)
- Re-test reranking with ensemble features

## Phase 2: Training Improvements
- Camera-aware adversarial training (gradient reversal)
- Knowledge distillation ViT-L → ViT-B (fix 09c bugs)

## Phase 3: Architecture Refinements
- Box-grained matching (detection-level similarity)
- Hand-annotated zone polygons