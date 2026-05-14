# People Tracking Support — Research & Compatibility Analysis

> **Generated**: 2026-03-27 | **Status**: Implementation In Progress

## Summary
- System is ~90% generic already — person model routing exists in Stage 2
- Main gap: trained person ReID weights (Market-1501)
- SOTA person ReID (Market-1501): SOLIDER Swin-B = 93.9% mAP, TransReID ViT CLIP = ~91% mAP
- Our target: TransReID ViT-B/16 CLIP on Market-1501, expected ~89-91% mAP, Rank-1 ~95-96%
- Codebase changes: config-only for pipeline, new training notebook 09p

## Codebase Audit
- Stage 0 (Ingestion): FULLY GENERIC
- Stage 1 (Tracking): CONFIG-ONLY — change detector.classes to [0]
- Stage 2 (Features): DUAL-MODEL IMPLEMENTED — person_reid routing at line ~310
- Stage 3 (Indexing): FULLY GENERIC
- Stage 4 (Association): CLASS-ADAPTIVE — person weights already in config
- Stage 5 (Evaluation): GENERIC — WILDTRACK eval implemented

## SOTA (Market-1501)
| Method | mAP | Rank-1 |
|--------|-----|--------|
| BagTricks (ResNet50) | 85.9% | 94.5% |
| AGW (ResNet50-IBN) | 87.8% | 95.1% |
| TransReID (ViT-B/16) | 89.0% | 95.1% |
| CLIP-ReID (ViT-B/16) | ~91% | ~96% |
| SOLIDER (Swin-B) | 93.9% | 96.9% |

## Implementation Status
- [x] Research complete
- [ ] 09p training notebook created
- [ ] Person MTMC config verified
- [ ] Market-1501 training run
- [ ] Baseline numbers obtained