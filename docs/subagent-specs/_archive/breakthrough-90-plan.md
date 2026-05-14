# Breakthrough Plan: MTMC IDF1 78% -> 85%+

## Honest Assessment
- 90% is beyond published SOTA (84.86%). Realistic target: 85%.
- Bottleneck is feature quality, not association. 220+ association configs exhausted.

## Priority Actions
1. **Fix ResNet101-IBN-a training** (mAP 12% -> 60%+): LR 1e-3->3.5e-4, warmup 5->10, circle_weight 0->0.5, label_smooth 0.05->0.1, batch 48->64
2. **2-model ensemble** with fixed ResNet101 + v80 TransReID: expect +1.5-2.5pp
3. **CID_BIAS per camera pair**: calibrate cross-camera similarity distributions
4. **Re-enable k-reciprocal reranking** once ensemble is strong

## Expected Progression
- Phase 1 (fix training + ensemble): 78% -> 81-83%
- Phase 2 (384px ViT + CID_BIAS + reranking): 83% -> 85%+