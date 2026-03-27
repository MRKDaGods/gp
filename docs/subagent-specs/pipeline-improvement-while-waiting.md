# Pipeline Improvement While Waiting — K-Reciprocal Reranking Re-test

## Context
- Current MTMC IDF1 = 0.782 (10c v34)
- ResNet101-IBN-a v18 training running on Kaggle
- Error profile: 87 fragmented (under-merge) vs 35 conflated (over-merge)

## Improvement: K-Reciprocal Reranking Re-test

### Why Re-test Now
v25 disabled reranking because it "hurt vehicles" — but under different conditions:
- 512D PCA (now 384D)
- FIC reg=3.0 (now 0.1)
- No QE (now enabled with aqe_k=3)
- No MNN filter (now active)
- Basic CC (now conflict_free_cc)

### Implementation
Zero code changes — config overrides only. Added to 10c v35 sweep:
- 5 reranking configs: lambda={0.3,0.5,0.7,0.9}, k1={20,30}
- Config paths: stage4.association.reranking.{enabled,k1,k2,lambda_value}

### Expected Impact
- Best case: +1.0-1.5pp IDF1
- Likely case: +0.3-0.8pp
- Worst case: -0.5pp (revert to baseline)

### Deployed
- 10c v35 pushed to ali369 (CPU, ~30 min)

## Future: Camera-Pair Similarity Normalization
If reranking doesn't help, consider per-camera-pair mean normalization (~30 lines in pipeline.py).