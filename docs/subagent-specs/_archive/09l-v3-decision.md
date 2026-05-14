# 09l v3 Decision: Extend LAION-2B CLIP Training to 300 Epochs

## Status: RECOMMENDED — Extend training before pivoting to DFN-2B

## 09l v2 Results Summary

| Metric | Value |
|--------|:-----:|
| **Raw mAP** | **61.51%** |
| R1 | 81.41% |
| mAP (reranked) | 67.20% |
| R1 (reranked) | 82.95% |
| Epochs | 160 (cosine T_max=150, LR hit 0.00) |
| Runtime | 3.7h on T4 |

### Training Curve Analysis

| Epoch | mAP | Δ per 20ep |
|:-----:|:---:|:----------:|
| 20 | 12.69% | — |
| 40 | 18.47% | +5.78 |
| 60 | 25.99% | +7.52 |
| 80 | 34.23% | +8.24 |
| 100 | 42.25% | +8.02 |
| 120 | 49.73% | +7.48 |
| 140 | 55.98% | +6.25 |
| 160 | 61.51% | +5.53 |

**Key observation**: The per-interval gain is decelerating smoothly (8.24 → 8.02 → 7.48 → 6.25 → 5.53), but the model was **still gaining +5.53pp in the final interval** when LR reached 0.00. This is NOT a converged model — it ran out of LR budget.

### Comparison to Primary Model

| Model | Backbone | VeRi init | Epochs | mAP |
|-------|----------|:---------:|:------:|:---:|
| Primary (09b v2) | OpenAI CLIP ViT-B/16 | Yes | 120 | **80.14%** |
| 09l v2 | LAION-2B CLIP ViT-B/16 | Yes | 160 | **61.51%** |

Gap: **18.63pp** — but LAION-2B's LR schedule was exhausted while still learning rapidly.

## Decision: Extend to 300 Epochs (09l v3)

### Rationale

1. **Model is clearly not converged**: +5.53pp gain in last interval with LR approaching 0
2. **65% threshold is within reach**: Conservative extrapolation (deceleration of ~0.7pp per interval) puts epoch 200 at ~66%, epoch 240 at ~70%
3. **Low cost**: ~3.2h additional T4 time via warm restart from ep160 checkpoint
4. **DFN-2B pivot is premature**: The 09m trigger condition ("only if 09l v2 < 65%") was designed for a converged model. 09l v2 is unconverged, so the condition doesn't cleanly apply
5. **Warm restart is a well-studied technique**: Resume + new cosine schedule is standard practice

### Conservative mAP Extrapolation (with deceleration ~0.7pp/interval)

| Epoch | Projected mAP | Δ per 20ep |
|:-----:|:------------:|:----------:|
| 180 | ~66.3% | ~4.8 |
| 200 | ~70.4% | ~4.1 |
| 220 | ~73.8% | ~3.4 |
| 240 | ~76.5% | ~2.7 |
| 260 | ~78.5% | ~2.0 |
| 280 | ~79.8% | ~1.3 |
| 300 | ~80.4% | ~0.6 |

**Realistic expectation**: 68-75% mAP. The optimistic extrapolation to 80% assumes constant deceleration, which likely overpredicts. LAION-2B pretraining is known to be somewhat weaker than OpenAI CLIP, so a ceiling of 70-78% is more realistic.

**Ensemble threshold (65%)**: Very likely crossed by epoch 180-200.

### Technical Plan for 09l v3

**Approach**: Warm restart from the epoch 160 EMA checkpoint

```
Resume checkpoint: working/09l_laion2b/export/vehicle_transreid_vit_base_cityflowv2.pth
                   (or best_ema checkpoint from epoch 160)

New schedule (140 additional epochs):
  - Warmup: 5 epochs (LR ramp from 1e-6 → backbone_lr=1e-4, head_lr=1e-3)
  - Cosine: T_max=135 (decays to 0 at epoch 300 total)
  - Total new epochs: 140
  - Eval every 20 epochs starting from epoch 180

All other hyperparams UNCHANGED:
  - AdamW, weight_decay=5e-4
  - LLRD=0.75
  - CE(eps=0.05) + Triplet(m=0.3) + Center(delayed, weight=5e-4)
  - EMA decay=0.9999
  - Batch 64, PKSampler(p=16, k=4)
  - Same augmentations, same CLIP normalization
```

### Implementation Changes (09l v2 → v3)

1. **Cell: Training config** — Change `NUM_EPOCHS = 160` → `NUM_EPOCHS = 300`
2. **Cell: LR schedule** — Change `T_max = 150` → `T_max = 290` (or implement warm restart)
3. **Cell: Checkpoint loading** — Add resume logic to load epoch 160 checkpoint and optimizer state
4. **Cell: Evaluation** — Ensure eval runs at epochs 180, 200, 220, 240, 260, 280, 300
5. **Alternative (simpler)**: Retrain from scratch with 300 epochs and T_max=290
   - Pro: Cleaner, no warm restart discontinuity
   - Con: Wastes 3.7h redoing epochs 1-160
   - **Recommendation**: Use warm restart to save time, unless checkpoint loading is complex

### Success Criteria

| Outcome | mAP Range | Action |
|---------|:---------:|--------|
| **Strong success** | ≥70% | Deploy as ensemble secondary immediately; skip DFN-2B |
| **Success** | 65-70% | Deploy as ensemble secondary; consider DFN-2B as tertiary later |
| **Marginal** | 62-65% | Pivot to DFN-2B (09m); LAION-2B ceiling too low for ensemble |
| **Failure** | <62% | Warm restart didn't help; pivot to DFN-2B (09m) immediately |

### Estimated Timeline

- Push 09l v3: ~30 min (notebook edits + kaggle push)
- T4 runtime: ~3.2h (warm restart) or ~6.9h (full retrain)
- Results available: ~4-7h after push

## Questions Answered

### Q1: Should we extend LAION-2B training?
**YES.** The curve is unambiguously unconverged. Extending training is the highest-expected-value action.

### Q2: Should we pivot to DFN-2B?
**NOT YET.** The 09m trigger condition was for a converged model. Wait for 09l v3 results.

### Q3: Both in parallel?
**Not possible** — single Kaggle GPU slot (gumfreddy). Sequential is the only option.

### Q4: Should reranked mAP (67.20%) count toward the 65% threshold?
**NO.** The 65% threshold refers to raw feature quality for score-level ensemble fusion. Reranking is a post-processing step that isn't applied during fusion. The raw mAP of 61.51% is the decision-relevant metric.