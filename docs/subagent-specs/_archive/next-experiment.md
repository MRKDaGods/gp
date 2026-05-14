# Next Experiment: AFLink Pure Addon to v52 Baseline

## Status: PROPOSED
## Date: 2026-04-16

## Goal

Test AFLink motion-based post-association as a **pure addon** to the current v52-optimal association config, without changing any base association parameters. This isolates AFLink's true impact from the confounded v46 joint sweep.

## Hypothesis

The v46 AFLink test (-3.95pp) was confounded because it ran a **joint sweep** over both base association parameters AND AFLink parameters. The sweep optimizer selected a suboptimal base config (non-AFLink peak was only IDF1=0.669) that is 10.6pp below the v52-optimal baseline (0.775). At the v52 operating point, where the initial association is much stronger, AFLink will see:
1. Fewer fragmented trajectories to merge (reducing false-positive risk)
2. Higher-quality initial trajectories (so any merges are more likely correct)

The error profile supports this: **87 fragmented** vs 35 conflated errors. Fragmentation is 2.5× more common, and AFLink specifically targets fragmentation through motion-consistent merging.

## Evidence

| Source | AFLink Effect | Context |
|--------|:---:|---------|
| 10c v46 (baseline features) | -3.95pp | Confounded joint sweep; non-AFLink peak was 0.669 (vs v52 at 0.775) |
| 10c v46 within-sweep | +6.3pp | Internal gain 0.669 → 0.732 with tight params |
| 10c v49 (augoverhaul features) | +4.7pp locally | Internal gain 0.675 → 0.722 |
| 10c v50 (SAM2 features) | +1.7pp locally | Internal gain 0.659 → 0.676 |
| v46 merge count | 57 merges | Reduced trajectories 239 → 182 (aggressive) |

**Key observation**: AFLink shows a consistent +1.7 to +6.3pp **local improvement** across three different feature sets. The v46 end-to-end regression was driven by the joint sweep selecting a worse base config, NOT by AFLink itself being net-negative on a good base.

## Changes Required

### Stage 4 (Association) — Config overrides only, no code changes

Add AFLink overrides on top of the v52-optimal config:

**v52 base config (keep unchanged):**
- `stage4.association.graph.similarity_threshold=0.50`
- `stage4.association.weights.vehicle.appearance=0.70`
- `stage4.association.fic.regularisation=0.50`
- `stage4.association.aqe.k=3`
- `stage4.association.gallery_expansion.threshold=0.48`
- `stage4.association.gallery_expansion.orphan_match_threshold=0.38`

**AFLink addon configs to sweep:**

| Config | `max_spatial_gap_px` | `min_direction_cos` | `min_velocity_ratio` | Rationale |
|--------|:---:|:---:|:---:|-----------|
| A (tight) | 150 | 0.85 | 0.5 | Best params from v46/v50 sweeps |
| B (tighter spatial) | 100 | 0.85 | 0.5 | More conservative spatial constraint |
| C (tighter direction) | 150 | 0.90 | 0.5 | Require near-parallel motion |
| D (strictest) | 100 | 0.90 | 0.7 | Tightest: small gap + parallel + similar speed |

All configs use:
- `stage4.association.aflink.enabled=true`
- `stage4.association.aflink.max_time_gap_frames=150`
- `stage4.association.aflink.velocity_window=5`

### CLI Override Template

```bash
python scripts/run_pipeline.py --config configs/default.yaml \
  --override stage4.association.graph.similarity_threshold=0.50 \
  --override stage4.association.weights.vehicle.appearance=0.70 \
  --override stage4.association.fic.regularisation=0.50 \
  --override stage4.association.aqe.k=3 \
  --override stage4.association.gallery_expansion.threshold=0.48 \
  --override stage4.association.gallery_expansion.orphan_match_threshold=0.38 \
  --override stage4.association.aflink.enabled=true \
  --override stage4.association.aflink.max_spatial_gap_px=150 \
  --override stage4.association.aflink.min_direction_cos=0.85 \
  --override stage4.association.aflink.min_velocity_ratio=0.5
```

## Expected Impact

- **Best case**: +0.3 to +0.5pp MTMC IDF1 (0.775 → 0.778-0.780), recovering some fragmented trajectories without adding conflation
- **Likely case**: Neutral (±0.1pp), confirming that motion cues don't help at this operating point
- **Worst case**: -0.5pp MTMC IDF1, confirming v46's conclusion that AFLink is a dead end

## Risks

1. **False merges on non-overlapping cameras** — CityFlowV2 cameras have minimal spatial overlap; vehicles moving in similar directions are often different identities
2. **Already-optimized base reduces headroom** — With 0.775 MTMC IDF1, most easy fragmentation is already resolved by appearance association
3. **Small expected gain** — Even if positive, unlikely to exceed +0.5pp

## Measurement

- **Command**: See CLI template above (4 configs × 1 run each)
- **Baseline**: v52 MTMC IDF1 = 0.775 (run without AFLink for comparison)
- **Success metric**: MTMC IDF1 > 0.776 (any measurable improvement)
- **GPU required**: No — CPU-only (stages 3-5 local evaluation)
- **Estimated time**: ~30 minutes total for all 4 configs
- **Data source**: Existing v52-era stage-2 features (no re-extraction needed)

## Decision Logic

- **If any config ≥ 0.778**: AFLink becomes part of the baseline. Update v52 config.
- **If all configs within ±0.002 of 0.775**: AFLink is confirmed neutral. Close this experiment.
- **If all configs < 0.773**: AFLink is confirmed harmful at all thresholds. Permanently close.

## Important Corrections

### CenterLoss is NOT untried
The copilot-instructions.md entry "Center loss for primary ViT training (never attempted)" is **incorrect**. The 09b v2 baseline model (80.14% mAP) was trained with **CE(eps=0.05) + Triplet(margin=0.3) + CenterLoss(weight=5e-4, start_epoch=15)**. CenterLoss is already part of the winning recipe and should be removed from the "remaining untried approaches" list.

### Remaining genuinely untried vehicle approaches
After removing CenterLoss, the remaining untried approaches are:
1. **GNN edge classification** — Heavy implementation, requires training data + GPU, uncertain payoff
2. **Min-cost-max-flow network solver** — Moderate implementation, CPU-only, but global-optimal assignment was -3.5pp on persons
3. **Different backbone for secondary model** — ConvNeXt-Base or EfficientNet-V2 instead of ResNet101-IBN-a, requires GPU training
4. **Center loss weight tuning** — Baseline uses 0.0005; higher weight (0.001-0.002) might force tighter clusters, but requires full GPU retraining with uncertain payoff

### Strategic Assessment

The vehicle pipeline is approaching full convergence at **IDF1 ≈ 0.775**. The 7.36pp gap to SOTA (0.8486) is caused by **single-model features vs. 3-5 model ensemble**. Without a genuinely complementary secondary model (≥65% mAP on CityFlowV2), the realistic ceiling is **77-78% MTMC IDF1**. This AFLink experiment is the last low-cost association-side test before declaring the pipeline fully converged.