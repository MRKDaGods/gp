# Path to 90%+ MTMC IDF1: Comprehensive Strategy Spec

**Date**: 2026-03-27
**Author**: MTMC Planner
**Scope**: Both vehicles (CityFlowV2) and people (Market-1501/WILDTRACK)
**Ambition Level**: EXTREME — 90%+ exceeds ALL published SOTA by 5-10pp

---

## Executive Summary

90%+ MTMC IDF1 on vehicles is beyond any published result — the AIC22 1st place is 84.86% with a 5-model ensemble. Our current 78.4% leaves an 11.6pp gap to 90%. For people, our WILDTRACK per-cam IDF1 is 16.8% (fundamentally limited by overlapping-FOV ground-plane protocol).

### Gap Analysis Table

| Target | Current | Gap | Realistic Ceiling | What's Needed for 90%+ |
|--------|:-------:|:---:|:-----------------:|----------------------|
| Vehicles (CityFlowV2) | 78.4% | 11.6pp | 86-88% | Paradigm shift beyond published methods |
| People (WILDTRACK GP) | 16.8% | 73.2pp | 70-80% | Different architecture for ground-plane eval |
| People (per-cam 2D) | N/A | ~90pp | 85-90% | Strong person ReID + association |

## A. Vehicle Track: 78.4% to 90%+ IDF1

### A.1 Tranche 1: Closing to SOTA (78.4% to 85%)

| Deficiency | Est. Impact | Status |
|------------|:----------:|--------|
| Single model vs 3-model ensemble | 2.5-3.5pp | ResNet101-IBN-a VeRi-776 pretrained COMPLETE |
| No CID_BIAS (per camera-pair) | 0.5-1.0pp | Script exists, config ready |
| Reranking disabled | 0.5-1.0pp | Currently hurts; should help with ensemble |
| No DMT camera-aware training | 1.0-1.5pp | Not implemented |
| No zone-based ST constraints | 0.5-1.0pp | Auto-zones hurt; hand-annotated untested |

### A.2 Tranche 2: Beyond SOTA (85% to 90%+)

| Innovation | Est. Impact | Difficulty |
|-----------|:----------:|:----------:|
| GNN edge classification (LMGP-style) | 1.0-3.0pp | High |
| Box-grained reranking | 0.5-1.5pp | High |
| Knowledge distillation ViT-L to ViT-B | 0.5-1.0pp | Medium |
| Temporal attention for tracklet embedding | 0.5-1.5pp | High |
| SAM2 foreground masking | 0.3-0.5pp | Medium |

### A.3 Feature Quality Improvements

#### Deploy ResNet101-IBN-a Ensemble (P1)
- VeRi-776 pretrained to 63.63% mAP (09e COMPLETE)
- CityFlowV2 fine-tune with VeRi-776 init just completed (09d mrkdagods) 
- Score-level fusion: alpha*sim_vit + (1-alpha)*sim_resnet, alpha in [0.55, 0.70]
- Expected: +1.5-2.5pp

#### CID_BIAS Per Camera-Pair (P1)
- Pre-compute per-camera-pair additive similarity bias from GT
- Script at scripts/compute_cid_bias.py ready
- Expected: +0.5-1.0pp

#### DMT Camera-Aware Training (P2)
- Stage 1: Standard ID + triplet + circle loss (40 epochs)
- Stage 2: + camera-aware adversarial loss (20 epochs)
- Expected: +1.0-1.5pp

## B. People Track: 0% to 90%+ IDF1

### B.1 Current State
- 09p person ReID RUNNING on yahia (TransReID ViT-B/16 CLIP on Market-1501)
- Expected mAP ~89-91% on Market-1501
- Pipeline 90% generic, person routing exists

### B.2 Person-Specific Association Config
```yaml
stage4.association:
  graph.similarity_threshold: 0.35  # lower than vehicles (0.53)
  weights.person:
    appearance: 0.80
    hsv: 0.10  
    spatiotemporal: 0.10
  reranking.enabled: true
```

### B.3 WILDTRACK Challenge
- Ground-plane protocol requires multi-view fusion (MVDeTr)
- Per-camera 2D protocol achievable: 80-90% IDF1
- Recommended: target per-camera protocol first

## C. Priority-Ordered Action Plan

### Phase 0: Immediate (No GPU)
- 0a: Get 09d mrkdagods results (VeRi-776 pretrained ResNet on CityFlowV2)
- 0b: Get 09p person ReID results when complete
- 0c: Upload upgraded ResNet weights to mtmc-weights

### Phase 1: Deploy Ensemble (No GPU, 10c only)
- 1a: Run 256px ViT + upgraded ResNet ensemble in 10c
- 1b: Sweep ensemble weights (0.15, 0.25, 0.35, 0.45)
- Expected: 80-82% IDF1

### Phase 2: CID_BIAS + Reranking (No GPU)
- 2a: Compute CID_BIAS from GT
- 2b: Deploy CID_BIAS
- 2c: Re-enable reranking with ensemble features
- Expected: 83-84% IDF1

### Phase 3: DMT + 3rd Model (GPU needed)
- 3a: Hand-annotate zone polygons
- 3b: Train ResNeXt101-IBN-a (3rd member)
- 3c: DMT Stage 2 fine-tuning
- Expected: 84-86% IDF1

### Phase 4: Advanced (GPU needed)
- 4a: GNN edge classification
- 4b: Box-grained reranking
- 4c: Knowledge distillation
- Expected: 87-89% IDF1

### People Pipeline (Parallel)
- P1: Get 09p results, upload person ReID weights
- P2: Prepare WILDTRACK on Kaggle
- P3: Create people pipeline notebooks
- P4: First WILDTRACK run (baseline)
- P5: Person association tuning
- P6: Enable reranking for people

## D. Realistic Assessment

| Timeline | Vehicle IDF1 | People IDF1 |
|----------|:----------:|:----------:|
| Now | 78.4% | 16.8% |
| 2 weeks | 80-82% | N/A |
| 1 month | 83-85% | 50-60% |
| 2 months | 85-87% | 65-75% |
| 3 months | 87-89% | 75-85% |

### Key Principle
Features > Association. 70% of MTMC performance comes from ReID quality.