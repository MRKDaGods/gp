# Post-Ensemble Analysis — Strategic Pivot Assessment

> **Date**: 2026-04-01
> **Triggered by**: Score-level fusion ensemble test (fus0.3_ter0.0) yielded MTMC IDF1 = 0.774 vs 0.775 baseline
> **Status**: Analysis complete — strategic pivot needed

---

## 1. Ensemble Result Summary

| Config | MTMC IDF1 | IDF1 | MOTA | HOTA |
|--------|:---------:|:----:|:----:|:----:|
| Baseline (v52 single-model) | 0.775 | 0.801 | 0.671 | 0.581 |
| Ensemble (fus0.3_ter0.0) | 0.774 | 0.803 | 0.676 | 0.581 |
| **Delta** | **-0.1pp** | +0.2pp | +0.5pp | 0.0pp |

The ensemble is statistically identical to baseline on MTMC IDF1, the only metric that matters.

---

## 2. Why the Ensemble Failed

### 2.1 Quality Gap Is Too Large

| Model | mAP (CityFlowV2) | Role in Fusion |
|-------|:-----------------:|:--------------:|
| TransReID ViT-B/16 CLIP 256px | ~80% | 70% weight |
| ResNet101-IBN-a (09d v18) | 52.77% | 30% weight |
| **Gap** | **~28pp** | — |

At 30% weight, the secondary model contributes 30% noise to every similarity score. For cross-camera pairs where the primary model produces a correct match at similarity ~0.55, the secondary model produces ~0.30 (near random), dragging the fused score down and making correct matches harder to distinguish from non-matches.

### 2.2 AIC22 Winners Used Balanced Ensembles

| System | Models | mAP Range | Ensemble Diversity |
|--------|:------:|:---------:|:------------------:|
| AIC22 1st (84.86%) | 5 models | All >70% on VeRi-776 | Architecture diversity (ResNet, ViT, etc.) |
| AIC22 2nd (84.37%) | 3 models | All >75% on VeRi-776 | ResNet101×2 + ResNeXt101, all IBN-a + DMT |
| **Our attempt** | **2 models** | **52-80%** | **One strong, one weak** |

The universal pattern: ensemble members must ALL be competent. Diversity only helps when all members carry useful signal separately.

### 2.3 ResNet101-IBN-a Training Has Hit a Ceiling

| Attempt | Config | mAP | Issue |
|---------|--------|:---:|-------|
| 09d v18 (ali369) | AdamW lr=1e-3, 60 epochs | 52.77% | Best result but still weak |
| 09d v18 (mrkdagods) | SGD lr=0.008 | 30.27% | SGD catastrophic |
| 09e v2 → 09f v3 | VeRi-776 pretrain → CityFlowV2 | 42.7% | Pretrain HURT transfer |
| 09d v17 | Circle loss + triplet | 29.6% | Loss conflict |
| 09d v12 | IBN layer3 bug | 21.9% | Fixed in v13 |

The 09d recipe (AdamW, lr=1e-3) is the only one that worked reasonably, but 52.77% is still too low for ensemble value. The VeRi-776→CityFlowV2 transfer path made things worse, not better.

---

## 3. GPU Budget Assessment

| Account | GPU Hours Remaining | Refresh |
|---------|:-------------------:|:-------:|
| mrkdagods | 0 | Next Tuesday |
| ali369 | 0 | Next Tuesday |
| gumfreddy | ~25-26h | Next Tuesday adds ~30h |

With ~25h on gumfreddy right now and refreshes coming Tuesday, we have roughly **85-95 GPU-hours** over the next week across all three accounts.

---

## 4. Strategic Options Analysis

### Option A: Continue Ensemble with Better Secondary

**What**: Retrain ResNet101-IBN-a with improved recipe to reach ≥65% mAP, then re-test ensemble.

**Pros**:
- Ensemble is the proven SOTA path (all AIC winners use it)
- Infrastructure already built (stage 2 + stage 4 support multi-model fusion)

**Cons**:
- Every ResNet training attempt except 09d v18 failed badly
- 09d v18's 52.77% may be near the ceiling for ImageNet→CityFlowV2 direct transfer
- VeRi-776 pretrain path made things worse
- Unclear what recipe change would bridge the 13pp gap to 65%+ mAP

**Required**:
- Better training recipe: longer training (120+ epochs), stronger augmentation (random erasing probability, color jitter), warm restarts
- Or different backbone: try ResNet50-IBN-a (smaller, may train more stably) or a different ViT variant
- ~10-15 GPU hours per training attempt

**Verdict**: HIGH RISK. No clear path to >65% mAP has been identified.

### Option B: Deploy 384px TransReID as Primary

**What**: Use the 384px ViT-B/16 CLIP (09b v2, 80.14% mAP) as primary instead of 256px.

**Status**: **CONFIRMED DEAD END** (2026-03-30).
- v43: MTMC IDF1 = 0.7585 (-2.6pp vs baseline)
- v44: MTMC IDF1 = 0.7562 (-2.8pp vs baseline)
- Root cause: higher resolution captures viewpoint-specific textures that hurt cross-camera matching

**Verdict**: DO NOT RETRY. Already tested definitively.

### Option C: Different Secondary Architecture

**What**: Instead of ResNet101-IBN-a, train a fundamentally different secondary model that may transfer better.

**Candidates**:
1. **ViT-S/16 (Small ViT)** — same CLIP pretraining path as primary but different capacity. Could reach 60-70% mAP with the same 3-stage pipeline (CLIP → VeRi-776 → CityFlowV2).
2. **ResNet50-IBN-a** — simpler, faster to train, may converge more reliably than ResNet101.
3. **SwinTransformer-T** — transformer diversity from CNN primary, but no existing training code.

**Pros**:
- ViT-S shares the proven CLIP → VeRi-776 → CityFlowV2 pipeline
- Architectural diversity (if mixing CNN + transformer) is what AIC winners do

**Cons**:
- New training code needed for most options
- Each attempt uses ~10-15 GPU hours
- No guarantee of reaching ≥65% mAP

**Verdict**: MODERATE RISK. ViT-S/16 is most promising due to shared infrastructure.

### Option D: CID_BIAS Camera-Pair Priors

**What**: Use ground-truth-derived or learned camera-pair biases to boost/suppress similarity scores.

**Status**: Tested once on 384px features → -0.52pp. Never tested on the 256px baseline.

**Pros**:
- Zero GPU cost (CPU-only, runs locally)
- Already implemented in the pipeline
- AIC22 2nd place used CID_BIAS successfully

**Cons**:
- Only 464/941 tracklets had GT matches for bias estimation → noisy estimates
- The 384px test was on already-degraded features
- Unclear if the current 6-camera CityFlowV2 layout provides enough camera-pair diversity for meaningful biases

**Verdict**: LOW RISK, LOW COST. Worth testing on 256px baseline as a quick experiment.

### Option E: Accept Current Ceiling and Focus on Person Pipeline

**What**: Acknowledge that vehicle MTMC IDF1 ~77.5% is the achievable ceiling with current resources and redirect all effort to the person pipeline (WILDTRACK), which is 0.6pp from its SOTA target.

**Pros**:
- Person pipeline has clear, achievable gains (0.6pp gap, known path via better detections)
- Lower GPU cost per experiment
- Less explored — higher chance of quick wins

**Cons**:
- Gives up on the vehicle IDF1 target
- The competition metric is vehicle MTMC IDF1

**Verdict**: PRAGMATIC FALLBACK if vehicle experiments continue to fail.

---

## 5. Recommended Strategy

### Immediate (This Week — ~25h gumfreddy GPU)

1. **CID_BIAS on 256px baseline** (0 GPU hours, CPU-only)
   - Test camera-pair bias matrix on the current v52 256px association recipe
   - This has never been tested on 256px features (only on the dead-end 384px)
   - If it helps even +0.3pp, it's free improvement

2. **Extended ResNet101-IBN-a training** (~12 GPU hours on gumfreddy)
   - Start from 09d v18 checkpoint (52.77% mAP) and continue training for 120 more epochs
   - Add stronger augmentation: random erasing p=0.5, color jitter
   - Use cosine annealing LR schedule with warm restarts
   - Target: ≥60% mAP
   - If successful, re-test ensemble at 0.15-0.30 weight

3. **Person pipeline — 12a v27 with ResNet34** (~3 GPU hours)
   - Train ResNet34 MVDeTr for 25 epochs (vs current ResNet18/10 epochs)
   - Close the remaining 0.6pp IDF1 gap on WILDTRACK

### Next Week (After GPU Refresh — ~90h total)

4. **If ResNet hits ≥60% mAP**: Run ensemble sweep (0.10, 0.15, 0.20, 0.25, 0.30) → ~5 GPU hours
5. **If ResNet fails again**: Try ViT-S/16 secondary via the same CLIP → VeRi-776 → CityFlowV2 pipeline → ~15 GPU hours
6. **Person pipeline refinements** based on 12a v27 results

### What NOT to Do

- Do NOT retry 384px (confirmed dead end)
- Do NOT retry DMT camera-aware training (confirmed -1.4pp)
- Do NOT run more association sweeps (225+ configs, exhausted)
- Do NOT ensemble at high weight (0.30+) with a <60% mAP secondary
- Do NOT use VeRi-776→CityFlowV2 transfer for ResNet (09f v3 proved it hurts)

---

## 6. Success Criteria

| Milestone | Target | Measurement |
|-----------|--------|-------------|
| CID_BIAS on 256px | MTMC IDF1 > 0.778 | Local 10c run |
| Extended ResNet training | CityFlowV2 mAP > 60% | 09d eval split |
| Ensemble with improved secondary | MTMC IDF1 > 0.785 | 10c Kaggle run |
| Person pipeline | IDF1 > 0.950 | 12b Kaggle run |

---

## 7. Key Insight

The ensemble approach is fundamentally correct — it is the universal pattern among SOTA methods. But ensemble only works when all members carry independent, high-quality signal. Our failure was not in the fusion mechanism or weight selection; it was in deploying a critically undertrained secondary model. The path forward is either:
1. **Make the secondary model competitive** (≥65% mAP), or
2. **Find a different secondary architecture** that can reach that threshold with the same training infrastructure

Until one of these is achieved, any fusion weight >0.0 will dilute the primary model's signal and produce neutral-to-negative results.