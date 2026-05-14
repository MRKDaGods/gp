# SOTA Battle Plan — Vehicle (CityFlowV2) + Person (WILDTRACK)

> **Created**: 2026-03-28 | **Objective**: Beat SOTA on both pipelines
> **Vehicle target**: IDF1 > 84.86% (AIC22 1st place)
> **Person target**: MTMC IDF1 > 60% (realistic), stretch 75%+
> **Accounts**: mrkdagods (30h GPU), ali369 (30h GPU), yahia (30h GPU) = 90h total

---

## Table of Contents
1. Executive Summary
2. Vehicle Pipeline Plan (Phases V1-V6)
3. Person Pipeline Plan (Phases P1-P5)
4. Parallel Execution Plan
5. Key Decisions
6. Risk Matrix

---

## 1. Executive Summary

### Vehicle Gap Analysis (6.5pp to SOTA)

| Deficiency | Est. Impact | Fix | Blocked By |
|-----------|:-----------:|-----|------------|
| Wrong 384px checkpoint on Kaggle | +1.0-2.5pp | Upload correct transreid_cityflowv2_384px_best.pth (local mAP=80.14%) | Nothing — immediate |
| Single model (no ensemble) | +1.5-2.5pp | Train ResNet101-IBN-a: VeRi-776 → CityFlowV2 | 09e training |
| No CID_BIAS | +0.5-1.0pp | Compute from GT, add to stage4 | Correct 384px features |
| Reranking disabled | +0.5-1.0pp | Re-enable after feature upgrade | Ensemble deployed |
| **Total recoverable** | **+3.5-7.0pp** | | |

### Person Gap Analysis (~50pp to published baselines)

| Deficiency | Est. Impact | Fix |
|-----------|:-----------:|-----|
| Extreme fragmentation (819+ tracklets for ~20 people) | Critical | Raise conf→0.70, min_hits→5, aggressive intra-merge |
| Low detector precision (~0.33) | Major | YOLO conf=0.70+, stricter NMS, ROI masks |
| ReID not fine-tuned on WILDTRACK | Moderate | Cannot fine-tune (only ~20 IDs); use EPFL/CUHK03 pretraining instead |
| PCA trained on wrong data | Moderate | Force refit with pca_transform_person.pkl.bak already renamed |
| Association params untested | Major | Full sweep: sim_thresh, weights, louvain, merge params |
| No spatiotemporal model | Moderate | WILDTRACK has overlapping FOV — temporal overlap bonus is key |

---

## 2. Vehicle Pipeline Plan

### Phase V1: Immediate Win — Fix 384px Deployment
**Priority**: CRITICAL | **GPU cost**: 0h (CPU only) | **Expected gain**: +1.0-2.5pp

Upload correct 384px ViT checkpoint (mAP=80.14%) to Kaggle mrkdagods/mtmc-weights dataset. Verify 10a doesn't override input_size back to 256.

Config for 10c (v80 best + 384px):
- AQE_K=3, SIM_THRESH=0.53, ALGORITHM=conflict_free_cc
- APPEARANCE_WEIGHT=0.70, HSV_WEIGHT=0.0, FUSION_WEIGHT=0.0

### Phase V2: ResNet101-IBN-a VeRi-776 Pretraining
**Priority**: HIGH | **GPU cost**: ~6h | **Expected gain**: enables ensemble (+1.5-2.5pp)

Check if models/reid/resnet101ibn_veri776_best.pth exists locally:
- If valid: Skip to 09d directly (VeRi-776 → CityFlowV2 fine-tune on yahia)
- If not: Run 09e first on mrkdagods (~3h), then 09d (~3h)

09d config: AdamW lr=3.5e-4, 120 epochs, batch_size=64, NO circle loss, NO SGD

### Phase V3: Ensemble Deployment
**Priority**: HIGH | **GPU cost**: ~2h | **Expected gain**: +1.5-2.5pp

10a: Add `stage2.reid.vehicle2.enabled=true` override
10c: Sweep FUSION_WEIGHT: 0.0, 0.1, 0.2, 0.3, 0.4

### Phase V4: CID_BIAS Camera-Pair Calibration
**Priority**: MEDIUM | **GPU cost**: 0h | **Expected gain**: +0.5-1.0pp

Run scripts/compute_cid_bias.py → upload NPY → enable in 10c

### Phase V5: Re-enable Reranking
**Priority**: LOW-MEDIUM | **GPU cost**: 0h | **Expected gain**: +0.5-1.0pp

Sweep: k1=[20,30,40], k2=[6,10], lambda=[0.3,0.4]

### Phase V6: Third Model (ViT 256px Legacy)
**Priority**: LOW | Only pursue if V3 shows +2pp

---

## 3. Person Pipeline Plan

### Phase P1: Detection & Tracking Re-engineering
**Priority**: CRITICAL | **Expected gain**: +15-25pp

Changes to wildtrack.yaml:
- confidence_threshold: 0.35 → 0.70
- min_hits: 1 → 5
- track_buffer: 60 → 120
- min_tracklet_length: 2 → 8
- max_gap (interpolation): 20 → 30
- intra_merge max_time_gap: 20 → 30

Target: ~80-150 tracklets (from 819)

### Phase P2: Person ReID Strategy
**Decision**: Do NOT fine-tune on WILDTRACK (only 20 IDs → overfit)
**Strategy**: Use existing Market-1501 models as ensemble
- Primary: TransReID ViT-B/16 (768D)
- Secondary: ResNet50-IBN-a (2048D)

Requires extending stage2 pipeline to support person2 (mirror vehicle2 logic).

### Phase P3: PCA Refit
Auto-refits when model file doesn't exist. Already set up (.bak renamed).
Test n_components: 128, 192, 256, 384, 512

### Phase P4: Association Parameter Sweep
Person-specific params (overlapping FOV = high temporal overlap bonus):
- sim_thresh sweep: 0.25, 0.30, 0.35, 0.40, 0.50
- weights: appearance=0.85, hsv=0.05, spatiotemporal=0.10
- temporal_overlap bonus: 0.05-0.20 (key signal for WILDTRACK)

### Phase P5: Notebook Modifications
- 11a: Add conf=0.70, min_hits=5 overrides
- 11c: Add tuning params cell + override lines

---

## 4. Parallel Execution Plan

| Account | Notebooks | GPU Hours | Priority |
|---------|-----------|:---------:|----------|
| ali369 | 10a→10b→10c (vehicle) | 4h/chain | Vehicle V1, V3 |
| mrkdagods | 09e, 11a→11b→11c (person) | 6h total | Vehicle V2 prereq, Person P1-P4 |
| yahia | 09d (ResNet CityFlowV2) | 3h | Vehicle V2 |

### Dependency Graph
- V1 (384px upload) → blocks V3 (ensemble) → blocks V4 (CID_BIAS) → blocks V5 (reranking)
- V2 (09e→09d) independent, feeds into V3
- P1 (detection) → P3 (PCA) → P4 (association sweep) — sequential chain
- Vehicle and Person pipelines are FULLY INDEPENDENT

---

## 5. Key Decisions

1. **Person fine-tuning**: NO (20 IDs → overfit). Use Market-1501 pretrained models.
2. **Person ensemble**: YES if code supports person2. Both models exist locally.
3. **Fastest to 85% vehicle IDF1**: V1+V2+V3+V4+V5 stacked = ~84-85%
4. **Realistic person IDF1**: 60-70% (85%+ requires multi-view detection we don't have)

---

## 6. Risk Matrix

| Risk | Severity | Mitigation |
|------|:--------:|------------|
| 384px weights upload corrupted | HIGH | Verify mAP locally before upload |
| ResNet101-IBN-a still weak after VeRi-776 | MEDIUM | Use 0.1 fusion weight; fallback to single ViT |
| Person pipeline still <30% after P1 | HIGH | Deep-dive error analysis; consider ROI masks |
| Kaggle quota exhausted | MEDIUM | Prioritize ali369 for vehicle |
