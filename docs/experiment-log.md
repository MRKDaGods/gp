# MTMC Tracker — Comprehensive Experiment Log

> **Purpose**: Prevent re-running experiments. Every parameter combination and approach is logged here.
> **Last updated**: 2026-03-25
> **Current best (Kaggle)**: MTMC IDF1 = **78.4%** (10c v44 / ali369 / code v80, min_hits=2)
> **Current best (local, recent)**: MTMC IDF1 = **77.7%** (10c v28, CamTTA + power_norm=0.5)
> **Historical local claim**: IDF1 = 82.97% (v47 — unverifiable, predates current experiment log)
> **SOTA target**: IDF1 ≈ 84.1% (AIC21 1st place) / 84.86% (AIC22 1st)
> **Gap**: ~5.7–6.5pp — caused by **feature quality**, NOT association tuning
> **Total experiments**: 225+ configs across ~14h GPU

---

## Table of Contents

1. [Current Best Configuration](#1-current-best-configuration)
2. [Complete Experiment History](#2-complete-experiment-history)
3. [Exhaustive Parameter Sweep Results](#3-exhaustive-parameter-sweep-results)
4. [Training History (09b, 09c, 09d)](#4-training-history-09b-09c-09d)
5. [What Has Been Exhaustively Tried](#5-what-has-been-exhaustively-tried)
6. [What Has Been Tried Poorly / Once](#6-what-has-been-tried-poorly--once)
7. [What Has NEVER Been Tried](#7-what-has-never-been-tried)
8. [Key Insights & Dead Ends](#8-key-insights--dead-ends)
9. [Error Profile](#9-error-profile)
10. [Currently Running / In-Progress](#10-currently-running--in-progress)
11. [Quick Lookup Index](#11-quick-lookup-index)

---

## 1. Current Best Configuration

**Kaggle best**: IDF1 = **78.4%** — 10c v44, ali369 account, code v80, 2026-03

```yaml
# Stage 4: Association
algorithm:             conflict_free_cc
sim_thresh:            0.53
appearance_weight:     0.70
hsv_weight:            0.0
fusion_weight:         0.10          # OSNet secondary (score-level)
fic_reg:               0.1
aqe_k:                 3
qe_alpha:              5.0
intra_merge:           true (thresh=0.80, gap=30)
bridge_prune:          0.0
max_component_size:    12
mutual_nn_top_k:       20
gallery_expansion:     0.50
length_weight_power:   0.3
temporal_overlap:      0.05 (max_mean_time=5.0)
exhaustive_min_sim:    0.10

# Disabled (all tested, all hurt)
camera_bias:           false         # -0.4pp
zone_model:            false         # -0.4pp
hierarchical:          false         # -1.0 to -5.1pp
csls:                  false         # CATASTROPHIC -34.7pp
cluster_verify:        false
temporal_split:        false
fac:                   false         # -2.5pp
reranking:             false         # hurts vehicles
dba:                   false         # zero effect

# Stage 5: Post-Processing
min_trajectory_frames:     40
cross_id_nms_iou:          0.40
min_trajectory_confidence: 0.30
min_submission_confidence: 0.15
stationary_filter:         true (disp=150, vel=2.0)
track_edge_trim:           false
track_smoothing:           false
mtmc_only_submission:      false     # true drops ~5pp!
gt_frame_clip:             true
gt_zone_filter:            true

# Stage 1: Tracker
min_hits:              2
confidence_threshold:  0.25
max_gap:               50
max_iou_distance:      0.7
intra_merge_time_gap:  40
denoise:               false

# Stage 2: Features
reid_model:            TransReID ViT-Base/16 CLIP @ 256x256
secondary_model:       OSNet (score-level @ 10%)
pca_n_components:      384
power_norm:            0.5
pooling:               CLS + GeM patches -> 1536D -> PCA 384D
crops_per_tracklet:    48
quality_temperature:   3.0
flip_augment:          true
clahe_clip_limit:      2.5
```

---

## 2. Complete Experiment History

### 2.1 10c Kaggle Version History

| 10c Ver | Code Ver | Date | Config Changes from Baseline | MTMC IDF1 | Verdict | Key Insight |
|:-------:|:--------:|:----:|------------------------------|:---------:|:-------:|-------------|
| v14-v22 | v54-v66 | 2026-02 | Initial grid scans (sim_thresh, app_w, bridge, gallery, orphan, rerank, algorithm). 384 combos. | ~74-78% | BASELINE | AQE_K=3 only win |
| v23-v26 | v67 | 2026-02 | Algorithm scan: CC, conflict_free_cc, Louvain (5 res), agglomerative | ~78% | conflict_free_cc +0.21pp | All algorithms within 0.3pp |
| v27-v30 | v68-v69 | 2026-02 | Post-proc scan: min_traj_frames 5-80, NMS IoU, stationary filter | ~78% | frames=40 optimal | stationary disp=200 catastrophic |
| v31-v33 | v70-v71 | 2026-02 | FIC reg sweep (0.05-15.0), sim_thresh refinement | ~78% | fic=0.1, sim=0.53 | FIC reg barely matters near 0.1 |
| v34 | v72 | 2026-02 | Intra-merge scan (36 configs: thresh x gap) | **78.28%** | thresh=0.80, gap=30 | Higher thresh = better |
| v35 | - | 2026-02 | PCA 512D test | 77.5% | REJECTED (-0.78pp) | 512D hurts |
| v36 | v73 | 2026-02 | app_w / DBA / mutual_nn top_k scan (14 configs) | 78.01% | app_w=0.70 | DBA=zero effect |
| v37 | v73 | 2026-02 | Clean confirmation run | 78.0% | Baseline confirmed | - |
| v38 | v74 | 2026-03 | CSLS, cluster_verify, temporal_split, gallery_exp, length_weight | 78.02% | ALL NO-OPS | CSLS catastrophic (-34.7pp) |
| v39 | v75 | 2026-03 | Consolidated + temporal_overlap / AQE_K extended | 78.01% | TO=0.05, K=3 | K>3 hurts |
| v40 | v76 | 2026-03 | quality_temperature=5.0, laplacian_min_var=50.0 | 77.3% | REJECTED (-0.7pp) | Quality temp hurt |
| v41 | v77 | 2026-03 | Tracker: max_gap=50, intra_merge_time=40 | **78.2%** | ACCEPTED | ID switches 131->99 |
| v42 | v78 | 2026-03 | Tracker: max_gap=80, merge_time=60 | 75.0% | REJECTED (-3.0pp) | TOO AGGRESSIVE |
| v43 | v79 | 2026-03 | Tracker: max_gap=60, merge_time=50 | 77.3% | REJECTED (-0.9pp) | Overshot |
| v44 | v80 | 2026-03 | Tracker: min_hits=2 | **78.4%** | **BEST KAGGLE** | +0.2pp |
| v45 | v81 | 2026-03 | confidence_threshold=0.20 | 75.6% | REJECTED (-2.8pp) | Noise |
| v46 | v82 | 2026-03 | denoise=true | 75.7% | REJECTED (-2.7pp) | Harmful |
| v47 | v83 | 2026-03 | max_iou_distance=0.5 | 76.8% | REJECTED (-1.6pp) | Too tight |
| v48 | v84 | 2026-03 | 384px model + PCA refit (mAP=44.9%) | ~74-75% | REJECTED | Bad model |

### 2.2 Recent Feature Experiments (local pipeline with CamTTA)

| Version | Config Changes | MTMC IDF1 | Verdict | Key Insight |
|:-------:|----------------|:---------:|:-------:|-------------|
| 10c v27 | PCA=512D | 74.7% | REJECTED (-3pp) | Higher PCA dims hurt |
| 10c v28 | CamTTA=ON, power_norm=0.5 | **77.7%** | LOCAL BEST | CamTTA + power_norm |
| 10c v29 | CamTTA=ON, power_norm=0 | 77.2% | REJECTED | Power norm +0.5pp |
| 10c v30 | CamTTA + MS-TTA [[320,320]] | 77.71% | NEUTRAL | MS-TTA useless |
| 10c v31 | Association param sweep (31 configs) | **78.0%** | ACCEPTED (+0.2pp from gallery_thresh only) | Core association params unchanged; gallery=0.48/orphan=0.38 |

Note: On CamTTA + power_norm features, the existing association optimum was effectively confirmed again: `sim_thresh=0.53`, `appearance_weight=0.70`, `fic_reg=0.10`, `aqe_k=3`, and `intra_merge=true` with `thresh=0.80` and `gap=30` stayed best or tied-best. The only measurable gain was a tiny +0.2pp from lowering `gallery_thresh` to 0.48 with `orphan_match=0.38`.

---

## 3. Exhaustive Parameter Sweep Results

### 3.1 Stage 4: Association Parameters (220+ configs tested - EXHAUSTED)

| Parameter | Values Tested | Optimal | Effect | Versions |
|-----------|:-------------|:-------:|--------|:--------:|
| `sim_thresh` | 0.35, 0.40, 0.45, 0.48, 0.50, 0.51, 0.52, **0.53**, 0.55, 0.58, 0.60, 0.65 | **0.53** | 0.50=-0.3pp, 0.55=-0.2pp | v58,v63,v70,v82 |
| `appearance_weight` | 0.60, 0.65, **0.70**, 0.75, 0.80, 0.85, 0.90, 0.95 | **0.70** | +0.76pp vs 0.75 | v65,v73 |
| `FIC regularisation` | 0.05, **0.1**, 0.2, 0.3, 0.5, 1.0, 3.0, 5.0, 10.0, 15.0 | **0.1** | +0.08pp vs 3.0 | v70-v71 |
| `AQE_K` | 2, **3**, 4, 5, 7, 10 | **3** | +0.9pp (2->3); k=5: -0.57pp | v59-v61,v75 |
| `QE alpha` | 1.0, 2.0, 3.0, **5.0**, 7.0 | **5.0** | All alternatives worse | v64,v72 |
| `fusion_weight` (OSNet) | **0.10**, 0.15, 0.20, 0.25, 0.30 | **0.10** | Higher = worse | v52-v53,v61 |
| `intra_merge thresh` | 0.60, 0.65, 0.70, 0.75, **0.80**, 0.85, OFF | **0.80** | +0.14pp vs 0.75 | v72 |
| `intra_merge gap` | **30**, 60, 90, 120, 180 | **30** | Gap-insensitive | v72 |
| `bridge_prune_margin` | **0.0**, 0.05, 0.10 | **0.0** | Pruning -1.4pp | v46 |
| `max_component_size` | 8, **12**, 16, 20 | **12** | Stable | v25 |
| `mutual_nn top_k` | 10, 15, **20**, 25, 30, 40 | **20** | ALL IDENTICAL | v73 |
| `gallery_expansion thresh` | 0.40, 0.45, **0.50**, 0.55, 0.60 | **0.50** | Lower hurts | v74 |
| `length_weight_power` | 0.0, 0.1, **0.3**, 0.5, 0.7, 1.0 | **0.3** | 0.0=-0.4pp | v74 |
| `temporal_overlap bonus` | 0.0(off), 0.02, **0.05**, 0.10, 0.15, 0.20 | **0.05** | OFF=-0.9pp! | v75 |
| `algorithm` | CC, **conflict_free_cc**, Louvain(5 res), agglomerative | **conflict_free_cc** | +0.21pp vs CC | v67 |

#### Catastrophic / Harmful Features (Stage 4)

| Feature | Result | Versions |
|---------|--------|:--------:|
| **CSLS** | **-34.7pp CATASTROPHIC** | v74 |
| **Hierarchical centroid** | **-1.0 to -5.1pp** | v54-v56,v62 |
| **FAC** | **-2.5pp** | v26 |
| **Reranking** | Hurts vehicles | v25 |
| **Camera bias** | **-0.4pp** | v54-v57 |
| **Zone model (auto)** | **-0.4pp** | v54-v57 |
| **DBA** | ZERO effect | v73 |
| **Cluster verify** | NO EFFECT | v74 |
| **Temporal split** | ZERO EFFECT | v74 |

### 3.2 Stage 5: Post-Processing

| Parameter | Values Tested | Optimal | Effect |
|-----------|:-------------|:-------:|--------|
| `min_trajectory_frames` | 5-80 (12 values) | **40** | Peaks at 40 |
| `cross_id_nms_iou` | 0.30-0.50 | **0.40** | +0.02pp |
| `stationary_filter disp` | 100-250 | **150** | 200=CATASTROPHIC (-1.5pp) |
| `track_edge_trim` | on/off | **off** | Hurts |
| `track_smoothing` | on/off | **off** | Hurts |
| `mtmc_only_submission` | true/false | **false** | true = **-5pp** |

### 3.3 Stage 1: Tracker Parameters

| Parameter | Values Tested | Optimal | Effect |
|-----------|:-------------|:-------:|--------|
| `min_hits` | **2**, 3 | **2** | +0.2pp |
| `confidence_threshold` | 0.20, **0.25** | **0.25** | 0.20 = -2.8pp |
| `denoise` | true, **false** | **false** | true = -2.7pp |
| `max_iou_distance` | 0.5, **0.7** | **0.7** | 0.5 = -1.6pp |
| `max_gap` | 30, **50**, 60, 80 | **50** | 80=-3pp |

### 3.4 Stage 2: Feature Extraction

| Parameter | Values Tested | Optimal | Effect |
|-----------|:-------------|:-------:|--------|
| `PCA n_components` | 256, **384**, 512 | **384** | 512 = -0.78 to -3pp |
| `ReID input resolution` | **256x256**, 384x384 | **256x256** | 384 = -1.3pp (not trained) |
| `Multi-scale TTA` | off, various | **off** | Always neutral/harmful |
| `Ensemble method` | **score-level 10%**, feature concat | **score-level** | Concat = -1.6pp |
| `quality_temperature` | **3.0**, 5.0 | **3.0** | 5.0 = -0.7pp |
| `power_norm` | 0.0, **0.5** | **0.5** | +0.5pp |
| `CamTTA` | enabled, **disabled** | **disabled** | Helps GLOBAL, hurts MTMC |

---

## 4. Training History

### 4.1 TransReID ViT-B/16 CLIP (09b)

| Version | Config | mAP | R1 | Status |
|:-------:|--------|:---:|:--:|:------:|
| 09b v1 | 384px from 256px checkpoint | 44.94% | - | FAILED |
| 09b v2 | 384px ViT-B/16 CLIP proper | **80.14%** | **92.27%** | SUCCESS |

### 4.2 Knowledge Distillation (09c)

| Version | Config | mAP | Status |
|:-------:|--------|:---:|:------:|
| 09c v1 | ViT-L/14 teacher -> ViT-B/16 student | 22% | ABANDONED (dim mismatch, bad temp) |

### 4.3 ResNet101-IBN-a (09d)

| Version | Account | Result | Issue |
|:-------:|:-------:|:------:|-------|
| v8 | mrkdagods | Had ckpt_e09 | Original |
| v9-v11 | mrkdagods | ERROR ~13min | Model download bug |
| v1-v2 (ali369) | ali369 | ERROR ~13min | Checkpoint ordering + download |
| v5 (ali369) | ali369 | COMPLETE | All 6 STEPs OK |
| v12 | mrkdagods | mAP=**21.9%** | IBN layer3 bug (layer1+2 only) |
| v13 | mrkdagods | mAP=11.98% (epoch 19 only, timed out) | IBN fix confirmed (layer1+2+3 all have IBN keys). Training recipe is the bottleneck - 12% mAP at epoch 20 is very low even for early training. |


The ResNet101-IBN-a training recipe itself needs investigation - both v12 (21.9% at e60) and v13 (12% at e19) show the model is not converging. Possible causes: wrong LR schedule, insufficient augmentation, or wrong optimizer settings. The ensemble plan is BLOCKED until training quality improves.

---

## 5. What Has Been Exhaustively Tried (DO NOT RE-TEST)

- All association parameters in Section 3.1 (220+ configs)
- All post-processing parameters in Section 3.2
- All tracker core parameters in Section 3.3
- PCA dimensions: 256D, 384D, 512D
- Algorithm variants: CC, conflict_free_cc, Louvain, agglomerative
- CamTTA: helps GLOBAL, hurts MTMC
- Multi-scale TTA: always neutral/harmful
- Track smoothing/edge trim: always harmful
- Feature concat ensemble: -1.6pp
- CSLS: catastrophic
- Hierarchical clustering: always harmful
- FAC: -2.5pp
- DBA: zero effect

---

## 6. What Has Been Tried Poorly / Once (Worth Revisiting)

| Approach | What Happened | Why Revisit | Priority |
|----------|---------------|-------------|:--------:|
| 384x384 Native Training | 09b v1: mAP=44.9% (wrong init) | All SOTA use 384px | **HIGH** |
| Knowledge Distillation | 09c: 22% mAP (dim mismatch) | All AIC24 top-3 used KD | **HIGH** |
| Score-Level Ensemble (2+ models) | Only OSNet@10% | Need proper 2nd backbone | **HIGH** |
| Multi-Scale TTA | Marginal + timeout | Selective approach needed | LOW |
| CamTTA | Hurts MTMC, helps GLOBAL | CLOSED | CLOSED |

---

## 7. What Has NEVER Been Tried (Real Opportunity Space)

| Approach | Stage | Est. Gain | Priority |
|----------|:-----:|:---------:|:--------:|
| 384px ViT-B/16 (proper training) | Training->S2 | +1.0-2.0pp | ***** |
| ResNet101-IBN-a ensemble | Training->S2->S4 | +1.0-1.5pp | ***** |
| Circle loss training | Training | +0.3-0.5pp | **** |
| Timestamp bias correction | S4 | +0.3-0.5pp | **** |
| Per-camera CLAHE tuning | S0 | +0.2-0.5pp | *** |
| Hand-annotated zone polygons | S4 | +0.5-1.5pp | *** |
| SAM2 foreground masking | S2 | +0.3-0.5pp | *** |
| CID_BIAS per camera pair | S4 | +0.5-1.0pp | *** |
| DMT camera-aware training | Training | +1.0-1.5pp | *** |
| Network flow / Hungarian | S4 | +0.3-1.0pp | ** |
| GNN edge classification | S4 | +1.0-3.0pp | ** |
| AFLink tracklet linking | S4 post | +0.3-0.5pp | ** |

---

## 8. Key Insights & Dead Ends

1. **"Association tuning is exhausted — the gap is feature quality."** 220+ configs; remaining gap needs better embeddings.
2. **Power normalization (+0.5pp)**: signed sqrt before PCA spreads similarity distribution.
3. **IBN-a layer3 bug**: 09d v12's 21.9% mAP was from IBN on layers 1+2 only. Fixed in v13.
4. **CamTTA hurts MTMC**: Adapts BN per camera -> features become camera-specific. FIC whitening in S4 is correct approach.
5. **Multi-scale TTA is useless**: Tested 4+ times, never helped.
6. **PCA=512D hurts by 0.78-3pp**. 384D is optimal.
7. **mtmc_only=True drops IDF1 by ~5pp**: Discards single-camera tracklets.
8. **CSLS is CATASTROPHIC (-34.7pp)**: Destroys similarity structure.
9. **Feature concat ensemble hurts (-1.6pp)**: Score-level fusion is correct.
10. **Reranking hurts with weak features**: Re-test after feature quality improvement.
11. **S02_c006 is catastrophic (74% IDF1)**: FP ratio 6.86x, GT covers only 43% of video.
12. **Fragmentation dominates**: Under-merging errors 1.7-2.5x over-merging. Feature quality problem.

### Gap Attribution

| Source | Est. IDF1 Cost |
|--------|:--------------:|
| Feature quality (single model, 256px, no KD) | 3.0-4.0pp |
| Spatio-temporal imprecision | 1.0-2.0pp |
| Association algorithm ceiling | 0.5-1.0pp |

---

## 9. Error Profile

| Error Type | Count | Meaning |
|:----------:|:-----:|---------|
| Fragmented GT IDs | 44-87 | Under-merging (features too dissimilar) |
| Conflated pred IDs | 26-35 | Over-merging (threshold too loose) |
| ID Switches | 99-152 | Frame-level identity swaps |

---

## 10. Currently Running / In-Progress

| Item | Status | Account | Details |
|------|:------:|:-------:|---------|
| 10c v31 | COMPLETE | ali369 | Association sweep complete: mtmc_idf1=78.0%; only gallery_thresh improved (+0.2pp) |
| 09d v13 | COMPLETE | mrkdagods | Timed out after epoch 19: mAP=11.98%; IBN layer1+2+3 fix confirmed, but training recipe is not converging |
| Next | BLOCKED | - | Ensemble plan blocked until ResNet101-IBN-a training quality improves |

---

## 11. Quick Lookup Index

| Question | Answer | Section |
|----------|--------|:-------:|
| sim_thresh=0.55? | Yes, -0.2pp vs 0.53 | 3.1 |
| PCA=512? | Yes, hurt -0.78 to -3pp | 3.4 |
| 384x384 inference? | Yes, -1.3pp (not trained) | 3.4 |
| 384x384 training? | Poorly (mAP=44.9%). Needs redo. | 4.1 |
| Reranking? | Hurts vehicles. Re-test after upgrade. | 3.1 |
| CSLS? | CATASTROPHIC -34.7pp | 3.1 |
| CamTTA? | Hurts MTMC. Keep off. | 3.4 |
| Hierarchical? | -1.0 to -5.1pp always | 3.1 |
| Louvain? | Identical to CC | 3.1 |
| Zone model? | Auto-zones hurt. Hand-annotated never tried. | 3.1 |
| Camera bias? | -0.4pp. CID_BIAS per-pair never tried. | 3.1 |
| FAC? | -2.5pp | 3.1 |
| mtmc_only? | -5pp. Always false. | 3.2 |
| Knowledge distillation? | Tried poorly (22%). Needs fix. | 4.2 |
| Circle loss? | Never used. Code exists. | 7 |
| ResNet101-IBN-a? | v13 complete: 11.98% at e19; recipe needs investigation. | 4.3 |
| Feature concat? | -1.6pp. Don't repeat. | 3.4 |
| Track smoothing? | Harmful. Don't repeat. | 3.2 |
| Power normalization? | +0.5pp. Enabled. | 3.4 |
| Multi-scale TTA? | Always neutral/harmful. | 3.4 |
| Denoise? | -2.7pp. Keep off. | 3.3 |