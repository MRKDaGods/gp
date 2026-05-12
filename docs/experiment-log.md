# MTMC Tracker — Comprehensive Experiment Log

> **Purpose**: Prevent re-running experiments. Every parameter combination and approach is logged here.
> **Last updated**: 2026-05-11
> **Current verified best (Kaggle)**: MTMC IDF1 = **77.94% headline** (14e B1 / 14f A20 / 14h M0 / 14i F0 / 14j W0 / 14k K0 / 14u U0, TTA features + `aqe_k=2`, `w_tertiary=0.525`, `similarity_threshold=0.48`, `fic_regularisation=0.5`); 14k K7 reached a non-promoted **78.08% MARGINAL** point in 4-way fusion
> **VeRi-776 reproducible best**: **mAP=0.9330 / R1=0.9845** (14t fusion, 2026-05-11)
> **Historical best (not reproducible)**: MTMC IDF1 = **78.4%** (10c v44 / ali369 / code v80, depended on the now-unavailable `vehicle_osnet_veri776.pth` checkpoint)
> **Current best (local, recent)**: MTMC IDF1 = **77.7%** (10c v28, CamTTA + power_norm=0.5)
> **Historical local claim**: IDF1 = 82.97% (v47 — unverifiable, predates current experiment log)
> **SOTA target**: IDF1 ≈ 84.1% (AIC21 1st place) / 84.86% (AIC22 1st)
> **Gap**: ~7.1–7.8pp — caused by **feature quality**, NOT association tuning
> **Total experiments**: 294+ configs across ~14h GPU plus VeRi-776 14p3/14q/14r single-model chase, 14t fusion WIN, and 14u CityFlow port FAIL

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

**Reference note**: The configuration below preserves the historical v80/v44 setup for comparison. It depended on `vehicle_osnet_veri776.pth`, a CityFlowV2-adapted OSNet checkpoint that is no longer present in `gumfreddy/mtmc-weights` or `mrkdagods/mtmc-weights`, so it is not currently reproducible.

**Historical Kaggle best**: IDF1 = **78.4%** — 10c v44, ali369 account, code v80, 2026-03

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

### 2.3 2026-04 Augoverhaul Downstream Follow-Ups

| Version | Upstream Model | Config Changes | MTMC IDF1 | Verdict | Key Insight |
|:-------:|----------------|----------------|:---------:|:-------:|-------------|
| 10c v48 | 09 v2 augoverhaul @ 256px | Corrected deployment size, 11-config association re-sweep | 72.2% | REJECTED | Fixing the 384px deployment bug did not rescue the model |
| 10c v49 | 09 v3 augoverhaul-EMA (`mAP=81.53%`, `R1=92.41%`) | Broader association sweep; best config `sim=0.45`, `app=0.60`, `st=0.40`, `fic=1.00`, `aqe_k=3`, `gallery=0.45`, `orphan=0.35`; AFLink `gap=150`, `dir=0.85` improved 0.675 -> 0.722 | 72.2% | REJECTED | Same 0.722 ceiling as v48; reranking off, camera-pair norm off, intra-merge negligible, so the augoverhaul model family is the bottleneck rather than association tuning |

### 2.4 2026-04 Controlled AFLink Retest on Restored Baseline

| Version | Upstream Model | Config Changes | MTMC IDF1 | Verdict | Key Insight |
|:-------:|----------------|----------------|:---------:|:-------:|-------------|
| 10c v52 AFLink addon retest | Fresh baseline features from 10a v30 | Exact v52 baseline recipe `sim=0.50`, `app=0.70`, `fic=0.50`, `aqe_k=3`, `gallery=0.48`, `orphan=0.38`; pure AFLink addon sweep over `gap/dir = 100/0.90, 150/0.85, 200/0.70` | Control **77.14%**; AFLink **73.32%**, **71.83%**, **63.94%** | REJECTED | Clean retest removes the v46 confound: even tight AFLink hurts **-3.82pp**, and wider gaps make false cross-camera merges much worse |

Note: The control run for this retest measured **IDF1 = 0.7921** and **HOTA = 0.5747** with AFLink disabled. This was a pure post-association addon test on the exact restored baseline operating point, not a joint association sweep.

### 2.5 2026-04 Structural Association Follow-Up

| Version | Upstream Model | Config Changes | MTMC IDF1 | Verdict | Key Insight |
|:-------:|----------------|----------------|:---------:|:-------:|-------------|
| 10c v53 | Fresh baseline features from 10a v30 | Replaced CC/conflict-free merge logic with a network-flow / Hungarian-style solver plus merge verification, compared directly against the controlled v52 CC baseline | CC baseline **77.14%**; network flow **76.9%** | REJECTED | Network flow is only **-0.24pp** on MTMC IDF1 and slightly improves MOTA/HOTA, but it fails its purpose by increasing conflation from **27 -> 30** predicted IDs instead of reducing false merges |

### 2.6 2026-04 CID_BIAS Topology Bias Follow-Up

| Version | Upstream Model | Config Changes | MTMC IDF1 | Verdict | Key Insight |
|:-------:|----------------|----------------|:---------:|:-------:|-------------|
| 10c v55 | Fresh baseline features from 10a v30 | Added topology-derived additive CID_BIAS terms on top of the restored baseline recipe; tested conservative **(+0.02/-0.10)**, default **(+0.04/-0.15)**, and aggressive **(+0.06/-0.20)** bias schedules against a no-bias control | Control **77.4%**; conservative **76.4%**; default **76.2%**; aggressive **76.3%** | REJECTED | All additive bias variants hurt by **-1.0 to -1.2pp**. FIC whitening already provides the useful camera calibration, and extra CID_BIAS offsets distort the calibrated similarity space |

### 2.7 2026-04 Improved R50-IBN Fusion Follow-Up

| Version | Upstream Model | Config Changes | MTMC IDF1 | Verdict | Key Insight |
|:-------:|----------------|----------------|:---------:|:-------:|-------------|
| 10c v60 | 09n FastReID SBS R50-IBN on 10a v37 -> 10b v22 | First fine-tuned R50-IBN fusion sweep across **w in [0.00, 0.05, ..., 0.50]** | Best **77.36%** at **w=0.10** | REJECTED | Fine-tuned R50-IBN improved the secondary model, but MTMC gain stayed marginal at only **+0.06pp** |
| 10c v61 | Improved 09p R50-IBN path on 10a `run_kaggle_20260420_201401` -> 10b v23 | Repeated the fusion sweep with the newer 10a chain and improved 09p secondary embeddings | Best **77.3595%** at **w=0.10** | REJECTED | Even with improved secondary training and more robust ingestion, the gain over **w=0.00** is only **+0.0006pp**, confirming the ceiling is still dominated by primary feature quality rather than association tuning |

### 2.8 2026-04-25 DINOv2 Downstream MTMC Evaluation

| Version | Upstream Model | Config Changes | MTMC IDF1 | Verdict | Key Insight |
|:-------:|----------------|----------------|:---------:|:-------:|-------------|
| 10c DINOv2 v2 baseline | 09s v1 DINOv2 ViT-L/14 @ 252px (mAP=86.79%, R1=96.15%) | v52 baseline params (`sim=0.46`, `app=0.60`, `fic=0.20`, `aqe_k=3`, `gallery=0.48`, `orphan=0.38`, `intra_merge=True/0.80`), no AFLink | 0.688 | REJECTED | Per-camera IDF1=0.794 (single-cam quality strong); MTMC -8.7pp vs ViT-B/16 CLIP baseline despite +6.65pp mAP |
| 10c DINOv2 v2 best | 09s v1 DINOv2 ViT-L/14 @ 252px | Same params + AFLink `gap=150px`, `dir_cos=0.85` | **0.744** | REJECTED | AFLink surprisingly helps DINOv2 (+5.6pp: 0.688→0.744); best full metrics: IDF1=0.755, MOTA=0.624, HOTA=0.547; still -3.1pp vs ViT-B/16 CLIP (0.775) |

### 2.9 2026-05-07 SAM2 Foreground Masking at Stage 2

| Version | Upstream Model | Config Changes | MTMC IDF1 | Verdict | Key Insight |
|:-------:|----------------|----------------|:---------:|:-------:|-------------|
| 14a v8 | 10a v7 Stage 0/1 outputs (`run_kaggle_20260425_202123`) + TransReID 256px CLIP/DINOv2 fusion | SAM2 `sam2_hiera_base_plus` box-prompt masking during Stage 2 ReID extraction; 5px dilation, `bbox_expand=1.10`, zeros background; downstream production-like `w_tertiary=0.60` fusion | **0.7647** | DEAD END | Clean foreground masks removed useful cross-camera context; -0.56pp vs production 0.7703 despite preserving vehicle silhouette |

The split between `trackeval_idf1=0.7866` and `mtmc_idf1=0.7647` suggests within-camera identity consistency held while cross-camera association regressed. Results are recorded in `outputs/14a_v8_summary/14a_summary.json`.

### 2.10 2026-05-07 Multi-Crop TTA at Stage 2

| Version | Upstream Model | Config Changes | MTMC IDF1 | Verdict | Key Insight |
|:-------:|----------------|----------------|:---------:|:-------:|-------------|
| 14c v2 | 10a v7 Stage 0/1 outputs (`run_kaggle_20260425_202123`) + TransReID 256px CLIP/DINOv2 fusion | Multi-crop TTA at Stage 2: primary {original, hflip, scale_0.95, scale_1.05}, tertiary DINOv2 {original, hflip}, L2-mean aggregation; downstream production-like `w_tertiary=0.60`, `sim_thresh=0.50` fusion | **0.77085** | MARGINAL | +0.05pp vs production 0.7703 — within ~0.24pp run-to-run noise; id_switches regressed 158→212 suggesting TTA smoothing slightly hurt within-camera disambiguation; Stage 2 walltime 99.42 min (faster than 14a SAM2 at 146.67 min) |

Per `docs/subagent-specs/post-14a-next.md` stop criteria, 0.77085 falls in the NEUTRAL band (0.7680–0.7720). Before escalating to a GNN edge classifier, the next experiment is a CPU-only fusion-weight sweep on the existing 14c v2 features (`14d`, see `docs/subagent-specs/post-14c-next.md`). Results recorded in `outputs/14c_v2_summary/14c_summary.json`.


### 2.11 2026-05-07 14d v1 — CPU Fusion Sweep on 14c v2 TTA Features

CPU-only sweep on the existing 14c v2 TTA Stage 2 outputs (no Stage 0/1/2 re-run). 8 configs across `w_tertiary∈{0.55,0.60,0.65,0.70}` and `sim_thresh∈{0.40,0.50}`. Total walltime 4.9 min on Kaggle CPU.

| Config | `w_tertiary` | `sim_thresh` | MTMC IDF1 | trackeval_idf1 | id_switches | Verdict |
|:------:|:------------:|:------------:|:---------:|:--------------:|:-----------:|:-------:|
| C0 control | 0.60 | 0.50 | 0.77085 | 0.7881 | 212 | reproduces 14c v2 |
| C1 | 0.55 | 0.50 | 0.77149 | 0.7896 | 214 | +0.06pp |
| C2 | 0.65 | 0.50 | 0.77124 | 0.7885 | 212 | +0.04pp |
| **C3 best** | **0.50** | **0.50** | **0.77155** | **0.7897** | 214 | **+0.07pp vs C0, +0.13pp vs production** |
| C4 | 0.70 | 0.50 | 0.77115 | 0.7887 | 212 | +0.03pp |
| C5 | 0.60 | 0.40 | 0.7566 | 0.7775 | 217 | -1.42pp (threshold dominates) |
| C6 | 0.55 | 0.40 | 0.7566 | — | — | -1.42pp |
| C7 | 0.65 | 0.40 | 0.7566 | — | — | -1.42pp |

**Verdict**: MARGINAL POSITIVE per `docs/subagent-specs/post-14c-next.md` stop criteria — best config 0.77155 falls in the NEUTRAL band (0.7680–0.7720) but the lift is consistent across `w_t∈[0.50,0.70]` at `thr=0.50`, the optimum shifted from production `w_t=0.60` to `w_t=0.50` on TTA features, and trackeval IDF1 rose +0.31pp. The `thr=0.40` family is universally -1.4pp worse, confirming the production threshold is correct. Next experiment is `14e`, a tighter `w_t × thr` grid around C3 plus AQE/FIC re-tune at the new optimum (single CPU kernel, ~10-15 min, zero GPU). Results recorded in `outputs/14d_v1_summary/14d_summary.json`. Spec: `docs/subagent-specs/post-14d-next.md`.

### 2.12 2026-05-07 14e v1 — Expanded TTA Fusion + AQE/FIC Sweep (WIN)

CPU-only sweep on the existing 14c v2 TTA Stage 2 outputs (no Stage 0/1/2 re-run). Two blocks: Block A is a fine `w_tertiary × similarity_threshold` grid at production AQE/FIC; Block B is an AQE/FIC sweep anchored at the Block A best.

#### Block A — fine `w_t × thr` at `aqe_k=3, fic_reg=0.5` (12 configs, all flat)

| Label | w_tertiary | sim_thresh | MTMC IDF1 | id_switches |
|:-----:|:----------:|:----------:|:---------:|:-----------:|
| A1 | 0.45 | 0.48 | 0.77157 | 213 |
| A2 | 0.45 | 0.50 | 0.77155 | 214 |
| A3 | 0.45 | 0.52 | 0.77167 | 212 |
| A4 | 0.475 | 0.48 | 0.77157 | 213 |
| A5 | 0.475 | 0.50 | 0.77155 | 214 |
| A6 | 0.475 | 0.52 | 0.77167 | 212 |
| A7 | 0.50 | 0.48 | 0.77157 | 213 |
| A8 | 0.50 | 0.50 | 0.77155 | 214 |
| A9 | 0.50 | 0.52 | 0.77070 | 212 |
| **A10** | **0.525** | **0.48** | **0.77171** | **213** |
| A11 | 0.525 | 0.50 | 0.77170 | 214 |
| A12 | 0.525 | 0.52 | 0.77100 | 212 |

A8 reproduces 14d C3 at 0.77155 (drift check ✓). All Block A points cluster in 0.7707–0.7717 (~0.001 spread) — fusion-weight surface is flat at production AQE/FIC.

#### Block B — AQE/FIC sweep at A10 anchor (w_t=0.525, thr=0.48)

| Label | aqe_k | fic_reg | MTMC IDF1 | id_switches | trackeval_idf1 | Δ vs A10 |
|:-----:|:-----:|:-------:|:---------:|:-----------:|:--------------:|:--------:|
| **B1** | **2** | **0.5** | **0.77936** | **154** | **0.79461** | **+0.77pp** |
| B2 | 4 | 0.5 | 0.77052 | 162 | 0.79255 | -1.19pp |
| B3 | 3 | 0.3 | 0.77268 | 213 | 0.79036 | +0.10pp |
| B4 | 3 | 0.7 | 0.77192 | 215 | 0.79005 | +0.02pp |

**Verdict**: WIN per `docs/subagent-specs/post-14d-next.md` thresholds (≥0.7720). **B1 = 0.77936** is +0.91pp vs production 0.7703 and +0.78pp vs the 14d floor 0.77155, with ID-switch count dropping from 213 → **154** (-28%). The discovery is **AQE k=2** unlocks the gain on TTA features — production has been at k=3 forever; TTA pre-smooths the embedding so production k=3 was over-smoothing. k=4 makes it worse (162 IDS, 0.77052). FIC sensitivity is small (B3 vs B4 differ by 0.0008 IDF1 at k=3). New reproducible headline = **0.77936**; production-deployed 0.7703 retained as previous baseline. Next experiment: 14f tighter AQE sweep around k=2 plus k=1 probes — spec `docs/subagent-specs/post-14e-next.md`. Results: `outputs/14e_v1_summary/14e_summary.json`.

### 2.13 2026-05-07 14f v1 — TTA AQE Confirmation Sweep (NEUTRAL, Plateau Confirmed)

CPU-only 54-config sweep on 14c v2 TTA features (45 Block A confirmation cells at `aqe_k=2`, 9 Block B `aqe_k=1` probes at FIC=0.5). Drift check passed: **A20 (B1 replicate at `w_t=0.525, thr=0.48, aqe_k=2, fic_reg=0.5`) = 0.77936 exact, id_switches=154 exact** — kernel and metric pipeline are stable.

**Plateau structure (8 ties at 0.77936, all with `id_switches=154`)**: at `aqe_k=2, w_t=0.525`, configs with `fic_reg ∈ {0.3, 0.4, 0.5, 0.6}` × `thr ∈ {0.46, 0.48}` all hit 0.77936. `thr=0.50` regresses by ~0.0012 with the same id_switches. `fic_reg=0.7` and `w_t ∈ {0.50, 0.55}` cluster at 0.7791 (~−0.0003pp). Stage-4 `(w_t, thr, FIC)` axis at `aqe_k=2` is **fully saturated** — no measurable signal remaining.

**k=1 family (9 configs, universally worse)**: MTMC IDF1 ∈ [0.76933, 0.77059], id_switches ∈ [143, 193]. AQE k=1 over-trims the gallery and re-introduces noise. **k=2 is the discrete optimum** — concave around k=2 (k=1 worse, k=3 worse, k=4 even worse per 14e B2).

**Verdict**: NEUTRAL per `docs/subagent-specs/post-14e-next.md` thresholds (best in 0.7785–0.7795 band by ≤0.001; effectively at-baseline). **The TTA × Stage-4-tuning family is EXHAUSTED at 0.77936** — this is a *confirmed-win plateau*, not a dead end. 0.77936 stays as the new reproducible headline. Stage-4 hyperparameter tuning cannot close the 0.77936 → 0.7810 gap. Next experiment is **14g DINOv2 4-view TTA expansion** at Stage 2 (GPU) — symmetrize the tertiary DINOv2 stream from 2 views {original, hflip} to 4 views {original, hflip, scale_0.95, scale_1.05} matching the primary, on the same Stage-2 TTA axis that delivered 14e's +0.91pp. With `w_t=0.525` the tertiary stream is ~half the fused score, so improving its stability has a fusion-level multiplier. Spec: `docs/subagent-specs/post-14f-next.md`. Results: `outputs/14f_v1_summary/14f_summary.json`.

### 2.14 2026-05-08 14g v1 — DINOv2 4-View TTA Expansion at Stage 2 (NEUTRAL)

GPU P100 Stage-2 rerun on the 14c v2 TTA recipe with the tertiary DINOv2 ViT-L/14 stream symmetrized from 2 views `{original, hflip}` to 4 views `{original, hflip, scale_0.95, scale_1.05}` (matching the primary CLIP TransReID stream). Followed by an 8-config CPU mini-sweep at the 14e B1 anchor.

**Drift check**: S0 (B1 anchor: `w_t=0.525, thr=0.48, aqe_k=2, fic_reg=0.5`) on the new tertiary features = **0.77902** MTMC IDF1, drift **−0.00034** vs 14e B1 0.77936 — within ±0.005 tolerance, gate passed.

**Mini-sweep result table (8 configs at the new feature build)**:

| Label | aqe_k | w_tertiary | sim_thresh | fic_reg | MTMC IDF1 | id_switches | Δ vs 14e B1 |
|:-----:|:-----:|:----------:|:----------:|:-------:|:---------:|:-----------:|:-----------:|
| S0 (anchor) | 2 | 0.525 | 0.48 | 0.5 | 0.77902 | 154 | −0.00034 |
| S1 | 2 | 0.500 | 0.48 | 0.5 | 0.77902 | 154 | −0.00034 |
| **S2** | **2** | **0.550** | **0.48** | **0.5** | **0.77926** | **154** | **−0.00010** |
| S3 | 2 | 0.575 | 0.48 | 0.5 | 0.77926 | 154 | −0.00010 |
| S4 | 2 | 0.525 | 0.46 | 0.5 | 0.77926 | 154 | −0.00010 |
| S5 | 2 | 0.525 | 0.50 | 0.5 | 0.77805 | 154 | −0.00131 |
| S6 | 3 | 0.525 | 0.48 | 0.5 | 0.77149 | 213 | −0.00787 |
| S7 | 2 | 0.525 | 0.48 | 0.4 | 0.77902 | 154 | −0.00034 |

**Verdict**: NEUTRAL per `docs/subagent-specs/post-14f-next.md` thresholds (best ∈ [0.7785, 0.7795]). No promotion. The 0.77936 14e B1 headline stands.

**Key signal**: every `aqe_k=2` config landed at `id_switches=154` exact — identical to 14e B1 on the original 2-view tertiary features. Adding two scale views to DINOv2 produced **zero change in association decisions**. With `w_t=0.525` the fused score is dominated by the primary CLIP TransReID stream, and the tertiary stream's TTA noise is already below the threshold where it could affect cross-camera matches. **The tertiary DINOv2 stream is saturated.** S6 at `aqe_k=3` reproduces the −0.79pp regression seen in 14e B3/B4 and the 14f k=3 family, confirming the AQE k=2 unlock is a property of TTA-smoothed primary features, not of how many DINOv2 views are used.

**Implication**: TTA expansion family is **fully saturated** — both primary 4-view (14c v2) and tertiary 4-view (14g) are now exhausted. Further IDF1 gains require a **non-view-coverage axis**: tracklet aggregation quality, feature diversity, or learned association.

**Next experiment**: 14h — robust tracklet pooling. Single GPU Stage-2 rerun on the proven 14c v2 TTA recipe with `stage2.multi_query.k=24` enabled, then CPU sweep over 7–8 aggregation rules (mean / median / geometric-median / medoid / trimmed-mean@10/25 / top-K-nearest-to-mean / top-K-nearest-to-medoid) computed from the saved per-tracklet K=24 best-quality TTA frame embeddings. Spec: `docs/subagent-specs/post-14g-next.md`.

**Files**: kernel `yahiaakhalafallah/14g-dinov2-4view-tta-stage2` v1; results `outputs/14g_v1_summary/14g_summary.json`; source 10a run `run_kaggle_20260425_202123`.

### 2.15 2026-05-08 14h v3 — Robust Tracklet Pooling Sweep (NEUTRAL)

GPU P100 Stage-2 rerun on the 14c v2 TTA recipe with `stage2.multi_query.k=24` enabled, saving the top-24 highest-quality TTA-smoothed embeddings per tracklet to `multi_query_embeddings.npz`. A CPU-side post-processor (`scripts/repool_stage2.py`) computed 8 robust aggregation modes from those rows, then Stages 3–5 were re-run for each mode at the 14e B1 anchor (`aqe_k=2, fic_reg=0.5, thr=0.48, w_t=0.525`). Stage-2 walltime was 124.0 min on P100; Stages 3–5 sweep walltime was 4.76 min CPU.

**Drift check**: M0 (existing softmax-quality mean, no robust repooling) reproduced **0.77936 EXACT**, `id_switches=154 EXACT` vs 14e B1; gate passed bit-identically.

**Robust-pooling result table (9 configs at the 14e B1 anchor)**:

| Label | mode | MTMC IDF1 | id_switches | trackeval_idf1 | Δ vs 14e B1 (0.77936) | fallback_count |
|:-----:|:-----|:---------:|:-----------:|:--------------:|:---------------------:|:--------------:|
| **M0** | **existing_softmax_quality_mean (drift gate)** | **0.77936** | **154** | **0.79461** | **+0.00000** | **0** |
| M1 | mean | 0.77829 | 163 | 0.79444 | −0.00107 | 158 |
| M2 | median | 0.77107 | 162 | 0.79299 | −0.00829 | 158 |
| M3 | geo_median | 0.77514 | 163 | 0.79367 | −0.00422 | 158 |
| M4 | medoid | 0.77234 | **134** | 0.79151 | −0.00702 | 158 |
| M5 | trimmed_mean_10 | 0.77522 | 167 | 0.79468 | −0.00414 | 158 |
| M6 | trimmed_mean_25 | 0.77146 | 162 | 0.79367 | −0.00790 | 158 |
| M7 | top12_to_mean | 0.76933 | 149 | 0.79291 | −0.01003 | 158 |
| M8 | top12_to_medoid | 0.76881 | 149 | 0.79177 | −0.01055 | 158 |

**Verdict**: NEUTRAL per `docs/subagent-specs/post-14g-next.md` thresholds. No promotion. The 0.77936 14e B1 headline stands.

**Key signal**: every robust mode underperforms the existing softmax-quality mean. Medoid achieves the lowest `id_switches` (134, −20 vs M0) but lower IDF1 (0.77234, −0.70pp), demonstrating ID-switch count is not a reliable proxy for IDF1 on this floor. The existing softmax pool is already optimal because TTA pre-smoothing removed the per-frame outliers that robust statistics would clip.

**Implication**: this is the third consecutive feature-side NEUTRAL after 14g (tertiary view expansion) and 14e Stage-4 saturation; the plateau is confirmed across three independent axes (view coverage, tracklet aggregation, Stage-4 tuning). Remaining levers: track-quality pre-filter (cheap hedge), new third feature stream (high cost), GNN edge classifier (very high cost).

**Next experiment**: 14i — track-quality pre-filter, CPU only, ~30 min, 20-config sweep. Spec: `docs/subagent-specs/post-14h-next.md`.

**Files**: kernel `yahiaakhalafallah/14h-robust-tracklet-pooling` v3; results `outputs/14h_v3_summary/14h_summary.json`; source 10a run `run_kaggle_20260425_202123`.

### 2.16 2026-05-08 14i v2 — Track-Quality Pre-Filter Sweep (MARGINAL)

CPU-only sweep on the existing 14h v3 Stage 1/2 outputs. F0 is the no-filter identity drift gate; F1–F20 sweep `min_length ∈ {3,5,8,12}` × `min_avg_confidence ∈ {0.30,0.35,0.40,0.45,0.50}` at the 14e B1 anchor (`aqe_k=2, fic_reg=0.5, thr=0.48, w_t=0.525`). Total Stage 3–5 walltime was 11.21 min CPU.

**v1 failure and fix**: 14i v1 returned all-zero metrics despite 929/929 pass-through because the notebook accepted an unvalidated Kaggle dataset root for Stage 5 ground truth. v2 validates that all six expected `Sxx_cxxx/gt/gt.txt` files are directly under the selected GT root and copies recovery artifacts for each label to `/kaggle/working/outputs/14i_v2_recovery/<label>`.

**F0 drift check**: F0 reproduced **0.77935962** with **id_switches=154**, retaining 929/929 tracklets and writing 6 MOT prediction files with 26,523 rows. Gate passed.

| Label | `min_length` | `min_avg_confidence` | kept | MTMC IDF1 | id_switches | Verdict |
|:-----:|-------------:|---------------------:|-----:|----------:|------------:|:--------|
| **F0** | **0** | **0.00** | **929** | **0.77935962** | **154** | drift gate passed |
| F1 | 3 | 0.30 | 845 | 0.77953820 | 125 | marginal |
| **F2** | **3** | **0.35** | **818** | **0.77963534** | **120** | best, not promoted |
| F6 | 5 | 0.30 | 794 | 0.77910981 | 131 | neutral |
| F7 | 5 | 0.35 | 769 | 0.77880991 | 126 | neutral |
| F8 | 5 | 0.40 | 751 | 0.77848421 | 118 | slight regression |

**Verdict**: MARGINAL, not deployable. F2 improves F0 by only **+0.00028 IDF1** (+0.03pp), below the 0.781 WIN threshold and within the established run-to-run noise band. More aggressive filters reduce ID switches but lose IDF1; e.g. F9 (`L=5`, `τ=0.45`) reaches 97 ID switches but drops to 0.77604.

**Implication**: low-quality/short tracklets are not the concentrated source of residual error. The 0.77936 plateau is now confirmed across Stage-4 tuning, TTA view count, robust tracklet pooling, and track-quality filtering. Remaining viable levers: genuinely new feature stream or learned association (GNN edge classifier).

**Files**: kernel `yahiaakhalafallah/14i-track-quality-prefilter` v2; results `outputs/14i_v2_recovery/14i_summary.json`; recovery artifacts under `outputs/14i_v2_recovery/outputs/14i_v2_recovery/<label>/`.

---

### 2.17 2026-05-08 14j v1 — R50-IBN 4-Way Score-Fusion Sweep (MARGINAL)

CPU-only sweep that adds FastReID R50-IBN-a (CityFlowV2-trained, from the 14j R50-IBN feature-extraction GPU kernel) as a fourth score-fusion stream on top of the existing primary CLIP TransReID + tertiary DINOv2 fusion. All 16 configs run at the 14e B1 anchor (`aqe_k=2, fic_reg=0.5, w_t=0.525`). At each `w_quaternary`, `w_primary` and `w_tertiary` are rescaled by `(1 − w_q)` preserving the `0.475 : 0.525` ratio. W0 is the no-quaternary identity drift gate.

**W0 drift check**: reproduced **0.77935962** with **id_switches=154 EXACT**, 929/929 tracklets pass-through, 6 MOT prediction files with 26,523 rows, 594 global trajectories. Gate passed bit-identically.

**Sweep result table (16 configs)**:

| Label | `w_q` | `thr` | `w_p` | MTMC IDF1 | id_switches | Verdict |
|:-----:|:-----:|:-----:|:-----:|:---------:|:-----------:|:--------|
| **W0** | **0.00** | **0.48** | **0.475** | **0.77935962** | **154** | drift gate passed |
| W1 | 0.05 | 0.46 | 0.425 | 0.77712784 | 200 | regression |
| W2 | 0.05 | 0.48 | 0.425 | 0.77950361 | 154 | neutral |
| W3 | 0.05 | 0.50 | 0.425 | 0.77853170 | 154 | neutral |
| W4 | 0.10 | 0.46 | 0.375 | 0.77741785 | 206 | regression |
| W5 | 0.10 | 0.48 | 0.375 | 0.77727183 | 200 | regression |
| W6 | 0.10 | 0.50 | 0.375 | 0.77853170 | 154 | neutral |
| W7 | 0.15 | 0.46 | 0.325 | 0.77828168 | 206 | neutral |
| W8 | 0.15 | 0.48 | 0.325 | 0.77795774 | 206 | neutral |
| W9 | 0.15 | 0.50 | 0.325 | 0.77698593 | 206 | regression |
| W10 | 0.20 | 0.46 | 0.275 | 0.77855537 | 207 | neutral |
| W11 | 0.20 | 0.48 | 0.275 | 0.77832761 | 206 | neutral |
| W12 | 0.20 | 0.50 | 0.275 | 0.77735591 | 206 | regression |
| W13 | 0.30 | 0.46 | 0.175 | 0.77917492 | 207 | neutral |
| **W14** | **0.30** | **0.48** | **0.175** | **0.78032197** | **207** | **best, MARGINAL, not promoted** |
| W15 | 0.30 | 0.50 | 0.175 | 0.77854740 | 206 | neutral |

**Verdict**: MARGINAL per the 14j spec bands (WIN ≥0.7810, MARGINAL 0.7795–0.7810). W14 reached 0.78032, +0.00097 over W0 and +0.0010 over the previous deployed baseline 0.7703, but the 0.7810 WIN threshold was not crossed. Headline 0.77936 stands.

**Boundary effect**: W14 sits on the upper boundary of the swept `w_q` grid (max tested = 0.30). At `thr=0.46` the trend across `w_q` is monotonic with no turnover: 0.77713 → 0.77742 → 0.77828 → 0.77856 → 0.77917. At `thr=0.48` after a regime change at `w_q=0.10` (id_switches jumps 154 → 200), the trend climbs cleanly to 0.78032 at the boundary. The optimum may extend into `w_q ∈ [0.35, 0.50]`.

**Primary-suppression caveat**: at `w_q=0.30`, `w_primary=0.175` (already minority vs `w_tertiary=0.525`). At higher `w_q`, primary CLIP TransReID is essentially zero'd out, so any further lift may be expert-rebalancing rather than ensemble diversity. The 14k follow-up includes a K13 sanity probe (`w_p=w_t=0.30, w_q=0.40`) to discriminate.

**Next experiment**: 14k — extended `w_q ∈ {0.35, 0.40, 0.45, 0.50}` × `thr ∈ {0.46, 0.48, 0.50}` sweep + K13 primary-balance sanity. CPU-only, ~10–15 min. Spec: `docs/subagent-specs/post-14j-next.md`.

**Files**: kernel `yahiaakhalafallah/14j-4way-fusion-sweep` v1; results `outputs/14j_4way_sweep/14j_4way_summary.json`; R50-IBN features `yahiaakhalafallah/14j-r50-ibn-features`; source `yahiaakhalafallah/14h-robust-tracklet-pooling` v3.

---

### 2.18 2026-05-08 14k Extended R50-IBN 4-way Fusion Sweep

CPU-only follow-up to 14j that extends the R50-IBN quaternary stream into the upper `w_q ∈ {0.35, 0.40, 0.45, 0.50}` range and adds K13, a literal balanced sanity probe (`w_p=0.30, w_t=0.30, w_q=0.40`). K1–K12 preserve the 14j residual split convention: after assigning `w_q`, the remaining weight is split between primary and tertiary in the `0.475 : 0.525` ratio. All configs use the 14e B1 anchor (`aqe_k=2, fic_reg=0.5`) and vary only `similarity_threshold` plus fusion weights.

**K0 drift check**: reproduced **0.77936** with **id_switches=154 EXACT**. Gate passed.

**Sweep result table (14 configs)**:

| Label | `w_q` | `thr` | `w_p` | `w_t` | MTMC IDF1 | id_switches | Verdict |
|:-----:|:-----:|:-----:|:-----:|:-----:|:---------:|:-----------:|:--------|
| **K0** | **0.00** | **0.48** | **0.475** | **0.525** | **0.77936** | **154** | drift gate passed |
| K1 | 0.35 | 0.46 | 0.150 | 0.500 | 0.77936 | 213 | neutral |
| K2 | 0.35 | 0.48 | 0.150 | 0.500 | 0.77917 | 207 | neutral |
| K3 | 0.35 | 0.50 | 0.150 | 0.500 | 0.77753 | 206 | regression |
| K4 | 0.40 | 0.46 | 0.125 | 0.475 | 0.78041 | 213 | marginal plateau |
| K5 | 0.40 | 0.48 | 0.125 | 0.475 | 0.78041 | 213 | marginal plateau |
| K6 | 0.40 | 0.50 | 0.125 | 0.475 | 0.78017 | 212 | marginal plateau |
| **K7** | **0.45** | **0.46** | **0.100** | **0.450** | **0.78079** | **213** | **peak, MARGINAL, not promoted** |
| K8 | 0.45 | 0.48 | 0.100 | 0.450 | 0.78048 | 213 | marginal plateau |
| K9 | 0.45 | 0.50 | 0.100 | 0.450 | 0.78048 | 213 | marginal plateau |
| K10 | 0.50 | 0.46 | 0.075 | 0.425 | 0.77964 | 213 | turnover |
| K11 | 0.50 | 0.48 | 0.075 | 0.425 | 0.78048 | 213 | marginal plateau |
| K12 | 0.50 | 0.50 | 0.075 | 0.425 | 0.78048 | 213 | marginal plateau |
| **K13** | **0.40** | **0.48** | **0.300** | **0.300** | **0.78048** | **213** | **literal sanity passed** |

**K13 sanity**: passes. The literal balanced probe (`0.30/0.30/0.40`) reaches 0.78048, matching the plateau and confirming the lift is not merely primary CLIP suppression. It is a real ensemble effect, but too small to promote.

**Turnover analysis**: 14j's boundary optimum at `w_q=0.30` extended into a shallow plateau, not a WIN. The curve improves through `w_q=0.40–0.45`, peaks at K7 = 0.78079, then shows turnover at `w_q=0.50` (K10 = 0.77964). Five configs cluster at ~0.78048, so the upper-quaternary regime is saturated rather than under-sampled.

**Verdict**: MARGINAL, not promoted. Best K7 is **0.78079**, +0.0014 vs 14e B1 but below the pre-registered WIN bar **0.7810** and below the historical noise band (~0.24pp). Headline **0.77936** stands. This closes the CPU-only 4-way fusion axis and confirms the feature-quality ceiling across 5 independent axes: Stage-4 tuning, tertiary view expansion, tracklet aggregation, track-quality filter, and 4-way score fusion.

**Next experiment**: 14l — genuinely new feature stream. Prioritize OSNet-IBN-x1.0 trained on VeRi-776 + CityFlowV2 first, then EVA-02-L/14, then CLIP TransReID L/14. Spec: `docs/subagent-specs/post-14k-next.md`.

**Files**: results `outputs/14k_extended/14k_extended_summary.json`; kernel `yahiaakhalafallah/14k-r50-ibn-fusion-extended`; source feature stack from 14h v3 plus 14j R50-IBN quaternary features.

---

### 2.19 2026-05-08 14m OSNet-IBN Training (resolved FAIL)

GPU-side attempt to train an architecture-diverse OSNet-IBN-x1.0 CityFlowV2 ReID model as the fifth stream for the planned 14n 5-way fusion ensemble. This replaced the lost ali369 v80 OSNet checkpoint path, but the completed run failed the single-camera eligibility gate by a wide margin.

**Final resolved run**: `gumfreddy/14m-osnet-ibn-cityflowv2-train` v1 completed **120/120 epochs** after the v3 memory-defense patches: single T4 GPU, batch 64, eval batch 8, `P_IDS=16`, `K=4`, no DataParallel, and per-epoch `gc.collect(); torch.cuda.empty_cache()`. Wall time was about **6.5h on T4**. Metrics below are from `data/outputs/14m_final_metrics.json`.

| Checkpoint | Epoch | mAP | R1 | R5 | R10 | Gate |
|:--|--:|--:|--:|--:|--:|:--|
| Best mAP | 60 | **24.27%** | 43.59% | **53.91%** | **60.65%** | FAIL |
| Best joint | 90 | 23.90% | **43.97%** | 53.89% | 60.32% | FAIL |
| Final eval | 120 | 23.80% | 43.89% | 53.72% | 60.28% | FAIL |

**Eligibility gate**: required **mAP >=75% AND R1 >=90%** before Stage-2 extraction or 14n fusion. Final eval was **mAP=23.80%, R1=43.89%, R5=53.72%, R10=60.28%**. Best checkpoint by mAP was epoch 60 with **mAP=24.27%, R1=43.59%, R5=53.91%, R10=60.65%**.

**Training trend**: the model peaked at epoch 60 and slowly degraded through epoch 120, so the failure is not an unfinished-training issue. It is over-trained and still far below the eligibility floor.

**Verdict**: **DEAD END** for 14n. OSNet-IBN-x1.0 in-domain CityFlowV2 from-scratch is too weak as a fusion stream: 23.80% final mAP is below the 52.77% R101-IBN floor and far below the 80%+ TransReID CLIP primary. Adding this stream would hurt rather than help.

**Root-cause hypotheses**: (a) 666-class CityFlowV2 is too small for OSNet's design when trained from scratch; (b) v3 single-GPU + batch-64 + `P=16/K=4` changed dynamics from the BoT recipe originally tuned for ResNet-family models and larger batches; (c) the BoT LR schedule may not suit OSNet. Do not retry OSNet-IBN-x1.0 from scratch on CityFlowV2 unless the new attempt explicitly addresses these three issues.

**Operational note**: the `KAGGLE_API_TOKEN` env-var auth pattern with `~/.kaggle/<account>_access_token` is proven for multi-account pushes; gumfreddy ran 14m successfully with this pattern.

---

### 2.20 2026-05-10 14p3 TransReID ViT-L/14 CLIP VeRi-776 (FAIL)

VeRi-first paper-direction experiment: test whether a larger TransReID ViT-L/14 CLIP backbone can beat the 09v v17 ViT-B/16 CLIP VeRi-776 baseline. The completed 14p3 run failed the gate and closes the capacity-only axis.

**Spec**: `docs/subagent-specs/14p-veri-sota-train.md`

**Result source**: `tmp_14p3_outputs/eval_results.json`

**Architecture and recipe**: TransReID ViT-L/14 CLIP @ 224², **304M params**, 100 epochs, LLRD factor **0.65**, `BACKBONE_LR=1.5e-4`, AMP fp16, `P=8/K=4`, batch 32, JPM 4 groups, BNNeck, SIE, AdamW + cosine + 10ep warmup, RandomErasing.

| Variant | mAP | R1 | Verdict |
|:--|--:|--:|:--|
| single_flip_cls_base | 80.90% | 96.90% | FAIL vs 09v base |
| single_flip_cls_aqe2_rerank | 87.88% | 97.50% | FAIL vs 09v post-rerank |
| concat_patch_flip_aqe2_rerank | **87.95%** | **97.32%** | best 14p3, FAIL |

**Verdict**: **FAIL**. 14p3 is about **2pp mAP / 1pp R1 below** the 09v v17 ViT-B/16 CLIP baseline at base (**89.97% mAP / 98.33% R1**) and about **3.6pp below** the 09v post-rerank mAP ceiling (~**91.54%**). Root cause: ViT-L/14 overfits VeRi-776's **576 train IDs / 37k images**; bigger model capacity is the wrong axis for this dataset size.

---

### 2.21 2026-05-10 14q ViT-B/16 CLIP 256² VeRi-776 Extended Training (FAIL)

Follow-up pivot from capacity to resolution/recipe: rerun the stronger 09v-style ViT-B/16 CLIP family at **256²** with extended training, then compare against the 09v v17 224² ceiling. This run failed the gate and closes the scale-only axis after 14p3.

**Spec**: `docs/subagent-specs/14q-veri-next.md`

**Kernel**: `mrkdagods/14q-veri-vit-b-16-clip-256-transreid-train`, COMPLETE on MRKDaGods. URL: https://www.kaggle.com/code/mrkdagods/14q-veri-vit-b-16-clip-256-transreid-train.

**Result source**: `tmp_14q_outputs/eval_results.json`

**Recipe**: ViT-B/16 CLIP @ 256², 160ep, `P=16/K=4`, batch 64, `BACKBONE_LR=3.5e-4`, head LR `3.5e-3`, LLRD **0.65**, 10ep warmup + cosine, AMP fp16, RandomErasing + geometric RandAugment, JPM 4-group, BNNeck, SIE, AdamW.

| Variant | mAP | R1 | Verdict |
|:--|--:|--:|:--|
| single_flip_cls_base | 79.68% | 96.84% | FAIL vs 09v base |
| single_flip_cls_aqe2_rerank | 88.06% | 96.66% | FAIL vs 09v post-rerank |
| concat_patch_flip_aqe2_rerank | 88.57% | 96.72% | FAIL vs 09v post-rerank |
| concat_patch_flip_aqe3_rerank | **89.15%** | **97.20%** | best 14q, FAIL |

**Verdict**: **FAIL**. 14q remains below the 09v v17 224² baseline (**89.97% mAP / 98.33% R1 base; ~91.54% post-rerank**). Triplet loss saturated to **0.005** by epoch 160, indicating full saturation under the current losses. Resolution bump **224->256** plus **60% more epochs** did not beat 09v v17, so the CLIP TransReID scale-only axis is exhausted under standard CE+triplet supervision.

| Band | Post-rerank mAP | R1 | Action |
|:--|:--|:--|:--|
| WIN | >=91.54% | >=98.33% | Promote over 09v v17 |
| MARGINAL | >=90.5% | >=98.0% | Document as near-ceiling but not a clear win |
| FAIL | otherwise | otherwise | Close this resolution/recipe pivot |

---

### 2.22 2026-05-10 14r Primary CLIP-ReID 2-Stage VeRi-776 (ABORTED, walltime guard)

Architecturally orthogonal follow-up after two scale-axis FAILs (14p3 ViT-L/14 and 14q ViT-B/16 @ 256²). This is the first run to use the CLIP text tower with learned ID-specific prompts rather than only the image tower as a TransReID backbone.

**Spec**: `docs/subagent-specs/14r-primary.md`

**Kernel**: `mrkdagods/14r-clip-reid-veri-776-train`, ABORTED by its own walltime guard on MRKDaGods. URL: https://www.kaggle.com/code/mrkdagods/14r-clip-reid-veri-776-train.

**Recipe**: Stage 1 freezes OpenAI CLIP ViT-B/16 image+text towers and learns 4 shared 512-d context tokens plus per-class 512-d ID token via image-text contrastive (`i2tce`), Adam lr 3.5e-4, wd 1e-4, 120ep, `P=8/K=4`, batch 32. Stage 2 unfreezes timm `vit_base_patch16_clip_224.openai` with CE + triplet + i2tce + JPM-CE, AdamW backbone 3.5e-4 / head 3.5e-3, LLRD 0.65, 120ep, 10ep warmup + cosine, AMP fp16, SIE, JPM 4-group, BNNeck.

**Outcome**: Stage 1 completed 120 epochs in **5.66h**, with loss **2.22 -> 1.53**. Saved prompts are present and finite: `text_features [576,512]`, `ctx [4,512]`, and `id_tokens [576,512]`. Stage 2 completed epoch 1 (loss **9.85**) before the walltime guard raised `RuntimeError`: projected Stage 2 runtime was **12.4h** (~6.2 min/epoch x 120 epochs), on top of the already-spent 5.66h Stage 1, projecting about **18h total > 14h cutoff**.

**Diagnosis**: **ABORTED / INCOMPLETE**, not a methodology failure. The published CLIP-ReID approach has not been evaluated yet. Root cause of slowness was batch 32 (`P=8/K=4`) versus 14q's batch 64 (`P=16/K=4`), causing roughly 2x more steps, plus `i2tce` auxiliary compute per step. Bug found: the kernel raised the abort `RuntimeError` before writing `train_log.json`, so the structured log was lost and only stdout preserved the result. Quota cost: about **9h MRKDaGods**.

**Rationale**: published CLIP-ReID evidence projects to roughly **91-93% mAP post-rerank**, and this tests a genuinely new supervision path rather than more model scale. The recovery run below continues this path from the saved Stage 1 prompts.

**Verdict bands**: WIN >=91.54% mAP AND >=98.33% R1; MARGINAL >=90.5% mAP AND >=98.0% R1; FAIL otherwise.

---

### 2.23 2026-05-10 14r Probe DINOv2 ViT-B/14 VeRi-776 Standalone (FAIL)

Backbone-swap probe that replicates the 09v v17 recipe shape with a DINOv2 ViT-B/14 backbone. This measures SSL-pretrained DINOv2 standalone on VeRi-776, which has not been done before in this repo.

**Spec**: `docs/subagent-specs/14r-probe.md`

**Kernel**: `gumfreddy/14r-probe-dinov2-veri-776-train`, COMPLETE on gumfreddy. URL: https://www.kaggle.com/code/gumfreddy/14r-probe-dinov2-veri-776-train.

**Result source**: `tmp_14r_probe_outputs/eval_results.json`

**Recipe**: timm `vit_base_patch14_dinov2.lvd142m`, 224² (16x16 patch grid), `P=8/K=4`, batch 32, `BACKBONE_LR=3.5e-4`, head LR `3.5e-3`, LLRD 0.65, 100ep, 10ep warmup + cosine, AMP fp16, JPM 4-group, BNNeck, SIE, CE label smoothing 0.1 + triplet 0.3 + JPM CE.

| Variant | mAP | R1 | Note |
|:--|:--:|:--:|:--|
| single_flip_cls_base | 81.43% | 97.44% | base |
| single_flip_cls_aqe2_rerank | 88.92% | 97.97% | rerank |
| concat_patch_flip_aqe2_rerank | 89.24% | **98.15%** | near-best |
| concat_patch_flip_aqe3_rerank | **89.27%** | **98.15%** | best 14r-probe, FAIL |

**Verdict**: **FAIL**. Best **89.27% mAP / 98.15% R1** is below WIN (**91.54% / 98.33%**) by about **2.3pp mAP / 0.2pp R1** and below the MARGINAL mAP threshold (**90.5%**) by about **1.2pp**.

**Rationale and significance**: DINOv2 offers an SSL-pretrained representation rather than CLIP image-text pretraining. This probe shows DINOv2 SSL pretraining alone underperforms CLIP pretraining for VeRi-776 (**89.27% vs CLIP's 91.54% post-rerank**), confirming **CLIP pretraining is necessary, not just any SSL**. The R1 result (**98.15%**) is close to 09v v17's **98.33%**, so DINOv2 features may still be diverse enough from CLIP for ensemble use.

**Verdict bands**: WIN >=91.54% mAP AND >=98.33% R1; MARGINAL >=90.5% mAP AND >=98.0% R1; FAIL otherwise.

---

### 2.24 2026-05-11 14r Recovery CLIP-ReID Stage-2 Resume (FAIL)

Stage-2-only recovery of 14r primary after the original kernel aborted from walltime projection. This run resumes from the saved Stage 1 prompts rather than repeating the 5.66h prompt-learning stage.

**Spec**: `docs/subagent-specs/14r-recovery.md` (LOCKED PLAN — 2026-05-10 section)

**Kernel**: `gumfreddy/14r-recovery-clip-reid-stage-2`, v4 COMPLETE on gumfreddy. URL: https://www.kaggle.com/code/gumfreddy/14r-recovery-clip-reid-stage-2.

**Result source**: `tmp_14r_recovery_outputs/14r_recovery_summary.json`. All `.pth` artifacts were deleted per disk-hygiene policy because the run failed and the checkpoints have no recovery value.

**Input**: saved Stage 1 prompts uploaded as Kaggle dataset `gumfreddy/14r-clip-reid-stage1-prompts`.

**Recipe deltas vs original 14r Stage 2**: batch `P=8/K=4 -> P=16/K=4` (size **32 -> 64**), epochs **120 -> 60**, LR sqrt-scaled from backbone/head **3.5e-4/3.5e-3 -> 4.95e-4/4.95e-3**, warmup **10ep -> 5ep**, periodic eval at **[20, 40, 50, 55, 60]**, Stage-2-only walltime guard **4.5h**, and `train_log.json` is written before any guard raise.

| Variant | mAP | R1 | R5 | R10 | Verdict |
|:--|--:|--:|--:|--:|:--|
| concat_patch_flip_aqe3_rerank_k1_80_k2_15_lambda_0_2 | **80.55%** | **93.68%** | 95.53% | 96.54% | FAIL |

**Verdict**: **FAIL**. Best concat AQE row reached only **80.55% mAP / 93.68% R1**, versus the 09v v17 ViT-B/16 CLIP baseline at **89.97% mAP / 98.33% R1**. This is a **-9.4pp mAP regression**, far below both WIN and MARGINAL gates.

**Conclusion**: Stage-2-only CLIP-ReID recovery from saved Stage 1 prompts did **not** recover the baseline. The likely failure mode is that prompts-only initialization is insufficient; Stage 2 needs the full Stage1->Stage2 coupled trajectory, including backbone co-training dynamics, not just final prompt vectors. A single-kernel T4 walltime budget of about **9h** is insufficient for the full CLIP-ReID Stage1+Stage2 chain on VeRi-776 at ViT-B/16 under this recipe.

**Verdict bands**: WIN >=91.54% mAP AND >=98.33% R1; MARGINAL >=90.5% mAP AND >=98.0% R1; FAIL otherwise.

---

### 2.25 2026-05-11 14t CLIP-SENet × TransReID Score-Level Fusion (VeRi-776, WIN)

VeRi-776 fusion follow-up after the single-model chase failed. This run tests whether the two strongest documented VeRi-776 experts in the repo, CLIP-SENet v6 and TransReID 09v v17, provide complementary rankings under simple fusion.

**Parents**: CLIP-SENet v6 (canonical 320², P=8/K=8; post-rerank **mAP=0.9154 / R1=0.9732**) + TransReID 09v v17 (base **mAP=0.8997 / R1=0.9833**).

**Dataset**: VeRi-776 query/gallery single-cam evaluation.

**Strategy**: swept both score-level fusion and feature concat. Score rows combine CLIP-SENet and TransReID distance/ranking scores; concat rows blend normalized feature spaces with `alpha_trans` / `alpha_clip`.

**Kernel**: `yahiaakhalafallah/14t-veri-fusion-clip-senet-x-transreid`, COMPLETE on yahiaakhalafallah. URL: https://www.kaggle.com/code/yahiaakhalafallah/14t-veri-fusion-clip-senet-x-transreid.

**Runtime**: ≈ **49.5 min** on T4.

**Result source**: `tmp_14t_outputs/14t_summary.json`.

| Variant | Config | mAP | R1 | Verdict |
|:--|:--|--:|--:|:--|
| **Best score-fusion** | `w_clipsenet=0.7, w_transreid=0.3`, `transreid_768`, AQE k=3 + rerank (k1=80, k2=15, λ=0.2) | **0.93304** | **0.98451** | **de-facto WIN** |
| Best concat | `alpha_trans=0.3, alpha_clip=0.7`, `transreid_768`, AQE k=3 + rerank (k1=80, k2=15, λ=0.2) | 0.93188 | 0.98271 | spec MARGINAL |

**Delta vs parents**: best score-fusion is **+3.33pp mAP / +0.12pp R1** vs 09v v17 base and **+1.76pp mAP / +1.13pp R1** vs CLIP-SENet v6 post-rerank alone.

**Plateau**: top score-fusion rows form a flat band with `w_clipsenet ∈ [0.6, 0.8]`. The best row uses the `transreid_768` global-token stream; transreid_768 clearly beat transreid_1536 for the final best configuration.

**Verdict**: strict spec = **MARGINAL** because the best concat row missed the R1≥0.9833 bar by **0.06pp**. De-facto = **WIN** because the best score-fusion row clears both mAP and R1 bars. This is the first experiment to reach the historical-claim **0.9845 R1** number as a true Rank-1 metric on this checkpoint family; the same value had previously only appeared as R5-via-AQE.

**Recommendation**: **Option B — Accept and freeze.** 14t is a paper-quality VeRi-776 fusion result, but it should **not** be auto-ported to CityFlow: 13d/13f/13h already showed CLIP-SENet × CityFlow fusion is negative because the VeRi-776 → CityFlowV2 domain gap dominates.

---

### 2.26 2026-05-12 14u CityFlow VeRi-Fusion Port (FAIL)

Final closure of the CityFlow VeRi-fusion family. Tests whether the 14t WIN mechanism — CLIP-SENet × TransReID-09v score fusion plus AQE k=3 + k-reciprocal rerank applied to the *fused* similarity matrix — ports to CityFlowV2 as a 4th score-fusion stream on top of the 14e B1 anchor (CLIP TransReID primary + DINOv2 tertiary, `aqe_k=2, w_t=0.525, thr=0.48, fic_reg=0.5`).

**Grid**: 19 configs, CPU-only. U0 = drift gate at `w_14t=0.00`. U1–U18 = sweep over `w_14t ∈ {0.05, 0.10, 0.15, 0.20, 0.25, 0.30}` × `thr ∈ {0.46, 0.48, 0.50}`.

**Result table (key rows)**:

| Label | `w_14t` | `thr` | MTMC IDF1 | id_switches | Verdict |
|:-----:|:-------:|:-----:|:---------:|:-----------:|:--------|
| **U0 (drift)** | **0.00** | **0.48** | **0.77936** | **154** | drift gate passed |
| U2 | 0.05 | 0.48 | 0.77950 | 154 | neutral |
| U3 | 0.05 | 0.50 | 0.77829 | 154 | regression |
| **U5 / U6 / U9 (best tie)** | **0.10 / 0.10 / 0.15** | **0.48 / 0.50 / 0.50** | **0.77995** | **160** | below MARGINAL bar |
| U7 | 0.15 | 0.46 | 0.77809 | 207 | regression |
| U8 | 0.15 | 0.48 | 0.77809 | 207 | regression |

**Verdict**: FAIL. Best 0.77995 = +0.00059 IDF1 vs U0, below the 14u spec MARGINAL bar of 0.7800. id_switches went UP (154→160) at the best point — fusion adds conflation, not signal. Optimum sits at the lower boundary of the sweep, exactly mirroring the 13d `w_cs=0.10` row in the prior CLIP-SENet × CityFlow fusion FAIL.

**Mechanistic interpretation**: 14u was the last untested fusion mechanism — applying AQE + k-reciprocal rerank to the *fused* similarity (the mechanism that drove the 14t VeRi WIN, +3.33pp mAP). It still cannot transfer the VeRi-776 lift to CityFlow cross-camera matching. Closes the 5th and final CityFlow VeRi-fusion branch (13d / 13f / 13g / 13h / 14u all FAIL or MARGINAL). The 0.77936 plateau is now confirmed across SIX independent axes: Stage-4 tuning, tertiary view expansion, tracklet aggregation, track-quality filter, 4-way score fusion, and VeRi-fusion-port.

**Next experiment**: none cheap or medium-cost remains. Remaining viable paths to >0.7900 require (a) AIC22-style 5-model GPU ensemble, (b) GNN edge classifier, or (c) zone-based ST + per-camera distance bias.

**Kernel**: `yahiaakhalafallah/14u-cityflow-veri-fusion-port`, COMPLETE. URL: https://www.kaggle.com/code/yahiaakhalafallah/14u-cityflow-veri-fusion-port.

**Files**: results `tmp_14u_outputs/14u_summary.json`.

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
| 09 v2 | Augoverhaul + CircleLoss | **81.59%** | - | HIGHER ReID, MTMC REGRESSION |
| 09 v3 | Augoverhaul + EMA training run (base checkpoint used downstream) | **81.53%** | **92.41%** | SAME MTMC CEILING AS 09 v2 |
| 09l v1 | ViT-B/16 LAION-2B CLIP, CircleLoss ablation recipe, 120 epochs | **20.36%** | **53.03%** | FAILED (`inf` loss throughout; gamma=128 CircleLoss overflowed fp16 autocast) |
| 09l v2 | ViT-B/16 LAION-2B CLIP, TripletLoss + EMA, 160 epochs | **61.51%** | **81.41%** | PROMISING BUT UNCONVERGED |
| 09l v3 | ViT-B/16 LAION-2B CLIP, resumed from v2 EMA checkpoint, 300 total epochs | **78.61%** | **90.43%** | READY FOR ENSEMBLE DEPLOYMENT |
| 09k v1 | ViT-Small/16 TransReID (ImageNet-21k pretrain), 120 epochs | **48.66%** | **62.01%** | BELOW ENSEMBLE THRESHOLD |
| 09o v1 | EVA02 ViT-B/16 CLIP, 120 epochs, AdamW, backbone_lr=1e-5, head_lr=5e-4, CE+Triplet+Center, cosine schedule | **48.17%** | **65.90%** | DEAD END FOR ENSEMBLE USE |

The full **09l** sequence now closes the loop on **LAION-2B CLIP** as a secondary-model candidate. After the broken **09l v1** CircleLoss recipe failed numerically, the stable **09l v2** rerun restored the backbone to **61.51% mAP**, **81.41% R1**, **67.20% mAP_rerank**, and **82.95% R1_rerank** in **160 epochs** with **TripletLoss + EMA(decay=0.9999)**.

Resuming that **v2 EMA checkpoint** for **09l v3** and extending training to **300 total epochs** lifted the model to **78.61% mAP**, **90.43% R1**, **81.09% mAP_rerank**, and **90.98% R1_rerank**. The resumed phase took **~3.3 hours on a Kaggle T4**, and the curve kept improving smoothly across **epoch 180/200/220/240/260/280/300 = 65.93/68.84/71.49/73.68/75.68/77.26/78.61 mAP**.

That result changes the interpretation of this backbone path: **LAION-2B CLIP is now a strong secondary model**, only **1.53pp** behind the deployed **OpenAI CLIP 09b v2** baseline at **80.14% mAP**, and comfortably above the **~65% mAP** bar required for score-level fusion. This path is now **ready for ensemble deployment** rather than needing more rescue training.

### 4.1 VeRi-776 Evaluation Reproduction (09v)

| Version | Account | Sweep | Best Result | Verdict | Key Insight |
|:-------:|:-------:|-------|:-----------:|:-------:|-------------|
| 09v v10 | yahiaakhalafallah | Full VeRi-776 eval reproduction on `vehicle_transreid_vit_base_veri776.pth`: rerank grid across 10 configs, AQE `k=0..7`, cross grid, and TTA comparisons | **mAP=84.77%**, **R1=98.03%**, **R5=98.75%**, **R10=99.28%** with **single_flip + rerank `k1=25,k2=8,λ=0.3`** | REPRODUCED | SIE camera embeddings lifted the baseline from **96.25% -> 97.50% R1** (+1.25pp); rerank `λ=0.3` with `k1` in the **20-30** range was the sweet spot; AQE did not beat the best pure-rerank config and 320/384px or checkpoint-B ensembling regressed |
| 09v v12 | yahiaakhalafallah | VeRi-776 frontier refresh on the same checkpoint: new `λ={0.15,0.2,0.25}` rerank sweep + AQE(k<=5) cross-grid around the best rerank configs | **Pareto frontier**: Best R1 **98.09 / 86.26**, Best mAP **89.84 / 97.79**, Joint **97.85 / 89.27**, Historical target match **97.91 / 87.15** (`R1 / mAP`) | SUPERSEDED BY v14 | **`λ=0.2` became the first strong joint optimum** on this checkpoint; later superseded by the 224x224 v14 rerun |
| 09v v14 | yahiaakhalafallah | Canonical VeRi-776 rerun at **224x224** to match kernel 08 training resolution, reusing the expanded `λ={0.15,0.2,0.25}` rerank sweep + AQE(k<=5) cross-grid | **Pareto frontier**: Best R1 **98.21 / 85.24**, Best mAP **89.91 / 97.62**, Joint **97.97 / 89.49**, Historical target match **97.85 / 87.32** (`R1 / mAP`) | SUPERSEDED BY v17 | **224x224 eval (matches kernel 08 training); +0.12pp R1 over v12 256x256** |
| 09v v15 | yahiaakhalafallah | Fine rerank grid around the v14 leader plus ten-crop TTA follow-up | **Best R1 confirmed at 98.33 / 85.14** with **single_flip rerank `k1=24,k2=8,λ=0.2`**; ten-crop loses about **-2.4pp R1** | PARTIAL WIN | **Finer `k1=24` rerank row adds +0.12pp R1; ten-crop is catastrophic and closed out** |
| 09v v16 | yahiaakhalafallah | Multi-scale eval attempt with **224+256** feature averaging | **ERROR** (`timm strict_img_size=True`) | BLOCKED | **Multi-scale would require model surgery; no result recorded** |
| 09v v17 | yahiaakhalafallah | Added **concat_patch_flip** GeM features on top of the v15 rerank grid and AQE cross-sweep | **Pareto frontier**: Best R1 **98.33 / 85.14**, Best mAP **89.97 / 97.80**, Joint **98.15 / 89.71**, Historical target match **98.33 / 85.14** (`R1 / mAP`) | CANONICAL | **single_flip remains the R1 leader; concat-patch GeM raises mAP by +0.06pp to 89.97 but does not lift the 98.33% R1 ceiling** |

Kernel **`yahiaakhalafallah/09v-veri-776-eval-transreid-rerank` v10** reproduced the current best known VeRi-776 result for the in-domain TransReID ViT-B/16 CLIP checkpoint at the trained **256x256** input size. The top four candidates all finished within **0.06pp R1**: pure rerank **`k1=25,k2=8,λ=0.3`** at **98.03%**, **AQE(k=4)+rerank `k1=30,k2=10,λ=0.3`** at **97.97%**, pure rerank **`k1=30,k2=10,λ=0.3`** at **97.97%**, and **AQE(k=2)+rerank `k1=25,k2=8,λ=0.3`** at **97.97%**. This closes the earlier **96.25% / 97.02% / 97.68%** claims with a verified full-sweep result recorded in `outputs/09v_veri_v4/veri776_eval_results_v4.json`.

Kernel **`yahiaakhalafallah/09v-veri-776-eval-transreid-rerank` v12** was the first authoritative VeRi-776 frontier for this checkpoint at the then-used **256x256** evaluation size. It added the missing **`λ=0.2`** operating point plus the surrounding **`λ=0.15/0.25`** checks and a focused **AQE(k<=5)** cross-grid. The resulting Pareto frontier was: **Best R1 = 98.09% / 86.26%** with pure rerank **`k1=30,k2=10,λ=0.15`**, **Best mAP = 89.84% / 97.79%** with **AQE(k=4)+rerank `k1=80,k2=15,λ=0.2`**, **Joint optimum = 97.85% / 89.27%** with pure rerank **`k1=80,k2=15,λ=0.2`**, and **Historical target match = 97.91% / 87.15%** with **AQE(k=5)+rerank `k1=30,k2=10,λ=0.2`**. This superseded the earlier **98.03 / 84.77** v10 frontier and established **`outputs/09v_veri_v5/veri776_eval_results_v5.json`** as the canonical **v12** artifact before the later v14 224x224 rerun replaced it.

Kernel **`yahiaakhalafallah/09v-veri-776-eval-transreid-rerank` v14** now replaces v12 as the authoritative VeRi-776 frontier for this checkpoint. The only intentional change was the evaluation resolution: **224x224** instead of **256x256**, matching the kernel 08 training metadata in `_scratch_old08/k08out/exported_models/vehicle_reid_sota_metadata.json`. That correction lifts the frontier to **Best R1 = 98.21% / 85.24%** with pure rerank **`k1=25,k2=8,λ=0.2`**, **Best mAP = 89.91% / 97.62%** with **AQE(k=5)+rerank `k1=80,k2=15,λ=0.2`**, **Joint optimum = 97.97% / 89.49%** with **AQE(k=2)+rerank `k1=80,k2=15,λ=0.2`**, and **Historical target match = 97.85% / 87.32%** with **AQE(k=4)+rerank `k1=30,k2=10,λ=0.15`**. This supersedes the prior v12 **256x256** canonical eval by **+0.12pp R1** and establishes **`outputs/09v_veri_v7/veri776_eval_results_v7.json`** as the canonical artifact.

Kernel **`yahiaakhalafallah/09v-veri-776-eval-transreid-rerank` v15** kept the same **224x224** setup and tested the obvious remaining eval-time headroom: a finer rerank sweep around the v14 leader plus ten-crop TTA. The rerank refinement mattered: **`k1=24,k2=8,λ=0.2`** lifted the absolute best row to **R1 = 98.33% / mAP = 85.14%** with **R5 = 99.05%** and **R10 = 99.34%**, a clean **+0.12pp R1** over v14. Ten-crop, by contrast, was decisively harmful at roughly **-2.4pp R1**, so it is now a closed dead end for this checkpoint.

Kernel **`yahiaakhalafallah/09v-veri-776-eval-transreid-rerank` v16** attempted multi-scale averaging over **224+256** views, but the run hit timm's **`strict_img_size=True`** guard before producing usable metrics. That blocks multi-scale eval on the current checkpoint unless the model is surgically rebuilt with compatible positional embedding handling.

Kernel **`yahiaakhalafallah/09v-veri-776-eval-transreid-rerank` v17** is now the authoritative VeRi-776 frontier. It preserves the v15 single_flip winner for headline **R1 = 98.33% / mAP = 85.14%** at **`k1=24,k2=8,λ=0.2`**, but adds a second feature bundle, **concat_patch_flip**, that raises the best-mAP row to **89.97% / 97.80%** with **AQE(k=3)+rerank `k1=80,k2=15,λ=0.2`** and yields a new joint optimum of **98.15% / 89.71%** at **AQE(k=2)+rerank** with the same rerank config. This establishes **`outputs/09v_veri_v9/veri776_eval_results_v9.json`** as the canonical artifact and closes the eval-time ceiling analysis: single_flip is still the R1 leader, concat-patch GeM helps mAP only, and pure evaluation tricks no longer move the checkpoint beyond **98.33% R1**.

Note: the remembered historical claim **`R1=0.984505`** is still reproduced in the canonical v17 artifact as **R5=98.4505%** for **single_flip AQE(k=3)+rerank `k1=30,k2=10,λ=0.2`**. The true metrics for that row are **R1=97.6758%**, **R5=98.4505%**, **R10=99.2253%**, and **mAP=86.9062%**, so the original number was **R5**, not **R1**.

The new **09k v1** ViT-Small/16 and **09o v1** EVA02 results are important because they rule out a simple "switch to another ViT backbone" explanation for secondary-model recovery. Despite using transformer backbones, the non-OpenAI-CLIP alternatives still topped out at only **48.66% mAP / 62.01% R1** for **ViT-Small/16 IN-21k** and **48.17% mAP / 65.90% R1** for **EVA02 ViT-B/16 CLIP**, both well below the quality needed for a useful ensemble partner. Combined with the later **09j v2** ResNeXt101-IBN-a ArcFace collapse to **36.88% mAP**, the evidence now points to **transfer quality and initialization compatibility**, not simply architecture choice: on CityFlowV2's **128 train IDs**, most alternative backbones are not producing a viable secondary model and the CNN variants remain capped well below the ensemble bar.

### 4.6 EVA02 ViT-B/16 CLIP (09o)

| Version | Account | Recipe | Result | Verdict | Key Insight |
|:-------:|:-------:|--------|:------:|:-------:|-------------|
| 09o v1 | gumfreddy | EVA02 ViT-B/16 CLIP, **120 epochs**, **AdamW**, **backbone_lr=1e-5**, **head_lr=5e-4**, **CE + Triplet + Center**, cosine schedule | **mAP=48.17%**, **R1=65.90%**, **R5=77.17%**, **R10=82.83%** | REJECTED | Considerably weaker than both the **80.14%** primary ViT-B/16 CLIP baseline and the **63.64%** fine-tuned R50-IBN secondary; EVA02 does not transfer well for vehicle ReID with the current recipe |

The **09o v1** result closes out the current **EVA02** path for CityFlowV2 vehicle ReID. Even with a conservative fine-tuning recipe and a full **120-epoch** run, it finished at only **48.17% mAP**, which is not merely below the practical ensemble threshold but also below the fine-tuned **R50-IBN** secondary. That makes **EVA02 ViT-B/16 CLIP** a dead end under the current training setup and removes it from the list of plausible near-term fusion candidates.

### 4.7 DINOv2 ViT-L/14 (09s)

| Version | Account | Recipe | Result | Verdict | Key Insight |
|:-------:|:-------:|--------|:------:|:-------:|-------------|
| 09s v1 | yahiaakhalafallah | DINOv2 ViT-L/14 (`vit_large_patch14_dinov2.lvd142m`), IMG_SIZE=252, STRIDE=14, 120 epochs, BATCH=32, stable ID+batch-hard triplet+delayed center loss | **mAP=86.79%**, **R1=96.15%**, best epoch 115/120 | BREAKTHROUGH ReID; MTMC REGRESSION | +6.65pp mAP / +3.88pp R1 over ViT-B/16 CLIP; downstream MTMC IDF1=0.744 (best, with AFLink) < ViT-B/16 CLIP 0.775 (-3.1pp). Single-cam IDF1=0.794 strong; cross-camera invariance weaker than TransReID recipe. |


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
| v17 | ali369 | mAP=29.6% | CircleLoss + triplet conflict |
| v18 | ali369 | mAP=**52.77%** | Best direct ImageNet→CityFlowV2 baseline |
| gumfreddy v3 | gumfreddy | mAP=50.61% | Extended fine-tuning from v18 regressed |
| 09e v2 | ali369 | mAP=62.52% on VeRi-776 | Pretraining succeeded, but transfer path later failed |
| 09f v3 | ali369 | mAP=42.7% | VeRi-776→CityFlowV2 fine-tune worse than direct baseline |
| 09i v1 | gumfreddy | mAP=50.80% at epoch 100/160, R1=73.46%, mAP_rerank=54.65% | ArcFace + Triplet + Center warm-start from 09d overfit after epoch 100 and stayed below the 52.77% ceiling |


The ResNet101-IBN-a path is now effectively closed. Across the original direct baseline, extended fine-tuning, VeRi-776 transfer, CircleLoss, SGD, and the new ArcFace follow-up, every serious variant ended at or below the **52.77% mAP** direct-training ceiling. The latest **09i v1** run confirms that even an ArcFace-based recipe does not rescue this backbone when warm-started from the existing CE-optimized checkpoint: it peaked at **50.80% mAP** by **epoch 100/160**, then overfit. This leaves the ensemble plan blocked by representation quality rather than missing optimizer or schedule tweaks.

### 4.4 ResNet101-IBN-a ArcFace Follow-Up (09i)

| Version | Account | Recipe | Result | Verdict | Key Insight |
|:-------:|:-------:|--------|:------:|:-------:|-------------|
| 09i v1 | gumfreddy | ArcFace (`s=30`, `m=0.35`) + Triplet (`m=0.3`) + Center loss, warm-start from 09d | Best **mAP=50.80%**, **R1=73.46%**, **mAP_rerank=54.65%** at **epoch 100/160** | REJECTED | Performance declined after epoch 100; CE-shaped warm-start geometry did not transfer cleanly into ArcFace, and four competing objectives on only 128 train IDs pushed the run into overfitting instead of a better optimum |

### 4.5 ResNeXt101-IBN-a ArcFace Attempt (09j)

| Version | Account | Recipe | Result | Verdict | Key Insight |
|:-------:|:-------:|--------|:------:|:-------:|-------------|
| 09j v2 | gumfreddy | ResNeXt101-32x8d-IBN-a + ArcFace + GeM + BNNeck, **160 epochs**, **batch 48**, **lr=3.5e-4**, **384x384** | **mAP=36.88%**, **R1=62.69%**, **mAP_rerank=40.49%** | REJECTED | The original IBN-Net ResNeXt weights were built for **32x32d grouped convolutions**, while the training model used **32x8d**. The v2 fix filtered loading with `strict=False`, but that only avoided the crash; it left many layers randomly initialized and crippled the model |

The **09j v2** result closes out the ResNeXt101-IBN-a path for this codebase. Even after fixing the explicit weight-shape mismatch from **v1**, the fallback partial-load strategy still produced a catastrophically weak model because too much of the backbone was effectively random at initialization. At **36.88% mAP**, this run finished not just below the **52.77%** ResNet101-IBN-a baseline, but far below the minimum quality needed for any ensemble use. The practical conclusion is that the current non-CLIP CNN routes are dead ends on CityFlowV2 unless a truly compatible pretrained checkpoint exists.

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
| Network flow / Hungarian solver | 10c v53 reached **76.9%** vs **77.14%** CC baseline and increased conflation **27 -> 30** | This implementation was not helpful; only worth revisiting if a materially different formulation is proposed | LOW |
| Multi-Scale TTA | Marginal + timeout | Selective approach needed | LOW |
| CamTTA | Hurts MTMC, helps GLOBAL | CLOSED | CLOSED |

---

## 7. What Has NEVER Been Tried (Real Opportunity Space)

| Approach | Stage | Est. Gain | Priority |
|----------|:-----:|:---------:|:--------:|
| 384px ViT-B/16 (proper training) | Training->S2 | +1.0-2.0pp | ***** |
| ResNet101-IBN-a ensemble | Training->S2->S4 | +1.0-1.5pp | ***** |
| Alternative CLIP-family backbone beyond LAION-2B and EVA02 (for example DFN-2B) | Training | +0.5-1.5pp | **** |
| Timestamp bias correction | S4 | +0.3-0.5pp | **** |
| Per-camera CLAHE tuning | S0 | +0.2-0.5pp | *** |
| Hand-annotated zone polygons | S4 | +0.5-1.5pp | *** |
| DMT camera-aware training | Training | +1.0-1.5pp | *** |
| GNN edge classification | S4 | +1.0-3.0pp | ** |

---

## 8. Key Insights & Dead Ends

1. **"Association tuning is exhausted — the gap is feature quality."** 220+ configs; remaining gap needs better embeddings.
2. **Power normalization (+0.5pp)**: signed sqrt before PCA spreads similarity distribution.
3. **IBN-a layer3 bug**: 09d v12's 21.9% mAP was from IBN on layers 1+2 only. Fixed in v13.
4. **CamTTA hurts MTMC**: Adapts BN per camera -> features become camera-specific. FIC whitening in S4 is correct approach.
5. **Multi-scale TTA is useless**: Tested 4+ times, never helped.
6. **PCA=512D hurts by 0.78-3pp**. 384D is optimal.
7. **mtmc_only=True drops IDF1 by ~5pp**: Discards single-camera tracklets. **Re-confirmed as active bug in 10c v8 (2026-04-22)** — all v8 results are ~5pp biased. Fixed in commit `69e67a0`.
8. **CSLS is CATASTROPHIC (-34.7pp)**: Destroys similarity structure.
9. **Feature concat ensemble hurts (-1.6pp)**: Score-level fusion is correct.
10. **Reranking hurts with weak features**: Re-test after feature quality improvement.
11. **S02_c006 is catastrophic (74% IDF1)**: FP ratio 6.86x, GT covers only 43% of video.
12. **Fragmentation dominates**: Under-merging errors 1.7-2.5x over-merging. Feature quality problem.
13. **SAM2 foreground masking is catastrophic for vehicle MTMC (-8.7pp)**: 10a v29 + 10c v50 peaked at only **0.688 MTMC IDF1** after a full **60-config** sweep, versus the **0.775** non-SAM2 baseline. Masking removes useful background context and likely clips vehicle boundary cues while raising runtime from **~65 min to 105.2 min**.
14. **Network flow did not solve conflation**: 10c v53 reached only **0.769 MTMC IDF1** vs the **0.7714** CC baseline. Although **MOTA/HOTA** ticked up slightly and **ID switches** fell by 2, conflation actually worsened from **27 -> 30** predicted IDs, so the current CC-based solver remains preferable.
15. **Topology CID_BIAS is also a dead end**: 10c v55 tested additive camera-pair offsets from **(+0.02/-0.10)** through **(+0.06/-0.20)** and all regressed to **0.764-0.762-0.763** versus a **0.774** control. FIC whitening already handles the useful cross-camera calibration; additive CID_BIAS just distorts those calibrated similarities.
16. **09l v3 makes LAION-2B CLIP ensemble-ready**: after resuming the stable **09l v2** EMA checkpoint from **epoch 160 -> 300**, **09l v3** reached **78.61% mAP / 90.43% R1 / 81.09% mAP_rr / 90.98% R1_rr**. The extension improved steadily through **epoch 180 -> 300** and finished only **1.53pp** behind the deployed **OpenAI CLIP 09b v2** baseline at **80.14% mAP**, so this backbone now clears the practical **~65%** ensemble threshold by a wide margin.
17. **EVA02 ViT-B/16 CLIP is a dead end with the current recipe**: **09o v1** reached only **48.17% mAP / 65.90% R1 / 77.17% R5 / 82.83% R10** after **120 epochs** with **AdamW** and **CE+Triplet+Center**. That is not just below ensemble usefulness, but even below the fine-tuned **R50-IBN** secondary at **63.64% mAP**, so the backbone does not transfer well to CityFlowV2 vehicles under the present setup.
18. **OSNet VeRi-776 secondary is closed out**: both reproduction strategies failed with currently available weights. Score-level fusion at **10%** reached only **76.7%**, concat reached **76.4%**, and the original `vehicle_osnet_veri776.pth` checkpoint that enabled the historical v80 path is no longer available in the weights datasets.

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
| 09s v1 | COMPLETE | yahiaakhalafallah | **DINOv2 ViT-L/14** (`vit_large_patch14_dinov2.lvd142m`) finished with **mAP=86.79%**, **R1=96.15%**, best epoch **115/120** (training BREAKTHROUGH). Full pipeline `10a→10b→10c` completed 2026-04-25: **MTMC IDF1=0.744** (best, with AFLink `gap=150`, `dir_cos=0.85`); **-3.1pp** vs ViT-B/16 CLIP (0.775). Higher mAP did NOT translate to better MTMC. |
| 09r v7 | FAILED | yahiaakhalafallah | **ViT-L TransReID** (`vit_large_patch16_224.augreg_in21k_ft_in1k`) finished cleanly on Kaggle **T4 x2** but only reached **mAP=60.38%**, **R1=76.57%**, best epoch **108/120**. Despite the larger backbone, it underperformed the **ViT-B/16 CLIP** baseline by **-19.76pp mAP**, confirming that pretraining quality matters far more than model size. |
| 10c v31 | COMPLETE | ali369 | Association sweep complete: mtmc_idf1=78.0%; only gallery_thresh improved (+0.2pp) |
| 09d v13 | COMPLETE | mrkdagods | Timed out after epoch 19: mAP=11.98%; IBN layer1+2+3 fix confirmed, but training recipe is not converging |
| 09l v1 | COMPLETE | gumfreddy | **mAP=20.36%**, **R1=53.03%**, **mAP_rr=27.16%**; catastrophic failure because **CircleLoss(gamma=128)** overflowed fp16 autocast and kept loss at `inf` throughout |
| 09l v2 | COMPLETE | gumfreddy | **mAP=61.51%**, **R1=81.41%**, **mAP_rr=67.20%**, **R1_rr=82.95%** after **160 epochs** with **TripletLoss + EMA(decay=0.9999)**; strong recovery over v1, but still clearly unconverged because mAP rose **55.98% -> 61.51%** from epoch **140 -> 160** as cosine LR ended |
| 09l v3 | COMPLETE | gumfreddy | resumed from the **09l v2 EMA** checkpoint to **300 total epochs**; **mAP=78.61%**, **R1=90.43%**, **mAP_rr=81.09%**, **R1_rr=90.98%**; strong secondary model only **1.53pp** behind **09b v2 (80.14% mAP)** and now ready for ensemble deployment |
| 09o v1 | COMPLETE | gumfreddy | **mAP=48.17%**, **R1=65.90%**, **R5=77.17%**, **R10=82.83%** after **120 epochs** with **AdamW** and **CE+Triplet+Center**; weaker than both the primary ViT baseline and the fine-tuned **R50-IBN** secondary, so EVA02 is a dead end for ensemble use under the current recipe |
| 09q | Exp B COMPLETE; v5 PENDING | gumfreddy | **Exp B (CLIP init)**: mAP=76.52% after 120 epochs — no improvement over 80.14%. **Exp A (resume 80.14%)**: never ran — checkpoint path bug fixed; **09q v5 pending**. |
| 14r primary | ABORTED | MRKDaGods | CLIP-ReID 2-stage VeRi-776 run aborted by walltime guard after Stage 1 completed 120ep in **5.66h** and Stage 2 completed epoch 1. Prompts saved and finite; methodology not evaluated yet. Kernel `mrkdagods/14r-clip-reid-veri-776-train`; quota cost about **9h**. |
| 14r probe | FAIL | gumfreddy | DINOv2 ViT-B/14 standalone VeRi-776 probe using the 09v-style recipe. Best post-rerank **89.27% mAP / 98.15% R1**, below WIN/MARGINAL mAP; confirms SSL alone underperforms CLIP pretraining. Kernel `gumfreddy/14r-probe-dinov2-veri-776-train`. |
| 14r recovery | RUNNING | gumfreddy | Stage-2-only CLIP-ReID resume from saved prompts dataset `gumfreddy/14r-clip-reid-stage1-prompts`; batch **32 -> 64**, epochs **120 -> 60**, sqrt-scaled LR **4.95e-4/4.95e-3**, eval at **[20, 40, 50, 55, 60]**. Kernel `gumfreddy/14r-recovery-clip-reid-stage-2`; ETA about **3.5-4.2h**. |

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
| Camera bias? | Basic camera bias: -0.4pp. CID_BIAS variants also failed: GT-learned -3.3pp, topology bias -1.0 to -1.2pp. | 3.1, 2.6 |
| FAC? | -2.5pp | 3.1 |
| mtmc_only? | -5pp. Always false. | 3.2 |
| Knowledge distillation? | Tried poorly (22%). Needs fix. | 4.2 |
| Circle loss? | Yes. 09 v4 collapsed to 18.45% mAP with `inf` loss; dead end. | 4.1 |
| EVA02? | 09o v1 reached 48.17% mAP / 65.90% R1. Dead end with current recipe. | 4.6 |
| ResNet101-IBN-a? | v13 complete: 11.98% at e19; recipe needs investigation. | 4.3 |
| Feature concat? | -1.6pp. Don't repeat. | 3.4 |
| Track smoothing? | Harmful. Don't repeat. | 3.2 |
| Power normalization? | +0.5pp. Enabled. | 3.4 |
| Multi-scale TTA? | Always neutral/harmful. | 3.4 |
| Denoise? | -2.7pp. Keep off. | 3.3 |

### 10c v60 — Fine-tuned R50-IBN Fusion (gumfreddy)
- **Date**: 2026-04-20
- **Notebook**: 10c v60 (10a v37 -> 10b v22 -> 10c v60)
- **Change**: Deploy fine-tuned R50-IBN (09n, mAP=63.64%) as secondary model, sweep fusion weights [0.0, 0.05, ..., 0.5]
- **Results**:
	- w=0.00: MTMC IDF1=0.7730 (baseline, primary only)
	- w=0.05: MTMC IDF1=0.7733
	- w=0.10: MTMC IDF1=0.7736 <- BEST (+0.06pp)
	- w=0.15: MTMC IDF1=0.7713
	- w=0.20: MTMC IDF1=0.7715
	- w=0.25: MTMC IDF1=0.7715
	- w=0.30: MTMC IDF1=0.7714
	- w=0.40: MTMC IDF1=0.7702
	- w=0.50: MTMC IDF1=0.7625
- **Conclusion**: Marginal gain (+0.06pp). R50-IBN at 63.64% mAP still too weak for meaningful ensemble. Need >=70% mAP or different architecture.

### 10c v61 — Improved 09p R50-IBN Fusion (gumfreddy)
- **Date**: 2026-04-20
- **Notebook**: 10c v61 (10a `run_kaggle_20260420_201401` -> 10b v23 -> 10c v61)
- **Kernel**: `gumfreddy/mtmc-10c-stages-4-5-association-eval` v61
- **Change**: Deploy the improved **09p** R50-IBN secondary through the newer 10a chain and repeat the fusion sweep over weights **[0.0, 0.05, ..., 0.5]**
- **Canonical result rule**: treat the **best fusion-sweep value** as canonical for this run, not the one-off Stage-5 log line that printed **[MTMC] IDF1=77.1%, MOTA=68.9%, HOTA=57.5%, IDSW=198** during one pass
- **Results**:
	- w=0.00: MTMC IDF1=0.773021
	- w=0.05: MTMC IDF1=0.773255
	- w=0.10: MTMC IDF1=0.773595 <- BEST
	- w=0.15: MTMC IDF1=0.772622
	- w=0.20: MTMC IDF1=0.771648
	- w=0.25: MTMC IDF1=0.771440
	- w=0.30: MTMC IDF1=0.771440
	- w=0.40: MTMC IDF1=0.770557
	- w=0.50: MTMC IDF1=0.761920
- **Conclusion**: The improved **09p** secondary and newer ingestion chain still produce only a marginal gain over baseline (**+0.000574** at best). This remains below the reproducible **0.775** ceiling and confirms the bottleneck is still primary feature quality / architecture rather than missed association tuning.

---

### Run 10c-v5 (yahia) — 3-way ensemble sweep (BROKEN — regression bug)
- **Date**: 2026-04-22
- **Kernel**: yahiaakhalafallah/mtmc-10c-stages-4-5-association-eval v5
- **Base**: 10a v4 (yahia) with WRONG overrides: \concat_patch=true\, \camera_bn.enabled=false- **Result**: baseline 73.57% (expected ~77.36%) — regression confirmed, -3.79pp
- **Best fusion**: w2=0.05, w3=0.15 → 73.73% (+0.16pp over broken baseline only)
- **Full sweep**:
  | Config | w2 | w3 | MTMC IDF1 |
  |---|---|---|---|
  | no_fusion_control | 0.00 | 0.00 | 73.57% |
  | baseline_floor | 0.10 | 0.00 | 73.55% |
  | w2_005_w3_015 | 0.05 | 0.15 | **73.73%** |
  | w2_000_w3_015 | 0.00 | 0.15 | 73.23% |
  | w2_000_w3_020 | 0.00 | 0.20 | 73.12% |
  | w2_005_w3_020 | 0.05 | 0.20 | 72.32% |
  | w2_010_w3_010 | 0.10 | 0.10 | 72.58% |
  | w2_010_w3_015 | 0.10 | 0.15 | 72.58% |
  | w2_015_w3_010 | 0.15 | 0.10 | 72.55% |
  | w2_020_w3_010 | 0.20 | 0.10 | 72.59% |
- **Conclusion**: results invalidated by regression bug; do not use these numbers

### Fix: 10a v5 — regression corrections
- **Date**: 2026-04-22
- **Kernel**: yahiaakhalafallah/mtmc-10a-stages-0-2 v5
- **Changes**: \concat_patch=true\ → \alse\, \camera_bn.enabled=false\ → \	rue- **Root cause of regression**:
  - \concat_patch=true\ changes ViT embedding from 768D → 1536D; PCA was trained on 768D, corrupting downstream features (-3.79pp impact)
  - \camera_bn.enabled=false\ disables cross-camera batch normalisation (~-2pp impact)
- **Status**: **COMPLETE** — 49.4 min runtime, 929 tracklets, primary + secondary + tertiary embeddings (all 384D); baseline restored
- **Followed by**: 10b v3 (FAISS index), 10c v8 (19-point ensemble sweep, biased by MTMC_ONLY=True), 10c v9 (MTMC_ONLY fix — running)

### 09q v4 — Extended TransReID training
- **Date**: 2026-04-22
- **Kernel**: mrkdagods/09q-transreid-cityflow-v10 v4
- **Status**: RUNNING — 120 epochs extended fine-tuning from 80.14% mAP checkpoint
- **Fixes applied**:
  - Added missing DataLoader cell (train_loader was never defined in v3)
  - Added \	hanhnguyenle/data-aicity-2023-track-2\ to \dataset_sources  - Added \mtmc-weights\ download cell for cross-account dataset access
- **Goal**: push primary mAP from 80.14% → 82–84%+ for stronger ensemble foundation


---

## 2026-04-25 Session: 09r ViT-L Failure and 09s DINOv2 Breakthrough

### 09s v1 - DINOv2 ViT-L/14 on CityFlowV2 (BREAKTHROUGH)
- **Date**: 2026-04-25
- **Kernel**: `yahiaakhalafallah/09s-dinov2-large-cityflowv2`
- **Status**: **BREAKTHROUGH**
- **Architecture**: **DINOv2 ViT-L/14** (`vit_large_patch14_dinov2.lvd142m`), **IMG_SIZE=252**, **STRIDE=14**, **120 epochs**, **BATCH=32**
- **Recipe**: same stable loss stack as 09r (**ID + batch-hard triplet + delayed center loss**) to isolate backbone/pretraining effects rather than recipe changes
- **Best result**: **mAP=86.79%**, **R1=96.15%**, best epoch **115/120**
- **Delta vs previous best**: **+6.65pp mAP**, **+3.88pp R1** relative to the prior best **ViT-B/16 CLIP** baseline (**80.14% mAP**, **92.27% R1**)
- **Training quality**: converged much better than 09r, ending around **loss=1.54** with **train_acc=0.9855** versus **loss=2.19** and **train_acc=0.9165** for the ViT-L AugReg run
- **Interpretation**: this is the strongest CityFlowV2 vehicle ReID model trained so far and the first concrete result that plausibly changes the MTMC ceiling rather than nudging it. DINOv2 large-scale self-supervised pretraining appears to transfer better to cross-camera vehicle identity than both **ViT-B/16 CLIP** and larger non-CLIP ImageNet-style pretraining.

### 09r v7 - ViT-L TransReID on CityFlowV2 (FAILED)
- **Date**: 2026-04-25
- **Kernel**: `yahiaakhalafallah/09r-vit-large-cityflowv2`
- **Status**: **FAILED**
- **Runtime**: **~2h37m** on Kaggle **T4 x2**
- **Architecture**: **ViT-Large patch16** (`vit_large_patch16_224.augreg_in21k_ft_in1k`), **256px**, **STRIDE=16**, **120 epochs**, **BATCH=32**, **CENTER_START=10**
- **Best result**: **mAP=60.38%**, **R1=76.57%**, best epoch **108/120**
- **Notebook fixes confirmed in this successful run**: gradient checkpointing, 2D `sie_embed`, positional `cam_ids` for DataParallel, and valid Kaggle notebook cell IDs
- **Delta vs previous best**: **-19.76pp mAP** versus the prior **ViT-B/16 CLIP** baseline (**80.14%**) despite the larger backbone
- **Interpretation**: 09r cleanly falsifies the idea that model size alone will rescue the vehicle ReID bottleneck. Large non-CLIP pretraining is not enough here; the learned feature geometry from stronger pretraining regimes matters far more than parameter count.

### April 25 Analysis
- **Main lesson**: pretraining quality is the decisive variable. **ViT-L without CLIP/DINOv2-quality pretraining failed badly**, while **DINOv2 ViT-L/14** delivered an immediate step change.
- **Strategic implication**: the feature bottleneck may no longer be fundamental. With **86.79% mAP / 96.15% R1**, the next required test is the full **10a -> 10b -> 10c** MTMC pipeline using the 09s checkpoint to measure actual downstream IDF1.

## 2026-04-22 Session: 3-Way Ensemble Sweep & MTMC_ONLY Bug Fix

### 10b v3 — FAISS Index from 10a v5
- **Date**: 2026-04-22
- **Input**: 10a v5 output (929 tracklets, 384D each)
- **Status**: **COMPLETE**
- **Output**: 12.6 MB FAISS checkpoint
- **Purpose**: Index 10a v5 features for downstream 10c v8/v9 evaluation

### 10c v8 — 3-Way Ensemble Sweep, 19 Configs (BIASED — MTMC_ONLY=True Bug)
- **Date**: 2026-04-22
- **Kernel**: `yahiaakhalafallah/mtmc-10c-stages-4-5-association-eval` v8
- **Status**: **COMPLETE** — results biased ~5pp low due to MTMC_ONLY=True bug
- **Bug**: `MTMC_ONLY=True` set in notebook AND `mtmc_only_submission: true` in `configs/datasets/cityflowv2.yaml` — drops single-camera tracks and costs ~5pp IDF1. Fixed in commit `69e67a0`.
- **Sweep**: 19 configs testing w2 (R50-IBN secondary weight) and w3 (LAION-2B tertiary weight)
- **Biased results** (add ~5pp for estimated unbiased values):

| Config | w2 (R50-IBN) | w3 (LAION ViT) | MTMC_IDF1 (biased) | Est. unbiased |
|---|---|---|---|---|
| no_fusion_control | 0.00 | 0.00 | 71.09% | ~76.09% |
| baseline_floor | 0.10 | 0.00 | 70.99% | ~75.99% |
| ter_005 | 0.00 | 0.05 | 71.13% | ~76.13% |
| ter_010 | 0.00 | 0.10 | 71.07% | ~76.07% |
| ter_015 | 0.00 | 0.15 | 71.13% | ~76.13% |
| ter_020 | 0.00 | 0.20 | 71.10% | ~76.10% |
| ter_025 | 0.00 | 0.25 | 71.14% | ~76.14% |
| ter_030 | 0.00 | 0.30 | 71.25% | ~76.25% |
| w2_005_w3_005 | 0.05 | 0.05 | 71.06% | ~76.06% |
| w2_005_w3_010 | 0.05 | 0.10 | 71.04% | ~76.04% |
| w2_005_w3_015 | 0.05 | 0.15 | 71.02% | ~76.02% |
| w2_005_w3_020 | 0.05 | 0.20 | 71.14% | ~76.14% |
| w2_005_w3_025 | 0.05 | 0.25 | 71.22% | ~76.22% |
| w2_010_w3_010 | 0.10 | 0.10 | 71.02% | ~76.02% |
| w2_010_w3_015 | 0.10 | 0.15 | 71.06% | ~76.06% |
| w2_010_w3_020 | 0.10 | 0.20 | 71.19% | ~76.19% |
| **w2_005_w3_030** | **0.05** | **0.30** | **71.28% (best)** | **~76.28%** |
| w2_010_w3_025 | 0.10 | 0.25 | 71.28% | ~76.28% |

- **Ensemble gain** (biased basis): best 3-way (71.28%) vs primary-only (71.09%) = **+0.19pp**
- **Pattern**: R50-IBN secondary (w2) hurts or is neutral at all tested weights; LAION tertiary (w3=0.30) gives marginal +0.16pp; 3-way best at w2=0.05 + w3=0.30 gives +0.19pp — all within noise
- **Interpretation**: Consistent with prior 2-way fusion dead ends — R50-IBN too weak; LAION CLIP too correlated with primary ViT. Ensemble rescue path remains exhausted.
- **Corrected run**: 10c v9 — **COMPLETE** (unbiased results below)

### 10c v9 — MTMC_ONLY=False Fix (COMPLETE)
- **Date**: 2026-04-22
- **Kernel**: `yahiaakhalafallah/mtmc-10c-stages-4-5-association-eval` v9
- **Status**: **COMPLETE** — unbiased results confirmed
- **Change**: Commit `69e67a0` — `MTMC_ONLY=False` in notebook, `mtmc_only_submission: false` in `configs/datasets/cityflowv2.yaml`
- **Baseline**: MTMC_IDF1=76.625%, IDF1=78.419%, MOTA=66.910%, HOTA=57.031%
- **Baseline note**: 76.625% is ~0.74pp below expected 77.36% — feature distribution shifted by camera_bn=true; V52 params need re-tuning.
- **Sweep results (19 configs, unbiased)**:

| Config | w2 (R50-IBN) | w3 (LAION ViT) | MTMC_IDF1 | Delta |
|---|---|---|---|---|
| no_fusion_control | 0.00 | 0.00 | 76.625% | — |
| baseline_floor / sec_010 | 0.10 | 0.00 | 76.561% | −0.064pp |
| ter_005 | 0.00 | 0.05 | 76.665% | +0.040pp |
| ter_010 | 0.00 | 0.10 | 76.586% | −0.039pp |
| ter_015 | 0.00 | 0.15 | 76.658% | +0.033pp |
| ter_020 | 0.00 | 0.20 | 76.641% | +0.016pp |
| ter_025 | 0.00 | 0.25 | 76.692% | +0.067pp |
| ter_030 | 0.00 | 0.30 | 76.779% | +0.154pp |
| w2_005_w3_005 | 0.05 | 0.05 | 76.582% | −0.043pp |
| w2_005_w3_010 | 0.05 | 0.10 | 76.637% | +0.012pp |
| w2_005_w3_015 | 0.05 | 0.15 | 76.615% | −0.010pp |
| w2_005_w3_020 | 0.05 | 0.20 | 76.692% | +0.067pp |
| w2_005_w3_025 | 0.05 | 0.25 | 76.764% | +0.138pp |
| **w2_005_w3_030** | **0.05** | **0.30** | **76.817% (BEST)** | **+0.192pp** |
| w2_010_w3_010 | 0.10 | 0.10 | 76.615% | −0.010pp |
| w2_010_w3_015 | 0.10 | 0.15 | 76.665% | +0.040pp |
| w2_010_w3_020 | 0.10 | 0.20 | 76.730% | +0.105pp |
| w2_010_w3_025 | 0.10 | 0.25 | 76.817% | +0.192pp |

- **Best fusion point**: w2=0.05, w3=0.30 → 76.817% (+0.192pp) — also tied w2=0.10, w3=0.25
- **R50-IBN secondary**: harmful alone (−0.064pp)
- **LAION tertiary alone**: +0.154pp at w3=0.30 — marginal, insufficient
- **Conclusion**: 3-way ensemble confirmed dead end; +0.192pp within noise


### Bug: MTMC_ONLY=True in 10c Notebook and cityflowv2.yaml (2026-04-22)
- **Bug**: `MTMC_ONLY=True` set in the 10c Kaggle notebook AND `mtmc_only_submission: true` in `configs/datasets/cityflowv2.yaml`
- **Impact**: ~5pp MTMC IDF1 penalty (single-camera tracklets dropped from submission, invisible to evaluator)
- **Affected runs**: All 10c v8 configs are biased ~5pp below their true values
- **Fix**: Commit `69e67a0` — `MTMC_ONLY=False` in notebook, `mtmc_only_submission: false` in config
- **First corrected run**: 10c v9 — **COMPLETE** (results above)

## 2026-04-25 Final Sprint — CLIP+DINOv2 Score-Level Ensemble Fusion Sweep

- **Date**: 2026-04-25
- **Kernel**: 10c v15 (gumfreddy account)
- **Pipeline chain**: 09s v1 (DINOv2 training) -> 10a DINOv2 features -> 10b DINOv2 FAISS -> 10c v15 fusion sweep
- **Design**: CLIP primary + DINOv2 tertiary score fusion, secondary slot disabled
- **Wall time**: ~7 min for 11 fusion pairs (vs 12 skipped pairs that required secondary)

| Label | w_tertiary | MTMC_IDF1 | IDF1 | MOTA | HOTA |
|---|---:|---:|---:|---:|---:|
| no_fusion_control | 0.00 | 0.7663 | 0.7842 | 0.6691 | 0.5703 |
| ter_005 | 0.05 | 0.7669 | 0.7846 | 0.6697 | 0.5706 |
| ter_010 | 0.10 | 0.7669 | 0.7846 | 0.6697 | 0.5706 |
| ter_015 | 0.15 | 0.7673 | 0.7851 | 0.6702 | 0.5710 |
| ter_020 | 0.20 | 0.7663 | 0.7840 | 0.6704 | 0.5706 |
| ter_025 | 0.25 | 0.7662 | 0.7840 | 0.6704 | 0.5706 |
| ter_030 | 0.30 | 0.7674 | 0.7853 | 0.6716 | 0.5717 |
| ter_040 | 0.40 | 0.7679 | 0.7851 | 0.6708 | 0.5719 |
| ter_050 | 0.50 | 0.7696 | 0.7857 | 0.6703 | 0.5721 |
| **ter_060** | **0.60** | **0.7703** | **0.7916** | **0.6725** | **0.5749** |
| ter_070 | 0.70 | 0.7693 | 0.7909 | 0.6716 | 0.5746 |

- **Skipped (12 pairs)**: `baseline_floor`, `sec_010`, `w2_005_w3_005`, `w2_005_w3_010`, `w2_005_w3_015`, `w2_005_w3_020`, `w2_005_w3_025`, `w2_005_w3_030`, `w2_010_w3_010`, `w2_010_w3_015`, `w2_010_w3_020`, `w2_010_w3_025`
- **Best**: `w_tertiary=0.60`, `MTMC IDF1=0.7703`, `IDF1=0.7916`, `MOTA=0.6725`, `HOTA=0.5749`
- **Conclusion**: +0.40pp local gain; this became the previous deployed best on available weights, later superseded by the 14e TTA + AQE k=2 headline at **0.77936**. It still remained below the historical **0.784** v80 peak, which depended on a now-unavailable OSNet checkpoint.
- **Reference**: DINOv2 ReID training in kernel 09s v1 (`mAP=86.79%`, `R1=96.15%`, epoch `115/120`)

## 2026-04-25 Session: OSNet Secondary Repro Investigation

- **Date**: 2026-04-25
- **Branch**: `repro/osnet-secondary`
- **Objective**: Reproduce the historical v80-era OSNet-assisted gain and resolve whether the missing performance was caused by code drift or by missing secondary-model weights.
- **Checkpoint audit**: The required `vehicle_osnet_veri776.pth` checkpoint is no longer present in either `gumfreddy/mtmc-weights` or `mrkdagods/mtmc-weights`. It was dropped when `mrkdagods/mtmc-weights` was regenerated on **2026-03-30**.

| Strategy | Pipeline Chain | Design | MTMC IDF1 | Verdict | Key Insight |
|:--------:|----------------|--------|:---------:|:-------:|-------------|
| A | 10a v10 -> 10b v7 -> 10c v18 | OSNet score-level fusion, `save_separate=True`, `w=0.10` | **76.7%** | REJECTED | **-0.8pp** vs the CLIP-only baseline; the public OSNet weights do not recover the historical gain |
| B | 10a v12 -> 10b v8 -> 10c v22 | OSNet concat, `save_separate=False`, **1280D -> 384D PCA** | **76.4%** | REJECTED | **-1.1pp** vs the CLIP-only baseline; concat also regresses with currently available weights |

- **Verdict**: **DEAD END**. The historical v80 **78.4%** result depended on OSNet concat with a CityFlowV2-adapted VeRi-776 checkpoint (`vehicle_osnet_veri776.pth`) that is now lost. The torchreid public VeRi-776 OSNet downloaded via `gdown` does not produce useful features for CityFlowV2, either as score-level fusion or concat.

### 09q — Extended TransReID Training Status (2026-04-22)
- **Date**: 2026-04-22
- **Kernel**: mrkdagods/09q-transreid-cityflow-v10
- **Goal**: Push primary ViT-B/16 CLIP mAP from 80.14% toward 82-84%+
- **Experiment B (CLIP init from scratch)**: Best **mAP=76.52%** after 120 epochs — **no improvement** over 80.14% baseline.
- **Experiment A (resume from 80.14% checkpoint)**: **NEVER RAN** — RESUME_CHECKPOINT_CANDIDATES did not include vehicle_transreid_vit_base_cityflowv2.pth. Bug fixed.
- **Status**: **09q v5 pending push** to re-run Exp A.

---

## 2026-04-25 Session: DINOv2 MTMC Pipeline Evaluation

### Pipeline Chain: 09s v1 → 10a DINOv2 → 10b DINOv2 → 10c DINOv2 v2

- **10a DINOv2**: `yahiaakhalafallah/mtmc-10a-dinov2` — Stages 0-2 with DINOv2 ViT-L/14 checkpoint from 09s v1 (`vit_large_patch14_dinov2.lvd142m`, IMG_SIZE=252). Run ID: `run_kaggle_20260425_035912`.
- **10b DINOv2**: `yahiaakhalafallah/mtmc-10b-dinov2-stage-3-faiss-indexing` — Stage 3 FAISS indexing of DINOv2 embeddings. COMPLETE.
- **10c DINOv2**: `yahiaakhalafallah/mtmc-10c-dinov2-stages-4-5-association-eval` — Stage 4-5 association sweep and evaluation. **v2 COMPLETE**.

### 10c DINOv2 v2 — Baseline (v52 params, no AFLink)
- **Date**: 2026-04-25
- **Config**: v52 baseline params — `sim_thresh=0.46`, `appearance_weight=0.60`, `fic_reg=0.20`, `aqe_k=3`, `gallery_thresh=0.48`, `orphan_match=0.38`, `intra_merge=True` (thresh=0.80). AFLink disabled.
- **Result**: **MTMC IDF1 = 0.688**, Per-camera IDF1 = 0.794
- **Comparison vs ViT-B/16 CLIP**: **-8.7pp** (0.688 vs 0.775)
- **Interpretation**: Single-camera quality is strong (0.794 per-cam IDF1), but cross-camera association badly underperforms the CLIP baseline despite 86.79% mAP. DINOv2 embeddings are not cross-camera invariant enough under the current pipeline.

### 10c DINOv2 v2 — Best (with AFLink gap=150, dir_cos=0.85)
- **Date**: 2026-04-25
- **Config**: Same baseline params + AFLink `max_spatial_gap_px=150`, `min_direction_cos=0.85`
- **Best swept params**: `sim_thresh=0.46`, `appearance_weight=0.60`, `fic_reg=0.20`, `aqe_k=3`, `gallery_thresh=0.48`, `orphan_match=0.38`, `intra_merge=True` (thresh=0.80)
- **Result**: **MTMC IDF1 = 0.744**, IDF1 = 0.755, MOTA = 0.624, HOTA = 0.547
- **AFLink effect on DINOv2**: **+5.6pp** (0.688 → 0.744)
- **AFLink comparison with ViT-B/16 CLIP**: ViT-B/16 CLIP lost **-3.82pp to -13.20pp** from AFLink in controlled retest (10c v52 addon). AFLink behavior is therefore **model-specific**.
- **Comparison vs ViT-B/16 CLIP best (no AFLink)**: **-3.1pp** (0.744 vs 0.775)
- **Interpretation**: AFLink provides partial compensation by bridging track gaps via motion consistency, suggesting DINOv2 features produce more short-gap track fragmentation than CLIP features. However, the cross-camera association deficit is structural and cannot be fully recovered with motion linking.

### Key Finding: ReID mAP Does NOT Predict MTMC IDF1 — Even for DINOv2
- **DINOv2 ViT-L/14**: mAP=86.79% → MTMC IDF1=0.744 (best, with AFLink)
- **ViT-B/16 CLIP**: mAP=80.14% → MTMC IDF1=0.775 (no AFLink)
- **Gap**: DINOv2 has **+6.65pp mAP** but **-3.1pp MTMC IDF1**
- **Root cause hypothesis**: The mAP metric evaluates within-distribution ranking (same camera split). MTMC requires cross-camera invariance. TransReID's training methodology (VeRi-776 pretraining with camera-aware labels → CityFlowV2 fine-tune with camera-pair ID loss) specifically targets cross-camera invariance. DINOv2's LVD-142M self-supervised pretraining builds powerful general visual representations but does not specifically optimize for cross-camera vehicle identity invariance.
- **Lesson**: The decisive variable for MTMC is **training methodology for cross-camera invariance** (TransReID recipe + CLIP pretraining), not raw model capacity or single-distribution mAP. This is now consistent with four data points: 384px (+0pp mAP → -2.8pp MTMC), augoverhaul (+1.45pp mAP → -5.3pp MTMC), DMT (+7pp mAP → -1.4pp MTMC), and DINOv2 (+6.65pp mAP → -3.1pp MTMC).
- **Best MTMC IDF1 remains**: **0.775 (ViT-B/16 CLIP, 10c v52)**

### Phase C - camera_bn=false Retest on v80-Restored Baseline (HYPOTHESIS FALSIFIED)
- **Date**: 2026-04-25
- **Kernels**: `yahiaakhalafallah/mtmc-10a-stages-0-2` v8, `yahiaakhalafallah/mtmc-10b-stage-3-faiss-indexing` v6, `yahiaakhalafallah/mtmc-10c-stages-4-5-association-eval` v17
- **Branch**: `fix/baseline-drift` (commit `7e242f6`, NOT merged)
- **Config change**: `camera_bn.enabled: true -> false`
- **Result**: **MTMC IDF1 = 0.7666** (CLIP-only baseline, no fusion sweep)
- **Delta vs current baseline (0.7663)**: **+0.03pp** (statistical noise)
- **Delta vs historical target (0.7750)**: **-0.84pp** (drift NOT recovered)
- **Verdict**: **HYPOTHESIS FALSIFIED** — `camera_bn=false` is not the baseline drift cause
