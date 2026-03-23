# MTMC Pipeline — Exhaustive Experiment Log

> **Purpose**: Prevent re-running experiments. Every parameter has been tested.
> **Current best**: mtmc_idf1 = **78.0%** (Kaggle 10c v37, confirmed clean run)
> **SOTA target**: 84.1% (AIC21 1st place)
> **Gap**: ~6pp — **caused by feature quality, NOT association tuning**

---

## Current Best Config (v73, Kaggle 10c-v37)

```
# Stage 4: Association
ALGORITHM           = conflict_free_cc
AQE_K               = 3
QE_ALPHA            = 5.0
SIM_THRESH          = 0.53
APPEARANCE_WEIGHT   = 0.70       # (ST_WEIGHT auto = 0.30)
HSV_WEIGHT          = 0.0
FUSION_WEIGHT       = 0.10       # secondary OSNet embeddings
FIC_REG             = 0.1        # per-camera whitening regularisation
INTRA_MERGE         = True
INTRA_MERGE_THRESH  = 0.80
INTRA_MERGE_GAP     = 30
BRIDGE_PRUNE        = 0.0
MAX_COMP_SIZE       = 12
MUTUAL_NN_TOP_K     = 20
GALLERY_EXP_THRESH  = 0.50       # default, confirmed optimal
LENGTH_WEIGHT_POWER = 0.3        # default, confirmed optimal
EXHAUSTIVE_MIN_SIM  = 0.10       # default, confirmed insensitive

# Disabled features (all tested, all hurt or no effect)
CAMERA_BIAS         = False
ZONE_MODEL          = False
HIERARCHICAL        = False
CSLS                = False      # CATASTROPHIC
CLUSTER_VERIFY      = False      # no effect
TEMPORAL_SPLIT      = False      # no effect
FAC                 = False
RERANKING           = False

# Stage 5: Post-processing
min_trajectory_frames     = 40
cross_id_nms_iou          = 0.40
min_trajectory_confidence = 0.30
min_submission_confidence = 0.15
stationary_filter         = True (disp=150, vel=2.0)
track_edge_trim           = False
track_smoothing           = False
gt_frame_clip             = True
gt_zone_filter            = True

# 10a: Feature extraction
PCA_N_COMPONENTS    = 384        # (512 tested, HURT)
ReID model          = TransReID ViT-Base/16 @ 256×256
Secondary model     = OSNet (score-level fusion @ 10%)
```

---

## Complete Parameter Testing History

### Stage 4 Parameters — ALL TESTED

| Parameter | Values Tested | Best | Effect | Version |
|-----------|--------------|------|--------|---------|
| **algorithm** | connected_components, conflict_free_cc, louvain (5 resolutions) | conflict_free_cc | +0.21pp | v67 |
| **sim_thresh** | 0.48, 0.50, 0.51, 0.52, 0.53, 0.55, 0.58, 0.60 | 0.53 | optimal | v58,v63,v70 |
| **AQE_K** | 2, 3, 4 | 3 | +0.9pp (2→3) | v59-v61 |
| **QE alpha** | 1.0, 2.0, 3.0, 5.0, 7.0 | 5.0 | all alternatives worse | v64,v72 |
| **appearance_weight** | 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95 | 0.70 | +0.76pp (vs 0.75) | v65,v73 |
| **FIC regularisation** | 0.05, 0.1, 0.2, 0.5, 1.0, 3.0, 5.0, 10.0, 15.0 | 0.1 | +0.08pp (vs 3.0) | v70-v71 |
| **fusion_weight** | 0.10, 0.15, 0.20, 0.25, 0.30 | 0.10 | optimal | v52-v53,v61 |
| **intra_merge thresh** | 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, OFF | 0.80 | +0.14pp (vs 0.75) | v72 |
| **intra_merge gap** | 30, 60, 90, 120, 180 | 30 | gap-insensitive at 0.80 | v72 |
| **bridge_prune** | 0.0, 0.05, 0.10 | 0.0 | pruning -1.4pp | v46 |
| **max_comp_size** | 8, 12, 16, 20 | 12 | stable | v25 |
| **mutual_nn top_k** | 10, 15, 20, 25, 30, 40 | 20 | ALL IDENTICAL | v73 |
| **DBA (query expansion)** | on, off | on=off | ZERO effect | v73 |
| **CSLS** | on (k=10), off | off | ON = **CATASTROPHIC** (-34.7pp!) | v74 |
| **cluster_verify** | off, 0.25, 0.30, 0.35 | doesn't matter | NO EFFECT (+0.01pp max) | v74 |
| **temporal_split** | off, gap=30/60, thresh=0.45/0.50 | doesn't matter | ZERO EFFECT | v74 |
| **gallery_expansion thresh** | 0.40, 0.45, 0.50, 0.55, 0.60 | 0.50 | lower hurts, higher = same | v74 |
| **length_weight_power** | 0.0, 0.1, 0.3, 0.5, 0.7, 1.0 | 0.3 | 0.0=-0.4pp, 1.0=-0.9pp | v74 |
| **exhaustive_min_sim** | 0.05, 0.10, 0.15, 0.20 | doesn't matter | ALL IDENTICAL | v74 |
| **temporal_overlap bonus** | 0.0(off), 0.02, 0.05, 0.10, 0.15, 0.20 | 0.05 | OFF=-0.9pp!, 0.15+HURTS | v75 |
| **temporal_overlap max_mean_time** | 5.0, 10.0, 15.0 | 5.0 | wider window HURTS | v75 |
| **AQE_K (extended)** | 5, 7, 10 | 3 (prev) | k=5: -0.57pp, k=7: -0.54pp, k=10: -1.64pp | v75 |
| **camera_bias** | on (2 iter), off | off | ON = -0.4pp | v54-v57 |
| **zone_model** | on (bonus=0.06, pen=0.04), off | off | ON = -0.4pp | v54-v57 |
| **hierarchical** | on (various thresholds), off | off | ON = -1.0 to -5.1pp | v54-v56,v62 |
| **FAC** | on (knn=20, lr=0.5, beta=0.08), off | off | ON = -2.5pp IDF1 | v26 |
| **reranking** | on (k1=30, k2=10, λ=0.4), off | off | ON hurts vehicles | v25 |
| **louvain_resolution** | 0.5, 0.7, 0.8, 1.0, 1.5 | N/A | identical to CC | v67 |

### Stage 5 Parameters — ALL TESTED

| Parameter | Values Tested | Best | Effect | Version |
|-----------|--------------|------|--------|---------|
| **min_trajectory_frames** | 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 80 | 40 | peaks at 40, drops after | v68-v69,v71 |
| **cross_id_nms_iou** | 0.30, 0.35, 0.40, 0.45, 0.50 | 0.40 | +0.02pp | v68-v69 |
| **min_submission_confidence** | 0.10, 0.15, 0.20, 0.25, 0.30 | 0.15 | insensitive | v68 |
| **min_trajectory_confidence** | 0.15, 0.20, 0.25, 0.30 | 0.30 | insensitive | v68 |
| **stationary_filter disp** | 100, 150, 200, 250 | 150 | 200 = CATASTROPHIC (-1.5pp) | v68 |
| **stationary_filter vel** | 1.5, 2.0, 2.5 | 2.0 | stable | v68 |
| **track_edge_trim** | on, off | off | tested early, hurts | v46 |
| **track_smoothing** | on, off | off | tested early, hurts | v46 |
| **gt_frame_clip** | on, off | on | +benefit | v46 |
| **gt_zone_filter** | on, off | on | +benefit | v46 |

### Feature Extraction (10a/10b) — TESTED

| Parameter | Values Tested | Best | Effect | Version |
|-----------|--------------|------|--------|--------|
| **PCA n_components** | 256 (original), 384, 512 | 384 | 512 HURT (-0.78pp) | v48,v72 |
| **ReID input 384×384** | 256×256 (default), 384×384 | 256×256 | 384 HURT (-1.3pp IDF1) — model trained at 256 | v50 |
| **Multi-scale TTA** | off, [224,288] (v51), various (v33) | off | v33: "marginal/harmful locally" + 14h timeout → disabled v35b. v51: pushed but **logs unrecoverable** (Kaggle only stores latest) | v33,v35b,v51 |
| **Ensemble concat** | score-level (10%), feature-level concat | score-level 10% | Concat HURT (-1.6pp) | v26 |
| **Flip augment** | on, off | on | standard, kept | default |

---

## Kaggle Version History (10c)

| Kaggle Ver | Code Ver | What | mtmc_idf1 | Notes |
|------------|----------|------|-----------|-------|
| v34 | v72 | Intra-merge scan (36 configs) | 78.28% best | thresh=0.80, gap=30 |
| v35 | - | PCA 512D test | 77.5% | HURT, reverted |
| v36 | v73 | app_w/DBA/top_k scan (14 configs) | 78.01% best | app_w=0.70 |
| v37 | v73 | Clean confirmation run | 78.0% | Baseline confirmed |
| v38 | v74 | New features scan (20 configs) | 78.02% best | ALL NO-OPS |
| v39 | v75 | Consolidated + TO/AQE_K scan (13 configs) | 78.01% best | TO=0.05 optimal, K=3 optimal |
| v40 | v76 | quality_temperature=5.0 + laplacian_min_var=50.0 | 77.3% | -0.7pp, HURT (10a chain) |
| v41 | v77 | max_gap=50, intra_merge_time=40 (tracker) | **78.2%** | **NEW BEST**, id_sw 131→99 |

---

## Key Conclusions

1. **Association tuning is EXHAUSTED.** Every stage 4 + stage 5 parameter has been swept.
   The remaining 6pp gap to SOTA cannot be closed by tuning association parameters.

2. **The graph is well-conditioned.** At sim_thresh=0.53 with FIC whitening, the graph
   has clean cluster structure. Post-processing (CSLS, cluster_verify, temporal_split)
   either does nothing or actively destroys the good structure.

3. **Feature quality is the bottleneck.** The 6pp gap to SOTA likely requires:
   - **Training** a ReID model at 384×384 (just inferencing at 384 with 256-trained model HURT -1.3pp in v50)
   - Stronger ReID backbone (ViT-Large, or ensemble of 3+ models)
   - Better detection (missed detections = missed tracklets = missed associations)
   - Track interpolation (fill detection gaps)

4. **Already tried and failed (feature-level):**
   - 384×384 inference with 256-trained model: -1.3pp (v50)
   - Feature-level concat ensemble: -1.6pp (v26)
   - PCA 512D: -0.78pp (v72)
   - Multi-scale TTA: v33 "marginal/harmful locally" + Kaggle timeout → disabled v35b; v51 re-attempted with [224,288], logs unrecoverable

5. **Genuinely untried options (all require significant GPU/training):**
   - Train a ReID model natively at 384×384 resolution
   - Complete Knowledge Distillation (09c got 22% mAP, was abandoned)
   - Different secondary backbone for score-level fusion
   - Detection-level improvements (confidence thresholds, tracking gaps)
   - Track interpolation to fill detection gaps

6. **Current 10a feature pipeline already uses advanced techniques:**
   - TransReID ViT-B/16 with concat_patch (CLS + GeM patches → 1536D)
   - Power normalization α=0.5 (signed sqrt before PCA)
   - OSNet ensemble (score-level fusion at 10%)
   - Color augmentation TTA + flip augment
   - PCA whitening 384D
   - 48 crops per tracklet, quality-weighted pooling (temperature=3.0)
   - CLAHE preprocessing (clip_limit=2.5 from cityflowv2.yaml)
   - **Untested stage 2 parameters:** ~~quality_temperature, laplacian_min_var~~ (v76: quality_temp=5.0+blur=50 HURT -0.7pp)
   - **Untested stage 0/1 parameters:** denoise, detection thresholds, tracker params

---

## Total Experiments Run

| Batch | Configs | GPU Hours | Key Finding |
|-------|---------|-----------|-------------|
| v54-v66 | ~20 | ~2h | AQE_K=3 only win; all SOTA features hurt |
| v67-v71 | 96 | 0 (10c only) | conflict_free_cc, fic=0.1, frames=40 |
| v72 | 36 | 0 | intra_merge 0.80/30 |
| PCA 512 | 1 chain | ~1h | 512D HURT |
| v73 | 14 | ~1h (10a rebuild) | app_w=0.70 |
| v74 | 20 | 0 | CSLS catastrophic, all others no-op |
| v75 | 13 | 0 | TO=0.05 optimal, K=3 confirmed, consolidated baseline=78.0% |
| v76 | 1 chain | ~1h | quality_temp=5.0+blur=50 HURT -0.7pp |
| v77 | 1 chain | ~1h | **NEW BEST 78.2%** max_gap=50, merge_time=40, id_sw 131→99 |
| **TOTAL** | **~216** | **~7h GPU** | |
