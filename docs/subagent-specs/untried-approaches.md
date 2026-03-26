# Untried Approaches to Beat MTMC SOTA by a Big Margin

**Date**: 2026-03-24  
**Current Best**: IDF1 = 78.4% (Kaggle v80 / 10c v44 / ali369)  
**SOTA Target**: IDF1 ≈ 84.1% (AIC21 1st place)  
**Gap**: 5.7pp  
**Experiments Run**: 225+ configs across 14h GPU  
**Verdict**: Association tuning is **exhausted**. The gap is **feature quality**.

---

## A. What's Been Exhaustively Tried (DON'T RE-TREAD)

### A1. Association Parameters — 220+ Configs (Stage 4)
| Parameter | Configs Tested | Optimal |
|-----------|---------------|---------|
| `sim_thresh` | 0.48–0.60 (8 values) | **0.53** |
| `appearance_weight` | 0.60–0.95 (8 values) | **0.70** |
| `FIC regularisation` | 0.05–15.0 (9 values) | **0.1** |
| `AQE_K` | 2–10 (6 values) | **3** |
| `QE alpha` | 1.0–7.0 (5 values) | **5.0** |
| `fusion_weight` (OSNet) | 0.10–0.30 (5 values) | **0.10** |
| `intra_merge threshold` | 0.60–0.85+OFF (7 values) | **0.80** |
| `bridge_prune_margin` | 0.0–0.10 (3 values) | **0.0** |
| `max_component_size` | 8–20 (4 values) | **12** |
| `mutual_nn top_k` | 10–40 (6 values) | **20** (all identical) |
| `length_weight_power` | 0.0–1.0 (6 values) | **0.3** |
| `temporal_overlap bonus` | 0.0–0.20 (6 values) | **0.05** |
| `gallery_expansion thresh` | 0.40–0.60 (5 values) | **0.50** |
| Algorithm | CC, conflict_free_cc, Louvain(5 res), agglomerative | **conflict_free_cc** |
| DBA | on/off | identical |
| CSLS | on/off | OFF (CATASTROPHIC -34.7pp) |
| Cluster verify | off, 0.25–0.35 | NO EFFECT |
| Temporal split | off, gap=30/60, thresh=0.45/0.50 | ZERO EFFECT |
| Camera bias | on/off | OFF (-0.4pp) |
| Zone model | on/off | OFF (-0.4pp) |
| Hierarchical centroid | on (various), off | OFF (-1.0 to -5.1pp) |
| FAC | on/off | OFF (-2.5pp) |
| Reranking | on/off | OFF (hurts vehicles) |

### A2. Post-Processing — Fully Swept (Stage 5)
| Parameter | Values | Optimal |
|-----------|--------|---------|
| `min_trajectory_frames` | 5–80 (12 values) | **40** |
| `cross_id_nms_iou` | 0.30–0.50 | **0.40** |
| `stationary_filter disp` | 100–250 | **150** |
| `track_edge_trim` | on/off | **OFF** |
| `track_smoothing` | on/off | **OFF** |
| `mtmc_only` | true/false | **false** (+5pp) |

### A3. Tracker Tuning (Stage 1)
| Parameter | Values | Optimal |
|-----------|--------|---------|
| `min_hits` | 2, 3 | **2** (+0.2pp) |
| `confidence_threshold` | 0.20, 0.25 | **0.25** (0.20 HURT -2.8pp) |
| `denoise` | true/false | **false** (-2.7pp) |
| `max_iou_distance` | 0.5, 0.7 | **0.7** (0.5 HURT -1.6pp) |
| `max_gap` | 30, 50, 60, 80 | **50** (+0.2pp) |
| `intra_merge.max_time_gap` | 5–60 | **40** |

### A4. Feature Dimensions
| PCA dims | Result |
|----------|--------|
| 256D | Suboptimal (historic) |
| **384D** | **Optimal** |
| 512D | HURT -0.78pp |

---

## B. Tried But Poorly or Only Once — Worth Revisiting

### B1. 384×384 Native Training (09b) — mAP=0.4494 → FAILED
**What happened**: Notebook 09b attempted to fine-tune TransReID at 384×384 starting from the 256×256 CityFlowV2 checkpoint. Got only mAP=0.4494 (vs ~0.55+ expected).  
**Why it failed**: Aggressive training schedule (40 epochs), potential position embedding interpolation issues, and the 256→384 resolution jump may need a gentler curriculum.  
**Why revisit**: Inference-only 384×384 hurt -1.3pp (v50) because the model NEVER SAW 384px crops during training. A properly trained 384px model is a completely different experiment. Every AIC top team uses 384×384+ input. **This is the single highest-ROI revisit.**  
**Specific fixes needed**:
- Longer training (80-120 epochs with cosine schedule)
- Bicubic position embedding interpolation (verify implementation)
- Start from CLIP pretrained weights at 384, not from 256-trained CityFlowV2 checkpoint
- Validate on VeRi-776 first before full pipeline test
- **Files**: `notebooks/kaggle/09b_vehicle_reid_384px/`

### B2. Knowledge Distillation (09c) — 22% mAP → ABANDONED
**What happened**: ViT-L/14-CLIP teacher → ViT-B/16 student. Teacher head warmup + KD loss. Got only 22% mAP.  
**Why it failed**: CLS token feature dimension mismatch between ViT-L (1024D) and ViT-B (768D) projector, improper KD temperature scaling, and potential normalization issues in cosine alignment loss.  
**Why revisit**: Every AIC 2024 top-3 team used KD. The technique is proven for +2-4% mAP. The failure was in implementation, not concept.  
**Specific fixes needed**:
- Fix projector: `nn.Linear(1024, 768)` with proper initialization
- Use temperature T=2 (not T=4 — too soft for fine-grained ReID)
- Add MSE feature alignment alongside KL loss
- Warm up student from the existing strong 256px CityFlowV2 checkpoint (not from scratch)
- **Files**: `notebooks/kaggle/09c_kd_vitl_teacher/`

### B3. Multi-Scale TTA — "Marginal/Harmful" + Timeout
**What happened**: v33 tested multi-scale but was "marginal/harmful locally" and caused a 14h Kaggle timeout (disabled in v35b). v51 retried with [224,288] but logs were lost.  
**Why revisit**: Multi-scale TTA is standard in all SOTA ReID systems. The Kaggle timeout suggests the implementation was too slow (processing ALL scales for ALL crops). A selective approach (only top-N crops, only 2 scales) might work within time budget.  
**Specific fixes needed**:
- Use only 2 scales: [256, 320] (not extreme ranges)
- Apply only to the top-16 crops per tracklet (not all 48)
- Profile Kaggle runtime to ensure <3h total
- **Files**: `src/stage2_features/reid_model.py` (multiscale_sizes config), `src/stage2_features/pipeline.py`

### B4. Score-Level Ensemble (TransReID + OSNet)
**What happened**: Feature-level concat (v26) HURT -1.6pp. Score-level fusion at 10% weight was tested and was slightly positive (+0.1pp in v52-53). But never tested properly at scale with the current v80 pipeline.  
**Why revisit**: The v26 concat failure is expected (different dimensionalities create misaligned spaces). Score-level fusion is fundamentally different — it lets each model vote independently. AIC21 1st place used a 3-model ensemble.  
**Specific fixes needed**:
- Vary fusion weight: sweep 0.05–0.30 for OSNet contribution
- Consider a third model (ResNet50-IBN with BoT recipe — training code already exists in `src/training/`)
- Test with current v80 features (v26 used weaker pipeline)
- **Files**: `src/stage2_features/pipeline.py` (vehicle_reid2), `src/stage4_association/pipeline.py` (fusion logic)

### B5. Camera-Specific TTA (CamTTA) — Implemented, NEVER Enabled
**What happened**: Full CamTTA infrastructure exists in `src/stage2_features/pipeline.py` (save_bn_state, restore_bn_state, warmup_camera_bn) and `src/stage2_features/reid_model.py`. It was NEVER enabled on the Kaggle pipeline.  
**Why it's free**: AIC24 1st and 3rd place teams used test-time BN adaptation. The code is already written and working. Just needs `stage2.camera_tta.enabled: true` in the 10a notebook overrides.  
**Risk**: Very low. Worst case it's neutral. The BN state is restored after each camera.  
**Files**: `src/stage2_features/pipeline.py` L270-320, `src/stage2_features/reid_model.py` L414-450

---

## C. What Has NEVER Been Tried — The Real Opportunity Space

### C1. Properly Trained 384×384 ReID Model
| Field | Details |
|-------|---------|
| **Technique** | Train TransReID ViT-B/16 natively at 384×384 input |
| **Stage** | Stage 2 (feature extraction) |
| **What changes** | `src/stage2_features/transreid_model.py` (img_size param), position embedding interpolation from CLIP 224→384. New Kaggle training notebook (09b rewrite). |
| **Expected impact** | +1.0–2.0pp IDF1. Higher resolution captures fine-grained details (license plates, trim, damage marks) that 256×256 blurs away. Every AIC top team uses ≥384px. |
| **Difficulty** | Medium — requires rewriting 09b with proper curriculum (warm up at 256, then fine-tune at 384) |
| **Risk** | Medium — 09b's failure proves it's non-trivial, but the technique is well-established |
| **Why never properly tried** | 09b used wrong approach (aggressive schedule, checkpoint-based rather than curriculum) |

### C2. Fixed Knowledge Distillation (ViT-L → ViT-B)
| Field | Details |
|-------|---------|
| **Technique** | ViT-L/14-CLIP frozen teacher → ViT-B/16 student with proper KD |
| **Stage** | Training pipeline (new checkpoint for stage 2) |
| **What changes** | `notebooks/kaggle/09c_kd_vitl_teacher/` — rewrite with fixed projector, proper temperature, MSE+KL loss |
| **Expected impact** | +1.0–2.0pp IDF1. ViT-L encodes 5× richer features. Distilling this into ViT-B preserves runtime while boosting quality. |
| **Difficulty** | Medium — 09c exists as a starting point, needs specific fixes |
| **Risk** | Medium — 09c failed, but the failures are diagnosable and fixable |
| **Why never properly tried** | 09c had architecture mismatch bugs; was abandoned after 22% mAP |

### C3. Part-Aware Transformer (PAT) Pooling
| Field | Details |
|-------|---------|
| **Technique** | Replace CLS+GeM pooling with learnable part tokens + cross-attention |
| **Stage** | Stage 2 — `src/stage2_features/transreid_model.py` forward() method |
| **What changes** | Add N learnable part query tokens (e.g., 6: front, rear, left, right, roof, underbody). Cross-attend these against ViT patch tokens. Concat part features → project → BNNeck. |
| **Expected impact** | +0.5–1.0pp IDF1. Vehicles have strong part structure. A car's license plate region is highly discriminative but gets averaged away in global pooling. |
| **Difficulty** | Medium — requires modifying TransReID architecture and retraining |
| **Risk** | Medium — proven for person ReID (PAT paper: arXiv 2307.02797), less tested for vehicles |
| **Why never tried** | Architecture change requiring retraining. Current pooling (CLS+GeM concat) is simpler. |

### C4. SAM2 Foreground Masking
| Field | Details |
|-------|---------|
| **Technique** | Run SAM2 on detection crops to segment vehicle foreground, zero out background before ReID |
| **Stage** | Stage 2 — between crop extraction and ReID inference |
| **What changes** | New preprocessing step in `src/stage2_features/pipeline.py` after `crop_extractor.extract_crops`. Add SAM2 inference, apply mask to crop. |
| **Expected impact** | +0.3–0.5pp IDF1. Removes background clutter (road markings, adjacent vehicles, trees) that pollute ReID features. AIC24 2nd place team used this. |
| **Difficulty** | Medium — SAM2 is a separate model requiring GPU inference |
| **Risk** | Low-Medium — AIC24 2nd place validated this approach. Kaggle T4 can handle it but adds ~30min to stage2 runtime. |
| **Why never tried** | Additional model complexity and GPU time on Kaggle |

### C5. Temporal Attention for Tracklet Embedding
| Field | Details |
|-------|---------|
| **Technique** | Replace quality-weighted crop averaging with a lightweight temporal transformer (2-4 layers) that models appearance change across a tracklet's lifetime |
| **Stage** | Stage 2 — `src/stage2_features/reid_model.py` `get_tracklet_embedding_from_scored_crops()` |
| **What changes** | Instead of `sum(quality_i × embedding_i) / sum(quality_i)`, feed the sequence of (embedding_i, quality_i, frame_timestamp_i) into a small transformer that outputs a single aggregated embedding. |
| **Expected impact** | +0.5–1.5pp IDF1. A vehicle seen head-on in frame 1 and from behind in frame 50 needs temporal modeling to bridge these views. The current weighted average pools appearance but loses temporal dynamics. |
| **Difficulty** | High — requires training a temporal aggregation module on CityFlowV2 tracklets |
| **Risk** | Medium — temporal pooling is proven in video-based ReID (MARS benchmark), but requires sufficient training data |
| **Why never tried** | Architecture change requiring training a new aggregation model |

### C6. ResNet50-IBN Trained on CityFlowV2 (Second Ensemble Member)
| Field | Details |
|-------|---------|
| **Technique** | Train ResNet50-IBN-a with full BoT recipe on CityFlowV2, use as ensemble member |
| **Stage** | Training → Stage 2 → Stage 4 (score-level fusion) |
| **What changes** | Use existing `src/training/train_reid.py` + `src/training/model.py` (ReIDModelBoT). Train on CityFlowV2 crops. Save as secondary model. Score-level fusion in stage4. |
| **Expected impact** | +0.5–1.0pp IDF1. CNN (ResNet) vs ViT capture fundamentally different features — local texture vs global context. Ensemble diversity is the key, not raw model quality. |
| **Difficulty** | Low-Medium — training code ALREADY EXISTS, just needs CityFlowV2 dataset configuration |
| **Risk** | Low — BoT recipe is well-established. v26's failure was concat-based; score-level is different. |
| **Why never tried** | Never trained ResNet50-IBN on CityFlowV2. Only ran OSNet (weaker). Training infra exists but wasn't connected to CityFlowV2 dataset. |

### C7. GNN Edge Classification (LMGP-Style)
| Field | Details |
|-------|---------|
| **Technique** | Train a Graph Neural Network to predict same/different-ID edges, replacing hand-tuned threshold+graph |
| **Stage** | Stage 4 — replace `src/stage4_association/graph_solver.py` |
| **What changes** | New module: GNN takes node features (appearance embedding, camera, timestamps) and edge features (similarity, ST score) to predict edge probability. Train on CityFlowV2 GT associations. |
| **Expected impact** | +1.0–3.0pp IDF1. The GNN learns camera-pair-specific biases, non-linear appearance interactions, and transition patterns that no fixed threshold can capture. LMGP paper (arXiv 2104.09018) shows +3-5% over hand-crafted. |
| **Difficulty** | High — requires PyTorch Geometric, training pipeline, GT label generation |
| **Risk** | Medium — proven in literature but complex to implement and tune |
| **Why never tried** | Paradigm shift from hand-tuned to learned association. Significant engineering effort. |

### C8. Network Flow / Hungarian Assignment
| Field | Details |
|-------|---------|
| **Technique** | Formulate cross-camera association as a minimum-cost network flow problem or bipartite matching |
| **Stage** | Stage 4 — alternative to graph connected components |
| **What changes** | Replace `graph_solver.py` clustering with global optimal assignment. Each camera pair's tracklets form a bipartite graph; min-cost flow finds the globally optimal assignment respecting one-to-one constraints. |
| **Expected impact** | +0.3–1.0pp IDF1. Avoids greedy/transitive chain errors. Hungarian ensures globally optimal matching within each camera pair. |
| **Difficulty** | Medium — scipy.optimize.linear_sum_assignment handles the core algorithm |
| **Risk** | Low — well-understood algorithm. May not handle multi-camera transitivity as well as CC. |
| **Why never tried** | Current conflict_free_cc works well; Hungarian is typically used for bipartite (2-camera) not multi-camera |

### C9. Timestamp Bias Correction
| Field | Details |
|-------|---------|
| **Technique** | Learn per-camera-pair timestamp offsets to correct CityFlowV2 sync errors |
| **Stage** | Stage 4 — `src/stage4_association/spatial_temporal.py` or pre-processing |
| **What changes** | Grid search timestamp offset per camera pair (±5s in 0.5s steps) that maximizes ST agreement on confident matches. Apply as correction before transition scoring. |
| **Expected impact** | +0.3–0.5pp IDF1. Even 1-2 second sync errors can shift ST scores for fast-moving vehicles, causing incorrect filtering. |
| **Difficulty** | Low — simple grid search, no training required |
| **Risk** | Low — does nothing if cameras are already synced |
| **Why never tried** | Assumed cameras were synced. Never measured actual sync accuracy. |

### C10. Refined Entry/Exit Zone Annotations
| Field | Details |
|-------|---------|
| **Technique** | Hand-annotate tight entry/exit zone polygons per camera, replacing auto-clustered k-means zones |
| **Stage** | Stage 4 — `src/stage4_association/zone_scoring.py`, `configs/datasets/cityflowv2_zones.json` |
| **What changes** | Create precise polygons where vehicles enter/exit each camera's FOV. Use these to constrain valid transitions and add zone-pair transition time priors. |
| **Expected impact** | +0.5–1.5pp IDF1. AIC21 1st place relied heavily on hand-tuned zone polygons. Our auto-zones are too coarse, causing the zone model to HURT (-0.4pp in v54). |
| **Difficulty** | Medium — requires manual annotation work (6 cameras × 2-4 zones each) |
| **Risk** | Low — if zones are accurate, this strictly improves ST constraints |
| **Why never tried** | Zone model was tested with auto-generated zones and hurt. The conclusion was "zones hurt" but the real problem may be "bad zones hurt." |

### C11. AFLink / Global Tracklet Linking
| Field | Details |
|-------|---------|
| **Technique** | Appearance-Free Link (AFLink) from StrongSORT — a lightweight MLP that links tracklet fragments using purely motion/temporal cues |
| **Stage** | Stage 4 post-processing (after initial clustering) |
| **What changes** | New module that takes unmatched tracklets and tries to link them based on spatio-temporal trajectory patterns (entry/exit points, velocity, direction) without appearance features. |
| **Expected impact** | +0.3–0.5pp IDF1. Addresses the 87 fragmented GT IDs where appearance alone fails (viewpoint change, occlusion). |
| **Difficulty** | Medium — requires training the link model on GT data |
| **Risk** | Medium — designed for single-camera, untested for cross-camera |
| **Why never tried** | StrongSORT's AFLink was designed for single-camera. Cross-camera adaptation is novel. |

### C12. Circle Loss Training
| Field | Details |
|-------|---------|
| **Technique** | Replace or augment triplet loss with circle loss during ReID training |
| **Stage** | Training pipeline — `src/training/losses.py` (CircleLoss class already exists!) |
| **What changes** | Modify training notebook to use circle loss instead of or alongside triplet loss. The code already exists in `src/training/losses.py`. |
| **Expected impact** | +0.3–0.5pp mAP. Circle loss has better convergence properties for fine-grained discrimination (Sun et al., CVPR 2020). |
| **Difficulty** | Low — loss function already implemented, just needs to be enabled in training config |
| **Risk** | Low — well-established technique, worst case it's neutral |
| **Why never tried** | Current CityFlowV2 training (08/09 notebooks) uses ID+triplet only. Circle loss code exists but was never plugged into CityFlowV2 training. |

### C13. CLAHE-Tuned Per-Camera Preprocessing
| Field | Details |
|-------|---------|
| **Technique** | Per-camera CLAHE parameters tuned to each camera's exposure characteristics |
| **Stage** | Stage 0 — `src/stage0_ingestion/pipeline.py` |
| **What changes** | Different `clahe_clip_limit` values per camera. Bright cameras (S01) need less enhancement; dark cameras (S02_c006, the worst performer at IDF1=74%) need more. |
| **Expected impact** | +0.2–0.5pp IDF1. S02_c006 is catastrophic (IDF1=74%, FP ratio 6.86×). Better preprocessing could improve its ReID features specifically. |
| **Difficulty** | Low — config-only change, no code needed |
| **Risk** | Low — per-camera preprocessing is a common technique |
| **Why never tried** | Global CLAHE clip_limit=2.5 applied uniformly. Never analyzed per-camera exposure profiles. |

### C14. Stronger Detector (YOLOv11 / RT-DETR)
| Field | Details |
|-------|---------|
| **Technique** | Replace YOLO26m with a stronger detector (YOLOv11m or RT-DETR) fine-tuned on CityFlowV2 |
| **Stage** | Stage 1 — `src/stage1_tracking/tracker.py` |
| **What changes** | New detector model, potentially new detection format handling |
| **Expected impact** | +0.2–0.5pp IDF1. Detection is NOT the main bottleneck (0 unmatched GT IDs), but better detections → better crops → better ReID features. More precise bounding boxes reduce background ratio in crops. |
| **Difficulty** | Low — ultralytics supports multiple models with same API |
| **Risk** | Low — detection quality improvement is monotonic |
| **Why never tried** | Assumed detection was "good enough" since GT recall is 100%. But tighter boxes improve crop quality. |

### C15. Viewpoint-Aware Feature Disentanglement
| Field | Details |
|-------|---------|
| **Technique** | Disentangle viewpoint-invariant identity features from viewpoint-specific appearance features |
| **Stage** | Stage 2 — modify TransReID architecture |
| **What changes** | Add a viewpoint classifier branch that forces the main feature branch to be viewpoint-invariant via gradient reversal layer (GRL). |
| **Expected impact** | +0.5–1.0pp IDF1. The 87 fragmented IDs are largely caused by viewpoint changes across cameras. A front-view car looks nothing like its rear view to a naive model. |
| **Difficulty** | High — requires architecture change + retraining with viewpoint labels |
| **Risk** | Medium — requires viewpoint annotations or pseudo-labels |
| **Why never tried** | Would need viewpoint labels for CityFlowV2 (could be auto-generated from camera geometry) |

---

## D. Prioritized Roadmap

### Priority Scoring
Each approach is scored on:
- **Expected gain** (1-5): 1=+0.2pp, 2=+0.5pp, 3=+1.0pp, 4=+1.5pp, 5=+2.0+pp
- **Effort** (1-5): 1=config change, 2=<1 day, 3=1-3 days, 4=1-2 weeks, 5=2+ weeks
- **Confidence** (1-5): 1=speculative, 2=plausible, 3=literature support, 4=AIC teams used it, 5=code exists ready to test
- **Priority** = (Expected gain × Confidence) / Effort

| Rank | Approach | Expected Gain | Effort | Confidence | Priority Score | Cumulative Est. |
|------|----------|:---:|:---:|:---:|:---:|:---:|
| **1** | **B5: Enable CamTTA** | 2 (+0.5pp) | 1 | 5 (code exists) | **10.0** | 78.9% |
| **2** | **C9: Timestamp bias correction** | 2 (+0.5pp) | 2 | 3 | **3.0** | 79.4% |
| **3** | **C12: Circle loss training** | 2 (+0.5pp) | 2 | 3 | **3.0** | 79.9% |
| **4** | **B1: Redo 384×384 training** | 5 (+2.0pp) | 4 | 4 | **5.0** | 81.9% |
| **5** | **C6: ResNet50-IBN ensemble** | 3 (+1.0pp) | 3 | 4 | **4.0** | 82.9% |
| **6** | **B2: Fix Knowledge Distillation** | 5 (+2.0pp) | 4 | 4 | **5.0** | 84.9% |
| **7** | **C10: Hand-annotated zones** | 3 (+1.0pp) | 3 | 4 | **4.0** | 85.9% |
| **8** | **B3: Multi-scale TTA (fixed)** | 2 (+0.5pp) | 2 | 3 | **3.0** | 86.4% |
| **9** | **C4: SAM2 foreground masking** | 2 (+0.5pp) | 3 | 4 | **2.7** | 86.9% |
| **10** | **C3: Part-aware pooling** | 3 (+1.0pp) | 4 | 3 | **2.3** | 87.9% |
| **11** | **C7: GNN edge classification** | 5 (+2.0pp) | 5 | 3 | **3.0** | 89.9% |
| **12** | **C5: Temporal attention** | 4 (+1.5pp) | 5 | 3 | **2.4** | 91.4% |
| **13** | **C13: Per-camera CLAHE** | 1 (+0.2pp) | 1 | 2 | **2.0** | 91.6% |
| **14** | **C8: Network flow assignment** | 2 (+0.5pp) | 3 | 2 | **1.3** | 92.1% |
| **15** | **C14: Stronger detector** | 1 (+0.2pp) | 2 | 2 | **1.0** | 92.3% |

> **Note**: Cumulative estimates assume partial additivity — real gains won't stack linearly. Conservative realistic estimate: **~85-87% IDF1** from approaches 1-7; **~88-90% IDF1** from all approaches.

### Phase 1: FREE / Near-Free Gains (Days, no training) — Target: 79.5%

| # | Action | Config/Code Change | Est. Gain |
|---|--------|-------------------|-----------|
| 1 | **Enable CamTTA** | `stage2.camera_tta.enabled: true` in 10a notebook | +0.5pp |
| 2 | **Timestamp bias correction** | New script: grid-search per camera-pair offsets on confident matches | +0.3–0.5pp |
| 3 | **Per-camera CLAHE tuning** | Analyze S02_c006 exposure; set camera-specific `clahe_clip_limit` | +0.1–0.3pp |

### Phase 2: Quick Retraining (1-2 weeks) — Target: 82-83%
| # | Action | Details | Est. Gain |
|---|--------|---------|-----------|
| 4 | **Circle loss CityFlowV2 training** | Use existing `src/training/losses.py` CircleLoss in 09 notebook | +0.3–0.5pp |
| 5 | **ResNet50-IBN on CityFlowV2** | Use existing `src/training/train_reid.py`, new Kaggle notebook | +0.5–1.0pp |
| 6 | **384×384 native training (redo 09b)** | Curriculum: 256→384, longer schedule, CLIP initialization | +1.0–2.0pp |
| 7 | **Multi-scale TTA (2 scales, top-16 crops)** | Profile for Kaggle runtime, test [256, 320] | +0.3–0.5pp |

### Phase 3: Architecture Improvements (2-4 weeks) — Target: 85-87%
| # | Action | Details | Est. Gain |
|---|--------|---------|-----------|
| 8 | **Fixed KD (ViT-L → ViT-B)** | Fix 09c projector + temperature + init from strong student | +1.0–2.0pp |
| 9 | **Hand-annotated zone polygons** | Annotate 6 cameras × 2-4 zones; enable zone_model | +0.5–1.5pp |
| 10 | **SAM2 foreground masking** | Pre-segment vehicle crops before ReID | +0.3–0.5pp |

### Phase 4: Paradigm Shift (4-8 weeks) — Target: 88-90%
| # | Action | Details | Est. Gain |
|---|--------|---------|-----------|
| 11 | **GNN edge classification** | PyTorch Geometric, train on CityFlowV2 GT | +1.0–3.0pp |
| 12 | **Temporal attention aggregation** | Small transformer for tracklet embedding | +0.5–1.5pp |
| 13 | **Part-aware transformer pooling** | Cross-attention part tokens in TransReID | +0.5–1.0pp |

---

## E. Key Insight

### The Literature Is Clear: Features > Association

The consistent finding across MTMC tracking literature (AIC21-24, LMGP, SUSHI) is:

1. **~70% of MTMC performance comes from ReID feature quality.** The AIC21 1st place team (84.1% IDF1) used a 3-model ensemble with 384×384+ input and extensive domain-specific fine-tuning.

2. **~20% comes from spatio-temporal constraints.** Zone-based transition models, timestamp correction, and camera topology are multiplicative with good features.

3. **~10% comes from association algorithm choice.** The difference between connected components and GNN-based matching is only ~2-3pp when features are strong.

### Where Does Our 5.7pp Gap Come From?

| Gap Source | Est. Contribution | Evidence |
|------------|:---:|---------|
| **ReID model quality** (single model, 256px, no KD) | **3.0–4.0pp** | 09b/09c failures, no ensemble, 256px vs 384px+ SOTA |
| **Spatio-temporal modeling** (auto-zones hurt, no timestamp correction) | **1.0–1.5pp** | Zone model hurts because auto-zones are coarse; never corrected sync |
| **Association algorithm ceiling** | **0.5–1.0pp** | 220+ configs prove diminishing returns; GNN could add ~1pp |

### The Path to Beating SOTA by a Big Margin

To beat 84.1% by a **big margin** (target 88%+), we need breakthroughs in **multiple dimensions simultaneously**:

1. **Feature quality revolution** (384px + KD + ensemble): +3-4pp → 82-83%
2. **Spatio-temporal precision** (zones + timestamp): +1-2pp → 84-85%  
3. **Learned association** (GNN edges): +1-2pp → 86-87%
4. **Advanced techniques** (SAM2 + temporal attention + PAT): +1-2pp → 88-89%

The 220+ exhausted configs **prove** that threshold tuning is a dead end. Every future improvement must come from **better representations** or **learned components**. The error profile (87 fragmented vs 35 conflated) specifically points to **under-merging caused by feature dissimilarity across viewpoints** — which is exactly what 384px training, KD, and viewpoint-invariant features would address.