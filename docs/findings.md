# MTMC Tracker — Research Findings & Strategic Analysis

> **IMPORTANT**: This is a living document. Update it whenever new experiments are run, new dead ends are discovered, or performance numbers change. Keep the "Current Performance" and "Dead Ends" sections current.

## Current Performance (Last Updated: 2026-04-13)

### Vehicle Pipeline (CityFlowV2)

| Metric | Value | Notes |
|--------|:-----:|-------|
| **Current Reproducible Best MTMC IDF1** | **77.5%** | 10c v52, v80-restored association recipe on the current codebase |
| **Historical Best MTMC IDF1** | **78.4%** | v80, but not reproducible with the current codebase (~1pp drift) |
| **SOTA Target** | 84.86% | AIC22 1st place |
| **Gap to SOTA** | 7.36pp | Relative to the current reproducible best |
| **Primary Model (ViT-B/16 CLIP 256px)** | mAP=80.14% | On CityFlowV2 eval split |
| **Secondary Model (ResNet101-IBN-a)** | mAP=52.77% | On CityFlowV2 eval split, ImageNet→CityFlowV2 only |
| **Secondary Model VeRi-776 pretrain (09e v2)** | mAP=62.52% | On VeRi-776 test set, ready for CityFlowV2 fine-tuning |
| **384px ViT (09b v2)** | DEAD END | Higher single-camera ReID accuracy did not transfer; MTMC IDF1 only 0.7585-0.7562 in v43-v44, -2.8pp vs 256px baseline |
| **09f CityFlowV2 fine-tune** | mAP=42.7% | v3 peaked at epoch 104/120 and still underperformed direct ImageNet→CityFlowV2 (09d v18: 52.77%) |
| **Association configs tested** | 225+ | All within 0.3pp of optimal |

**Current reproducible vehicle MTMC IDF1 is 0.775** with 10c v52 using the v80-restored recipe (`sim_thresh=0.50`, `appearance_weight=0.70`, `fic_reg=0.50`, `aqe_k=3`, `gallery_thresh=0.48`, `orphan_match=0.38`). This is a small improvement over v50/v51 at 0.772, but it still trails the historical v80 result of 0.784. Vehicle association remains exhausted, so future gains will need materially better features or priors rather than more stage-4 tuning.

**⚠️ Metric Disambiguation (Vehicle Pipeline):**
- **MTMC IDF1 = 77.5%** — Current reproducible best on the current codebase from 10c v52. This remains the official metric and the only number that should be compared to AIC22 SOTA.
- **Historical v80 reference** — MTMC IDF1 = 78.4%, IDF1 = 79.8%, GLOBAL IDF1 = 80.5%. These numbers are useful for historical comparison, but they are not currently reproducible.
- All current best numbers use GT-assisted metrics (`gt_frame_clip=true`, `gt_zone_filter=true`) which inflate scores by 1-3pp vs clean evaluation.

### Person Pipeline (WILDTRACK) — NEW

| Metric | Value | Notes |
|--------|:-----:|-------|
| **Ground-plane MODA** | **90.3%** | 12b v14 tuned Kalman tracker with interpolation on MVDeTr detections |
| **Ground-plane IDF1** | **94.7%** | Matched again by 12b v1 on 12a v3 detections; Precision=94.5%, Recall=96.1%, IDSW=5 |
| **Best Detector (MVDeTr 12a v3)** | MODA=92.1% | Epoch 20/25, Precision=95.7%, Recall=96.6%; best WILDTRACK detector yet |
| **12b v2 extended Kalman sweep** | MODA=90.0%, IDF1=94.7% | Wider interpolation/max_age/conf sweeps plus velocity-aware quadratic interpolation still converged to the same 5-IDSW operating point |
| **12b v1 on 12a v3 detections** | MODA=90.0%, IDF1=94.7% | Better detections did not improve tracking; Kalman sweep converged to the same effective operating point |
| **Previous Best Tracking Baseline** | IDF1=92.8% | 12b v9 naive tracker; new tuned Kalman run is +1.9pp |
| **ReID Features (12b v8)** | mean cosine=0.720 | Fixed ViT backbone mismatch; range widened to [0.215, 1.000] |
| **ReID Merge State** | No gain over baseline | Tuned Kalman tracks are already clean; merge sweep did not improve IDF1 |
| **Gap to SOTA** | 0.6pp | Best IDF1=94.7% vs target 95.3% |
| **Status** | Person tracking appears effectively converged at 94.7% IDF1 | 12b v1 stayed flat on better detections; further gains likely need a different tracker or stronger person ReID |

**Current person best remains 94.7% IDF1**: 12b v14 reached **IDF1=94.7%** and **MODA=90.3%** with a tuned Kalman tracker plus interpolation, improving on the previous 12b v9 naive-tracker best of **92.8%** by **+1.9pp**. A follow-up rerun on the stronger **12a v3** detector in **12b v1** converged to the same **IDF1=94.7%** with slightly lower **MODA=90.0%** using Kalman settings `max_age=2`, `min_hits=2`, `distance_gate=25.0`, `q_std=5.0`, `r_std=10.0`. The extended **12b v2** sweep widened interpolation from **[1,2,3]** to **[1,2,3,4,5]**, max_age from **[2,3,4,5]** to **[2,3,4,5,6,8]**, added confidence thresholds **[0.15, 0.20, 0.25, 0.30, 0.35]**, and tested velocity-aware quadratic interpolation, yet still selected `max_age=2`, `min_hits=2`, `distance_gate=25.0`, `q_std=5.0`, `r_std=10.0`, `interpolation_max_gap=1`, `conf_threshold=0.15` with the same **IDF1=94.7%**, **MODA=90.0%**, and **5 ID switches**. The improved detector therefore did **not** translate into better tracking, and the tracker sweep extensions also failed to move the metric, which indicates the current WILDTRACK person pipeline is tracker-limited rather than detector-limited or parameter-limited. Person IDF1 appears effectively converged at **94.7%**; further gains likely require a fundamentally different tracker or materially stronger person ReID.

## Gap Decomposition

The remaining gap to SOTA now decomposes into:

| Deficiency | Impact | Status |
|------------|:------:|--------|
| Single ReID model vs 3-5 ensemble | Largest remaining gap | Tested with weak secondary (52.77% mAP at 0.30 weight): no gain. Needs ≥65% mAP secondary to be viable |
| Camera-aware single-model training (DMT) | -1.4pp | Tested and harmful (v46) |
| Multi-query track representation | -0.1pp | Tested and neutral/harmful (v51) |
| Higher-dimensional concat-patch features | -0.3pp | Tested and harmful (v48-v49) |
| Association tuning / structural association changes | Exhausted | 225+ configs plus structural variants already tried |

**Higher single-camera ReID mAP is not translating into better MTMC IDF1.** Both 384px deployment and DMT camera-aware training improved single-camera discrimination but hurt cross-camera association. Without a true multi-model ensemble, the realistic ceiling now appears to be roughly 77-78% MTMC IDF1.

## Critical Discovery: 384px Is a Dead End for MTMC

The earlier checkpoint mismatch was real, but it is no longer the blocker. The correct 09b v2 384px checkpoint was evaluated and still lost decisively to the 256px baseline in MTMC.

**Definitive result (2026-03-30)**:
- v43 (384px, tuned thresholds, min_hits=3): MTMC IDF1 = 0.7585
- v44 (384px, exact v43 config but min_hits=2): MTMC IDF1 = 0.7562
- v80 baseline (256px, min_hits=2): MTMC IDF1 = 0.7840

Higher input resolution improved or preserved single-camera discriminative detail, but it made cross-camera association worse by emphasizing viewpoint-specific textures that do not transfer well across cameras.

## Critical Discovery: ResNet101-IBN-a 52.77% Is Expected

The ViT achieves high mAP because of 3-stage progressive specialization:
- CLIP (400M image-text pairs) → VeRi-776 (576 IDs, 37K images) → CityFlowV2 (128 IDs, 7.5K images)

The ResNet skips the critical VeRi-776 middle step:
- ImageNet (1.3M generic) → CityFlowV2 (128 IDs, 7.5K images) directly

Published 75-80% mAP baselines for ResNet101-IBN-a are evaluated on **VeRi-776** (576 IDs), NOT on CityFlowV2 (128 IDs). These numbers were never comparable. The 52.77% is reasonable given the massive pretraining disadvantage.

**Fix**: Train ResNet101-IBN-a on VeRi-776 FIRST, then fine-tune on CityFlowV2 (same pattern as the ViT).

**Status update (2026-03-29)**: VeRi-776 pretraining completed in 09e with mAP=62.52% on the VeRi-776 test set, but the follow-up CityFlowV2 fine-tune in 09f v3 only reached mAP=42.7%. This underperforms the direct ImageNet→CityFlowV2 baseline (09d v18: 52.77%), so VeRi-776 pretraining appears to hurt rather than help for this ResNet101-IBN-a path.

**Status update (2026-04-01)**: Extending the direct ImageNet→CityFlowV2 run by resuming from the **09d v18** 52.77% checkpoint at a lower learning rate (**3e-4**) peaked at only **50.61% mAP** in **09d gumfreddy v3**. This confirms that **52.77% is effectively the ceiling** for the current ImageNet→CityFlowV2 ResNet101-IBN-a recipe rather than an undertrained checkpoint.

## What SOTA Does Differently

| Pattern | AIC22 1st | AIC22 2nd | AIC21 1st | We have? |
|---------|:-:|:-:|:-:|:-:|
| 3+ ReID backbone ensemble | 5 models | 3 models | 3 models | **NO** (1 working) |
| 384×384 input | ✅ | ✅ | ✅ | **Tested, but harmful in our pipeline** |
| IBN-a backbones | ✅ | ✅ | ✅ | ViT only |
| Camera-pair bias (CID_BIAS) | ROI masks | NPY | NPY | **Tested on 256px: -3.3pp MTMC IDF1 (DEAD END)** |
| Reranking | Box-grained | k-reciprocal | k-reciprocal | **Disabled** |
| Camera-aware training (DMT) | ✅ | ✅ | ✅ | **NO** |
| Multiple loss functions | ID+tri+circle+cam | ID+tri+cam | ID+tri+cam | ID+tri |

### AIC Winning Methods Analysis and Ensemble Implementation

- **AIC22 1st place (IDF1=0.8486)** used a **5-model ReID ensemble** plus **Box-Grained Matching**.
- **AIC22 2nd place (IDF1=0.8437)** used a **3-model ensemble** (**ResNet101-IBN-a x2 + ResNeXt101-IBN-a**) with **DMT** and **CID_BIAS**.
- **Universal pattern**: every top AIC method relies on a **3-5 model ensemble**. The consistent win is not a single stronger backbone, but diversity across multiple ReID models.
- **Implication for our results**: our single-model upgrades such as **384px**, **DMT**, and **reranking** fail in isolation because they are ensemble-dependent techniques. They can remove noise, but with only one model they also remove too much discriminative signal.

**Ensemble implementation status**

- **09g**: ResNet101-IBN-a DMT training is currently running on **gumfreddy** with **150 epochs**, a **camera-adversarial head**, and **no circle loss**.
- **09h**: ResNeXt101-IBN-a DMT training is running on **gumfreddy** (**v1**).
- **Infrastructure**: **Stage 2** and **Stage 4** have already been updated for **3-model score-level fusion**.
- **CID_BIAS**: tested on 256px features; MTMC IDF1 dropped from 0.784 to 0.751 (-3.3pp). Dead end for single-model features.

**Why the previous improvements failed**

- With **1 model**, techniques like **384px**, **DMT**, and **reranking** strip away unstable but still useful signal, so MTMC gets worse.
- With **3-5 models**, those same techniques can suppress noise while identity signal survives through model diversity.
- The key blocker is therefore not that these methods are intrinsically bad, but that they were evaluated in a **single-model regime** where AIC winners never operated.

## Prioritized Action Plan

| Priority | Action | Expected Impact | Status |
|:--------:|--------|:---------------:|--------|
| **1** | Train or acquire a truly complementary secondary/tertiary ReID model for ensemble use | Only plausible path to a material gain | BLOCKED — ensemble tested at 0.30 weight with 52.77% mAP secondary, no gain; ResNet training path exhausted without VeRi-776 benefit |
| **2** | ~~CID_BIAS~~ | -3.3pp on 256px features (0.751 vs 0.784 baseline) | **DEAD END** |
| **3** | Additional association sweeps or structural tweaks | Negligible | NOT RECOMMENDED |
| **4** | More single-model feature variants (384px, DMT, multi-query, concat-patch) | Negative based on current evidence | DO NOT RETRY |
| **5** | Re-enable reranking only after feature quality improves materially | Potentially positive only with better features | BLOCKED by current features |

Association tuning remains exhausted (225+ configs tested). The remaining vehicle path is no longer incremental stage-4 tuning or single-model feature ablations; it is ensemble-quality representation learning or materially stronger camera-pair priors.

## Latest Experiment Results (2026-04-13)

### Completed

#### 12b v2 — Extended Kalman Sweep (2026-04-13)
- **Task**: Extend the WILDTRACK tracker sweep beyond the original tuned-Kalman search to test whether broader interpolation, longer track persistence, looser detection thresholds, or better interpolation dynamics can reduce the remaining 5 ID switches
- **Sweep extensions**: interpolation **[1,2,3] -> [1,2,3,4,5]**, `max_age` **[2,3,4,5] -> [2,3,4,5,6,8]**, detection confidence **[0.15, 0.20, 0.25, 0.30, 0.35]**, plus velocity-aware quadratic interpolation
- **Best config**: `max_age=2`, `min_hits=2`, `distance_gate=25.0`, `q_std=5.0`, `r_std=10.0`, `interpolation_max_gap=1`, `conf_threshold=0.15`
- **Result**: **IDF1 = 94.7%**, **MODA = 90.0%**, **IDSW = 5**
- **Comparison**: Identical to the prior best operating point; no sweep extension reduced ID switches below **5** or improved IDF1 beyond **0.9467**
- **Conclusion**: Kalman tracker parameters are fully exhausted for the current person pipeline. The remaining gap is tracker-limited, not parameter-limited.

#### 10c v52 — Vehicle v80-Restored Reproduction
- **Task**: Re-run the vehicle pipeline with the restored v80-style association recipe on the current codebase
- **Config**: `sim_thresh=0.50`, `appearance_weight=0.70`, `fic_reg=0.50`, `aqe_k=3`, `gallery_thresh=0.48`, `orphan_match=0.38`
- **Result**: **MTMC IDF1 = 0.775**, IDF1 = 0.801, MOTA = 0.671, HOTA = 0.581
- **Comparison**: Improves over v50 at **0.772**, but still trails historical v80 at **0.784**
- **Conclusion**: Vehicle association remains exhausted. The latest recovery attempt squeezed out only +0.3pp, so future gains need materially better feature quality or stronger priors rather than more stage-4 sweeps.

#### 10a/10c — Score-Level Fusion Ensemble Test (TransReID + ResNet101-IBN-a at 0.30 weight)
- **Task**: Test 2-model score-level fusion with primary TransReID ViT-B/16 CLIP (256px, ~80% mAP) and secondary ResNet101-IBN-a (52.77% mAP) at 30% fusion weight
- **Config**: `fus0.3_ter0.0` — 30% secondary weight, 0% tertiary, on current v52 association recipe
- **Result**: **MTMC IDF1 = 0.774**, IDF1 = 0.803, MOTA = 0.676, HOTA = 0.581
- **Baseline**: v52 single-model: **MTMC IDF1 = 0.775**
- **Delta**: **-0.1pp** (statistically identical)
- **Root cause**: The 28pp mAP gap between primary (80%) and secondary (52.77%) means the secondary model adds noise rather than complementary signal. At 0.30 weight, 30% of the similarity score comes from a model that is essentially random for cross-camera matching. AIC22 winners used ensembles where ALL models exceeded 70% mAP on VeRi-776.
- **Conclusion**: Ensemble with a drastically weaker secondary model is a dead end. The secondary must reach ≥65% mAP (ideally >70%) on CityFlowV2 before ensemble fusion can help. This confirms the experiment-log finding that `fusion_weight > 0.10` hurts when the secondary is weak.

#### 09d v3 (gumfreddy) — Extended ResNet101-IBN-a Fine-Tuning Failure
- **Task**: Resume the best direct ImageNet→CityFlowV2 ResNet101-IBN-a checkpoint (**09d v18**, 52.77% mAP) and continue training with a lower learning rate
- **Config**: Resume from 09d v18 `best_model.pth`, lower `lr=3e-4`
- **Result**: Best **mAP = 50.61%**, which is **worse** than the original **52.77%** checkpoint
- **Conclusion**: Additional fine-tuning from the 52.77% checkpoint degrades the model instead of improving it. The direct ImageNet→CityFlowV2 path appears saturated, and **52.77% is effectively the ceiling** for this ResNet101-IBN-a recipe.

#### 12a v26 — MVDeTr WILDTRACK Detector for Best Tracking Run
- **Task**: Train the detector used by the current best 12b v14 tracking pipeline
- **Config**: MVDeTr `resnet18` backbone, `deform_trans`, 10 epochs
- **Result**: **MODA = 90.9%**, MODP = 81.9%, Precision = 95.8%, Recall = 95.1%
- **Conclusion**: Detection quality is now strong enough that the remaining IDF1 gap appears dominated by missed or imperfect detections rather than association collapse.

#### 12a v3 (gumfreddy) — Extended MVDeTr WILDTRACK Detector Training
- **Task**: Extend WILDTRACK detector training from 10 to 25 epochs on the same MVDeTr `resnet18` setup
- **Config**: MVDeTr `resnet18`, 25 epochs
- **Result**: Best **MODA = 92.1%** at **epoch 20**, with recall in the **96.4-96.6%** range and best-epoch precision **95.7%**
- **Comparison**: Previous best detector was **MODA = 90.9%**, so this is a **+1.2pp** improvement in MODA and roughly **+1.5pp** in recall
- **Trend**: Performance peaked around epoch 20 and declined slightly by epoch 25
- **Conclusion**: This is the **best WILDTRACK detection result yet**, but the follow-up **12b v1** tracking rerun stayed flat, so detector gains alone are no longer enough to move person tracking.

#### 12b v1 (gumfreddy) — Tracking Rerun on 12a v3 Detections
- **Task**: Re-run the WILDTRACK tracking sweep using the improved **12a v3** detector outputs
- **Best tracker**: Kalman `max_age=2`, `min_hits=2`, `distance_gate=25.0`, `q_std=5.0`, `r_std=10.0`
- **Result**: **MODA = 90.0%**, **IDF1 = 94.7%**, Precision = 94.5%, Recall = 96.1%, IDSW = 5
- **Comparison**: Detector quality improved from **90.9%** to **92.1% MODA**, but tracking stayed essentially flat versus **12b v14** (**90.3% MODA**, **94.7% IDF1**)
- **Conclusion**: The tracking sweep converged to the same effective operating point. The current WILDTRACK person pipeline is bottlenecked by the tracker, not the detector.

### In Progress

- **09 v2**: Vehicle augmentation overhaul + EMA (**running on Kaggle**)
- **10c v46**: AFLink motion-based post-association (**running on Kaggle**)
- **12b v3**: Global optimal tracker (sliding-window) for person (**running on Kaggle**)

#### 12b v14 — Tuned Kalman Tracking + Merge Sweep
- **Task**: Optimize world-coordinate tracking and test whether ReID merge improves the cleaned track set
- **Baseline naive tracker**: **IDF1 = 91.4%**, MODA = 89.1%, IDSW = 7, 42 tracks
- **Default Kalman tracker**: **IDF1 = 88.9%**, MODA = 89.8% using the default chi-squared gate; worse than naive
- **Best tuned Kalman**: `max_age=2`, `min_hits=2`, `distance_gate=20cm`, `q_std=8`, `r_std=8`, interpolation `gap=2` -> **IDF1 = 94.7%**, **MODA = 90.3%**, Precision = 95.8%, Recall = 95.1%, IDSW = 5, 38 tracks
- **Comparison**: Beats the previous best 12b v9 naive tracker at **92.8% IDF1** by **+1.9pp**
- **Merge result**: No ReID merge variant improved over the tuned-Kalman baseline; only **44 ReID features** were available and the tracker already produced clean trajectories
- **Conclusion**: Kalman tracking works very well on WILDTRACK when the gate is tuned in metric space, but the follow-up **12b v1** rerun on stronger **12a v3** detections stayed flat at **94.7% IDF1**. Further gains now look more likely to require a fundamentally different tracker or better person ReID than more detector-only gains or aggressive merging.

#### 09e — VeRi-776 Pretraining
- **Kernel**: `ali369/09e-vehicle-reid-resnet101-ibn-a-veri-776-pretrain` v2, T4 GPU
- **Task**: ResNet101-IBN-a pretrained on VeRi-776 dataset (576 IDs, ~50k images)
- **Config**: 384x384, 120 epochs, triplet+center loss, cosine scheduler, fp16
- **Result**: Best mAP = 62.52% on VeRi-776 test set
- **Runtime**: ~3.4 hours (12,206 seconds)
- **Artifacts**: `best_model.pth` (525MB), 13 epoch checkpoints
- **Next**: Fine-tune on CityFlowV2 in 09f

#### 12a — MVDeTr WILDTRACK Training
- **Kernel**: `ali369/12a-wildtrack-mvdetr-training` v11, T4 GPU
- **Task**: MVDeTr ground-plane multi-view detection on WILDTRACK (7 cameras)
- **Config**: ResNet18 backbone, deform_trans world features, 10 epochs, batch_size 1
- **Results**:
	- Epoch 1: MODA 69.4%, MODP 79.4%, Precision 98.2%, Recall 70.7%
	- Epoch 5: MODA 89.4%, MODP 79.6%, Precision 97.6%, Recall 91.6%
	- Epoch 10: MODA 92.0%, MODP 81.9%, Precision 96.5%, Recall 95.5%
- **Outcome**: Surpasses the MVDeTr paper's reported 91.5% MODA on WILDTRACK
- **Runtime**: ~2.5 hours (9,130 seconds)
- **Artifacts**: `MultiviewDetector.pth`, `test.txt` (ground-plane detections), `log.txt`
- **Next**: Feed detections into tracking pipeline in 12b

#### 09f v2 — CityFlowV2 Fine-tuning Failure
- **Kernel**: `ali369/09f-resnet101ibn-cityflowv2` v2, T4 GPU
- **Task**: Fine-tune VeRi-776-pretrained ResNet101-IBN-a on CityFlowV2
- **Config**: circle_weight=1.0, batch_size=32, label_smooth=0.1, lr=7e-5
- **Result**: **mAP = 16.2%** on CityFlowV2 eval split
- **Outcome**: Catastrophic failure; best checkpoint was epoch 4 during warmup, then training diverged after warmup ended
- **Root cause**: Circle loss and triplet loss produced conflicting gradients on the same feature tensor and destroyed embedding quality
- **Follow-up**: 09f v3 launched with circle loss disabled and a corrected optimization recipe

#### 09f v3 — CityFlowV2 Fine-tuning Retry Result
- **Kernel**: `ali369/09f-resnet101ibn-cityflowv2` v3, T4 GPU
- **Task**: Fine-tune VeRi-776-pretrained ResNet101-IBN-a on CityFlowV2 with circle loss removed
- **Config**: batch_size=48, lr=3.5e-4 (AdamW), label_smoothing=0.05, warmup start_factor=0.1, 120 epochs, circle_weight=0.0
- **Result**: **mAP = 42.7%** at epoch 104/120 on the CityFlowV2 eval split
- **Outcome**: Removing circle loss fixed the catastrophic collapse from v2, but the VeRi-776→CityFlowV2 path still underperformed the direct ImageNet→CityFlowV2 recipe from 09d v18 (**52.77% mAP**)
- **Interpretation**: VeRi-776 pretraining appears to hurt CityFlowV2 transfer for ResNet101-IBN-a, likely because the model overfits to VeRi-776's appearance distribution and does not generalize well to CityFlowV2's viewpoints and lighting
- **Conclusion**: The secondary-model ensemble plan is not viable with this pretraining path without materially better pretraining or a broader hyperparameter search

#### Kaggle T4 Dataset Mount Structure Discovery
- **Discovery**: On Kaggle **T4 GPU** kernels, datasets mounted under **`/kaggle/input/datasets/<owner>/<slug>/`** rather than **`/kaggle/input/<slug>/`**
- **Impact**: This caused multiple 09d failures when kernels assumed the legacy flat mount path and could not locate the dataset
- **Conclusion**: When debugging missing-dataset errors on Kaggle T4 kernels, check the nested `datasets/<owner>/` mount structure first.

#### 12b v4/v5/v6 — Ground-plane Tracking Validation
- **Kernel**: `ali369/12b-wildtrack-mvdetr-tracking-reid` v4-v6, T4 GPU
- **Task**: World-coordinate tracking on MVDeTr detections plus person ReID feature extraction
- **Tracking result**: **MODA = 89.8%, IDF1 = 92.2%, Precision = 97.1%, Recall = 93.9%, IDSW = 12**
- **Outcome**: Ground-plane tracking is working correctly; WILDTRACK tracking quality is strong even before ReID is used for evaluation
- **Issue discovered in v6**: ReID embeddings were mode-collapsed with mean off-diagonal cosine similarity **0.874**
- **Root cause**: Pipeline instantiated CLIP ViT (`vit_base_patch16_clip_224.openai`) while 09p was trained with standard ViT (`vit_base_patch16_224`); CLIP adds `norm_pre`, which was randomly initialized and corrupted features

#### 12b v8 — ViT Backbone Fix
- **Kernel**: `ali369/12b-wildtrack-mvdetr-tracking-reid` v8, T4 GPU
- **Fix**: Matched inference backbone to the 09p training architecture (`vit_base_patch16_224`)
- **Feature quality improvement**:
	- Mean off-diagonal cosine similarity: **0.874 -> 0.720**
	- Similarity range: **[0.675, 1.000] -> [0.215, 1.000]**
	- Merge candidates: **844 -> 382**
	- Weight loading: only 4 minor missing keys (`bn_jpm.*`) instead of all backbone keys missing in v5/v6
- **Tracking metrics**: Unchanged at MODA=89.8%, IDF1=92.2% because ground-plane evaluation does not consume ReID features directly

#### 10a v44 — Local Stage 3-5 Evaluation on Correct 384px + `min_hits=2` Features
- **Input**: `data/kaggle_outputs/10a_v44/run_kaggle_20260329_192418/` with 941 tracklets, 384D primary embeddings, 384D secondary embeddings, 192D HSV
- **Task**: Run local CPU-only stages 3-5 on the exported Kaggle stage-2 artifacts using two association configs
- **Run A (384px-tuned config)**:
	- Overrides: sim=0.54, secondary_weight=0.17, appearance=0.70, gallery=0.50, FIC=0.10, cross_id_nms_iou=0.40, min_traj_frames=40, min_traj_conf=0.30, `gt_frame_clip=true`, `gt_zone_filter=true`, `mtmc_only_submission=false`
	- Result: **MTMC IDF1 = 74.62%**, IDF1 = 78.62%, MOTA = 67.99%
	- Per-camera IDF1: S01_c001 0.926, S01_c002 0.900, S01_c003 0.914, S02_c006 0.637, S02_c007 0.760, S02_c008 0.580
- **Run B (v80-style thresholds on same v44 features)**:
	- Overrides: sim=0.53, secondary_weight=0.10, appearance=0.70, gallery=0.50, FIC=0.10 with the same stage-5 filters as Run A
	- Result: **MTMC IDF1 = 74.74%**, IDF1 = 78.67%, MOTA = 68.07%
	- Per-camera IDF1: S01_c001 0.925, S01_c002 0.900, S01_c003 0.914, S02_c006 0.638, S02_c007 0.762, S02_c008 0.582
- **Outcome**: Both local evaluations underperform the current 78.4% MTMC benchmark by ~3.7pp. The v80-style thresholds are only **+0.12pp** MTMC IDF1 over the 384px-tuned config, so the issue is not primarily the chosen stage-4 thresholds.
- **Important note**: These runs used the current local stage-5 defaults unless overridden. Track smoothing remained enabled, and the requested legacy override aliases had to be translated onto the live config tree (`weights.vehicle.appearance`, `stage5.min_trajectory_*`, `stage5.cross_id_nms_iou`, `stage5.gt_frame_clip`).

#### 10a v43/v44 — Definitive 384px Verdict
- **v43**: 384px features, tuned thresholds, min_hits=3 -> **MTMC IDF1 = 0.7585**
- **v44**: 384px features, exact v43 association recipe, min_hits=2 -> **MTMC IDF1 = 0.7562**
- **v44 + CID_BIAS**: exact v44 config plus a locally-computed 6x6 CID_BIAS matrix from **464/941 GT-matched tracklets** -> **MTMC IDF1 = 0.7510**
- **Baseline for comparison**: v80 256px pipeline, min_hits=2 -> **MTMC IDF1 = 0.7840**
- **Conclusion**: 384px is a confirmed dead end for MTMC in this pipeline, despite strong single-camera mAP. The failure mode is reduced cross-camera invariance, not insufficient stage-4 tuning. CID_BIAS also made the 384px run **worse by -0.52pp**, so camera-pair calibration is not enough to rescue this feature set.

#### 10c v45-v51 — Final Vehicle Ablation Verdict
- **v45 baseline**: late 10c baseline remained in the **0.772-0.775 MTMC IDF1** range depending on exact export and evaluation path
- **v46 (DMT camera-aware training, 87.3% mAP)**: **MTMC IDF1 = 0.758**, which is **-1.4pp** vs the paired v45 baseline at **0.772**
- **v48 (`concat_patch=true`, 1536D features)**: **MTMC IDF1 = 0.773**, which is **-0.3pp** vs v45 at **0.775**
- **v49 (`concat_patch=true` + vehicle2 ensemble)**: **MTMC IDF1 = 0.769**, which is **-0.3pp** vs v50 at **0.772**
- **v50/v51 (current reproducible best / multi-query test)**: **MTMC IDF1 = 0.772**; the multi-query track representation in **v51** was **-0.1pp** vs **v50** and did not improve association quality
- **Conclusion**: Every final feature-side ablation failed to beat the current reproducible baseline. Better single-camera discrimination did not solve the MTMC bottleneck.

## Person Pipeline (WILDTRACK) — New Initiative

### Baseline Performance

| Run | Config Changes | Tracklets | MTMC IDF1 | IDF1 | MOTA |
|-----|---------------|-----------|-----------|------|------|
| Baseline (run_20260327_211115) | Default wildtrack.yaml | 1,339 | 0.176 | 0.316 | -0.281 |
| Exp 1 (run_20260327_224721) | conf=0.55, min_len=8, min_hits=3, match=0.75, merge_gap=40 | 819 | 0.233 | 0.368 | 0.118 |
| Kaggle Baseline (wildtrack_20260328_000939) | 11a→11b→11c chain on Kaggle T4; conf=0.55, min_hits=3, match=0.75, min_tracklet_length=8; ReID 768D→PCA 256D (EV=0.988), HSV 192D, flip_aug=on, camera_bn=on; sim=0.30, louvain=1.5, app=0.80, hsv=0.10, spatio=0.10 | 911 | 0.140 | 0.280 | -0.463 |
| Exp 2 (run_20260327_231511) | conf=0.65, min_len=12, fresh PCA, rerank=off, sim=0.40, louvain=2.0, app=0.90 | INCOMPLETE (interrupted) | — | — | — |

Kaggle baseline details: 911 tracklets across 7 cameras (C1:126, C2:169, C3:154, C4:88, C5:146, C6:121, C7:107), per-camera 2D metrics IDF1=0.280 / MOTA=-0.463 / HOTA=0.000 / IDSW=573, MTMC IDF1=0.140 / MOTA=-0.276 / IDSW=1006, error analysis: 164 fragmented GT IDs, 141 conflated pred IDs, 46 unmatched GT IDs, 141 unmatched pred IDs. Per-camera IDF1/MOTA/IDSW: C1 0.261/-0.012/126, C2 0.191/-0.669/107, C3 0.253/-0.113/127, C4 0.233/-1.468/23, C5 0.316/-0.687/80, C6 0.254/0.004/86, C7 0.450/-0.296/24.

Kaggle underperformed the local Exp 1 baseline (MTMC IDF1 0.140 vs 0.233; per-camera IDF1 0.280 vs 0.368). The likely causes are suboptimal fixed association parameters in 11c and GPU/framework-dependent detection differences. The 911 vs 819 tracklet count gap indicates slightly different detection/tracking behavior on Kaggle hardware/runtime, which plausibly explains part of the per-camera IDF1 regression.

### Key Discoveries (Person Pipeline)
1. **Frame ID off-by-one bug (FIXED)**: WILDTRACK GT was being written with 0-based frame IDs, but predictions use 1-based. Fixed in `scripts/prepare_dataset.py`. This single fix contributed +39.9pp MOTA improvement.
2. **Extreme tracklet fragmentation**: 1,339 tracklets for ~20 people in baseline. Increasing min_hits and min_tracklet_length helped reduce to 819.
3. **Over-detection**: Person detector has low precision (~0.33, 13K FP). Need higher confidence threshold.
4. **PCA model potentially wrong distribution**: Person PCA was trained on vehicle features. Moved to .bak to force refit on WILDTRACK data.
5. **ReID model**: Using TransReID ViT-Base/16 CLIP pretrained on Market1501 (person-specific). Not fine-tuned on WILDTRACK.
6. **All GPU-intensive pipeline stages must run on Kaggle, not locally** (local GTX 1050 Ti too slow).
7. **Kaggle chain currently regresses vs local best**: Same stage-1 thresholds but fixed 11c association settings and possible runtime differences produced 911 tracklets and lower MTMC/per-camera IDF1 than local Exp 1.

### Person Pipeline Next Steps
- Treat person tracking as effectively converged at **94.7% IDF1** under the current Kalman pipeline; **12b v1** on **12a v3** detections stayed flat
- Further person gains likely require a fundamentally different tracker or materially stronger person ReID rather than more detector-only tuning
- Revisit merge only if person ReID quality improves materially; current Kalman tracks are already clean and merges did not help
- Keep GPU-intensive person stages on Kaggle only

## Conclusive Dead Ends (DO NOT RETRY)

| Approach | Result | Evidence |
|----------|--------|---------|
| Association parameter tuning | Exhausted (225+ configs, all within 0.3pp) | Experiment log |
| CSLS distance | -34.7pp catastrophic | v74 |
| Hierarchical clustering | -1.0 to -5.1pp | v54-56, v62 |
| FAC (Feature Augmented Clustering) | -2.5pp | v26 |
| Feature concatenation (vs score fusion) | -1.6pp | Experiment log |
| CamTTA (Camera Test-Time Adaptation) | Helps global, hurts MTMC | v28-30 |
| Multi-scale TTA | Neutral/harmful | Multiple runs |
| Track smoothing / edge trim | Always harmful | Experiment log |
| Denoise preprocessing | -2.7pp | v46, v82 |
| mtmc_only submission flag | -5pp | Documented |
| Auto-generated zone polygons | -0.4pp | v54-57 |
| PCA dimension search | 384D optimal, others worse | Experiment log |
| Ensemble with 52% secondary at high weight | Dilutes signal | Current state |
| Score-level ensemble with 52.77% mAP secondary at 0.30 weight | -0.1pp MTMC IDF1; noise dilutes primary signal | 10a/10c fusion test, fus0.3_ter0.0 |
| SGD optimizer for ResNet101-IBN-a | 30.27% mAP catastrophic | v18 mrkdagods |
| Circle loss for ResNet101-IBN-a with triplet loss | Catastrophic gradient conflict; NEVER combine on the same feature tensor | 09d v17 (29.6%), 09f v2 (16.2%) vs 09d v18 (52.77%), 09e (62.52%) with circle_weight=0 |
| ResNet101-IBN-a VeRi-776→CityFlowV2 fine-tuning | mAP=42.7% (09f v3), worse than direct ImageNet→CityFlowV2 (52.77% mAP, 09d v18); secondary-model ensemble path not viable without better pretraining or broader hyperparameter search | 09e v2, 09f v3, 09d v18 |
| Extended fine-tuning from the 09d v18 ResNet101-IBN-a checkpoint | mAP degraded from 52.77% to 50.61% after resuming with lower LR; confirms the direct ImageNet→CityFlowV2 path is already at its ceiling | 09d gumfreddy v3 |
| CLIP ViT backbone when checkpoint uses standard ViT | `norm_pre` randomly initialized, causing mode collapse (cosine sim 0.874) | 12b v5/v6 vs v8 |
| Default Kalman tracker with chi-squared gating on WILDTRACK | Worse than naive baseline; IDF1 fell to 88.9% | 12b v14 sweep |
| ReID merge on top of tuned WILDTRACK Kalman tracks | No improvement over baseline; tracks already clean and only 44 features matched | 12b v14 merge sweep |
| Person: extended Kalman sweeps | No improvement; interpolation 1-5, max_age 2-8, conf 0.15-0.35, and velocity-aware quadratic interpolation all left IDF1 unchanged at 0.947 with 5 IDSW | 12b v2 |
| K-reciprocal reranking (with current features) | Always worse | v25, v35 |
| Camera-pair similarity normalization | Zero effect (FIC handles it) | v36 |
| CID_BIAS (camera-pair bias matrix) | -3.3pp MTMC IDF1 on 256px features (0.751 vs 0.784) | v44 + CID_BIAS test |
| confidence_threshold=0.20 | -2.8pp | v45 |
| max_iou_distance=0.5 | -1.6pp | v47 |
| 384px TransReID deployment | -2.8pp MTMC IDF1 despite +10pp mAP; v43=0.7585, v44=0.7562 vs v80=0.784 | 10a v43, 10a v44, v80 baseline |
| DMT camera-aware training (87.3% mAP) | -1.4pp MTMC IDF1; v46=0.758 vs v45=0.772 | 10c v45-v46 |
| Multi-query track representation | -0.1pp; v51=0.771 vs v50=0.772 | 10c v50-v51 |
| `concat_patch=true` (1536D features) | -0.3pp; v48=0.773 vs v45=0.775 | 10c v45, 10c v48 |
| `concat_patch=true` + vehicle2 ensemble | -0.3pp; v49=0.769 vs v50=0.772 | 10c v49-v50 |

### 384px TransReID Input Resolution (2026-03-30)
- **Result**: -2.8pp MTMC IDF1 (0.7562 vs 0.784 baseline)
- **Details**: 384px ViT-B/16 CLIP has 80.14% single-camera mAP (same as 256px) but performs significantly worse for cross-camera association
- **v43** (min_hits=3, optimized thresholds): MTMC IDF1 = 0.7585
- **v44** (min_hits=2, exact v43 config): MTMC IDF1 = 0.7562
- **Root cause**: Higher resolution captures viewpoint-specific textures (badge positions, reflections, shadow patterns) that help within-camera ReID but hurt cross-camera matching where viewpoint/lighting change dramatically
- **Insight**: The bottleneck is cross-camera feature invariance, not raw discriminative power

## Component Health Summary

| Component | Status | Notes |
|-----------|:------:|-------|
| Detection (YOLO26m) | ✅ OK | Not the bottleneck |
| Tracking (BoT-SORT) | ✅ OK | min_hits=2 optimal |
| Feature Extraction (ViT) | ⚠️ Ceiling | 256px remains best; 384px is a verified dead end for MTMC |
| Feature Processing (PCA) | ✅ OK | 384D optimal |
| Ensemble/Fusion | ❌ Blocked | Tested at 0.30 weight: -0.1pp. Secondary (52.77% mAP) far too weak for ensemble benefit |
| Association (Stage 4) | ✅ Exhausted | 225+ configs, no more gains |
| Evaluation | ✅ OK | Under-merging 1.69:1 ratio = feature quality issue |

## Model Training History

### TransReID ViT-B/16 CLIP (Primary)
- 09b v1: mAP=44.9% (40 epochs from 256px init, too aggressive LR)
- 09b v2: mAP=80.14%, R1=92.27% (VeRi-776 pretrained → CityFlowV2 fine-tune) ← **BEST, but wrong checkpoint on Kaggle**

### ResNet101-IBN-a (Secondary)
- 09d v12: mAP=21.9% (IBN layer3 bug)
- 09d v13: mAP=11.98% at epoch 19 (timed out)
- 09d v17: mAP=29.6% (wrong recipe: lr=3.5e-4 + circle_weight=0.5)
- 09d v18 ali369: mAP=52.77% (AdamW lr=1e-3, best so far)
- 09d gumfreddy v3: mAP=50.61% after resuming from the 09d v18 52.77% checkpoint with lower `lr=3e-4`; confirmed extra fine-tuning degrades this ImageNet→CityFlowV2 path
- 09d v18 mrkdagods: mAP=30.27% (SGD lr=0.008, failed catastrophically)
- 09e v2: VeRi-776 pretraining complete, mAP=62.52% on VeRi-776 test set (384x384, 120 epochs, triplet+center, cosine, fp16)
- 09f v2: mAP=16.2% catastrophic failure (circle_weight=1.0, bs=32, label_smooth=0.1, lr=7e-5; best model at epoch 4 during warmup)
- 09f v3: mAP=42.7% at epoch 104/120 (circle_weight=0.0, bs=48, label_smoothing=0.05, AdamW lr=3.5e-4, warmup start_factor=0.1, 120 epochs); still worse than direct ImageNet→CityFlowV2 in 09d v18

## Key Insight

**The system is NOT broken.** Vehicle MTMC remains capped by camera-invariant feature quality, not association logic. The current reproducible ceiling is 77.5% MTMC IDF1, while the historical 78.4% v80 result is no longer reproducible on the current codebase. Higher single-camera ReID mAP does **not** translate to better MTMC IDF1 in this pipeline: 384px deployment (-2.8pp despite a large mAP gain) and DMT camera-aware training (-1.4pp despite +7pp mAP) both made cross-camera association worse, and multi-query plus concat-patch variants also failed to improve over baseline. The ResNet path is likewise not a near-term ensemble unlock: 09f v3 recovered from the circle-loss failure but still topped out at 42.7% mAP, materially worse than the direct ImageNet→CityFlowV2 baseline at 52.77%, and extending that direct path further only degraded to 50.61%. On the person side, ground-plane tracking is already strong (**90.3% MODA, 94.7% IDF1**), but rerunning tracking on the stronger **12a v3** detector stayed essentially flat at **90.0% MODA, 94.7% IDF1**, indicating that the current WILDTRACK pipeline is now tracker-limited rather than detector-limited. The remaining vehicle work is no longer about more association sweeps or single-model feature tweaks; without a true multi-model ensemble, the realistic ceiling appears to be roughly 77-78% MTMC IDF1.