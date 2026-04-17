# MTMC Tracker — Research Findings & Strategic Analysis

> **IMPORTANT**: This is a living document. Update it whenever new experiments are run, new dead ends are discovered, or performance numbers change. Keep the "Current Performance" and "Dead Ends" sections current.

## Current Performance (Last Updated: 2026-04-17)

### Vehicle Pipeline (CityFlowV2)

| Metric | Value | Notes |
|--------|:-----:|-------|
| **Current Reproducible Best MTMC IDF1** | **77.5%** | 10c v52, v80-restored association recipe on the current codebase |
| **Historical Best MTMC IDF1** | **78.4%** | v80, but not reproducible with the current codebase (~1pp drift) |
| **SOTA Target** | 84.86% | AIC22 1st place |
| **Gap to SOTA** | 7.36pp | Relative to the current reproducible best |
| **Primary Model (ViT-B/16 CLIP 256px, 09 v2 aug overhaul)** | mAP=81.59% | On CityFlowV2 eval split; +1.45pp vs the prior 80.14% baseline |
| **Experiment B (CircleLoss ablation, 09 v4)** | mAP=18.45%, R1=48.84% | Catastrophic failure with baseline augmentations; training loss was `inf` at every epoch |
| **10c v48 (09 v2 augoverhaul @ 256px)** | MTMC IDF1=0.722 | Best result after an 11-sweep association re-optimization; single-cam IDF1 only 0.752 |
| **10c v49 (09 v3 augoverhaul-EMA @ 256px)** | MTMC IDF1=0.722 | Best result after a broader association sweep; AFLink recovered 0.675 -> 0.722 but could not break the augoverhaul ceiling |
| **Secondary Model (ResNet101-IBN-a)** | mAP=52.77% | On CityFlowV2 eval split, ImageNet→CityFlowV2 only |
| **Secondary Model VeRi-776 pretrain (09e v2)** | mAP=62.52% | On VeRi-776 test set, ready for CityFlowV2 fine-tuning |
| **384px ViT (09b v2)** | DEAD END | Higher single-camera ReID accuracy did not transfer; MTMC IDF1 only 0.7585-0.7562 in v43-v44, -2.8pp vs 256px baseline |
| **09f CityFlowV2 fine-tune** | mAP=42.7% | v3 peaked at epoch 104/120 and still underperformed direct ImageNet→CityFlowV2 (09d v18: 52.77%) |
| **Association configs tested** | 225+ | All within 0.3pp of optimal |

**Current reproducible vehicle MTMC IDF1 is 0.775** with 10c v52 using the v80-restored recipe (`sim_thresh=0.50`, `appearance_weight=0.70`, `fic_reg=0.50`, `aqe_k=3`, `gallery_thresh=0.48`, `orphan_match=0.38`). This is a small improvement over v50/v51 at 0.772, but it still trails the historical v80 result of 0.784. Vehicle association remains exhausted, so future gains will need materially better features or priors rather than more stage-4 tuning.

The latest structural association follow-up, **10c v53 network flow solver**, confirms that conclusion. Against the same controlled **10c v52** baseline, network flow reached only **MTMC IDF1 = 0.769** versus **0.7714** for the CC baseline (**-0.24pp**). It slightly improved **MOTA** (**0.689 -> 0.694**) and **HOTA** (**0.5747 -> 0.577**) with **2 fewer ID switches** (**199 -> 197**), but it **increased conflation from 27 to 30 predicted IDs** instead of reducing it. The current **conflict_free_cc** pipeline remains the preferred solver for this problem.

The corrected **10c v48** evaluation of the **09 v2 augoverhaul** model at the intended **256px** resolution still regressed sharply: the best full sweep reached only **MTMC IDF1 = 0.722** with `sim_thresh=0.45`, `appearance_weight=0.60`, `fic_reg=1.00`, `aqe_k=3`, `aflink_gap=150`, and `aflink_dir=0.85`. The follow-up **10c v49** sweep on the **09 v3 augoverhaul-EMA** training run reproduced the same **0.722** ceiling with a broader parameter search, confirming that the regression persists across augoverhaul variants and is driven by the model family rather than missed association tuning. Single-camera **IDF1 = 0.752** in **10c v48** was also materially below the baseline (~0.82), confirming that the earlier **10c v47** collapse was partly a deployment bug, but the underlying augoverhaul recipe is itself a vehicle-MTMC regression.

The new **Experiment B** CircleLoss-only ablation on the primary vehicle ReID path failed catastrophically. Kernel **`gumfreddy/09-vehicle-reid-cityflowv2-circleloss-ablation` v1** used the original baseline augmentation stack with **CE+LS(eps=0.05) + CircleLoss(m=0.25, gamma=128) + CenterLoss** for **120 epochs**, but the training loss was **`inf` at every epoch** and the run collapsed to only **mAP = 18.45%** and **R1 = 48.84%**. This independently confirms that **CircleLoss is a dead end** on this CityFlowV2 TransReID recipe. It also sharpens the interpretation of the augoverhaul regression: either the augoverhaul augmentations themselves caused the **81.59% mAP -> 0.722 MTMC IDF1** failure, or **CircleLoss was not actually active** in that training run due to a config/path mismatch. In either case, **CircleLoss is not a viable explanation for a healthy high-mAP regime** because when it is definitely active, it destroys training entirely.

**⚠️ Metric Disambiguation (Vehicle Pipeline):**
- **MTMC IDF1 = 77.5%** — Current reproducible best on the current codebase from 10c v52. This remains the official metric and the only number that should be compared to AIC22 SOTA.
- **Historical v80 reference** — MTMC IDF1 = 78.4%, IDF1 = 79.8%, GLOBAL IDF1 = 80.5%. These numbers are useful for historical comparison, but they are not currently reproducible.
- All current best numbers use GT-assisted metrics (`gt_frame_clip=true`, `gt_zone_filter=true`) which inflate scores by 1-3pp vs clean evaluation.

### Person Pipeline (WILDTRACK) — NEW

| Metric | Value | Notes |
|--------|:-----:|-------|
| **Ground-plane MODA** | **90.0%** | Best converged operating point reproduced across the final 12b tracking runs |
| **Ground-plane IDF1** | **94.7%** | Confirmed independently by 12b v1, v2, and v3; Precision=94.5%, Recall=96.1%, IDSW=5 |
| **Best Detector (MVDeTr 12a v3)** | MODA=92.1% | Epoch 20/25, Precision=95.7%, Recall=96.6%; best WILDTRACK detector yet |
| **12b v2 extended Kalman sweep** | MODA=90.0%, IDF1=94.7% | Wider interpolation/max_age/conf sweeps plus velocity-aware quadratic interpolation still converged to the same 5-IDSW operating point |
| **12b v3 global optimal tracker** | MODA=88.2%, IDF1=91.2% | Sliding-window Hungarian assignment underperformed badly; immediate frame-level costs overrode the motion-prediction advantage of Kalman and produced 15 ID switches |
| **12b v1 on 12a v3 detections** | MODA=90.0%, IDF1=94.7% | Better detections did not improve tracking; Kalman sweep converged to the same effective operating point |
| **Previous Best Tracking Baseline** | IDF1=92.8% | 12b v9 naive tracker; new tuned Kalman run is +1.9pp |
| **ReID Features (12b v8)** | mean cosine=0.720 | Fixed ViT backbone mismatch; range widened to [0.215, 1.000] |
| **ReID Merge State** | No gain over baseline | Tuned Kalman tracks are already clean; merge sweep did not improve IDF1 |
| **Gap to SOTA** | 0.6pp | Best IDF1=94.7% vs target 95.3% |
| **Status** | FULLY CONVERGED | Kalman, naive, and global-optimal trackers tested across 59 configs; remaining path is better person appearance features or graph-based approaches |

**Current person best remains 94.7% IDF1 and is now fully converged**: 12b v14 first reached **IDF1=94.7%** with a tuned Kalman tracker, and that operating point has now been independently matched by **12b v1**, **12b v2**, and **12b v3**-era validation runs around the same final configuration. The stronger **12a v3** detector did not improve tracking, broader Kalman sweeps across interpolation/max_age/confidence thresholds did not improve tracking, and the new sliding-window **GlobalOptimalTracker** regressed sharply to **IDF1=91.17%**, **MODA=88.2%**, and **15 ID switches**. Across **59 tracker configs** spanning **Kalman**, **naive**, and **global optimal** trackers, the best Kalman solutions all clustered at **IDF1=0.9467 ± 0.0004**, with confidence thresholds from **0.15-0.35** making no meaningful difference. The root cause is that global optimal assignment over-commits to immediate frame-level costs, while BoT-SORT's predictive Kalman model handles occlusions and missed detections better over time. The current WILDTRACK person pipeline is therefore **fully tracker-converged at 94.7% IDF1**. The remaining **0.6pp** gap to SOTA is most likely due to rare occlusion and re-entry failures that need better person appearance features or graph-based multi-view association, not more tracker tuning.

## Gap Decomposition

The remaining gap to SOTA now decomposes into:

| Deficiency | Impact | Status |
|------------|:------:|--------|
| Single ReID model vs 3-5 ensemble | Largest remaining gap | Tested with weak secondary (52.77% mAP at 0.30 weight): no gain. Needs ≥65% mAP secondary to be viable |
| Camera-aware single-model training (DMT) | -1.4pp | Tested and harmful (v46) |
| Multi-query track representation | -0.1pp | Tested and neutral/harmful (v51) |
| Higher-dimensional concat-patch features | -0.3pp | Tested and harmful (v48-v49) |
| Association tuning / structural association changes | Exhausted | 225+ configs plus structural variants already tried |

**Higher single-camera ReID mAP is not translating into better MTMC IDF1.** Three experiments now confirm this: the augmentation-overhaul plus CircleLoss recipe, 384px deployment, and DMT camera-aware training all improved or preserved validation mAP while hurting downstream MTMC. Without a true multi-model ensemble, the realistic ceiling now appears to be roughly 77-78% MTMC IDF1.

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
| Camera-pair bias (CID_BIAS) | ROI masks | NPY | NPY | **Tested twice and harmful: GT-learned -3.3pp, topology bias -1.0 to -1.2pp (DEAD END)** |
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
- **CID_BIAS**: both tested variants are dead ends. The original GT-learned CID_BIAS dropped MTMC IDF1 from **0.784 -> 0.751** (**-3.3pp**), and the later topology-bias sweep in **10c v55** reached only **0.764-0.762-0.763** versus a **0.774** control (**-1.0 to -1.2pp**). FIC whitening already handles the useful camera calibration, and additive CID_BIAS terms distort those calibrated similarities.

**Why the previous improvements failed**

- With **1 model**, techniques like **384px**, **DMT**, and **reranking** strip away unstable but still useful signal, so MTMC gets worse.
- With **3-5 models**, those same techniques can suppress noise while identity signal survives through model diversity.
- The key blocker is therefore not that these methods are intrinsically bad, but that they were evaluated in a **single-model regime** where AIC winners never operated.

## Prioritized Action Plan

| Priority | Action | Expected Impact | Status |
|:--------:|--------|:---------------:|--------|
| **1** | Train or acquire a truly complementary secondary/tertiary ReID model for ensemble use | Only plausible path to a material gain | BLOCKED — ensemble tested at 0.30 weight with 52.77% mAP secondary, no gain; ResNet training path exhausted without VeRi-776 benefit |
| **2** | ~~CID_BIAS~~ | Both variants are dead ends: GT-learned **-3.3pp** and topology bias **-1.0 to -1.2pp**; FIC whitening already covers camera calibration | **DEAD END** |
| **3** | Additional association sweeps or structural tweaks | Negligible | NOT RECOMMENDED |
| **4** | More single-model feature variants (384px, DMT, multi-query, concat-patch) | Negative based on current evidence | DO NOT RETRY |
| **5** | Re-enable reranking only after feature quality improves materially | Potentially positive only with better features | BLOCKED by current features |

Association tuning remains exhausted (225+ configs tested). The remaining vehicle path is no longer incremental stage-4 tuning or single-model feature ablations; it is ensemble-quality representation learning or materially stronger camera-pair priors.

## Latest Experiment Results (2026-04-17)

### Completed

#### 10c v55 — CID_BIAS Topology Bias Sweep (2026-04-17)
- **Task**: Test whether a lightweight **topology-derived CID_BIAS** improves Stage-4 association by adding camera-pair-specific similarity offsets on top of the restored baseline recipe
- **Baseline**: Fresh baseline features from **10a v30** with a control run at **MTMC IDF1 = 0.774** and no additive bias
- **Conservative bias**: **+0.02 / -0.10** reached **MTMC IDF1 = 0.764** (**-1.0pp**)
- **Default bias**: **+0.04 / -0.15** reached **MTMC IDF1 = 0.762** (**-1.2pp**)
- **Aggressive bias**: **+0.06 / -0.20** reached **MTMC IDF1 = 0.763** (**-1.2pp**)
- **Interpretation**: Every tested additive bias degraded MTMC, including the most conservative setting. This indicates the current pipeline is already getting the useful camera-pair calibration from **FIC whitening**, and extra CID_BIAS offsets only warp the calibrated similarity geometry.
- **Conclusion**: **Topology CID_BIAS is a confirmed dead end** for the current CityFlowV2 single-model pipeline. Together with the earlier **GT-learned CID_BIAS** regression (**-3.3pp**), this now rules out both learned and hand-shaped additive CID_BIAS variants.

#### 10c v53 — Network Flow Solver vs CC Baseline (2026-04-17)
- **Task**: Test a **network flow / Hungarian-based structural association solver** as a replacement for the existing connected-components merge logic, with merge verification intended to reduce cross-camera conflation
- **Baseline**: Controlled **10c v52** CC run at **MTMC IDF1 = 0.7714**, **MOTA = 0.689**, **HOTA = 0.5747**, **ID switches = 199**, **45 fragmented GT IDs**, **27 conflated predicted IDs**
- **Network flow result**: **MTMC IDF1 = 0.769**, **MOTA = 0.694**, **HOTA = 0.577**, **ID switches = 197**, **46 fragmented GT IDs**, **30 conflated predicted IDs**
- **Delta vs baseline**: **-0.24pp MTMC IDF1**, **+0.5pp MOTA**, **+0.2pp HOTA**, **-2 ID switches**, **+1 fragmented GT ID**, **+3 conflated predicted IDs**
- **Interpretation**: The solver improved some frame-level metrics slightly, but it failed its main design goal. The Hungarian assignment plus merge verification did **not** prevent false merges and instead created **more conflation** than the CC baseline.
- **Conclusion**: **Network flow is neutral to slightly negative** for the current CityFlowV2 vehicle pipeline. It is **not a hard dead end** on the scale of AFLink or CSLS, but it is **not helpful** and does not justify replacing the existing **conflict_free_cc** approach.

#### 10a v29 / 10c v50 — SAM2 Foreground Masking Before Vehicle ReID (2026-04-16)
- **Task**: Test inference-time **SAM2 foreground masking** on vehicle crops before Stage-2 ReID feature extraction using **`facebook/sam2.1-hiera-tiny`**
- **Masking config**: **center-point prompt**, **per-crop mean fill**, **`min_crop_size=48`**
- **10a v29 runtime / scale**: **929 tracklets**, **105.2 min**, versus roughly **~65 min** for the same pipeline without SAM2 masking
- **10c v50 evaluation**: Full **60-config** Stage-4 parameter sweep with **FEATURE_TEST enabled** across **11 parameter dimensions**, including AFLink and camera-pair normalization variants
- **Best result**: **MTMC IDF1 = 0.688**
- **Per-camera 2D metrics**: **MOTA = 0.540**, **IDF1 = 0.704**, **HOTA = 0.515**
- **Baseline comparison**: Current reproducible non-SAM2 baseline is **MTMC IDF1 = 0.775** from **10c v52**, so SAM2 masking is **-8.7pp** worse despite exhaustive association retuning
- **Interpretation**: Foreground masking removes background context such as road surface and nearby scene cues that appear to help cross-camera vehicle re-identification. The masks also likely clip vehicle edges and suppress boundary features that carry identity signal.
- **Conclusion**: **SAM2 foreground masking is a confirmed dead end** for the current CityFlowV2 vehicle pipeline. It adds major runtime cost while degrading both single-camera and MTMC quality, and even aggressive downstream retuning could not recover baseline performance.

#### 09 v4 / Experiment B — CircleLoss Ablation on Baseline Augmentations (2026-04-16)
- **Kernel**: `gumfreddy/09-vehicle-reid-cityflowv2-circleloss-ablation` **v1**
- **Purpose**: Isolate whether **CircleLoss** caused the augoverhaul regression by keeping the original baseline augmentation stack and disabling EMA
- **Config**: **TransReID ViT-B/16 CLIP 256px**, **CE+LS(eps=0.05) + CircleLoss(m=0.25, gamma=128) + CenterLoss**, **120 epochs**, baseline augmentations only: `RandomHorizontalFlip`, `Pad+RandomCrop`, `ColorJitter(brightness=0.2, contrast=0.15, saturation=0.1, hue=0.0)`, `Normalize`, `RandomErasing`
- **Result**: **Catastrophic failure** with **best mAP = 18.45%** and **best R1 = 48.84%**
- **Training stability**: Loss was **`inf` at every epoch**, indicating complete numerical instability rather than a weak-but-valid training regime
- **Interpretation**: This matches the pre-existing dead-end pattern that **circle-style metric learning can catastrophically destabilize training** in this codebase. It also means the prior augoverhaul regression cannot be explained as “CircleLoss helped ReID but hurt MTMC.” When CircleLoss is definitely active on the primary ViT recipe, it destroys training outright.
- **Conclusion**: **CircleLoss is a confirmed dead end** for the primary CityFlowV2 TransReID path. The augoverhaul regression is therefore most likely attributable to the augoverhaul augmentations themselves, unless the earlier augoverhaul training never actually activated CircleLoss because of a config mismatch.

#### 09 v2 — Augmentation Overhaul + EMA (2026-04-13)
- **Task**: Test a stronger augmentation recipe plus model EMA on the primary CityFlowV2 TransReID ViT-B/16 CLIP training path
- **Augmentation overhaul**: Added `RandomGrayscale(p=0.1)`, stronger `ColorJitter(0.3,0.25,0.2,0.05)`, `GaussianBlur(k=5,p=0.2)`, `RandomPerspective(0.1,p=0.2)`, and a wider `RandomErasing` scale range
- **EMA config**: Model Exponential Moving Average with `decay=0.9999`
- **Base model result**: **mAP = 81.59%**, **+1.45pp** over the prior **80.14%** baseline
- **Base model with reranking**: **mAP = 83.12%**
- **EMA model result**: **mAP = 39.09%**
- **EMA model with reranking**: **mAP = 47.76%**
- **Analysis**: The augmentation overhaul clearly works and improves generalization. The EMA branch failed because `decay=0.9999` is too high for a 120-epoch schedule; the averaged weights converge too slowly and were still improving at epoch 120.
- **Downstream outcome**: The corrected end-to-end follow-up in **10c v48** reached only **0.722 MTMC IDF1**, so the higher validation mAP did not transfer to vehicle MTMC.

#### 09 v3 — EMA Training Results (2026-04-14)
- **Task**: Re-test EMA on the primary CityFlowV2 TransReID ViT-B/16 CLIP path using the augoverhaul augmentation stack with the standard **TripletLoss + CenterLoss** recipe and a lower **EMA decay = 0.999**
- **Base model result**: **mAP = 81.53%**, **R1 = 92.41%**
- **EMA model result**: **mAP = 81.44%**, **R1 = 92.74%**
- **EMA delta vs base**: **-0.09pp mAP**, **+0.33pp R1**
- **Baseline comparison**: Relative to the prior **80.14% mAP / 92.27% R1** baseline, the augoverhaul + EMA training run still gains **+1.39pp mAP** overall
- **Interpretation**: EMA converges to essentially the same solution as the base model. The tiny validation difference is not meaningful enough to justify carrying a second checkpoint or treating EMA as a distinct improvement path.
- **Downstream implication**: Combined with **10c v48 = 0.722** and **10c v49 = 0.722 MTMC IDF1** (**-5.3pp** vs baseline in both cases), this further confirms that higher single-camera **mAP** is **not** the MTMC bottleneck in the current vehicle pipeline.
- **Conclusion**: **EMA is a dead end** for this recipe. It produces nearly identical validation quality to the base model and does not change the core finding that feature-space changes with better mAP are failing to improve MTMC.

#### 10c v47 — Augmentation Overhaul Model with 384px Deployment Bug (2026-04-14)
- **Task**: Evaluate the new **09 v2 augmentation-overhaul** primary model in the downstream **10a -> 10b -> 10c** CityFlowV2 pipeline
- **Result**: **MTMC IDF1 = 0.702**, a **-7.3pp** regression versus the current reproducible **0.775** baseline
- **Root cause**: **BUG**. The **10a** notebook deployed the augoverhaul model at **384x384** input even though it was trained for **256x256**. This happened because `configs/datasets/cityflowv2.yaml` still had `input_size: [384, 384]` from the earlier 384px dead-end work, and the notebook did not explicitly override it for the 256px model.
- **Why this matters**: This result does **not** show that the augmentation-overhaul model is worse. The model itself improved single-camera quality to **mAP = 81.59%** versus the previous **80.14%** baseline, so it should be expected to perform **better**, not worse, when deployed at the correct **256x256** resolution.
- **Fix applied**: Added an explicit `stage2.reid.vehicle.input_size=[256,256]` override to the **10a** notebook and fixed the default `cityflowv2.yaml` input size so 256px deployment is now the safe default for the primary vehicle model.
- **Status**: The fix was verified in **10a v20**, and the corrected **256px** deployment was evaluated end-to-end in **10c v48**.
- **Conclusion**: Treat **10c v47** as a configuration failure, not as evidence against the augmentation overhaul. The apparent regression was caused by a silent deployment mismatch, not by weaker learned features.

#### 10c v48 — Corrected 256px Evaluation of the 09 v2 Augoverhaul Model (2026-04-14)
- **Task**: Re-run the full downstream **10a -> 10b -> 10c** CityFlowV2 pipeline with the **09 v2 augoverhaul** model deployed at the correct **256x256** input size
- **Best association config**: `sim_thresh=0.45`, `appearance_weight=0.60`, `fic_reg=1.00`, `aqe_k=3`, `aflink_gap=150`, `aflink_dir=0.85`
- **Sweep size**: **11 association configs**
- **Result**: **MTMC IDF1 = 0.722**, which is **-5.3pp** versus the current reproducible **0.775** baseline
- **Single-camera quality**: **IDF1 = 0.752**, also materially below the baseline (~0.82)
- **Interpretation**: The 384px deployment bug in **10c v47** was real, but fixing it did **not** rescue the model. The underlying **09 v2** recipe still regresses badly for MTMC even at the correct deployment resolution.
- **Root cause**: The experiment was confounded because it changed both the augmentation stack and the loss at the same time: it added `RandomGrayscale`, stronger `ColorJitter`, `GaussianBlur`, and `RandomPerspective`, while also replacing **TripletLoss** with **CircleLoss**. The stronger augmentations likely suppress color and texture cues that are still needed to distinguish same-model vehicles across cameras, and the simultaneous loss change prevents precise attribution.
- **Conclusion**: The **09 v2 augoverhaul + CircleLoss** recipe is a confirmed **DEAD END** for CityFlowV2 vehicle MTMC despite its higher validation **mAP = 81.59%**.

#### 10c v49 — Association Sweep on the 09 v3 Augoverhaul-EMA Model (2026-04-14)
- **Task**: Re-run the downstream **10c** association sweep using the **09 v3 augoverhaul-EMA** training run (`mAP = 81.53%`, `R1 = 92.41%`) to test whether the EMA-trained augoverhaul recipe changes the MTMC operating point
- **Best association config**: `sim_thresh=0.45`, `appearance_weight=0.60`, `spatiotemporal_weight=0.40`, `fic_reg=1.00`, `aqe_k=3`, `gallery_thresh=0.45`, `orphan_match=0.35`
- **Best result**: **MTMC IDF1 = 0.722**, exactly matching **10c v48** and therefore confirming a persistent augoverhaul ceiling
- **AFLink effect**: With the best non-AFLink operating point, MTMC IDF1 peaked at **0.675**; enabling AFLink with `max_spatial_gap_px=150` and `min_direction_cos=0.85` lifted it to **0.722**, but still did not recover the baseline
- **Reranking**: **Off** was best throughout the sweep
- **Camera-pair normalization**: **Off** was best throughout the sweep
- **Intra-camera merge**: Negligible effect at `thresh=0.80` and `gap=60`
- **Interpretation**: Even after swapping to the **09 v3** augoverhaul checkpoint and expanding the association search, the ceiling stayed fixed at **0.722**. That makes the failure mode robust to association variants and points back to the augoverhaul feature space itself rather than missed stage-4 tuning.
- **Conclusion**: The augoverhaul regression is now confirmed across both **10c v48** and **10c v49**. Different sweeps and slightly different augoverhaul checkpoints converge to the same ceiling, so the model family itself is the issue, not the association parameters.

#### 10c v46 — AFLink Motion-Based Post-Association (2026-04-13)
- **Task**: Test AFLink as a motion-based post-association stage after the standard CityFlowV2 appearance-driven association
- **Merge effect**: AFLink produced **57 motion-based merges**, reducing trajectories from **239 -> 182**
- **Best AFLink params**: `spatial_gap=150px`, `direction_cos=0.85`, `camera_pair_norm=on`
- **Best result**: **MTMC IDF1 = 0.7355**, which is **down from the 0.775 baseline by -3.95pp**
- **Within-sweep behavior**: AFLink gained roughly **+6.3pp** inside its own sweep (**0.669 -> 0.732**) when tuning `spatial_gap` and `direction_cos`
- **Retest status**: This result is now **confirmed by a clean controlled retest** on **10c v52** using the exact restored baseline association recipe (`sim_thresh=0.50`, `appearance_weight=0.70`, `fic_reg=0.50`, `aqe_k=3`, `gallery_thresh=0.48`, `orphan_match=0.38`) and fresh baseline features from **10a v30**.
- **Controlled retest**: Control without AFLink reached **MTMC IDF1 = 0.7714**, **IDF1 = 0.7921**, **HOTA = 0.5747**. AFLink then degraded to **0.7332** at `gap=100`, `dir_cos=0.90` (**-3.82pp**), **0.7183** at `gap=150`, `dir_cos=0.85` (**-5.31pp**), and **0.6394** at `gap=200`, `dir_cos=0.70` (**-13.20pp**).
- **Root cause**: The original v46 result was initially suspected to be partly confounded by joint sweep interactions, but the clean retest removes that explanation. Even with a tight **100px** gap and strict **0.90** direction cosine, motion consistency across non-overlapping CityFlowV2 cameras is not reliable enough for identity linking. AFLink creates false cross-camera merges that fragment the identity space, and wider motion windows make the damage worse.
- **Conclusion**: AFLink is **confirmed harmful** on **CityFlowV2** and should be treated as a **DEAD END** for this dataset. The true penalty is now established at roughly **-3.8pp to -13.2pp MTMC IDF1**, not just the original **-3.95pp** point estimate from v46.
- **Key insight**: The fact that even `gap=100` with `dir_cos=0.90` still loses **-3.82pp** shows the problem is structural, not a loose-threshold artifact.

#### 12b v2 — Extended Kalman Sweep (2026-04-13)
- **Task**: Extend the WILDTRACK tracker sweep beyond the original tuned-Kalman search to test whether broader interpolation, longer track persistence, looser detection thresholds, or better interpolation dynamics can reduce the remaining 5 ID switches
- **Sweep extensions**: interpolation **[1,2,3] -> [1,2,3,4,5]**, `max_age` **[2,3,4,5] -> [2,3,4,5,6,8]**, detection confidence **[0.15, 0.20, 0.25, 0.30, 0.35]**, plus velocity-aware quadratic interpolation
- **Best config**: `max_age=2`, `min_hits=2`, `distance_gate=25.0`, `q_std=5.0`, `r_std=10.0`, `interpolation_max_gap=1`, `conf_threshold=0.15`
- **Result**: **IDF1 = 94.7%**, **MODA = 90.0%**, **IDSW = 5**
- **Comparison**: Identical to the prior best operating point; no sweep extension reduced ID switches below **5** or improved IDF1 beyond **0.9467**
- **Conclusion**: Kalman tracker parameters are fully exhausted for the current person pipeline. The remaining gap is tracker-limited, not parameter-limited.

#### 12b v3 — Global Optimal Tracker (2026-04-13)
- **Task**: Implement and test a new `GlobalOptimalTracker` with sliding-window globally optimal assignment (Hungarian algorithm with a configurable window) as a possible replacement for the current Kalman tracker on WILDTRACK
- **Global-optimal config**: `window_size=10`, `birth_death_cost=100`
- **Sweep size**: **59 tracker configs** across **Kalman**, **naive**, and **global optimal** trackers
- **Best Kalman result**: **IDF1 = 94.7%**, **MODA = 90.0%**, **IDSW = 5**
- **Best naive result**: **IDF1 = 93.25%**, **MODA = 88.6%**, **IDSW = 12**
- **Best global-optimal result**: **IDF1 = 91.17%**, **MODA = 88.2%**, **IDSW = 15**
- **Comparison**: Global optimal was the **worst** of all three tracker families, trailing Kalman by **-3.5pp IDF1** with **3x more ID switches**
- **Additional finding**: All tested Kalman configs clustered at **IDF1 = 0.9467 ± 0.0004**, and confidence threshold sweeps across **0.15-0.35** made no meaningful difference
- **Root cause**: Global optimal assignment over-commits to immediate frame-level costs and loses the motion-prediction advantage of Kalman filtering. The predictive model in BoT-SORT handles occlusions and missed detections better than pure assignment optimization.
- **Conclusion**: The person pipeline is **fully converged at IDF1=0.947**. Kalman is the optimal tracker for the current system, and further tracker architecture changes are unlikely to close the remaining **0.6pp** gap to SOTA. That gap is more likely caused by rare occlusion/re-entry cases that need stronger person appearance features or graph-based multi-view fusion, not more tracker improvements.

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

#### Kaggle T4 Dataset Mount Structure Discovery
- **Discovery**: On Kaggle **T4 GPU** kernels, datasets mounted under **`/kaggle/input/datasets/<owner>/<slug>/`** rather than **`/kaggle/input/<slug>/`**
- **Impact**: This caused multiple 09d failures when kernels assumed the legacy flat mount path and could not locate the dataset
- **Conclusion**: When debugging missing-dataset errors on Kaggle T4 kernels, check the nested `datasets/<owner>/` mount structure first.

## Bugs Found / Configuration Pitfalls

### CityFlowV2 `input_size` default leaked a dead-end 384px setting (2026-04-14)
- **Bug**: `configs/datasets/cityflowv2.yaml` still contained `input_size: [384, 384]`, even though **384px deployment is a confirmed dead end** for the main vehicle MTMC pipeline.
- **Failure mode**: The **10a** notebook silently inherited that config and exported the new **09 v2 augmentation-overhaul** model at **384x384** instead of its intended **256x256** input size.
- **Observed impact**: This caused the catastrophic **10c v47 = 0.702 MTMC IDF1** result and could easily have been misread as a model-quality regression.
- **Verification**: The fix was verified in **10a v20**, and the corrected **256px** deployment was then evaluated in **10c v48**.
- **Lesson**: Leaving dead-end experimental defaults in shared config files is dangerous because later notebooks can silently inherit them.
- **Fix**: Set the safe default back to **256x256** and add explicit notebook-level overrides for `stage2.reid.vehicle.input_size` whenever testing alternate deployment resolutions.

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
- Treat person tracking as fully converged at **94.7% IDF1** under the current Kalman pipeline; **12b v1**, **12b v2**, and **12b v3** all confirmed the same ceiling
- Further person gains likely require materially stronger person appearance features or graph-based multi-view association rather than more tracker tuning
- Revisit merge only if person ReID quality improves materially; current Kalman tracks are already clean and merges did not help
- Keep GPU-intensive person stages on Kaggle only

## What Actually Worked

- **Augmentation overhaul**: **+1.45pp mAP** on the primary vehicle ReID model via `RandomGrayscale`, `GaussianBlur`, `RandomPerspective`, stronger `ColorJitter`, and a wider `RandomErasing` range
- **Deployment-bug correction**: The **10c v47** collapse to **0.702 MTMC IDF1** was correctly traced to a **384px deployment mismatch**; the **10a v20** fix and **10c v48** rerun verified the bug was real, even though the underlying **09 v2** recipe still regressed for MTMC.

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
| CircleLoss on TransReID ViT-B/16 CLIP (Experiment B, baseline augmentations) | Catastrophic numerical instability; training loss was `inf` at every epoch and the run collapsed to **18.45% mAP / 48.84% R1** | `gumfreddy/09-vehicle-reid-cityflowv2-circleloss-ablation` v1 |
| ResNet101-IBN-a VeRi-776→CityFlowV2 fine-tuning | mAP=42.7% (09f v3), worse than direct ImageNet→CityFlowV2 (52.77% mAP, 09d v18); secondary-model ensemble path not viable without better pretraining or broader hyperparameter search | 09e v2, 09f v3, 09d v18 |
| Extended fine-tuning from the 09d v18 ResNet101-IBN-a checkpoint | mAP degraded from 52.77% to 50.61% after resuming with lower LR; confirms the direct ImageNet→CityFlowV2 path is already at its ceiling | 09d gumfreddy v3 |
| CLIP ViT backbone when checkpoint uses standard ViT | `norm_pre` randomly initialized, causing mode collapse (cosine sim 0.874) | 12b v5/v6 vs v8 |
| Default Kalman tracker with chi-squared gating on WILDTRACK | Worse than naive baseline; IDF1 fell to 88.9% | 12b v14 sweep |
| ReID merge on top of tuned WILDTRACK Kalman tracks | No improvement over baseline; tracks already clean and only 44 features matched | 12b v14 merge sweep |
| Extended Kalman parameter sweeps (person) | No improvement; 59 tracker configs clustered within +-0.0004 IDF1 around the same 0.947 Kalman operating point, so the tracker parameter space is fully exhausted | 12b v1-v3 |
| Global optimal tracker (person) | -3.5pp IDF1 vs Kalman; immediate assignment costs lost the motion-prediction advantage and produced 15 ID switches | 12b v3 |
| K-reciprocal reranking (with current features) | Always worse | v25, v35 |
| Camera-pair similarity normalization | Zero effect (FIC handles it) | v36 |
| CID_BIAS (camera-pair bias matrix) | -3.3pp MTMC IDF1 on 256px features (0.751 vs 0.784) | v44 + CID_BIAS test |
| confidence_threshold=0.20 | -2.8pp | v45 |
| max_iou_distance=0.5 | -1.6pp | v47 |
| AFLink motion linking | Confirmed harmful in controlled retest: **-3.82pp** even at `gap=100`, `dir_cos=0.90`; broader gaps degrade to **-5.31pp** (`150/0.85`) and **-13.20pp** (`200/0.70`). Motion consistency across non-overlapping cameras is unreliable and false merges dominate. | 10c v46, 10c v52 |
| SAM2 foreground masking | -8.7pp MTMC IDF1 (0.688 vs 0.775 baseline); removes useful background context and clips vehicle boundary features, while increasing runtime from ~65 min to 105.2 min | 10a v29, 10c v50 |
| 384px TransReID deployment | -2.8pp MTMC IDF1 despite +10pp mAP; v43=0.7585, v44=0.7562 vs v80=0.784 | 10a v43, 10a v44, v80 baseline |
| Augmentation overhaul + CircleLoss (09 v2 augoverhaul model) | -5.3pp MTMC IDF1 (0.722 vs 0.775 baseline) despite +1.45pp mAP; confounded recipe changed both augmentations and loss. Follow-up CircleLoss-only ablation collapsed to **18.45% mAP** with `inf` loss every epoch, so CircleLoss is independently a dead end and the augoverhaul augmentations are the most likely regression source unless the original run had a CircleLoss config mismatch | 10c v48, 09 v2, 09 v4 Experiment B |
| EMA model averaging | Model **mAP=81.53%** vs EMA **mAP=81.44%** with **R1 92.41% vs 92.74%**; EMA converges to essentially the same place and is not a meaningful improvement path | 09 v3 |
| DMT camera-aware training (87.3% mAP) | -1.4pp MTMC IDF1; v46=0.758 vs v45=0.772 | 10c v45-v46 |
| EMA (`decay=0.9999`, 120 epochs) | mAP=39.09%; too slow to converge under the current training schedule and needs a much longer run to be viable | 09 v2 |
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
| Association (Stage 4) | ✅ Exhausted | 225+ configs plus network-flow v53; CC remains preferred after network flow lost -0.24pp and increased conflation |
| Evaluation | ✅ OK | Under-merging 1.69:1 ratio = feature quality issue |

## Model Training History

### TransReID ViT-B/16 CLIP (Primary)
- 09b v1: mAP=44.9% (40 epochs from 256px init, too aggressive LR)
- 09b v2: mAP=80.14%, R1=92.27% (VeRi-776 pretrained → CityFlowV2 fine-tune) ← **BEST, but wrong checkpoint on Kaggle**
- 09 v2: mAP=81.59%, rerank mAP=83.12% with augmentation overhaul; EMA branch failed at mAP=39.09% with `decay=0.9999` over 120 epochs

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

**The system is NOT broken.** Vehicle MTMC remains capped by camera-invariant feature quality, not association logic. The current reproducible ceiling is 77.5% MTMC IDF1, while the historical 78.4% v80 result is no longer reproducible on the current codebase. Higher single-camera ReID mAP does **not** translate to better MTMC IDF1 in this pipeline: augmentation overhaul plus CircleLoss (**+1.45pp mAP -> -5.3pp MTMC IDF1**), 384px deployment (**same mAP -> -2.8pp MTMC IDF1**), and DMT camera-aware training (**+7pp mAP -> -1.4pp MTMC IDF1**) all made cross-camera association worse, and the new CircleLoss-only Experiment B showed that when CircleLoss is definitely active on the primary ViT recipe it fails catastrophically (**18.45% mAP, `inf` loss every epoch**). That means the augoverhaul regression is most likely driven by the augoverhaul augmentations themselves, unless the earlier training had a CircleLoss config mismatch. This now confirms that **mAP != MTMC IDF1** for CityFlowV2 vehicle tracking: the MTMC graph needs features with clean, thresholdable similarity distributions, not just good ranking on the validation split. The ResNet path is likewise not a near-term ensemble unlock: 09f v3 recovered from the circle-loss failure but still topped out at 42.7% mAP, materially worse than the direct ImageNet→CityFlowV2 baseline at 52.77%, and extending that direct path further only degraded to 50.61%. On the person side, ground-plane tracking is already strong (**90.3% MODA, 94.7% IDF1**), but rerunning tracking on the stronger **12a v3** detector stayed essentially flat at **90.0% MODA, 94.7% IDF1**, indicating that the current WILDTRACK pipeline is now tracker-limited rather than detector-limited. The remaining vehicle work is no longer about more association sweeps or single-model feature tweaks; without a true multi-model ensemble, the realistic ceiling appears to be roughly 77-78% MTMC IDF1.